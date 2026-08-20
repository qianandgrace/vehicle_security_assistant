"""用 30 题评测 query 优化方案（重写 / 意图路由 / HyDE）。

对比矩阵（P0/P1 复用上轮结果，本脚本只跑 3 个新方案）：
    P2  HyPE + LLM 全量重写   -> query_opt/eval_rewrite.csv
    P3  HyPE + 意图路由重写   -> query_opt/eval_route.csv
    P4  基线 + HyDE           -> query_opt/eval_hyde.csv

输出：res/optimization/query_opt/
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_milvus import Milvus

from config import (
    DB_NAME,
    HYPE_COLLECTION,
    MD_COLLECTION,
    MILVUS_URI,
    OPT_RES_DIR,
    TESTSET_30,
    TOP_K,
)
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics
from utils.eval_rag import answer_with_context
from utils.llm_utils import get_llm, get_embedding_model, get_index_llm
from utils.query_opt import classify_and_rewrite, hyde_document, rewrite_query

HYPE_MULTIPLIER = 4


def retrieve_contexts(vs, search_query, k, collection):
    """检索并返回上下文文本列表（HyPE 需要把假设性问题映射回原 chunk 并去重）。"""
    if collection == HYPE_COLLECTION:
        docs = vs.similarity_search(search_query, k=k * HYPE_MULTIPLIER)
        seen, ctx = set(), []
        for d in docs:
            orig = d.metadata.get("original_content", d.page_content)
            if orig and orig not in seen:
                seen.add(orig)
                ctx.append(orig)
            if len(ctx) >= k:
                break
        return ctx
    docs = vs.similarity_search(search_query, k=k)
    return [d.page_content for d in docs]


def build_search_queries(testset, mode, llm):
    """根据模式生成每条问题的检索查询。mode: rewrite | route | hyde | raw"""
    search_queries = []
    for i, it in enumerate(testset, 1):
        q = it["question"]
        if mode == "rewrite":
            sq = rewrite_query(q, llm)
        elif mode == "route":
            c = classify_and_rewrite(q, llm)
            sq = c["rewritten_query"] if c["need_rewrite"] else q
        elif mode == "hyde":
            sq = hyde_document(q, llm, chunk_size=512)
        else:
            sq = q
        search_queries.append(sq)
        if i % 10 == 0 or i == len(testset):
            print(f"  查询预处理 {i}/{len(testset)}")
    return search_queries


def evaluate(mode, collection, testset, llm, idx_llm, embedding, k):
    name = {"rewrite": "HyPE+全量重写", "route": "HyPE+意图路由", "hyde": "基线+HyDE"}[mode]
    print(f"\n===== 评测：{name} ({collection}) =====")
    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection,
        connection_args={"uri": MILVUS_URI, "db_name": DB_NAME},
    )
    search_queries = build_search_queries(testset, mode, idx_llm)

    questions = [t["question"] for t in testset]
    references = [t["answer"] for t in testset]
    retrieved_contexts, answers = [], []
    for i, (q, sq) in enumerate(zip(questions, search_queries), 1):
        ctxs = retrieve_contexts(vs, sq, k, collection)
        retrieved_contexts.append(ctxs)
        answers.append(answer_with_context(llm, q, "\n\n".join(ctxs)))
        if i % 10 == 0 or i == len(questions):
            print(f"  进度 {i}/{len(questions)}")

    dataset = construct_rag_dataset(questions, references, retrieved_contexts, answers)
    print(f"  ragas 评测中...")
    df = get_evaluation_metrics(dataset, llm, embedding)
    df["question"] = questions
    df["reference"] = references
    df["answer"] = answers
    df["collection"] = collection
    df["mode"] = mode
    return df


def main():
    out_dir = OPT_RES_DIR / "query_opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    print(f"评测集：{len(testset)} 题 | top_k={TOP_K}")

    llm = get_llm()          # 评测/问答
    idx_llm = get_index_llm()  # 重写/HyDE 生成
    embedding = get_embedding_model()

    summary_rows = []
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    variants = [
        ("rewrite", HYPE_COLLECTION),
        ("route", HYPE_COLLECTION),
        ("hyde", MD_COLLECTION),
    ]
    for mode, collection in variants:
        df = evaluate(mode, collection, testset, llm, idx_llm, embedding, TOP_K)
        out = out_dir / f"eval_{mode}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存：{out}")
        row = {"mode": mode, "collection": collection}
        for c in metric_cols:
            row[c] = round(float(df[c].mean()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "对比汇总.csv", index=False, encoding="utf-8-sig")
    print("\n===== query 优化方案平均指标 =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
