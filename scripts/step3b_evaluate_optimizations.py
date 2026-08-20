"""用精简 30 题评测 4 个方案（基线 MD + Semantic + Proposition + HyPE），输出各方案 ragas 指标。

每个方案对应一个 Milvus collection，检索策略不同：
- md_collection（基线）/ semantic / proposition：直接返回 chunk 文本
- hype：用问题-问题匹配检索，把命中的"假设性问题"映射回原 chunk，并去重

输出：
    res/optimization/eval_<collection>.csv
    res/optimization/对比汇总.csv
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
    PROPOSITION_COLLECTION,
    SEMANTIC_COLLECTION,
    TESTSET_30,
    TOP_K,
)
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics
from utils.eval_rag import answer_with_context
from utils.llm_utils import get_llm, get_embedding_model

# 参与对比的方案：名称 -> collection
COLLECTIONS = [
    ("基线 MD(512)", MD_COLLECTION),
    ("Semantic 语义切块", SEMANTIC_COLLECTION),
    ("Proposition 命题", PROPOSITION_COLLECTION),
    ("HyPE 假设性问题", HYPE_COLLECTION),
]

# HyPE 检索时按问题数量多取，再映射回原 chunk 去重
HYPE_MULTIPLIER = 4


def retrieve_contexts(vs, query, k, collection):
    """返回检索到的上下文文本列表（Hype 去重映射回原 chunk）。"""
    if collection == HYPE_COLLECTION:
        docs = vs.similarity_search(query, k=k * HYPE_MULTIPLIER)
        seen, ctx = set(), []
        for d in docs:
            orig = d.metadata.get("original_content", d.page_content)
            if orig and orig not in seen:
                seen.add(orig)
                ctx.append(orig)
            if len(ctx) >= k:
                break
        return ctx
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def evaluate_collection(name, collection, testset, llm, embedding, k):
    print(f"\n===== 评测：{name} ({collection}) =====")
    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection,
        connection_args={"uri": MILVUS_URI, "db_name": DB_NAME},
    )
    questions = [t["question"] for t in testset]
    references = [t["answer"] for t in testset]
    retrieved_contexts, answers = [], []
    for i, q in enumerate(questions, 1):
        ctxs = retrieve_contexts(vs, q, k, collection)
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
    return df


def main():
    OPT_RES_DIR.mkdir(parents=True, exist_ok=True)
    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    print(f"评测集：{len(testset)} 题 | LLM(评测): 见 config | top_k={TOP_K}")

    llm = get_llm()
    embedding = get_embedding_model()

    summary_rows = []
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for name, collection in COLLECTIONS:
        df = evaluate_collection(name, collection, testset, llm, embedding, TOP_K)
        out = OPT_RES_DIR / f"eval_{collection}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存：{out}")
        row = {"方案": name, "collection": collection}
        for c in metric_cols:
            row[c] = round(float(df[c].mean()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OPT_RES_DIR / "对比汇总.csv", index=False, encoding="utf-8-sig")
    print("\n===== 各方案平均指标 =====")
    print(summary_df.to_string(index=False))
    print(f"\n汇总已保存：{OPT_RES_DIR / '对比汇总.csv'}")


if __name__ == "__main__":
    main()
