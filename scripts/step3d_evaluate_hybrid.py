"""用 30 题评测混合检索（HyPE 稠密 + BM25 稀疏）+ 4 种重排。

与基线（HyPE + 意图路由，P3）保持一致：查询先经意图路由处理，再送入检索。
对比矩阵：
    R0 基线 HyPE+意图路由（复用 res/optimization/query_opt/eval_route.csv）
    R1 混合 + RRF
    R2 混合 + CrossEncoder(bce-reranker)
    R3 混合 + RankLLM
    R4 混合 + ColBERT(bge-m3)

输出：res/optimization/hybrid/eval_<method>.csv
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_milvus import Milvus

from config import (
    DB_NAME,
    HYPE_COLLECTION,
    MILVUS_URI,
    OPT_RES_DIR,
    TESTSET_30,
    TOP_K,
    TRANSFER_DIR,
)
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics
from utils.chunking import base_chunks
from utils.eval_rag import answer_with_context
from utils.hybrid import BM25Index, hybrid_retrieve
from utils.llm_utils import get_llm, get_embedding_model, get_index_llm
from utils.query_opt import classify_and_rewrite

HYPE_MULTIPLIER = 4
CAND_K = 30
BASE_CHUNK_SIZE = 1500
BASE_CHUNK_OVERLAP = 200

METHODS = ["rrf", "crossencoder", "rankllm", "colbert"]


def routed_queries(testset, llm):
    """意图路由：仅对需重写的查询改写（与基线 P3 一致）。"""
    out = []
    for i, it in enumerate(testset, 1):
        c = classify_and_rewrite(it["question"], llm)
        out.append(c["rewritten_query"] if c["need_rewrite"] else it["question"])
        if i % 10 == 0 or i == len(testset):
            print(f"  路由预处理 {i}/{len(testset)}")
    return out


def build_dense_fn(vs):
    def dense_fn(query, k):
        docs = vs.similarity_search(query, k=k * HYPE_MULTIPLIER)
        seen, out = set(), []
        for d in docs:
            orig = d.metadata.get("original_content", d.page_content)
            if orig and orig not in seen:
                seen.add(orig)
                out.append(Document(page_content=orig))
            if len(out) >= k:
                break
        return out
    return dense_fn


def evaluate(method, testset, sqs, llm, idx_llm, embedding, vs, bm25, k):
    print(f"\n===== 评测：混合+{method} =====")
    dense_fn = build_dense_fn(vs)
    questions = [t["question"] for t in testset]
    references = [t["answer"] for t in testset]
    retrieved_contexts, answers = [], []
    for i, (q, sq) in enumerate(zip(questions, sqs), 1):
        docs = hybrid_retrieve(sq, dense_fn, bm25, method, cand_k=CAND_K,
                               top_k=k, llm=idx_llm)
        ctxs = [d.page_content for d in docs]
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
    df["method"] = method
    return df


def main():
    out_dir = OPT_RES_DIR / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    print(f"评测集：{len(testset)} 题 | top_k={TOP_K} | cand_k={CAND_K}")

    llm = get_llm()
    idx_llm = get_index_llm()
    embedding = get_embedding_model()

    # 意图路由（与基线 P3 一致的查询处理）
    sqs = routed_queries(testset, idx_llm)

    # BM25 稀疏索引（与 HyPE 相同的基块）
    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    base = base_chunks(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
    bm25 = BM25Index(base)
    print(f"BM25 索引：{len(base)} 块")

    # HyPE 稠密检索
    vs = Milvus(embedding_function=embedding, collection_name=HYPE_COLLECTION,
                connection_args={"uri": MILVUS_URI, "db_name": DB_NAME})

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary_rows = []
    for method in METHODS:
        df = evaluate(method, testset, sqs, llm, idx_llm, embedding, vs, bm25, TOP_K)
        out = out_dir / f"eval_{method}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存：{out}")
        row = {"method": method}
        for c in metric_cols:
            row[c] = round(float(df[c].mean()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "对比汇总.csv", index=False, encoding="utf-8-sig")
    print("\n===== 混合检索+重排 平均指标 =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
