"""用 30 题评测 4 种上下文优化，与基线（混合+RRF）对比。

基线（R0）：res/optimization/hybrid/eval_rrf.csv（复用）
上下文优化（检索完全相同，仅改变送入 LLM 的 context）：
    C1 RSE 段提取        -> context/eval_rse.csv
    C2 上下文窗口增强     -> context/eval_window.csv
    C3 上下文压缩(LLM)   -> context/eval_compress.csv
    C4 U型排序           -> context/eval_ushape.csv

输出：res/optimization/context/
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
from utils.context_opt import (
    build_chunk_map,
    compress_context,
    rse_context,
    ushape_context,
    window_context,
)
from utils.eval_rag import answer_with_context
from utils.hybrid import BM25Index, hybrid_retrieve
from utils.llm_utils import get_llm, get_embedding_model, get_index_llm
from utils.query_opt import classify_and_rewrite

HYPE_MULTIPLIER = 4
CAND_K = 30
BASE_CHUNK_SIZE = 1500
BASE_CHUNK_OVERLAP = 200

METHODS = ["rse", "window", "compress", "ushape"]


def routed_queries(testset, llm):
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


def apply_context(method, query, docs, chunk_map, idx_llm):
    if method == "rse":
        return rse_context(docs, chunk_map)
    if method == "window":
        return window_context(docs, chunk_map, radius=1)
    if method == "compress":
        return compress_context(query, docs, idx_llm)
    if method == "ushape":
        return ushape_context(docs, top_n=TOP_K)
    raise ValueError(method)


def evaluate(method, testset, sqs, llm, idx_llm, embedding, vs, bm25, chunk_map, k):
    print(f"\n===== 评测：上下文-{method} =====")
    dense_fn = build_dense_fn(vs)
    questions = [t["question"] for t in testset]
    references = [t["answer"] for t in testset]
    retrieved_contexts, answers = [], []
    for i, (q, sq) in enumerate(zip(questions, sqs), 1):
        docs = hybrid_retrieve(sq, dense_fn, bm25, "rrf", cand_k=CAND_K,
                               top_k=k, llm=idx_llm)
        ctx_docs = apply_context(method, q, docs, chunk_map, idx_llm)
        ctxs = [d.page_content for d in ctx_docs]
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
    df["context_method"] = method
    return df


def main():
    out_dir = OPT_RES_DIR / "context"
    out_dir.mkdir(parents=True, exist_ok=True)
    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    print(f"评测集：{len(testset)} 题 | top_k={TOP_K}")

    llm = get_llm()
    idx_llm = get_index_llm()
    embedding = get_embedding_model()
    sqs = routed_queries(testset, idx_llm)

    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    base = base_chunks(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
    bm25 = BM25Index(base)
    chunk_map = build_chunk_map(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
    print(f"BM25 索引 {len(base)} 块 | chunk_map {len(chunk_map)} 条")

    vs = Milvus(embedding_function=embedding, collection_name=HYPE_COLLECTION,
                connection_args={"uri": MILVUS_URI, "db_name": DB_NAME})

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary_rows = []
    for method in METHODS:
        out = out_dir / f"eval_{method}.csv"
        if out.exists():
            print(f"[跳过] {method}：{out} 已存在")
            df = pd.read_csv(out, encoding="utf-8-sig")
            row = {"context_method": method}
            for c in metric_cols:
                row[c] = round(float(df[c].mean()), 4)
            summary_rows.append(row)
            continue
        df = evaluate(method, testset, sqs, llm, idx_llm, embedding, vs, bm25, chunk_map, TOP_K)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存：{out}")
        row = {"context_method": method}
        for c in metric_cols:
            row[c] = round(float(df[c].mean()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "对比汇总.csv", index=False, encoding="utf-8-sig")
    print("\n===== 上下文优化 平均指标 =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
