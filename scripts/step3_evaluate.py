"""第三步：用第一步生成的测试集，对 PDF / MD 两个集合跑 RAG 并用 ragas 对比评测。

流程（对 pdf_collection 与 md_collection 各跑一遍）：
    1. 对每个问题做向量检索 top_k，得到 retrieved_contexts
    2. 用同一份 context 让 LLM 生成答案
    3. 用 ragas 计算 Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall
    4. 保存结果 CSV（res/ 目录），并打印两个集合的平均分对比

输出：
    res/eval_pdf.csv
    res/eval_md.csv
    res/对比汇总.csv
"""
import json
import sys
from pathlib import Path

# 避免 Windows 控制台 GBK 编码打印中文/特殊字符时崩溃
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_milvus import Milvus

from config import (
    DB_NAME,
    EVAL_MD_CSV,
    EVAL_PDF_CSV,
    LLM_MODEL,
    MD_COLLECTION,
    MILVUS_URI,
    PDF_COLLECTION,
    REPORT_SUMMARY_CSV,
    RES_DIR,
    TESTSET_JSON,
    TOP_K,
)
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics
from utils.eval_rag import answer_with_context, format_docs
from utils.llm_utils import get_llm, get_embedding_model


def evaluate_collection(collection_name: str, testset, llm, embedding, top_k: int):
    """对一个集合跑完整 RAG + ragas 评测，返回 DataFrame。"""
    print(f"\n===== 评测集合：{collection_name} =====")
    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={"uri": MILVUS_URI, "db_name": DB_NAME},
    )

    questions = [item["question"] for item in testset]
    references = [item["answer"] for item in testset]

    retrieved_contexts = []
    answers = []
    for i, q in enumerate(questions, 1):
        docs = vs.similarity_search(q, k=top_k)
        contexts = [d.page_content for d in docs]
        answer = answer_with_context(llm, q, format_docs(docs))
        retrieved_contexts.append(contexts)
        answers.append(answer)
        if i % 10 == 0 or i == len(questions):
            print(f"  进度 {i}/{len(questions)}")

    dataset = construct_rag_dataset(questions, references, retrieved_contexts, answers)
    print(f"  开始 ragas 评测（{len(questions)} 个问题）...")
    result_df = get_evaluation_metrics(dataset, llm, embedding)
    result_df["question"] = questions
    result_df["reference"] = references
    result_df["answer"] = answers
    result_df["collection"] = collection_name
    return result_df


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TESTSET_JSON, "r", encoding="utf-8") as fp:
        testset = json.load(fp)
    print(f"测试集规模：{len(testset)} 个问题 | LLM: {LLM_MODEL}")

    llm = get_llm()
    embedding = get_embedding_model()

    pdf_df = evaluate_collection(PDF_COLLECTION, testset, llm, embedding, TOP_K)
    pdf_df.to_csv(EVAL_PDF_CSV, index=False, encoding="utf-8-sig")
    print(f"\nPDF 评测结果已保存：{EVAL_PDF_CSV}")

    md_df = evaluate_collection(MD_COLLECTION, testset, llm, embedding, TOP_K)
    md_df.to_csv(EVAL_MD_CSV, index=False, encoding="utf-8-sig")
    print(f"MD 评测结果已保存：{EVAL_MD_CSV}")

    # ---- 对比汇总 ----
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {}
    for name, df in [("PDF", pdf_df), ("MD", md_df)]:
        row = {}
        for col in metric_cols:
            if col in df.columns:
                row[col] = round(float(df[col].mean()), 4)
        summary[name] = row
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(REPORT_SUMMARY_CSV, index=True, encoding="utf-8-sig")
    print(f"\n对比汇总已保存：{REPORT_SUMMARY_CSV}")

    print("\n===== 对比结果（各指标均值）=====")
    print(summary_df.to_string())
    print("\n第三步完成")


if __name__ == "__main__":
    main()
