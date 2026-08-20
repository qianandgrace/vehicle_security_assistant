"""评测不同 LLM 在当前最终 pipeline（HyPE+路由+BM25+RRF+U型）下的 RAG 效果。

用于"三模型对比"：deepseek 云 / 本地 qwen2:7b（未微调）/ 微调后 qwen2:7b。
评测裁判固定用 DeepSeek（保证公平），检索完全相同，只有答案生成模型不同。

用法：
    python scripts/step3f_evaluate_models.py deepseek
    python scripts/step3f_evaluate_models.py ollama
    python scripts/step3f_evaluate_models.py ollama-finetuned
输出：res/models/eval_<llm_type>.csv
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RES_DIR, TESTSET_30, TOP_K
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics
from utils.llm_utils import get_llm, get_embedding_model
from utils.peft_llm import answer_with_peft
from utils.rag_pipeline import VehicleRAG

# 评测裁判（固定 deepseek）
JUDGE_LLM = get_llm()


def main():
    llm_type = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    out_dir = RES_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    print(f"评测 {llm_type} | {len(testset)} 题 | 裁判=deepseek | top_k={TOP_K}")

    # 检索用同一 pipeline（任意 llm_type 均可，检索与生成模型无关）
    pipe = VehicleRAG(llm_type="ollama", top_k=TOP_K)
    embedding = get_embedding_model()

    questions = [t["question"] for t in testset]
    references = [t["answer"] for t in testset]
    retrieved_contexts, answers = [], []
    for i, q in enumerate(questions, 1):
        docs, _ = pipe.retrieve(q, top_k=TOP_K)
        ctx = [d.page_content for d in docs]
        retrieved_contexts.append(ctx)
        if llm_type == "peft":
            answers.append(answer_with_peft(q, "\n\n".join(ctx)))
        else:
            out = pipe.answer(q, top_k=TOP_K)
            answers.append(out["answer"])
        if i % 10 == 0 or i == len(questions):
            print(f"  进度 {i}/{len(questions)}")

    dataset = construct_rag_dataset(questions, references, retrieved_contexts, answers)
    print("  ragas 评测中（裁判 deepseek）...")
    df = get_evaluation_metrics(dataset, JUDGE_LLM, embedding)
    df["question"] = questions
    df["reference"] = references
    df["answer"] = answers
    df["model"] = llm_type

    out = out_dir / f"eval_{llm_type}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"已保存：{out}")
    for c in cols:
        print(f"  {c}: {df[c].mean():.4f}")


if __name__ == "__main__":
    main()
