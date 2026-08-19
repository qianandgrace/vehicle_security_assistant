"""第四步：根据 PDF / MD 两份评测结果生成对比报告。

读取 res/eval_pdf.csv 与 res/eval_md.csv（由 step3 生成），
计算各指标均值、逐题胜负，输出：
    res/对比报告.md      （Markdown 报告）
    res/对比报告_per_question.csv （逐题对比明细）
"""
import sys
from datetime import datetime
from pathlib import Path

# 避免 Windows 控制台 GBK 编码打印中文/特殊字符时崩溃
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EVAL_MD_CSV,
    EVAL_PDF_CSV,
    LLM_MODEL,
    REPORT_MD,
    RES_DIR,
    TOP_K,
)

METRICS = [
    ("faithfulness", "忠实度 Faithfulness"),
    ("answer_relevancy", "答案相关性 Answer Relevancy"),
    ("context_precision", "上下文精确率 Context Precision"),
    ("context_recall", "上下文召回率 Context Recall"),
]


def load_and_align():
    pdf = pd.read_csv(EVAL_PDF_CSV, encoding="utf-8-sig")
    md = pd.read_csv(EVAL_MD_CSV, encoding="utf-8-sig")
    # 对齐问题（两份结果应同序，仍按 question 对齐更稳）
    pdf = pdf.set_index("question")
    md = md.set_index("question")
    common = pdf.index.intersection(md.index)
    return pdf.loc[common], md.loc[common]


def build_summary(pdf, md) -> pd.DataFrame:
    rows = []
    for col, name in METRICS:
        p = float(pdf[col].mean())
        m = float(md[col].mean())
        rows.append({
            "指标": name,
            "PDF 均值": round(p, 4),
            "MD 均值": round(m, 4),
            "差值 (MD-PDF)": round(m - p, 4),
            "占优": "MD" if m > p else ("PDF" if p > m else "持平"),
        })
    return pd.DataFrame(rows)


def per_question_wins(pdf, md) -> pd.DataFrame:
    cols = [c for c, _ in METRICS]
    data = {"question": pdf.index.tolist()}
    for col in cols:
        data[f"{col}_PDF"] = pdf[col].values
        data[f"{col}_MD"] = md[col].values
        data[f"{col}_winner"] = ["MD" if m > p else ("PDF" if p > m else "=")
                                 for p, m in zip(pdf[col].values, md[col].values)]
    return pd.DataFrame(data)


def build_conclusion(summary):
    lines = []
    lines.append("基于上表均值对比：\n")
    for _, r in summary.iterrows():
        diff = r["差值 (MD-PDF)"]
        tag = "优于" if diff > 0 else ("劣于" if diff < 0 else "持平于")
        lines.append(f"- **{r['指标']}**：MD {tag} PDF（{r['PDF 均值']:.4f} vs {r['MD 均值']:.4f}，差值 {diff:+.4f}）")
    return "\n".join(lines)


def build_report(pdf, md, summary, perq) -> str:
    total = len(pdf)
    lines = []
    lines.append("# PDF 与 MD 作为 RAG 知识源的对比评测报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测框架**：ragas 0.4.3（`evaluate` + Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall）")
    lines.append(f"- **LLM**：{LLM_MODEL}")
    lines.append(f"- **Embedding**：BAAI/bge-large-zh-v1.5（1024 维）")
    lines.append(f"- **向量库**：Milvus，库 `vehicle`，集合 `pdf_collection`（原始 PDF）与 `md_collection`（MinerU 转换 MD）")
    lines.append(f"- **分块**：chunk_size={CHUNK_SIZE}，overlap={CHUNK_OVERLAP}；**检索**：similarity_search，top_k={TOP_K}")
    lines.append(f"- **测试集**：{total} 个问答对，覆盖 16 个新能源汽车标准文档（GB/GBT/QCT）\n")
    lines.append("## 2. 平均指标对比\n")
    lines.append("| 指标 | PDF 均值 | MD 均值 | 差值 (MD-PDF) | 占优 |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['指标']} | {r['PDF 均值']:.4f} | {r['MD 均值']:.4f} | {r['差值 (MD-PDF)']:+.4f} | {r['占优']} |")
    lines.append("\n## 3. 逐指标胜场统计\n")
    lines.append("| 指标 | PDF 胜 | MD 胜 | 持平 |")
    lines.append("|---|---|---|---|")
    for col, name in METRICS:
        counts = perq[f"{col}_winner"].value_counts()
        lines.append(f"| {name} | {counts.get('PDF', 0)} | {counts.get('MD', 0)} | {counts.get('=', 0)} |")
    lines.append("\n## 4. 结论\n")
    lines.append(build_conclusion(summary))
    lines.append("\n### 解读\n")
    lines.append("- **检索能力（召回/精确）**：MD 在上下文召回率上小幅领先（+0.017），上下文精确率基本持平，说明 MinerU 处理过的表格、图片转文字等信息，让检索命中正确答案略好；但两者都未拉开显著差距。")
    lines.append("- **生成质量（忠实度/相关性）**：PDF 的忠实度明显更高（-0.042），MD 的答案相关性略高（+0.017）。可能原因：PDF 抽取的文本更“原汁原味”，模型倾向直接引用原文，因此更忠实；MD 文本更通顺，模型回答更流畅、相关性更好，但也更容易产生上下文未直接支持的表述。")
    lines.append("- **总体**：MD 作为知识源在检索与答案相关性上略优于 PDF，但差距不大；若更看重答案忠实度，PDF 更稳。可在后续工作中尝试以 MD 为主、结合 PDF 原文纠偏的混合方案，或针对 MD 增加“仅依据上下文回答”的约束提示来提升忠实度。")
    lines.append("\n## 5. 逐题明细\n")
    lines.append("完整逐题对比见 `res/对比报告_per_question.csv`。")
    return "\n".join(lines)


def main():
    if not EVAL_PDF_CSV.exists() or not EVAL_MD_CSV.exists():
        raise FileNotFoundError("缺少评测结果文件，请先运行 scripts/step3_evaluate.py")
    RES_DIR.mkdir(parents=True, exist_ok=True)

    pdf, md = load_and_align()
    summary = build_summary(pdf, md)
    perq = per_question_wins(pdf, md)

    report = build_report(pdf, md, summary, perq)
    REPORT_MD.write_text(report, encoding="utf-8")
    perq.to_csv(RES_DIR / "对比报告_per_question.csv", index=False, encoding="utf-8-sig")

    print("===== 指标均值对比 =====")
    print(summary.to_string(index=False))
    print(f"\n报告已生成：{REPORT_MD}")
    print(f"逐题明细已生成：{RES_DIR / '对比报告_per_question.csv'}")


if __name__ == "__main__":
    main()
