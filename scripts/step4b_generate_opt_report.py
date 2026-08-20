"""根据 step3b 的评测结果生成优化方案对比报告。

读取 res/optimization/eval_*.csv（每方案一份），汇总指标均值，
输出 res/optimization/对比报告.md。
"""
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OPT_RES_DIR, TOP_K

METRICS = [
    ("faithfulness", "忠实度 Faithfulness"),
    ("answer_relevancy", "答案相关性 Answer Relevancy"),
    ("context_precision", "上下文精确率 Context Precision"),
    ("context_recall", "上下文召回率 Context Recall"),
]

# 方案展示名映射
NAME_MAP = {
    "md_collection": "基线 MD(512)",
    "md_semantic_collection": "Semantic 语义切块",
    "md_proposition_collection": "Proposition 命题",
    "md_hype_collection": "HyPE 假设性问题",
}


def main():
    if not OPT_RES_DIR.exists():
        raise FileNotFoundError("缺少评测结果，请先运行 scripts/step3b_evaluate_optimizations.py")

    files = sorted(OPT_RES_DIR.glob("eval_*.csv"))
    frames = {f.stem[len("eval_"):]: pd.read_csv(f, encoding="utf-8-sig") for f in files}

    # 汇总均值
    rows = []
    for coll, df in frames.items():
        row = {"collection": coll, "方案": NAME_MAP.get(coll, coll)}
        for c, _ in METRICS:
            row[c] = round(float(df[c].mean()), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)

    total = len(next(iter(frames.values())))
    lines = []
    lines.append("# RAG 分块/索引优化方案对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测集**：30 个问答对（精选自 117 题，覆盖 16 个新能源汽车标准）")
    lines.append(f"- **检索**：Milvus（db=vehicle），similarity_search，top_k={TOP_K}")
    lines.append(f"- **Embedding**：BAAI/bge-large-zh-v1.5（1024 维）")
    lines.append(f"- **评测**：ragas（Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall）")
    lines.append(f"- **LLM(评测/问答)**：deepseek-v4-flash；**LLM(Proposition/HyPE 索引生成)**：qwen-max\n")
    lines.append("## 2. 平均指标对比\n")
    lines.append("| 方案 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['方案']} | {r['faithfulness']:.4f} | {r['answer_relevancy']:.4f} "
            f"| {r['context_precision']:.4f} | {r['context_recall']:.4f} |"
        )

    lines.append("\n## 3. 逐指标最优方案\n")
    lines.append("| 指标 | 最优方案 | 数值 |")
    lines.append("|---|---|---|")
    for c, name in METRICS:
        best = summary.loc[summary[c].idxmax()]
        lines.append(f"| {name} | {best['方案']} | {best[c]:.4f} |")

    lines.append("\n## 4. 结论\n")
    lines.append("各方案在不同指标上各有优劣，综合表现见上表。相对于基线 MD(512)：\n")
    # 与基线对比
    base_row = summary[summary["collection"] == "md_collection"]
    if len(base_row) == 1:
        base = base_row.iloc[0]
        for _, r in summary.iterrows():
            if r["collection"] == "md_collection":
                continue
            diffs = []
            for c, _ in METRICS:
                d = r[c] - base[c]
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
                diffs.append(f"{arrow}{abs(d):.3f}")
            lines.append(f"- **{r['方案']}** vs 基线：忠实度{diffs[0]} / 相关性{diffs[1]} / 精确率{diffs[2]} / 召回率{diffs[3]}")
    lines.append("\n## 5. 逐题明细\n")
    lines.append("完整逐题数据见 `res/optimization/eval_*.csv`。")

    out = OPT_RES_DIR / "对比报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
