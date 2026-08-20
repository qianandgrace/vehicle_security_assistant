"""生成上下文优化对比报告（合并基线 混合+RRF 与 4 种上下文优化）。

读取：
    res/optimization/hybrid/eval_rrf.csv          R0 基线 混合+RRF
    res/optimization/context/eval_rse.csv         C1 RSE 段提取
    res/optimization/context/eval_window.csv      C2 上下文窗口增强
    res/optimization/context/eval_compress.csv    C3 上下文压缩
    res/optimization/context/eval_ushape.csv      C4 U型排序

输出：res/optimization/context/对比报告.md
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

PLAN = [
    ("R0 基线 混合+RRF", OPT_RES_DIR / "hybrid" / "eval_rrf.csv"),
    ("C1 RSE 段提取", OPT_RES_DIR / "context" / "eval_rse.csv"),
    ("C2 上下文窗口增强", OPT_RES_DIR / "context" / "eval_window.csv"),
    ("C3 上下文压缩", OPT_RES_DIR / "context" / "eval_compress.csv"),
    ("C4 U型排序", OPT_RES_DIR / "context" / "eval_ushape.csv"),
]


def main():
    frames = []
    for name, path in PLAN:
        if not path.exists():
            print(f"[跳过] {name}：缺少 {path}")
            continue
        frames.append((name, pd.read_csv(path, encoding="utf-8-sig")))
    if not frames:
        raise FileNotFoundError("没有任何评测结果可读")

    rows = []
    for name, df in frames:
        row = {"方案": name}
        for c, _ in METRICS:
            row[c] = round(float(df[c].mean()), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)

    lines = []
    lines.append("# 上下文优化方案对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测集**：30 个问答对（覆盖 16 个新能源汽车标准）")
    lines.append(f"- **检索基线**：HyPE + BM25 + RRF 重排（上一步最优），top_k={TOP_K}")
    lines.append(f"- **上下文优化**：四种方案均在检索结果之上、送 LLM 生成之前做后处理")
    lines.append(f"- **C1 RSE**：把连续位置的相关块合并成段，补上被夹在中间的块")
    lines.append(f"- **C2 窗口增强**：为每个检索块加入前后相邻块（radius=1）")
    lines.append(f"- **C3 上下文压缩**：LLM 提取每个块中与问题相关的部分，剔除无关块")
    lines.append(f"- **C4 U型排序**：最高分放开头、次高分放结尾，缓解 lost-in-the-middle")
    lines.append(f"- **评测**：ragas；Embedding：bge-large-zh-v1.5\n")
    lines.append("## 2. 平均指标对比\n")
    lines.append("| 方案 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['方案']} | {r['faithfulness']:.4f} | {r['answer_relevancy']:.4f} "
            f"| {r['context_precision']:.4f} | {r['context_recall']:.4f} |"
        )

    lines.append("\n## 3. 与基线（R0）对比\n")
    base = summary.iloc[0]
    for _, r in summary.iterrows():
        if r["方案"] == base["方案"]:
            continue
        diffs = []
        for c, _ in METRICS:
            d = r[c] - base[c]
            diffs.append(("▲" if d > 0 else ("▼" if d < 0 else "=")) + f"{abs(d):.3f}")
        lines.append(f"- **{r['方案']}** vs R0：忠实度{diffs[0]} / 相关性{diffs[1]} / 精确率{diffs[2]} / 召回率{diffs[3]}")

    lines.append("\n## 4. 逐指标最优方案\n")
    lines.append("| 指标 | 最优方案 | 数值 |")
    lines.append("|---|---|---|")
    for c, name in METRICS:
        best = summary.loc[summary[c].idxmax()]
        lines.append(f"| {name} | {best['方案']} | {best[c]:.4f} |")

    lines.append("\n## 5. 结论\n")
    lines.append("- **基线（混合+RRF）仍是最均衡方案**：忠实度 0.9544 为最高，其余指标也都较强。4 种上下文优化大多是在指标间做取舍，没有一种全面超越。")
    lines.append("- **U型排序（C4）是安全且零成本的增益**：相关性 +0.010，其余基本持平，无需额外模型/LLM 开销。作为默认上下文排列方式推荐。")
    lines.append("- **RSE 段提取（C1）相关性最优**（0.8643，+0.013），但精确率/召回率略降；适合更关注答案相关性的场景。")
    lines.append("- **窗口增强（C2）召回率拉满**（1.0000，+0.033），但加入邻居带来噪音使精确率明显下降（-0.13，且该指标有 11/30 缺失、结果不可靠）；适合召回优先且能容忍噪音的场景。")
    lines.append("- **上下文压缩（C3）高精确率但忠实度崩塌**：精确率 0.9380（+0.123）为最优，但压缩去掉了支撑信息，忠实度暴跌至 0.6894（-0.265）——压缩后的答案失去原文依据，不推荐当前实现。")
    lines.append("- **最终建议**：检索层保持 HyPE+BM25+RRF；上下文层默认采用 **U型排序**；若追求答案相关性可叠加 RSE；压缩需改进提示词（保留关键数值与限定）后再评估。")
    lines.append("\n## 6. 逐题明细\n")
    lines.append("见 `res/optimization/context/eval_*.csv`。")

    out = OPT_RES_DIR / "context" / "对比报告.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
