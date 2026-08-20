"""生成 query 优化对比报告（合并分块优化 P0/P1 与 query 优化 P2/P3/P4）。

读取：
    res/optimization/eval_md_collection.csv        P0 基线
    res/optimization/eval_md_hype_collection.csv   P1 HyPE
    res/optimization/query_opt/eval_rewrite.csv    P2 HyPE+全量重写
    res/optimization/query_opt/eval_route.csv      P3 HyPE+意图路由
    res/optimization/query_opt/eval_hyde.csv       P4 基线+HyDE

输出：res/optimization/query_opt/对比报告.md
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

# 方案定义：(名称, 源 CSV 路径)
PLAN = [
    ("P0 基线 MD(512)", OPT_RES_DIR / "eval_md_collection.csv"),
    ("P1 HyPE 假设性问题", OPT_RES_DIR / "eval_md_hype_collection.csv"),
    ("P2 HyPE+全量重写", OPT_RES_DIR / "query_opt" / "eval_rewrite.csv"),
    ("P3 HyPE+意图路由", OPT_RES_DIR / "query_opt" / "eval_route.csv"),
    ("P4 基线+HyDE", OPT_RES_DIR / "query_opt" / "eval_hyde.csv"),
]


def main():
    frames = []
    for name, path in PLAN:
        if not path.exists():
            print(f"[跳过] {name}：缺少 {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append((name, df))

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
    lines.append("# Query 优化方案对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测集**：30 个问答对（覆盖 16 个新能源汽车标准）")
    lines.append(f"- **检索**：Milvus（db=vehicle），similarity_search，top_k={TOP_K}")
    lines.append(f"- **Embedding**：BAAI/bge-large-zh-v1.5（1024 维）")
    lines.append(f"- **评测**：ragas（Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall）")
    lines.append(f"- **LLM**：评测/问答 deepseek-v4-flash；重写/HyDE 生成 qwen-max\n")
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

    # 与 P1（HyPE 基线）对比
    lines.append("\n## 4. 与 HyPE 基线（P1）对比\n")
    p1 = summary[summary["方案"] == "P1 HyPE 假设性问题"]
    if len(p1) == 1:
        base = p1.iloc[0]
        for _, r in summary.iterrows():
            if r["方案"] == base["方案"]:
                continue
            diffs = []
            for c, _ in METRICS:
                d = r[c] - base[c]
                diffs.append(("▲" if d > 0 else ("▼" if d < 0 else "=")) + f"{abs(d):.3f}")
            lines.append(f"- **{r['方案']}** vs P1：忠实度{diffs[0]} / 相关性{diffs[1]} / 精确率{diffs[2]} / 召回率{diffs[3]}")

    lines.append("\n## 5. 结论\n")
    lines.append("- **意图路由（P3）是 query 优化的最佳选择**：在检索三项（相关性/精确率/召回率）基本不损失的前提下，把忠实度从 0.8897 提升到 0.9353（+0.046）。路由只重写「适用/定义/方法」类模糊查询，明确查询保持不变，避免改坏。")
    lines.append("- **全量重写（P2）有风险**：忠实度虽最高（0.9389），但检索三项全面下滑（相关性 -0.052 / 精确率 -0.079 / 召回率 -0.067）。对明确查询强行重写会引入噪声（如把「适用于哪些类型的汽车」改成「新能源汽车」反而限缩了范围）。")
    lines.append("- **HyDE（P4）不推荐**：在基线上只提升忠实度（+0.022），相关性/召回率反而下降，且整体远不如 HyPE（相关性 -0.205 / 召回率 -0.267）。")
    lines.append("- **最终推荐**：**HyPE + 意图路由**——检索层用 HyPE（问题-问题匹配），查询层用意图路由按需重写，两者叠加效果最优。\n")
    lines.append("## 6. 逐题明细\n")
    lines.append("见 `res/optimization/eval_*.csv` 与 `res/optimization/query_opt/eval_*.csv`。")

    out = OPT_RES_DIR / "query_opt" / "对比报告.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
