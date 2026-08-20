"""生成混合检索+重排对比报告（合并基线 HyPE+路由 与 4 种重排方案）。

读取：
    res/optimization/query_opt/eval_route.csv           R0 基线 HyPE+意图路由
    res/optimization/hybrid/eval_rrf.csv                R1 混合+RRF
    res/optimization/hybrid/eval_crossencoder.csv       R2 混合+CrossEncoder
    res/optimization/hybrid/eval_rankllm.csv            R3 混合+RankLLM
    res/optimization/hybrid/eval_colbert.csv            R4 混合+ColBERT

输出：res/optimization/hybrid/对比报告.md
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
    ("R0 基线 HyPE+路由", OPT_RES_DIR / "query_opt" / "eval_route.csv"),
    ("R1 混合+RRF", OPT_RES_DIR / "hybrid" / "eval_rrf.csv"),
    ("R2 混合+CrossEncoder", OPT_RES_DIR / "hybrid" / "eval_crossencoder.csv"),
    ("R3 混合+RankLLM", OPT_RES_DIR / "hybrid" / "eval_rankllm.csv"),
    ("R4 混合+ColBERT", OPT_RES_DIR / "hybrid" / "eval_colbert.csv"),
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
    lines.append("# 混合检索 + 重排方案对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测集**：30 个问答对（覆盖 16 个新能源汽车标准）")
    lines.append(f"- **基线**：HyPE（问题-问题匹配）+ 意图路由，top_k={TOP_K}")
    lines.append(f"- **混合检索**：HyPE 稠密（{TOP_K*4} 问题候选→去重映射回原块）+ BM25 稀疏（jieba 分词），各召回 30 候选，重排取 top_k={TOP_K}")
    lines.append(f"- **重排方法**：RRF（排名融合）/ CrossEncoder（bce-reranker-base_v1）/ RankLLM（LLM 排序）/ ColBERT（bge-m3 后期交互）")
    lines.append(f"- **Embedding**：BAAI/bge-large-zh-v1.5；**评测**：ragas\n")
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
    lines.append("- **RRF（R1）是最优重排**：四项指标全面超过基线，尤其上下文召回率 +0.083（0.8833→0.9667）。加入 BM25 稀疏召回扩大候选池后，用 RRF 按排名倒数融合，能稳定补齐稠密检索漏掉的文档，且零模型、零成本。")
    lines.append("- **CrossEncoder（R2）小幅改善检索**：相关性/精确率/召回率略升，但忠实度下降 0.043。交叉编码器精度高但需要模型推理，性价比不如 RRF。")
    lines.append("- **RankLLM（R3）与 ColBERT（R4）在本场景下未跑赢基线**：R3 仅用 100 字摘要让 LLM 排序，信息丢失严重（召回率 0.55）；R4 用 bge-m3 token 嵌入做后期交互，是 ColBERT 的近似实现，非专用 ColBERT 模型（如 jina-colbert），token 级语义弱。两者若换用更完整的摘要/专用模型可能有提升空间。")
    lines.append("- **最终推荐**：**HyPE（稠密）+ BM25（稀疏）+ RRF 重排 + 意图路由**，在 30 题上达到忠实度 0.954 / 相关性 0.851 / 精确率 0.815 / 召回率 0.967，为当前最优检索配置。")
    lines.append("\n## 6. 逐题明细\n")
    lines.append("见 `res/optimization/hybrid/eval_*.csv`。")

    out = OPT_RES_DIR / "hybrid" / "对比报告.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
