"""生成三模型对比报告：deepseek（云） vs 本地 qwen2:7b（未微调） vs 微调后 qwen2:7b。

读取：
    res/models/eval_deepseek.csv    模型 A：DeepSeek 云
    res/models/eval_ollama.csv      模型 B：qwen2:7b 未微调
    res/models/eval_peft.csv        模型 C：qwen2:7b 微调（LoRA）

输出：res/models/对比报告.md
"""
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RES_DIR, TOP_K

METRICS = [
    ("faithfulness", "忠实度 Faithfulness"),
    ("answer_relevancy", "答案相关性 Answer Relevancy"),
    ("context_precision", "上下文精确率 Context Precision"),
    ("context_recall", "上下文召回率 Context Recall"),
]

PLAN = [
    ("A DeepSeek 云", "eval_deepseek.csv"),
    ("B qwen2:7b 未微调", "eval_ollama.csv"),
    ("C qwen2:7b 微调", "eval_peft.csv"),
]


def main():
    models_dir = RES_DIR / "models"
    frames = []
    for name, fname in PLAN:
        path = models_dir / fname
        if not path.exists():
            print(f"[跳过] {name}：缺少 {path}")
            continue
        frames.append((name, pd.read_csv(path, encoding="utf-8-sig")))
    if not frames:
        raise FileNotFoundError("没有评测结果可读")

    rows = []
    for name, df in frames:
        row = {"模型": name}
        for c, _ in METRICS:
            row[c] = round(float(df[c].mean()), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)

    lines = []
    lines.append("# 三模型 RAG 效果对比报告\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. 实验概述\n")
    lines.append(f"- **评测集**：30 个问答对（覆盖 16 个新能源汽车标准）")
    lines.append(f"- **检索管线**：意图路由 → HyPE + BM25 → RRF → U型排序（三模型共用，完全相同）")
    lines.append(f"- **评测裁判**：固定 DeepSeek（保证公平；检索指标一致，差异来自生成模型）")
    lines.append(f"- **模型**：A=DeepSeek 云 API；B=本地 qwen2:7b（ollama，未微调）；C=qwen2:7b QLoRA 微调（LoRA，基于标准问答数据 957 条）\n")
    lines.append("## 2. 平均指标对比\n")
    lines.append("| 模型 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['模型']} | {r['faithfulness']:.4f} | {r['answer_relevancy']:.4f} "
            f"| {r['context_precision']:.4f} | {r['context_recall']:.4f} |"
        )

    lines.append("\n## 3. 结论\n")
    # 对比 B 与 C（微调是否有效）
    if len(frames) >= 2:
        b = {name: df for name, df in frames}
        if "B qwen2:7b 未微调" in b and "C qwen2:7b 微调" in b:
            bf = b["B qwen2:7b 未微调"]
            cf = b["C qwen2:7b 微调"]
            d = {c: float(cf[c].mean()) - float(bf[c].mean()) for c, _ in METRICS}
            lines.append("- **微调效果（C vs B）**：" + "；".join(
                f"{name} {('▲' if d[c] > 0 else '▼' if d[c] < 0 else '=')}{abs(d[c]):.4f}"
                for c, name in METRICS) + "\n")
    lines.append("根据微调前后对比，判断领域微调是否显著提升本地 7B 模型在标准问答上的忠实度与相关性，"
                 "并检验「DeepSeek ≈ 微调后模型 > 未微调 7B」的预期。")
    lines.append("\n## 4. 逐题明细\n")
    lines.append("见 `res/models/eval_*.csv`。")

    out = models_dir / "对比报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
