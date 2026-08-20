"""从 117 题测试集中精选 30 题，作为后续优化方案的评测集。

策略：按 source_doc 分层，每个文档均匀抽取 2 题（覆盖 16 个文档的多样性）；
若超出 30 题，则剔除答案最短（信息量最少）的多余题目。
输出：output/testset_30.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, TESTSET_JSON

TARGET = 30


def evenly_sample(items, k):
    """从有序列表中均匀抽取 k 个（索引尽量分散）。"""
    if len(items) <= k:
        return list(items)
    n = len(items)
    idxs = sorted(set(round(i * (n - 1) / (k - 1)) for i in range(k)))
    return [items[i] for i in idxs]


def main():
    data = json.load(open(TESTSET_JSON, encoding="utf-8"))
    by_doc = defaultdict(list)
    for d in data:
        by_doc[d["source_doc"]].append(d)

    selected = []
    for doc, items in by_doc.items():
        selected.extend(evenly_sample(items, 2))
    print(f"分层抽样后：{len(selected)} 题")

    # 若超出目标数，剔除答案最短的题目（信息量相对少）
    if len(selected) > TARGET:
        selected.sort(key=lambda d: len(d["answer"]))
        selected = selected[len(selected) - TARGET:]

    # 按文档恢复顺序
    selected.sort(key=lambda d: (d["source_doc"], d["id"]))
    for i, d in enumerate(selected, 1):
        d["id"] = i

    out = OUTPUT_DIR / "testset_30.json"
    out.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    c = Counter(d["source_doc"] for d in selected)
    print(f"精选 {len(selected)} 题 -> {out}")
    print("各文档分布:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
