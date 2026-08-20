"""父子块适用性分析（v2）：答案信息跨度分析。

父子块的核心问题是"上下文被切太细"——一个 chunk 装不下回答问题所需的完整上下文。
对每个问题：
1. 从参考答案提取关键 token（数字 / 拉丁词 / 中文词），并在源文档中定位；
2. 计算"最小覆盖窗口"：能覆盖全部（或核心 80%）关键 token 的最短连续文档区间；
3. 对比该跨度与当前 chunk_size=512。

结论：若多数答案的跨度远大于 chunk_size，说明单块装不下完整上下文，
父子块（小块检索 + 父文档生成）值得实施；否则价值有限。
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRANSFER_DIR

# 常见停用词（不参与定位）
STOP = set("的了是在应和与及于对为此按按到从要向被把由自等并或如若则如果时中所其之各该将还都要可最不大二三四五六七八九十比较相应依据规定作为包括根据按照进行具有这些那些部分")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def key_tokens(answer: str):
    """从答案中提取有区分度的 token。"""
    toks = re.findall(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z₀-₉]+|[一-鿿]{2,}", answer)
    out = []
    for t in toks:
        if t in STOP:
            continue
        # 去掉纯单位/常见词
        if re.fullmatch(r"(分钟|小时|毫米|厘米|米|千米|公里|牛|克|伏|安|秒|第|条)", t):
            continue
        out.append(t)
    return list(dict.fromkeys(out))


def minimal_span(positions):
    """positions: 每个 token 的位置列表。返回能覆盖最多 token 的最短区间。"""
    # 展平所有 (pos, token_idx)，按 pos 排序
    events = []
    for ti, plist in enumerate(positions):
        for p in plist:
            events.append((p, ti))
    events.sort()
    n = len(positions)
    # 滑动窗口：找覆盖 >= target 个不同 token 的最小 span
    best_span = None
    count = {}
    l = 0
    covered = 0
    for r in range(len(events)):
        pos_r, ti_r = events[r]
        if count.get(ti_r, 0) == 0:
            covered += 1
        count[ti_r] = count.get(ti_r, 0) + 1
        while l <= r:
            pos_l, ti_l = events[l]
            if count[ti_l] > 1:
                count[ti_l] -= 1
                l += 1
            else:
                break
        # 只记录覆盖所有/绝大多数 token 的窗口
        if covered >= 0.8 * n:
            span = events[r][0] - events[l][0]
            if best_span is None or span < best_span:
                best_span = span
    return best_span


def main():
    testset = json.load(open(OUTPUT := PROJECT_ROOT / "output/testset_30.json", encoding="utf-8"))
    CHUNK = 512
    results = []
    for item in testset:
        doc_name = item["source_doc"]
        md_path = TRANSFER_DIR / doc_name
        if not md_path.exists():
            results.append((item, None, None))
            continue
        doc = norm(md_path.read_text(encoding="utf-8"))
        toks = key_tokens(item["answer"])
        positions = []
        found_toks = []
        for t in toks:
            idxs = [m.start() for m in re.finditer(re.escape(t), doc)]
            if idxs:
                positions.append(idxs)
                found_toks.append(t)
        if len(found_toks) < max(2, len(toks) * 0.4):
            results.append((item, None, len(toks)))
            continue
        span = minimal_span(positions)
        results.append((item, span, len(found_toks)))

    spans = [r[1] for r in results if r[1] is not None]
    n_measured = len(spans)
    print(f"===== 答案信息跨度分析（{len(testset)} 题，chunk_size={CHUNK}）=====")
    print(f"成功定位到答案关键信息的问题：{n_measured} 题\n")
    if spans:
        spans_sorted = sorted(spans)
        import statistics
        print(f"最小覆盖窗口（字符数）分布：")
        print(f"  中位数 {statistics.median(spans):.0f} | 均值 {statistics.mean(spans):.0f} | "
              f"最小 {spans_sorted[0]} | 最大 {spans_sorted[-1]}")
        print(f"  <={CHUNK}（单块够用）: {sum(1 for s in spans if s <= CHUNK)} 题")
        print(f"  >{CHUNK}（需更大上下文）: {sum(1 for s in spans if s > CHUNK)} 题")
        ratio = sum(1 for s in spans if s > CHUNK) / n_measured
        print(f"\n结论倾向：{ratio:.0%} 的题目其答案上下文跨度超过单块容量，"
              f"{'需要' if ratio >= 0.4 else '暂不需要'}父子块/大块生成优化。")

    print("\n明细（未定位到的标 -）：")
    for item, span, found in results:
        tag = f"{span}" if span is not None else "-"
        flag = " <== 超块" if (span is not None and span > CHUNK) else ""
        print(f"  [{item['id']:>2}] span={tag:>6}{flag}  {item['question'][:40]}")


if __name__ == "__main__":
    main()
