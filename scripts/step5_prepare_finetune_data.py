"""构造 QLoRA 微调数据集：从标准文档生成标准问答指令数据。

- 从每个 MD 基块用 LLM 生成 2 个问答对（风格与测试集一致）
- 排除与精选测试集（30 题）相同的问题，避免评测数据泄漏
- 输出 Qwen2 chat 格式的 JSONL 指令数据

输出：output/finetune_data.jsonl（~1000 条）
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, TESTSET_30, TRANSFER_DIR
from utils.chunking import base_chunks
from utils.llm_utils import get_index_llm

QA_PER_CHUNK = 2
BASE_CHUNK_SIZE = 1500
BASE_CHUNK_OVERLAP = 200

PROMPT = """你是一名新能源汽车标准问答数据生成助手。请基于下面的标准文档内容，生成 {n} 个高质量问答对。

要求：
1. 问题具体、多样化（数值/限值、定义、试验方法、适用范围、判定规则等），是用户可能真实提出的问题；
2. 答案必须直接从文档内容中提取，具体、准确、简洁；
3. 不要生成与示例测试集相同或高度相似的问题。

严格只输出一个 JSON 数组，不要输出解释或代码块：
[{{"question": "问题", "answer": "标准答案"}}]

文档内容：
{content}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main():
    # 测试集问题（用于排除泄漏）
    testset = json.load(open(TESTSET_30, encoding="utf-8"))
    test_q = {norm(t["question"]) for t in testset}
    print(f"测试集 {len(test_q)} 题，将排除相同问题")

    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    chunks = base_chunks(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
    print(f"基块数：{len(chunks)}")

    llm = get_index_llm()

    def _one(doc):
        resp = llm.invoke(PROMPT.format(n=QA_PER_CHUNK, content=doc.page_content))
        m = re.search(r"\[.*\]", resp.content, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
        pairs = []
        for item in data:
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                q, a = str(item["question"]).strip(), str(item["answer"]).strip()
                if norm(q) not in test_q:
                    pairs.append({"question": q, "answer": a, "doc": doc.metadata["doc_name"]})
        return pairs

    all_pairs = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one, c) for c in chunks]
        for f in as_completed(futures):
            all_pairs.extend(f.result())

    # 去重
    seen, dedup = set(), []
    for p in all_pairs:
        if norm(p["question"]) not in seen:
            seen.add(norm(p["question"]))
            dedup.append(p)
    print(f"生成 {len(all_pairs)} 对，去重后 {len(dedup)} 对")

    # 输出 Qwen2 chat 格式 JSONL
    out = OUTPUT_DIR / "finetune_data.jsonl"
    with open(out, "w", encoding="utf-8") as fp:
        for p in dedup:
            record = {
                "messages": [
                    {"role": "system", "content": "你是新能源汽车标准领域的专业问答助手，请依据标准文档内容准确回答问题。"},
                    {"role": "user", "content": p["question"]},
                    {"role": "assistant", "content": p["answer"]},
                ]
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已保存：{out}（{len(dedup)} 条）")


if __name__ == "__main__":
    main()
