"""第一步：基于数据集（MinerU 转换后的 MD）生成测试集。

用 DeepSeek 从每个标准文档中抽取多样化、场景化的问答对（问题 + 标准答案），
作为后续 PDF / MD 两种向量库对比评测的统一测试集。

输出：
    output/testset.json
    output/testset.csv
"""
import json
import re
import sys
from pathlib import Path

# 避免 Windows 控制台 GBK 编码打印中文/特殊字符时崩溃
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import OUTPUT_DIR, TESTSET_CSV, TESTSET_JSON, TRANSFER_DIR
from utils.llm_utils import get_deepseek_llm

# 每个片段生成的问题数 / 片段字符数 / 每个文档最多处理的片段数
QUESTIONS_PER_CHUNK = 3
CHARS_PER_CHUNK = 7000
MAX_CHUNKS_PER_DOC = 3

PROMPT = """你是一名新能源汽车标准测试集生成助手。请基于下面给定的标准文档内容，生成 {n} 个高质量的问答对，用于评测 RAG 检索效果。

要求：
1. 问题要多样化、场景化，覆盖文档中的具体事实，例如：数值/阈值、日期/时间、术语定义、技术要求、试验方法、判定规则、适用范围等，避免问题类型单一。
2. 每个问题必须能从给定文档内容中直接找到明确答案，答案要具体、准确、简洁（尽量直接引用原文中的关键信息，不要主观推断）。
3. 问题用中文，尽量是"什么/多少/如何/哪些/何时/是否"等具体问句。
4. 严格只输出一个 JSON 数组，不要输出任何解释性文字、不要使用代码块标记。格式如下：
[{{"question": "问题", "answer": "标准答案"}}]

文档内容：
{content}"""


def chunk_text(text: str, size: int = CHARS_PER_CHUNK, max_chunks: int = MAX_CHUNKS_PER_DOC):
    """把长文档切成若干片段，限制每个文档处理的片段数。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks[:max_chunks]


def extract_json_array(text: str):
    """从 LLM 回复中稳健地提取 JSON 数组。"""
    if not text:
        return []
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        # 兜底：尝试逐行修复常见转义问题
        try:
            data = json.loads(text.replace("'", '"'))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            result.append({"question": str(item["question"]).strip(), "answer": str(item["answer"]).strip()})
    return result


def main():
    llm = get_deepseek_llm()
    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    print(f"共 {len(md_files)} 个 MD 文档待处理\n")

    all_qa = []
    for f in md_files:
        content = f.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        print(f"[{f.name}] 全文 {len(content)} 字 -> 取 {len(chunks)} 个片段")
        for ci, chunk in enumerate(chunks, 1):
            prompt = PROMPT.format(n=QUESTIONS_PER_CHUNK, content=chunk)
            try:
                resp = llm.invoke(prompt)
                qas = extract_json_array(resp.content)
            except Exception as e:
                print(f"    片段 {ci} 调用失败：{e}")
                continue
            for qa in qas:
                qa["source_doc"] = f.name
                all_qa.append(qa)
            print(f"    片段 {ci} -> 生成 {len(qas)} 个问题")

    # 去重（相同问题保留一个）
    seen = set()
    dedup = []
    for qa in all_qa:
        key = qa["question"]
        if key not in seen:
            seen.add(key)
            dedup.append(qa)

    # 编号
    for i, qa in enumerate(dedup, 1):
        qa["id"] = i

    print(f"\n共生成 {len(dedup)} 个问答对（去重后）")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TESTSET_JSON, "w", encoding="utf-8") as fp:
        json.dump(dedup, fp, ensure_ascii=False, indent=2)
    df = pd.DataFrame(dedup)
    df.to_csv(TESTSET_CSV, index=False, encoding="utf-8-sig")

    print(f"测试集已保存：\n  {TESTSET_JSON}\n  {TESTSET_CSV}")
    print("\n样例：")
    for qa in dedup[:3]:
        print(f"  Q: {qa['question']}\n  A: {qa['answer']}\n")


if __name__ == "__main__":
    main()
