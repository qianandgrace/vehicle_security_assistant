"""三种分块/索引策略的实现：Semantic / Proposition / HyPE。

统一约定：每个策略产出一个 `list[Document]`，直接交给向量库存储与检索。
- Semantic  : 每个 Document = 语义连贯块，page_content 即检索返回内容
- Proposition: 每个 Document = 原子事实句（proposition），page_content 即检索返回内容
- HyPE      : 每个 Document = 一个假设性问题，page_content 为问题，
              metadata["original_content"] 指向原 chunk（检索后映射回原 chunk）
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 通用 ----------

def clean(text: str) -> str:
    return text.replace("\t", " ")


def base_chunks(files, chunk_size=1500, chunk_overlap=200):
    """把每个 MD 文件按固定长度切块，作为 Proposition / HyPE 的基块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    docs = []
    for f in files:
        content = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(clean(content)), 1):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": str(f), "doc_name": f.stem, "chunk_id": i},
            ))
    return docs


# ---------- Semantic ----------

# SemanticChunker 默认按英文句号切句，中文"。"不识别会导致整篇不切分。
# 这里显式按中文句号/分号切句。
CN_SENTENCE_REGEX = r"(?<=[。！？；])\s*"


def semantic_chunks(files, embedding, threshold_type="percentile", threshold_amount=90):
    """按语义边界切分：相邻句子向量距离超过阈值即切开。"""
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings=embedding,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
        sentence_split_regex=CN_SENTENCE_REGEX,
    )
    docs = []
    for f in files:
        content = clean(f.read_text(encoding="utf-8"))
        chunks = splitter.create_documents([content])
        for i, c in enumerate(chunks, 1):
            c.metadata.update({"source": str(f), "doc_name": f.stem, "chunk_id": i})
        docs.extend(chunks)
    return docs


# ---------- Proposition ----------

PROPOSITION_SYSTEM = """请把给定的文本拆分成简单、自包含、原子化的事实陈述（propositions）。

要求：
1. 每条只表达一个单一事实或要点；
2. 无需额外上下文即可理解（自包含，不要用代词指代，使用全称）；
3. 保留必要的数值、日期、单位、限定条件，使事实精确；
4. 一条陈述只含一个主谓关系，不要用并列句；
5. 尽量忠实原文，不要臆造原文没有的内容。

严格只输出一个 JSON 数组，不要输出任何解释或代码块标记。格式：
[{{"proposition": "事实陈述"}}]"""


def _parse_propositions(text: str):
    if not text:
        return []
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("proposition"):
            out.append(str(item["proposition"]).strip())
        elif isinstance(item, str):
            out.append(item.strip())
    return [p for p in out if p]


def propositionize(base_docs, llm, workers=8):
    """对每个基块调用 LLM 拆成命题，返回 Document 列表。"""
    out = []

    def _one(doc):
        resp = llm.invoke(PROPOSITION_SYSTEM + "\n\n文本：\n" + doc.page_content)
        return doc, _parse_propositions(resp.content)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, d) for d in base_docs]
        for f in as_completed(futures):
            doc, props = f.result()
            for p in props:
                out.append(Document(
                    page_content=p,
                    metadata={
                        "source": doc.metadata["source"],
                        "doc_name": doc.metadata["doc_name"],
                        "parent_chunk_id": doc.metadata["chunk_id"],
                    },
                ))
    return out


# ---------- HyPE ----------

HYPE_SYSTEM = """分析下面的文本，生成 {n} 个能覆盖其核心要点的"假设性问题"。

要求：
1. 每个问题应像是用户可能提出的真实查询（question），语言自然；
2. 问题要能引出该文本的关键信息（数值、要求、定义、方法等）；
3. 一个问题一行，不要编号、不要前缀、不要空行。

只输出问题，每行一个："""


def _parse_questions(text: str):
    if not text:
        return []
    # 去掉编号/前缀，按行拆分
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉常见编号前缀（1. 、- 、• 、(1) 等）
        line = re.sub(r"^\s*(?:\d+[\.、)）]|[-\*•·]|\(?\d+\)|（\d+）)\s*", "", line)
        line = line.strip().strip("：:，。")
        if len(line) >= 4:
            lines.append(line)
    return lines


def hype_questions(base_docs, llm, questions_per_chunk=4, workers=8):
    """对每个基块生成若干假设性问题，返回 Document 列表（问题作索引，原块在 metadata）。"""
    out = []

    def _one(doc):
        resp = llm.invoke(HYPE_SYSTEM.format(n=questions_per_chunk) + "\n\n文本：\n" + doc.page_content)
        return doc, _parse_questions(resp.content)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, d) for d in base_docs]
        for f in as_completed(futures):
            doc, questions = f.result()
            for q in questions:
                out.append(Document(
                    page_content=q,
                    metadata={
                        "source": doc.metadata["source"],
                        "doc_name": doc.metadata["doc_name"],
                        "parent_chunk_id": doc.metadata["chunk_id"],
                        "original_content": doc.page_content,  # 检索后映射回原 chunk
                    },
                ))
    return out
