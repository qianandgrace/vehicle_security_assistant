"""上下文优化：RSE 段提取 / 窗口增强 / 上下文压缩 / U型排序。

所有方法都接收"检索返回的 docs"并输出调整后的上下文文档列表，
在生成前替换原始 context。
"""
from collections import defaultdict

from utils.chunking import base_chunks


def build_chunk_map(files, chunk_size=1500, chunk_overlap=200):
    """构建 (doc_name, chunk_id) -> Document 的映射（用于找回相邻块）。"""
    base = base_chunks(files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {(d.metadata["doc_name"], d.metadata["chunk_id"]): d for d in base}


def _recover_meta(docs, chunk_map):
    """按内容匹配，为检索返回的 docs 补充 doc_name / chunk_id 元数据。"""
    content_map = {}
    for (doc_name, cid), d in chunk_map.items():
        content_map.setdefault(d.page_content, []).append((doc_name, cid))
    for doc in docs:
        if doc.metadata.get("chunk_id") is not None:
            continue
        for doc_name, cid in content_map.get(doc.page_content, []):
            doc.metadata.setdefault("doc_name", doc_name)
            doc.metadata.setdefault("chunk_id", cid)
            break
    return docs


def rse_context(docs, chunk_map, overall_max_chunks=6):
    """Relevant Segment Extraction：把检索结果按文档聚合，合并连续 chunk_id 为段。

    相关块在原文档中往往聚簇；把连续位置的相关块合并成段，并补上被"夹"在
    中间但未命中的块，返回连续段（段内保持原文档顺序）。
    """
    _recover_meta(docs, chunk_map)
    by_doc = defaultdict(list)
    for d in docs:
        dn, cid = d.metadata.get("doc_name"), d.metadata.get("chunk_id")
        if dn is None or cid is None:
            continue
        by_doc[dn].append((cid, d))

    segments = []
    for dn, items in by_doc.items():
        items.sort(key=lambda x: x[0])
        run = [items[0]]
        for cid, d in items[1:]:
            if cid == run[-1][0] + 1:
                run.append((cid, d))
            else:
                segments.append(run)
                run = [(cid, d)]
        segments.append(run)

    scored = []
    for run in segments:
        total = sum(float(d.metadata.get("rrf_score", 0.0)) for _, d in run)
        scored.append((total, run))
    scored.sort(key=lambda x: -x[0])

    ctx = []
    for _, run in scored:
        if len(ctx) + len(run) > overall_max_chunks:
            continue
        ctx.extend(d for _, d in run)
        if len(ctx) >= overall_max_chunks:
            break
    return ctx[:overall_max_chunks]


def window_context(docs, chunk_map, radius=1, top_n=5):
    """上下文窗口增强：为每个检索块加入其前后 radius 个相邻块，丰富上下文。"""
    _recover_meta(docs, chunk_map)
    ctx, seen = [], set()
    for d in docs[:top_n]:
        dn, cid = d.metadata.get("doc_name"), d.metadata.get("chunk_id")
        if dn is None or cid is None:
            if d.page_content not in seen:
                seen.add(d.page_content)
                ctx.append(d)
            continue
        for delta in range(-radius, radius + 1):
            n = chunk_map.get((dn, cid + delta))
            if n is not None and n.page_content not in seen:
                seen.add(n.page_content)
                ctx.append(n)
    return ctx


COMPRESS_PROMPT = """根据问题，从下面的文档片段中提取与问题直接相关的关键信息（数值、要求、定义、试验方法等），压缩为简洁的相关内容，去掉无关表述。若片段与问题无关，只输出两个字：无关。

问题：{query}

片段：
{chunk}

相关内容："""


def compress_context(query, docs, llm):
    """上下文压缩：用 LLM 提取每个块中与问题相关的部分，减少噪音。

    返回与输入一致的 Document 列表（内容为压缩后的文本）。
    """
    from langchain_core.documents import Document

    out = []
    for d in docs:
        try:
            resp = llm.invoke(COMPRESS_PROMPT.format(query=query, chunk=d.page_content))
            text = resp.content.strip()
        except Exception:
            text = d.page_content
        if text and text != "无关":
            out.append(Document(page_content=text, metadata=dict(d.metadata)))
    if not out:  # 全部被判定无关时回退到原块
        return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs]
    return out


def ushape_context(docs, top_n=5):
    """U型排序：最高分放开头、次高分放结尾，缓解"关键信息在中间被忽略"（lost-in-the-middle）。"""
    scored = sorted(docs, key=lambda d: -float(d.metadata.get("rrf_score", 0.0)))
    top = scored[:top_n]
    if len(top) <= 2:
        return top
    return [top[0]] + top[2:] + [top[1]]
