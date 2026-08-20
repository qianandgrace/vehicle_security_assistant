"""最终 RAG pipeline：整合全部优化后的检索链路。

流程：
    用户问题 -> 意图路由(按需重写) -> HyPE 稠密检索 + BM25 稀疏检索
            -> RRF 融合重排 -> U型排序上下文 -> LLM 生成答案

LLM 可切换：deepseek 等云 API（get_llm）或本地 ollama（qwen2:7b）。
"""
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama

from config import (
    DB_NAME,
    HYPE_COLLECTION,
    MILVUS_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TRANSFER_DIR,
)
from utils.chunking import base_chunks
from utils.context_opt import build_chunk_map, ushape_context
from utils.eval_rag import answer_with_context
from utils.hybrid import BM25Index, rrf_rerank
from utils.llm_utils import get_llm, get_embedding_model, get_index_llm
from utils.query_opt import classify_and_rewrite

BASE_CHUNK_SIZE = 1500
BASE_CHUNK_OVERLAP = 200
HYPE_MULTIPLIER = 4
CAND_K = 30
DEFAULT_TOP_K = 5


class VehicleRAG:
    def __init__(self, llm_type: str = "deepseek", top_k: int = DEFAULT_TOP_K):
        """llm_type: deepseek | qwen | openai | ollama"""
        self.top_k = top_k
        self.llm_type = llm_type
        self.embedding = get_embedding_model()
        self.route_llm = get_index_llm()  # 意图路由/重写用（qwen，稳定）
        self.llm = self._build_llm(llm_type)

        # HyPE 稠密检索（问题-问题匹配，映射回原块）
        self.hype_vs = Milvus(
            embedding_function=self.embedding,
            collection_name=HYPE_COLLECTION,
            connection_args={"uri": MILVUS_URI, "db_name": DB_NAME},
        )
        # BM25 稀疏索引（与 HyPE 相同的基块）
        md_files = sorted(TRANSFER_DIR.glob("*.md"))
        base = base_chunks(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
        self.bm25 = BM25Index(base)
        self.chunk_map = build_chunk_map(md_files, chunk_size=BASE_CHUNK_SIZE,
                                         chunk_overlap=BASE_CHUNK_OVERLAP)

    def _build_llm(self, llm_type: str):
        if llm_type == "ollama":
            return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)
        if llm_type in ("deepseek", "qwen", "openai"):
            return get_llm()
        raise ValueError(f"未知 llm_type: {llm_type}")

    # ---- 检索 ----
    def _dense_fn(self, query, k):
        docs = self.hype_vs.similarity_search(query, k=k * HYPE_MULTIPLIER)
        seen, out = set(), []
        for d in docs:
            orig = d.metadata.get("original_content", d.page_content)
            if orig and orig not in seen:
                seen.add(orig)
                out.append(Document(page_content=orig))
            if len(out) >= k:
                break
        return out

    def retrieve(self, query: str, top_k: int | None = None):
        """意图路由 -> HyPE + BM25 -> RRF -> U型排序。返回 (docs, 路由信息)。"""
        top_k = top_k or self.top_k
        route = classify_and_rewrite(query, self.route_llm)
        sq = route["rewritten_query"] if route["need_rewrite"] else query

        dense = self._dense_fn(sq, CAND_K)
        sparse = self.bm25.retrieve(sq, CAND_K)
        docs = rrf_rerank(dense, sparse, top=top_k)
        docs = ushape_context(docs, top_n=top_k)
        return docs, {"query_type": route["type"], "need_rewrite": route["need_rewrite"],
                      "search_query": sq}

    # ---- 问答 ----
    def answer(self, query: str, top_k: int | None = None):
        docs, route = self.retrieve(query, top_k)
        ctx = [d.page_content for d in docs]
        result = answer_with_context(self.llm, query, "\n\n".join(ctx))
        return {
            "answer": result,
            "contexts": ctx,
            "sources": [d.metadata.get("doc_name", "?") for d in docs],
            "route": route,
        }


@lru_cache(maxsize=1)
def get_pipeline(llm_type: str = "deepseek") -> VehicleRAG:
    """缓存单例 pipeline，供 gradio 与应用复用。"""
    return VehicleRAG(llm_type=llm_type)


if __name__ == "__main__":
    import json
    q = "车窗防夹力应不大于多少牛？"
    pipe = get_pipeline("deepseek")
    out = pipe.answer(q)
    print("问题:", q)
    print("回答:", out["answer"])
    print("来源:", out["sources"])
