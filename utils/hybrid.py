"""混合检索 + 重排：稠密检索（HyPE）+ 稀疏检索（BM25）+ 4 种重排。

- BM25Index         : 基于 rank_bm25 + jieba 中文分词的稀疏索引
- rrf_rerank        : Reciprocal Rank Fusion（纯排名融合，零样本）
- CrossEncoderReranker : bce-reranker-base_v1 交叉编码器精排
- RankLLMReranker   : LLM 判断相关性并排序
- ColBERTReranker   : bge-m3 token 级后期交互（MaxSim）重排
- hybrid_retrieve   : 稠密+稀疏召回候选 -> 指定方法重排 -> top_k
"""
import json
import re
from functools import lru_cache

import jieba
import numpy as np
from langchain_core.documents import Document

# ---------- 中文分词 ----------

_STOP = set("的了是在和与及于有我就他她它为其这不也都一个")
_PUNC = re.compile(r"[^\w一-鿿]+")


def tokenize(text: str) -> list[str]:
    """jieba 中文分词 + 过滤停用词/标点/单字。"""
    toks = []
    for w in jieba.cut(text):
        w = _PUNC.sub("", w).strip()
        if len(w) >= 2 and w not in _STOP:
            toks.append(w)
    return toks


# ---------- BM25 ----------

class BM25Index:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.corpus = [tokenize(d.page_content) for d in docs]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int) -> list[Document]:
        """返回 BM25 命中的文档（按分数降序）。"""
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in order]


# ---------- RRF ----------

def rrf_rerank(dense_docs: list[Document], sparse_docs: list[Document],
               k: int = 60, top: int = 5) -> list[Document]:
    """Reciprocal Rank Fusion：按两路检索的排名倒数融合，纯排名、零样本。"""
    scores: dict[str, float] = {}
    objs: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        did = hash(doc.page_content)
        objs[did] = doc
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(sparse_docs):
        did = hash(doc.page_content)
        objs[did] = doc
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top]
    for did, s in ranked:
        objs[did].metadata["rrf_score"] = s
    return [objs[did] for did, _ in ranked]


# ---------- Cross-Encoder ----------

@lru_cache(maxsize=1)
def _crossencoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("maidalun1020/bce-reranker-base_v1")


class CrossEncoderReranker:
    def __init__(self):
        self.model = _crossencoder()

    def rerank(self, query: str, docs: list[Document], top: int = 5) -> list[Document]:
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1][:top]
        return [docs[i] for i in order]


# ---------- RankLLM ----------

RANKLLM_PROMPT = """你是检索结果重排助手。根据用户问题，对下列候选文档按相关性从高到低排序。

文档列表（编号. 前100字摘要）：
{docs}

问题：{query}

只输出一个 JSON 数组，按相关性降序排列，元素为 {{"id": 编号, "relevance": 1-10 分数}}。不要输出解释。"""


class RankLLMReranker:
    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, docs: list[Document], top: int = 5) -> list[Document]:
        if not docs:
            return docs
        pool = docs[:40]  # 限制候选数，避免超长
        lines = [f"{i+1}. {d.page_content[:100]}" for i, d in enumerate(pool)]
        resp = self.llm.invoke(RANKLLM_PROMPT.format(
            docs="\n".join(lines), query=query))
        m = re.search(r"\[.*\]", resp.content, re.DOTALL)
        if not m:
            return pool[:top]
        try:
            ranking = json.loads(m.group(0))
            ids = [int(x["id"]) - 1 for x in ranking if "id" in x]
        except Exception:
            return pool[:top]
        ordered = [pool[i] for i in ids if 0 <= i < len(pool)]
        return (ordered + [d for d in pool if d not in ordered])[:top]


# ---------- ColBERT（后期交互）----------

@lru_cache(maxsize=1)
def _colbert_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3").to(device).eval()
    return tok, model, device


class ColBERTReranker:
    def __init__(self):
        self.tokenizer, self.model, self.device = _colbert_model()

    def _encode_tokens(self, texts):
        import torch
        import torch.nn.functional as F
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).last_hidden_state
        emb = F.normalize(emb, p=2, dim=-1)
        return emb, inputs["attention_mask"]

    def rerank(self, query: str, docs: list[Document], top: int = 5) -> list[Document]:
        import torch
        if not docs:
            return docs
        q_emb, q_mask = self._encode_tokens([query])
        q_emb, q_len = q_emb[0], int(q_mask[0].sum())
        scores = []
        for i in range(0, len(docs), 16):
            batch = [d.page_content for d in docs[i:i + 16]]
            d_emb, d_mask = self._encode_tokens(batch)
            for j in range(d_emb.shape[0]):
                dlen = int(d_mask[j].sum())
                sim = torch.matmul(q_emb[:q_len], d_emb[j][:dlen].T)  # (q_len, d_len)
                scores.append(float(sim.max(dim=1).values.sum()))
        order = np.argsort(scores)[::-1][:top]
        return [docs[i] for i in order]


# ---------- 混合检索 ----------

def _dedup(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


def hybrid_retrieve(query, dense_fn, bm25, method, cand_k=30, top_k=5, llm=None):
    """稠密（HyPE）+ 稀疏（BM25）召回候选 -> 指定方法重排 -> top_k。

    dense_fn(query, k) -> list[Document]（稠密检索返回的原始 chunk）
    method: rrf | crossencoder | rankllm | colbert
    """
    dense_docs = dense_fn(query, cand_k)
    sparse_docs = bm25.retrieve(query, cand_k)

    if method == "rrf":
        return rrf_rerank(dense_docs, sparse_docs, top=top_k)

    merged = _dedup(dense_docs + sparse_docs)
    if method == "crossencoder":
        return CrossEncoderReranker().rerank(query, merged, top=top_k)
    if method == "rankllm":
        return RankLLMReranker(llm).rerank(query, merged, top=top_k)
    if method == "colbert":
        return ColBERTReranker().rerank(query, merged, top=top_k)
    raise ValueError(f"未知重排方法: {method}")
