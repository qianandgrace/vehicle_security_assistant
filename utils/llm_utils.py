"""LLM 与 bge embedding 的统一初始化。

- get_llm()        : 评测/问答用 LLM（由 config.LLM_TYPE 决定，默认 deepseek-v4-flash）
- get_index_llm()  : 索引构建（Proposition/HyPE 生成）用 LLM（由 config.INDEX_LLM_TYPE 决定，默认 qwen）
  原因：deepseek-v4-flash 对"提取命题/假设性问题（长 JSON/列表）"这类提示会挂起，
  而 qwen-max / gpt-4o-mini 表现稳定。
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_MODEL,
    INDEX_LLM_TYPE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CONFIGS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TYPE,
)


def _build_llm(llm_type: str, *, max_tokens: int | None = None) -> ChatOpenAI:
    cfg = LLM_CONFIGS[llm_type]
    return ChatOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=LLM_TEMPERATURE,
        max_retries=2,
        timeout=90,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回当前配置的评测/问答 LLM 实例（默认 deepseek-v4-flash）。"""
    return _build_llm(LLM_TYPE)


@lru_cache(maxsize=1)
def get_index_llm() -> ChatOpenAI:
    """返回索引构建（Proposition/HyPE 生成）LLM 实例（默认 qwen-max）。"""
    return _build_llm(INDEX_LLM_TYPE, max_tokens=3000)


# 向后兼容别名
get_deepseek_llm = get_llm


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """返回 bge-large-zh-v1.5 embedding（1024 维，归一化，复用单例）。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
