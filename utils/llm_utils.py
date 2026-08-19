"""LLM 与 bge embedding 的统一初始化。

LLM 供应商由 config.LLM_TYPE 决定（deepseek | qwen | openai），
可用环境变量 LLM_TYPE 切换。
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TYPE,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回当前配置的 LLM 实例（ChatOpenAI 兼容接口，复用单例）。"""
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_retries=2,
    )


# 向后兼容别名
get_deepseek_llm = get_llm


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """返回 bge-large-zh-v1.5 embedding（1024 维，归一化，复用单例）。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
