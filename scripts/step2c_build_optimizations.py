"""构建三个优化方案的 Milvus 集合（db=vehicle）：

    1. md_semantic_collection  —— SemanticChunker 语义切块
    2. md_proposition_collection —— LLM 拆成原子命题（proposition）嵌入
    3. md_hype_collection      —— LLM 生成假设性问题（HyPE）嵌入

用法：
    python scripts/step2c_build_optimizations.py            # 全量 16 个文档
    python scripts/step2c_build_optimizations.py --limit 2  # 仅前 2 个文档（快速验证）
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_milvus import Milvus

from config import (
    DB_NAME,
    HYPE_COLLECTION,
    MILVUS_URI,
    PROPOSITION_COLLECTION,
    SEMANTIC_COLLECTION,
    TRANSFER_DIR,
)
from utils.chunking import base_chunks, hype_questions, propositionize, semantic_chunks
from utils.llm_utils import get_embedding_model, get_index_llm

# Proposition / HyPE 的基块参数
BASE_CHUNK_SIZE = 1500
BASE_CHUNK_OVERLAP = 200
QUESTIONS_PER_CHUNK = 4


def drop_and_store(collection_name, docs, embedding, db_name):
    from pymilvus import MilvusClient
    client = MilvusClient(uri=MILVUS_URI, db_name=db_name)
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)
        print(f"  已删除旧集合 {collection_name}")
    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={"uri": MILVUS_URI, "db_name": db_name},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    vs.add_documents(docs)
    return vs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文档（验证用）")
    args = parser.parse_args()

    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    if args.limit:
        md_files = md_files[: args.limit]
    print(f"待处理文档：{len(md_files)} 个")
    if args.limit:
        print("  [验证模式] 仅处理前 %d 个" % args.limit)

    embedding = get_embedding_model()
    llm = get_index_llm()

    # ---- 1. Semantic ----
    print("\n===== 1. SemanticChunker 语义切块 =====")
    sem_docs = semantic_chunks(md_files, embedding, threshold_type="percentile", threshold_amount=90)
    print(f"  语义块数：{len(sem_docs)}")
    drop_and_store(SEMANTIC_COLLECTION, sem_docs, embedding, DB_NAME)
    print(f"  集合 {SEMANTIC_COLLECTION} 写入完成")

    # ---- 2. Proposition ----
    print("\n===== 2. Proposition 原子命题 =====")
    base = base_chunks(md_files, chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_CHUNK_OVERLAP)
    print(f"  基块数：{len(base)}")
    prop_docs = propositionize(base, llm)
    print(f"  命题数：{len(prop_docs)}")
    drop_and_store(PROPOSITION_COLLECTION, prop_docs, embedding, DB_NAME)
    print(f"  集合 {PROPOSITION_COLLECTION} 写入完成")

    # ---- 3. HyPE ----
    print("\n===== 3. HyPE 假设性问题 =====")
    hype_docs = hype_questions(base, llm, questions_per_chunk=QUESTIONS_PER_CHUNK)
    print(f"  问题数：{len(hype_docs)}")
    drop_and_store(HYPE_COLLECTION, hype_docs, embedding, DB_NAME)
    print(f"  集合 {HYPE_COLLECTION} 写入完成")

    # 汇总
    from pymilvus import MilvusClient
    client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
    print(f"\n数据库 {DB_NAME} 现有集合：{client.list_collections()}")
    print("构建完成")


if __name__ == "__main__":
    main()
