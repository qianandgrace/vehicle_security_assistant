"""第二步：分别用 PDF 和 MD 建两个 Milvus 集合。

- 数据库：vehicle（已创建）
- 集合 1：pdf_collection（来自 data/origin_data 原始 PDF）
- 集合 2：md_collection （来自 data/transfer_data MinerU 转换后的 MD）

两者使用相同的 embedding（bge-large-zh-v1.5）、分块参数与索引参数，保证对比公平。
"""
import sys
from pathlib import Path

# 避免 Windows 控制台 GBK 编码打印中文/特殊字符时崩溃
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import MilvusClient

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_NAME,
    MD_COLLECTION,
    MILVUS_URI,
    ORIGIN_DIR,
    PDF_COLLECTION,
    TRANSFER_DIR,
)
from utils.data_process import encode_md, encode_pdf
from utils.llm_utils import get_embedding_model


def drop_collection_if_exists(db_name: str, collection_name: str):
    client = MilvusClient(uri=MILVUS_URI, db_name=db_name)
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)
        print(f"  已删除旧集合 {collection_name}")
    else:
        print(f"  集合 {collection_name} 不存在，跳过删除")


def ensure_db(db_name: str):
    client = MilvusClient(uri=MILVUS_URI)
    if db_name not in client.list_databases():
        client.create_database(db_name)
        print(f"已创建数据库 {db_name}")
    else:
        print(f"数据库 {db_name} 已存在")


def build_collection(collection_name: str, docs, embedding, db_name: str):
    """把文档写入一个集合（构造函数会按 embedding 维度自动建 schema）。"""
    from langchain_milvus import Milvus

    vs = Milvus(
        embedding_function=embedding,
        collection_name=collection_name,
        connection_args={"uri": MILVUS_URI, "db_name": db_name},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    if docs:
        vs.add_documents(docs)
    return vs


def main():
    ensure_db(DB_NAME)
    embedding = get_embedding_model()

    # ---- PDF 集合 ----
    print("\n===== 构建 PDF 集合 =====")
    drop_collection_if_exists(DB_NAME, PDF_COLLECTION)
    pdf_files = sorted(ORIGIN_DIR.glob("*.pdf"))
    pdf_docs = []
    for f in pdf_files:
        try:
            chunks = encode_pdf(f, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            pdf_docs.extend(chunks)
            print(f"  [PDF] {f.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  [PDF] {f.name} 解析失败：{e}")
    print(f"  PDF 总 chunks: {len(pdf_docs)}")
    build_collection(PDF_COLLECTION, pdf_docs, embedding, DB_NAME)
    print(f"  集合 {PDF_COLLECTION} 写入完成")

    # ---- MD 集合 ----
    print("\n===== 构建 MD 集合 =====")
    drop_collection_if_exists(DB_NAME, MD_COLLECTION)
    md_files = sorted(TRANSFER_DIR.glob("*.md"))
    md_docs = []
    for f in md_files:
        try:
            chunks = encode_md(f, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            md_docs.extend(chunks)
            print(f"  [MD] {f.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  [MD] {f.name} 解析失败：{e}")
    print(f"  MD 总 chunks: {len(md_docs)}")
    build_collection(MD_COLLECTION, md_docs, embedding, DB_NAME)
    print(f"  集合 {MD_COLLECTION} 写入完成")

    # 汇总
    client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
    print(f"\n数据库 {DB_NAME} 中的集合：{client.list_collections()}")
    print("第二步完成")


if __name__ == "__main__":
    main()
