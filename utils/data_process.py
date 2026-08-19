"""PDF 与 MD 文档的加载与分块。"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _clean_docs(docs):
    """将制表符替换为空格，并补充 doc_name 元数据便于追踪来源。"""
    for doc in docs:
        doc.page_content = doc.page_content.replace("\t", " ")
        source = doc.metadata.get("source", "")
        if source:
            doc.metadata["doc_name"] = Path(source).stem
    return docs


def _split(docs, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def encode_pdf(path, chunk_size=512, chunk_overlap=100):
    """加载 PDF 并分块。

    PDF 直接抽取的是原始文本流，表格与图片中的信息通常会丢失或错位。
    """
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    docs = _clean_docs(docs)
    return _split(docs, chunk_size, chunk_overlap)


def encode_md(path, chunk_size=512, chunk_overlap=100):
    """加载 MinerU 转换后的 MD 并分块。

    MD 已把表格结构化、图片转成文字说明，信息更完整、更利于检索。
    """
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    docs = _clean_docs(docs)
    return _split(docs, chunk_size, chunk_overlap)
