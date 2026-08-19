"""
通过milvus和langchain快速使用rag
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings


from utils.data_process import encode_pdf_langchain
from utils.eval_rag import answer_with_rag, evaluate_rag
from rag_eval.ragas_eval import construct_rag_dataset, get_evaluation_metrics, questions, ground_truths


load_dotenv()

class SimpleRAG:
    def __init__(self, milvus_uri: str = "http://localhost:19530",
                 db_name: str = "test_db",
                 collection_name: str = "test_collection"):
        # 初始化embedding模型
        self.milvis_uri = milvus_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True}
            )  # 使用bge-large-zh-v1.5模型进行向量化
        # 连接到运行在本地的 Milvus 服务器
        self.vector_store = Milvus(
                embedding_function=self.embedding_model,
                collection_name=collection_name,
                connection_args={"uri": milvus_uri, "db_name":db_name,},
                index_params={"index_type": "FLAT", "metric_type": "L2"},
        )
        # 这里传入的大模型需要是chatopenAI，不能是原生的
        self.llm = ChatOpenAI(
            api_key=os.getenv("LAOZHANG_API_KEY"), # type: ignore
            # 以下是北京地域base_url
            base_url=os.getenv("LAOZHANG_BASE_URL"),
            model="gpt-4o-mini")

    def save_vector(self, file_path, collection_name):
        # 
        parent_chunks = encode_pdf_langchain(file_path, chunk_size=512, chunk_overlap=100)
        # 存入milvus数据库中
        self.vector_store = Milvus.from_documents(
            parent_chunks,
            self.embedding_model,
            collection_name=collection_name,
            connection_args={"uri": self.milvis_uri, "db_name": self.db_name},
        )

    def answer(self, query: str, top_k: int = 10):
        """向量检索"""
        # search type 包括mmr， search_kwagrs
        search_retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
        res = answer_with_rag(search_retriever, self.llm, query)
        return res

    def evaluate(self, questions, ground_truths, top_k: int = 10):
        """rag效果评估"""
        search_retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
        retrieved_contexts = []
        answers = []
        for question in questions:
            # Retrieve documents
            docs = self.vector_store.similarity_search(question, k=top_k)
            retrieved_contexts.append([doc.page_content for doc in docs])
            # Generate answer
            answer = answer_with_rag(search_retriever, self.llm, question)
            answers.append(answer["answer"])
        
        dataset = construct_rag_dataset(questions, ground_truths, retrieved_contexts, answers)
        metrics_result = get_evaluation_metrics(dataset, self.llm, self.embedding_model)
        return metrics_result
    

if __name__ == "__main__":
    pdf_path = r"C:\Users\qian gao\git_project\my_rag_tech\data\company_documents\浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf"
    rag =  SimpleRAG(collection_name="langchain_example")
    # rag.save_vector(pdf_path, "langchain_example")
    # res = rag.answer("客户经理被投诉了，投诉一次扣多少分", top_k=3)
    eval_result = rag.evaluate(questions, ground_truths, top_k=3)
    # 保存到csv文件中
    import pandas as pd
    eval_result.to_csv("rag_eval_result.csv", index=False)

    print(f"测评效果：{eval_result}")

