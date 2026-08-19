"""RAG 问答工具函数。"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_with_context(llm, question, context_text):
    """基于给定上下文回答问题。

    与检索环节共用同一份 context，保证 ragas 评测的对象一致。
    """
    system = (
        "You are an assistant for question-answering tasks. Answer the question "
        "based ONLY on the provided documents. If the answer is not in the "
        "documents, say that you don't know. Use three to five sentences maximum "
        "and keep the answer concise."
    )
    human = (
        "Retrieved documents:\n\n<docs>{docs}</docs>\n\n"
        "User question: <question>{question}</question>"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"docs": context_text, "question": question})
