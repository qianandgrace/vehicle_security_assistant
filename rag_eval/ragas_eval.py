"""ragas 评测：构建数据集 + 计算指标。

适配 ragas 0.4.x：沿用旧版指标路径（from ragas.metrics import ...），
此时指标构造时 llm/embeddings 可选，由 evaluate() 自动注入，
与参考代码保持一致。旧路径会打印弃用警告，这里静默处理。
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


def construct_rag_dataset(questions, ground_truths, contexts, answers):
    """构建 ragas 评估所需的 Dataset。"""
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
    return Dataset.from_dict(data)


def get_evaluation_metrics(data, llm, embeddings):
    """计算 Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall。"""
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    # DeepSeek 的 deepseek-chat 不支持 n>1（一次生成多个候选），
    # 而 ragas 的 AnswerRelevancy 等指标默认请求 n=3。用 bypass_n=True 包装，
    # 让 ragas 改为发起 n 次 n=1 的请求，规避 400 报错。
    ragas_llm = LangchainLLMWrapper(llm, bypass_n=True)
    result = evaluate(
        dataset=data,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=embeddings,
    )
    return result.to_pandas()
