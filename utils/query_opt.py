"""Query 侧优化：意图识别 + 查询重写 + HyDE。

- classify_and_rewrite : 一次 LLM 调用完成"类型分类 + 是否需重写 + 重写"
- rewrite_query        : 无条件重写
- hyde_document        : 生成假设文档（HyDE），用于基线集合检索

生成用 LLM 默认取 get_index_llm()（qwen，稳定；deepseek-v4-flash 对长列表
生成会挂起，但这里都是单对象/短文输出，也可自行传入 get_llm()）。
"""
import json
import re

CLASSIFY_PROMPT = """你是新能源汽车标准领域的查询分析助手。分析下面的用户查询，判断其类型并决定是否需要重写以提升检索效果。

查询类型：
- 数值限值类：询问具体数值、限值、范围（如"限值是多少""速度应为多少""不大于多少"）
- 日期发布类：询问发布/实施日期
- 定义概念类：询问术语定义（如"什么是""如何定义"）
- 试验方法类：询问试验方法、条件、步骤、判定规则
- 适用范围类：询问适用对象/范围

重写规则：
- 数值限值类、日期发布类：通常已明确，不需要重写
- 定义概念类、试验方法类、适用范围类：可能需要重写，补全标准名称/关键词/限定词使其更利于检索

若不需要重写，rewritten_query 保持原查询不变。
严格只输出一个 JSON 对象，不要输出解释或代码块：
{{"type": "数值限值类", "need_rewrite": false, "rewritten_query": "原查询文本"}}

查询：{query}"""

REWRITE_PROMPT = """你是新能源汽车标准领域的查询重写助手。将下面的用户查询改写为更利于向量检索的形式：
- 补全标准名称、术语全称；
- 补充关键限定词（车型、对象、条件等）；
- 保持原意，不改变问题本质，不臆造。

只输出重写后的查询文本，不要解释、不要编号。

查询：{query}"""

HYDE_PROMPT = """给定问题：'{query}'

请生成一段约 {chunk_size} 字的"假设性文档"，直接回答该问题。该文档将作为检索的代理查询。
要求：风格接近标准文档条款（可含数值、要求、定义），内容具体、忠实推断，不要编造标准号或数值。
只输出文档正文。"""


def _parse_classify(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        return {
            "type": str(data.get("type", "")),
            "need_rewrite": bool(data.get("need_rewrite", False)),
            "rewritten_query": str(data.get("rewritten_query", "")),
        }
    except Exception:
        return None


def classify_and_rewrite(query: str, llm) -> dict:
    """返回 {"type", "need_rewrite", "rewritten_query"}。"""
    resp = llm.invoke(CLASSIFY_PROMPT.format(query=query))
    result = _parse_classify(resp.content)
    if result is None:
        result = {"type": "未知", "need_rewrite": False, "rewritten_query": query}
    if not result["rewritten_query"]:
        result["rewritten_query"] = query
    return result


def rewrite_query(query: str, llm) -> str:
    """无条件重写查询。"""
    resp = llm.invoke(REWRITE_PROMPT.format(query=query))
    text = resp.content.strip().strip("""\"'`""")
    return text if text else query


def hyde_document(query: str, llm, chunk_size: int = 512) -> str:
    """生成假设文档（HyDE 代理查询）。"""
    resp = llm.invoke(HYDE_PROMPT.format(query=query, chunk_size=chunk_size))
    return resp.content.strip() or query
