"""新能源汽车标准 RAG 对话界面（gradio）。

启动：python app.py，浏览器打开 http://127.0.0.1:7860
LLM 可选：deepseek 云 API / 本地 ollama（qwen2:7b）
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from utils.rag_pipeline import VehicleRAG

LLM_CHOICES = [
    ("DeepSeek 云 API (deepseek-v4-flash)", "deepseek"),
    ("本地 ollama (qwen2:7b)", "ollama"),
    ("本地微调后模型 (qwen2:7b LoRA)", "peft"),
]

_pipelines = {}


def get_pipeline(llm_type: str) -> VehicleRAG:
    if llm_type not in _pipelines:
        _pipelines[llm_type] = VehicleRAG(llm_type=llm_type)
    return _pipelines[llm_type]


def chat(question: str, llm_label: str, top_k: int, show_sources: bool):
    if not question or not question.strip():
        return "请输入问题。", None, None, None
    llm_type = dict(LLM_CHOICES)[llm_label]
    # 微调后模型：检索用同一 pipeline，生成用 peft 模型
    if llm_type == "peft":
        pipe = get_pipeline("ollama")
        docs, route = pipe.retrieve(question, top_k=top_k)
        ctx = [d.page_content for d in docs]
        from utils.peft_llm import answer_with_peft
        answer = answer_with_peft(question, "\n\n".join(ctx))
        sources_docs = docs
    else:
        pipe = get_pipeline(llm_type)
        out = pipe.answer(question, top_k=top_k)
        answer = out["answer"]
        route = out["route"]
        ctx = out["contexts"]
        sources_docs = [type("D", (), {"metadata": {"doc_name": s}})() for s in out["sources"]]

    route_info = (
        f"**查询类型**：{route['query_type']}\n\n"
        f"**检索查询**：{route['search_query']}\n\n"
        f"**是否重写**：{'是' if route['need_rewrite'] else '否'}"
    )

    sources = None
    if show_sources:
        blocks = []
        for i, (ctx, doc) in enumerate(zip(ctx, sources_docs), 1):
            src = doc.metadata.get("doc_name", "?")
            blocks.append(f"**[{i}] 来源：{src}**\n\n{ctx}\n\n---")
        sources = "\n".join(blocks)

    return answer, route_info, sources, f"共召回 {len(ctx)} 个片段"


with gr.Blocks(title="新能源汽车标准 RAG 问答") as demo:
    gr.Markdown(
        "# 🚗 新能源汽车标准 RAG 问答助手\n\n"
        "**检索链路**：意图路由 → HyPE 稠密 + BM25 稀疏 → RRF 重排 → U型排序 → LLM 生成。\n\n"
        "覆盖 16 个 GB/GBT/QCT 新能源汽车标准（碰撞安全、动力电池、智能网联、儿童约束等）。"
    )
    with gr.Row():
        llm = gr.Dropdown(choices=[label for label, _ in LLM_CHOICES],
                          value=LLM_CHOICES[0][0], label="LLM 模型")
        top_k = gr.Slider(3, 10, value=5, step=1, label="检索片段数 top_k")
        show_src = gr.Checkbox(value=True, label="显示检索来源")
    with gr.Row():
        qbox = gr.Textbox(placeholder="例如：车窗防夹力应不大于多少牛？", label="你的问题", lines=2)
        submit = gr.Button("检索问答", variant="primary")
    with gr.Row():
        answer = gr.Textbox(label="回答", lines=6)
        route_info = gr.Markdown("**查询类型**：")
    status = gr.Markdown("")
    sources = gr.Markdown("")

    submit.click(chat, inputs=[qbox, llm, top_k, show_src],
                 outputs=[answer, route_info, sources, status])
    qbox.submit(chat, inputs=[qbox, llm, top_k, show_src],
                outputs=[answer, route_info, sources, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
