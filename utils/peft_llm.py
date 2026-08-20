"""微调后模型（LoRA）的推理封装：加载 Qwen2-7B-Instruct + LoRA adapter，直接生成答案。

用于评测与 gradio 中"微调后模型"选项，避免 GGUF 转换。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR

DEFAULT_ADAPTER_DIR = OUTPUT_DIR / "qwen2_7b_lora"
DEFAULT_MODEL_ID = "Qwen/Qwen2-7B-Instruct"


def _resolve_default_adapter():
    # 完整训练会保存最终 adapter 到顶层；被中断时用最新的 checkpoint
    if (DEFAULT_ADAPTER_DIR / "adapter_config.json").exists():
        return DEFAULT_ADAPTER_DIR
    ckpts = sorted(
        DEFAULT_ADAPTER_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    return ckpts[-1] if ckpts else DEFAULT_ADAPTER_DIR


DEFAULT_ADAPTER = _resolve_default_adapter()

_model_cache = {}


def load_peft_model(model_id: str = DEFAULT_MODEL_ID, adapter_path=DEFAULT_ADAPTER):
    """加载 4-bit 基础模型 + LoRA adapter（复用单例）。"""
    key = (model_id, str(adapter_path))
    if key not in _model_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"加载基础模型 {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # 8GB 显存下 bge(约1.2GB) 与 7B 同存，auto 会误判显存不足而分到 CPU；显式限 6GiB 强制全上 GPU
        base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, max_memory={0: "6GiB"},
        )
        print(f"加载 LoRA adapter {adapter_path} ...")
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        _model_cache[key] = (tokenizer, model)
    return _model_cache[key]


def answer_with_peft(question: str, context: str, max_new_tokens: int = 512,
                     model_id: str = DEFAULT_MODEL_ID, adapter_path=DEFAULT_ADAPTER) -> str:
    """用微调模型基于检索上下文生成答案（应用 Qwen chat 模板）。"""
    import torch

    tokenizer, model = load_peft_model(model_id, adapter_path)
    system = "你是新能源汽车标准领域的专业问答助手。请严格依据给定的标准文档内容回答问题；若文档未提及，请直接说明不知道，不要编造。"
    user = f"检索到的标准文档内容：\n\n<docs>{context}</docs>\n\n用户问题：<question>{question}</question>"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip() or "我不知道。"


if __name__ == "__main__":
    # 简单自测（adapter 存在时）
    if DEFAULT_ADAPTER.exists():
        print(answer_with_peft("车窗防夹力应不大于多少牛？", "汽车防夹系统标准中，车窗防夹力应不大于 100 N。"))
    else:
        print("尚未找到微调 adapter，跳过自测。")
