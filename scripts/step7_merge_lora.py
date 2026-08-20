"""把 LoRA adapter 合并进基础模型，保存为 fp16 全量模型（供 GGUF 转换）。

用法：
    python scripts/step7_merge_lora.py
输出：output/qwen2_7b_merged/（fp16 safetensors）
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR

BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
ADAPTER = OUTPUT_DIR / "qwen2_7b_lora" / "checkpoint-120"
OUT_DIR = OUTPUT_DIR / "qwen2_7b_merged"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"加载基础模型（CPU fp16）：{BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"加载 LoRA adapter：{ADAPTER}")
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("合并 LoRA 权重（merge_and_unload）...")
    model = model.merge_and_unload()
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"保存全量模型：{OUT_DIR}")
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)
    print("合并完成。")


if __name__ == "__main__":
    main()
