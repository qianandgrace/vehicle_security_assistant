"""QLoRA 微调 Qwen2 模型（基于标准问答指令数据）。

用法：
    # 先用小模型验证训练栈
    python scripts/step6_finetune_qwen.py --smoke
    # 正式微调 7B（需先下载 Qwen2-7B-Instruct）
    python scripts/step6_finetune_qwen.py --model Qwen/Qwen2-7B-Instruct --output output/qwen2_7b_lora

输出：LoRA adapter（含 tokenizer）到 output/ 指定目录。
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR

DEFAULT_DATA = OUTPUT_DIR / "finetune_data.jsonl"


def build_dataset(tokenizer, data_path, max_len=1024):
    from datasets import Dataset

    rows = []
    for line in open(data_path, encoding="utf-8"):
        rec = json.loads(line)
        # 只取 user 与 assistant 轮（Qwen chat 模板）
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": rec["messages"][1]["content"]},
             {"role": "assistant", "content": rec["messages"][2]["content"]}],
            tokenize=False,
        )
        rows.append({"text": text})
    ds = Dataset.from_list(rows)

    def tokenize_fn(ex):
        out = tokenizer(ex["text"], truncation=True, max_length=max_len, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tokenize_fn, remove_columns=["text"])
    return ds


def train(args):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = args.model
    print(f"加载模型：{model_name}")

    if args.smoke:
        # 4-bit 量化对小模型也生效，验证栈
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = build_dataset(tokenizer, args.data, max_len=args.max_len)
    print(f"训练样本数：{len(data)}")

    args_t = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=args.fp16,
        bf16=not args.fp16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args_t, train_dataset=data)
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter 已保存：{args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--output", default=str(OUTPUT_DIR / "qwen2_7b_lora"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--fp16", action="store_true", help="FP16；默认 bf16")
    p.add_argument("--smoke", action="store_true", help="用 0.5B 小模型验证训练栈")
    args = p.parse_args()
    if args.smoke:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
        args.output = str(OUTPUT_DIR / "smoke_lora")
        args.epochs = 1
    train(args)


if __name__ == "__main__":
    main()
