"""QLoRA fine-tune Qwen2.5-1.5B-Instruct on the dialogue rewrite dataset.

Designed for T4 16GB (Kaggle/Colab) with bitsandbytes 4-bit.
Falls back to LoRA-only (no quantization) if `load_in_4bit: false` in config —
useful on machines without bitsandbytes (e.g. Apple Silicon).

Usage:
    python src/train/train.py --config configs/qwen_lora_sft.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = cfg.get("load_in_4bit", True)
    model_kwargs = {"trust_remote_code": True}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen_lora_sft.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_files = {"train": cfg["train_file"], "validation": cfg["valid_file"]}
    dataset = load_dataset("json", data_files=data_files)
    # Drop `meta` column — SFTTrainer only needs `messages`.
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c != "messages"])

    model, tokenizer = build_model_and_tokenizer(cfg)

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"],
        eval_strategy="steps",
        fp16=cfg["fp16"],
        seed=cfg["seed"],
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # max_seq_length and packing moved from SFTConfig to SFTTrainer in TRL ≥ 0.16
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=cfg["cutoff_len"],
        packing=False,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
