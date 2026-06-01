"""QLoRA fine-tune Qwen2.5-1.5B-Instruct on the dialogue rewrite dataset.

Designed for T4 16GB (Kaggle/Colab) with bitsandbytes 4-bit.
Falls back to LoRA-only (no quantization) if `load_in_4bit: false` in config —
useful on machines without bitsandbytes (e.g. Apple Silicon).

Usage:
    python src/train/train.py --config configs/qwen_lora_sft.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL ≥ 0.17 dropped max_seq_length from SFTTrainer; set it on the tokenizer instead
    # so truncation still applies during internal collation.
    tokenizer.model_max_length = cfg["cutoff_len"]

    load_in_4bit = cfg.get("load_in_4bit", True)
    model_kwargs = {"trust_remote_code": True}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # Qwen2.5 native dtype; avoids fp16 GradScaler conflict
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb
        # Under DDP (torchrun sets LOCAL_RANK) each process must load the full model
        # onto its OWN gpu so the 2× T4 run is data-parallel. "auto" would spread one
        # copy across both gpus (model-parallel) and leave the 2nd gpu idle for a 1.5B.
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        model_kwargs["device_map"] = {"": local_rank} if local_rank != -1 else "auto"
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Train from scratch even if a checkpoint exists in output_dir (default: auto-resume).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Override config to push the adapter to HuggingFace Hub (needs HF_TOKEN with write).",
    )
    parser.add_argument("--hub-model-id", default=None, help="Override hub_model_id from config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides — let the Kaggle/Colab notebook toggle Hub push without editing the yaml.
    if args.push_to_hub:
        cfg["push_to_hub"] = True
    if args.hub_model_id:
        cfg["hub_model_id"] = args.hub_model_id

    data_files = {"train": cfg["train_file"], "validation": cfg["valid_file"]}
    dataset = load_dataset("json", data_files=data_files)

    # Processed data is Llama-Factory sharegpt format: a `conversations` column of
    # [{"from": system|human|gpt, "value": ...}]. SFTTrainer's chat-template path
    # expects a `messages` column of [{"role": system|user|assistant, "content":
    # ...}], so convert here. (Legacy `messages`-format files pass through.)
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}

    def to_messages(example):
        return {
            "messages": [
                {"role": role_map.get(t["from"], t["from"]), "content": t["value"]}
                for t in example["conversations"]
            ]
        }

    cols = dataset["train"].column_names
    if "conversations" in cols:
        dataset = dataset.map(to_messages, remove_columns=cols)
    else:
        dataset = dataset.remove_columns([c for c in cols if c != "messages"])

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
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        seed=cfg["seed"],
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DDP: only LoRA params have requires_grad, so there are no unused params to
        # search for — disabling the search avoids the extra all-reduce overhead.
        ddp_find_unused_parameters=False,
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id"),
        hub_strategy=cfg.get("hub_strategy", "end"),
        hub_private_repo=cfg.get("hub_private_repo", False),
    )

    # TRL ≥ 0.16: tokenizer → processing_class
    # TRL ≥ 0.17: max_seq_length removed from SFTTrainer entirely; truncation via tokenizer.model_max_length
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Auto-resume from the latest checkpoint so a killed Kaggle session (9–12h cap,
    # idle timeout) continues instead of restarting from step 0.
    resume_from = None
    out_dir = Path(cfg["output_dir"])
    if not args.no_resume and out_dir.is_dir():
        resume_from = get_last_checkpoint(str(out_dir))
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved LoRA adapter to {cfg['output_dir']}")

    if cfg.get("push_to_hub"):
        trainer.push_to_hub(commit_message="End of training")
        print(f"Pushed adapter to https://huggingface.co/{cfg['hub_model_id']}")


if __name__ == "__main__":
    main()
