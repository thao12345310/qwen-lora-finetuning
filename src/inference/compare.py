"""Run base model vs fine-tuned LoRA on the test set, side-by-side.

Output: outputs/eval_results/comparison.jsonl with one row per test sample:
    {dialogue, gold, base_output, finetuned_output, meta}

The script keeps both models in memory simultaneously is expensive; instead we
run all samples through the base model first, unload, then load the adapter
and run all samples through the fine-tuned model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.inference.predict import format_conversation, load_config, load_model
import torch


def parse_user_block(text: str) -> list[dict]:
    """Reverse of format_conversation: 'user: ..\\nbot: ..' -> list of dicts."""
    turns = []
    for line in text.split("\n"):
        if ":" not in line:
            continue
        role, content = line.split(":", 1)
        turns.append({"role": role.strip(), "content": content.strip()})
    return turns


def run_split(samples, cfg, adapter_path: str | None):
    model, tokenizer = load_model(
        cfg["model_name_or_path"], adapter_path, cfg.get("load_in_4bit", False)
    )
    outputs = []
    for s in tqdm(samples, desc="base" if adapter_path is None else "lora"):
        user_text = s["messages"][1]["content"]
        messages = [
            {"role": "system", "content": cfg["system_prompt"].strip()},
            {"role": "user", "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                do_sample=cfg["do_sample"],
                repetition_penalty=cfg["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--test_file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/eval_results/comparison.jsonl")
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = [json.loads(l) for l in args.test_file.open(encoding="utf-8")]
    if args.limit:
        samples = samples[: args.limit]

    base_outputs = run_split(samples, cfg, adapter_path=None)
    # Free up GPU before loading the adapter version (lru_cache holds the base
    # in memory; clearing it is safest).
    load_model.cache_clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    lora_outputs = run_split(samples, cfg, adapter_path=cfg["adapter_path"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s, b, l in zip(samples, base_outputs, lora_outputs):
            f.write(
                json.dumps(
                    {
                        "dialogue": s["messages"][1]["content"],
                        "gold": s["messages"][2]["content"],
                        "base_output": b,
                        "finetuned_output": l,
                        "meta": s.get("meta", {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(samples)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
