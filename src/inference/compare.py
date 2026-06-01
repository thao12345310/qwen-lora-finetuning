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

from tqdm import tqdm

from src.inference.predict import load_config, load_model
import torch


# Llama Factory sharegpt `from` -> chat-template role.
_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def parse_sample(s: dict) -> tuple[list[dict], str]:
    """Split a `conversations`-schema sample into (prompt_messages, gold).

    The final gpt turn is the gold rewrite; everything before it is the prompt
    context fed through the chat template (matching how the model was trained).
    """
    turns = [
        {"role": _ROLE_MAP[t["from"]], "content": t["value"]}
        for t in s["conversations"]
    ]
    if not turns or turns[-1]["role"] != "assistant":
        raise ValueError(
            "Expected each sample's last conversation turn to be 'gpt' (the gold "
            f"rewrite); got: {turns[-1] if turns else 'empty'}"
        )
    return turns[:-1], turns[-1]["content"]


def run_split(parsed, cfg, adapter_path: str | None):
    model, tokenizer = load_model(
        cfg["model_name_or_path"], adapter_path, cfg.get("load_in_4bit", False)
    )
    outputs = []
    for prompt_messages, _gold in tqdm(
        parsed, desc="base" if adapter_path is None else "lora"
    ):
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
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
    parsed = [parse_sample(s) for s in samples]

    base_outputs = run_split(parsed, cfg, adapter_path=None)
    # Free up GPU before loading the adapter version (lru_cache holds the base
    # in memory; clearing it is safest).
    load_model.cache_clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    adapter_path = cfg["adapter_path"]
    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at '{adapter_path}'.\n"
            "Please run training first:\n"
            "  python src/train/train.py --config configs/qwen_lora_sft.yaml"
        )
    lora_outputs = run_split(parsed, cfg, adapter_path=adapter_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s, (prompt_messages, gold), b, l in zip(
            samples, parsed, base_outputs, lora_outputs
        ):
            f.write(
                json.dumps(
                    {
                        "dialogue": [
                            m for m in prompt_messages if m["role"] != "system"
                        ],
                        "gold": gold,
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
