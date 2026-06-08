"""Run the rewriter on a single conversation (base model or fine-tuned).

Used both standalone (CLI) and as a library by src/api/main.py.

Example:
    python src/inference/predict.py \
        --config configs/inference.yaml \
        --conversation '[{"role":"user","content":"mở điều hoà"},{"role":"bot","content":"bạn muốn đặt bao nhiêu độ?"},{"role":"user","content":"27 độ"}]'
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prompts import REWRITE_TAG, SYSTEM_PROMPT_FOR_TRAINING


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def format_conversation(turns: list[dict]) -> str:
    """Render turns into the legacy labeled text format."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def build_trained_messages(conversation: list[dict]) -> list[dict]:
    """Render API turns into the same multi-message prompt used for SFT/eval.

    All previous turns are passed as chat history. The final user turn is tagged
    with <REWRITE> so the model knows exactly which utterance to rewrite.
    """
    if not conversation:
        raise ValueError("conversation must contain at least one turn")
    if conversation[-1].get("role") != "user":
        raise ValueError("the final turn must have role='user'")

    messages = [{"role": "system", "content": SYSTEM_PROMPT_FOR_TRAINING}]
    for i, turn in enumerate(conversation):
        role = turn.get("role")
        if role not in {"user", "bot"}:
            raise ValueError(f"unsupported role: {role!r}")
        content = str(turn.get("content", ""))
        if i == len(conversation) - 1:
            content = f"{REWRITE_TAG}\n{content}"
        messages.append({
            "role": "user" if role == "user" else "assistant",
            "content": content,
        })
    return messages


def parse_rewrite(raw: str) -> str:
    """Extract rewrite_message from model JSON, falling back to raw text."""
    txt = (raw or "").strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "rewrite_message" in obj:
            return str(obj["rewrite_message"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return txt


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def load_model(model_path: str, adapter_path: str | None, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _pick_device()
    model_kwargs = {"trust_remote_code": True}

    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    # CUDA path is already placed via device_map; MPS/CPU need an explicit move.
    if device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer


def rewrite(
    conversation: list[dict],
    cfg: dict,
    use_adapter: bool = True,
) -> str:
    adapter_path = cfg.get("adapter_path") if use_adapter else None
    if adapter_path and not Path(adapter_path).exists():
        adapter_path = None

    model, tokenizer = load_model(
        cfg["model_name_or_path"],
        adapter_path,
        cfg.get("load_in_4bit", False),
    )

    messages = build_trained_messages(conversation)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            do_sample=cfg["do_sample"],
            repetition_penalty=cfg["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help='JSON list of {"role": "user"|"bot", "content": "..."}',
    )
    parser.add_argument("--base-only", action="store_true", help="Skip the LoRA adapter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    turns = json.loads(args.conversation)
    out = rewrite(turns, cfg, use_adapter=not args.base_only)
    print(out)


if __name__ == "__main__":
    main()
