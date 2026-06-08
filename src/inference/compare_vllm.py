"""Compare base vs LoRA via a vLLM OpenAI-compatible server.

vLLM doesn't run on Apple Silicon — point this client at a vLLM instance running
on Colab/Kaggle/Linux+CUDA (e.g. via ngrok).

Server side (on a CUDA box):
    vllm serve Qwen/Qwen2.5-1.5B-Instruct \\
      --enable-lora \\
      --lora-modules vi-rewriter=outputs/qwen-dialogue-rewriter-lora \\
      --max-lora-rank 8 --port 8000

Client side:
    python -m src.inference.compare_vllm \\
      --url http://<ngrok-url>  \\
      --conversation '[{"role":"user","content":"mở điều hoà"},
                       {"role":"bot","content":"bạn muốn đặt bao nhiêu độ?"},
                       {"role":"user","content":"27 độ"}]'
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

from src.inference.predict import build_trained_messages


def chat(url: str, model: str, messages: list[dict], cfg: dict) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": cfg["max_new_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-name", default="vi-rewriter")
    parser.add_argument(
        "--conversation",
        required=True,
        help='JSON list of {"role": "user"|"bot", "content": "..."}',
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.open())
    turns = json.loads(args.conversation)
    messages = build_trained_messages(turns)

    base = chat(args.url, args.base_model, messages, cfg)
    lora = chat(args.url, args.lora_name, messages, cfg)

    print(json.dumps(
        {"base_model_output": base, "fine_tuned_output": lora},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
