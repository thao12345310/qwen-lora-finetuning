"""Convert template-generated OpenAI-format jsonl → Llama Factory `conversations` format.

The legacy generators (`generate_dataset.py`, `generate_multi_turn.py`) emit:

    {
      "messages": [
        {"role": "system",    "content": "<old system>"},
        {"role": "user",      "content": "user: bật nhạc\\nbot: bài gì?\\nuser: Em Của..."},
        {"role": "assistant", "content": "Tôi muốn phát bài Em Của..."}
      ],
      "meta": {...}
    }

This script re-parses the single user blob into individual turns, wraps the
final user turn with the <REWRITE> tag, and produces:

    {
      "conversations": [
        {"from": "system", "value": "<NEW rewrite-only system prompt>"},
        {"from": "human",  "value": "bật nhạc"},
        {"from": "gpt",    "value": "bài gì?"},
        {"from": "human",  "value": "<REWRITE>\\nEm Của..."},
        {"from": "gpt",    "value": "{\\"rewrite_message\\": \\"Tôi muốn phát bài Em Của...\\"}"}
      ],
      "meta": {...}
    }

Usage:
    python -m src.data.convert_to_lf \\
        --input data/raw/dialogues_merged.jsonl \\
        --output data/train/dialogues_train_template.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data.prompts import REWRITE_TAG, SYSTEM_PROMPT_FOR_TRAINING

ROLE_LINE_RE = re.compile(r"^(user|bot)\s*:\s*(.+)$", re.IGNORECASE)


def parse_turns(blob: str) -> list[dict]:
    """Parse 'user: X\\nbot: Y\\nuser: Z' → [{'role':'user','content':'X'},...]."""
    turns: list[dict] = []
    current: dict | None = None
    for line in blob.split("\n"):
        m = ROLE_LINE_RE.match(line)
        if m:
            if current:
                turns.append(current)
            current = {"role": m.group(1).lower(), "content": m.group(2).strip()}
        elif current and line.strip():
            # Continuation line (e.g., wrapped content) — append to current turn.
            current["content"] += " " + line.strip()
    if current:
        turns.append(current)
    return turns


def to_lf_record(record: dict, system_prompt: str) -> dict | None:
    msgs = record["messages"]
    if len(msgs) < 3:
        return None
    user_blob = next((m["content"] for m in msgs if m["role"] == "user"), None)
    assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if user_blob is None or assistant is None:
        return None

    turns = parse_turns(user_blob)
    if not turns or turns[-1]["role"] != "user":
        return None

    conversations = [{"from": "system", "value": system_prompt}]
    for t in turns[:-1]:
        from_ = "human" if t["role"] == "user" else "gpt"
        conversations.append({"from": from_, "value": t["content"]})
    final_user = turns[-1]["content"]
    conversations.append({"from": "human", "value": f"{REWRITE_TAG}\n{final_user}"})
    answer = json.dumps({"rewrite_message": assistant.strip()}, ensure_ascii=False)
    conversations.append({"from": "gpt", "value": answer})

    return {"conversations": conversations, "meta": record.get("meta", {})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/dialogues_merged.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/train/dialogues_train_template.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    converted: list[dict] = []
    skipped = 0
    for r in records:
        lf = to_lf_record(r, SYSTEM_PROMPT_FOR_TRAINING)
        if lf is None:
            skipped += 1
            continue
        converted.append(lf)

    with args.output.open("w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Converted {len(converted)} samples (skipped {skipped}) → {args.output}")


if __name__ == "__main__":
    main()
