#!/usr/bin/env python3
"""Convert merged.jsonl (role/content/label turns) → eval_bench format (from/value).

eval_bench.lf_to_messages reads each turn as {"from", "value"} and maps `from`
via ROLE_MAP = {"system","human","gpt"}. merged.jsonl instead uses
{"role","content","label"} with roles {"system","user","assistant"}. This script
remaps role→from (user→human, assistant→gpt) and content→value, dropping `label`,
so the file can be fed to eval_bench unchanged.

Usage:
    python scripts/convert_merged_to_bench.py data/merged.jsonl data/bench/merged_bench.jsonl
"""
import json
import sys
from pathlib import Path

ROLE_TO_FROM = {"system": "system", "user": "human", "assistant": "gpt"}


def convert_record(rec: dict) -> dict:
    convs = rec["conversations"]
    out = []
    for c in convs:
        role = c["role"]
        if role not in ROLE_TO_FROM:
            raise ValueError(f"unknown role {role!r}")
        out.append({"from": ROLE_TO_FROM[role], "value": c["content"]})
    # eval_bench asserts the final turn is the 'gpt' gold answer with JSON value.
    assert out[-1]["from"] == "gpt", f"last turn must be assistant, got {out[-1]['from']}"
    json.loads(out[-1]["value"])  # sanity: gold is parseable JSON
    return {"conversations": out}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/merged.jsonl")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/bench/merged_bench.jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(convert_record(json.loads(line)), ensure_ascii=False) + "\n")
            n += 1
    print(f"Converted {n} records → {dst}")


if __name__ == "__main__":
    main()
