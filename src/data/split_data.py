"""Split data/raw/dialogues.jsonl into train/valid/test, stratified by group."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/dialogues.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = [json.loads(line) for line in args.input.open(encoding="utf-8")]

    # Stratify by domain × context_required so every split keeps the domain mix
    # AND the context / no-context ratio balanced. Falls back gracefully if a
    # sample is missing either key.
    def strat_key(s: dict) -> str:
        meta = s.get("meta", {})
        domain = meta.get("domain", "unknown")
        ctx = meta.get("context_required")
        ctx = "na" if ctx is None else str(ctx).lower()
        return f"{domain}|{ctx}"

    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[strat_key(s)].append(s)

    train, valid, test = [], [], []
    for group, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        # round() instead of floor() so a 0.1 valid ratio on small strata does not
        # collapse to 0; guarantee at least 1 valid item per non-trivial bucket so
        # the eval set is never starved.
        n_valid = round(n * args.valid_ratio)
        if n_valid == 0 and n >= 3:
            n_valid = 1
        n_train = min(round(n * args.train_ratio), n - n_valid)
        train.extend(items[:n_train])
        valid.extend(items[n_train : n_train + n_valid])
        test.extend(items[n_train + n_valid :])

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in split:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} -> {path}")


if __name__ == "__main__":
    main()
