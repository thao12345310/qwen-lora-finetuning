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

    # Stratify by group so each split has balanced coverage.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s["meta"]["group"]].append(s)

    train, valid, test = [], [], []
    for group, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * args.train_ratio)
        n_valid = int(n * args.valid_ratio)
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
