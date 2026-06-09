"""Split a jsonl dataset into train/valid/test, stratified by metadata fields."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_STRATIFY_BY = ["meta.domain", "meta.online_offline", "meta.context_required"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def nested_get(row: dict, dotted_key: str):
    value = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def strat_key(sample: dict, stratify_by: list[str] | None = None) -> str:
    keys = stratify_by or DEFAULT_STRATIFY_BY
    values = []
    for key in keys:
        value = nested_get(sample, key)
        if value is None:
            value = "na"
        elif isinstance(value, bool):
            value = str(value).lower()
        values.append(str(value))
    return "|".join(values)


def split_samples(
    samples: list[dict],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return deterministic stratified train/valid/test splits."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        buckets[strat_key(sample, stratify_by)].append(sample)

    train, valid, test = [], [], []
    for _group, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        # round() instead of floor() so a 0.1 valid ratio on small strata does not
        # collapse to 0; guarantee at least 1 valid item per non-trivial bucket so
        # the eval set is never starved.
        n_valid = round(n * valid_ratio)
        if n_valid == 0 and n >= 3:
            n_valid = 1
        n_train = min(round(n * train_ratio), n - n_valid)
        train.extend(items[:n_train])
        valid.extend(items[n_train : n_train + n_valid])
        test.extend(items[n_train + n_valid :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/dialogues.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    train, valid, test = split_samples(
        samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, split)
        print(f"  {name}: {len(split)} -> {path}")


if __name__ == "__main__":
    main()
