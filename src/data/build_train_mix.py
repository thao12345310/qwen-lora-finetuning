"""Build versioned train/valid/test data from a YAML recipe.

Example:
    python3 -m src.data.build_train_mix --recipe data/recipes/v3.yaml

The recipe controls which sources are split and which sources are appended only
to train. Outputs default to the recipe's output_dir, usually data/processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Keep the CLI usable before requirements.txt is installed.
    yaml = None

from src.data.split_data import read_jsonl, split_samples, write_jsonl


SHAREGPT_DATASET_INFO = {
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"},
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "system_tag": "system",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed: int, name: str) -> int:
    blob = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(blob[:8], "big")


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line.rstrip()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(text: str) -> Any:
    """Small YAML subset parser for repo recipes when PyYAML is unavailable."""
    raw_lines = []
    for line in text.splitlines():
        line = strip_comment(line)
        if line.strip():
            raw_lines.append(line)

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def parse_key_value(content: str) -> tuple[str, str]:
        if ":" not in content:
            raise ValueError(f"Expected key: value, got {content!r}")
        key, value = content.split(":", 1)
        return key.strip(), value.strip()

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(raw_lines):
            return {}, index
        content = raw_lines[index][indent:]
        if content.startswith("- "):
            return parse_list(index, indent)
        return parse_map(index, indent)

    def parse_map(index: int, indent: int) -> tuple[dict[str, Any], int]:
        out: dict[str, Any] = {}
        while index < len(raw_lines):
            line = raw_lines[index]
            cur_indent = indent_of(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError(f"Unexpected indentation: {line!r}")
            content = line[indent:]
            if content.startswith("- "):
                break
            key, value = parse_key_value(content)
            index += 1
            if value:
                out[key] = parse_scalar(value)
            else:
                next_indent = indent_of(raw_lines[index]) if index < len(raw_lines) else indent + 2
                out[key], index = parse_block(index, next_indent)
        return out, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        out: list[Any] = []
        while index < len(raw_lines):
            line = raw_lines[index]
            cur_indent = indent_of(line)
            if cur_indent < indent:
                break
            if cur_indent != indent:
                raise ValueError(f"Unexpected list indentation: {line!r}")
            content = line[indent:]
            if not content.startswith("- "):
                break
            item = content[2:].strip()
            index += 1
            if not item:
                next_indent = indent_of(raw_lines[index]) if index < len(raw_lines) else indent + 2
                value, index = parse_block(index, next_indent)
                out.append(value)
            elif ":" in item:
                key, value_text = parse_key_value(item)
                value: dict[str, Any] = {}
                if value_text:
                    value[key] = parse_scalar(value_text)
                else:
                    next_indent = indent_of(raw_lines[index]) if index < len(raw_lines) else indent + 2
                    value[key], index = parse_block(index, next_indent)
                if index < len(raw_lines) and indent_of(raw_lines[index]) > indent:
                    rest, index = parse_map(index, indent + 2)
                    value.update(rest)
                out.append(value)
            else:
                out.append(parse_scalar(item))
        return out, index

    parsed, index = parse_block(0, 0)
    if index != len(raw_lines):
        raise ValueError("Could not parse entire recipe")
    return parsed


def load_recipe(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        recipe = yaml.safe_load(text)
    else:
        recipe = simple_yaml_load(text)
    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe must be a YAML mapping: {path}")
    if "sources" not in recipe or not isinstance(recipe["sources"], list):
        raise ValueError("Recipe must define a sources list")
    return recipe


def source_take_count(take: Any, n: int) -> tuple[int, int]:
    if take is None or take == "all":
        return n, 0
    if not isinstance(take, int) or take < 0:
        raise ValueError(f"source.take must be a non-negative int or 'all', got {take!r}")
    return min(take, n), max(take - n, 0)


def get_gold(row: dict) -> str | None:
    for turn in reversed(row.get("conversations", [])):
        if turn.get("from") != "gpt":
            continue
        value = turn.get("value", "")
        try:
            parsed = json.loads(value)
        except Exception:
            return value
        return parsed.get("rewrite_message") or value
    return None


def load_bench_golds(path: Path | None) -> set[str]:
    if path is None:
        return set()
    golds = set()
    for row in read_jsonl(path):
        gold = get_gold(row)
        if gold:
            golds.add(gold)
    return golds


def dedupe_key(row: dict, mode: str | None) -> str | None:
    if not mode:
        return None
    if mode == "conversations":
        value = row.get("conversations", row)
    elif mode == "gold":
        value = get_gold(row)
    else:
        raise ValueError(f"Unsupported guards.dedupe_by: {mode}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def wrap(row: dict, source_name: str) -> dict[str, Any]:
    return {"record": row, "source": source_name, "meta": row.get("meta", {})}


def load_source_rows(source: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = source.get("name")
    if not name:
        raise ValueError("Each source must have a name")
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    rows = read_jsonl(path)
    count, shortfall = source_take_count(source.get("take", "all"), len(rows))
    selected = list(rows)
    if count < len(selected):
        rng = random.Random(stable_seed(seed, str(name)))
        rng.shuffle(selected)
        selected = selected[:count]

    stats = {
        "name": name,
        "path": str(path),
        "placement": source.get("placement", "split"),
        "requested_take": source.get("take", "all"),
        "loaded": len(rows),
        "selected_before_guards": len(selected),
        "shortfall": shortfall,
        "dropped_duplicate": 0,
        "dropped_bench_leak": 0,
        "accepted": 0,
    }
    return [wrap(row, str(name)) for row in selected], stats


def apply_guards(
    rows: list[dict[str, Any]],
    *,
    source_stats: dict[str, dict[str, Any]],
    seen: set[str],
    dedupe_by: str | None,
    bench_golds: set[str],
    bench_policy: str,
) -> list[dict[str, Any]]:
    accepted = []
    for item in rows:
        stats = source_stats[item["source"]]
        row = item["record"]

        key = dedupe_key(row, dedupe_by)
        if key is not None:
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)

        gold = get_gold(row)
        if bench_golds and gold in bench_golds:
            if bench_policy == "fail":
                raise ValueError(f"Bench leak in source {item['source']}: {gold}")
            if bench_policy == "drop":
                stats["dropped_bench_leak"] += 1
                continue

        stats["accepted"] += 1
        accepted.append(item)
    return accepted


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): counter[k] for k in sorted(counter, key=lambda x: str(x))}


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source, by_domain, by_context, by_pattern = Counter(), Counter(), Counter(), Counter()
    by_online_offline = Counter()
    for item in rows:
        meta = item["record"].get("meta", {})
        by_source[item["source"]] += 1
        by_domain[meta.get("domain", "unknown")] += 1
        by_online_offline[meta.get("online_offline", "unknown")] += 1
        ctx = meta.get("context_required")
        by_context["unknown" if ctx is None else str(ctx).lower()] += 1
        by_pattern[meta.get("pattern") or meta.get("generator") or "unknown"] += 1
    return {
        "total": len(rows),
        "by_source": counter_dict(by_source),
        "by_domain": counter_dict(by_domain),
        "by_online_offline": counter_dict(by_online_offline),
        "by_context_required": counter_dict(by_context),
        "by_pattern": counter_dict(by_pattern),
    }


def build_dataset_info() -> dict[str, dict[str, Any]]:
    out = {}
    for name in ["train", "valid", "test"]:
        spec = {"file_name": f"{name}.jsonl"}
        spec.update(SHAREGPT_DATASET_INFO)
        out[name] = spec
    return out


def render_report(recipe: dict[str, Any], counts: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    lines = [
        f"# Dataset mix report: {recipe.get('version', 'unknown')}",
        "",
        "## Split totals",
    ]
    for split in ["train", "valid", "test"]:
        lines.append(f"- {split}: {counts[split]['total']}")

    lines.extend(["", "## Source counts"])
    for split in ["train", "valid", "test"]:
        bits = ", ".join(f"{k}={v}" for k, v in counts[split]["by_source"].items())
        lines.append(f"- {split}: {bits or 'none'}")

    lines.extend(["", "## Train distribution"])
    for key in ["by_domain", "by_online_offline", "by_context_required", "by_pattern"]:
        bits = ", ".join(f"{k}={v}" for k, v in counts["train"][key].items())
        lines.append(f"- {key}: {bits or 'none'}")

    lines.extend(["", "## Source guard summary"])
    for src in sources:
        lines.append(
            "- {name}: loaded={loaded}, selected={selected_before_guards}, "
            "accepted={accepted}, dup_drop={dropped_duplicate}, "
            "bench_drop={dropped_bench_leak}, shortfall={shortfall}".format(**src)
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    outputs: dict[str, list[dict[str, Any]]],
    *,
    recipe_path: Path,
    recipe: dict[str, Any],
    sources: list[dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for split, rows in outputs.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, [item["record"] for item in rows])
        output_files[split] = {
            "file_name": path.name,
            "count": len(rows),
            "sha256": sha256_file(path),
        }

    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(build_dataset_info(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_files["dataset_info"] = {
        "file_name": dataset_info_path.name,
        "sha256": sha256_file(dataset_info_path),
    }

    report = render_report(recipe, counts, sources)
    report_path = output_dir / "dataset_report.md"
    report_path.write_text(report, encoding="utf-8")
    output_files["dataset_report"] = {
        "file_name": report_path.name,
        "sha256": sha256_file(report_path),
    }

    manifest = {
        "version": recipe.get("version"),
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256_file(recipe_path),
        "output_dir": str(output_dir),
        "seed": recipe.get("seed", 42),
        "split": recipe.get("split", {}),
        "guards": recipe.get("guards", {}),
        "sources": sources,
        "counts": counts,
        "outputs": output_files,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_from_recipe(recipe_path: Path, output_dir_override: Path | None, dry_run: bool) -> str:
    recipe = load_recipe(recipe_path)
    seed = int(recipe.get("seed", 42))
    output_dir = output_dir_override or Path(recipe.get("output_dir", "data/processed"))
    split_cfg = recipe.get("split", {})
    guards = recipe.get("guards", {})
    dedupe_by = guards.get("dedupe_by")
    bench_path = Path(guards["bench_leak_check"]) if guards.get("bench_leak_check") else None
    bench_policy = guards.get("bench_leak_policy", "drop")
    if bench_policy not in {"drop", "fail", "ignore"}:
        raise ValueError("guards.bench_leak_policy must be one of: drop, fail, ignore")
    bench_golds = set() if bench_policy == "ignore" else load_bench_golds(bench_path)

    source_stats: dict[str, dict[str, Any]] = {}
    split_pool: list[dict[str, Any]] = []
    train_only_pool: list[dict[str, Any]] = []
    for source in recipe["sources"]:
        rows, stats = load_source_rows(source, seed)
        source_stats[stats["name"]] = stats
        placement = stats["placement"]
        if placement == "split":
            split_pool.extend(rows)
        elif placement == "train_only":
            train_only_pool.extend(rows)
        else:
            raise ValueError(f"Unsupported source placement: {placement}")

    seen: set[str] = set()
    split_pool = apply_guards(
        split_pool,
        source_stats=source_stats,
        seen=seen,
        dedupe_by=dedupe_by,
        bench_golds=bench_golds,
        bench_policy=bench_policy,
    )

    train, valid, test = split_samples(
        split_pool,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        valid_ratio=float(split_cfg.get("valid_ratio", 0.1)),
        seed=int(split_cfg.get("seed", seed)),
        stratify_by=split_cfg.get("stratify_by"),
    )

    train_only = apply_guards(
        train_only_pool,
        source_stats=source_stats,
        seen=seen,
        dedupe_by=dedupe_by,
        bench_golds=bench_golds,
        bench_policy=bench_policy,
    )
    train = train + train_only
    random.Random(seed).shuffle(train)

    outputs = {"train": train, "valid": valid, "test": test}
    counts = {split: count_rows(rows) for split, rows in outputs.items()}
    sources = list(source_stats.values())
    report = render_report(recipe, counts, sources)

    if not dry_run:
        write_outputs(
            output_dir,
            outputs,
            recipe_path=recipe_path,
            recipe=recipe,
            sources=sources,
            counts=counts,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_from_recipe(args.recipe, args.output_dir, args.dry_run)
    print(report, end="")
    if args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
