#!/usr/bin/env python3
"""Run a 100-case stratified smoke benchmark against two remote deployments.

The first deployment is a Gradio app with /rewrite. The second is a vLLM
OpenAI-compatible endpoint, evaluated with the vi-rewriter LoRA model by default.

No third-party packages are required; this is intentionally stdlib-only so it can
run in a fresh local checkout.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}
TAG_PREFIXES = ("<REWRITE_AND_CLASSIFY>\n", "<REWRITE>\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("data/bench/dialogues_bench_v3.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--gradio-url", default="https://helpful-quilt-aground.ngrok-free.dev")
    parser.add_argument("--vllm-url", default="https://overtime-freely-glider.ngrok-free.dev")
    parser.add_argument("--vllm-model", default="vi-rewriter")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").lower().replace("°c", " độ ").replace("°", " độ ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def strip_tag(text: str) -> str:
    for prefix in TAG_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def record_to_case(record: dict[str, Any], index: int) -> dict[str, Any]:
    convs = record["conversations"]
    gold_obj = json.loads(convs[-1]["value"])
    gold_rewrite = str(gold_obj.get("rewrite_message", "")).strip()
    gold_domain = gold_obj.get("domain") or record.get("meta", {}).get("online_offline")

    messages = [{"role": ROLE_MAP[c["from"]], "content": c["value"]} for c in convs[:-1]]
    turns = []
    body = [c for c in convs if c["from"] != "system"][:-1]
    for c in body:
        turns.append({
            "role": "user" if c["from"] == "human" else "bot",
            "content": strip_tag(str(c["value"])),
        })

    history_lines = [
        f"{'user' if t['role'] == 'user' else 'assistant'}: {t['content']}"
        for t in turns[:-1]
    ]
    return {
        "case_id": index,
        "meta": record.get("meta", {}),
        "turns": turns,
        "messages": messages,
        "history_text": "\n".join(history_lines),
        "final_user": turns[-1]["content"] if turns else "",
        "gold_rewrite": gold_rewrite,
        "gold_domain": gold_domain,
    }


def select_stratified(records: list[dict[str, Any]], limit: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    by_pattern: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, record in enumerate(records):
        pattern = record.get("meta", {}).get("pattern", "unknown")
        by_pattern[pattern].append((i, record))

    rng = random.Random(seed)
    patterns = sorted(by_pattern)
    base = limit // len(patterns)
    remainder = limit % len(patterns)
    selected: list[tuple[int, dict[str, Any]]] = []
    for pos, pattern in enumerate(patterns):
        bucket = by_pattern[pattern][:]
        rng.shuffle(bucket)
        take = base + (1 if pos < remainder else 0)
        selected.extend(bucket[:take])

    rng.shuffle(selected)
    return selected[:limit]


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_gradio_event(text: str) -> list[Any]:
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    raise ValueError(f"no data line in Gradio response: {text[:200]}")


def parse_model_json(raw: str) -> tuple[str, str | None, bool]:
    text = (raw or "").strip()
    if not text:
        return "", None, False
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return text, None, False
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return text, None, False
    if not isinstance(obj, dict):
        return text, None, False
    rewrite = str(obj.get("rewrite_message", "")).strip()
    domain = obj.get("domain")
    if domain is not None:
        domain = str(domain).strip().lower()
    return rewrite, domain, bool(rewrite)


def call_with_retry(fn, max_retries: int) -> dict[str, Any]:
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 6) + random.random() * 0.25)
    return {"error": str(last_error)}


def call_gradio(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()

    def once() -> dict[str, Any]:
        payload = {"data": [case["final_user"], case["history_text"], ""]}
        first = http_json(
            f"{args.gradio_url.rstrip('/')}/gradio_api/call/rewrite",
            payload,
            timeout=args.timeout,
        )
        event_id = first["event_id"]
        data = parse_gradio_event(
            http_text(
                f"{args.gradio_url.rstrip('/')}/gradio_api/call/rewrite/{event_id}",
                timeout=args.timeout,
            )
        )
        display = data[0] if len(data) > 0 else ""
        raw = data[1] if len(data) > 1 else display
        rewrite, domain, valid = parse_model_json(raw)
        return {
            "raw": raw,
            "display": display,
            "rewrite": rewrite,
            "domain": domain,
            "output_valid": valid,
        }

    result = call_with_retry(once, args.max_retries)
    result["latency_s"] = round(time.perf_counter() - start, 3)
    return result


def call_vllm(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()

    def once() -> dict[str, Any]:
        payload = {
            "model": args.vllm_model,
            "messages": case["messages"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        data = http_json(
            f"{args.vllm_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout=args.timeout,
        )
        raw = data["choices"][0]["message"]["content"].strip()
        rewrite, domain, valid = parse_model_json(raw)
        return {
            "raw": raw,
            "rewrite": rewrite,
            "domain": domain,
            "output_valid": valid,
            "usage": data.get("usage"),
        }

    result = call_with_retry(once, args.max_retries)
    result["latency_s"] = round(time.perf_counter() - start, 3)
    return result


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def token_scores(pred: str, gold: str) -> dict[str, float]:
    pred_tokens = tokens(pred)
    gold_tokens = tokens(gold)
    if not pred_tokens and not gold_tokens:
        return {"token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": fbeta(precision, recall, 1.0),
    }


def restoration_scores(pred: str, gold: str, final_user: str) -> dict[str, float | None]:
    last = set(tokens(final_user))
    target = set(tokens(gold)) - last
    if not target:
        return {"restoration_f1": None, "restoration_f2": None}
    pred_extra = set(tokens(pred)) - last
    hits = len(pred_extra & target)
    precision = hits / len(pred_extra) if pred_extra else 0.0
    recall = hits / len(target)
    return {
        "restoration_f1": fbeta(precision, recall, 1.0),
        "restoration_f2": fbeta(precision, recall, 2.0),
    }


def score_prediction(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    rewrite = pred.get("rewrite", "")
    domain = pred.get("domain")
    gold_rewrite = case["gold_rewrite"]
    gold_domain = case["gold_domain"]
    scores = {
        "exact_norm": int(normalize_text(rewrite) == normalize_text(gold_rewrite)),
        "domain_correct": int(domain == gold_domain) if gold_domain else None,
        "length_ratio": len(rewrite) / max(len(gold_rewrite), 1),
    }
    scores.update(token_scores(rewrite, gold_rewrite))
    scores.update(restoration_scores(rewrite, gold_rewrite, case["final_user"]))
    return scores


def run_endpoint(
    name: str,
    cases: list[dict[str, Any]],
    call_fn,
    args: argparse.Namespace,
    out_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_i = {executor.submit(call_fn, case, args): i for i, case in enumerate(cases)}
        for fut in futures.as_completed(future_to_i):
            i = future_to_i[fut]
            case = cases[i]
            pred = fut.result()
            scores = score_prediction(case, pred) if not pred.get("error") else {}
            rows[i] = {
                "case_id": case["case_id"],
                "meta": case["meta"],
                "final_user": case["final_user"],
                "gold_rewrite": case["gold_rewrite"],
                "gold_domain": case["gold_domain"],
                "prediction": pred,
                "scores": scores,
            }
            done += 1
            if done % 10 == 0 or done == len(cases):
                print(f"{name}: {done}/{len(cases)}")

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def mean(values: list[Any]) -> float | None:
    xs = [x for x in values if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    errors = sum(1 for r in rows if r["prediction"].get("error"))
    latencies = [r["prediction"].get("latency_s") for r in rows if r["prediction"].get("latency_s") is not None]
    scores = [r["scores"] for r in rows if r.get("scores")]
    by_pattern = defaultdict(list)
    by_domain = defaultdict(list)
    for row in rows:
        pattern = row["meta"].get("pattern", "unknown")
        domain = row["gold_domain"] or row["meta"].get("domain", "unknown")
        by_pattern[pattern].append(row)
        by_domain[domain].append(row)

    def avg_score(key: str, subset: list[dict[str, Any]]) -> float | None:
        return mean([r.get("scores", {}).get(key) for r in subset])

    return {
        "n": n,
        "errors": errors,
        "output_valid": mean([int(r["prediction"].get("output_valid", False)) for r in rows]),
        "exact_norm": mean([s.get("exact_norm") for s in scores]),
        "domain_correct": mean([s.get("domain_correct") for s in scores]),
        "token_f1": mean([s.get("token_f1") for s in scores]),
        "token_recall": mean([s.get("token_recall") for s in scores]),
        "restoration_f2": mean([s.get("restoration_f2") for s in scores]),
        "latency_avg_s": mean(latencies),
        "latency_p95_s": sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1] if latencies else None,
        "by_pattern": {
            k: {
                "n": len(v),
                "exact_norm": avg_score("exact_norm", v),
                "domain_correct": avg_score("domain_correct", v),
                "token_f1": avg_score("token_f1", v),
                "restoration_f2": avg_score("restoration_f2", v),
                "errors": sum(1 for r in v if r["prediction"].get("error")),
            }
            for k, v in sorted(by_pattern.items())
        },
        "by_gold_domain": {
            k: {
                "n": len(v),
                "domain_correct": avg_score("domain_correct", v),
                "token_f1": avg_score("token_f1", v),
                "errors": sum(1 for r in v if r["prediction"].get("error")),
            }
            for k, v in sorted(by_domain.items())
        },
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    def pct(x: Any) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lines = [
        "# 100-case two-endpoint benchmark",
        "",
        "Metrics are deterministic string/token checks against the benchmark gold.",
        "They are useful for smoke/regression comparison, but not a semantic LLM judge.",
        "",
        "## Summary",
        "",
        "| endpoint | n | errors | valid | domain acc | exact norm | token F1 | restoration F2 | avg latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary["endpoints"].items():
        lines.append(
            "| {name} | {n} | {errors} | {valid} | {domain} | {exact} | {f1} | {rf2} | {avg:.2f}s | {p95:.2f}s |".format(
                name=name,
                n=data["n"],
                errors=data["errors"],
                valid=pct(data["output_valid"]),
                domain=pct(data["domain_correct"]),
                exact=pct(data["exact_norm"]),
                f1=pct(data["token_f1"]),
                rf2=pct(data["restoration_f2"]),
                avg=data["latency_avg_s"] or 0.0,
                p95=data["latency_p95_s"] or 0.0,
            )
        )
    lines.extend(["", "## Pattern Mix", ""])
    pattern_counts = summary["case_mix"]["patterns"]
    for pattern, count in pattern_counts.items():
        lines.append(f"- {pattern}: {count}")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.bench.open(encoding="utf-8") if line.strip()]
    selected = select_stratified(records, args.limit, args.seed)
    cases = [record_to_case(record, index) for index, record in selected]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("outputs/eval_results") / f"two_endpoints_100_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "selected_cases.jsonl").open("w", encoding="utf-8") as fh:
        for case in cases:
            export = {k: v for k, v in case.items() if k != "messages"}
            fh.write(json.dumps(export, ensure_ascii=False) + "\n")

    case_mix = {
        "patterns": dict(Counter(case["meta"].get("pattern", "unknown") for case in cases)),
        "task_domains": dict(Counter(case["meta"].get("domain", "unknown") for case in cases)),
        "online_offline": dict(Counter(case["gold_domain"] for case in cases)),
    }
    print(f"Selected {len(cases)} cases from {args.bench}")
    print("Pattern mix:", json.dumps(case_mix["patterns"], ensure_ascii=False, sort_keys=True))
    print("Output dir:", out_dir)

    gradio_rows = run_endpoint(
        "gradio_helpful",
        cases,
        call_gradio,
        args,
        out_dir / "preds_gradio_helpful.jsonl",
    )
    vllm_rows = run_endpoint(
        f"vllm_{args.vllm_model}",
        cases,
        call_vllm,
        args,
        out_dir / f"preds_vllm_{args.vllm_model.replace('/', '_')}.jsonl",
    )

    summary = {
        "created_at": timestamp,
        "bench": str(args.bench),
        "case_mix": case_mix,
        "endpoints": {
            "gradio_helpful": summarize(gradio_rows),
            f"vllm_{args.vllm_model}": summarize(vllm_rows),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir, summary)

    print("\nSummary:")
    for name, data in summary["endpoints"].items():
        print(
            f"{name}: errors={data['errors']}/{data['n']} "
            f"valid={data['output_valid']:.3f} "
            f"domain={data['domain_correct']:.3f} "
            f"exact={data['exact_norm']:.3f} "
            f"token_f1={data['token_f1']:.3f} "
            f"restoration_f2={data['restoration_f2']:.3f}"
        )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
