"""Smoke test: re-judge a small sample of existing preds with a chosen judge model
(default: Claude Haiku via Anthropic's OpenAI-compat endpoint) using the current
JUDGE_SYSTEM prompt. Prints OLD (stored) vs NEW flags so you can eyeball the diff
WITHOUT touching the saved preds file or re-running vLLM.

Usage:
    export JUDGE_API_KEY=$ANTHROPIC_API_KEY   # or it falls back to ANTHROPIC_API_KEY
    /opt/homebrew/bin/python3.11 -m src.eval._rejudge_smoke
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.eval.eval_bench import (
    JUDGE_FLAGS,
    format_old_user_msg,
    judge,
    lf_to_old_turns,
)

load_dotenv()

PREDS = Path("data/bench/eval_results/pilot500/preds_vi-rewriter.jsonl")
BENCH = Path("data/bench/dialogues_bench_browser.jsonl")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.anthropic.com/v1/")
N = int(os.environ.get("N", "12"))


def pick_indices(prev: list[dict], n: int) -> list[int]:
    """A mix: prior score==0 / judge-errored (the interesting cases) first, then some
    score==1 to confirm the prompt doesn't start failing easy passes."""
    errored = [i for i, r in enumerate(prev) if str(r.get("reason", "")).startswith("__JUDGE_ERROR__")]
    zeros = [i for i, r in enumerate(prev) if r.get("score") == 0 and i not in errored]
    ones = [i for i, r in enumerate(prev) if r.get("score") == 1]
    out, seen = [], set()
    for bucket in (zeros, errored, ones):
        for i in bucket:
            if i not in seen:
                out.append(i); seen.add(i)
            if len(out) >= n:
                return sorted(out)
    return sorted(out)


async def main():
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set JUDGE_API_KEY or ANTHROPIC_API_KEY")

    prev = [json.loads(l) for l in PREDS.open(encoding="utf-8")]
    records = [json.loads(l) for l in BENCH.open(encoding="utf-8")][: len(prev)]
    assert len(prev) == len(records), f"{len(prev)} preds vs {len(records)} bench rows — misaligned"

    idx = pick_indices(prev, N)
    client = AsyncOpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    sem = asyncio.Semaphore(4)

    print(f"Judge = {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print(f"Re-judging {len(idx)} rows from {PREDS.name}\n")

    async def one(i):
        turns, _ = lf_to_old_turns(records[i])
        dialogue = format_old_user_msg(turns)
        new = await judge(client, JUDGE_MODEL, dialogue, prev[i]["gold"], prev[i]["pred"], sem)
        return i, new

    results = await asyncio.gather(*(one(i) for i in idx))

    flips = 0
    for i, new in sorted(results):
        old = prev[i]
        of = [old.get(k) for k in JUDGE_FLAGS]
        nf = [new.get(k) for k in JUDGE_FLAGS]
        changed = old.get("score") != new.get("score")
        flips += changed
        mark = "  <-- SCORE FLIP" if changed else ""
        print(f"[{i}] {old['meta'].get('pattern')}")
        print(f"    gold: {old['gold']}")
        print(f"    pred: {old.get('pred_text') or old.get('pred')}")
        print(f"    OLD flags={of} score={old.get('score')}")
        print(f"    NEW flags={nf} score={new.get('score')}{mark}")
        print(f"    NEW reason: {new.get('reason')}")
        print()

    print(f"Score flips: {flips}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
