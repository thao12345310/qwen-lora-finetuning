"""Evaluate the current vLLM-served model on the 300-sample benchmark.

The bench uses the NEW Llama Factory format (<REWRITE> tag, JSON output). The
existing LoRA adapter was trained on the OLD plain-text format. So this script
RE-FORMATS each bench sample to match what the adapter expects, and judges the
plain-text prediction semantically against the gold rewrite via GPT-4o.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m src.eval.eval_bench \
        --vllm-url https://overtime-freely-glider.ngrok-free.dev \
        --models vi-rewriter Qwen/Qwen2.5-1.5B-Instruct \
        --judge-model gpt-4o
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()

OLD_SYSTEM_PROMPT = (
    "Bạn là model rewrite hội thoại. Nhiệm vụ của bạn là biến câu nói cuối của user "
    "thành một yêu cầu độc lập, rõ ràng, giữ nguyên ý định, không thêm thông tin "
    "không chắc chắn. Chỉ trả về câu rewrite."
)

JUDGE_SYSTEM = """Bạn là giám khảo đánh giá câu rewrite tiếng Việt cho task hội thoại trợ lý xe.

Bạn nhận: dialogue (lịch sử hội thoại), gold (câu rewrite chuẩn), prediction (câu model dự đoán).

Tiêu chí ĐÚNG (score=1) — prediction phải:
- Bảo toàn TẤT CẢ slot quan trọng có trong gold (tên người, địa điểm, số, nhiệt độ, bài hát, hãng, chế độ, v.v.).
- Cùng intent với gold (bật/tắt/đổi/hủy/thêm).
- Có thể khác cách diễn đạt, opener, dấu câu — KHÔNG sao.
- KHÔNG được thêm slot hoặc thông tin sai (vd kéo nhầm số bot mention).
- KHÔNG được bỏ slot.

Sai (score=0): thiếu slot, sai intent, thêm thông tin bịa.

Trả về JSON: {"score": 0 hoặc 1, "reason": "1 câu ngắn"}."""


def lf_to_old_turns(record: dict) -> tuple[list[dict], str]:
    """Convert LF record → (turns list in old format, gold rewrite plain text).

    Roles: 'human' → 'user', 'gpt' → 'bot' (except final gpt which is the answer).
    Final human's value strips the leading '<REWRITE>\\n' tag.
    """
    convs = record["conversations"]
    # Drop system message; the adapter has its own injected system.
    body = [c for c in convs if c["from"] != "system"]
    # Last gpt is the gold answer
    assert body[-1]["from"] == "gpt"
    gold = json.loads(body[-1]["value"])["rewrite_message"]
    # Everything before is dialogue history
    dialogue = body[:-1]
    turns = []
    for c in dialogue:
        role = "user" if c["from"] == "human" else "bot"
        content = c["value"]
        if content.startswith("<REWRITE>\n"):
            content = content[len("<REWRITE>\n"):]
        turns.append({"role": role, "content": content})
    return turns, gold


def format_old_user_msg(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


async def predict(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    sem: asyncio.Semaphore,
    max_tokens: int = 96,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"__ERROR__ {e}"
                await asyncio.sleep(2 ** attempt)
        return "__ERROR__ unreachable"


async def judge(
    client: AsyncOpenAI,
    model: str,
    dialogue: str,
    gold: str,
    pred: str,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    user = (
        f"dialogue:\n{dialogue}\n\n"
        f"gold:\n{gold}\n\n"
        f"prediction:\n{pred}\n\n"
        f"Đánh giá theo schema."
    )
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=200,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"score": 0, "reason": f"__JUDGE_ERROR__ {e}"}
                await asyncio.sleep(2 ** attempt)
        return {"score": 0, "reason": "__JUDGE_ERROR__ unreachable"}


async def _gather_with_progress(coros: list, desc: str):
    """asyncio.gather with a tqdm bar that ticks as each task completes."""
    pbar = tqdm(total=len(coros), desc=desc)
    results = [None] * len(coros)

    async def wrapper(i, coro):
        results[i] = await coro
        pbar.update(1)

    await asyncio.gather(*(wrapper(i, c) for i, c in enumerate(coros)))
    pbar.close()
    return results


async def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    records = [json.loads(l) for l in args.bench.open(encoding="utf-8")]
    print(f"Loaded {len(records)} bench samples from {args.bench}")

    vllm_client = AsyncOpenAI(base_url=args.vllm_url.rstrip("/") + "/v1", api_key="EMPTY")
    judge_client = AsyncOpenAI(api_key=api_key)
    pred_sem = asyncio.Semaphore(args.vllm_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for model in args.models:
        print(f"\n=== Predicting with {model} ===")
        tasks = []
        contexts = []
        for r in records:
            turns, gold = lf_to_old_turns(r)
            user_msg = format_old_user_msg(turns)
            contexts.append({"turns": turns, "user_msg": user_msg, "gold": gold, "meta": r["meta"]})
            tasks.append(predict(vllm_client, model, OLD_SYSTEM_PROMPT, user_msg, pred_sem))

        preds = await _gather_with_progress(tasks, desc=f"predict {model}")

        print(f"  Judging {len(preds)} predictions with {args.judge_model}…")
        judge_tasks = [
            judge(judge_client, args.judge_model, ctx["user_msg"], ctx["gold"], pred, judge_sem)
            for ctx, pred in zip(contexts, preds)
        ]
        scores = await _gather_with_progress(judge_tasks, desc=f"judge {model}")

        # Persist raw
        slug = model.replace("/", "_")
        out_path = args.output_dir / f"preds_{slug}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ctx, pred, sc in zip(contexts, preds, scores):
                f.write(json.dumps({
                    "meta": ctx["meta"],
                    "gold": ctx["gold"],
                    "pred": pred,
                    "score": sc.get("score", 0),
                    "reason": sc.get("reason", ""),
                }, ensure_ascii=False) + "\n")
        print(f"  Saved {out_path}")

        all_results[model] = [
            {"meta": ctx["meta"], "gold": ctx["gold"], "pred": pred, **sc}
            for ctx, pred, sc in zip(contexts, preds, scores)
        ]

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for model, rows in all_results.items():
        overall = sum(r["score"] for r in rows) / len(rows)
        print(f"\n{model}  →  overall {overall*100:.1f}% ({sum(r['score'] for r in rows)}/{len(rows)})")
        by_pat: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_pat[r["meta"]["pattern"]].append(r["score"])
        for p in sorted(by_pat):
            scs = by_pat[p]
            print(f"  {p:24s}  {sum(scs)}/{len(scs)}  ({sum(scs)/len(scs)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("data/bench/dialogues_bench.jsonl"))
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--models", nargs="+", default=["vi-rewriter", "Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench/eval_results"))
    parser.add_argument("--vllm-concurrency", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
