"""Evaluate the current vLLM-served model on the benchmark.

The bench AND the current adapter both use the Llama-Factory format (system =
SYSTEM_PROMPT_FOR_TRAINING, multi-turn messages, final user turn tagged <REWRITE>,
assistant answer = JSON {"rewrite_message": ...}). So each bench record is sent to
the model AS-IS (its own conversations[:-1] become the chat messages) — identical
to what the model saw at training — and the JSON prediction is judged semantically
against the gold rewrite.

Usage (judge can be MiMo via --judge-base-url, or GPT-4o by default):
    export JUDGE_API_KEY=...        # or OPENAI_API_KEY for the GPT-4o judge
    python -m src.eval.eval_bench \
        --vllm-url https://overtime-freely-glider.ngrok-free.dev \
        --models vi-rewriter Qwen/Qwen2.5-1.5B-Instruct \
        --judge-model mimo-v2.5-pro \
        --judge-base-url https://token-plan-sgp.xiaomimimo.com/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from src.data.prompts import BASELINE_SYSTEM_PROMPT

load_dotenv()

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}

JUDGE_SYSTEM = """Bạn là giám khảo đánh giá câu rewrite tiếng Việt cho task hội thoại trợ lý xe.

Bạn nhận: dialogue (lịch sử hội thoại), gold (câu rewrite chuẩn), prediction (câu model dự đoán).
Prediction có thể khác gold về cách diễn đạt, opener, dấu câu — điều đó KHÔNG sao. Chỉ đánh giá Ý NGHĨA.

Chấm 4 cờ, mỗi cờ 0 hoặc 1:
- "intent_ok": cùng hành động với gold (bật/tắt/đổi/hủy/thêm/gọi/gửi/dẫn đường…). 1 nếu đúng intent.
- "slots_complete": KHÔNG bỏ sót slot quan trọng có trong gold (tên người, địa điểm, số, nhiệt độ, bài hát, hãng, chế độ, kênh, tần số…). 1 nếu đủ.
- "no_hallucination": KHÔNG thêm slot/thông tin sai hoặc không được xác nhận (vd kéo nhầm số bot mention, tự thêm ràng buộc). 1 nếu sạch.
- "negation_ok": giữ đúng phủ định/khẳng định của gold (đừng đảo "không bật" thành "bật"). 1 nếu đúng; nếu gold không có phủ định thì để 1.

"score" = 1 chỉ khi CẢ 4 cờ đều = 1; ngược lại 0.

Trả về DUY NHẤT JSON:
{"intent_ok":0|1,"slots_complete":0|1,"no_hallucination":0|1,"negation_ok":0|1,"score":0|1,"reason":"1 câu ngắn"}"""

JUDGE_FLAGS = ("intent_ok", "slots_complete", "no_hallucination", "negation_ok")


def _normalize_judge(d: dict) -> dict:
    """Coerce raw judge JSON into the canonical flag schema, filling gaps safely."""
    score = d.get("score")
    default = int(bool(score)) if score is not None else 1
    out = {k: int(bool(d.get(k, default))) for k in JUDGE_FLAGS}
    out["score"] = int(bool(score)) if score is not None else int(all(out[k] for k in JUDGE_FLAGS))
    out["reason"] = d.get("reason", "")
    return out


def _judge_error(msg: str) -> dict:
    return {k: 0 for k in JUDGE_FLAGS} | {"score": 0, "reason": msg}


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


def lf_to_messages(record: dict) -> tuple[list[dict], str]:
    """Convert an LF record → (chat messages to send the model, gold rewrite).

    messages = conversations[:-1] (system + the multi-turn history, final user turn
    already carries the <REWRITE> tag) mapped to role/content — EXACTLY the training
    input. The dropped last `gpt` turn is the gold answer.
    """
    convs = record["conversations"]
    assert convs[-1]["from"] == "gpt"
    gold = json.loads(convs[-1]["value"])["rewrite_message"]
    messages = [{"role": ROLE_MAP[c["from"]], "content": c["value"]} for c in convs[:-1]]
    return messages, gold


# ---- Baseline (untrained) prompting: detailed system + few-shot --------------
# The trained adapter is sent the record AS-IS (lf_to_messages). An untrained base
# model instead gets BASELINE_SYSTEM_PROMPT plus few-shot demos drawn from the
# frontier-generated train data, so the comparison reflects capability, not a
# prompt handicap.

def lf_to_baseline_query(record: dict) -> tuple[str, str]:
    """Render an LF record as (query_text, gold_json_str) for baseline prompting.

    query_text = the dialogue history + final <REWRITE> turn as labeled lines.
    gold_json_str = the assistant's JSON answer (used as the few-shot target).
    """
    convs = [c for c in record["conversations"] if c["from"] != "system"]
    answer = convs[-1]["value"]
    lines = []
    for c in convs[:-1]:
        label = "Người dùng" if c["from"] == "human" else "Trợ lý"
        lines.append(f"{label}: {c['value']}")
    return "\n".join(lines), answer


def _final_user_text(record: dict) -> str:
    """Final human turn of an LF record, tag-stripped + normalized (for dedup)."""
    for c in reversed(record["conversations"]):
        if c["from"] == "human":
            txt = c["value"]
            if txt.startswith("<REWRITE>\n"):
                txt = txt[len("<REWRITE>\n"):]
            return normalize_text(txt)
    return ""


def load_fewshot(path: Path, n: int, bench_records: list[dict], seed: int = 13) -> list[dict]:
    """Sample n few-shot LF records from `path`, excluding any whose final user turn
    matches a bench sample (leakage guard). Deterministic given seed."""
    if n <= 0:
        return []
    pool = [json.loads(l) for l in path.open(encoding="utf-8")]
    bench_finals = {_final_user_text(r) for r in bench_records}
    pool = [r for r in pool if _final_user_text(r) not in bench_finals]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def build_baseline_messages(record: dict, fewshot: list[dict]) -> list[dict]:
    """system (detailed) + few-shot (user query → assistant JSON) + the real query."""
    messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]
    for fr in fewshot:
        q, a = lf_to_baseline_query(fr)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    q, _ = lf_to_baseline_query(record)
    messages.append({"role": "user", "content": q})
    return messages


def format_old_user_msg(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def last_user_utterance(turns: list[dict]) -> str:
    """The final user turn (the <REWRITE> target, already tag-stripped by lf_to_old_turns)."""
    for t in reversed(turns):
        if t["role"] == "user":
            return t["content"]
    return ""


# ---- Tier-0: deterministic, gold-free checks + Restoration-F ------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(s: str) -> str:
    """Lowercase, fold degree marks, strip punctuation, collapse whitespace."""
    s = s.lower().replace("°c", " độ ").replace("°", " độ ")
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


def tokenize(s: str) -> list[str]:
    return normalize_text(s).split()


def extract_pred_text(raw: str) -> tuple[str, bool]:
    """Return (clean_text, output_valid). Handles JSON ({"rewrite_message":...}) or plain text."""
    if not raw or raw.startswith("__ERROR__"):
        return "", False
    txt = raw.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "rewrite_message" in obj:
            txt = str(obj["rewrite_message"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return txt, bool(txt)


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def restoration_scores(pred_text: str, gold: str, last_user: str):
    """Token-set restoration F1/F2 over the info that had to be recovered from context.

    target = tokens(gold) - tokens(last_user)  → what the rewrite added beyond the last turn.
    Returns (f1, f2), or (None, None) when there is nothing to restore (target empty),
    so context-free samples don't dilute the score.
    """
    last = set(tokenize(last_user))
    target = set(tokenize(gold)) - last
    if not target:
        return None, None
    pred_extra = set(tokenize(pred_text)) - last
    hit = len(pred_extra & target)
    precision = hit / len(pred_extra) if pred_extra else 0.0
    recall = hit / len(target)
    return _fbeta(precision, recall, 1.0), _fbeta(precision, recall, 2.0)


def deterministic_metrics(raw_pred: str, gold: str, last_user: str) -> dict:
    pred_text, valid = extract_pred_text(raw_pred)
    is_copy = bool(pred_text) and normalize_text(pred_text) == normalize_text(last_user)
    ratio = len(pred_text) / max(len(gold), 1)
    is_degenerate = (not pred_text) or ratio < 0.3 or ratio > 3.0
    f1, f2 = restoration_scores(pred_text, gold, last_user)
    return {
        "pred_text": pred_text,
        "output_valid": int(valid),
        "is_copy": int(is_copy),
        "is_degenerate": int(is_degenerate),
        "restoration_f1": f1,
        "restoration_f2": f2,
    }


async def predict(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    sem: asyncio.Semaphore,
    max_tokens: int = 160,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
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
    max_retries: int = 6,
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
                return _normalize_judge(json.loads(resp.choices[0].message.content))
            except Exception as e:
                if attempt == max_retries - 1:
                    return _judge_error(f"__JUDGE_ERROR__ {e}")
                # Backoff + jitter để vượt rate-limit (429). Trần 5s để 429 không ngốn thời gian.
                await asyncio.sleep(min(2 ** attempt, 5) + random.uniform(0, 1))
        return _judge_error("__JUDGE_ERROR__ unreachable")


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
    # Judge can run on any OpenAI-compatible endpoint (OpenAI, MiMo, …).
    # JUDGE_API_KEY takes priority; fall back to OPENAI_API_KEY for the default OpenAI judge.
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set JUDGE_API_KEY (or OPENAI_API_KEY) for the judge")

    records = [json.loads(l) for l in args.bench.open(encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} bench samples from {args.bench}")

    baseline_set = set(args.baseline_models or [])
    fewshot = []
    if baseline_set:
        sampled = load_fewshot(args.baseline_fewshot_file, args.baseline_fewshot_n, records)
        hard = []
        if args.baseline_fewshot_hard_file and args.baseline_fewshot_hard_file.exists():
            # All curated hard cases (no sampling cap); placed AFTER the random ones so
            # the trickiest patterns sit closest to the query (recency).
            hard = load_fewshot(args.baseline_fewshot_hard_file, 10**9, records)
        fewshot = sampled + hard
        print(
            f"Baseline models {sorted(baseline_set)} → detailed prompt + "
            f"{len(sampled)} sampled ({args.baseline_fewshot_file}) + "
            f"{len(hard)} hard ({args.baseline_fewshot_hard_file}) = {len(fewshot)} few-shot"
        )

    vllm_client = AsyncOpenAI(base_url=args.vllm_url.rstrip("/") + "/v1", api_key="EMPTY")
    judge_client = AsyncOpenAI(api_key=api_key, base_url=args.judge_base_url)
    pred_sem = asyncio.Semaphore(args.vllm_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    for model in args.models:
        is_baseline = model in baseline_set
        tag = "  (baseline: detailed prompt + few-shot)" if is_baseline else ""
        print(f"\n=== Predicting with {model} ==={tag}")
        tasks = []
        contexts = []
        for r in records:
            _, gold = lf_to_messages(r)                  # gold is format-independent
            if is_baseline:
                messages = build_baseline_messages(r, fewshot)
            else:
                messages, _ = lf_to_messages(r)          # NEW-format input (matches training)
            turns, _ = lf_to_old_turns(r)               # readable turns for judge + metrics
            dialogue = format_old_user_msg(turns)
            contexts.append({"turns": turns, "dialogue": dialogue, "gold": gold, "meta": r["meta"]})
            tasks.append(predict(vllm_client, model, messages, pred_sem))

        preds = await _gather_with_progress(tasks, desc=f"predict {model}")

        print(f"  Judging {len(preds)} predictions with {args.judge_model}…")
        judge_tasks = [
            judge(judge_client, args.judge_model, ctx["dialogue"], ctx["gold"], pred, judge_sem)
            for ctx, pred in zip(contexts, preds)
        ]
        scores = await _gather_with_progress(judge_tasks, desc=f"judge {model}")

        # Build rows: deterministic (Tier-0) + judge (Tier-1) metrics per sample.
        rows = []
        for ctx, pred, sc in zip(contexts, preds, scores):
            det = deterministic_metrics(pred, ctx["gold"], last_user_utterance(ctx["turns"]))
            rows.append({"meta": ctx["meta"], "gold": ctx["gold"], "pred": pred, **det, **sc})

        # Persist raw
        slug = model.replace("/", "_")
        out_path = args.output_dir / f"preds_{slug}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {out_path}")

        all_results[model] = rows

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = {model: print_report(model, rows) for model, rows in all_results.items()}

    # Machine-readable summary for downstream comparison/plots.
    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics summary → {summary_path}")


def _mean(xs: list) -> float:
    """Mean over non-None values; NaN if empty (e.g. restoration on all context-free)."""
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _is_judge_error(row: dict) -> bool:
    return str(row.get("reason", "")).startswith("__JUDGE_ERROR__")


def aggregate(rows: list[dict]) -> dict:
    """All metrics for one model. Deterministic metrics use every row; judge-based
    metrics use only successfully-judged rows (judge errors reported separately)."""
    judged = [r for r in rows if not _is_judge_error(r)]
    neg = [r for r in judged if r["meta"].get("pattern") == "negation"]
    return {
        "n": len(rows),
        "judge_error_rate": _mean([_is_judge_error(r) for r in rows]),
        # Tier-0
        "output_validity": _mean([r["output_valid"] for r in rows]),
        "copy_rate": _mean([r["is_copy"] for r in rows]),
        "degenerate_rate": _mean([r["is_degenerate"] for r in rows]),
        # Tier-1 (judge)
        "command_acc": _mean([r["score"] for r in judged]),
        "intent_acc": _mean([r["intent_ok"] for r in judged]),
        "slot_completeness": _mean([r["slots_complete"] for r in judged]),
        "halluc_rate": _mean([1 - r["no_hallucination"] for r in judged]),
        "negation_preservation": _mean([r["negation_ok"] for r in neg]),
        # Tier-2 (reference-overlap, auto)
        "restoration_f1": _mean([r["restoration_f1"] for r in rows]),
        "restoration_f2": _mean([r["restoration_f2"] for r in rows]),
    }


def _slice(rows: list[dict], key: str) -> dict[str, dict]:
    """Command-acc + halluc-rate per value of meta[key], skipping judge errors and
    rows lacking the key (so optional fields like context_required degrade gracefully)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if _is_judge_error(r):
            continue
        v = r["meta"].get(key)
        if v is None:
            continue
        groups[str(v)].append(r)
    return {
        v: {
            "n": len(g),
            "command_acc": _mean([r["score"] for r in g]),
            "halluc_rate": _mean([1 - r["no_hallucination"] for r in g]),
        }
        for v, g in groups.items()
    }


def print_report(model: str, rows: list[dict]) -> dict:
    agg = aggregate(rows)
    slices = {k: _slice(rows, k) for k in ("context_required", "pattern", "domain")}

    print(f"\n{model}   (n={agg['n']})")
    print(f"  Output Validity    : {agg['output_validity']*100:5.1f}%")
    print(f"  Copy Rate          : {agg['copy_rate']*100:5.1f}%   (cao ở context_required=true là xấu)")
    print(f"  Degenerate Rate    : {agg['degenerate_rate']*100:5.1f}%")
    print(f"  Command Accuracy   : {agg['command_acc']*100:5.1f}%   ← headline")
    print(f"  Intent Accuracy    : {agg['intent_acc']*100:5.1f}%")
    print(f"  Slot Completeness  : {agg['slot_completeness']*100:5.1f}%")
    print(f"  Halluc. Slot Rate  : {agg['halluc_rate']*100:5.1f}%   ← càng thấp càng tốt")
    print(f"  Negation Preserv.  : {agg['negation_preservation']*100:5.1f}%  (trên nhóm negation)")
    print(f"  Restoration-F1/F2  : {agg['restoration_f1']:.3f} / {agg['restoration_f2']:.3f}")
    if agg["judge_error_rate"] > 0:
        print(f"  Judge error rate   : {agg['judge_error_rate']*100:5.1f}%  (loại khỏi metric ngữ nghĩa)")

    for key in ("context_required", "pattern", "domain"):
        sl = slices[key]
        if not sl:
            if key == "context_required":
                print("  [context_required: chưa có field trong bench meta — bỏ qua slice]")
            continue
        print(f"  ── Command Acc theo {key}:")
        for v in sorted(sl):
            s = sl[v]
            print(f"     {v:22s} {s['command_acc']*100:5.1f}%  (n={s['n']}, halluc {s['halluc_rate']*100:.1f}%)")

    return {"overall": agg, "slices": slices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=Path("data/bench/dialogues_bench_browser.jsonl"))
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--models", nargs="+", default=["vi-rewriter", "Qwen/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible base URL for the judge (e.g. MiMo: "
        "https://token-plan-sgp.xiaomimimo.com/v1). Default: OpenAI.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench/eval_results"))
    parser.add_argument("--vllm-concurrency", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chấm N mẫu đầu (smoke test).")
    parser.add_argument(
        "--baseline-models",
        nargs="*",
        default=[],
        help="Các model (trong --models) là baseline CHƯA train → nhận prompt chi tiết "
        "BASELINE_SYSTEM_PROMPT + few-shot, thay vì prompt train ngắn. "
        "VD: --baseline-models Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--baseline-fewshot-file",
        type=Path,
        default=Path("data/processed/train.jsonl"),
        help="File LF (frontier-gen) để lấy few-shot cho baseline.",
    )
    parser.add_argument(
        "--baseline-fewshot-n", type=int, default=20, help="Số ví dụ few-shot cho baseline."
    )
    parser.add_argument(
        "--baseline-fewshot-hard-file",
        type=Path,
        default=Path("data/seed/fewshot_hard.jsonl"),
        help="File LF chứa few-shot tuyển chọn cho case khó/hay sai (negation, "
        "tham chiếu ngầm, slot bắc cầu, chống hallucinate…). Luôn nạp TOÀN BỘ, "
        "đặt sau các ví dụ random.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
