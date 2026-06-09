"""Semantic judge pass over the generated LF dataset via the MiMo API (temp 0).

gen_data.md requires cross-evaluation with a SEPARATE low-temperature judge prompt:
the structural validator in generate_api.py only checks turns/numbers/format, not
whether the rewrite resolves the right referent, preserves negation/compound, or
avoids repeating an already-done action. This pass scores each sample 0/1 with the
strict JUDGE_SYSTEM prompt and writes only the score==1 samples.

Resumable: scores are cached by sample key, so a killed run continues. Concurrent
with retry/backoff + key rotation, same as generate_api.py.

    MIMO_API_KEYS="k1,k2" python src/data/judge_api.py \
        --in data/train/dialogues_train_api.jsonl \
        --out data/train/dialogues_train_api.judged.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vllm import JUDGE_SYSTEM, parse_score  # noqa: E402
from openai import OpenAI  # noqa: E402

REWRITE_PREFIX = "<REWRITE_AND_CLASSIFY>\n"


def lf_to_judge_inputs(rec):
    """Reconstruct (dialogue_str, context_required, prediction) from an LF record."""
    convs = rec["conversations"]
    body = [c for c in convs if c["from"] != "system"]
    pred = json.loads(body[-1]["value"]).get("rewrite_message", "")
    turns = []
    for c in body[:-1]:
        role = "user" if c["from"] == "human" else "bot"
        v = c["value"]
        if v.startswith(REWRITE_PREFIX):
            v = v[len(REWRITE_PREFIX):]
        turns.append(f"{role}: {v}")
    ctx = rec.get("meta", {}).get("context_required")
    return "\n".join(turns), ctx, pred


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"))
    p.add_argument("--base-url", default=os.environ.get(
        "MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    p.add_argument("--in", dest="inp", default="data/train/dialogues_train_api.jsonl")
    p.add_argument("--out", default="data/train/dialogues_train_api.judged.jsonl")
    p.add_argument("--score-cache", default=None,
                   help="resume cache of scores (default: <out>.scores.jsonl)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--judge-max-tokens", type=int, default=192)
    return p.parse_args()


def main():
    args = parse_args()
    keys = [k.strip() for k in os.environ.get("MIMO_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        sys.exit("set MIMO_API_KEYS='key1,key2'")
    clients = [OpenAI(api_key=k, base_url=args.base_url) for k in keys]
    score_cache = args.score_cache or args.out + ".scores.jsonl"

    records = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"{len(records)} records | {len(clients)} key(s) | cache {score_cache}")

    def key_of(rec):
        d, ctx, pred = lf_to_judge_inputs(rec)
        return d.split("\n")[-1] + "||" + pred  # last user turn + prediction

    scores = {}
    if os.path.exists(score_cache):
        for l in open(score_cache, encoding="utf-8"):
            l = l.strip()
            if l:
                o = json.loads(l)
                scores[o["k"]] = o["s"]
        print(f"resumed {len(scores)} cached scores")

    todo = [r for r in records if key_of(r) not in scores]
    print(f"to judge: {len(todo)}")

    rr = {"i": 0}
    rr_lock = threading.Lock()
    cache_fh = open(score_cache, "a", encoding="utf-8")
    write_lock = threading.Lock()

    def judge_one(rec):
        d, ctx, pred = lf_to_judge_inputs(rec)
        user = (f"dialogue:\n{d}\n\ncontext_required: {str(ctx).lower()}\n\n"
                f"prediction:\n{pred}\n\nChấm theo schema.")
        for attempt in range(args.max_retries):
            with rr_lock:
                c = clients[rr["i"] % len(clients)]
                rr["i"] += 1
            try:
                r = c.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=args.judge_max_tokens,
                    response_format={"type": "json_object"},
                )
                return parse_score(r.choices[0].message.content)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(min(1.5 ** attempt + random.random(), 20))
                else:
                    time.sleep(1 + random.random())
        return None  # unjudged -> treated as keep (fail-open) at finalize

    t0 = ok = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, r): r for r in todo}
        done = 0
        for fut in as_completed(futs):
            rec = futs[fut]
            s = fut.result()
            done += 1
            if s is None:
                err += 1
                continue
            ok += 1
            k = key_of(rec)
            scores[k] = s
            with write_lock:
                cache_fh.write(json.dumps({"k": k, "s": s}, ensure_ascii=False) + "\n")
                cache_fh.flush()
            if done % 500 == 0:
                kept = sum(scores.values())
                print(f"  judged {done}/{len(todo)} | so far kept {kept} | err {err} "
                      f"| {done/max(time.time()-t0,1):.1f}/s")
    cache_fh.close()

    # finalize: keep score==1 (and fail-open unjudged so the run is never blocked)
    kept = [r for r in records if scores.get(key_of(r), 1) == 1]
    judged = sum(1 for r in records if key_of(r) in scores)
    rejected = judged - sum(scores.get(key_of(r), 0) for r in records if key_of(r) in scores)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"judged {judged}/{len(records)} | rejected {rejected} | "
          f"kept {len(kept)} → {args.out}")


if __name__ == "__main__":
    main()
