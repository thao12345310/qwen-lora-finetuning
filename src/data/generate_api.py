"""Phase-2 bulk generation via an OpenAI-compatible API (MiMo) instead of local vLLM.

Same quota plan / validator / Llama-Factory output as generate_vllm.py, but:
  - engine = remote chat-completions API (concurrent ThreadPool + retry/backoff + key rotation),
  - NO self-judge (the hard validator alone gates quality),
  - generated `domain` is coerced back into the fixed enum so stratify stays correct.

Resumes from --cache, so re-running fills only buckets still short of quota — important
when the API key has a short expiry and gets rate-limited mid-run.

    MIMO_API_KEYS="tp-aaa,tp-bbb" python src/data/generate_api.py \
        --target-total 500 --out data/train/dialogues_train_api.jsonl --workers 6
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vllm import (  # noqa: E402
    GEN_SYSTEM, DOMAINS, DOMAIN_WEIGHTS, build_quota, select_capped,
    load_seed_pool, make_validator, parse_samples, to_lf, find_first,
)
from openai import OpenAI  # noqa: E402

# --- domain drift → enum -----------------------------------------------------
DOMAIN_ALIAS = {
    "media_control": "music", "media": "music", "audio": "music", "player": "music",
    "climate_control": "climate", "ac": "climate", "hvac": "climate", "air_conditioning": "climate",
    "vehicle_control": "vehicle", "car_control": "vehicle", "car": "vehicle",
    "phone": "calling", "call": "calling", "dialer": "calling",
    "sms": "messaging", "message": "messaging", "chat": "messaging",
    "smarthome": "smart_home", "home": "smart_home", "house": "smart_home",
    "nav": "navigation", "navi": "navigation", "maps": "navigation", "route": "navigation",
    "charge": "charging", "ev_charging": "charging", "charger": "charging",
    "adas": "driver_assist", "assist": "driver_assist", "autopilot": "driver_assist",
    "driver_assistance": "driver_assist",
}


def coerce_domain(dom, fallback):
    d = (dom or "").strip().lower().replace(" ", "_").replace("-", "_")
    if d in DOMAIN_WEIGHTS:
        return d
    return DOMAIN_ALIAS.get(d, fallback)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"))
    p.add_argument("--base-url", default=os.environ.get(
        "MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    p.add_argument("--target-total", type=int, default=500)
    p.add_argument("--context-fraction", type=float, default=2 / 3)
    p.add_argument("--gen-temperature", type=float, default=1.0)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--gen-max-tokens", type=int, default=3072)
    p.add_argument("--samples-per-call", type=int, default=6)
    p.add_argument("--long-samples-per-call", type=int, default=2)
    p.add_argument("--n-fewshot", type=int, default=3)
    p.add_argument("--workers", type=int, default=6, help="concurrent in-flight API calls")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--max-waves", type=int, default=30)
    p.add_argument("--oversample", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap-total", type=int, default=0)
    p.add_argument("--bench", default=None)
    p.add_argument("--seed-pool", default=None)
    p.add_argument("--out", default="data/train/dialogues_train_api.jsonl")
    p.add_argument("--cache", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    keys = [k.strip() for k in os.environ.get("MIMO_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        sys.exit("set MIMO_API_KEYS='key1,key2'")
    clients = [OpenAI(api_key=k, base_url=args.base_url) for k in keys]
    print(f"{len(clients)} key(s) | model={args.model} | base={args.base_url}")

    bench_path = find_first(args.bench, ["/kaggle/input/**/dialogues_bench.jsonl"],
                            ["data/bench/dialogues_bench.jsonl"])
    seed_path = find_first(args.seed_pool, ["/kaggle/input/**/fewshot_seed.jsonl"],
                           ["data/seed/fewshot_seed.jsonl"])
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cache_path = args.cache or os.path.join(os.path.dirname(out_path) or ".", ".cache_gen_api.jsonl")
    print("seed:", seed_path, "| bench:", bench_path, "| cache:", cache_path)

    quota, n_ctx, n_noctx = build_quota(args.target_total, args.context_fraction)
    print(f"context-required: {n_ctx} | no-context: {n_noctx}")

    reject = {}
    validate = make_validator(reject)
    accepted = {}

    def _key(s):
        return (s["turns"][-1]["content"], s["rewrite"])

    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            line = line.strip()
            if line:
                s = json.loads(line)
                accepted[_key(s)] = s
        print(f"resumed {len(accepted)} cached samples")

    def have_counts():
        return Counter((s["context_required"], s["user_turns"]) for s in accepted.values())

    seed_pool, n_curated = load_seed_pool(seed_path, bench_path)
    print(f"seed pool: {len(seed_pool)} ({n_curated} curated)")

    def samples_per_call(ut):
        return args.long_samples_per_call if ut >= 4 else (3 if ut == 3 else args.samples_per_call)

    def fewshot_block(ctx_req, want_ut, k=args.n_fewshot):
        if not seed_pool:
            return ""
        same = [p for p in seed_pool if p.get("context_required") == ctx_req] or seed_pool
        same = sorted(same, key=lambda p: 0 if p.get("user_turns") == want_ut else 1)[:max(k * 3, 6)]
        picks = random.sample(same, min(k, len(same)))
        ex = [{"turns": p["turns"], "rewrite": p["rewrite"], "domain": p["domain"]} for p in picks]
        return ("Vài ví dụ THAM KHẢO về độ khó & văn phong (đừng sao chép nội dung):\n"
                + json.dumps(ex, ensure_ascii=False, indent=1) + "\n\n")

    def schema_instructions(want_ut, ctx_req):
        bots = want_ut - 1
        n = samples_per_call(want_ut)
        skeleton = " → ".join(["user", "bot"] * bots + ["user"])
        kind = ("context_required=true (lượt user cuối THIẾU thông tin, rewrite phải dùng ngữ cảnh trước)"
                if ctx_req else
                "context_required=false (lượt user cuối ĐÃ đầy đủ, rewrite giữ nguyên ý lượt cuối)")
        return (f"Sinh {n} mẫu KHÁC NHAU, loại: {kind}.\n"
                f"CẤU TRÚC BẮT BUỘC — đếm cho đúng số lượt:\n    {skeleton}\n"
                f"→ ĐÚNG {want_ut} lượt user và {bots} lượt bot, xen kẽ, KẾT THÚC bằng user. "
                f"Sai số lượt hoặc kết thúc bằng bot sẽ bị LOẠI.\n\n"
                "Schema JSON bắt buộc:\n"
                '{ "samples": [ { "turns": [ {"role":"user","content":"..."}, {"role":"bot","content":"..."} ], '
                '"rewrite":"câu rewrite cho lượt user cuối", "domain":"navigation|climate|music|calling|messaging|'
                'charging|smart_home|vehicle|driver_assist", "rationale":"1 câu vì sao" } ] }')

    def pick_domains(k=3):
        return random.choices(DOMAINS, weights=[DOMAIN_WEIGHTS[d] for d in DOMAINS], k=k)

    rr = {"i": 0}
    rr_lock = threading.Lock()

    def call_one(ctx, ut, doms):
        user = (fewshot_block(ctx, ut) + schema_instructions(ut, ctx)
                + f"\n\nƯu tiên các domain: {', '.join(doms)}. seed={random.randint(1, 10**7)}")
        for attempt in range(args.max_retries):
            with rr_lock:
                c = clients[rr["i"] % len(clients)]
                rr["i"] += 1
            try:
                r = c.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": GEN_SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=args.gen_temperature, top_p=args.gen_top_p,
                    max_tokens=args.gen_max_tokens,
                    response_format={"type": "json_object"},
                )
                return r.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(min(1.5 ** attempt + random.random(), 20))
                else:
                    time.sleep(1 + random.random())
        return None

    cache_fh = open(cache_path, "a", encoding="utf-8")
    write_lock = threading.Lock()
    t0 = time.time()

    for wave in range(1, args.max_waves + 1):
        have = have_counts()
        deficits = {b: quota[b] - have.get(b, 0) for b in quota if quota[b] - have.get(b, 0) > 0}
        if not deficits:
            print("all quotas filled ✓")
            break

        specs = []
        for (ctx, ut), need in deficits.items():
            n_calls = max(1, math.ceil(need * args.oversample / samples_per_call(ut)))
            for _ in range(n_calls):
                specs.append((ctx, ut, pick_domains(3)))
        random.shuffle(specs)
        print(f"\n── wave {wave}: deficits={deficits} → {len(specs)} calls")

        reject.clear()
        kept = ok = err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(call_one, ctx, ut, doms): (ctx, ut, doms) for ctx, ut, doms in specs}
            for fut in as_completed(futs):
                ctx, ut, doms = futs[fut]
                txt = fut.result()
                if not txt:
                    err += 1
                    continue
                ok += 1
                for raw in parse_samples(txt):
                    v = validate(raw, ctx, ut)
                    if not v:
                        continue
                    v["domain"] = coerce_domain(v.get("domain"), doms[0])
                    k = _key(v)
                    with write_lock:
                        if k in accepted:
                            continue
                        accepted[k] = v
                        cache_fh.write(json.dumps(v, ensure_ascii=False) + "\n")
                        cache_fh.flush()
                        kept += 1
        rate = ok / max(1, ok + err)
        print(f"   calls ok={ok} failed={err} (success {rate:.0%}) | kept {kept} "
              f"| total {len(accepted)}/{args.target_total} | rejects {dict(reject)}")
        if ok == 0:
            print("   !! every call failed (rate-limit/expired key?) — stopping")
            break

    cache_fh.close()
    dt = time.time() - t0
    print(f"\nDONE — {len(accepted)} accepted in {dt/60:.1f} min "
          f"({len(accepted)/max(dt,1):.2f} samples/s)")

    # finalize → Llama-Factory
    if args.cap_total and len(accepted) > args.cap_total:
        selected = select_capped(accepted, quota, args.cap_total)
    else:
        selected = list(accepted.values())
    records, seen = [], set()
    for s in selected:
        k = _key(s)
        if k in seen:
            continue
        seen.add(k)
        r = to_lf(s)
        r["meta"]["source"] = args.model
        records.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} → {out_path}")
    print("context_required:", dict(Counter(r["meta"]["context_required"] for r in records)))
    print("user_turns:", dict(sorted(Counter(r["meta"]["user_turns"] for r in records).items())))
    print("domain:", dict(Counter(r["meta"]["domain"] for r in records).most_common()))


if __name__ == "__main__":
    main()
