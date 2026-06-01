"""Smoke test: gen a few Vietnamese rewrite samples via the MiMo OpenAI-compatible
API and eyeball quality vs data/cache_gen_vllm.jsonl.

Reuses GEN_SYSTEM + validator from generate_vllm so the comparison is apples-to-apples.

    MIMO_API_KEY=tp-xxx MIMO_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1 \
        python src/data/smoke_mimo.py
"""
from __future__ import annotations
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vllm import (  # noqa: E402
    GEN_SYSTEM, make_validator, parse_samples, load_seed_pool,
)
from openai import OpenAI  # noqa: E402

MODEL = os.environ.get("MIMO_MODEL", "MiMo-V2.5-Pro")
client = OpenAI(api_key=os.environ["MIMO_API_KEY"],
                base_url=os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))

SEED = os.path.join(os.path.dirname(__file__), "../../data/seed/fewshot_seed.jsonl")
BENCH = os.path.join(os.path.dirname(__file__), "../../data/bench/dialogues_bench.jsonl")
seed_pool, _ = load_seed_pool(SEED, BENCH)


def fewshot_block(ctx_req, want_ut, k=3):
    same = [p for p in seed_pool if p.get("context_required") == ctx_req] or seed_pool
    same = sorted(same, key=lambda p: 0 if p.get("user_turns") == want_ut else 1)[:max(k * 3, 6)]
    picks = random.sample(same, min(k, len(same)))
    ex = [{"turns": p["turns"], "rewrite": p["rewrite"], "domain": p["domain"]} for p in picks]
    return ("Vài ví dụ THAM KHẢO về độ khó & văn phong (đừng sao chép nội dung):\n"
            + json.dumps(ex, ensure_ascii=False, indent=1) + "\n\n")


def schema_instr(want_ut, ctx_req, n=3):
    bots = want_ut - 1
    skeleton = " → ".join(["user", "bot"] * bots + ["user"])
    kind = ("context_required=true (lượt user cuối THIẾU thông tin, rewrite phải dùng ngữ cảnh trước)"
            if ctx_req else
            "context_required=false (lượt user cuối ĐÃ đầy đủ, rewrite giữ nguyên ý lượt cuối)")
    return (f"Sinh {n} mẫu KHÁC NHAU, loại: {kind}.\n"
            f"CẤU TRÚC BẮT BUỘC:\n    {skeleton}\n"
            f"→ ĐÚNG {want_ut} lượt user và {bots} lượt bot, xen kẽ, KẾT THÚC bằng user.\n\n"
            "Schema JSON bắt buộc:\n"
            '{ "samples": [ { "turns": [ {"role":"user","content":"..."}, {"role":"bot","content":"..."} ], '
            '"rewrite":"...", "domain":"navigation|climate|music|calling|messaging|charging|smart_home|vehicle|driver_assist", '
            '"rationale":"1 câu vì sao" } ] }')


def call(ctx_req, want_ut):
    user = (fewshot_block(ctx_req, want_ut) + schema_instr(want_ut, ctx_req)
            + f"\n\nseed={random.randint(1, 10**7)}")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
        temperature=1.0, top_p=0.95, max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def main():
    random.seed(1)
    reject = {}
    validate = make_validator(reject)
    buckets = [(True, 2), (True, 3), (False, 1), (False, 2)]
    kept = []
    for ctx, ut in buckets:
        try:
            txt = call(ctx, ut)
        except Exception as e:
            print(f"!! API error for ctx={ctx} ut={ut}: {type(e).__name__}: {e}")
            continue
        for raw in parse_samples(txt):
            v = validate(raw, ctx, ut)
            if v:
                kept.append(v)
    print(f"\n=== kept {len(kept)} valid samples | rejects: {dict(reject)} ===\n")
    for s in kept:
        print(json.dumps(s, ensure_ascii=False))
    if kept:
        out = os.path.join(os.path.dirname(__file__), "../../data/smoke_mimo.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(kept)} → {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
