"""Phase-2 bulk training-data generation with a vLLM teacher + self-judge.

Standalone script form of `kaggle_gen_data.ipynb`. Run it as a SUBPROCESS from
the notebook (`!python src/data/generate_vllm.py`) rather than importing torch/
vllm into the Kaggle kernel: Kaggle pre-imports torch/numpy at kernel startup, and
the vllm install downgrades numpy/torch on disk. Importing the downgraded packages
back into that same long-lived kernel triggers a numpy ABI ValueError (papermill /
"Save & Run All" never restarts the kernel, so the usual "Run > Restart" fix can't
apply). A fresh `python` process loads the downgraded stack cleanly.

A quantized Qwen2.5-14B-Instruct-AWQ served by vLLM bulk-generates Vietnamese
in-car dialogues, and the SAME model self-judges them (strict prompt, temp 0).
Output is the Llama-Factory `conversations` schema.

The wave loop RESUMES from --cache, so re-running fills only the buckets still
short of quota. Use --cap-total to trim the final file to exactly N samples
(quota-balanced: scarce buckets are protected, overflow buckets are trimmed), and
--no-generate to do that finalize step from the cache alone (no GPU needed).

Examples:
    # Kaggle (GPU): top up the 7-turn bucket, resume from an existing cache,
    # then write exactly 3000 quota-balanced samples.
    python src/data/generate_vllm.py --target-total 3000 --cap-total 3000 \\
        --cache /kaggle/working/data/train/.cache_gen_vllm.jsonl \\
        --max-waves 40 --oversample 4

    # Local (no GPU): just assemble exactly 3000 from a finished cache.
    python src/data/generate_vllm.py --no-generate --cap-total 3000 \\
        --cache data/cache_gen_vllm.jsonl \\
        --out data/train/dialogues_train_vllm.jsonl
"""
from __future__ import annotations

# spawn (not fork) for vLLM TP workers — MUST be set before any torch/vllm import,
# otherwise TP=2 crashes with "Cannot re-initialize CUDA in forked subprocess".
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import glob
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict

# Import the training system prompt from the shared module so generated data stays
# aligned with deployment. The script lives in src/data/ next to prompts.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT_FOR_TRAINING, REWRITE_TAG  # noqa: E402


# ===========================================================================
# System prompts (generator + judge are generation-only; training prompt is
# imported from prompts.py)
# ===========================================================================
GEN_SYSTEM = """Bạn là chuyên gia tạo dữ liệu HUẤN LUYỆN tiếng Việt cho task rewrite hội thoại trong xe ô tô.

Mỗi mẫu gồm một đoạn hội thoại nhiều lượt (xen kẽ user/bot, kết thúc bằng user) và MỘT câu rewrite chuẩn cho lượt user CUỐI.

NGUYÊN TẮC SINH:
- Đa dạng domain; slot phải CỤ THỂ và có thật (tên người, địa điểm Hà Nội/Sài Gòn thật, bài Vpop thật, hãng EV thật, nhiệt độ/khoảng cách hợp lý).
- Văn nói tự nhiên trong xe: có thể lược chủ ngữ, nói nhanh, slang nhẹ.
- Biến tấu opener của câu rewrite — KHÔNG luôn mở đầu bằng "Tôi muốn...". Dùng "Hãy...", "Làm ơn...", "Cho tôi...", "Mình cần...", câu mệnh lệnh ngắn.
- KHÔNG lặp scenario/từ vựng giữa các mẫu trong cùng batch.

HAI LOẠI MẪU (rất quan trọng):
(A) context_required = true — lượt user CUỐI THIẾU thông tin (đại từ "cái đó/bài đó", quá ngắn, slot rải rác ở các lượt trước, hoặc tham chiếu ngầm). Câu rewrite PHẢI giải/gộp slot từ các lượt TRƯỚC.
    • TUYỆT ĐỐI KHÔNG lặp lại hành động mà BOT ĐÃ LÀM XONG ở lượt trước — chỉ rewrite hành động CHƯA được thực hiện.
(B) context_required = false — lượt user CUỐI ĐÃ ĐẦY ĐỦ, tự nó thực hiện được (vd "Bật đèn pha như tôi yêu cầu đi"). Câu rewrite GIỮ NGUYÊN ý lượt cuối, chỉ làm sạch/chuẩn hoá; KHÔNG kéo slot lạ từ các lượt trước.

Giữ negation ("đừng/trừ/không/tránh") và compound ("và/rồi/sau đó") nếu có — KHÔNG làm mất.

Chỉ trả về JSON đúng schema yêu cầu, không thêm chữ nào ngoài JSON."""


JUDGE_SYSTEM = """Bạn là GIÁM KHẢO chấm câu rewrite tiếng Việt cho task hội thoại trợ lý xe. Hãy chấm CỰC KỲ NGHIÊM KHẮC và theo đúng luật.

Bạn nhận:
- dialogue: lịch sử hội thoại (xen kẽ user/bot, kết thúc bằng user).
- context_required: true nếu lượt user cuối cần ngữ cảnh để hiểu, false nếu lượt cuối đã đầy đủ.
- prediction: câu rewrite cần chấm.

CHO score = 1 KHI VÀ CHỈ KHI prediction thoả MỌI điều:
1. Là MỘT câu tiếng Việt hoàn chỉnh, đứng một mình hiểu được, đúng ý định lượt user cuối.
2. Bảo toàn TẤT CẢ slot cần thiết (tên người, địa điểm, số, nhiệt độ, bài hát, hãng, chế độ).
3. KHÔNG thêm thông tin/slot bịa, KHÔNG kéo nhầm số hay slot mà bot chỉ nhắc tới nhưng user không yêu cầu.
4. Nếu context_required=true: phải GIẢI đại từ/tham chiếu và GỘP đủ slot rải rác từ các lượt trước; và KHÔNG lặp lại hành động bot đã làm xong.
5. Nếu context_required=false: phải bám sát lượt user cuối, KHÔNG thêm slot từ các lượt trước.
6. Giữ đúng polarity negation (đừng/trừ/không/tránh) và đủ cả 2 vế của compound intent.

NGƯỢC LẠI cho score = 0 (thiếu slot, sai intent, bịa thông tin, mất negation, lặp hành động đã làm, hoặc câu không tự đứng được).

Chỉ trả về JSON: {"score": 0 hoặc 1, "reason": "1 câu ngắn tiếng Việt"}."""


# ===========================================================================
# CLI / config
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # teacher model
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    p.add_argument("--tensor-parallel", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--device", default="cuda",
                   help="vLLM device. 'cuda' avoids the vllm 0.6.3 "
                        "'Failed to infer device type' auto-detect bug; use 'auto' to re-enable detection.")
    # volume + mix
    p.add_argument("--target-total", type=int, default=3000)
    p.add_argument("--context-fraction", type=float, default=2 / 3,
                   help="fraction of samples that require context (default 2/3)")
    # generation
    p.add_argument("--gen-temperature", type=float, default=1.0)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--gen-max-tokens", type=int, default=3072)
    p.add_argument("--samples-per-call", type=int, default=5)
    p.add_argument("--long-samples-per-call", type=int, default=1,
                   help="samples/call for 4-user-turn (7-msg) prompts; 1 = most complete (default)")
    p.add_argument("--n-fewshot", type=int, default=3)
    # judge
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--judge-max-tokens", type=int, default=192)
    # wave loop
    p.add_argument("--max-waves", type=int, default=14)
    p.add_argument("--oversample", type=float, default=2.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=64, help="prompts per llm.generate() call")
    p.add_argument("--max-num-seqs", type=int, default=64)
    # finalize
    p.add_argument("--cap-total", type=int, default=0,
                   help="trim final output to exactly N samples (quota-balanced); 0 = keep all")
    p.add_argument("--no-generate", action="store_true",
                   help="skip GPU generation; finalize from --cache only (runs without a GPU)")
    # paths
    p.add_argument("--bench", default=None, help="few-shot/bench pool (auto-detected if omitted)")
    p.add_argument("--seed-pool", default=None, help="curated few-shot seed (auto-detected if omitted)")
    p.add_argument("--out", default="/kaggle/working/data/train/dialogues_train_vllm.jsonl")
    p.add_argument("--cache", default=None, help="resume cache (default: <out-dir>/.cache_gen_vllm.jsonl)")
    return p.parse_args()


def find_first(explicit, patterns, fallbacks):
    if explicit and os.path.exists(explicit):
        return explicit
    cands = []
    for pat in patterns:
        cands += glob.glob(pat, recursive=True)
    cands += fallbacks
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# ===========================================================================
# Quota plan (turn-count x context x domain)
# ===========================================================================
DOMAIN_WEIGHTS = {
    "navigation": 1.0, "climate": 1.0, "music": 1.0, "calling": 1.0,
    "messaging": 2.0, "charging": 2.5, "smart_home": 3.0,
    "vehicle": 3.0, "driver_assist": 3.0,
}
DOMAINS = list(DOMAIN_WEIGHTS)

# user_turns distribution within each context class.
TURN_DIST = {
    True:  {2: 0.45, 3: 0.35, 4: 0.20},   # context-required -> >=2 user turns
    False: {1: 0.55, 2: 0.45},            # no-context -> mostly single turn
}


def build_quota(target_total, context_fraction):
    def _alloc(total, dist):
        out = {k: round(total * w) for k, w in dist.items()}
        out[max(out)] += total - sum(out.values())  # fix rounding
        return out

    n_ctx = round(target_total * context_fraction)
    n_noctx = target_total - n_ctx
    quota = {}
    for ut, c in _alloc(n_ctx, TURN_DIST[True]).items():
        quota[(True, ut)] = c
    for ut, c in _alloc(n_noctx, TURN_DIST[False]).items():
        quota[(False, ut)] = c
    return quota, n_ctx, n_noctx


def select_capped(accepted, quota, cap):
    """Pick exactly `cap` samples, quota-balanced. Each bucket first gets up to its
    quota (so scarce buckets like 4-user-turn keep ALL their samples); leftover
    slots are filled from buckets that have a surplus. Returns a list of samples."""
    groups = defaultdict(list)
    for s in accepted.values():
        groups[(s["context_required"], s["user_turns"])].append(s)

    take = {b: min(len(items), quota.get(b, 0)) for b, items in groups.items()}
    total = sum(take.values())

    if total < cap:                       # under quota somewhere -> fill from surplus
        need = cap - total
        surplus = {b: len(groups[b]) - take[b] for b in groups}
        for b in sorted(surplus, key=lambda x: -surplus[x]):
            if need <= 0:
                break
            add = min(surplus[b], need)
            take[b] += add
            need -= add
    elif total > cap:                     # quotas exceed cap -> trim biggest buckets
        over = total - cap
        for b in sorted(take, key=lambda x: -take[x]):
            if over <= 0:
                break
            red = min(take[b], over)
            take[b] -= red
            over -= red

    out = []
    for b, items in groups.items():
        out.extend(items[:take[b]])
    return out


# ===========================================================================
# Seed few-shot pool
# ===========================================================================
def bench_to_compact(rec):
    convs = [c for c in rec["conversations"] if c["from"] != "system"]
    gold = json.loads(convs[-1]["value"])["rewrite_message"]
    turns = []
    for c in convs[:-1]:
        role = "user" if c["from"] == "human" else "bot"
        v = c["value"]
        if v.startswith("<REWRITE>\n"):
            v = v[len("<REWRITE>\n"):]
        turns.append({"role": role, "content": v})
    n_user = sum(1 for t in turns if t["role"] == "user")
    return {"turns": turns, "rewrite": gold, "user_turns": n_user,
            "context_required": True,  # bench samples are all hard/context cases
            "domain": rec.get("meta", {}).get("domain", "navigation")}


def load_seed_pool(seed_path, bench_path):
    pool = []
    if seed_path:
        for line in open(seed_path, encoding="utf-8"):
            try:
                pool.append(json.loads(line))
            except Exception:
                pass
    n_curated = len(pool)
    if bench_path:
        for line in open(bench_path, encoding="utf-8"):
            try:
                pool.append(bench_to_compact(json.loads(line)))
            except Exception:
                pass
    return pool, n_curated


# ===========================================================================
# Validators
# ===========================================================================
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _norm(t):
    t = unicodedata.normalize("NFC", t).lower().strip()
    return re.sub(r"[\s,.!?;:\"'()\-]+", " ", t).strip()


def _nums(t):
    return {m.group(0).replace(",", ".") for m in NUMBER_RE.finditer(t)}


def make_validator(reject_counter):
    def _rej(reason):
        reject_counter[reason] = reject_counter.get(reason, 0) + 1
        return None

    def validate(s, ctx_req, want_ut):
        if not isinstance(s, dict):
            return _rej("not_dict")
        turns, rw = s.get("turns"), (s.get("rewrite") or "").strip()
        domain = (s.get("domain") or "").strip()
        if not isinstance(turns, list) or not rw or not domain:
            return _rej("missing_field")
        if not (5 <= len(rw) <= 240):
            return _rej("rewrite_length")

        cleaned, prev = [], None
        for t in turns:
            if not isinstance(t, dict):
                return _rej("turn_not_dict")
            role, content = t.get("role"), (t.get("content") or "").strip()
            if role not in ("user", "bot") or not content:
                return _rej("bad_turn")
            if role == prev:
                return _rej("not_alternating")
            prev = role
            cleaned.append({"role": role, "content": content})
        if not cleaned or cleaned[0]["role"] != "user":
            return _rej("not_start_user")
        if cleaned[-1]["role"] != "user":
            return _rej("not_end_user")

        n_user = sum(1 for t in cleaned if t["role"] == "user")
        if n_user != want_ut:
            return _rej("wrong_user_turns")

        last_user = cleaned[-1]["content"]
        all_nums = _nums(" ".join(t["content"] for t in cleaned))
        rw_nums = _nums(rw)
        if not rw_nums.issubset(all_nums):
            return _rej("hallucinated_number")

        if ctx_req:
            if _norm(last_user) == _norm(rw):
                return _rej("ctx_but_identical")
        else:
            # keep-as-is: rewrite must not pull numbers absent from the final user turn
            if not rw_nums.issubset(_nums(last_user)):
                return _rej("noctx_pulled_number")

        return {"turns": cleaned, "rewrite": rw, "domain": domain,
                "context_required": ctx_req, "user_turns": n_user,
                "rationale": (s.get("rationale") or "").strip()}

    return validate


def to_lf(s):
    convs = [{"from": "system", "value": SYSTEM_PROMPT_FOR_TRAINING}]
    for t in s["turns"][:-1]:
        convs.append({"from": "human" if t["role"] == "user" else "gpt", "value": t["content"]})
    convs.append({"from": "human", "value": f"{REWRITE_TAG}\n{s['turns'][-1]['content']}"})
    convs.append({"from": "gpt", "value": json.dumps({"rewrite_message": s["rewrite"]}, ensure_ascii=False)})
    return {"conversations": convs,
            "meta": {"domain": s["domain"], "context_required": s["context_required"],
                     "user_turns": s["user_turns"], "total_turns": len(s["turns"]),
                     "source": "qwen2.5-14b-awq"}}


# ===========================================================================
# Parsing (resilient to truncated JSON)
# ===========================================================================
def _recover_objects(text):
    """Recover COMPLETE {...} objects inside the "samples" array even when JSON is
    truncated (output exceeded max_tokens). Brace counting respects strings &
    escapes, so a half-cut final sample only drops itself."""
    i = text.find('"samples"')
    start = text.find('[', i) if i != -1 else text.find('[')
    if start == -1:
        return []
    out, depth, obj_start = [], 0, None
    in_str = esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            if depth == 0:
                obj_start = j
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    out.append(json.loads(text[obj_start:j + 1]))
                except Exception:
                    pass
                obj_start = None
    return out


def parse_samples(text):
    try:
        s = json.loads(text).get("samples", [])
        if s:
            return s
    except Exception:
        pass
    return _recover_objects(text)


def parse_score(text):
    try:
        return int(json.loads(text).get("score", 0))
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                return int(json.loads(m.group(0)).get("score", 0))
            except Exception:
                pass
        return 1 if re.search(r'"score"\s*:\s*1', text) else 0


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    # --- resolve paths -----------------------------------------------------
    bench_path = find_first(
        args.bench,
        ["/kaggle/input/**/dialogues_bench.jsonl"],
        ["data/bench/dialogues_bench.jsonl",
         "/kaggle/working/qwen-lora-finetuning/data/bench/dialogues_bench.jsonl"])
    seed_path = find_first(
        args.seed_pool,
        ["/kaggle/input/**/fewshot_seed.jsonl"],
        ["data/seed/fewshot_seed.jsonl",
         "/kaggle/working/qwen-lora-finetuning/data/seed/fewshot_seed.jsonl"])
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cache_path = args.cache or os.path.join(os.path.dirname(out_path), ".cache_gen_vllm.jsonl")

    print("seed  :", seed_path or "NOT FOUND (curated few-shot)")
    print("bench :", bench_path or "NOT FOUND")
    print("out   :", out_path)
    print("cache :", cache_path)

    # --- quota -------------------------------------------------------------
    quota, n_ctx, n_noctx = build_quota(args.target_total, args.context_fraction)
    print(f"\ncontext-required: {n_ctx}  |  no-context: {n_noctx}")
    for k in sorted(quota, key=lambda x: (str(x[0]), x[1])):
        print(f"  ctx={str(k[0]):5s} user_turns={k[1]} -> {quota[k]}")

    # --- load / resume cache (needed for both generate and finalize) -------
    reject = {}
    validate = make_validator(reject)
    accepted = {}  # (final_user, rewrite) -> sample

    def _key(s):
        return (s["turns"][-1]["content"], s["rewrite"])

    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            accepted[_key(s)] = s
        print(f"\nresumed {len(accepted)} cached samples")

    def have_counts():
        return Counter((s["context_required"], s["user_turns"]) for s in accepted.values())

    # ===================================================================
    # GENERATE (skipped entirely with --no-generate; no GPU touched)
    # ===================================================================
    if args.no_generate:
        print("\n--no-generate: skipping GPU generation, finalizing from cache")
    else:
        def pick_domains(k=3):
            return random.choices(DOMAINS, weights=[DOMAIN_WEIGHTS[d] for d in DOMAINS], k=k)

        seed_pool, n_curated = load_seed_pool(seed_path, bench_path)
        print(f"seed pool: {len(seed_pool)} ({n_curated} curated + {len(seed_pool)-n_curated} bench)")

        def fewshot_block(ctx_req, want_ut, k=args.n_fewshot):
            if not seed_pool:
                return ""
            same_ctx = [p for p in seed_pool if p.get("context_required") == ctx_req]
            pool = same_ctx or seed_pool
            pool = sorted(pool, key=lambda p: 0 if p.get("user_turns") == want_ut else 1)
            picks = pool[: max(k * 3, 6)]
            picks = random.sample(picks, min(k, len(picks)))
            ex = [{"turns": p["turns"], "rewrite": p["rewrite"], "domain": p["domain"]} for p in picks]
            return ("Vài ví dụ THAM KHẢO về độ khó & văn phong (đừng sao chép nội dung):\n"
                    + json.dumps(ex, ensure_ascii=False, indent=1) + "\n\n")

        def samples_per_call(ut):
            # Longer dialogues overflow gen-max-tokens and get truncated, corrupting
            # the batch's JSON. Ask for fewer samples/call for long buckets.
            if ut >= 4:
                return args.long_samples_per_call
            return 3 if ut == 3 else args.samples_per_call

        def schema_instructions(want_ut, ctx_req):
            bots = want_ut - 1
            n = samples_per_call(want_ut)
            skeleton = " → ".join(["user", "bot"] * (want_ut - 1) + ["user"])
            kind = ("context_required=true (lượt user cuối THIẾU thông tin, rewrite phải dùng ngữ cảnh trước)"
                    if ctx_req else
                    "context_required=false (lượt user cuối ĐÃ đầy đủ, rewrite giữ nguyên ý lượt cuối)")
            return (
                f"Sinh {n} mẫu KHÁC NHAU, loại: {kind}.\n"
                f"CẤU TRÚC BẮT BUỘC — đếm cho đúng số lượt:\n"
                f"    {skeleton}\n"
                f"→ ĐÚNG {want_ut} lượt user và {bots} lượt bot, xen kẽ, KẾT THÚC bằng user. "
                f"Sai số lượt hoặc kết thúc bằng bot sẽ bị LOẠI.\n\n"
                "Schema JSON bắt buộc:\n"
                '{ "samples": [ { "turns": [ {"role":"user","content":"..."}, {"role":"bot","content":"..."} ], '
                '"rewrite":"câu rewrite cho lượt user cuối", "domain":"navigation|climate|music|calling|messaging|'
                'charging|smart_home|vehicle|driver_assist", "rationale":"1 câu vì sao" } ] }'
            )

        # --- engine (heavy import happens HERE, in this fresh process) ------
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            device=args.device,           # explicit -> skips vllm 0.6.3 auto device inference
            quantization="awq",
            tensor_parallel_size=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            dtype="float16",
            trust_remote_code=True,
            enforce_eager=True,           # TP=2 on Kaggle: skip CUDA-graph capture -> stabler startup
            enable_chunked_prefill=True,  # avoids preemption assert in 0.6.3 scheduler
            max_num_seqs=args.max_num_seqs,
        )

        def build_prompt(system, user):
            return tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )

        def generate_chunked(prompts, sp, chunk=args.chunk):
            outs = []
            for i in range(0, len(prompts), chunk):
                outs.extend(llm.generate(prompts[i:i + chunk], sp))
            return outs

        gen_sp = SamplingParams(temperature=args.gen_temperature, top_p=args.gen_top_p,
                                max_tokens=args.gen_max_tokens)
        judge_sp = SamplingParams(temperature=args.judge_temperature, max_tokens=args.judge_max_tokens)
        print("engine ready")

        def dialogue_str(s):
            return "\n".join(f"{t['role']}: {t['content']}" for t in s["turns"])

        def judge_prompt(s):
            return build_prompt(JUDGE_SYSTEM,
                                f"dialogue:\n{dialogue_str(s)}\n\n"
                                f"context_required: {str(s['context_required']).lower()}\n\n"
                                f"prediction:\n{s['rewrite']}\n\nChấm theo schema.")

        # --- wave loop -----------------------------------------------------
        cache_fh = open(cache_path, "a", encoding="utf-8")
        for wave in range(1, args.max_waves + 1):
            have = have_counts()
            deficits = {b: quota[b] - have.get(b, 0) for b in quota if quota[b] - have.get(b, 0) > 0}
            if not deficits:
                print("all quotas filled ✓")
                break

            prompts, specs = [], []
            for (ctx, ut), need in deficits.items():
                n_calls = max(1, math.ceil(need * args.oversample / samples_per_call(ut)))
                for _ in range(n_calls):
                    doms = pick_domains(3)
                    user = (fewshot_block(ctx, ut)
                            + schema_instructions(ut, ctx)
                            + f"\n\nƯu tiên các domain: {', '.join(doms)}. seed={random.randint(1, 10_000_000)}")
                    prompts.append(build_prompt(GEN_SYSTEM, user))
                    specs.append((ctx, ut))

            print(f"\n── wave {wave}: deficits={deficits} → {len(prompts)} gen prompts")
            outs = generate_chunked(prompts, gen_sp)

            reject.clear()
            cands = []
            for (ctx, ut), o in zip(specs, outs):
                for raw in parse_samples(o.outputs[0].text):
                    v = validate(raw, ctx, ut)
                    if v and _key(v) not in accepted:
                        cands.append(v)
            print(f"   validated candidates: {len(cands)}  | rejects: {dict(reject)}")
            if not cands:
                continue

            jouts = generate_chunked([judge_prompt(s) for s in cands], judge_sp)
            kept = 0
            for s, jo in zip(cands, jouts):
                if parse_score(jo.outputs[0].text) == 1:
                    k = _key(s)
                    if k in accepted:
                        continue
                    accepted[k] = s
                    cache_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                    cache_fh.flush()
                    kept += 1
            print(f"   judged → kept {kept} (total {len(accepted)}/{args.target_total})")

        cache_fh.close()
        print(f"\nDONE — {len(accepted)} accepted samples")

    # ===================================================================
    # FINALIZE: select (optional cap) + convert + save + report
    # ===================================================================
    if args.cap_total and len(accepted) > args.cap_total:
        selected = select_capped(accepted, quota, args.cap_total)
        print(f"\ncapped {len(accepted)} → {len(selected)} samples (quota-balanced)")
    else:
        selected = list(accepted.values())
        if args.cap_total:
            print(f"\ncap-total={args.cap_total} but only {len(accepted)} available — keeping all")

    records, seen = [], set()
    for s in selected:
        k = _key(s)
        if k in seen:
            continue
        seen.add(k)
        records.append(to_lf(s))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} samples → {out_path}\n")

    by_ctx = Counter(r["meta"]["context_required"] for r in records)
    by_ut = Counter(r["meta"]["user_turns"] for r in records)
    by_dom = Counter(r["meta"]["domain"] for r in records)
    print("context_required:", dict(by_ctx), f"  (target {args.context_fraction:.0%} True)")
    print("user_turns      :", dict(sorted(by_ut.items())))
    print("domain          :")
    for d, n in by_dom.most_common():
        print(f"   {d:14s} {n}")
    if reject:
        print("\nvalidator rejections (last wave):")
        for k, v in sorted(reject.items(), key=lambda x: -x[1]):
            print(f"   {k:24s} {v}")


if __name__ == "__main__":
    main()
