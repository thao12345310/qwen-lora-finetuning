"""Diagnose whether train.jsonl is skewed toward 'short, info-near-final-turn' rewrites.

Tests the root-cause hypothesis (REPORT.md §C): the model learns
"rewrite = tidy up the last turn", so it drops (a) negation clauses and
(b) qualifiers that were said ONCE in a middle turn. If train rarely contains
either, that under-coverage is the shared root of the negation and
multi_turn_slot weaknesses.

Outputs distribution stats; no judge, no network. Pure local count.

Run:  /opt/homebrew/bin/python3.11 -m src.eval.analyze_train_distribution
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = Path("data/processed/train.jsonl")

# Negation cues (word-boundary; lowercased text). "không" dominates but is also
# the yes/no QUESTION particle ("... có ... không?") and a discourse decline
# ("Không, cảm ơn") — both must be excluded or the negation rate is wildly
# inflated. _is_real_negation() filters those before we trust a NEG hit.
NEG = re.compile(r"(?<![\wàáảãạăắằẳẵặâấầẩẫậ])"
                 r"(không|đừng|chớ|chẳng|đâu có|tránh|khỏi|miễn|ngoại trừ)"
                 r"(?![\wàáảãạăắằẳẵặâấầẩẫậ])", re.IGNORECASE)

# "không" as interrogative tail: clause-final, optionally trailed by ? or a
# question/politeness particle. e.g. "... bật không?", "... được không nhỉ".
_Q_KHONG = re.compile(r"không\s*(nhỉ|nha|nhé|à|ạ|vậy|thế|chưa|chăng)?\s*[?\.…]*\s*$",
                      re.IGNORECASE)
# "Không" opening a turn as a bare decline: "Không," / "Không." / "Không cảm ơn"
_DECLINE = re.compile(r"^\s*không\s*[,\.]", re.IGNORECASE)


# "không X" compounds where không is part of a NOUN, not negation.
_NOUN_KHONG = re.compile(r"không\s+(gian|khí|trung|lực|tặc)", re.IGNORECASE)


def is_real_negation(text: str) -> bool:
    """NEG.search hits, but NOT purely as a question particle, bare decline, or
    a không-headed noun ('không gian', 'không khí')."""
    if not NEG.search(text):
        return False
    # Strip the known non-negation constructs, then re-test what's left.
    stripped = _Q_KHONG.sub("", text)
    stripped = _DECLINE.sub("", stripped)
    stripped = _NOUN_KHONG.sub("", stripped)
    return bool(NEG.search(stripped))

# Coordinating cues that hint at ≥2 clauses in one rewrite.
COMPOUND = re.compile(r"\b(nhưng|còn|rồi|sau đó|đồng thời|và cả|, )\b", re.IGNORECASE)

# Light Vietnamese stopword set — function words + the task's filler verbs.
# Goal is content-token overlap, not perfect NLP.
STOP = set("""
và là của có cho với các một những này đó kia ấy nhé nha à ạ ơi thì mà ra vào lên
xuống đi đến tới về cho tôi mình bạn anh chị em ông bà nó họ chúng ta được bị đã
đang sẽ vừa mới luôn nữa rồi cũng vẫn còn chỉ rất quá lắm hơn nhất cái con chiếc
ở tại trong ngoài trên dưới giúp hãy xin làm ơn muốn cần phải nên hay hoặc nếu khi
đây đấy gì sao nào ai đâu bao giờ thế vậy ừ vâng dạ ok okay theo từ bằng để cùng
như giờ bây giờ hiện tại luôn đi nhé chứ chăng phần message rewrite
""".split())

CONTENT = re.compile(r"[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ0-9]+",
                     re.IGNORECASE)


def content_tokens(text: str) -> set[str]:
    toks = CONTENT.findall(text.lower())
    return {t for t in toks if t not in STOP and len(t) > 1}


def parse(rec: dict):
    """Return (context_turns, final_user, gold) where context_turns is a list of
    (role, text) BEFORE the final <REWRITE> human turn, ordered chronologically."""
    convs = [c for c in rec["conversations"] if c["from"] != "system"]
    # final human turn = the one carrying <REWRITE>
    final_idx = max(i for i, c in enumerate(convs) if c["from"] == "human")
    final_user = re.sub(r"^\s*<REWRITE_AND_CLASSIFY>\s*", "", convs[final_idx]["value"]).strip()
    context = [(c["from"], c["value"]) for c in convs[:final_idx]]
    gold_raw = convs[-1]["value"]
    try:
        gold = json.loads(gold_raw).get("rewrite_message", "")
    except Exception:
        m = re.search(r'"rewrite_message"\s*:\s*"(.*?)"\s*}', gold_raw, re.S)
        gold = m.group(1) if m else gold_raw
    return context, final_user, gold


def restoration_distance(context, final_user, gold):
    """Content tokens that appear in gold but NOT in the final user turn must have
    been restored from context. For each, find the NEAREST earlier turn holding it
    and record its distance (1 = turn immediately before final, 2 = one further, …).
    Returns (max_distance, n_restored_tokens). max_distance=0 => nothing restored."""
    g, fu = content_tokens(gold), content_tokens(final_user)
    restored = g - fu
    if not restored or not context:
        return 0, 0
    # distance: last context turn is distance 1, going back.
    dists = []
    for tok in restored:
        best = None
        for d, (_, text) in enumerate(reversed(context), start=1):
            if tok in content_tokens(text):
                best = d
                break
        if best is not None:
            dists.append(best)
    if not dists:
        return 0, 0
    return max(dists), len(dists)


def main():
    recs = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
    n = len(recs)
    print(f"# Train distribution diagnostic — {n} samples ({TRAIN})\n")

    ut, tt = Counter(), Counter()
    ctx_req = Counter()
    src = Counter()
    neg_final = neg_gold = 0
    neg_final_kept = neg_final_dropped = 0   # negation in final user → kept/dropped in gold
    compound_gold = 0
    far_restore = Counter()         # max restoration distance bucket (context_required only)
    restore_examples = []
    by_domain_neg = defaultdict(lambda: [0, 0])  # domain -> [n_neg_final, n]

    for r in recs:
        m = r.get("meta", {})
        ut[m.get("user_turns")] += 1
        tt[m.get("total_turns")] += 1
        cr = bool(m.get("context_required"))
        ctx_req[cr] += 1
        src[m.get("source", "?")] += 1
        dom = m.get("domain", "?")

        context, final_user, gold = parse(r)
        fn, gn = is_real_negation(final_user), is_real_negation(gold)
        neg_final += fn
        neg_gold += gn
        by_domain_neg[dom][1] += 1
        if fn:
            by_domain_neg[dom][0] += 1
            if gn:
                neg_final_kept += 1
            else:
                neg_final_dropped += 1
        if COMPOUND.search(gold):
            compound_gold += 1

        if cr:
            maxd, nrest = restoration_distance(context, final_user, gold)
            bucket = "0 (none)" if maxd == 0 else ("1 (liền kề)" if maxd == 1
                     else "2" if maxd == 2 else "≥3 (xa)")
            far_restore[bucket] += 1
            if maxd >= 3 and len(restore_examples) < 8:
                restore_examples.append((final_user, gold))

    def pct(x, d=n):
        return f"{100*x/d:.1f}%" if d else "—"

    print("## 1. Độ dài & context")
    print(f"context_required: True={ctx_req[True]} ({pct(ctx_req[True])}) | "
          f"False={ctx_req[False]} ({pct(ctx_req[False])})")
    print("user_turns:", dict(sorted((k, v) for k, v in ut.items() if k is not None)))
    print("total_turns:", dict(sorted((k, v) for k, v in tt.items() if k is not None)))
    short = sum(v for k, v in ut.items() if k is not None and k <= 2)
    print(f"→ user_turns ≤ 2 (hội thoại ngắn): {short} ({pct(short)})")
    print("source:", dict(src.most_common()))

    print("\n## 2. Phủ định (negation)")
    print(f"final user-turn chứa phủ định : {neg_final} ({pct(neg_final)})")
    print(f"gold chứa phủ định            : {neg_gold} ({pct(neg_gold)})")
    if neg_final:
        print(f"  trong số final có phủ định → gold GIỮ : {neg_final_kept} "
              f"({pct(neg_final_kept, neg_final)})")
        print(f"  trong số final có phủ định → gold RỤNG: {neg_final_dropped} "
              f"({pct(neg_final_dropped, neg_final)})")

    print("\n## 3. Đa mệnh đề (compound)")
    print(f"gold có cue đa mệnh đề (nhưng/rồi/, …): {compound_gold} ({pct(compound_gold)})")

    print("\n## 4. Khoảng cách khôi phục slot (chỉ context_required) — GIẢ THUYẾT GỐC")
    crn = ctx_req[True]
    for k in ["0 (none)", "1 (liền kề)", "2", "≥3 (xa)"]:
        print(f"  max restore distance = {k:12s}: {far_restore[k]} ({pct(far_restore[k], crn)})")
    far = far_restore["2"] + far_restore["≥3 (xa)"]
    print(f"→ slot khôi phục từ lượt KHÔNG liền kề (≥2): {far} ({pct(far, crn)} của context_required)")

    print("\n## 5. Phủ định theo domain (top theo tỉ lệ final có phủ định)")
    rows = sorted(((d, c[0], c[1]) for d, c in by_domain_neg.items() if c[1] >= 50),
                  key=lambda x: x[1] / x[2], reverse=True)[:12]
    for d, k, tot in rows:
        print(f"  {d:16s} {k:5d}/{tot:5d}  ({100*k/tot:.1f}%)")

    if restore_examples:
        print("\n## 6. Ví dụ slot khôi phục từ lượt xa (≥3)")
        for fu, gold in restore_examples[:6]:
            print(f"  final: {fu[:70]}")
            print(f"  gold : {gold[:90]}\n")


if __name__ == "__main__":
    main()
