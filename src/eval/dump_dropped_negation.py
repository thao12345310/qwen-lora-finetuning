"""Extract train samples where the final user turn has a negation cue but the gold
rewrite drops it. Splits by a cheap heuristic so we can eyeball gold quality:

  paraphrased — gold likely re-expressed the negation positively
                (final "đừng dùng sạc nhanh" -> gold "sạc chậm/thường"); acceptable.
  dropped     — gold appears to have simply removed the constraint; suspect bad gold.

This is a triage view, not ground truth — it tells us roughly how much of the
57.5% drop is genuine gold rot (load_bearing dropped) vs benign rephrase.

Run: /opt/homebrew/bin/python3.11 -m src.eval.dump_dropped_negation [N_per_domain]
"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
from src.eval.analyze_train_distribution import parse, NEG, is_real_negation, content_tokens

TRAIN = Path("data/processed/train.jsonl")
OUT = Path("data/bench/eval_results/full_gptoss/train_dropped_negation.txt")

# If the negated head noun/verb survives in gold, it's likely a positive paraphrase
# of the same constraint rather than a true drop. We grab the 1-3 tokens right
# after the negation cue in the final turn and check their survival in gold.
def negated_payload(final_user: str) -> set[str]:
    toks = []
    for m in NEG.finditer(final_user):
        tail = final_user[m.end():m.end() + 40]
        toks += content_tokens(tail)
    return set(toks)


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    recs = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
    buckets = {"paraphrased": [], "dropped": []}
    by_dom = Counter()
    dom_dropped = Counter()

    for r in recs:
        _, final_user, gold = parse(r)
        if not is_real_negation(final_user) or is_real_negation(gold):
            continue  # only: final has real neg, gold lost the neg word
        dom = r.get("meta", {}).get("domain", "?")
        payload = negated_payload(final_user)
        g = content_tokens(gold)
        survived = payload & g
        kind = "paraphrased" if survived else "dropped"
        by_dom[dom] += 1
        if kind == "dropped":
            dom_dropped[dom] += 1
        buckets[kind].append((dom, final_user, gold, sorted(survived)))

    total = len(buckets["paraphrased"]) + len(buckets["dropped"])
    print(f"# Dropped-negation triage — {total} cases (final has neg, gold word-drops it)\n")
    print(f"paraphrased (payload survives in gold, likely OK): {len(buckets['paraphrased'])} "
          f"({100*len(buckets['paraphrased'])/total:.1f}%)")
    print(f"dropped     (payload gone, SUSPECT bad gold)     : {len(buckets['dropped'])} "
          f"({100*len(buckets['dropped'])/total:.1f}%)\n")
    print("## suspect-dropped by domain")
    for d, c in dom_dropped.most_common():
        print(f"  {d:16s} {c:4d}/{by_dom[d]:4d}")

    lines = []
    for kind in ("dropped", "paraphrased"):
        lines.append(f"\n{'='*70}\n{kind.upper()}  (n={len(buckets[kind])})\n{'='*70}")
        seen = Counter()
        for dom, fu, gold, surv in buckets[kind]:
            if seen[dom] >= per:
                continue
            seen[dom] += 1
            lines.append(f"[{dom}] survived={surv}")
            lines.append(f"  final: {fu}")
            lines.append(f"  gold : {gold}\n")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ {per} examples/domain/bucket written to {OUT}")


if __name__ == "__main__":
    main()
