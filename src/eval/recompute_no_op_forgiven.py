"""A4 — recompute negation/command metrics post-hoc, forgiving no_op clause-drops.

NO re-judging: the per-flag judge verdicts already live in the preds_*.jsonl
(negation_ok / slots_complete / …). We only re-aggregate them under a corrected
rubric: a negation clause classified `no_op` in negation_types.jsonl is, by
definition, non-load-bearing — omitting it changes neither the tool-call set nor
the end-state — so the negation requirement for that sample is vacuously
satisfied and must not be penalized.

Forgiveness (applied identically to BOTH models, judge stays frozen gpt-oss):
  no_op sample  ->  negation_ok := 1     (primary, defensible)
  variant +slot ->  also slots_complete := 1   (reported separately; looser, since
                    the dropped no_op clause is the most common missing 'slot')

We first reproduce the ORIGINAL numbers to validate the pipeline, then forgive.

Run: /opt/homebrew/bin/python3.11 -m src.eval.recompute_no_op_forgiven
"""
from __future__ import annotations
import json
from pathlib import Path
from src.eval.analyze_train_distribution import NEG, content_tokens

DIR = Path("data/bench/eval_results/full_gptoss")
PREDS = {
    "LoRA (vi-rewriter)": DIR / "preds_vi-rewriter.jsonl",
    "Baseline (Qwen2.5-1.5B)": DIR / "preds_Qwen_Qwen2.5-1.5B-Instruct.jsonl",
}
NEG_TYPES = DIR / "negation_types.jsonl"
FLAGS = ("intent_ok", "slots_complete", "no_hallucination", "negation_ok")


def load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def score(r):  # command_acc unit: all four flags must hold
    return 1 if all(r[f] == 1 for f in FLAGS) else 0


def pred_text(r):
    p = r.get("pred", r.get("pred_text", ""))
    if isinstance(p, str):
        try:
            p = json.loads(p).get("rewrite_message", p)
        except Exception:
            pass
    return str(p)


# Generic action verbs — shared by almost every positive command, so they can't
# distinguish "inverted the target" from "issued an unrelated command". Strip them
# from the negated payload so matching keys on the TARGET object only
# (e.g. 'auto pilot', 'lane assist', 'đèn trần'), not on 'bật'.
ACTION_VERBS = set("bật tắt mở đóng phát dùng gọi nhắn gửi kích hoạt chuyển đặt giữ "
                   "can thiệp sử khóa mở_khóa bấm chỉnh để".split())


def negated_payload(gold: str) -> set[str]:
    """DISTINGUISHING target tokens of the negated clause: everything after the
    first negation cue, minus (a) generic action verbs and (b) tokens that also
    appear in the positive part before the cue. The remainder uniquely identifies
    the forbidden target (e.g. 'auto pilot', 'lane assist', 'trần', 'sau', 'gốc') —
    a shared noun like 'đèn'/'bản'/'cảnh báo' is NOT distinguishing and is dropped."""
    m = NEG.search(gold)
    if not m:
        return set()
    before = content_tokens(gold[:m.start()])
    after = content_tokens(gold[m.end():])
    return (after - before) - ACTION_VERBS


def is_clean_drop(r) -> bool:
    """A no_op forgiveness is only valid when the model OMITTED the negated target.
    If pred still names the distinguishing target (inverts 'Không bật Lane Assist'
    into 'Bật Lane Assist', or issues 'tắt khóa trẻ em'), it touched a forbidden
    control — a real error, not a harmless drop — so do NOT forgive."""
    payload = negated_payload(r["gold"])
    if not payload:
        return True
    return not (payload & content_tokens(pred_text(r)))


def recompute(rows, no_op_idx, forgive_slot=False, strict=True):
    """Return dict of metrics after forgiving no_op clause-drops.
    strict=True  -> forgive only CLEAN drops (model didn't touch the negated target).
    strict=False -> blanket forgive every no_op sample (overcounts inversions)."""
    neg_ok, slot, sc = [], [], []
    neg_ok_negpat, sc_negpat = [], []
    forgiven = 0
    for i, r in enumerate(rows):
        n_ok, s_ok = r["negation_ok"], r["slots_complete"]
        if i in no_op_idx and (not strict or is_clean_drop(r)):
            if n_ok == 0:
                forgiven += 1
            n_ok = 1
            if forgive_slot:
                s_ok = 1
        r2 = {**r, "negation_ok": n_ok, "slots_complete": s_ok}
        neg_ok.append(n_ok)
        slot.append(s_ok)
        sc.append(score(r2))
        if r["meta"].get("pattern") == "negation":
            neg_ok_negpat.append(n_ok)
            sc_negpat.append(score(r2))
    return {
        "negation_preservation": mean(neg_ok),
        "slot_completeness": mean(slot),
        "command_acc": mean(sc),
        "neg_pattern_command_acc": mean(sc_negpat),
        "neg_pattern_negation_ok": mean(neg_ok_negpat),
        "_forgiven": forgiven,
    }


def main():
    neg_types = {json.loads(l)["idx"]: json.loads(l)["type"]
                 for l in NEG_TYPES.open(encoding="utf-8")}
    no_op_idx = {i for i, t in neg_types.items() if t == "no_op"}
    print(f"negation_types: {len(neg_types)} classified, no_op={len(no_op_idx)}\n")

    results = {}
    for name, path in PREDS.items():
        rows = load(path)
        assert len(rows) == 1044, f"{name}: {len(rows)} rows"
        orig = recompute(rows, set())                              # no forgiveness
        forg = recompute(rows, no_op_idx)                          # strict clean-drop
        forg_slot = recompute(rows, no_op_idx, forgive_slot=True)  # + slot, strict
        blanket = recompute(rows, no_op_idx, strict=False)         # blanket (overcounts)
        results[name] = (orig, forg, forg_slot, blanket)
        n_clean = sum(1 for i, r in enumerate(rows)
                      if i in no_op_idx and r["negation_ok"] == 0 and is_clean_drop(r))
        n_dirty = sum(1 for i, r in enumerate(rows)
                      if i in no_op_idx and r["negation_ok"] == 0 and not is_clean_drop(r))
        print(f"{name}: no_op negation_ok=0 → clean-drop(forgive)={n_clean}, "
              f"inversion(keep)={n_dirty}")
    print()

    def row(metric, label):
        cells = []
        for name in PREDS:
            o, f, fs, bl = results[name]
            cells.append(f"{100*o[metric]:5.1f} → {100*f[metric]:5.1f}")
        return f"  {label:36s} " + " | ".join(cells)

    print("Forgiveness = STRICT (clean-drop only). Judge frozen gpt-oss, applied to both.\n")
    print(f"{'':38s} {'LoRA orig→forgiven':>21s} | {'Baseline orig→forgiven':>21s}")
    print(row("neg_pattern_negation_ok", "Negation Preserv. (negation pattern) ★"))
    print(row("negation_preservation", "negation_ok (all 1044, mostly trivial)"))
    print(row("command_acc", "Command Acc (all 1044)"))
    print(row("neg_pattern_command_acc", "  ↳ command_acc (negation pattern)"))
    print(row("slot_completeness", "Slot Completeness (all 1044)"))

    print("\n[strict vs blanket — negation pattern negation_ok]")
    for name in PREDS:
        o, f, fs, bl = results[name]
        print(f"  {name:24s} orig {100*o['neg_pattern_negation_ok']:.1f} | "
              f"strict {100*f['neg_pattern_negation_ok']:.1f} | "
              f"blanket {100*bl['neg_pattern_negation_ok']:.1f}")

    # validation vs metrics_summary.json / REPORT.md
    lo, ba = results["LoRA (vi-rewriter)"][0], results["Baseline (Qwen2.5-1.5B)"][0]
    print("\n[validation — original must match REPORT.md]")
    print(f"  Negation(pattern) orig  LoRA={100*lo['neg_pattern_negation_ok']:.1f} (exp 74.8) | "
          f"base={100*ba['neg_pattern_negation_ok']:.1f} (exp 84.9)")
    print(f"  Command(all)      orig  LoRA={100*lo['command_acc']:.1f} (exp 65.9) | "
          f"base={100*ba['command_acc']:.1f} (exp 52.8)")

    # corrected metrics, machine-readable
    corrected = {
        name: {
            "negation_pattern_negation_ok": {
                "original": results[name][0]["neg_pattern_negation_ok"],
                "no_op_forgiven_strict": results[name][1]["neg_pattern_negation_ok"],
                "no_op_forgiven_blanket": results[name][3]["neg_pattern_negation_ok"],
            },
            "command_acc_all": {
                "original": results[name][0]["command_acc"],
                "no_op_forgiven_strict": results[name][1]["command_acc"],
            },
            "forgiven_clean_drops": results[name][1]["_forgiven"],
        }
        for name in PREDS
    }
    (DIR / "A4_corrected_metrics.json").write_text(
        json.dumps({"forgiveness": "strict clean-drop only; judge frozen gpt-oss",
                    "models": corrected}, ensure_ascii=False, indent=2), encoding="utf-8")

    # dump per-sample forgiven flips for audit
    out = DIR / "no_op_forgiven_flips.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for name, path in PREDS.items():
            for i, r in enumerate(load(path)):
                if i in no_op_idx and r["negation_ok"] == 0:
                    fh.write(json.dumps({"model": name, "idx": i, "gold": r["gold"],
                                         "pred": r.get("pred", r.get("pred_text", "")),
                                         "slots_complete": r["slots_complete"],
                                         "reason": r.get("reason", "")[:200]},
                                        ensure_ascii=False) + "\n")
    print(f"\n→ forgiven flips (negation_ok 0→1 on no_op) dumped to {out}")


if __name__ == "__main__":
    main()
