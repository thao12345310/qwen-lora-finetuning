"""Migrate REWRITE-only LF data → REWRITE_AND_CLASSIFY in place (cheap, no LLM).

Dữ liệu cũ (vd data/train/dialogues_train_api.jsonl, các file bench) ở format
rewrite-only: tag <REWRITE>, system prompt cũ, gpt output {"rewrite_message": ...}.
Task mới cần: tag <REWRITE_AND_CLASSIFY>, system prompt mới (có định nghĩa
online/offline), output {"rewrite_message", "domain"}.

Vì MỌI nguồn cũ đều là tác vụ điều khiển/nav/media → domain = "offline". Script này:
  1. Thay system message bằng SYSTEM_PROMPT_FOR_TRAINING mới (nếu --system).
  2. Đổi tiền tố tag ở lượt human cuối: "<REWRITE>\\n" → "<REWRITE_AND_CLASSIFY>\\n".
  3. Thêm "domain": <label> vào gpt output JSON (giữ nguyên rewrite_message).
  4. Thêm meta.online_offline = <label>.

Mặc định label = "offline". Dùng cho nguồn online đã có sẵn nhãn thì truyền
--label online (hiếm — nguồn online nên sinh trực tiếp bằng generate_online.py).

Idempotent: chạy lại trên file đã migrate không làm hỏng (nhận diện tag mới).

Ví dụ:
    python -m src.data.migrate_to_classify \\
        --input data/train/dialogues_train_api.jsonl --inplace --system
    python -m src.data.migrate_to_classify \\
        --input data/bench/dialogues_bench_browser.jsonl \\
        --output data/bench/dialogues_bench_browser.jsonl --system
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from src.data.prompts import REWRITE_TAG, SYSTEM_PROMPT_FOR_TRAINING
except ModuleNotFoundError:  # support flat `python src/data/migrate_to_classify.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompts import REWRITE_TAG, SYSTEM_PROMPT_FOR_TRAINING

OLD_TAG = "<REWRITE>"
NEW_TAG = REWRITE_TAG  # "<REWRITE_AND_CLASSIFY>"


def migrate_record(rec: dict, label: str, set_system: bool) -> dict:
    convs = rec.get("conversations")
    if not isinstance(convs, list) or not convs:
        return rec

    for c in convs:
        if c.get("from") == "system" and set_system:
            c["value"] = SYSTEM_PROMPT_FOR_TRAINING

    # Retag the final human turn (only the leading tag prefix; keep idempotent).
    for c in reversed(convs):
        if c.get("from") == "human":
            v = c.get("value", "")
            if v.startswith(NEW_TAG + "\n"):
                break  # already migrated
            if v.startswith(OLD_TAG + "\n"):
                c["value"] = NEW_TAG + "\n" + v[len(OLD_TAG) + 1:]
            break

    # Add domain to the final gpt output JSON.
    last = convs[-1]
    if last.get("from") == "gpt":
        try:
            obj = json.loads(last["value"])
        except (json.JSONDecodeError, TypeError, ValueError):
            obj = None
        if isinstance(obj, dict) and "rewrite_message" in obj:
            obj.setdefault("domain", label)
            last["value"] = json.dumps(obj, ensure_ascii=False)

    meta = rec.setdefault("meta", {})
    meta.setdefault("online_offline", label)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--inplace", action="store_true", help="overwrite --input")
    ap.add_argument("--label", default="offline", choices=["offline", "online"],
                    help="domain label to stamp (default offline; old data is all offline)")
    ap.add_argument("--system", action="store_true",
                    help="also replace the system message with the new training prompt")
    args = ap.parse_args()

    out = args.input if args.inplace else args.output
    if out is None:
        raise SystemExit("provide --output or --inplace")

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    migrated = [migrate_record(r, args.label, args.system) for r in rows]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in migrated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Migrated {len(migrated)} records ({args.label}) → {out}")


if __name__ == "__main__":
    main()
