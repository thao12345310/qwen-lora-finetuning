"""Generate and append a confirmation benchmark slice using Ollama Cloud.

The slice targets the v2.1 blind spot:
assistant proposes a concrete action -> final user only says a minimal affirmation
("ờ", "ừm", "ok", ...) -> gold rewrite is the proposed action with all slots resolved.

Default model: gpt-oss:120b-cloud via Ollama's OpenAI-compatible endpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.data.build_benchmark import BenchSample, to_lf_record, validate_sample
from src.data.prompts import SYSTEM_PROMPT_FOR_TRAINING

BENCH = Path("data/bench/dialogues_bench_browser.jsonl")
SIDECAR = Path("data/bench/dialogues_bench_confirmation_ollama.jsonl")
TRAIN_PATHS = [
    Path("data/processed/train.jsonl"),
    Path("data/processed/patch_b2b3.jsonl"),
    Path("data/processed/patch_b4.jsonl"),
]

AFFIRMS = [
    "ờ", "ừ", "ừm", "uhm", "ok", "oki", "okê", "vâng", "dạ", "được",
    "đúng", "chuẩn", "đúng rồi", "ừ nhỉ", "ờ được", "ừ ok", "ừm đúng",
]
DOMAINS = [
    "navigation", "climate", "music", "calling", "messaging", "charging",
    "smart_home", "driver_assist", "vehicle",
]
ACTION_RE = re.compile(
    r"\b(bật|tắt|mở|đóng|dẫn|chỉ|tìm|gọi|nhắn|gửi|phát|đặt|lên lịch|"
    r"kết nối|khóa|sạc|giảm|tăng|chuyển|hẹn|kiểm tra|lưu|thêm|bỏ|"
    r"kích hoạt|báo|nhắc)\b",
    re.IGNORECASE,
)
CJK_RE = re.compile(r"[一-鿿가-힯]")
CONCRETE_RE = re.compile(
    r"(\d|[\"'“”]|\b(?:VinFast|EVN|V-GREEN|Tesla|Spotify|Google Maps|Apple Maps|"
    r"YouTube Music|Zing MP3|Zalo|Bluetooth|CarPlay|VietMap|Cruise Control|"
    r"Lane Assist|Auto Pilot|Lotte|Aeon|Vincom|Crescent|Sala|Landmark|"
    r"Nguyễn|Trần|Lê|Phạm|Hoàng|Hồ|Quận|Hà Nội|Sài Gòn|Đà Nẵng|"
    r"Lan|Minh|Hương|Nam|Thành|Trang|Tuấn|Mai|Linh|Vy|Hà|Phúc|Đức)\b)",
    re.IGNORECASE,
)
POLARITY_CLASH_RE = re.compile(r"\b(giảm)\b.{0,40}\b(lên)\b|\b(tăng)\b.{0,40}\b(xuống)\b", re.IGNORECASE)
GENERIC_BAD = [
    "gần nhất", "gần đây", "trạm xăng", "bãi đỗ", "trung tâm thương mại",
    "công viên trung tâm", "địa chỉ nhà cô ấy", "liên hệ đã lưu",
    "liên hệ đã lưu gần nhất", "anh bạn", "mẹ bạn", "bus station",
    "ga cuối tuần", "ngay bây giờ",
]
STOPWORDS = {
    "em", "anh", "chị", "ạ", "nhé", "nha", "được", "không", "cho", "giúp",
    "tôi", "mình", "bạn", "với", "và", "là", "có", "muốn", "không", "đi",
    "luôn", "ngay", "thì", "này", "đó", "ở", "tại", "vào", "ra", "lên",
    "xuống", "một", "cái", "nhỉ",
}

SYSTEM = """Bạn là chuyên gia tạo benchmark tiếng Việt cho task rewrite hội thoại trong xe.

Hãy sinh dữ liệu KHÓ nhưng sạch. Chỉ trả về JSON object hợp lệ, không markdown.
Mọi mẫu phải thuộc pattern confirmation:
- Lượt bot ngay trước cuối phải ĐỀ XUẤT một hành động cụ thể có slot rõ.
- Lượt user cuối chỉ là một lời đồng ý tối giản.
- rewrite chuẩn phải là hành động được đề xuất, đủ slot, không bịa thêm.
"""

FEWSHOTS: list[dict[str, Any]] = [
    {
        "turns": [
            {"role": "user", "content": "Kính sau mờ quá."},
            {"role": "bot", "content": "Em bật sấy kính sau mức nhẹ trong 10 phút nhé?"},
            {"role": "user", "content": "ừm"},
        ],
        "rewrite": "Bật sấy kính sau mức nhẹ trong 10 phút.",
        "domain": "vehicle",
        "rationale": "User chỉ xác nhận, cần bind vào đề xuất bật sấy kính sau của bot.",
    },
    {
        "turns": [
            {"role": "user", "content": "Tôi sắp tới khu đô thị Sala rồi."},
            {"role": "bot", "content": "Em dẫn đường vào cổng Nguyễn Cơ Thạch của khu đô thị Sala nhé?"},
            {"role": "user", "content": "ok"},
        ],
        "rewrite": "Dẫn đường vào cổng Nguyễn Cơ Thạch của khu đô thị Sala.",
        "domain": "navigation",
        "rationale": "Gold phải lấy đúng cổng được bot đề xuất, không tự thêm điểm đến khác.",
    },
    {
        "turns": [
            {"role": "user", "content": "Pin còn 18% thôi."},
            {"role": "bot", "content": "Em tìm trạm sạc EVN công suất 120 kW gần cầu Sài Gòn nhé?"},
            {"role": "user", "content": "dạ"},
        ],
        "rewrite": "Tìm trạm sạc EVN công suất 120 kW gần cầu Sài Gòn.",
        "domain": "charging",
        "rationale": "Lời đồng ý phải giữ brand, công suất và vị trí trong đề xuất.",
    },
    {
        "turns": [
            {"role": "user", "content": "Mai nhớ nhắc tôi việc bảo dưỡng xe."},
            {"role": "bot", "content": "Em đặt nhắc lịch bảo dưỡng xe lúc 8 giờ sáng mai nhé?"},
            {"role": "user", "content": "chuẩn"},
        ],
        "rewrite": "Đặt nhắc lịch bảo dưỡng xe lúc 8 giờ sáng mai.",
        "domain": "vehicle",
        "rationale": "Final turn không có nội dung lệnh, toàn bộ intent nằm ở đề xuất của bot.",
    },
]


def strip_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_payload(text: str) -> dict[str, Any]:
    text = strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def golds_from(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for c in rec.get("conversations", []):
            if c.get("from") == "gpt":
                try:
                    out.add(json.loads(c["value"])["rewrite_message"])
                except Exception:
                    pass
    return out


def final_user(turns: list[dict[str, str]]) -> str:
    return turns[-1]["content"].strip()


def prev_bot(turns: list[dict[str, str]]) -> str:
    for t in reversed(turns[:-1]):
        if t["role"] == "bot":
            return t["content"].strip()
    return ""


def meaningful_tokens(text: str) -> set[str]:
    text = re.sub(r"[^\wÀ-ỹ]+", " ", text.lower(), flags=re.UNICODE)
    return {t for t in text.split() if len(t) >= 3 and t not in STOPWORDS}


def validate_confirmation(raw: dict[str, Any], forbidden_golds: set[str]) -> tuple[BenchSample | None, str]:
    raw = dict(raw)
    raw["pattern"] = "confirmation"
    raw["context_required"] = True
    sample, reason = validate_sample(raw, "confirmation", min_turns=3)
    if sample is None:
        return None, reason
    if sample.domain not in DOMAINS:
        return None, "bad_domain"
    if final_user(sample.turns).lower() not in AFFIRMS:
        return None, "final_not_minimal_affirm"
    bot = prev_bot(sample.turns)
    if not bot:
        return None, "missing_prev_bot"
    if not ("?" in bot or "nhé" in bot.lower() or "không" in bot.lower() or "được không" in bot.lower()):
        return None, "prev_bot_not_proposal"
    if not ACTION_RE.search(bot) or not ACTION_RE.search(sample.rewrite):
        return None, "missing_action_verb"
    if CJK_RE.search(sample.rewrite):
        return None, "cjk_in_rewrite"
    low_rewrite = sample.rewrite.lower()
    if any(bad in low_rewrite for bad in GENERIC_BAD):
        return None, "generic_slot"
    if POLARITY_CLASH_RE.search(sample.rewrite):
        return None, "polarity_clash"
    if not CONCRETE_RE.search(sample.rewrite):
        return None, "missing_concrete_slot"
    if sample.rewrite in forbidden_golds:
        return None, "gold_overlap_train_or_bench"

    # Loose grounding check: gold should share concrete words with the proposed action.
    # This catches common bad generations that agree but invent a new place/person.
    overlap = meaningful_tokens(bot) & meaningful_tokens(sample.rewrite)
    if len(overlap) < 2:
        return None, "weak_grounding_to_proposal"
    return sample, "ok"


def build_prompt(batch_size: int, seed: int, domain_hint: list[str]) -> str:
    return json.dumps({
        "task": "Sinh benchmark confirmation cho Vietnamese in-car dialogue rewrite.",
        "batch_size": batch_size,
        "seed": seed,
        "domain_hint": domain_hint,
        "allowed_final_user_values": AFFIRMS,
        "domains": DOMAINS,
        "rules": [
            "Mỗi sample có đúng 3-5 turns, kết thúc bằng role=user.",
            "Lượt bot ngay trước cuối phải là đề xuất hành động cụ thể có slot rõ.",
            "Final user content phải EXACTLY là một giá trị trong allowed_final_user_values.",
            "rewrite là một câu tiếng Việt độc lập, bind đúng hành động bot đề xuất.",
            "Không thêm địa danh, người, app, brand, con số, thời gian ngoài dialogue.",
            "Mỗi đề xuất phải có slot cụ thể: tên người/địa điểm/app/brand/bài hát/nội dung tin nhắn/con số/thời gian.",
            "Tránh cụm generic như gần nhất, gần đây, liên hệ đã lưu, trung tâm thương mại, công viên trung tâm, địa chỉ nhà cô ấy.",
            "Ưu tiên tên riêng Việt Nam thật: Nguyễn Huệ, Lotte Mall Tây Hồ, chị Lan Anh, Spotify, Zalo, VinFast, EVN.",
            "Không copy few-shot; tạo scenario mới, đa dạng domain.",
            "Không sinh true-abstain/filler khi bot chưa đề xuất hành động.",
        ],
        "schema": {
            "samples": [
                {
                    "turns": [
                        {"role": "user", "content": "..."},
                        {"role": "bot", "content": "...?"},
                        {"role": "user", "content": "ừm"},
                    ],
                    "rewrite": "...",
                    "domain": "navigation|climate|music|calling|messaging|charging|smart_home|driver_assist|vehicle",
                    "rationale": "1 câu giải thích ngắn",
                }
            ]
        },
        "few_shot_examples_do_not_copy": FEWSHOTS,
    }, ensure_ascii=False, indent=2)


def generate_batch(client: OpenAI, model: str, batch_size: int, seed: int, domain_hint: list[str]) -> list[dict[str, Any]]:
    extra: dict[str, Any] = {}
    if "gpt-oss" in model.lower():
        extra["reasoning_effort"] = "low"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(batch_size, seed, domain_hint)},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
        top_p=0.95,
        max_tokens=6000,
        **extra,
    )
    payload = extract_payload(resp.choices[0].message.content or "")
    samples = payload.get("samples", [])
    return [s for s in samples if isinstance(s, dict)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--sidecar", type=Path, default=SIDECAR)
    ap.add_argument("--model", default="gpt-oss:120b-cloud")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--max-waves", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace-existing", action="store_true",
                    help="Drop existing pattern=confirmation rows from --bench before appending")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    forbidden = golds_from(args.bench)
    for path in TRAIN_PATHS:
        forbidden |= golds_from(path)

    existing_records = []
    if args.bench.exists():
        existing_records = [json.loads(l) for l in args.bench.open(encoding="utf-8") if l.strip()]
    if args.replace_existing:
        before = len(existing_records)
        existing_records = [
            r for r in existing_records
            if r.get("meta", {}).get("pattern") != "confirmation"
        ]
        print(f"replace-existing: removed {before - len(existing_records)} confirmation rows")
    seen_keys = {
        (r["conversations"][-2]["value"], r["conversations"][-1]["value"])
        for r in existing_records
    }

    accepted: list[dict[str, Any]] = []
    reject = Counter()
    for wave in range(1, args.max_waves + 1):
        if len(accepted) >= args.target:
            break
        domain_hint = rng.sample(DOMAINS, k=4)
        seed = rng.randint(1, 10_000_000)
        print(f"wave {wave}: requesting {args.batch_size} ({len(accepted)}/{args.target}) domains={domain_hint}")
        try:
            raw_samples = generate_batch(client, args.model, args.batch_size, seed, domain_hint)
        except Exception as exc:
            reject[f"model_error:{type(exc).__name__}"] += 1
            print(f"  model error: {exc}")
            time.sleep(2)
            continue

        for raw in raw_samples:
            sample, reason = validate_confirmation(raw, forbidden)
            if sample is None:
                reject[reason] += 1
                continue
            rec = to_lf_record(sample, SYSTEM_PROMPT_FOR_TRAINING)
            rec["meta"]["pattern"] = "confirmation"
            rec["meta"]["context_required"] = True
            rec["meta"]["intent"] = "confirm_proposed"
            rec["meta"]["source"] = "ollama_gptoss_120b_cloud"
            rec["meta"]["generator_model"] = args.model
            key = (rec["conversations"][-2]["value"], rec["conversations"][-1]["value"])
            gold = json.loads(rec["conversations"][-1]["value"])["rewrite_message"]
            if key in seen_keys or gold in forbidden:
                reject["duplicate_or_forbidden_after_lf"] += 1
                continue
            seen_keys.add(key)
            forbidden.add(gold)
            accepted.append(rec)
            if len(accepted) >= args.target:
                break

    print(f"accepted={len(accepted)} target={args.target}")
    if reject:
        print("rejects:", dict(reject.most_common()))
    if len(accepted) < args.target:
        raise SystemExit(f"only accepted {len(accepted)}/{args.target}")
    accepted = accepted[:args.target]

    by_domain = Counter(r["meta"]["domain"] for r in accepted)
    by_affirm = Counter(
        r["conversations"][-2]["value"].replace("<REWRITE>\n", "").strip().lower()
        for r in accepted
    )
    print("domains:", dict(sorted(by_domain.items())))
    print("affirms:", dict(sorted(by_affirm.items())))

    if args.dry_run:
        return

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in accepted) + "\n",
        encoding="utf-8",
    )
    with args.bench.open("w", encoding="utf-8") as f:
        for r in existing_records + accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote sidecar: {args.sidecar}")
    print(f"appended benchmark: {len(existing_records)} + {len(accepted)} = {len(existing_records) + len(accepted)} -> {args.bench}")


if __name__ == "__main__":
    main()
