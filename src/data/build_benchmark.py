"""Build a hard, high-quality benchmark for the Vietnamese dialogue rewrite task.

Calls GPT-4o (frontier) to generate dialogues across 8 hard patterns:
    1. pronoun_resolution    — anaphora ("đó", "người ấy", "cái thứ hai")
    2. irrelevant_context    — bot mentions a number user must NOT pull into rewrite
    3. multi_turn_slot       — slots scattered across 3-4 user turns, rewrite consolidates
    4. correction            — user changes/cancels mid-conversation
    5. code_switching        — mix English tokens in Vietnamese ("play Despacito", "Bluetooth")
    6. implicit_reference    — "nhà tôi", "công ty", "trường con" — slot resolved via world knowledge
    7. negation              — exclude / forbid a slot ("đừng đi qua cầu X", "trừ Y ra")
    8. compound_intent       — 2 intents in a single final turn ("đổi bài rồi giảm volume")

Pipeline:
    GPT-4o generate → heuristic validate (hallucination + leak checks)
        → gpt-4o-mini difficulty filter (reject samples mini solves easily)
        → cache + dedup → Llama Factory `conversations` output

Output:
    data/bench/dialogues_bench.jsonl  (Llama Factory `conversations` format)

Usage:
    export OPENAI_API_KEY=sk-...
    python -m src.data.build_benchmark
    python -m src.data.build_benchmark --samples-per-pattern 125 --batch-size 5

The script is resumable: per-pattern progress is cached in
`data/bench/.cache_<pattern>.jsonl`. Pre-existing cached samples (from earlier
runs) are loaded as-is without re-validation — only NEW generations go through
the tightened validators + difficulty filter.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from src.data.prompts import SYSTEM_PROMPT_FOR_TRAINING

load_dotenv()


META_SYSTEM = """Bạn là chuyên gia tạo dữ liệu huấn luyện tiếng Việt cho task rewrite hội thoại trong xe ô tô.

Bạn sẽ sinh dữ liệu BENCHMARK KHÓ — mục tiêu là làm model phải suy luận, không phải copy. Mỗi mẫu phải thật sự thử thách: slot ẩn, đại từ, thông tin nhiễu, hoặc thay đổi ý.

YÊU CẦU TUYỆT ĐỐI:
- Đa dạng domain trong cùng pattern (navigation, AC/climate, music, calling, messaging, charging, smart-home, driver-assist, vehicle-controls). Đừng tất cả về navigation.
- Slot phải cụ thể (tên người, địa điểm Hà Nội/Sài Gòn thật, bài hát Vpop thật, hãng EV thật, nhiệt độ hợp lý).
- Cách diễn đạt phải tự nhiên kiểu nói chuyện trong xe — có thể lược chủ ngữ, dùng tiếng nói nhanh, slang nhẹ.
- Câu rewrite mục tiêu PHẢI biến tấu opener — KHÔNG bao giờ luôn mở đầu bằng "Tôi muốn ...". Dùng đa dạng: "Hãy ...", "Làm ơn ...", "Bạn ... giúp tôi", "Mình muốn ...", "Cho tôi ...", câu mệnh lệnh ngắn gọn, v.v.
- Không lặp scenario/từ vựng giữa các mẫu trong cùng batch.

Trả về JSON đúng schema yêu cầu."""


PATTERN_PROMPTS = {
    "pronoun_resolution": """Pattern: PRONOUN RESOLUTION (giải đại từ).

User dùng đại từ chỉ trỏ ở lượt cuối (đó / đấy / cái này / cái thứ hai / người ấy / anh ta / chỗ đó / địa điểm đó / bài đó / số đó / hãng đó...). Model rewrite PHẢI resolve đại từ thành slot cụ thể từ lượt bot trước đó.

Phải có ÍT NHẤT 3 turn (user-bot-user). Lượt bot giữa cung cấp thông tin chứa slot.
Bạn có thể thử cả lượt 5 turn (user-bot-user-bot-user) với 2 slot, user pick 1 qua đại từ.

Ví dụ đại từ nâng cao: "cái thứ hai", "cái rẻ hơn", "địa điểm gần hơn", "người vừa nói tới"...
""",

    "irrelevant_context": """Pattern: IRRELEVANT CONTEXT FILTERING (lọc thông tin nhiễu).

Bot mention một con số / giá trị (nhiệt độ ngoài trời, giá xăng/điện, giờ, %, khoảng cách, tốc độ giới hạn...) ở turn giữa. Sau đó user request hành động với MỘT SLOT KHÁC HẲN. Model rewrite KHÔNG ĐƯỢC kéo nhầm số/giá trị bot mention.

Cấu trúc tiêu chuẩn: user hỏi info → bot trả lời có số → user yêu cầu hành động độc lập.

Cố gắng làm "nhiễu" gần với "tín hiệu" để model dễ nhầm — ví dụ: bot báo nhiệt độ ngoài 38 độ, user xin bật điều hoà 22 độ. Bot báo còn 15% pin, user xin đặt cruise 60 km/h.
""",

    "multi_turn_slot": """Pattern: MULTI-TURN SLOT TRACKING (gộp slot rải rác).

3 đến 4 user-turn, mỗi turn cung cấp ĐÚNG MỘT slot. Bot giữa các turn hỏi slot tiếp theo. Lượt user cuối thường rất ngắn (chỉ 1 token / 1 cụm). Rewrite PHẢI GỘP TẤT CẢ slot từ tất cả các turn user lại thành 1 câu.

Domain thường có slot dày: AC (nhiệt + chế độ + vùng), SMS (người + nội dung), nav (địa điểm + ưu tiên lộ trình + giờ rời), gọi điện (người + qua loa ngoài), nghe nhạc (nguồn + playlist + âm lượng), tìm sạc (hãng + tốc độ).

KHÔNG được skip slot — rewrite phải có TẤT CẢ slot user đã cung cấp.
""",

    "correction": """Pattern: CORRECTION & CANCELLATION (đổi ý hoặc huỷ).

Có 2 sub-type — sinh đều cả 2:
(a) CORRECTION: user yêu cầu X → bot xác nhận/đang xử lý → user nói "khoan, đổi sang Y" / "không phải X, là Y" / "đổi điểm đến thành Y" / "cho tôi đổi sang Y". Rewrite phản ánh **intent cuối** (Y), KHÔNG còn X.
(b) CANCELLATION: user yêu cầu X → bot xác nhận → user nói "thôi huỷ đi" / "không cần nữa" / "bỏ đi" / "thôi đừng làm nữa". Rewrite là yêu cầu HUỶ X.

Đảm bảo X và Y khác nhau ý nghĩa (đổi địa điểm, đổi bài hát, đổi liên lạc, đổi nhiệt độ...). Cancellation phải làm rõ là HUỶ chứ không phải hoàn thành.
""",

    "code_switching": """Pattern: CODE-SWITCHING (chèn từ tiếng Anh / thương hiệu / từ kỹ thuật).

Người Việt khi nói chuyện với xe rất hay mix tiếng Anh: tên bài hát, tên ca sĩ, brand (Spotify, Bluetooth, CarPlay, Tesla, VinFast), thuật ngữ (cruise control, auto pilot, lane assist, navigation, low-beam, defrost, voice memo), số (FM 99.9, channel 5).

User turn cuối CHỨA token tiếng Anh hoặc chuyển ngữ. Rewrite PHẢI giữ NGUYÊN viết hoa / phiên âm của các token đó (KHÔNG dịch sang tiếng Việt, KHÔNG bỏ).

Ví dụ:
- "Mở Spotify lên rồi play playlist Chill Vibes." → rewrite giữ "Spotify", "playlist Chill Vibes".
- "Bật auto pilot đi." → rewrite giữ "auto pilot".
- "Kết nối Bluetooth với iPhone của em." → rewrite giữ "Bluetooth", "iPhone".

Có thể kết hợp với pronoun (lượt trước bot nói brand, lượt cuối user dùng "cái đó") — nhưng rewrite vẫn phải resolve về tên brand TIẾNG ANH gốc.
""",

    "implicit_reference": """Pattern: IMPLICIT REFERENCE (slot ngầm, dựa world knowledge / context cá nhân).

User KHÔNG nêu tên cụ thể, dùng cụm sở hữu/quan hệ: "nhà tôi", "nhà mình", "công ty", "trường con", "cơ quan", "phòng gym của em", "siêu thị quen", "bác sĩ riêng", "mẹ tôi", "vợ", "anh hai", "đồng nghiệp Lan", "sếp"...

Có 2 sub-mode:
(a) RESOLVABLE: turn trước đó có info giúp resolve (vd: user nói "lưu nhà mình ở Trần Duy Hưng", sau đó "dẫn về nhà" → resolve "Trần Duy Hưng"). Rewrite phải resolve cụ thể.
(b) UNRESOLVABLE: không có info trước, rewrite GIỮ NGUYÊN cụm ngầm (vd "Dẫn về nhà tôi.", "Gọi cho mẹ tôi.") — KHÔNG được bịa địa chỉ / số điện thoại.

Sinh tỉ lệ ~50/50 cả 2 sub-mode. Mỗi mẫu rationale phải nói rõ sub-mode nào.
""",

    "negation": """Pattern: NEGATION & EXCLUSION (loại trừ slot).

User yêu cầu hành động NHƯNG kèm điều kiện loại trừ: "đừng đi qua cầu Chương Dương", "tránh đường Nguyễn Trãi", "trừ bài của Coldplay ra", "không bật điều hoà lạnh", "không gọi cho Lan", "đừng cài đặt vào lịch", "không qua loa ngoài"...

Có 2 sub-mode:
(a) ACTION + EXCLUSION: positive intent + 1 slot bị loại ("dẫn tới sân bay, đừng đi cao tốc", "phát nhạc Vpop, trừ Sơn Tùng").
(b) PURE NEGATION: huỷ/cấm hẳn ("đừng nhắc nữa", "tắt thông báo từ Zalo", "không gọi cho ai cả").

Rewrite PHẢI giữ rõ negation — KHÔNG được rút gọn rồi mất "đừng / trừ / không". Test xem model có hiểu polarity không.

Có thể thêm distractor: bot turn giữa gợi ý điều bị loại (vd bot đề xuất "đi qua cầu Chương Dương cho nhanh", user negation lại).
""",

    "compound_intent": """Pattern: COMPOUND INTENT (2 intent trong cùng 1 lượt cuối).

Lượt cuối user chứa 2 hành động độc lập nhưng nói liền: "đổi bài đi rồi giảm volume xuống", "gọi cho Lan xong nhắn tin cho mẹ luôn", "tăng điều hoà lên 25 độ và bật chế độ tự động", "đặt báo thức 6h sáng và đóng cửa garage giúp anh".

Rewrite PHẢI giữ CẢ 2 intent với CẢ 2 slot. Tốt nhất nối bằng "và" / "rồi" / "sau đó". KHÔNG được bỏ 1 trong 2.

Có thể có pronoun trong intent thứ 2 ("phát Despacito, tăng âm lượng cho nó lên 80") — model phải resolve "nó" = bài Despacito.

2 intent KHÔNG cần cùng domain (vd 1 calling + 1 climate). Đảm bảo cả 2 đều actionable.
""",
}


DOMAIN_WEIGHTS = {
    "navigation": 1.0,
    "climate": 1.0,
    "music": 1.0,
    "calling": 1.0,
    "messaging": 2.0,
    "charging": 2.5,
    "smart_home": 3.0,
    "vehicle": 3.0,
    "driver_assist": 3.0,
}


def pick_batch_domains(k: int = 3) -> list[str]:
    domains = list(DOMAIN_WEIGHTS.keys())
    weights = list(DOMAIN_WEIGHTS.values())
    return random.choices(domains, weights=weights, k=k)


SCHEMA_INSTRUCTIONS = """Trả về JSON object đúng schema:

{
  "samples": [
    {
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "bot",  "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "rewrite": "câu rewrite chuẩn cho lượt user cuối",
      "domain": "navigation|climate|music|calling|messaging|charging|smart_home|driver_assist|vehicle",
      "rationale": "1 câu giải thích tại sao ca này khó"
    }
  ]
}

LƯU Ý:
- turns: list xen kẽ user/bot, BẮT BUỘC kết thúc bằng role="user".
- rewrite: 1 câu duy nhất, KHÔNG có dấu ngoặc kép thừa, KHÔNG ghi {"rewrite_message"...}.
- domain: chọn đúng 1 trong list.
- rationale: ngắn gọn 1 câu tiếng Việt."""


@dataclass
class BenchSample:
    pattern: str
    turns: list[dict]
    rewrite: str
    domain: str
    rationale: str


reject_counter: dict[str, int] = {}
filter_counter: dict[str, int] = {}


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[\s,.!?;:\"'()\-]+", " ", text)
    return text.strip()


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", ".") for m in NUMBER_RE.finditer(text)}


def validate_sample(s: dict, pattern: str) -> tuple[BenchSample | None, str]:
    """Return (sample, reason). Sample is None when rejected."""
    if not isinstance(s, dict):
        return None, "not_dict"
    turns = s.get("turns")
    rewrite = s.get("rewrite", "").strip()
    domain = s.get("domain", "").strip()
    rationale = s.get("rationale", "").strip()
    if not isinstance(turns, list) or len(turns) < 3:
        return None, "too_few_turns"
    if not rewrite or not domain:
        return None, "missing_field"
    if not (5 <= len(rewrite) <= 240):
        return None, "rewrite_length"

    cleaned: list[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            return None, "turn_not_dict"
        role = t.get("role")
        content = (t.get("content") or "").strip()
        if role not in ("user", "bot") or not content:
            return None, "bad_role_or_empty"
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "user":
        return None, "not_ending_user"

    last_user = cleaned[-1]["content"]
    if len(last_user.split()) > 25:
        return None, "final_user_too_long"
    if _norm(last_user) == _norm(rewrite):
        return None, "rewrite_equals_final_turn"

    dialogue_numbers = _numbers(" ".join(t["content"] for t in cleaned))
    rewrite_numbers = _numbers(rewrite)
    if not rewrite_numbers.issubset(dialogue_numbers):
        return None, "hallucinated_numbers"

    if pattern == "irrelevant_context":
        user_numbers = _numbers(" ".join(t["content"] for t in cleaned if t["role"] == "user"))
        bot_only = dialogue_numbers - user_numbers
        if bot_only & rewrite_numbers:
            return None, "leaked_bot_number"

    if pattern == "negation":
        if not re.search(r"\b(đừng|trừ|không|tránh|cấm|ngoại trừ|bỏ qua)\b", rewrite.lower()):
            return None, "negation_marker_missing"

    if pattern == "compound_intent":
        if not re.search(r"\b(và|rồi|sau đó|đồng thời|cùng lúc|xong)\b", rewrite.lower()):
            return None, "compound_connector_missing"

    return BenchSample(
        pattern=pattern, turns=cleaned, rewrite=rewrite,
        domain=domain, rationale=rationale,
    ), "ok"


def to_lf_record(sample: BenchSample, system_prompt: str) -> dict:
    """Convert internal sample → Llama Factory `conversations` schema."""
    conversations = [{"from": "system", "value": system_prompt}]
    history = sample.turns[:-1]
    for t in history:
        from_ = "human" if t["role"] == "user" else "gpt"
        conversations.append({"from": from_, "value": t["content"]})
    last_user = sample.turns[-1]
    conversations.append({
        "from": "human",
        "value": f"<REWRITE>\n{last_user['content']}",
    })
    answer = json.dumps({"rewrite_message": sample.rewrite}, ensure_ascii=False)
    conversations.append({"from": "gpt", "value": answer})
    return {
        "conversations": conversations,
        "meta": {
            "pattern": sample.pattern,
            "domain": sample.domain,
            "rationale": sample.rationale,
            "user_turns": sum(1 for t in sample.turns if t["role"] == "user"),
            "total_turns": len(sample.turns),
        },
    }


WEAK_BASELINE_SYSTEM = (
    "Bạn là model rewrite hội thoại tiếng Việt cho trợ lý xe ô tô. "
    "Nhiệm vụ: biến câu nói cuối của user thành MỘT câu yêu cầu độc lập, "
    "rõ ràng, giữ nguyên ý định, dùng thông tin từ các lượt trước nếu cần. "
    "Chỉ trả về câu rewrite, KHÔNG giải thích, KHÔNG JSON, KHÔNG markdown."
)


WEAK_JUDGE_SYSTEM = """Bạn là giám khảo đánh giá câu rewrite tiếng Việt cho task hội thoại trợ lý xe.

Nhận: dialogue (history), gold (rewrite chuẩn), prediction (model dự đoán).

ĐÚNG (score=1) khi prediction:
- Bảo toàn tất cả slot quan trọng có trong gold (tên người, địa điểm, số, nhiệt độ, bài hát, hãng, chế độ).
- Cùng intent với gold (bật/tắt/đổi/huỷ/thêm/loại trừ).
- Khác cách diễn đạt / opener / dấu câu — vẫn coi là đúng.

SAI (score=0): thiếu slot, sai intent, thêm thông tin bịa, mất polarity (negation).

Trả về JSON: {"score": 0 hoặc 1, "reason": "1 câu ngắn"}."""


async def weak_predict(
    client: AsyncOpenAI,
    model: str,
    dialogue_str: str,
    sem: asyncio.Semaphore,
    max_retries: int = 2,
) -> str | None:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": WEAK_BASELINE_SYSTEM},
                        {"role": "user", "content": dialogue_str},
                    ],
                    max_tokens=120,
                    temperature=0.1,
                    top_p=0.9,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def judge_match(
    client: AsyncOpenAI,
    model: str,
    dialogue_str: str,
    gold: str,
    pred: str,
    sem: asyncio.Semaphore,
    max_retries: int = 2,
) -> int | None:
    user = (
        f"dialogue:\n{dialogue_str}\n\n"
        f"gold:\n{gold}\n\n"
        f"prediction:\n{pred}\n\n"
        "Đánh giá theo schema."
    )
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": WEAK_JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=120,
                )
                return int(json.loads(resp.choices[0].message.content).get("score", 0))
            except Exception:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def filter_difficulty(
    client: AsyncOpenAI,
    weak_model: str,
    judge_model: str,
    samples: list[BenchSample],
    sem: asyncio.Semaphore,
) -> list[BenchSample]:
    """Keep only samples the weak baseline gets WRONG."""
    if not samples:
        return []

    dialogues = [
        "\n".join(f"{t['role']}: {t['content']}" for t in s.turns)
        for s in samples
    ]
    preds = await asyncio.gather(*[
        weak_predict(client, weak_model, d, sem) for d in dialogues
    ])

    judge_tasks = []
    for s, d, p in zip(samples, dialogues, preds):
        if p is None:
            judge_tasks.append(asyncio.sleep(0, result=None))
        else:
            judge_tasks.append(judge_match(client, judge_model, d, s.rewrite, p, sem))
    scores = await asyncio.gather(*judge_tasks)

    kept: list[BenchSample] = []
    for s, sc in zip(samples, scores):
        if sc is None:
            filter_counter["judge_error"] = filter_counter.get("judge_error", 0) + 1
            kept.append(s)
        elif sc == 1:
            filter_counter["rejected_too_easy"] = filter_counter.get("rejected_too_easy", 0) + 1
        else:
            filter_counter["kept_hard"] = filter_counter.get("kept_hard", 0) + 1
            kept.append(s)
    return kept


async def generate_batch(
    client: AsyncOpenAI,
    model: str,
    pattern: str,
    pattern_prompt: str,
    batch_size: int,
    sem: asyncio.Semaphore,
    seed_hint: int,
    domain_hint: list[str],
    max_retries: int = 3,
) -> list[BenchSample]:
    quota_line = (
        "QUOTA DOMAIN BẮT BUỘC cho batch này — phải có ÍT NHẤT 1 mẫu thuộc MỖI domain sau: "
        + ", ".join(domain_hint)
        + ". Còn lại tự do chọn domain khác trong list cho phép."
    )
    user_msg = (
        f"{pattern_prompt}\n\n"
        f"Sinh {batch_size} mẫu khác biệt nhau.\n\n"
        f"{quota_line}\n\n"
        f"Đây là seed để bạn vary scenario: {seed_hint}.\n\n"
        f"{SCHEMA_INSTRUCTIONS}"
    )
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": META_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=1.0,
                    max_tokens=4000,
                )
                payload = json.loads(resp.choices[0].message.content)
                samples = payload.get("samples", [])
                validated: list[BenchSample] = []
                for s in samples:
                    v, reason = validate_sample(s, pattern)
                    if v is not None:
                        validated.append(v)
                    else:
                        reject_counter[reason] = reject_counter.get(reason, 0) + 1
                return validated
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n[{pattern} seed={seed_hint}] failed: {e}")
                    return []
                await asyncio.sleep(2 ** attempt)
        return []


async def run_pattern(
    client: AsyncOpenAI,
    args,
    pattern: str,
    sem: asyncio.Semaphore,
    target: int,
) -> list[BenchSample]:
    cache_path = args.output_dir / f".cache_{pattern}.jsonl"
    cached: list[BenchSample] = []
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            rec = json.loads(line)
            cached.append(BenchSample(**rec))
        print(f"  [{pattern}] resumed {len(cached)} cached samples")

    if len(cached) >= target:
        return cached[:target]

    cache_fh = cache_path.open("a", encoding="utf-8")
    accepted: list[BenchSample] = list(cached)
    pbar = tqdm(total=target, desc=pattern, initial=len(accepted))

    waves = 0
    while len(accepted) < target and waves < args.max_waves:
        waves += 1
        deficit = target - len(accepted)
        # Oversample: hard validator + difficulty filter typically reject 40-60%.
        oversample = 2.5 if args.use_difficulty_filter else 1.6
        n_batches = max(1, int((deficit * oversample + args.batch_size - 1) // args.batch_size))

        tasks = [
            generate_batch(
                client, args.model, pattern, PATTERN_PROMPTS[pattern],
                args.batch_size, sem,
                seed_hint=random.randint(1, 10_000_000),
                domain_hint=pick_batch_domains(k=3),
            )
            for _ in range(n_batches)
        ]
        wave_samples: list[BenchSample] = []
        for coro in asyncio.as_completed(tasks):
            batch = await coro
            wave_samples.extend(batch)

        if args.use_difficulty_filter and wave_samples:
            wave_samples = await filter_difficulty(
                client, args.weak_model, args.judge_model, wave_samples, sem,
            )

        # Dedup against accepted (within-pattern)
        seen_keys = {(s.turns[-1]["content"], s.rewrite) for s in accepted}
        for s in wave_samples:
            key = (s.turns[-1]["content"], s.rewrite)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cache_fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            accepted.append(s)
            pbar.update(1)
            if len(accepted) >= target:
                break
        cache_fh.flush()

    pbar.close()
    cache_fh.close()
    if len(accepted) < target:
        print(f"  [{pattern}] WARN: only {len(accepted)}/{target} after {waves} waves")
    return accepted[:target]


async def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in environment")

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    target = args.samples_per_pattern
    patterns = args.patterns or list(PATTERN_PROMPTS.keys())
    unknown = [p for p in patterns if p not in PATTERN_PROMPTS]
    if unknown:
        raise SystemExit(f"unknown pattern(s): {unknown}. Valid: {list(PATTERN_PROMPTS)}")

    filt = "ON" if args.use_difficulty_filter else "OFF"
    print(f"Generating ~{target} samples × {len(patterns)} patterns "
          f"= {target * len(patterns)} total")
    print(f"  generator={args.model}  weak={args.weak_model}  judge={args.judge_model}  "
          f"difficulty-filter={filt}\n")

    all_samples: list[BenchSample] = []
    for pattern in patterns:
        samples = await run_pattern(client, args, pattern, sem, target)
        all_samples.extend(samples)
        print(f"  [{pattern}] kept {len(samples)}")

    # Dedup by (final user turn, rewrite)
    seen = set()
    deduped: list[BenchSample] = []
    for s in all_samples:
        key = (s.turns[-1]["content"], s.rewrite)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"\nDedup: {len(all_samples)} → {len(deduped)}")

    records = [to_lf_record(s, SYSTEM_PROMPT_FOR_TRAINING) for s in deduped]
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} samples to {args.output}")

    # Distribution print
    by_pat: dict[str, int] = {}
    by_dom: dict[str, int] = {}
    for s in deduped:
        by_pat[s.pattern] = by_pat.get(s.pattern, 0) + 1
        by_dom[s.domain] = by_dom.get(s.domain, 0) + 1
    print("\nPattern distribution:")
    for k, v in sorted(by_pat.items()):
        print(f"  {k:24s} {v}")
    print("Domain distribution:")
    for k, v in sorted(by_dom.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v}")

    if reject_counter:
        print("\nValidator rejections:")
        for k, v in sorted(reject_counter.items(), key=lambda x: -x[1]):
            print(f"  {k:28s} {v}")
    if filter_counter:
        print("\nDifficulty filter:")
        for k, v in sorted(filter_counter.items(), key=lambda x: -x[1]):
            print(f"  {k:28s} {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/bench/dialogues_bench.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--samples-per-pattern", type=int, default=125,
                        help="125 × 8 patterns = 1000 samples target")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weak-model", default="gpt-4o-mini",
                        help="Weak baseline used in the difficulty filter")
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="Judge used to score weak-baseline predictions")
    parser.add_argument("--use-difficulty-filter", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Drop samples the weak baseline already solves")
    parser.add_argument("--max-waves", type=int, default=6,
                        help="Safety cap on retry waves per pattern")
    parser.add_argument("--patterns", nargs="+", default=None,
                        help="Restrict to a subset of patterns (default: all 8)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
