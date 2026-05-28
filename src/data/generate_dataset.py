"""Generate Vietnamese dialogue rewrite dataset from templates + paraphrases.

Output: data/raw/dialogues.jsonl — one JSON per line, each with `messages` field
suitable for instruction tuning (system / user / assistant).
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Callable

SYSTEM_PROMPT = (
    "Bạn là model rewrite hội thoại. Nhiệm vụ của bạn là biến câu nói cuối "
    "của user thành một yêu cầu độc lập, rõ ràng, giữ nguyên ý định, không "
    "thêm thông tin không chắc chắn. Chỉ trả về câu rewrite."
)

# ---------------------------------------------------------------------------
# Paraphrase pools
# ---------------------------------------------------------------------------

AC_ON_VERBS = [
    "bật điều hoà", "mở điều hoà", "cho điều hoà chạy", "bật máy lạnh",
    "mở máy lạnh", "khởi động điều hoà",
]
AC_OFF_VERBS = ["tắt điều hoà", "tắt máy lạnh", "đóng điều hoà"]
TEMP_SLOTS = ["{n} độ", "{n} độ C", "mức {n}", "để {n}", "tầm {n} độ"]
TEMP_VALUES = [18, 20, 22, 24, 25, 26, 27, 28, 30]

MUSIC_PLAY_VERBS = ["bật nhạc", "mở nhạc", "phát nhạc", "cho nhạc chạy"]
MUSIC_PAUSE_VERBS = ["tạm dừng nhạc", "dừng nhạc", "pause nhạc lại"]
SONGS = ["Chúng Ta Của Hiện Tại", "Hoa Nở Không Màu", "Đế Vương", "Có Chàng Trai Viết Lên Cây"]
ARTISTS = ["Sơn Tùng MTP", "Hoài Lâm", "Đình Dũng", "Phan Mạnh Quỳnh"]

NAV_VERBS = ["dẫn đường", "chỉ đường", "điều hướng", "đi đến"]
PLACES = [
    "VinFast Times City", "Aeon Mall Long Biên", "sân bay Nội Bài",
    "bệnh viện Bạch Mai", "trạm sạc gần nhất", "Hồ Gươm",
]
NAV_REWRITE_OPENINGS = [
    "Tôi muốn điều hướng đến",
    "Tôi muốn đi đến",
    "Tôi muốn đi tới",
    "Hãy chỉ đường đến",
    "Dẫn tôi đến",
    "Đưa tôi tới",
]


def nav_rewrite(place: str) -> str:
    return f"{random.choice(NAV_REWRITE_OPENINGS)} {place}."

CALL_VERBS = ["gọi", "gọi điện", "gọi cho", "liên lạc"]
CONTACTS = ["Mẹ", "Bố", "anh Nam", "chị Lan", "sếp", "vợ", "chồng"]

# Bot affirmations / questions / acknowledgements
BOT_TEMP_Q = [
    "bạn muốn đặt bao nhiêu độ?",
    "bạn muốn để bao nhiêu độ?",
    "mức nhiệt độ bao nhiêu được không?",
]
BOT_CONFIRM_CALL = [
    "Bạn muốn gọi cho {x} đúng không?",
    "Xác nhận gọi cho {x}?",
]
BOT_FOUND_PLACE = [
    "{x} cách bạn {km}km",
    "Tôi tìm thấy {x}, cách đây khoảng {km}km",
]

CONFIRM_USER = ["đúng rồi", "ừ đúng", "phải rồi", "chính xác", "ok đi"]
CANCEL_USER = ["thôi hủy đi", "không cần nữa", "huỷ đi", "thôi bỏ đi"]
ADJUST_DOWN = ["giảm thêm {n} độ", "hạ {n} độ nữa", "thấp hơn {n} độ"]
ADJUST_UP = ["tăng thêm {n} độ", "lên {n} độ nữa", "cao hơn {n} độ"]


def temp_phrase(n: int) -> str:
    return random.choice(TEMP_SLOTS).format(n=n)


# ---------------------------------------------------------------------------
# Sample = (dialogue_lines, rewritten_query, metadata)
# ---------------------------------------------------------------------------


def fmt_dialogue(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def make_sample(turns, rewrite, intent, group, domain):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fmt_dialogue(turns)},
            {"role": "assistant", "content": rewrite},
        ],
        "meta": {"intent": intent, "group": group, "domain": domain},
    }


# ---------------------------------------------------------------------------
# Generators per scenario group
# ---------------------------------------------------------------------------


def gen_complete_ac():
    samples = []
    for verb, t in itertools.product(AC_ON_VERBS, TEMP_VALUES):
        turns = [("user", f"{verb} {temp_phrase(t)}")]
        rewrite = f"Tôi muốn bật điều hoà ở {t} độ."
        samples.append(make_sample(turns, rewrite, "set_ac_temperature", "complete_utterance", "ac"))
    for verb in AC_OFF_VERBS:
        turns = [("user", verb)]
        samples.append(make_sample(turns, "Tôi muốn tắt điều hoà.", "turn_off_ac", "complete_utterance", "ac"))
    return samples


def gen_complete_music():
    samples = []
    for verb, song in itertools.product(MUSIC_PLAY_VERBS, SONGS):
        turns = [("user", f"{verb} bài {song}")]
        rewrite = f"Tôi muốn phát bài {song}."
        samples.append(make_sample(turns, rewrite, "play_music", "complete_utterance", "music"))
    for verb in MUSIC_PAUSE_VERBS:
        turns = [("user", verb)]
        samples.append(make_sample(turns, "Tôi muốn tạm dừng nhạc.", "pause_music", "complete_utterance", "music"))
    return samples


def gen_complete_nav():
    samples = []
    for verb, place in itertools.product(NAV_VERBS, PLACES):
        turns = [("user", f"{verb} đến {place}")]
        rewrite = nav_rewrite(place)
        samples.append(make_sample(turns, rewrite, "navigate_to_location", "complete_utterance", "navigation"))
    return samples


def gen_complete_call():
    samples = []
    for verb, name in itertools.product(CALL_VERBS, CONTACTS):
        turns = [("user", f"{verb} {name}" if verb != "gọi cho" else f"gọi cho {name}")]
        rewrite = f"Tôi muốn gọi cho {name}."
        samples.append(make_sample(turns, rewrite, "call_contact", "complete_utterance", "calling"))
    return samples


def gen_missing_intent_ac():
    """user only says temperature; AC intent must come from earlier turn."""
    samples = []
    for verb, t in itertools.product(AC_ON_VERBS, TEMP_VALUES):
        turns = [
            ("user", verb),
            ("bot", random.choice(BOT_TEMP_Q)),
            ("user", temp_phrase(t)),
        ]
        rewrite = f"Tôi muốn bật điều hoà ở {t} độ."
        samples.append(make_sample(turns, rewrite, "set_ac_temperature", "missing_intent", "ac"))
    return samples


def gen_missing_intent_call():
    samples = []
    for name in CONTACTS:
        turns = [
            ("user", "gọi điện"),
            ("bot", "bạn muốn gọi cho ai?"),
            ("user", name),
        ]
        rewrite = f"Tôi muốn gọi cho {name}."
        samples.append(make_sample(turns, rewrite, "call_contact", "missing_intent", "calling"))
    return samples


def gen_missing_intent_music():
    samples = []
    for song in SONGS:
        turns = [
            ("user", "bật nhạc"),
            ("bot", "bạn muốn nghe bài gì?"),
            ("user", song),
        ]
        rewrite = f"Tôi muốn phát bài {song}."
        samples.append(make_sample(turns, rewrite, "play_music", "missing_intent", "music"))
    return samples


def gen_pronoun_nav():
    samples = []
    for place, km in itertools.product(PLACES, [1, 2, 3, 5]):
        bot_msg = random.choice(BOT_FOUND_PLACE).format(x=place, km=km)
        pronouns = ["đó", "nơi đó", "địa điểm đó"]
        for pn in pronouns:
            turns = [
                ("user", f"tìm {place.lower() if 'trạm' in place else place}"),
                ("bot", bot_msg),
                ("user", f"dẫn đường tới {pn}"),
            ]
            rewrite = nav_rewrite(place)
            samples.append(make_sample(turns, rewrite, "navigate_to_location", "pronoun_resolution", "navigation"))
    return samples


def gen_pronoun_music():
    samples = []
    for song, artist in itertools.product(SONGS, ARTISTS):
        turns = [
            ("user", f"có bài {song} không"),
            ("bot", f"Bài {song} của {artist}."),
            ("user", "phát bài đó đi"),
        ]
        rewrite = f"Tôi muốn phát bài {song}."
        samples.append(make_sample(turns, rewrite, "play_music", "pronoun_resolution", "music"))
    return samples


def gen_confirmation_call():
    samples = []
    for name, conf in itertools.product(CONTACTS, CONFIRM_USER):
        bot = random.choice(BOT_CONFIRM_CALL).format(x=name)
        turns = [
            ("user", f"gọi cho {name}"),
            ("bot", bot),
            ("user", conf),
        ]
        rewrite = f"Tôi muốn gọi cho {name}."
        samples.append(make_sample(turns, rewrite, "call_contact", "confirmation", "calling"))
    return samples


def gen_cancellation_ac():
    samples = []
    for t in TEMP_VALUES:
        for cancel in CANCEL_USER:
            turns = [
                ("user", f"bật điều hoà {t} độ"),
                ("bot", f"Tôi sẽ bật điều hoà {t} độ."),
                ("user", cancel),
            ]
            rewrite = f"Tôi muốn hủy yêu cầu bật điều hoà {t} độ."
            samples.append(make_sample(turns, rewrite, "cancel_action", "cancellation", "ac"))
    return samples


def gen_adjustment_ac():
    samples = []
    for t, delta in itertools.product(TEMP_VALUES, [1, 2, 3]):
        # decrease
        phrase = random.choice(ADJUST_DOWN).format(n=delta)
        turns = [
            ("user", f"bật điều hoà {t} độ"),
            ("bot", f"Đã bật điều hoà {t} độ."),
            ("user", phrase),
        ]
        rewrite = f"Tôi muốn giảm nhiệt độ điều hoà thêm {delta} độ."
        samples.append(make_sample(turns, rewrite, "decrease_temperature", "parameter_adjustment", "ac"))
        # increase
        phrase = random.choice(ADJUST_UP).format(n=delta)
        turns = [
            ("user", f"bật điều hoà {t} độ"),
            ("bot", f"Đã bật điều hoà {t} độ."),
            ("user", phrase),
        ]
        rewrite = f"Tôi muốn tăng nhiệt độ điều hoà thêm {delta} độ."
        samples.append(make_sample(turns, rewrite, "increase_temperature", "parameter_adjustment", "ac"))
    return samples


def gen_irrelevant_context_ac():
    """Earlier turns mention an unrelated number — model must NOT pull it in."""
    samples = []
    weather_temps = [35, 38, 40, 42]
    for outdoor, target in itertools.product(weather_temps, TEMP_VALUES):
        if outdoor == target:
            continue
        turns = [
            ("user", "thời tiết hôm nay thế nào?"),
            ("bot", f"Hôm nay khoảng {outdoor} độ."),
            ("user", f"bật điều hoà {target} độ"),
        ]
        rewrite = f"Tôi muốn bật điều hoà {target} độ."
        samples.append(make_sample(turns, rewrite, "set_ac_temperature", "irrelevant_context", "ac"))
    return samples


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


GENERATORS: list[Callable[[], list[dict]]] = [
    gen_complete_ac,
    gen_complete_music,
    gen_complete_nav,
    gen_complete_call,
    gen_missing_intent_ac,
    gen_missing_intent_call,
    gen_missing_intent_music,
    gen_pronoun_nav,
    gen_pronoun_music,
    gen_confirmation_call,
    gen_cancellation_ac,
    gen_adjustment_ac,
    gen_irrelevant_context_ac,
]


def generate(target: int, seed: int) -> list[dict]:
    random.seed(seed)
    pool: list[dict] = []
    for gen in GENERATORS:
        pool.extend(gen())

    # Deduplicate by (user_input, assistant_output)
    seen = set()
    unique = []
    for s in pool:
        key = (s["messages"][1]["content"], s["messages"][2]["content"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    random.shuffle(unique)
    if target and len(unique) > target:
        # Stratified downsample by group to keep coverage balanced.
        by_group: dict[str, list[dict]] = {}
        for s in unique:
            by_group.setdefault(s["meta"]["group"], []).append(s)
        per_group = max(1, target // len(by_group))
        sampled = []
        for group, items in by_group.items():
            sampled.extend(items[:per_group])
        # Top up if short.
        if len(sampled) < target:
            remaining = [s for s in unique if s not in sampled]
            sampled.extend(remaining[: target - len(sampled)])
        random.shuffle(sampled)
        return sampled[:target]
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Approx target sample count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dialogues.jsonl"))
    args = parser.parse_args()

    samples = generate(args.target, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Print group distribution
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["meta"]["group"]] = counts.get(s["meta"]["group"], 0) + 1
    print(f"Wrote {len(samples)} samples to {args.output}")
    for g, c in sorted(counts.items()):
        print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
