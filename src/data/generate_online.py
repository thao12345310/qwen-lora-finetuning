"""Generate ONLINE-domain Vietnamese car-assistant rewrite+classify data.

Task mới REWRITE_AND_CLASSIFY yêu cầu model vừa viết lại câu vừa phân loại
domain {online, offline}. Toàn bộ generator điều khiển/nav/media hiện có đều là
OFFLINE; file này bổ sung phần còn THIẾU: dữ liệu ONLINE.

Theo định nghĩa nhãn:
- online = trò chuyện phiếm / trò chuyện chung, hỏi đáp & tìm kiếm thông tin
  chung, cách sử dụng & cách khắc phục lỗi liên quan tới xe.

LƯU Ý RANH GIỚI (dễ nhầm sang offline):
- "nghe kể truyện cười", "tìm kiếm địa điểm/quán ăn", "điều khiển xe" là OFFLINE
  → KHÔNG sinh ở đây.
- Phần xe ở đây là HỎI-ĐÁP về xe ("... là sao?", "cách ... thế nào?",
  "vì sao ...?"), KHÁC hẳn câu LỆNH điều khiển ("bật/mở/đặt ..."). Cố ý đặt cùng
  meta.domain vehicle/charging với phần offline để model phân biệt theo BẢN CHẤT
  tác vụ, không theo từ khoá domain.

Mọi mẫu vẫn giữ tính chất multi-turn + giải đại từ/ellipsis như phần offline:
~2/3 context_required=True (câu cuối tham chiếu lượt trước), ~1/3 False (tự đủ).

Output: data/raw/dialogues_online.jsonl ở Llama-Factory `conversations` format,
output JSON là {"rewrite_message": "...", "domain": "online"}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

try:
    from src.data.generate_multi_turn import make_sample, pick
except ModuleNotFoundError:  # support `python src/data/generate_online.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_multi_turn import make_sample, pick


def osample(turns, rewrite, intent, pattern, domain, *, context_required=True):
    """make_sample wrapper cố định online_offline='online' cho mọi mẫu file này."""
    return make_sample(turns, rewrite, intent, pattern, domain,
                       context_required=context_required, online_offline="online")


# Các cách user lược câu hỏi sang đối tượng B (ellipsis cần resolve về câu hỏi đầy đủ).
FOLLOWUP_PHRASINGS = [
    "thế còn {x}?", "còn {x} thì sao?", "{x} thì sao?", "vậy {x} thì thế nào?",
]


# ===========================================================================
# VOCAB POOLS
# ===========================================================================

# --- Chit-chat (trò chuyện phiếm / chung) ----------------------------------
CHITCHAT_TOPICS = [
    ("màu sắc", "màu xanh dương", "Bạn thích màu gì?"),
    ("món ăn", "phở bò", "Bạn thích món ăn nào nhất?"),
    ("mùa trong năm", "mùa thu", "Bạn thích mùa nào nhất?"),
    ("thể loại nhạc", "nhạc trữ tình", "Bạn thích thể loại nhạc nào?"),
    ("môn thể thao", "bóng đá", "Bạn thích môn thể thao nào?"),
    ("loài vật", "chó", "Bạn thích con vật nào nhất?"),
]
CHITCHAT_OPENERS = [
    "nói chuyện với tôi chút đi", "buồn quá, tâm sự với tôi đi",
    "kể tôi nghe chuyện gì đó đi", "đường còn xa, trò chuyện chút nhỉ",
]
MOOD_LINES = [
    ("hôm nay tôi hơi mệt", "Bạn có thể chia sẻ thêm để tôi nghe cùng nhé."),
    ("tôi đang vui lắm", "Thật tuyệt, có chuyện gì vui vậy bạn?"),
    ("dạo này tôi stress quá", "Mình ở đây nghe bạn nói nè."),
]

# --- Hỏi đáp / tìm kiếm thông tin chung -------------------------------------
CAPITALS = [
    ("Pháp", "Paris"), ("Đức", "Berlin"), ("Nhật Bản", "Tokyo"),
    ("Hàn Quốc", "Seoul"), ("Ý", "Rome"), ("Anh", "London"),
    ("Thái Lan", "Bangkok"), ("Úc", "Canberra"), ("Nga", "Moscow"),
    ("Trung Quốc", "Bắc Kinh"), ("Indonesia", "Jakarta"), ("Tây Ban Nha", "Madrid"),
    ("Brazil", "Brasília"), ("Canada", "Ottawa"), ("Ấn Độ", "New Delhi"),
]
PEOPLE_BIRTH = [
    ("Sơn Tùng M-TP", "năm 1994"), ("Mỹ Tâm", "năm 1981"),
    ("Đen Vâu", "năm 1989"), ("Hồ Ngọc Hà", "năm 1984"),
    ("Hoài Lâm", "năm 1995"), ("Hà Anh Tuấn", "năm 1984"),
    ("Bích Phương", "năm 1989"), ("Trấn Thành", "năm 1987"),
]
UNIT_CONVERTS = [
    ("1 hải lý", "1,852 km"), ("1 dặm", "1,609 km"),
    ("1 feet", "0,3048 mét"), ("1 inch", "2,54 cm"),
    ("1 pound", "0,4536 kg"),
]
DEFINITIONS = [
    "lạm phát", "blockchain", "GDP", "hiệu ứng nhà kính",
    "trí tuệ nhân tạo", "năng lượng tái tạo", "lãi suất kép",
    "điện toán đám mây", "tỷ giá hối đoái", "khủng hoảng kinh tế",
]
SIMPLE_FACTS = [
    "Nước sôi ở bao nhiêu độ C?",
    "Ai là người phát minh ra bóng đèn?",
    "Một năm ánh sáng dài bao nhiêu?",
    "Trái Đất quay quanh Mặt Trời mất bao lâu?",
    "Đỉnh núi cao nhất thế giới tên là gì?",
    "Đại dương nào lớn nhất thế giới?",
    "Con sông nào dài nhất thế giới?",
    "Ánh sáng đi nhanh bao nhiêu?",
    "Cơ thể người có bao nhiêu xương?",
    "Hành tinh nào lớn nhất hệ Mặt Trời?",
    "Ai vẽ bức tranh Mona Lisa?",
    "Vạn Lý Trường Thành dài bao nhiêu km?",
]

# --- Hướng dẫn dùng & khắc phục lỗi xe (HỎI-ĐÁP, không phải lệnh) -----------
CAR_FEATURES = [
    "chế độ off-road", "chế độ thể thao", "ga tự động adaptive cruise",
    "hỗ trợ giữ làn", "chế độ một bàn đạp", "camera 360 độ",
    "khoá vi sai", "chế độ tiết kiệm pin", "cảnh báo điểm mù",
    "hỗ trợ đỗ xe tự động", "chế độ tản nhiệt pin", "sạc hai chiều V2L",
]
WARNING_LIGHTS = [
    "đèn cảnh báo động cơ", "đèn áp suất lốp", "đèn ắc-quy",
    "đèn ABS", "đèn nhiệt độ nước làm mát", "đèn túi khí",
    "đèn phanh tay", "đèn trợ lực lái", "đèn cảnh báo pin EV",
]
CAR_ISSUES = [
    "xe không vào sạc được", "màn hình trung tâm bị treo",
    "cửa sổ trời không đóng được", "xe sạc rất chậm",
    "điều hoà không mát", "kết nối Bluetooth bị rớt liên tục",
    "cảm biến lùi kêu liên tục", "gạt mưa tự động không hoạt động",
    "khoá thông minh không nhận", "đèn pha tự động không bật",
    "xe báo lỗi hệ thống phanh", "ghế chỉnh điện bị kẹt",
]
CHARGING_QUESTIONS = [
    "Sạc đầy pin xe điện mất bao lâu?",
    "Có nên sạc pin tới 100% mỗi lần không?",
    "Sạc nhanh DC khác sạc thường ở điểm nào?",
    "Đi được bao nhiêu km khi pin đầy?",
    "Sạc qua đêm ở nhà có hại pin không?",
    "Thời tiết lạnh ảnh hưởng thế nào tới quãng đường?",
    "Pin xe điện dùng được bao nhiêu năm?",
    "Nên giữ pin ở mức bao nhiêu phần trăm khi đỗ lâu?",
]


# ===========================================================================
# GENERATORS — chit-chat
# ===========================================================================
def gen_chitchat_preference_followup():
    """user hỏi sở thích A → bot trả lời → user lược 'thế còn B' → resolve."""
    out = []
    for topic, ans, q_full in CHITCHAT_TOPICS:
        for o_topic, _o_ans, o_full in CHITCHAT_TOPICS:
            if o_topic == topic:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", q_full),
                    ("bot", f"Mình thích {ans}. Còn bạn thì sao?"),
                    ("user", phr.format(x=o_topic)),
                ]
                out.append(osample(turns, o_full, "chitchat_preference",
                                   "reference_resolution", "chitchat"))
    return out


def gen_chitchat_mood():
    """trò chuyện phiếm tự đủ (no-retrieve)."""
    out = []
    for line, reply in MOOD_LINES:
        turns = [
            ("user", pick(CHITCHAT_OPENERS)),
            ("bot", "Mình luôn sẵn lòng nghe bạn nè."),
            ("user", line),
        ]
        # câu cuối tự đủ nghĩa → giữ nguyên ý, chuẩn hoá nhẹ
        rewrite = line[0].upper() + line[1:] + "."
        out.append(osample(turns, rewrite, "chitchat_mood", "no_retrieve", "chitchat",
                           context_required=False))
    return out


# ===========================================================================
# GENERATORS — hỏi đáp / thông tin chung
# ===========================================================================
def gen_qa_capital_followup():
    out = []
    for country, cap in CAPITALS:
        for o_country, _o_cap in CAPITALS:
            if o_country == country:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"thủ đô của {country} là gì?"),
                    ("bot", f"Thủ đô của {country} là {cap}."),
                    ("user", phr.format(x=o_country)),
                ]
                rewrite = f"Thủ đô của {o_country} là gì?"
                out.append(osample(turns, rewrite, "qa_capital", "reference_resolution", "general_qa"))
    return out


def gen_qa_birthyear_followup():
    out = []
    for name, year in PEOPLE_BIRTH:
        for o_name, _o_year in PEOPLE_BIRTH:
            if o_name == name:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"{name} sinh năm nào?"),
                    ("bot", f"{name} sinh {year}."),
                    ("user", phr.format(x=o_name)),
                ]
                rewrite = f"{o_name} sinh năm nào?"
                out.append(osample(turns, rewrite, "qa_birthyear", "reference_resolution", "general_qa"))
    return out


def gen_qa_unit_convert():
    """câu hỏi đổi đơn vị, tự đủ (no-retrieve)."""
    out = []
    for unit, _val in UNIT_CONVERTS:
        turns = [
            ("user", pick(["cho hỏi cái này", "tôi hỏi chút"])),
            ("bot", "Bạn cứ hỏi nhé."),
            ("user", f"{unit} bằng bao nhiêu km?" if "km" not in unit else f"{unit} bằng bao nhiêu?"),
        ]
        rewrite = f"{unit[0].upper() + unit[1:]} bằng bao nhiêu?"
        out.append(osample(turns, rewrite, "qa_unit_convert", "no_retrieve", "general_qa",
                           context_required=False))
    return out


def gen_qa_definition_followup():
    out = []
    for term in DEFINITIONS:
        # nhánh "thế còn B": resolve sang khái niệm B
        for other in DEFINITIONS:
            if other == term:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"{term} là gì?"),
                    ("bot", f"Đó là một khái niệm liên quan tới {term}."),
                    ("user", phr.format(x=other)),
                ]
                rewrite = f"{other[0].upper() + other[1:]} là gì?"
                out.append(osample(turns, rewrite, "qa_definition", "reference_resolution", "general_qa"))
        # nhánh đại từ "cái đó": resolve về chính term gốc
        turns = [
            ("user", f"{term} là gì?"),
            ("bot", f"Đó là một khái niệm liên quan tới {term}."),
            ("user", pick(["giải thích kỹ hơn cái đó đi", "nói rõ hơn về cái đó được không"])),
        ]
        out.append(osample(turns, f"Giải thích kỹ hơn về {term}.",
                           "qa_definition", "pronoun_resolution", "general_qa"))
    return out


def gen_qa_simple_fact():
    """câu hỏi kiến thức tự đủ (no-retrieve)."""
    out = []
    for q in SIMPLE_FACTS:
        turns = [
            ("user", pick(["hỏi tí", "cho hỏi cái"])),
            ("bot", "Vâng, bạn hỏi đi."),
            ("user", q.lower()),
        ]
        out.append(osample(turns, q, "qa_fact", "no_retrieve", "general_qa",
                           context_required=False))
    return out


# ===========================================================================
# GENERATORS — hướng dẫn dùng & khắc phục lỗi xe (HỎI-ĐÁP về xe)
# ===========================================================================
def gen_car_howto_feature_followup():
    """Hỏi CÁCH DÙNG tính năng (online), tương phản với LỆNH bật tính năng (offline)."""
    out = []
    for feat in CAR_FEATURES:
        for other in CAR_FEATURES:
            if other == feat:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"cách bật {feat} trên xe như thế nào?"),
                    ("bot", f"Bạn vào phần cài đặt rồi tìm mục {feat} để kích hoạt."),
                    ("user", phr.format(x=other)),
                ]
                rewrite = f"Cách bật {other} trên xe như thế nào?"
                out.append(osample(turns, rewrite, "car_howto", "reference_resolution", "vehicle"))
    return out


def gen_car_warning_followup():
    """Hỏi Ý NGHĨA đèn cảnh báo (online troubleshooting)."""
    out = []
    for light in WARNING_LIGHTS:
        for other in WARNING_LIGHTS:
            if other == light:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"{light} sáng là sao?"),
                    ("bot", f"{light[0].upper() + light[1:]} sáng thường báo một vấn đề cần kiểm tra."),
                    ("user", phr.format(x=other)),
                ]
                rewrite = f"{other[0].upper() + other[1:]} sáng nghĩa là gì?"
                out.append(osample(turns, rewrite, "car_warning", "reference_resolution", "vehicle"))
    return out


def gen_car_troubleshoot():
    """Hỏi cách khắc phục lỗi xe, tự đủ (no-retrieve)."""
    out = []
    for issue in CAR_ISSUES:
        turns = [
            ("user", pick(["tôi gặp vấn đề với xe", "xe đang bị lỗi"])),
            ("bot", "Bạn mô tả lỗi cụ thể giúp mình nhé."),
            ("user", f"{issue} thì phải làm sao?"),
        ]
        rewrite = f"{issue[0].upper() + issue[1:]} thì phải làm sao?"
        out.append(osample(turns, rewrite, "car_troubleshoot", "no_retrieve", "vehicle",
                           context_required=False))
    return out


def gen_charging_howto():
    """Hỏi đáp kiến thức về sạc (online), tương phản với LỆNH 'tìm trạm sạc' (offline)."""
    out = []
    for q in CHARGING_QUESTIONS:
        turns = [
            ("user", pick(["cho hỏi về sạc xe điện", "tôi muốn hỏi chuyện sạc pin"])),
            ("bot", "Vâng, bạn cứ hỏi về việc sạc nhé."),
            ("user", q.lower()),
        ]
        out.append(osample(turns, q, "charging_howto", "no_retrieve", "charging",
                           context_required=False))
    return out


def gen_charging_howto_followup():
    """multi-turn charging Q&A có tham chiếu."""
    out = []
    methods = ["sạc nhanh DC", "sạc thường AC", "sạc tại nhà", "trạm sạc công cộng"]
    for a in methods:
        for b in methods:
            if a == b:
                continue
            for phr in FOLLOWUP_PHRASINGS:
                turns = [
                    ("user", f"sạc đầy bằng {a} mất bao lâu?"),
                    ("bot", f"Thời gian sạc bằng {a} tuỳ dung lượng pin và công suất trạm."),
                    ("user", phr.format(x=b)),
                ]
                rewrite = f"Sạc đầy bằng {b} mất bao lâu?"
                out.append(osample(turns, rewrite, "charging_howto", "reference_resolution", "charging"))
    return out


# ===========================================================================
# REGISTRATION
# ===========================================================================
GENERATORS: list[Callable[[], list[dict]]] = [
    gen_chitchat_preference_followup,
    gen_chitchat_mood,
    gen_qa_capital_followup,
    gen_qa_birthyear_followup,
    gen_qa_unit_convert,
    gen_qa_definition_followup,
    gen_qa_simple_fact,
    gen_car_howto_feature_followup,
    gen_car_warning_followup,
    gen_car_troubleshoot,
    gen_charging_howto,
    gen_charging_howto_followup,
]


# Trần số mẫu được phép DÙNG CHUNG một gold rewrite. Các generator follow-up sinh
# nhiều hội thoại KHÁC nhau (đổi câu hỏi distractor A + cách hỏi) nhưng cùng gold B
# → đa dạng INPUT (tốt cho classifier) mà không để gold nào bị lặp tới mức memorize.
MAX_PER_GOLD = 24


def _conv_key(s: dict) -> str:
    # dedup theo TOÀN hội thoại (giữ các biến thể context distractor khác nhau).
    return json.dumps(s["conversations"], ensure_ascii=False, sort_keys=True)


def _gold_of(s: dict) -> str:
    return json.loads(s["conversations"][-1]["value"])["rewrite_message"]


def generate(target: int, seed: int, repeats: int = 1) -> list[dict]:
    random.seed(seed)
    pool: list[dict] = []
    # mỗi vòng repeats re-roll các pick() ngẫu nhiên → tăng đa dạng/khối lượng.
    for _ in range(max(1, repeats)):
        for gen in GENERATORS:
            pool.extend(gen())

    # Dedup theo toàn hội thoại, rồi giới hạn lặp gold ≤ MAX_PER_GOLD.
    seen = set()
    gold_count: dict[str, int] = {}
    unique = []
    random.shuffle(pool)
    for s in pool:
        key = _conv_key(s)
        if key in seen:
            continue
        g = _gold_of(s)
        if gold_count.get(g, 0) >= MAX_PER_GOLD:
            continue
        seen.add(key)
        gold_count[g] = gold_count.get(g, 0) + 1
        unique.append(s)

    random.shuffle(unique)

    # target<=0 → trả TOÀN BỘ unique (no_retrieve online vốn ít, không ép cap;
    # tỷ lệ retrieve:no_retrieve toàn dataset đã do phần offline + `take` ở
    # build_train_mix điều tiết). target>0 → ép ~2/3 retrieve : ~1/3 no-retrieve.
    if not target or target <= 0:
        return unique

    retrieve = [s for s in unique if s["meta"]["context_required"]]
    no_retrieve = [s for s in unique if not s["meta"]["context_required"]]
    n_no = min(len(no_retrieve), target // 3)
    n_re = min(len(retrieve), target - n_no)
    selected = retrieve[:n_re] + no_retrieve[:n_no]
    random.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=0, help="Số mẫu mục tiêu (0 = giữ toàn bộ).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Số vòng re-roll pick() ngẫu nhiên để tăng khối lượng/đa dạng.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/dialogues_online.jsonl"))
    args = parser.parse_args()

    samples = generate(args.target, args.seed, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} ONLINE samples → {args.output}")


if __name__ == "__main__":
    main()
