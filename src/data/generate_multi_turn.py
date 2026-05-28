"""Generate multi-turn (2–4 user turns) Vietnamese car-assistant rewrite data.

A "turn" = một lượt nói của user. Mỗi mẫu có 2 đến 4 user-turn, xen kẽ bot,
final luôn là user. Tổng số message = 3 / 5 / 7 (hoặc bot mở đầu: 4 / 6).

Mục tiêu: bài rewrite chuẩn phải gói GỌN mọi tham số quan trọng rải rác qua
các turn trước (slot, ràng buộc, thay đổi, hủy bỏ, lựa chọn...), không bỏ sót
và không kéo theo thông tin nhiễu.

Output: data/raw/dialogues_multi_turn.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Callable

SYSTEM_PROMPT = (
    "Bạn là model rewrite hội thoại trên xe ô tô. Nhiệm vụ: biến lượt nói "
    "cuối của user thành một yêu cầu độc lập, đầy đủ ý — gộp mọi tham số và "
    "ràng buộc đã xuất hiện ở các lượt trước, bỏ qua thông tin nhiễu. Chỉ "
    "trả về câu rewrite, không giải thích."
)

# ===========================================================================
# VOCAB POOLS
# ===========================================================================

# --- Climate ---------------------------------------------------------------
AC_ON_VERBS = ["bật điều hoà", "mở điều hoà", "khởi động điều hoà", "bật máy lạnh", "mở AC"]
AC_OFF_VERBS = ["tắt điều hoà", "tắt máy lạnh", "đóng điều hoà"]
HEATER_ON_VERBS = ["bật sưởi", "mở sưởi ấm", "bật heater"]
DEFROST_VERBS = ["bật sấy kính", "mở defrost", "bật chế độ sấy kính"]
FAN_LEVELS = [1, 2, 3, 4, 5]
TEMP_VALUES = [16, 18, 20, 22, 23, 24, 25, 26, 27, 28]
AC_MODES = ["mát", "lạnh sâu", "tự động", "tiết kiệm", "êm"]
AC_ZONES = ["ghế lái", "ghế phụ", "hàng ghế sau", "cả xe", "phía trước"]
WIND_DIRS = ["thổi lên mặt", "thổi xuống chân", "thổi đều", "ra kính lái"]

# --- Music / Audio ---------------------------------------------------------
SONGS = [
    "Chúng Ta Của Hiện Tại", "Hoa Nở Không Màu", "Đế Vương",
    "Có Chàng Trai Viết Lên Cây", "Nắng Ấm Xa Dần", "Lạc Trôi",
    "Em Của Ngày Hôm Qua", "Người Lạ Ơi", "Yêu 5",
]
ARTISTS = ["Sơn Tùng MTP", "Hoài Lâm", "Đình Dũng", "Phan Mạnh Quỳnh", "Đen Vâu", "Mỹ Tâm"]
PLAYLISTS = ["nhạc trữ tình", "nhạc thư giãn", "nhạc rap Việt", "nhạc EDM", "nhạc thiền", "Top 50 Vpop"]
RADIO_STATIONS = ["VOV Giao thông", "VOV1", "XoneFM", "Joy FM"]
PODCASTS = ["Tâm Sự Kinh Doanh", "Have A Sip", "Người Trong Muôn Nghề", "Vietcetera"]
VOLUME_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80]

# --- Navigation ------------------------------------------------------------
PLACES = [
    "VinFast Times City", "Aeon Mall Long Biên", "sân bay Nội Bài",
    "bệnh viện Bạch Mai", "Hồ Gươm", "phố cổ Hà Nội", "Lotte Center",
    "Royal City", "Big C Thăng Long", "Đại học Bách Khoa",
]
POI_CATEGORIES = ["quán cafe", "trạm xăng", "trạm sạc", "nhà hàng", "bãi đỗ xe", "ATM", "siêu thị"]
ROUTE_PREFS = ["tránh đường thu phí", "tránh cao tốc", "đi đường nhanh nhất", "tránh tắc đường", "đi đường ngắn nhất"]
DISTANCES = [0.8, 1.2, 2.5, 3.0, 4.5, 7.0]
NAV_REWRITE_OPENINGS = [
    "Tôi muốn điều hướng đến",
    "Tôi muốn đi đến",
    "Tôi muốn đi tới",
    "Hãy chỉ đường đến",
    "Dẫn tôi đến",
    "Đưa tôi tới",
]
# Verb chunks usable inside a chained "Tôi muốn … và …" sentence (no "Tôi muốn" prefix).
CHAINED_NAV_VERBS = ["điều hướng đến", "đi tới", "đi đến", "ghé qua"]

# --- Calling / Messaging ---------------------------------------------------
CONTACTS = ["Mẹ", "Bố", "anh Nam", "chị Lan", "sếp", "vợ", "chồng", "em Mai", "anh Tuấn"]
SMS_CONTENTS = [
    "tôi đang trên đường, sắp đến",
    "kẹt xe, đến muộn 15 phút",
    "tối nay ăn cơm ngoài nhé",
    "xong việc rồi, về luôn đây",
    "đợi anh 10 phút",
]

# --- Vehicle controls ------------------------------------------------------
WINDOWS = ["cửa kính lái", "cửa kính phụ", "cửa kính sau bên trái", "cửa kính sau bên phải", "tất cả cửa kính"]
SEATS = ["ghế lái", "ghế phụ", "ghế sau bên trái", "ghế sau bên phải"]
SEAT_HEAT_LEVELS = [1, 2, 3]
LIGHTS = ["đèn pha", "đèn cốt", "đèn sương mù", "đèn nội thất", "đèn cabin"]
WIPER_SPEEDS = ["chậm", "trung bình", "nhanh", "tự động"]

# --- Trip / energy ---------------------------------------------------------
BATTERY_LEVELS = [8, 12, 15, 20, 25]
RANGE_KM = [30, 50, 80, 120]
CHARGING_NETWORKS = ["VinFast", "EV One", "EBOOST"]

# --- Driving modes / assistance --------------------------------------------
DRIVE_MODES = ["thể thao", "tiết kiệm", "tiêu chuẩn", "tuyết", "off-road"]
CRUISE_SPEEDS = [60, 70, 80, 90, 100, 110]

# --- Smart home ------------------------------------------------------------
HOME_DEVICES = ["đèn phòng khách", "điều hoà phòng ngủ", "rèm cửa", "cổng nhà", "máy lọc không khí"]

# --- Confirm / cancel cues -------------------------------------------------
USER_CONFIRM = ["đúng rồi", "ừ đúng đó", "ok đi", "chuẩn rồi", "phải"]
USER_CANCEL = ["thôi hủy đi", "không cần nữa", "bỏ đi", "không gọi nữa", "thôi đừng"]


# ===========================================================================
# HELPERS
# ===========================================================================


def fmt_dialogue(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def make_sample(turns, rewrite, intent, pattern, domain):
    user_turns = sum(1 for r, _ in turns if r == "user")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fmt_dialogue(turns)},
            {"role": "assistant", "content": rewrite},
        ],
        "meta": {
            "intent": intent,
            "pattern": pattern,
            "group": pattern,  # alias for compat with split_data.py (stratify key)
            "domain": domain,
            "user_turns": user_turns,
            "messages": len(turns),
        },
    }


def pick(seq):
    return random.choice(seq)


def nav_rewrite(place: str, *, pref: str | None = None, depart: str | None = None) -> str:
    s = f"{random.choice(NAV_REWRITE_OPENINGS)} {place}"
    if pref:
        s += f", ưu tiên {pref}"
    if depart:
        s += f" và khởi hành {depart}"
    return s + "."


# ===========================================================================
# 2-USER-TURN GENERATORS  (user-bot-user, 3 messages)
# ===========================================================================


def gen_2t_slot_ac_temp():
    """user mở AC; bot hỏi nhiệt độ; user cho nhiệt độ."""
    out = []
    for verb, t in itertools.product(AC_ON_VERBS, TEMP_VALUES):
        turns = [
            ("user", verb),
            ("bot", pick(["bạn muốn đặt bao nhiêu độ?", "để mức bao nhiêu độ ạ?", "nhiệt độ bao nhiêu?"])),
            ("user", pick([f"{t}", f"{t} độ", f"để {t}", f"khoảng {t}"])),
        ]
        rw = f"Tôi muốn bật điều hoà ở {t} độ."
        out.append(make_sample(turns, rw, "set_ac_temperature", "slot_filling", "climate"))
    return out


def gen_2t_slot_ac_mode():
    """user yêu cầu AC ở mức nhiệt; bot hỏi chế độ; user chọn chế độ."""
    out = []
    for t, mode in itertools.product(TEMP_VALUES[:6], AC_MODES):
        turns = [
            ("user", f"bật điều hoà {t} độ"),
            ("bot", "bạn muốn chế độ nào: mát, lạnh sâu, tự động, tiết kiệm hay êm?"),
            ("user", f"chế độ {mode}"),
        ]
        rw = f"Tôi muốn bật điều hoà {t} độ ở chế độ {mode}."
        out.append(make_sample(turns, rw, "set_ac_temperature_with_mode", "slot_filling", "climate"))
    return out


def gen_2t_slot_fan_zone():
    """user bật quạt; bot hỏi vùng; user chọn vùng."""
    out = []
    for level, zone in itertools.product(FAN_LEVELS, AC_ZONES):
        turns = [
            ("user", f"chỉnh quạt gió mức {level}"),
            ("bot", "bạn muốn áp dụng cho vùng nào?"),
            ("user", f"{zone}"),
        ]
        rw = f"Tôi muốn chỉnh quạt gió mức {level} cho {zone}."
        out.append(make_sample(turns, rw, "set_fan_level_zone", "slot_filling", "climate"))
    return out


def gen_2t_slot_call_contact():
    out = []
    for name in CONTACTS:
        turns = [
            ("user", pick(["gọi điện", "tôi muốn gọi", "thực hiện cuộc gọi"])),
            ("bot", "bạn muốn gọi cho ai?"),
            ("user", name),
        ]
        rw = f"Tôi muốn gọi cho {name}."
        out.append(make_sample(turns, rw, "call_contact", "slot_filling", "calling"))
    return out


def gen_2t_slot_sms_recipient():
    """user gửi sms; bot hỏi gửi cho ai; user trả lời (giữ nội dung từ turn 1)."""
    out = []
    for name, content in itertools.product(CONTACTS, SMS_CONTENTS):
        turns = [
            ("user", f"gửi tin nhắn '{content}'"),
            ("bot", "gửi cho ai ạ?"),
            ("user", name),
        ]
        rw = f"Tôi muốn gửi tin nhắn '{content}' cho {name}."
        out.append(make_sample(turns, rw, "send_sms", "slot_filling", "messaging"))
    return out


def gen_2t_slot_music_song():
    out = []
    for verb, song in itertools.product(["bật nhạc", "mở nhạc đi", "phát nhạc"], SONGS):
        turns = [
            ("user", verb),
            ("bot", "bạn muốn nghe bài gì?"),
            ("user", song),
        ]
        rw = f"Tôi muốn phát bài {song}."
        out.append(make_sample(turns, rw, "play_music", "slot_filling", "music"))
    return out


def gen_2t_slot_music_playlist():
    out = []
    for pl in PLAYLISTS:
        turns = [
            ("user", "mở nhạc"),
            ("bot", "bạn muốn nghe playlist nào?"),
            ("user", pl),
        ]
        rw = f"Tôi muốn phát playlist {pl}."
        out.append(make_sample(turns, rw, "play_playlist", "slot_filling", "music"))
    return out


def gen_2t_slot_nav_place():
    out = []
    for place in PLACES:
        turns = [
            ("user", pick(["dẫn đường đi", "chỉ đường giúp", "điều hướng"])),
            ("bot", "bạn muốn đi đâu?"),
            ("user", place),
        ]
        rw = nav_rewrite(place)
        out.append(make_sample(turns, rw, "navigate_to_location", "slot_filling", "navigation"))
    return out


def gen_2t_slot_window():
    out = []
    for w in WINDOWS:
        turns = [
            ("user", "mở cửa kính"),
            ("bot", "bạn muốn mở cửa nào?"),
            ("user", w),
        ]
        rw = f"Tôi muốn mở {w}."
        out.append(make_sample(turns, rw, "open_window", "slot_filling", "vehicle"))
    return out


def gen_2t_slot_seat_heat():
    out = []
    for seat, lvl in itertools.product(SEATS, SEAT_HEAT_LEVELS):
        turns = [
            ("user", "bật sưởi ghế"),
            ("bot", "ghế nào và mức bao nhiêu?"),
            ("user", f"{seat} mức {lvl}"),
        ]
        rw = f"Tôi muốn bật sưởi {seat} mức {lvl}."
        out.append(make_sample(turns, rw, "set_seat_heat", "slot_filling", "vehicle"))
    return out


def gen_2t_slot_cruise_speed():
    out = []
    for s in CRUISE_SPEEDS:
        turns = [
            ("user", "bật cruise control"),
            ("bot", "đặt tốc độ bao nhiêu km/h?"),
            ("user", f"{s} km/h"),
        ]
        rw = f"Tôi muốn bật cruise control ở tốc độ {s} km/h."
        out.append(make_sample(turns, rw, "set_cruise_control", "slot_filling", "driver_assist"))
    return out


def gen_2t_confirm_call():
    out = []
    for name, conf in itertools.product(CONTACTS, USER_CONFIRM):
        turns = [
            ("user", f"gọi cho {name}"),
            ("bot", f"Bạn muốn gọi cho {name} đúng không?"),
            ("user", conf),
        ]
        rw = f"Tôi muốn gọi cho {name}."
        out.append(make_sample(turns, rw, "call_contact", "confirmation", "calling"))
    return out


def gen_2t_cancel_call():
    out = []
    for name, cancel in itertools.product(CONTACTS, USER_CANCEL):
        turns = [
            ("user", f"gọi cho {name}"),
            ("bot", f"Đang gọi cho {name}…"),
            ("user", cancel),
        ]
        rw = f"Tôi muốn hủy cuộc gọi đến {name}."
        out.append(make_sample(turns, rw, "cancel_call", "cancellation", "calling"))
    return out


def gen_2t_adjust_temp():
    out = []
    for t, d in itertools.product(TEMP_VALUES[:6], [1, 2, 3]):
        # increase
        turns = [
            ("user", f"bật điều hoà {t} độ"),
            ("bot", f"Đã bật điều hoà {t} độ."),
            ("user", pick([f"tăng thêm {d} độ", f"nóng quá, lên {d} độ nữa", f"cao thêm {d} độ"])),
        ]
        rw = f"Tôi muốn tăng nhiệt độ điều hoà từ {t} độ thêm {d} độ."
        out.append(make_sample(turns, rw, "adjust_ac_temperature", "adjustment", "climate"))
        # decrease
        turns = [
            ("user", f"bật điều hoà {t} độ"),
            ("bot", f"Đã bật điều hoà {t} độ."),
            ("user", pick([f"giảm thêm {d} độ", f"lạnh hơn {d} độ nữa", f"thấp xuống {d} độ"])),
        ]
        rw = f"Tôi muốn giảm nhiệt độ điều hoà từ {t} độ thêm {d} độ."
        out.append(make_sample(turns, rw, "adjust_ac_temperature", "adjustment", "climate"))
    return out


def gen_2t_adjust_volume():
    out = []
    for v, d in itertools.product([20, 40, 60], [10, 20]):
        turns = [
            ("user", f"để âm lượng {v}"),
            ("bot", f"Đã đặt âm lượng mức {v}."),
            ("user", pick([f"to thêm {d}", f"tăng {d} nữa", f"lớn lên {d}"])),
        ]
        rw = f"Tôi muốn tăng âm lượng từ mức {v} thêm {d}."
        out.append(make_sample(turns, rw, "adjust_volume", "adjustment", "music"))
    return out


def gen_2t_reference_nav():
    """bot vừa kể địa điểm; user dùng đại từ 'đó'."""
    out = []
    for place, km in itertools.product(PLACES, DISTANCES):
        turns = [
            ("user", f"tìm {place}"),
            ("bot", f"Tôi thấy {place}, cách đây {km}km."),
            ("user", pick(["dẫn tới đó đi", "đi đến đó", "chỉ đường tới đấy", "đến chỗ đó"])),
        ]
        rw = nav_rewrite(place)
        out.append(make_sample(turns, rw, "navigate_to_location", "reference_resolution", "navigation"))
    return out


def gen_2t_reference_music():
    out = []
    for song, artist in itertools.product(SONGS, ARTISTS):
        turns = [
            ("user", f"có bài {song} không"),
            ("bot", f"Có bài {song} của {artist}."),
            ("user", pick(["phát bài đó", "mở bài đấy đi", "cho nghe bài đó"])),
        ]
        rw = f"Tôi muốn phát bài {song} của {artist}."
        out.append(make_sample(turns, rw, "play_music", "reference_resolution", "music"))
    return out


def gen_2t_reference_call():
    """user vừa hỏi số của ai đó; bot trả lời; user yêu cầu gọi 'người đó'."""
    out = []
    for name in CONTACTS:
        turns = [
            ("user", f"số của {name} là gì"),
            ("bot", f"Số của {name} đây."),
            ("user", pick(["gọi luôn đi", "gọi cho người đó", "gọi đi"])),
        ]
        rw = f"Tôi muốn gọi cho {name}."
        out.append(make_sample(turns, rw, "call_contact", "reference_resolution", "calling"))
    return out


def gen_2t_disambig_nav():
    """bot đưa nhiều lựa chọn; user pick 1."""
    out = []
    pairs = list(itertools.combinations(PLACES, 2))
    random.shuffle(pairs)
    for p1, p2 in pairs[:30]:
        turns = [
            ("user", "tìm chỗ ăn"),
            ("bot", f"Có hai gợi ý: {p1} và {p2}. Bạn chọn nơi nào?"),
            ("user", pick([f"chọn {p1}", f"đi {p1}", "cái thứ nhất"])),
        ]
        # When user says "cái thứ nhất" the model must resolve from bot's listing order.
        chosen = p1
        rw = nav_rewrite(chosen)
        out.append(make_sample(turns, rw, "navigate_to_location", "disambiguation", "navigation"))
    return out


def gen_2t_chained_ac_music():
    """user yêu cầu 2 hành động cùng lúc trong 1 câu cuối."""
    out = []
    for t, song in itertools.product([22, 24, 26], SONGS[:4]):
        turns = [
            ("user", "tôi vừa lên xe"),
            ("bot", "Tôi giúp gì được cho bạn?"),
            ("user", f"bật điều hoà {t} độ và mở bài {song}"),
        ]
        rw = f"Tôi muốn bật điều hoà {t} độ và phát bài {song}."
        out.append(make_sample(turns, rw, "chained_climate_music", "chained_intent", "multi"))
    return out


def gen_2t_irrelevant_outdoor_temp():
    """bot kể nhiệt độ ngoài trời; user yêu cầu AC ở nhiệt độ khác — KHÔNG dính nhiệt ngoài."""
    out = []
    for outdoor, target in itertools.product([35, 38, 40], TEMP_VALUES):
        if outdoor == target:
            continue
        turns = [
            ("user", "ngoài trời bao nhiêu độ?"),
            ("bot", f"Hiện tại ngoài trời {outdoor} độ."),
            ("user", f"bật điều hoà {target} độ"),
        ]
        rw = f"Tôi muốn bật điều hoà {target} độ."
        out.append(make_sample(turns, rw, "set_ac_temperature", "irrelevant_context", "climate"))
    return out


def gen_2t_bot_alert_charge():
    """bot báo pin yếu trước; user yêu cầu tìm trạm sạc."""
    out = []
    for batt, net in itertools.product(BATTERY_LEVELS, CHARGING_NETWORKS):
        turns = [
            ("bot", f"Cảnh báo: pin còn {batt}%, nên tìm trạm sạc sớm."),
            ("user", pick(["tìm trạm sạc gần nhất", "chỉ trạm sạc đi", "đến trạm sạc gần đây"])),
            ("bot", f"Bạn muốn ưu tiên hãng nào?"),
            ("user", net),
        ]
        rw = f"Tôi muốn tìm trạm sạc {net} gần nhất."
        out.append(make_sample(turns, rw, "find_charging_station", "slot_filling", "navigation"))
    return out


# ===========================================================================
# 3-USER-TURN GENERATORS (5 messages: u-b-u-b-u)
# ===========================================================================


def gen_3t_multi_slot_nav():
    """nav: place → bot hỏi route pref → user trả lời → bot xác nhận → user confirm."""
    out = []
    for place, pref in itertools.product(PLACES, ROUTE_PREFS):
        turns = [
            ("user", f"dẫn đường đến {place}"),
            ("bot", "bạn muốn ưu tiên lộ trình thế nào?"),
            ("user", pref),
            ("bot", f"Sẽ dẫn đến {place} với tuỳ chọn {pref}, đúng chưa?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = nav_rewrite(place, pref=pref)
        out.append(make_sample(turns, rw, "navigate_with_preference", "multi_slot", "navigation"))
    return out


def gen_3t_multi_slot_sms():
    """user gửi sms → bot hỏi gửi cho ai → user trả lời → bot hỏi nội dung → user trả lời."""
    out = []
    for name, content in itertools.product(CONTACTS, SMS_CONTENTS):
        turns = [
            ("user", "gửi tin nhắn"),
            ("bot", "gửi cho ai ạ?"),
            ("user", name),
            ("bot", "nội dung tin nhắn là gì?"),
            ("user", content),
        ]
        rw = f"Tôi muốn gửi tin nhắn '{content}' cho {name}."
        out.append(make_sample(turns, rw, "send_sms", "multi_slot", "messaging"))
    return out


def gen_3t_multi_slot_ac_full():
    """AC: bật → bot hỏi nhiệt độ → user → bot hỏi chế độ → user."""
    out = []
    for t, mode in itertools.product([20, 22, 24, 26], AC_MODES):
        turns = [
            ("user", "bật điều hoà"),
            ("bot", "bạn muốn để bao nhiêu độ?"),
            ("user", f"{t} độ"),
            ("bot", "chế độ gì ạ?"),
            ("user", mode),
        ]
        rw = f"Tôi muốn bật điều hoà ở {t} độ, chế độ {mode}."
        out.append(make_sample(turns, rw, "set_ac_temperature_with_mode", "multi_slot", "climate"))
    return out


def gen_3t_multi_slot_call_then_speaker():
    """gọi → bot hỏi contact → user trả lời → bot xác nhận → user xin loa ngoài."""
    out = []
    for name in CONTACTS:
        turns = [
            ("user", "gọi điện"),
            ("bot", "gọi cho ai ạ?"),
            ("user", name),
            ("bot", f"Đang chuẩn bị gọi {name}."),
            ("user", "bật loa ngoài luôn"),
        ]
        rw = f"Tôi muốn gọi cho {name} qua loa ngoài."
        out.append(make_sample(turns, rw, "call_contact_speaker", "multi_slot", "calling"))
    return out


def gen_3t_correction_song():
    """user yêu cầu bài X → bot xác nhận → user đổi sang bài Y."""
    out = []
    for x, y in itertools.permutations(SONGS, 2):
        turns = [
            ("user", f"mở bài {x}"),
            ("bot", f"Đang phát {x}."),
            ("user", f"khoan, đổi sang bài {y} đi"),
            ("bot", f"Bạn chắc muốn đổi sang {y}?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = f"Tôi muốn đổi bài đang phát sang {y}."
        out.append(make_sample(turns, rw, "play_music", "correction", "music"))
        if len(out) >= 60:
            break
    return out


def gen_3t_correction_nav():
    """user yêu cầu đi X → bot xác nhận → user đổi đích → bot hỏi lại → user xác nhận."""
    out = []
    pairs = list(itertools.permutations(PLACES, 2))
    random.shuffle(pairs)
    for x, y in pairs[:40]:
        turns = [
            ("user", f"dẫn đến {x}"),
            ("bot", f"Đang dẫn đến {x}."),
            ("user", f"đổi điểm đến thành {y}"),
            ("bot", f"Bạn muốn đổi đích sang {y}, đúng không?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = f"Tôi muốn đổi điểm đến sang {y}."
        out.append(make_sample(turns, rw, "navigate_to_location", "correction", "navigation"))
    return out


def gen_3t_compare_then_pick():
    """user tìm POI → bot đưa 2 lựa chọn → user hỏi cái nào gần hơn → bot trả lời → user chọn."""
    out = []
    for cat in POI_CATEGORIES:
        p1, p2 = random.sample(PLACES, 2)
        d1, d2 = sorted(random.sample(DISTANCES, 2))
        turns = [
            ("user", f"tìm {cat} gần đây"),
            ("bot", f"Có {p1} cách {d2}km và {p2} cách {d1}km."),
            ("user", "cái nào gần hơn?"),
            ("bot", f"{p2} gần hơn, cách {d1}km."),
            ("user", "đến đó đi"),
        ]
        rw = nav_rewrite(p2)
        out.append(make_sample(turns, rw, "navigate_to_location", "comparison", "navigation"))
    return out


def gen_3t_conditional_charge():
    """user hỏi range → bot trả lời → user yêu cầu charging nếu pin yếu → bot xác nhận → user confirm."""
    out = []
    for batt, rng in itertools.product(BATTERY_LEVELS, RANGE_KM):
        turns = [
            ("user", "pin còn bao nhiêu?"),
            ("bot", f"Pin còn {batt}%, đi được khoảng {rng}km."),
            ("user", "nếu vậy tìm trạm sạc gần nhất"),
            ("bot", "Bạn có yêu cầu hãng sạc nào không?"),
            ("user", "hãng nào cũng được"),
        ]
        rw = "Tôi muốn tìm trạm sạc gần nhất, không yêu cầu hãng cụ thể."
        out.append(make_sample(turns, rw, "find_charging_station", "conditional", "navigation"))
    return out


def gen_3t_disambig_call():
    """user gọi tên chung chung (Anh) → bot hỏi anh nào → user → bot xác nhận → user."""
    out = []
    matched = [c for c in CONTACTS if c.startswith("anh ") or c in ("Bố", "sếp", "chồng")]
    for name in matched:
        turns = [
            ("user", "gọi cho anh"),
            ("bot", "bạn muốn gọi anh nào?"),
            ("user", name),
            ("bot", f"Xác nhận gọi {name}?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = f"Tôi muốn gọi cho {name}."
        out.append(make_sample(turns, rw, "call_contact", "disambiguation", "calling"))
    return out


def gen_3t_chained_climate_window():
    """user trên xe nóng → bot gợi ý → user yêu cầu AC + mở cửa kính."""
    out = []
    for t, w in itertools.product([20, 22, 24], WINDOWS):
        turns = [
            ("user", "nóng quá"),
            ("bot", "Tôi có thể bật điều hoà hoặc mở cửa kính, bạn muốn cái nào?"),
            ("user", f"bật điều hoà {t} độ"),
            ("bot", f"Đã bật điều hoà {t} độ."),
            ("user", f"và hé {w} một chút"),
        ]
        rw = f"Tôi muốn bật điều hoà {t} độ và mở hé {w}."
        out.append(make_sample(turns, rw, "chained_climate_window", "chained_intent", "multi"))
    return out


def gen_3t_radio_then_volume():
    out = []
    for st, v in itertools.product(RADIO_STATIONS, VOLUME_LEVELS):
        turns = [
            ("user", "mở radio"),
            ("bot", "bạn muốn nghe đài nào?"),
            ("user", st),
            ("bot", f"Đang phát {st}."),
            ("user", f"để âm lượng {v}"),
        ]
        rw = f"Tôi muốn phát đài {st} ở âm lượng {v}."
        out.append(make_sample(turns, rw, "play_radio_volume", "multi_slot", "music"))
    return out


def gen_3t_smart_home_garage():
    """user sắp về nhà → bot xác nhận eta → user yêu cầu mở cổng + bật đèn."""
    out = []
    for dev in HOME_DEVICES[:3]:
        turns = [
            ("user", "tôi sắp về tới nhà"),
            ("bot", "Còn khoảng 5 phút nữa."),
            ("user", f"mở cổng nhà trước và {pick(['bật', 'mở'])} {dev}"),
            ("bot", f"Xác nhận mở cổng và bật {dev}?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = f"Tôi muốn mở cổng nhà và bật {dev}."
        out.append(make_sample(turns, rw, "smart_home_combo", "chained_intent", "smart_home"))
    return out


# ===========================================================================
# 4-USER-TURN GENERATORS (7 messages: u-b-u-b-u-b-u)
# ===========================================================================


def gen_4t_full_nav_dialog():
    """nav từ đầu đến cuối với 4 slot: POI category → tên → tuỳ chọn → confirm."""
    out = []
    for cat, place, pref in itertools.product(POI_CATEGORIES[:4], PLACES[:5], ROUTE_PREFS[:3]):
        turns = [
            ("user", f"tìm {cat}"),
            ("bot", "có nhiều kết quả, bạn nhớ tên cụ thể không?"),
            ("user", place),
            ("bot", f"Có {place}, bạn muốn đi không?"),
            ("user", "đi"),
            ("bot", "bạn ưu tiên lộ trình thế nào?"),
            ("user", pref),
        ]
        rw = nav_rewrite(place, pref=pref)
        out.append(make_sample(turns, rw, "navigate_with_preference", "multi_slot", "navigation"))
    return out


def gen_4t_full_sms_dialog():
    """sms với 4 slot: trigger → contact → nội dung → confirm gửi."""
    out = []
    for name, content in itertools.product(CONTACTS, SMS_CONTENTS):
        turns = [
            ("user", "tôi muốn nhắn tin"),
            ("bot", "gửi cho ai?"),
            ("user", name),
            ("bot", "nội dung là gì?"),
            ("user", content),
            ("bot", f"Tôi sẽ gửi '{content}' cho {name}, đúng chưa?"),
            ("user", pick(USER_CONFIRM)),
        ]
        rw = f"Tôi muốn gửi tin nhắn '{content}' cho {name}."
        out.append(make_sample(turns, rw, "send_sms", "multi_slot", "messaging"))
    return out


def gen_4t_full_ac_dialog():
    """AC với 4 slot: trigger → nhiệt độ → chế độ → vùng."""
    out = []
    combos = list(itertools.product([20, 22, 24, 26], AC_MODES[:3], AC_ZONES[:3]))
    random.shuffle(combos)
    for t, mode, zone in combos[:60]:
        turns = [
            ("user", "bật điều hoà"),
            ("bot", "đặt bao nhiêu độ?"),
            ("user", f"{t} độ"),
            ("bot", "chế độ nào?"),
            ("user", mode),
            ("bot", "áp dụng cho vùng nào?"),
            ("user", zone),
        ]
        rw = f"Tôi muốn bật điều hoà {t} độ, chế độ {mode}, cho {zone}."
        out.append(make_sample(turns, rw, "set_ac_full", "multi_slot", "climate"))
    return out


def gen_4t_full_music_dialog():
    """music: bật → chọn nguồn → chọn playlist/bài → chọn volume."""
    out = []
    for pl, v in itertools.product(PLAYLISTS[:3], VOLUME_LEVELS[:4]):
        turns = [
            ("user", "tôi muốn nghe nhạc"),
            ("bot", "bạn muốn nghe từ Spotify hay nhạc trong xe?"),
            ("user", "Spotify"),
            ("bot", "playlist nào?"),
            ("user", pl),
            ("bot", "âm lượng bao nhiêu?"),
            ("user", f"{v}"),
        ]
        rw = f"Tôi muốn phát playlist {pl} trên Spotify ở âm lượng {v}."
        out.append(make_sample(turns, rw, "play_playlist_source_volume", "multi_slot", "music"))
    return out


def gen_4t_charging_full():
    """alert pin → user yêu cầu sạc → bot hỏi hãng → user → bot hỏi nhanh hay thường → user."""
    out = []
    for batt, net in itertools.product([10, 15, 20], CHARGING_NETWORKS):
        for speed in ["sạc nhanh", "sạc thường"]:
            turns = [
                ("bot", f"Pin còn {batt}%, tôi đề xuất tìm trạm sạc."),
                ("user", "tìm trạm sạc gần nhất"),
                ("bot", "bạn muốn hãng nào?"),
                ("user", net),
                ("bot", "sạc nhanh hay sạc thường?"),
                ("user", speed),
                ("bot", f"Xác nhận tìm trạm {net} có {speed} gần nhất?"),
                ("user", pick(USER_CONFIRM)),
            ]
            rw = f"Tôi muốn tìm trạm sạc {net} hỗ trợ {speed} gần nhất."
            out.append(make_sample(turns, rw, "find_charging_station", "multi_slot", "navigation"))
    return out


def gen_4t_correction_then_chain():
    """user yêu cầu nav X + nhạc A → bot xác nhận → user đổi nhạc → bot xác nhận → user thêm cuộc gọi."""
    out = []
    triples = []
    for place, song, name in itertools.product(PLACES[:4], SONGS[:3], CONTACTS[:3]):
        triples.append((place, song, name))
    random.shuffle(triples)
    for place, song, name in triples[:40]:
        new_song = pick([s for s in SONGS if s != song])
        turns = [
            ("user", f"đi đến {place} và mở bài {song}"),
            ("bot", f"Đã đặt đích {place}, đang phát {song}."),
            ("user", f"đổi nhạc sang {new_song}"),
            ("bot", f"Đã đổi sang {new_song}."),
            ("user", f"và gọi {name} qua loa ngoài"),
        ]
        rw = (
            f"Tôi muốn {pick(CHAINED_NAV_VERBS)} {place}, phát bài {new_song}, "
            f"và gọi {name} qua loa ngoài."
        )
        out.append(make_sample(turns, rw, "chained_nav_music_call", "correction", "multi"))
    return out


def gen_4t_calendar_then_nav():
    """user hỏi lịch → bot trả lời → user yêu cầu nav đến địa điểm trong lịch → bot hỏi giờ rời → user."""
    out = []
    meetings = [
        ("cuộc họp với khách hàng", "VinFast Times City", "14:00"),
        ("khám sức khoẻ", "bệnh viện Bạch Mai", "9:00"),
        ("hẹn ăn trưa", "Lotte Center", "11:30"),
        ("đón con", "Đại học Bách Khoa", "17:00"),
    ]
    leave_offsets = ["ngay bây giờ", "trong 15 phút nữa", "trong 30 phút nữa"]
    for (event, loc, hr), leave in itertools.product(meetings, leave_offsets):
        turns = [
            ("user", "lịch hôm nay có gì?"),
            ("bot", f"Bạn có {event} lúc {hr} tại {loc}."),
            ("user", f"dẫn đường tới {loc}"),
            ("bot", "bạn muốn rời đi khi nào?"),
            ("user", leave),
        ]
        rw = nav_rewrite(loc, depart=leave)
        out.append(make_sample(turns, rw, "navigate_scheduled", "multi_slot", "navigation"))
    return out


def gen_4t_drive_mode_full():
    """user đổi chế độ lái → bot xác nhận → user yêu cầu thêm regen cao → bot xác nhận → user thêm cruise."""
    out = []
    for mode, spd in itertools.product(DRIVE_MODES[:3], CRUISE_SPEEDS):
        turns = [
            ("user", f"đổi chế độ lái sang {mode}"),
            ("bot", f"Đã đổi sang chế độ {mode}."),
            ("user", "đặt regen ở mức cao nhất"),
            ("bot", "Đã đặt regen mức cao."),
            ("user", f"bật cruise control ở {spd} km/h"),
        ]
        rw = (
            f"Tôi muốn đặt chế độ lái {mode}, regen mức cao, và cruise control "
            f"ở {spd} km/h."
        )
        out.append(make_sample(turns, rw, "drive_mode_full", "multi_slot", "driver_assist"))
    return out


def gen_4t_ignore_irrelevant_chain():
    """user hỏi thời tiết → bot trả lời (mention nhiệt) → user nói tới điểm khác → bot trả lời → user yêu cầu AC ở nhiệt khác hẳn."""
    out = []
    for outdoor, target in itertools.product([35, 38, 40], [20, 22, 24]):
        turns = [
            ("user", "hôm nay thời tiết thế nào?"),
            ("bot", f"Hôm nay nắng, khoảng {outdoor} độ."),
            ("user", "đường có tắc không?"),
            ("bot", "Đường Trường Chinh đang tắc nhẹ."),
            ("user", f"bật điều hoà {target} độ chế độ tự động"),
        ]
        rw = f"Tôi muốn bật điều hoà {target} độ ở chế độ tự động."
        out.append(make_sample(turns, rw, "set_ac_temperature_with_mode", "irrelevant_context", "climate"))
    return out


def gen_4t_defrost_full():
    """user kêu kính mờ → bot gợi ý → user bật defrost → bot hỏi mức quạt → user → bot hỏi nhiệt độ → user."""
    out = []
    for fan, t in itertools.product(FAN_LEVELS[:3], [22, 24, 26]):
        turns = [
            ("user", "kính mờ quá"),
            ("bot", "Tôi có thể bật sấy kính giúp bạn."),
            ("user", "ừ bật defrost đi"),
            ("bot", "mức quạt bao nhiêu?"),
            ("user", f"mức {fan}"),
            ("bot", "kèm nhiệt độ?"),
            ("user", f"{t} độ"),
        ]
        rw = f"Tôi muốn bật sấy kính ở mức quạt {fan} và {t} độ."
        out.append(make_sample(turns, rw, "defrost_with_settings", "multi_slot", "climate"))
    return out


# ===========================================================================
# REGISTRATION
# ===========================================================================

GENERATORS: list[Callable[[], list[dict]]] = [
    # 2-user-turn
    gen_2t_slot_ac_temp,
    gen_2t_slot_ac_mode,
    gen_2t_slot_fan_zone,
    gen_2t_slot_call_contact,
    gen_2t_slot_sms_recipient,
    gen_2t_slot_music_song,
    gen_2t_slot_music_playlist,
    gen_2t_slot_nav_place,
    gen_2t_slot_window,
    gen_2t_slot_seat_heat,
    gen_2t_slot_cruise_speed,
    gen_2t_confirm_call,
    gen_2t_cancel_call,
    gen_2t_adjust_temp,
    gen_2t_adjust_volume,
    gen_2t_reference_nav,
    gen_2t_reference_music,
    gen_2t_reference_call,
    gen_2t_disambig_nav,
    gen_2t_chained_ac_music,
    gen_2t_irrelevant_outdoor_temp,
    gen_2t_bot_alert_charge,
    # 3-user-turn
    gen_3t_multi_slot_nav,
    gen_3t_multi_slot_sms,
    gen_3t_multi_slot_ac_full,
    gen_3t_multi_slot_call_then_speaker,
    gen_3t_correction_song,
    gen_3t_correction_nav,
    gen_3t_compare_then_pick,
    gen_3t_conditional_charge,
    gen_3t_disambig_call,
    gen_3t_chained_climate_window,
    gen_3t_radio_then_volume,
    gen_3t_smart_home_garage,
    # 4-user-turn
    gen_4t_full_nav_dialog,
    gen_4t_full_sms_dialog,
    gen_4t_full_ac_dialog,
    gen_4t_full_music_dialog,
    gen_4t_charging_full,
    gen_4t_correction_then_chain,
    gen_4t_calendar_then_nav,
    gen_4t_drive_mode_full,
    gen_4t_ignore_irrelevant_chain,
    gen_4t_defrost_full,
]


def generate(target: int, seed: int) -> list[dict]:
    random.seed(seed)
    pool: list[dict] = []
    for gen in GENERATORS:
        pool.extend(gen())

    # Dedup by (dialogue, rewrite)
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
        # Stratified downsample theo (pattern, user_turns) để cân bằng.
        buckets: dict[tuple, list[dict]] = {}
        for s in unique:
            k = (s["meta"]["pattern"], s["meta"]["user_turns"])
            buckets.setdefault(k, []).append(s)
        per_bucket = max(1, target // len(buckets))
        sampled = []
        for items in buckets.values():
            sampled.extend(items[:per_bucket])
        if len(sampled) < target:
            extras = [s for s in unique if s not in sampled]
            sampled.extend(extras[: target - len(sampled)])
        random.shuffle(sampled)
        return sampled[:target]
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000, help="Số mẫu mục tiêu (0 = giữ toàn bộ).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dialogues_multi_turn.jsonl"))
    args = parser.parse_args()

    samples = generate(args.target, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples → {args.output}")

    # Phân phối theo pattern × user_turns
    dist: dict[tuple, int] = {}
    for s in samples:
        k = (s["meta"]["pattern"], s["meta"]["user_turns"])
        dist[k] = dist.get(k, 0) + 1
    print("\nDistribution (pattern, user_turns):")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")


if __name__ == "__main__":
    main()
