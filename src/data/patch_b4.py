"""B4 targeted training patch — narrow hard samples for the failure modes the v2.0
bench error-analysis surfaced (see reports/v2.1_error_analysis_next.md). Same proven
recipe as patch_b2b3: plain SFT positives of the exact tricky structure, a self-check
asserts each gold, and a leak-check drops any gold colliding with the eval bench.

clause-preservation (chống over-compression — SFT đang vứt vế phụ/phủ định):
  n1 keep_negative_clause — vế khẳng định (ở context) + vế "không/đừng/tránh X" ở lượt
                            cuối phải CÙNG sống trong gold. [neg drop, 81 ca]
  n2 anti_inversion       — câu chỉ-phủ-định giữ ĐÚNG cực; cấm đảo sang khẳng định/đối
                            nghĩa ("Không bật Lane Assist" KHÔNG thành "Bật…"). [inversion]
  n3 exclusion_in_list    — danh sách + vế loại trừ ("gọi A,B,C, không gọi D") giữ cả vế
                            loại trừ. [correction 27 ca]

slot-retention (chống rớt slot bổ nghĩa — 302 ca slot-incomplete):
  s1 keep_qualifier       — slot bổ nghĩa nói ở lượt GIỮA (brand/cổng/app/nguồn) phải
                            sống vào lệnh cuối. [VinFast, CCS2, Google Maps, offline]
  s2 compound_two_actions — 2 hành động trong 1 lượt giữ cả hai, không gom về một.

confirmation (bind lời đồng ý vào hành động đề xuất — lỗ hổng demo, bench mù):
  n4 confirm_proposed     — assistant ĐỀ XUẤT hành động → user đồng ý tối giản ("ờ"/"ừ"
                            /"ok") → gold = chính hành động đề xuất, resolve đủ slot;
                            cấm bịa địa danh/intent ngoài hội thoại. [idx demo "ờ"]

v2.2 adds p1 pronoun_resolution and c1 language_control, then uses severity-weighted
per-generator counts (1700 samples total); --per is a fallback for any generator not
listed in WEIGHTS.

Run: /opt/homebrew/bin/python3.11 -m src.data.patch_b4 --per 200
Kaggle: chạy SAU split_data.py (cell 5b), song song patch_b2b3 --inplace.
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

# reuse the proven framework — single source of truth for prompt/format/leak-check
from src.data.patch_b2b3 import (
    SYSTEM_PROMPT, sample, bench_golds, pick, word_in, cap, TRAIN,
)

OUT = Path("data/processed/patch_b4.jsonl")
TRAIN_V3 = Path("data/processed/train_v3.jsonl")
NEG_WORDS = ("không", "đừng", "tránh")


def has_neg(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in NEG_WORDS)


def join_vi(items):
    """Vietnamese list join: 'a, b và c'."""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " và " + items[-1]


# =========================================================== n1 keep_negative_clause
# per-domain (positive action stated in an EARLIER turn) + (trailing negative clause
# stated in the FINAL turn). gold must restore BOTH — the bug drops one of them.
N1 = {
    "driver_assist": {
        "pos": ["bật hỗ trợ giữ làn đường", "bật ga tự động Cruise Control",
                "bật cảnh báo điểm mù", "bật hỗ trợ phanh khẩn cấp"],
        "neg": ["không bật Auto Pilot", "không bật tự động chuyển làn",
                "đừng bật cảnh báo rung vô lăng", "đừng bật chế độ thể thao"],
    },
    "navigation": {
        "pos": ["chỉ đường về sân bay Nội Bài theo tuyến nhanh nhất",
                "dẫn đường tới Hồ Gươm", "dẫn đường về nhà theo quốc lộ 1"],
        "neg": ["không đi qua cầu Nhật Tân", "không đi vào cao tốc",
                "tránh đường Phạm Văn Đồng", "không đi đường vòng"],
    },
    "charging": {
        "pos": ["sạc xe tại trạm VinFast Hải Dương",
                "sạc xe tại điểm sạc Vincom Ocean Park"],
        "neg": ["không sạc quá 90%", "đừng dùng sạc nhanh", "không sạc quá 80%"],
    },
    "music": {
        "pos": ["phát nhạc Vpop trên Apple Music", "phát playlist EDM Drive",
                "phát nhạc acoustic của Soobin Hoàng Sơn"],
        "neg": ["không phát bài của Sơn Tùng M-TP", "đừng bật shuffle",
                "không phát bản gốc"],
    },
    "climate": {
        "pos": ["chỉnh điều hoà lên 24 độ, hướng gió ra giữa",
                "bật điều hoà hàng ghế sau", "mở cửa sổ trời 50%"],
        "neg": ["không thổi xuống chân", "đừng bật máy nén lạnh",
                "không bật chế độ sấy kính"],
    },
}


def n1_keep_negative_clause():
    dom = pick(list(N1))
    pos = pick(N1[dom]["pos"])
    neg = pick(N1[dom]["neg"])
    open_g = pick(["Vâng, em ghi nhận. Anh/chị cần lưu ý thêm gì không ạ?",
                   "Được ạ. Còn điều gì cần em chú ý không?",
                   "Em thực hiện ngay. Có yêu cầu nào nữa không ạ?"])
    final_u = pick([f"À mà {neg} nhé", f"Nhưng {neg} nha", f"Nhớ là {neg}",
                    f"{cap(neg)} nhé"])
    turns = [
        ("human", f"{cap(pos)}."),
        ("gpt", open_g),
        ("human", final_u),
    ]
    gold = f"{cap(pos)}, {neg}."
    s = sample(turns, gold, dom, src="patch_b4")
    # both clauses survive + a negation marker is present
    assert pos.lower() in gold.lower() and neg.lower() in gold.lower(), (pos, neg, gold)
    assert has_neg(gold), gold
    return s


# =================================================================== n2 anti_inversion
# negation-only command: gold keeps the negative verb; the ANTONYM verb must NOT appear
# (that would be the inversion bug, e.g. "Không mở" → "Đóng", "Không bật" → "Tắt").
ANTONYM = {"bật": "tắt", "tắt": "bật", "mở": "đóng", "đóng": "mở"}
N2 = {
    "bật": ["chế độ tiết kiệm pin", "đèn viền nội thất", "âm thanh cảnh báo va chạm",
            "khóa trẻ em", "chế độ một bàn đạp", "sưởi ghế lái", "sưởi vô lăng",
            "đèn chào mừng", "chế độ off-road", "cảnh báo vượt tốc độ",
            "khởi động từ xa", "đèn gầm", "chế độ massage ghế", "đèn đọc sách sau",
            "hỗ trợ đỗ xe song song", "giữ phanh tự động"],
    "mở": ["cửa sổ hành khách trước", "nắp bình sạc", "gương chiếu hậu",
           "cửa sổ trời phía sau", "rèm cửa sổ trời", "cửa kính tam giác",
           "khoang để đồ trung tâm"],
    "tắt": ["đèn nội thất nền", "quạt gió tốc độ cao", "âm báo thắt dây an toàn",
            "chế độ chờ", "đèn taplo", "loa siêu trầm"],
    "đóng": ["cửa sổ hành khách sau", "cửa sổ trời nghiêng", "nắp khoang động cơ"],
}


def n2_anti_inversion():
    verb = pick(list(N2))
    item = pick(N2[verb])
    final_u = pick([f"Đừng {verb} {item} nhé", f"{cap(item)} thì đừng {verb} nha",
                    f"Nhớ đừng {verb} {item}", f"{cap(item)} khỏi {verb} nhé",
                    f"Thôi đừng {verb} {item}"])
    turns = [("human", final_u)]
    gold = f"Không {verb} {item}."
    s = sample(turns, gold, "vehicle", ctx=False, src="patch_b4")
    assert has_neg(gold), gold
    # the inversion (antonym verb) must not leak into the gold
    assert not word_in(ANTONYM[verb], gold.lower()), (verb, gold)
    return s


# ================================================================ n3 exclusion_in_list
N3_NAMES = ["bố Văn Thịnh", "mẹ Ngọc Mai", "em Bảo Châu", "anh Minh Đức",
            "chị Phương Thảo", "anh Quang Huy", "chị Bích Ngọc", "anh Hoàng Phúc",
            "anh Trường Giang", "chị Ngọc Anh", "anh Quốc Bảo", "chú Hải"]


def n3_exclusion_in_list():
    verb = pick(["Gọi nhóm", "Gọi hội nghị"])
    pool = random.sample(N3_NAMES, 4)
    incl, excl = pool[:3], pool[3]
    inc_str = join_vi(incl)
    final_u = pick([f"{verb} cho {inc_str}, đừng gọi {excl} nhé",
                    f"{verb} cho {inc_str} thôi, không gọi {excl}"])
    turns = [("human", final_u)]
    gold = f"{verb} cho {inc_str}, không gọi {excl}."
    s = sample(turns, gold, "calling", ctx=False, src="patch_b4")
    assert excl in gold and "không gọi" in gold.lower(), (excl, gold)
    for nm in incl:
        assert nm in gold, (nm, gold)
    return s


# ================================================================== s1 keep_qualifier
# qualifier slot (brand/port/app/source) stated in a MIDDLE turn must survive into the
# final short command — the bug drops it (idx: sạc VinFast cổng CCS2; navigate Google Maps).
S1_BRANDS = ["VinFast", "EVN", "VinFast Power", "EVN Genco", "Porsche Charging"]
S1_PORTS = ["CCS2", "DC", "Type 2", "CHAdeMO"]
S1_PLACES = ["Vincom Mega Mall Ocean Park", "trạm dừng nghỉ Hải Dương",
             "Aeon Mall Long Biên", "Vinhomes Grand Park", "Lotte Mall Tây Hồ",
             "trạm dừng nghỉ Pháp Vân"]
S1_TIMES = ["19 giờ", "7 giờ sáng", "21 giờ", "6 giờ chiều", "20 giờ 30", "5 giờ sáng"]
S1_APPS = ["Google Maps", "Apple Maps", "VietMap", "Vietmap Live", "Here WeGo"]
S1_DESTS = ["làn D1 ga quốc tế sân bay Tân Sơn Nhất", "Bảo tàng Dân tộc học",
            "khu công nghệ cao Hòa Lạc", "bến xe Mỹ Đình", "phố cổ Hội An",
            "khu nghỉ dưỡng Bà Nà Hills"]
S1_PLAYLISTS = ["EDM Drive", "Chill Lofi", "Top Hits Vpop", "Acoustic Cafe",
                "Workout Energy", "Indie Việt"]
S1_SOURCES = ["offline từ bộ nhớ xe", "từ thẻ nhớ", "offline trong USB",
              "offline từ điện thoại", "từ ổ cứng xe"]


def s1_keep_qualifier():
    kind = pick(["charging", "navigation", "music"])
    if kind == "charging":
        brand, port, place, time = (pick(S1_BRANDS), pick(S1_PORTS),
                                    pick(S1_PLACES), pick(S1_TIMES))
        turns = [
            ("human", "Đặt lịch sạc xe giúp tôi."),
            ("gpt", "Bạn muốn sạc ở trạm nào ạ?"),
            ("human", f"Trạm {brand} ở {place}."),
            ("gpt", "Dùng cổng sạc nào ạ?"),
            ("human", f"Cổng {port}."),
            ("gpt", "Mấy giờ bắt đầu ạ?"),
            ("human", f"Lúc {time}."),
        ]
        gold = f"Đặt lịch sạc {brand} bằng cổng {port} tại {place} lúc {time}."
        anchors = [brand, port, place, time]
    elif kind == "navigation":
        app, dest = pick(S1_APPS), pick(S1_DESTS)
        turns = [
            ("human", "Dẫn đường giúp tôi."),
            ("gpt", "Bạn muốn tới đâu ạ?"),
            ("human", f"{cap(dest)}."),
            ("gpt", "Dùng ứng dụng bản đồ nào ạ?"),
            ("human", f"{app}."),
            ("gpt", "Em bắt đầu nhé?"),
            ("human", "Đi thôi."),
        ]
        gold = f"Dẫn đường tới {dest} bằng {app}."
        anchors = [dest, app]
    else:  # music
        pl, src = pick(S1_PLAYLISTS), pick(S1_SOURCES)
        turns = [
            ("human", "Phát nhạc cho tôi."),
            ("gpt", "Bạn muốn phát playlist nào ạ?"),
            ("human", f"Playlist '{pl}'."),
            ("gpt", "Phát từ nguồn nào ạ?"),
            ("human", f"{cap(src)}."),
            ("gpt", "Bật chế độ ngẫu nhiên không ạ?"),
            ("human", "Bật shuffle luôn."),
        ]
        gold = f"Phát playlist '{pl}' {src}, bật shuffle."
        anchors = [pl, src, "shuffle"]
    s = sample(turns, gold, kind, src="patch_b4")
    for a in anchors:
        assert a in gold, (a, gold)  # every middle-turn qualifier survives
    return s


# ============================================================ s2 compound_two_actions
S2_A = [("bật đèn phòng khách", "đèn phòng khách"), ("khóa toàn bộ cửa", "khóa"),
        ("bật điều hoà 24 độ", "điều hoà"), ("đóng cửa sổ trời", "cửa sổ trời"),
        ("bật đèn sân vườn", "đèn sân vườn"), ("bật bình nóng lạnh", "bình nóng lạnh"),
        ("mở cổng gara", "cổng gara"), ("bật đèn hành lang", "đèn hành lang"),
        ("hạ rèm phòng khách", "rèm phòng khách"), ("bật đèn bếp", "đèn bếp")]
S2_B = [("đặt thermostat 22 độ", "thermostat"), ("bật quạt thông gió", "quạt thông gió"),
        ("kéo rèm phòng ngủ xuống", "rèm phòng ngủ"), ("bật chế độ ngủ cho nhà", "chế độ ngủ"),
        ("bật máy lọc không khí", "máy lọc không khí"), ("bật robot hút bụi", "robot hút bụi"),
        ("tưới vườn tự động", "tưới vườn"), ("bật loa phòng khách", "loa phòng khách"),
        ("khóa cửa sau", "cửa sau"), ("bật camera an ninh", "camera an ninh")]


def s2_compound_two_actions():
    (a_text, a_tok) = pick(S2_A)
    (b_text, b_tok) = pick(S2_B)
    while b_tok == a_tok:
        (b_text, b_tok) = pick(S2_B)
    final_u = pick([f"{cap(a_text)} và {b_text} nhé",
                    f"{cap(a_text)}, rồi {b_text} luôn",
                    f"Vừa {a_text} vừa {b_text}"])
    turns = [("human", final_u)]
    gold = f"{cap(a_text)} và {b_text}."
    s = sample(turns, gold, "smart_home", ctx=False, src="patch_b4")
    assert a_tok.lower() in gold.lower() and b_tok.lower() in gold.lower(), (a_tok, b_tok, gold)
    return s


# ============================================================== n4 confirm_proposed
# (opening, proposed action phrase, gold, keytoken, domain) — user affirms the proposal.
N4_PROPOSALS = [
    ("Trời nóng quá.", "bật điều hoà 24 độ cho mát", "Bật điều hoà 24 độ.", "điều hoà", "climate"),
    ("Tôi sắp về nhà.", "dẫn đường về nhà theo tuyến nhanh nhất", "Dẫn đường về nhà theo tuyến nhanh nhất.", "Dẫn đường về nhà", "navigation"),
    ("Pin xe sắp cạn.", "tìm trạm sạc VinFast gần nhất", "Tìm trạm sạc VinFast gần nhất.", "trạm sạc VinFast", "charging"),
    ("Buồn ngủ quá.", "phát một playlist nhạc sôi động", "Phát một playlist nhạc sôi động.", "playlist nhạc sôi động", "music"),
    ("Trong xe tối quá.", "bật đèn nội thất lên", "Bật đèn nội thất.", "đèn nội thất", "vehicle"),
    ("Tôi muốn gọi cho vợ.", "gọi cho chị Lan qua số di động", "Gọi cho chị Lan qua số di động.", "chị Lan", "calling"),
    ("Sắp tới nhà rồi.", "bật điều hoà ở nhà trước", "Bật điều hoà ở nhà.", "điều hoà", "smart_home"),
    ("Đường này hay tắc.", "tìm tuyến đường khác ít kẹt xe hơn", "Tìm tuyến đường khác ít kẹt xe hơn.", "tuyến đường khác", "navigation"),
    ("Trời lạnh thật.", "bật sưởi ghế lái", "Bật sưởi ghế lái.", "sưởi ghế lái", "climate"),
    ("Kính bị mờ hết rồi.", "bật sấy kính trước", "Bật sấy kính trước.", "sấy kính trước", "climate"),
    ("Nhạc đang nhỏ quá.", "tăng âm lượng lên mức 15", "Tăng âm lượng lên mức 15.", "âm lượng", "music"),
    ("Tôi cần đổ xăng à nhầm sạc.", "tìm trạm sạc nhanh DC trên đường đi", "Tìm trạm sạc nhanh DC trên đường đi.", "trạm sạc nhanh DC", "charging"),
    ("Sắp tới giờ họp rồi.", "gọi video cho anh Minh Đức", "Gọi video cho anh Minh Đức.", "anh Minh Đức", "calling"),
    ("Tôi muốn nghe tin tức.", "phát bản tin thời sự buổi sáng", "Phát bản tin thời sự buổi sáng.", "bản tin thời sự", "music"),
    ("Xe bẩn quá nhỉ.", "tìm trạm rửa xe gần nhất", "Tìm trạm rửa xe gần nhất.", "trạm rửa xe", "navigation"),
    ("Trời tối rồi đấy.", "bật đèn pha tự động", "Bật đèn pha tự động.", "đèn pha tự động", "vehicle"),
    ("Tôi đói rồi.", "tìm quán phở gần đây", "Tìm quán phở gần đây.", "quán phở", "navigation"),
    ("Sắp ra cao tốc rồi.", "bật ga tự động Cruise Control", "Bật ga tự động Cruise Control.", "Cruise Control", "driver_assist"),
    ("Con đang ngủ ở ghế sau.", "giảm âm lượng nhạc xuống mức 5", "Giảm âm lượng nhạc xuống mức 5.", "âm lượng nhạc", "music"),
    ("Tôi về tới cổng rồi.", "mở cổng nhà và bật đèn sân", "Mở cổng nhà và bật đèn sân.", "cổng nhà", "smart_home"),
    ("Sương mù dày quá.", "bật đèn sương mù trước", "Bật đèn sương mù trước.", "đèn sương mù trước", "vehicle"),
    ("Tôi cần gọi gấp cho bố.", "gọi cho bố Văn Thịnh", "Gọi cho bố Văn Thịnh.", "bố Văn Thịnh", "calling"),
    ("Ngồi lâu mỏi lưng quá.", "bật massage ghế lái", "Bật massage ghế lái.", "massage ghế lái", "vehicle"),
    ("Tôi muốn đi Đà Lạt.", "dẫn đường tới Đà Lạt theo đường đèo", "Dẫn đường tới Đà Lạt theo đường đèo.", "Đà Lạt", "navigation"),
]
AFFIRM = ["ờ", "ừ", "ừm", "uhm", "ok", "oki", "okê", "vâng", "dạ", "được",
          "đúng", "chuẩn", "đúng rồi", "ừ nhỉ", "ờ được", "ừ ok", "ừm đúng"]


def n4_confirm_proposed():
    opening, proposal, gold, keytok, dom = pick(N4_PROPOSALS)
    yn = pick(AFFIRM)
    turns = [
        ("human", opening),
        ("gpt", f"Em {proposal} nhé ạ?"),
        ("human", yn),
    ]
    s = sample(turns, gold, dom, src="patch_b4")
    assert keytok in gold, (keytok, gold)
    assert len(yn) <= 8, yn  # final turn really is a minimal affirmation
    return s


# ============================================================ p1 pronoun_resolution
# lượt cuối dùng đại từ/chỉ định → gold dùng referent đã resolve từ ngữ cảnh, KHÔNG
# để lại đại từ. pattern tệ nhất bench (52.6%) và 0 coverage trước v2.2.
P1_SONGS = [("Hào Quang", "Đức Phúc"), ("Chúng Ta Của Hiện Tại", "Sơn Tùng M-TP"),
            ("Nơi Này Có Anh", "Sơn Tùng M-TP"), ("Lạc Trôi", "Sơn Tùng M-TP"),
            ("Ngày Đầu Tiên", "Đức Phúc"), ("Đưa Em Về Nhà", "Grey D"),
            ("Có Chàng Trai Viết Lên Cây", "Phan Mạnh Quỳnh"), ("Trốn Tìm", "Đen"),
            ("Bài Này Chill Phết", "Đen"), ("Sài Gòn Đau Lòng Quá", "Hứa Kim Tuyền"),
            ("Đi Đu Đưa Đi", "Bích Phương"), ("Vợ Người Ta", "Phan Mạnh Quỳnh"),
            ("Thằng Điên", "JustaTee"), ("Ghen", "Erik")]
P1_PLACES = ["Bảo tàng Mỹ thuật", "công viên Yên Sở", "phố sách Đinh Lễ", "hồ Tây",
             "làng gốm Bát Tràng", "chùa Trấn Quốc", "Hoàng thành Thăng Long",
             "công viên Thống Nhất", "phố cổ Hà Nội", "Văn Miếu Quốc Tử Giám",
             "đảo Tuần Châu", "thác Bản Giốc", "vịnh Lan Hạ", "núi Hàm Lợn"]
P1_PERSONS = ["anh Tuấn", "chị Mai", "bố", "mẹ", "anh Hùng kế toán", "chị Lan hàng xóm",
              "em Trang", "chú Ba", "anh Dũng sửa xe", "cô giáo Hằng", "bác sĩ Nam",
              "anh Phong trưởng phòng", "chị Yến", "ông Tư"]
P1_STATIONS = ["VinFast Long Biên", "EVN Mỹ Đình", "VinFast Times City",
               "VinFast Royal City", "EVN Cầu Giấy", "VinFast Ocean Park",
               "VinFast Smart City", "EVN Hà Đông", "VinFast Aeon Hà Đông",
               "EVN Thanh Xuân", "VinFast Gardenia", "VinFast Sunshine"]
P1_COMPARE = [("Highlands Coffee", "The Coffee House"), ("Circle K", "GS25"),
              ("BigC", "Aeon Mall"), ("Phúc Long", "Katinat"),
              ("Trung Nguyên Legend", "Cộng Cà Phê"), ("Lotteria", "Jollibee"),
              ("KFC", "McDonald's"), ("Vincom", "Lotte Mart"),
              ("Bách Hóa Xanh", "WinMart"), ("Pharmacity", "Long Châu")]
P1_MUSIC_FIN = ["Phát lại bài đó đi", "Mở lại bài vừa nãy", "Bật bài đó lên",
                "Cho nghe lại bài đó", "Phát bài vừa nãy nữa đi", "Mở bài đó nghe lại"]
P1_NAV_FIN = ["Dẫn đường tới chỗ đó đi", "Chỉ đường đến nơi đó", "Đưa tôi tới chỗ đó",
              "Dẫn tôi đến nơi đó", "Chỉ đường tới chỗ đó", "Đi tới nơi đó đi"]
P1_CALL_FIN = ["Gọi cho người đó đi", "Gọi người đó giúp tôi", "Quay số người đó",
               "Gọi điện cho người đó", "Liên lạc người đó giúp tôi", "Gọi người đó luôn"]
P1_CHG_FIN = ["Dẫn tới trạm đó đi", "Tới trạm đó", "Chỉ đường trạm đó",
              "Đưa tôi tới trạm đó", "Dẫn đường trạm đó luôn", "Đi tới trạm đó"]
P1_CMP_FIN = ["Chỗ gần hơn đi", "Cho tôi chỗ gần hơn", "Chọn chỗ gần hơn", "Đến chỗ gần hơn"]

DEICTIC_FORBID = ["cái đó", "chỗ đó", "chỗ kia", "cái kia", "bài đó", "bài vừa nãy",
                  "bài vừa", "số đó", "người đó", "chỗ gần hơn", "chỗ gần", "cái thứ",
                  "nơi đó", "trạm đó"]


def p1_pronoun_resolution():
    kind = pick(["music", "navigation", "calling", "charging", "compare"])
    extra_forbid = None
    if kind == "music":
        song, artist = pick(P1_SONGS)
        turns = [("human", "Phát nhạc cho tôi nghe"),
                 ("gpt", "Anh muốn nghe bài nào ạ?"),
                 ("human", f"Bài {song} của {artist}"),
                 ("gpt", "Vâng, em chuẩn bị ạ."),
                 ("human", pick(P1_MUSIC_FIN))]
        gold = f"Phát lại bài {song} của {artist}."
        refs, dom = [song, artist], "music"
    elif kind == "navigation":
        place = pick(P1_PLACES)
        turns = [("human", "Cuối tuần tôi muốn đi chơi"),
                 ("gpt", "Anh định đi đâu ạ?"),
                 ("human", place),
                 ("gpt", "Vâng ạ."),
                 ("human", pick(P1_NAV_FIN))]
        gold = f"Dẫn đường tới {place}."
        refs, dom = [place], "navigation"
    elif kind == "calling":
        person = pick(P1_PERSONS)
        turns = [("human", "Tôi cần gọi điện"),
                 ("gpt", "Anh muốn gọi cho ai ạ?"),
                 ("human", cap(person)),
                 ("gpt", "Vâng ạ."),
                 ("human", pick(P1_CALL_FIN))]
        gold = f"Gọi cho {person}."
        refs, dom = [person], "calling"
    elif kind == "charging":
        station = pick(P1_STATIONS)
        turns = [("human", "Pin xe sắp hết rồi"),
                 ("gpt", f"Gần đây có trạm {station}, anh muốn tới đó chứ?"),
                 ("human", "Ừ"),
                 ("gpt", "Anh xác nhận tới trạm nào ạ?"),
                 ("human", pick(P1_CHG_FIN))]
        gold = f"Dẫn đường đến trạm {station}."
        refs, dom = [station], "charging"
    else:  # compare — chọn nơi GẦN HƠN trong hai lựa chọn
        a, b = pick(P1_COMPARE)
        da, db = pick([(1.2, 3.5), (0.8, 2.4), (1.5, 4.0)])
        turns = [("human", "Tìm quán cà phê gần đây"),
                 ("gpt", f"Có {a} cách {da}km và {b} cách {db}km, anh chọn nơi nào?"),
                 ("human", pick(P1_CMP_FIN))]
        gold = f"Dẫn đường đến {a}."
        refs, dom, extra_forbid = [a], "navigation", b
    s = sample(turns, gold, dom, src="patch_b4")
    low = gold.lower()
    for r in refs:
        assert r in gold, (r, gold)            # referent resolve được
    for d in DEICTIC_FORBID:
        assert d not in low, (d, gold)         # không để lại đại từ/chỉ định
    assert not word_in("nó", low), gold
    if extra_forbid:
        assert extra_forbid not in gold, (extra_forbid, gold)  # chọn đúng 1 trong 2
    return s


# ============================================================ c1 language_control
# input trộn Anh-Việt → gold tiếng Việt, GIỮ brand/proper-noun Anh, KHÔNG CJK, không
# dịch tên riêng. Chống regress code_switching (−4.6) + rò CJK (室外门廊…) ở v2.1.
C1_GENRES = ["chill", "lo-fi", "acoustic", "EDM", "ballad", "indie", "R&B", "jazz"]
C1_APPS = ["Spotify", "Apple Music", "YouTube Music", "SoundCloud", "Zing MP3"]
C1_PLACES = ["Lotte Mall", "Aeon Mall", "Landmark 81", "Vincom Center", "Crescent Mall",
             "Times City", "Royal City", "Diamond Plaza", "Saigon Centre", "Bitexco Tower"]
C1_MAPS = ["Google Maps", "Apple Maps"]
C1_PLAYLISTS = ["Discover Weekly", "Top Hits", "Daily Mix", "Chill Vibes",
                "Workout Energy", "Focus Flow", "Road Trip", "Morning Coffee"]
C1_VI_VERBS = ["Phát", "Dẫn đường", "Mở", "Đặt", "Gọi", "Bật", "Tìm"]
_CJK = re.compile(r"[一-鿿가-힯]")


def c1_language_control():
    kind = pick(["music", "navigate", "playlist"])
    if kind == "music":
        g, app = pick(C1_GENRES), pick(C1_APPS)
        inp = pick([f"play nhạc {g} trên {app}", f"mở {app} phát nhạc {g}",
                    f"bật nhạc {g} bằng {app}"])
        gold, brands, dom = f"Phát nhạc {g} trên {app}.", [app], "music"
    elif kind == "navigate":
        place, m = pick(C1_PLACES), pick(C1_MAPS)
        inp = pick([f"navigate to {place} bằng {m}", f"chỉ đường to {place} bằng {m}",
                    f"direction tới {place} qua {m}"])
        gold, brands, dom = f"Dẫn đường đến {place} bằng {m}.", [place, m], "navigation"
    else:  # playlist
        pl, app = pick(C1_PLAYLISTS), pick(C1_APPS)
        inp = pick([f"phát playlist {pl} trên {app}", f"mở {app} bật playlist {pl}"])
        gold, brands, dom = f"Phát playlist {pl} trên {app}.", [pl, app], "music"
    s = sample([("human", inp)], gold, dom, ctx=False, src="patch_b4")
    for b in brands:
        assert b in gold, (b, gold)            # brand/proper-noun giữ nguyên
    assert not _CJK.search(gold), gold          # tuyệt đối không rò CJK
    assert any(v in gold for v in C1_VI_VERBS), gold  # phần còn lại là tiếng Việt
    return s


GENERATORS = {
    "n1_keep_negative_clause": n1_keep_negative_clause,
    "n2_anti_inversion": n2_anti_inversion,
    "n3_exclusion_in_list": n3_exclusion_in_list,
    "s1_keep_qualifier": s1_keep_qualifier,
    "s2_compound_two_actions": s2_compound_two_actions,
    "n4_confirm_proposed": n4_confirm_proposed,
    "p1_pronoun_resolution": p1_pronoun_resolution,
    "c1_language_control": c1_language_control,
}

# Per-generator sample counts (severity-weighted; reports/v2.2). Dồn budget vào
# pronoun (tệ nhất, mới) + code_switching (regress, mới) + correction/negation/slot;
# trim compound/confirm. `--per` chỉ là fallback. Tổng B4 = 1700 (+ B2/B3 790 = 2490).
WEIGHTS = {
    "n1_keep_negative_clause": 250,
    "n2_anti_inversion": 150,
    "n3_exclusion_in_list": 250,
    "s1_keep_qualifier": 250,
    "s2_compound_two_actions": 150,
    "n4_confirm_proposed": 150,
    "p1_pronoun_resolution": 300,
    "c1_language_control": 200,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=200, help="samples per generator")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--merge", action="store_true",
                    help="also write train_v3 = train + patch")
    ap.add_argument("--inplace", action="store_true",
                    help="append patch into data/processed/train.jsonl (Kaggle: run "
                         "AFTER split_data.py; idempotent since split regenerates it)")
    args = ap.parse_args()
    random.seed(args.seed)

    bench = bench_golds()
    rows, counts, seen, leak = [], {}, set(), 0
    for name, fn in GENERATORS.items():
        target = WEIGHTS.get(name, args.per)
        n = 0
        tries = 0
        while n < target and tries < target * 60:
            tries += 1
            r = fn()
            gold = json.loads(r["conversations"][-1]["value"])["rewrite_message"]
            if gold in bench:          # never train on an eval-bench gold
                leak += 1
                continue
            full = json.dumps(r["conversations"], ensure_ascii=False)
            if full in seen:
                continue
            seen.add(full)
            rows.append(r)
            n += 1
        counts[name] = n
    print(f"(skipped {leak} samples colliding with bench golds)")
    random.shuffle(rows)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"patch: {len(rows)} samples → {OUT}")
    for k, v in counts.items():
        print(f"  {k:26s} {v}")

    if args.merge:
        base = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
        merged = base + rows
        random.shuffle(merged)
        TRAIN_V3.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                            encoding="utf-8")
        print(f"merged: {len(base)} + {len(rows)} = {len(merged)} → {TRAIN_V3}")

    if args.inplace:
        base = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
        merged = base + rows
        random.shuffle(merged)
        TRAIN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                         encoding="utf-8")
        print(f"inplace: {len(base)} + {len(rows)} = {len(merged)} → {TRAIN} (overwritten)")


if __name__ == "__main__":
    main()
