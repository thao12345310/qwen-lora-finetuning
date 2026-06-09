"""Short-contextual everyday-utterance patch — teach the model *when NOT to rewrite a
short final utterance into a tool command*. Same proven recipe as patch_b2b3/patch_b4
(plain SFT positives of the exact tricky structure, a self-check asserts each gold, a
leak-check drops any gold colliding with the eval bench), but with two outputs:

  • train pool   → data/patches/patch_short_contextual.jsonl   (wired in via recipe v4)
  • held-out eval → data/bench/bench_short_contextual.jsonl     (scored separately)

The two pools draw from DISJOINT entity/number sub-pools (the `held` flag), so the eval
is a genuine held-out slice, not just disjoint conversations.

Patterns (meta.pattern):
  ack_bind_proposal          ack "ừ/ok/vâng" after a clear proposal → bind to the proposal
  ack_no_action              ack after a pure info answer            → echo "Ừ."/"Ok."
  ack_after_done             ack after assistant already did it      → echo, don't repeat
  reject_subproposal         reject a sub-proposal ("Thôi.")         → echo, don't flip to action
  short_correction           short value update ("24", "90%")        → bind + keep new value
  confirm_with_modifier       "Ừ nhưng tránh cao tốc"                → action + keep modifier
  vague_execute               "Gửi đi/Chọn đi" WITH a draft          → execute the draft
  vague_execute_no_proposal   "Làm đi/Cứ thế" with NO proposal       → echo, don't fabricate
  pronoun_recent_entity       "cái đó/bài đó/chỗ đó"                  → resolve nearest entity
  list_choice                 "chỗ gần hơn/cái thứ hai"              → resolve from offered list

CRITICAL: the no-op golds ("Ừ.", "Ok.", "Không.", "Thôi.", "Gửi đi.") are deliberately
identical across many samples. build_train_mix drops any train gold that matches a gold
in guards.bench_leak_check. So the held-out file MUST stay OUT of that leak-check
reference (keep it pointed at dialogues_bench_browser.jsonl only), or the no-op training
signal gets mass-dropped.

Run:
    /opt/homebrew/bin/python3.11 -m src.data.patch_short_contextual
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

# reuse the proven framework — single source of truth for prompt/format/leak-check
from src.data.patch_b2b3 import sample, pick, cap, bench_golds

OUT = Path("data/patches/patch_short_contextual.jsonl")
BENCH_OUT = Path("data/bench/bench_short_contextual.jsonl")
TRAIN = Path("data/processed/train.jsonl")
SRC = "patch_short_contextual"

# --------------------------------------------------------------------------- forms
AFFIRM = ["Ừ", "Ừm", "Ok", "Okê", "Vâng", "Dạ", "Được", "Được rồi", "Có", "Ờ"]
THANKS = ["Cảm ơn", "Cảm ơn nhé", "Cảm ơn nha", "Ờ cảm ơn"]
ACK = AFFIRM + THANKS
REJECT = ["Thôi", "Thôi khỏi", "Không", "Không cần", "Khỏi", "Thôi không cần đâu"]
# command-shaped utterances that must STAY a no-op when there is no proposal to bind to
VAGUE_NP = ["Làm đi", "Cứ thế", "Triển", "Gửi đi", "Chọn đi", "Bật luôn", "Ok triển"]

# --------------------------------------------------------------------------- banks
DEST = ["Vincom Bà Triệu", "Times City", "Aeon Mall Long Biên", "Hồ Gươm",
        "sân bay Nội Bài", "bến xe Mỹ Đình", "công viên Thống Nhất", "Royal City",
        "Lotte Center Hà Nội", "Big C Thăng Long", "Hồ Tây", "Landmark 81",
        "chợ Bến Thành", "Bà Nà Hills"]
CONTACT = ["anh Nam", "chị Hương", "mẹ", "ba", "anh Tuấn", "chị Lan", "anh Minh",
           "chị Mai", "anh Hùng", "chú Hải", "chị Thu", "anh Phong", "cô Lan", "bác Tư"]
PLAYLIST = ["Lofi Drive", "Vpop Chill", "EDM Drive", "Acoustic Sáng", "Nhạc Trịnh",
            "Top Hits Việt", "Indie Việt", "Rap Việt Hits", "Ballad Buồn"]
SONG = [("Nàng Thơ", "Hoàng Dũng"), ("Chúng Ta Của Hiện Tại", "Sơn Tùng M-TP"),
        ("Ước Gì", "Mỹ Tâm"), ("Em Của Ngày Hôm Qua", "Sơn Tùng M-TP"),
        ("Đưa Nhau Đi Trốn", "Đen Vâu"), ("Lạc Trôi", "Sơn Tùng M-TP"),
        ("Có Chàng Trai Viết Lên Cây", "Phan Mạnh Quỳnh"),
        ("Hôm Nay Tôi Buồn", "Phùng Khánh Linh"), ("Họa Mi Tóc Nâu", "Mỹ Tâm"),
        ("Nơi Này Có Anh", "Sơn Tùng M-TP")]
STATION = ["trạm sạc nhanh ở Vincom Bà Triệu", "trạm sạc ở Aeon Long Biên",
           "trạm VinFast ở Times City", "trạm sạc ở Mê Linh Plaza",
           "trạm sạc nhanh ở Long Biên", "trạm sạc ở Royal City", "trạm sạc ở Ocean Park"]
CAFE = ["Highlands Coffee", "The Coffee House", "Phúc Long", "Cộng Cà Phê", "Katinat",
        "Trung Nguyên Legend", "Starbucks Hồ Gươm", "Runam Bistro"]
DEVICE_BT = ["điện thoại 'Pixel 7 Pro'", "điện thoại 'iPhone 15'",
             "điện thoại 'Galaxy S24'", "tai nghe 'AirPods Pro'", "máy tính bảng 'iPad Air'"]
STREET = ["Nguyễn Trãi", "Lê Lợi", "Trần Hưng Đạo", "Bà Triệu", "Nguyễn Huệ",
          "Láng Hạ", "Cầu Giấy", "Xuân Thủy"]
NAMES = ["Long", "Minh", "Hà", "Linh", "Tuấn", "Hương", "Nam", "Quân"]
ARTIST_SONGS = {
    "Mỹ Tâm": ["Ước Gì", "Họa Mi Tóc Nâu", "Đâu Chỉ Riêng Em"],
    "Sơn Tùng M-TP": ["Lạc Trôi", "Chúng Ta Của Hiện Tại", "Nơi Này Có Anh"],
    "Đen Vâu": ["Đưa Nhau Đi Trốn", "Bài Này Chill Phết", "Mười Năm"],
    "Hoàng Dũng": ["Nàng Thơ", "Đoạn Kết Mới", "Yên"],
    "Vũ": ["Lạ Lùng", "Đông Kiếm Em", "Bước Qua Nhau"],
    "Hà Anh Tuấn": ["Tháng Tư Là Lời Nói Dối", "Ve` Đâu Mái Tóc Người Thương", "Nơi Tình Yêu Bắt Đầu"],
}

# number sub-pools — (train, held) are disjoint so numeric scenarios don't overlap
TEMP = ([18, 19, 20, 21, 22, 23, 24], [25, 26, 27])
VOL = ([30, 40, 45, 50, 55, 60, 70], [35, 65, 80])
PCT = ([60, 70, 75, 80, 85], [50, 90, 95])
SPEED = ([80, 90, 100, 110, 120], [60, 70, 130])
ADDR = ([12, 15, 21, 28, 33, 40, 45], [7, 55, 88])
HOUR = ([6, 7, 8, 9, 10], [5, 11, 12])
PIN = ([22, 35, 48, 55, 63, 72], [18, 80, 90])
KM = ([120, 180, 220, 260, 300], [90, 330, 400])
MINUTE = ([12, 18, 25, 30, 35], [8, 45, 50])
DIST = ([1, 2, 3, 4, 5], [6, 7])


# --------------------------------------------------------------------------- helpers
def _pool(bank, held, frac=5):
    k = max(1, len(bank) // frac)
    return bank[-k:] if held else bank[:-k]


def P(bank, held, frac=5):
    """Pick from a bank, reserving the last 1/frac of items for the held-out pool."""
    return pick(_pool(bank, held, frac))


def two_of(bank, held, frac=5):
    """Two distinct items from the held/train sub-pool, widening to the full bank if the
    sub-pool is too small to yield two (held pools can shrink to a single entity)."""
    sub = _pool(bank, held, frac)
    if len(sub) < 2:
        sub = bank
    a = pick(sub)
    b = pick(sub)
    while b == a:
        b = pick(sub)
    return a, b


def N(pair, held):
    return pick(pair[1] if held else pair[0])


def two(pair, held):
    seq = pair[1] if held else pair[0]
    a = pick(seq)
    b = pick(seq)
    while b == a:
        b = pick(seq)
    return a, b


def echo(form: str) -> str:
    """Normalise a short utterance into its echo gold: capitalise + terminal period."""
    f = form.strip()
    f = f[0].upper() + f[1:]
    if f[-1] not in ".!?":
        f += "."
    return f


def emit(turns, gold, domain, ctx=True):
    return sample(turns, gold, domain, ctx=ctx, src=SRC)


def assert_echo(gold, form):
    assert gold == echo(form), (gold, form)
    assert len(gold.split()) <= 4, gold  # an echo stays short; it never expands


def assert_action(gold, form):
    assert gold != echo(form), gold          # it DID expand past the bare ack
    assert gold[-1] in ".'\"", gold          # ends on a period or a quoted body
    assert "<REWRITE_AND_CLASSIFY>" not in gold, gold


# =================================================================== ack_bind_proposal
def _bind_case(held):
    """(domain, trigger, proposal, gold) — assistant proposes a concrete action."""
    d = pick("nav msg call clim charge music assist veh".split())
    if d == "nav":
        x = P(DEST, held)
        return ("navigation", f"Tôi muốn tới {x}.",
                f"Tôi dẫn đường tới {x} theo tuyến nhanh nhất nhé?",
                f"Dẫn đường tới {x} theo tuyến nhanh nhất.")
    if d == "msg":
        x = P(CONTACT, held)
        return ("messaging", f"Nhắn cho {x} là tôi sắp tới.",
                "Bạn muốn gửi ngay tin nhắn này không?",
                f"Gửi tin nhắn cho {x} với nội dung: 'Tôi sắp tới.'")
    if d == "call":
        x = P(CONTACT, held)
        return ("calling", f"Mình muốn gọi cho {x}.",
                f"Tôi gọi cho {x} qua danh bạ nhé?", f"Gọi cho {x} qua danh bạ.")
    if d == "clim":
        t = N(TEMP, held)
        return ("climate", "Trong xe hơi nóng.",
                f"Bạn muốn hạ điều hòa xuống {t} độ không?", f"Hạ điều hòa xuống {t} độ.")
    if d == "charge":
        return ("charging", "Pin xe còn ít quá.",
                "Tôi tìm và dẫn đường tới trạm sạc nhanh gần nhất nhé?",
                "Tìm và dẫn đường tới trạm sạc nhanh gần nhất.")
    if d == "music":
        x = P(PLAYLIST, held)
        return ("music", "Mình muốn nghe nhạc lái xe.",
                f"Tôi phát playlist '{x}' trên Spotify nhé?",
                f"Phát playlist '{x}' trên Spotify.")
    if d == "assist":
        return ("driver_assist", "Đường này đông xe quá.",
                "Bạn muốn bật cảnh báo điểm mù không?", "Bật cảnh báo điểm mù.")
    return ("vehicle", "Ghế lái vị trí này vừa rồi.",
            "Bạn muốn lưu vị trí ghế lái hiện tại không?", "Lưu vị trí ghế lái hiện tại.")


def ack_bind_proposal(held):
    dom, trig, prop, gold = _bind_case(held)
    form = pick(AFFIRM)
    turns = [("human", trig), ("gpt", prop), ("human", form)]
    s = emit(turns, gold, dom)
    assert_action(gold, form)
    return s


# ====================================================================== ack_no_action
def _info_case(held):
    """(domain, question, answer) — a PURE info answer, no proposed action."""
    d = pick("veh clim music call charge nav assist msg home".split())
    if d == "veh":
        return ("vehicle", "Pin xe còn bao nhiêu?",
                f"Pin còn {N(PIN, held)}%, ước tính đi được khoảng {N(KM, held)} km nữa.")
    if d == "clim":
        return ("climate", "Điều hòa đang bao nhiêu độ?",
                f"Điều hòa đang đặt ở {N(TEMP, held)} độ.")
    if d == "music":
        t, a = P(SONG, held)
        return ("music", "Bài đang phát là bài gì?", f"Bạn đang nghe '{t}' của {a}.")
    if d == "call":
        return ("calling", "Có cuộc gọi nhỡ nào không?",
                f"Không có cuộc gọi nhỡ nào trong {N(MINUTE, held)} phút gần đây.")
    if d == "charge":
        return ("charging", "Trạm sạc gần nhất cách bao xa?",
                f"Trạm sạc nhanh gần nhất cách {N(DIST, held)} km.")
    if d == "nav":
        return ("navigation", f"Đi tới {P(DEST, held)} mất bao lâu?",
                f"Tuyến hiện tại mất khoảng {N(MINUTE, held)} phút.")
    if d == "assist":
        return ("driver_assist", "Xe có cảnh báo điểm mù không?",
                "Xe có cảnh báo điểm mù, hiện đang tắt.")
    if d == "msg":
        c = P(CONTACT, held)
        return ("messaging", "Đọc tin nhắn mới nhất.",
                f"Tin nhắn mới nhất từ {c}: 'Em tới đâu rồi?'")
    return ("vehicle", "Bluetooth đang nối với máy nào?",
            f"Bluetooth đang kết nối với {P(DEVICE_BT, held)}.")


def ack_no_action(held):
    dom, q, a = _info_case(held)
    form = pick(ACK)
    turns = [("human", q), ("gpt", a), ("human", form)]
    gold = echo(form)
    s = emit(turns, gold, dom)
    assert_echo(gold, form)
    return s


# ====================================================================== ack_after_done
def _done_case(held):
    """(domain, command, done-confirmation) — assistant already executed it."""
    d = pick("music msg nav clim assist call veh charge".split())
    if d == "music":
        x = P(PLAYLIST, held)
        return ("music", f"Bật playlist {x} đi.", f"Đã bật playlist {x}.")
    if d == "msg":
        x = P(CONTACT, held)
        return ("messaging", f"Gửi tin nhắn cho {x} là tôi tới muộn.", f"Đã gửi tin nhắn cho {x}.")
    if d == "nav":
        x = P(DEST, held)
        return ("navigation", f"Dẫn đường tới {x}.", f"Đã đặt lộ trình tới {x}.")
    if d == "clim":
        t = N(TEMP, held)
        return ("climate", f"Đặt điều hòa {t} độ.", f"Đã đặt điều hòa ở {t} độ.")
    if d == "assist":
        return ("driver_assist", "Tắt cảnh báo điểm mù.", "Đã tắt cảnh báo điểm mù.")
    if d == "call":
        x = P(CONTACT, held)
        return ("calling", f"Gọi cho {x}.", f"Đã gọi cho {x}.")
    if d == "veh":
        return ("vehicle", "Lưu vị trí ghế lái hiện tại.", "Đã lưu vị trí ghế lái hiện tại.")
    return ("charging", "Bắt đầu sạc xe.", "Đã bắt đầu sạc xe.")


def ack_after_done(held):
    dom, cmd, done = _done_case(held)
    form = pick(ACK)
    turns = [("human", cmd), ("gpt", done), ("human", form)]
    gold = echo(form)
    s = emit(turns, gold, dom)
    assert_echo(gold, form)
    return s


# ================================================================== reject_subproposal
def _reject_case(held):
    """(domain, command, 'done + a follow-up sub-proposal?')."""
    sub = pick(["Bạn muốn {x} không?", "Có cần {x} nữa không?"])
    d = pick("music nav clim call assist charge".split())
    if d == "music":
        x = P(PLAYLIST, held)
        return ("music", f"Bật playlist {x} đi.",
                f"Đã bật playlist {x}. " + sub.format(x="tăng âm lượng"))
    if d == "nav":
        x = P(DEST, held)
        return ("navigation", f"Dẫn đường tới {x}.",
                f"Đã dẫn đường tới {x}. " + sub.format(x="thêm điểm dừng"))
    if d == "clim":
        t = N(TEMP, held)
        return ("climate", f"Hạ điều hòa xuống {t} độ.",
                f"Đã hạ điều hòa xuống {t} độ. " + sub.format(x="bật thêm sấy kính"))
    if d == "call":
        x = P(CONTACT, held)
        return ("calling", f"Gọi cho {x}.",
                f"Đang gọi cho {x}. " + sub.format(x="bật loa ngoài"))
    if d == "assist":
        return ("driver_assist", "Bật giữ làn đường.",
                "Đã bật giữ làn đường. " + sub.format(x="bật thêm cảnh báo rung vô lăng"))
    return ("charging", f"Tìm {P(STATION, held)}.",
            "Đã tìm thấy trạm sạc gần nhất. " + sub.format(x="đặt lịch sạc trước"))


def reject_subproposal(held):
    dom, cmd, gpt = _reject_case(held)
    form = pick(REJECT)
    turns = [("human", cmd), ("gpt", gpt), ("human", form)]
    gold = echo(form)
    s = emit(turns, gold, dom)
    assert_echo(gold, form)
    return s


# ============================================================ vague_execute_no_proposal
def vague_execute_no_proposal(held):
    dom, q, a = _info_case(held)
    form = pick(VAGUE_NP)
    turns = [("human", q), ("gpt", a), ("human", form)]
    gold = echo(form)
    s = emit(turns, gold, dom)
    assert_echo(gold, form)
    return s


# ===================================================================== short_correction
def short_correction(held):
    d = pick("clim music nav charge assist call msg".split())
    if d == "clim":
        a, b = two(TEMP, held)
        turns = [("human", f"Bật điều hòa {a} độ."), ("gpt", f"Đã đặt điều hòa ở {a} độ."),
                 ("human", pick([f"À {b}.", f"{b} độ.", f"À {b} độ thôi."]))]
        gold, key = f"Đặt điều hòa ở {b} độ.", str(b)
        dom = "climate"
    elif d == "music":
        a, b = two(VOL, held)
        turns = [("human", f"Đặt âm lượng {a}%."), ("gpt", f"Đã đặt âm lượng {a}%."),
                 ("human", pick([f"{b}%.", f"À {b}%."]))]
        gold, key, dom = f"Đặt âm lượng {b}%.", f"{b}%", "music"
    elif d == "nav":
        a, b = two(ADDR, held)
        st = P(STREET, held)
        turns = [("human", f"Dẫn đường tới {a} {st}."), ("gpt", f"Đã đặt điểm đến là {a} {st}."),
                 ("human", pick([f"À {b} {st}.", f"{b} {st} cơ."]))]
        gold, key, dom = f"Dẫn đường tới {b} {st}.", f"{b} {st}", "navigation"
    elif d == "charge":
        a, b = two(PCT, held)
        turns = [("human", f"Sạc xe đến {a}% rồi dừng."), ("gpt", f"Đã đặt giới hạn sạc là {a}%."),
                 ("human", pick([f"{b}%.", f"À {b}% thôi."]))]
        gold, key, dom = f"Đặt giới hạn sạc xe là {b}%.", f"{b}%", "charging"
    elif d == "assist":
        a, b = two(SPEED, held)
        turns = [("human", f"Đặt giới hạn tốc độ {a} km/h."),
                 ("gpt", f"Đã đặt giới hạn tốc độ {a} km/h."),
                 ("human", pick([f"{b} thôi.", f"À {b} km/h."]))]
        gold, key, dom = f"Đặt giới hạn tốc độ {b} km/h.", f"{b}", "driver_assist"
    elif d == "call":
        c1, c2 = two_of(CONTACT, held)
        turns = [("human", f"Gọi cho {c1}."), ("gpt", f"Tôi gọi cho {c1} nhé?"),
                 ("human", f"{cap(c2)}.")]
        gold, key, dom = f"Gọi cho {c2}.", c2, "calling"
    else:
        c = P(CONTACT, held)
        a, b = two(HOUR, held)
        turns = [("human", f"Nhắn cho {c} là em tới lúc {a} giờ."),
                 ("gpt", f"Tôi đã soạn tin nhắn cho {c}: 'Em tới lúc {a} giờ.'"),
                 ("human", pick([f"{b} giờ.", f"À {b} giờ."]))]
        gold, key, dom = f"Nhắn cho {c} nội dung: 'Em tới lúc {b} giờ.'", f"{b} giờ", "messaging"
    s = emit(turns, gold, dom)
    assert key in gold, (key, gold)
    return s


# ================================================================ confirm_with_modifier
def confirm_with_modifier(held):
    d = pick("nav msg call clim charge music".split())
    if d == "nav":
        x = P(DEST, held)
        trig, prop = f"Dẫn đường tới {x}.", "Tôi chọn tuyến nhanh nhất nhé?"
        mod, gold = "tránh cao tốc", f"Dẫn đường tới {x} theo tuyến nhanh nhất nhưng tránh cao tốc."
        dom = "navigation"
    elif d == "msg":
        x = P(CONTACT, held)
        trig, prop = f"Nhắn cho {x} là tôi đến muộn.", "Bạn muốn gửi tin nhắn này ngay không?"
        mod, gold = "gửi qua Zalo", f"Gửi tin nhắn qua Zalo cho {x} với nội dung: 'Tôi đến muộn.'"
        dom = "messaging"
    elif d == "call":
        x = P(CONTACT, held)
        trig, prop = f"Gọi cho {x}.", f"Tôi gọi cho {x} ngay nhé?"
        mod, gold = "đừng bật loa ngoài", f"Gọi cho {x} và không bật loa ngoài."
        dom = "calling"
    elif d == "clim":
        t = N(TEMP, held)
        trig, prop = "Trong xe nóng quá.", f"Tôi bật điều hòa {t} độ nhé?"
        mod, gold = "chỉ hàng ghế sau", f"Bật điều hòa {t} độ chỉ cho hàng ghế sau."
        dom = "climate"
    elif d == "charge":
        p = N(PCT, held)
        trig, prop = "Cắm sạc xe giúp tôi.", "Bạn muốn bắt đầu sạc ngay không?"
        mod, gold = f"không sạc quá {p}%", f"Bắt đầu sạc xe ngay nhưng không sạc quá {p}%."
        dom = "charging"
    else:
        x = P(PLAYLIST, held)
        trig, prop = f"Phát playlist {x}.", f"Tôi phát playlist {x} trên Spotify nhé?"
        mod, gold = "phát nhỏ thôi", f"Phát playlist {x} trên Spotify với âm lượng nhỏ."
        dom = "music"
    form = pick(["Ừ", "Ok", "Vâng", "Được", "Có"])
    turns = [("human", trig), ("gpt", prop), ("human", f"{form}, nhưng {mod}.")]
    s = emit(turns, gold, dom)
    # the modifier must survive into the gold (the over-compression bug drops it)
    head = mod.split()[-1] if mod.split()[-1] != "%" else mod
    assert any(w in gold.lower() for w in mod.lower().split() if len(w) > 2), (mod, gold)
    return s


# messaging bodies that read naturally as a standalone tin-nhắn content
VEXEC_MSG = ["con đang lái xe", "con sắp tới rồi", "anh đang trên đường",
             "em ra trễ 10 phút", "tối nay về muộn", "đang kẹt xe chút nhé"]


# ======================================================================= vague_execute
def vague_execute(held):
    d = pick("msg nav call assist charge music".split())
    if d == "msg":
        x = P(CONTACT, held)
        body = pick(VEXEC_MSG)
        turns = [("human", f"Nhắn cho {x} là {body}."),
                 ("gpt", f"Tôi đã soạn tin nhắn cho {x}: '{cap(body)}.'"),
                 ("human", pick(["Gửi đi.", "Gửi luôn đi.", "Ừ gửi đi.", "Gửi ngay đi."]))]
        gold, dom = f"Gửi tin nhắn cho {x} với nội dung: '{cap(body)}.'", "messaging"
    elif d == "nav":
        x = P(DEST, held)
        turns = [("human", f"Tìm đường tới {x}."),
                 ("gpt", "Tuyến ít kẹt xe đang phù hợp nhất. Bạn muốn chọn tuyến này không?"),
                 ("human", pick(["Chọn đi.", "Chọn tuyến đó đi.", "Đi tuyến đó đi.", "Ừ chọn đi."]))]
        gold, dom = f"Dẫn đường tới {x} theo tuyến ít kẹt xe.", "navigation"
    elif d == "call":
        x = P(CONTACT, held)
        turns = [("human", f"Tìm số {x} trong danh bạ."),
                 ("gpt", f"Tôi đã tìm thấy số di động của {x}."),
                 ("human", pick(["Gọi đi.", "Gọi luôn đi.", "Gọi ngay đi.", "Ừ gọi đi."]))]
        gold, dom = f"Gọi cho {x} bằng số di động.", "calling"
    elif d == "assist":
        feat = P(["cảnh báo điểm mù", "hỗ trợ giữ làn đường", "cảnh báo va chạm phía trước",
                  "hỗ trợ phanh khẩn cấp", "ga tự động"], held)
        turns = [("human", "Đường này hơi đông."),
                 ("gpt", f"Bạn muốn bật {feat} không?"),
                 ("human", pick(["Bật luôn.", "Bật đi.", "Ừ bật đi.", "Bật ngay đi."]))]
        gold, dom = f"Bật {feat}.", "driver_assist"
    elif d == "charge":
        x = P(STATION, held)
        turns = [("human", "Tìm trạm sạc nhanh quanh đây."),
                 ("gpt", f"{cap(x)} gần nhất và còn trống. Bạn muốn đi tới đó không?"),
                 ("human", pick(["Đi đi.", "Tới đó đi.", "Dẫn đi.", "Ừ đi đi."]))]
        gold, dom = f"Dẫn đường tới {x} đang còn trống.", "charging"
    else:
        x = P(PLAYLIST, held)
        turns = [("human", "Mở nhạc đi."),
                 ("gpt", f"Tôi đã chọn playlist '{x}'. Bạn muốn phát ngay không?"),
                 ("human", pick(["Phát đi.", "Phát luôn đi.", "Mở đi.", "Ừ phát đi."]))]
        gold, dom = f"Phát playlist '{x}'.", "music"
    s = emit(turns, gold, dom)
    assert gold[-1] in ".'\"", gold
    return s


# ================================================================ pronoun_recent_entity
def pronoun_recent_entity(held):
    d = pick("music nav call veh charge".split())
    if d == "music":
        t, a = P(SONG, held)
        turns = [("human", f"Phát bài '{t}' của {a}."), ("gpt", f"Đang phát '{t}' của {a}."),
                 ("human", pick(["Thêm bài đó vào yêu thích.", "Lưu bài đó vào playlist yêu thích."]))]
        gold, dom = f"Thêm bài '{t}' của {a} vào danh sách yêu thích.", "music"
    elif d == "nav":
        x = P(CAFE, held)
        st = P(STREET, held)
        turns = [("human", "Tìm quán cà phê yên tĩnh gần đây."),
                 ("gpt", f"Có {x} ở {st} phù hợp nhất."),
                 ("human", pick(["Dẫn tới chỗ đó đi.", "Đi tới chỗ đó."]))]
        gold, dom = f"Dẫn đường tới {x} ở {st}.", "navigation"
    elif d == "call":
        x = P(CONTACT, held)
        turns = [("human", f"Tìm {x} trong danh bạ."), ("gpt", f"Tôi tìm thấy {x} trong danh bạ."),
                 ("human", pick(["Gọi người đó đi.", "Gọi cho người đó."]))]
        gold, dom = f"Gọi cho {x}.", "calling"
    elif d == "veh":
        x = P(DEVICE_BT, held)
        turns = [("human", "Bluetooth đang kết nối với máy nào?"),
                 ("gpt", f"Bluetooth đang kết nối với {x}."),
                 ("human", pick(["Ngắt cái đó đi.", "Ngắt kết nối cái đó."]))]
        gold, dom = f"Ngắt kết nối Bluetooth với {x}.", "vehicle"
    else:
        x = P(STATION, held)
        turns = [("human", "Tìm trạm sạc nhanh gần nhất."), ("gpt", f"Gần nhất là {x}."),
                 ("human", pick(["Dẫn tới trạm đó đi.", "Đi tới trạm đó."]))]
        gold, dom = f"Dẫn đường tới {x}.", "charging"
    s = emit(turns, gold, dom)
    assert " đó" not in gold.lower() and "cái đó" not in gold.lower(), gold  # pronoun resolved
    return s


# =========================================================================== list_choice
def list_choice(held):
    d = pick("nav_near call music charge route".split())
    if d == "nav_near":
        a, b = two_of(CAFE, held)
        da, db = N(DIST, held) * 100 + 500, N(KM, held) // 100 + 2
        turns = [("human", "Tìm quán cà phê gần đây."),
                 ("gpt", f"Có {a} cách {da} m và {b} cách {db} km."),
                 ("human", pick(["Chỗ gần hơn.", "Cái gần hơn đi."]))]
        gold, dom = f"Dẫn đường tới {a}.", "navigation"
    elif d == "call":
        nm = P(NAMES, held)
        turns = [("human", f"Gọi cho {nm}."),
                 ("gpt", f"Có {nm} công ty và {nm} em họ trong danh bạ."),
                 ("human", pick(["Cái thứ hai.", "Người thứ hai."]))]
        gold, dom = f"Gọi cho {nm} em họ.", "calling"
    elif d == "music":
        artist = pick(list(ARTIST_SONGS))
        s1, s2, s3 = ARTIST_SONGS[artist]
        turns = [("human", f"Tìm nhạc của {artist}."),
                 ("gpt", f"Có bài '{s1}', '{s2}' và '{s3}'."),
                 ("human", pick(["Bài thứ ba.", "Cái thứ ba."]))]
        gold, dom = f"Phát bài '{s3}' của {artist}.", "music"
    elif d == "charge":
        a, b = two_of(DEST, held)
        turns = [("human", "Tìm trạm sạc quanh đây."),
                 ("gpt", f"Có trạm thường ở {a} và trạm sạc nhanh ở {b}."),
                 ("human", pick(["Trạm sạc nhanh.", "Cái sạc nhanh đi."]))]
        gold, dom = f"Dẫn đường tới trạm sạc nhanh ở {b}.", "charging"
    else:
        x = P(DEST, held)
        turns = [("human", f"Tìm đường tới {x}."),
                 ("gpt", "Có tuyến nhanh nhất qua cao tốc và tuyến ít kẹt xe qua đường ven biển."),
                 ("human", pick(["Tuyến ít kẹt xe.", "Tuyến ven biển đi."]))]
        gold, dom = f"Dẫn đường tới {x} theo tuyến ít kẹt xe qua đường ven biển.", "navigation"
    s = emit(turns, gold, dom)
    assert gold[-1] in ".'\"", gold
    return s


# --------------------------------------------------------------------------- registry
GENERATORS = {
    "ack_bind_proposal": ack_bind_proposal,
    "ack_no_action": ack_no_action,
    "ack_after_done": ack_after_done,
    "reject_subproposal": reject_subproposal,
    "vague_execute_no_proposal": vague_execute_no_proposal,
    "short_correction": short_correction,
    "confirm_with_modifier": confirm_with_modifier,
    "vague_execute": vague_execute,
    "pronoun_recent_entity": pronoun_recent_entity,
    "list_choice": list_choice,
}

# train targets — weight the "don't over-help" restraint classes higher
WEIGHTS = {
    "ack_no_action": 300,
    "ack_after_done": 220,
    "reject_subproposal": 220,
    "vague_execute_no_proposal": 240,
    "ack_bind_proposal": 240,
    "short_correction": 200,
    "confirm_with_modifier": 160,
    "vague_execute": 160,
    "pronoun_recent_entity": 140,
    "list_choice": 140,
}
BENCH_PER = 20  # held-out eval samples per pattern


def generate(targets, held, bench, seen):
    """Run each generator up to its target; dedupe by conversation, drop bench-gold leaks."""
    rows, counts, leak = [], {}, 0
    for name, fn in GENERATORS.items():
        target = targets[name]
        n, tries = 0, 0
        while n < target and tries < target * 60:
            tries += 1
            r = fn(held)
            r.setdefault("meta", {})["pattern"] = name
            gold = json.loads(r["conversations"][-1]["value"])["rewrite_message"]
            if gold in bench:                # never train on an eval-bench gold
                leak += 1
                continue
            full = json.dumps(r["conversations"], ensure_ascii=False)
            if full in seen:
                continue
            seen.add(full)
            rows.append(r)
            n += 1
        counts[name] = n
    return rows, counts, leak


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bench-out", type=Path, default=BENCH_OUT)
    ap.add_argument("--bench-per", type=int, default=BENCH_PER)
    ap.add_argument("--inplace", action="store_true",
                    help="also append the train pool into data/processed/train.jsonl (Kaggle "
                         "pipeline: run AFTER split_data.py, parallel to patch_b4 --inplace). "
                         "Idempotent: split regenerates train.jsonl each run, and a dedupe "
                         "guards against re-appending.")
    args = ap.parse_args()
    random.seed(args.seed)

    bench = bench_golds()        # main eval bench — leak-check the TRAIN pool against it
    seen: set[str] = set()       # shared so the held-out pool never repeats a train conv

    train_rows, train_counts, train_leak = generate(WEIGHTS, held=False, bench=bench, seen=seen)
    random.shuffle(train_rows)
    write_jsonl(args.out, train_rows)
    print(f"train: {len(train_rows)} samples → {args.out}  (skipped {train_leak} bench-gold leaks)")
    for k in WEIGHTS:
        print(f"  {k:28s} {train_counts[k]}")

    bench_targets = {k: args.bench_per for k in GENERATORS}
    held_rows, held_counts, _ = generate(bench_targets, held=True, bench=set(), seen=seen)
    random.shuffle(held_rows)
    write_jsonl(args.bench_out, held_rows)
    print(f"held-out eval: {len(held_rows)} samples → {args.bench_out}")
    for k in bench_targets:
        print(f"  {k:28s} {held_counts[k]}")

    if args.inplace:
        if not TRAIN.exists():
            raise FileNotFoundError(f"{TRAIN} not found — run split_data.py first")
        base = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
        present = {json.dumps(r["conversations"], ensure_ascii=False) for r in base}
        new = [r for r in train_rows
               if json.dumps(r["conversations"], ensure_ascii=False) not in present]
        with TRAIN.open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"inplace: appended {len(new)} samples → {TRAIN} (total {len(base) + len(new)})")


if __name__ == "__main__":
    main()
