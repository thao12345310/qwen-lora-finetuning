"""B2/B3 targeted training patch — narrow hard samples for the two failure modes
that dominate multi_turn_slot errors in the v1.0 bench (see errors_multi_turn_slot.txt):

B2 messaging:
  G1 no-pronoun-content  — message body must NOT echo the recipient pronoun
                           ("Bảo anh ấy X" → body "X", recipient is a separate slot). [idx 26]
  G2 separate-time-slot  — keep send-time as its OWN slot AND update it inside the
                           content; don't merge time into the body only.          [idx 280]
  G3 email-intent        — an email request (address + subject) stays 'gửi email'
                           with the subject slot; never collapses to 'nhắn tin'.    [idx 152]

B3 smart_home:
  G4 keep-location/device — device + location/room said in a MIDDLE turn must survive
                            into the final short reference.                  [idx 61/119/305]
  G5 no-carry-prev-action — earlier already-done actions must NOT be re-issued in the
                            final rewrite; only the new final action (with its room). [idx 61]

These are plain SFT positives: more correct examples of the exact tricky structure
(no negative loss in sharegpt SFT). Output is the train.jsonl format directly so it
can be concatenated and re-split. A self-check asserts each gold omits the forbidden
tokens (pronoun-in-body / carried device) before writing.

Run: /opt/homebrew/bin/python3.11 -m src.data.patch_b2b3 --per 200
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

OUT = Path("data/processed/patch_b2b3.jsonl")
TRAIN = Path("data/processed/train.jsonl")
TRAIN_V2 = Path("data/processed/train_v2.jsonl")
BENCH = Path("data/bench/dialogues_bench_browser.jsonl")


def bench_golds():
    """Gold rewrites of the eval bench — patch samples must avoid these (no leakage)."""
    out = set()
    if not BENCH.exists():
        return out
    for l in BENCH.open(encoding="utf-8"):
        for c in json.loads(l)["conversations"]:
            if c["from"] == "gpt":
                try:
                    out.add(json.loads(c["value"])["rewrite_message"])
                except Exception:
                    pass
    return out

SYSTEM_PROMPT = (
    "Bạn là một module xử lý NGÔN NGỮ cho hệ thống trợ lý trong xe.\n\n"
    "Khi người dùng gửi yêu cầu có tag <REWRITE>, bạn PHẢI:\n"
    "1. Viết lại câu ở phía sau tag này thành MỘT câu hoàn chỉnh, đầy đủ ý nghĩa.\n"
    "2. Ngắn gọn, rõ nghĩa.\n"
    "3. Chỉ sử dụng thông tin có trong hội thoại trước đó nếu cần — KHÔNG thêm thông tin mới.\n"
    '4. Chỉ trả về JSON hợp lệ dạng: {"rewrite_message": "..."}'
)

# --------------------------------------------------------------------------- pools
RECIPIENTS = ["anh Nam", "chị Lan", "anh Tuấn Kiệt", "chị Hương", "sếp Cường",
              "anh Thắng", "chị Loan", "em Vân", "Mai", "chú Hải", "bố", "mẹ"]
PRONS = ["anh ấy", "cô ấy", "ông ấy", "bà ấy", "ảnh", "chị ấy"]
# message content cores — read naturally as standalone body AND after "Bảo <pron> ..."
MSG_CORES = [
    "họp dời sang 3 giờ chiều", "cuộc họp chuyển sang phòng A",
    "lịch hẹn đổi sang sáng mai", "deadline lùi đến thứ Sáu",
    "tối nay team liên hoan lúc 7 giờ", "ngày mai công ty nghỉ",
    "xe đã sửa xong, chiều qua lấy", "buổi training hoãn sang tuần sau",
    "chuyến công tác dời sang thứ Hai", "nhớ chuẩn bị tài liệu trước",
]
# content-agnostic closings so any extra fits any core
MSG_EXTRAS = ["nhớ phản hồi sớm", "gửi giúp trước 5 giờ", "có gì gọi lại nhé",
              "nhắn lại khi nhận được", "cảm ơn nhiều", "đừng quên nha"]
SMS_TIMES = ["7h", "8h", "9h", "10h", "6h30", "8h15"]
SMS_BASE = ["Hẹn {t} sáng ngày mai", "Gặp lúc {t} ở văn phòng",
            "Đón em lúc {t}", "Bắt đầu họp {t}"]
SMS_ADD = ["Mang hồ sơ", "Nhớ đem ô", "Mang theo USB", "Đi sớm 10 phút"]

EMAIL_WHO = ["sếp", "anh Hà", "phòng nhân sự", "khách hàng", "chị kế toán"]
EMAILS = ["nguyenvanha@techcorp.vn", "linh.tran@abc.com.vn", "hr@vinfast.vn",
          "contact@partner.com", "ketoan@company.vn", "minh.le@xyz.com.vn"]
SUBJECTS = ["Báo cáo tuần 24", "Kế hoạch Q3", "Đơn xin nghỉ phép",
            "Xác nhận đơn hàng", "Biên bản họp", "Đề xuất ngân sách"]
EMAIL_CORES = [
    "xin lỗi vì trễ hạn và cam kết nộp báo cáo trước 5 giờ chiều thứ Sáu",
    "xác nhận tham dự cuộc họp ngày mai lúc 9 giờ",
    "đề nghị duyệt ngân sách cho dự án mới",
    "thông báo sẽ nghỉ phép từ thứ Hai tuần sau",
    "gửi kèm biên bản và mong phản hồi trước cuối tuần",
]

HOME_PLACES = ["căn hộ Vinhomes Grand Park", "nhà ở Gamuda Gardens",
               "căn hộ Masteri Thảo Điền", "nhà ở Ecopark", "biệt thự Vinhomes Riverside"]
# device -> rooms where it plausibly lives (keeps generated pairings sensible)
DEVICE_ROOMS = {
    "bình nóng lạnh": ["phòng tắm", "phòng tắm tầng hai"],
    "máy nước nóng": ["phòng tắm", "phòng bếp"],
    "máy lọc không khí Xiaomi": ["phòng ngủ", "phòng khách"],
    "điều hoà": ["phòng ngủ", "phòng khách", "phòng làm việc"],
    "đèn sân vườn": ["sân vườn", "khu vườn trước nhà"],
}
SCHED_DEVICES = list(DEVICE_ROOMS)
CLEAN_ROOMS = ["phòng bếp", "phòng khách", "phòng ngủ", "phòng làm việc"]  # vacuum-able
INDOOR_ROOMS = ["phòng ngủ", "phòng khách", "phòng bếp", "phòng làm việc"]
VAC_MODES = ["im lặng", "tiêu chuẩn", "mạnh", "tiết kiệm"]
SH_TIMES = ["18 giờ 30", "19 giờ 10", "6 giờ sáng", "21 giờ", "17 giờ 45"]
SH_DURS = ["25 phút", "15 phút", "30 phút", "1 tiếng", "45 phút"]
# (device, done_phrase, room) already executed by the bot before the final turn.
# room=None → device carries no location; room set → establishes the scene location
# in a MIDDLE turn that the final rewrite must carry forward (while dropping the action).
DONE_DEVICES = [("đèn phòng ngủ", "bật đèn phòng ngủ", "phòng ngủ"),
                ("đèn phòng khách", "bật đèn phòng khách", "phòng khách"),
                ("đèn phòng bếp", "bật đèn phòng bếp", "phòng bếp"),
                ("máy lọc không khí Xiaomi", "bật máy lọc không khí Xiaomi chế độ turbo", None),
                ("điều hoà", "bật điều hoà 24 độ", None)]
# final NEW action — target noun is distinct from any light/AC device (so the
# no-carry self-check is unambiguous), and coherent with an indoor room.
NEW_ACTIONS = [
    ("kéo rèm cửa sổ xuống hết", "Kéo rèm cửa sổ {room} xuống hoàn toàn", "rèm"),
    ("đóng rèm lại", "Đóng rèm {room}", "rèm"),
    ("mở quạt trần", "Mở quạt trần {room}", "quạt trần"),
    ("bật đèn ngủ dịu", "Bật đèn ngủ dịu {room}", "đèn ngủ"),
]


def pick(s):
    return random.choice(s)


def word_in(word, text):
    """Whole-word containment (so pron 'ảnh' doesn't match inside 'sảnh')."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def cap(s):
    return s[0].upper() + s[1:] if s else s


def sample(turns, gold, domain, *, ctx=True, src="patch_b2b3"):
    """turns: list of (role, text) with role in {human,gpt}; final must be human."""
    conv = [{"from": "system", "value": SYSTEM_PROMPT}]
    for i, (role, text) in enumerate(turns):
        val = ("<REWRITE>\n" + text) if (role == "human" and i == len(turns) - 1) else text
        conv.append({"from": role, "value": val})
    conv.append({"from": "gpt", "value": json.dumps({"rewrite_message": gold}, ensure_ascii=False)})
    user_turns = sum(1 for r, _ in turns if r == "human")
    return {"conversations": conv,
            "meta": {"domain": domain, "context_required": ctx,
                     "user_turns": user_turns, "total_turns": len(turns), "source": src}}


# --------------------------------------------------------------------------- B2
def g1_no_pronoun_content():
    name, pron, core, extra = pick(RECIPIENTS), pick(PRONS), pick(MSG_CORES), pick(MSG_EXTRAS)
    turns = [
        ("human", f"Soạn tin nhắn cho {name}"),
        ("gpt", "Nội dung tin nhắn là gì ạ?"),
        ("human", f"Bảo {pron} {core}"),
        ("gpt", f"Tin nhắn: '{cap(core)} nhé'. Bạn muốn thêm gì không?"),
        ("human", f"Thêm vào '{extra}' rồi gửi đi"),
    ]
    gold = f"Gửi tin nhắn cho {name}: '{cap(core)}, {extra}'."
    s = sample(turns, gold, "messaging")
    assert not word_in(pron, gold), (pron, gold)  # body must not echo the pronoun
    return s


def g2_separate_time():
    name, t1, t2 = pick(RECIPIENTS), pick(SMS_TIMES), pick(SMS_TIMES)
    while t2 == t1:
        t2 = pick(SMS_TIMES)
    base, add = pick(SMS_BASE), pick(SMS_ADD)
    c1 = base.format(t=t1)
    c2 = base.format(t=t2)
    turns = [
        ("human", f"Gửi tin cho {name}: \"{c1}.\""),
        ("gpt", "Bạn muốn gửi ngay hay lên lịch?"),
        ("human", f"Thay giờ thành {t2}."),
        ("gpt", f"Giờ đã cập nhật thành {t2}."),
        ("human", f"Thêm nội dung: \"{add}.\""),
    ]
    gold = f"Gửi tin đến {name} vào {t2} nội dung \"{c2}. {add}.\""
    s = sample(turns, gold, "messaging")
    assert t2 in gold and gold.count(t2) >= 2, gold  # time as slot AND inside content
    return s


def g3_email_intent():
    who, email, subj, core = pick(EMAIL_WHO), pick(EMAILS), pick(SUBJECTS), pick(EMAIL_CORES)
    turns = [
        ("human", f"Soạn email cho {who}"),
        ("gpt", "Gửi đến địa chỉ nào ạ?"),
        ("human", f"{email}, tiêu đề là {subj}"),
        ("gpt", "Nội dung email là gì ạ?"),
        ("human", f"Bảo {core}"),
    ]
    gold = f"Soạn và gửi email đến {email}, tiêu đề '{subj}', nội dung: {core}."
    s = sample(turns, gold, "messaging")
    assert "email" in gold and subj in gold, gold  # intent + subject slot preserved
    return s


# --------------------------------------------------------------------------- B3
def g4_keep_location_device():
    place, time, dur = pick(HOME_PLACES), pick(SH_TIMES), pick(SH_DURS)
    if random.random() < 0.5:  # scheduled heater/AC with auto-off + location
        dev = pick(SCHED_DEVICES)
        room = pick(DEVICE_ROOMS[dev])
        turns = [
            ("human", f"Chuẩn bị sẵn cho tôi trước khi về {place}."),
            ("gpt", "Bạn muốn bật thiết bị nào trong nhà?"),
            ("human", f"{cap(dev)} {room}."),
            ("gpt", "Bạn muốn bắt đầu lúc nào?"),
            ("human", f"{time}."),
            ("gpt", "Có cần tự động tắt sau một khoảng thời gian không?"),
            ("human", f"Sau {dur}, lưu lịch này."),
        ]
        gold = f"Lưu lịch bật {dev} {room} tại {place} lúc {time} và tự động tắt sau {dur}."
        anchors = [dev, room, place, time, dur]
    else:  # robot vacuum with mode + location
        mode, room = pick(VAC_MODES), pick(CLEAN_ROOMS)
        turns = [
            ("human", "Lên lịch dọn nhà khi tôi đang lái xe về."),
            ("gpt", "Bạn muốn điều khiển thiết bị nào ở nhà?"),
            ("human", f"Robot hút bụi trong {place}."),
            ("gpt", "Bạn muốn dọn khu vực nào và vào lúc mấy giờ?"),
            ("human", f"{cap(room)}, lúc {time}."),
            ("gpt", "Bạn muốn dùng chế độ làm sạch nào?"),
            ("human", f"Chế độ {mode} nhé."),
        ]
        gold = f"Lên lịch cho robot hút bụi dọn {room} trong {place} lúc {time} bằng chế độ {mode}."
        anchors = ["robot hút bụi", room, place, time, mode]
    s = sample(turns, gold, "smart_home")
    for a in anchors:
        assert a in gold, (a, gold)  # every middle-turn slot survives
    return s


def g5_no_carry_prev_action():
    # dev_a establishes the scene ROOM in a middle turn (must be carried);
    # dev_b is a second already-done action (must NOT be carried).
    with_room = [d for d in DONE_DEVICES if d[2]]
    dev_a, done_a, room = pick(with_room)
    dev_b, done_b, _ = pick([d for d in DONE_DEVICES if d[0] != dev_a])
    final_user, gold_tmpl, target = pick(NEW_ACTIONS)
    open_u = pick([f"Tôi sắp về, {done_a} lên", f"Sắp tới nhà rồi, {done_a} giúp tôi",
                   f"Về tới nơi, {done_a} đi", f"Chuẩn bị về, {done_a}"])
    final_u = pick([f"Và {final_user} luôn nhé", f"{cap(final_user)} nữa nhé",
                    f"Tiện thể {final_user} giúp tôi", f"Rồi {final_user} luôn"])
    turns = [
        ("human", open_u),
        ("gpt", f"Đã {done_a}."),
        ("human", f"{cap(done_b)}"),
        ("gpt", f"Đã {done_b} trong {room}."),
        ("human", final_u),  # final turn: NO room → must infer from context
    ]
    gold = gold_tmpl.format(room=room) + "."
    s = sample(turns, gold, "smart_home")
    # already-done devices must NOT be carried; scene room + new target MUST appear
    assert dev_a not in gold and dev_b not in gold, (dev_a, dev_b, gold)
    assert room in gold and target in gold, (room, target, gold)
    return s


GENERATORS = {
    "g1_no_pronoun_content": g1_no_pronoun_content,
    "g2_separate_time": g2_separate_time,
    "g3_email_intent": g3_email_intent,
    "g4_keep_location_device": g4_keep_location_device,
    "g5_no_carry_prev_action": g5_no_carry_prev_action,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=200, help="samples per generator")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--merge", action="store_true", help="also write train_v2 = train + patch")
    ap.add_argument("--inplace", action="store_true",
                    help="append patch into data/processed/train.jsonl (for the Kaggle "
                         "pipeline: run AFTER split_data.py; idempotent since split "
                         "regenerates train.jsonl each run)")
    args = ap.parse_args()
    random.seed(args.seed)

    bench = bench_golds()
    rows, counts, seen, leak = [], {}, set(), 0
    for name, fn in GENERATORS.items():
        n = 0
        tries = 0
        while n < args.per and tries < args.per * 40:
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
        TRAIN_V2.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                            encoding="utf-8")
        print(f"merged: {len(base)} + {len(rows)} = {len(merged)} → {TRAIN_V2}")

    if args.inplace:
        base = [json.loads(l) for l in TRAIN.open(encoding="utf-8")]
        merged = base + rows
        random.shuffle(merged)
        TRAIN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                         encoding="utf-8")
        print(f"inplace: {len(base)} + {len(rows)} = {len(merged)} → {TRAIN} (overwritten)")


if __name__ == "__main__":
    main()
