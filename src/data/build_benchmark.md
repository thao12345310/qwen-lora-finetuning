Việc xây dựng bộ dữ liệu benchmark cho dự án được thực hiện với mục đích duy nhất là **để đánh giá mô hình**, tuyệt đối không dùng làm dữ liệu huấn luyện (train). Quá trình xây dựng bộ benchmark này yêu cầu sự tỉ mỉ và tuân theo các tiêu chí sau:

**1. Cấu trúc và phân bổ dữ liệu**

- **Đảm bảo tỷ lệ thực tế:** Tương tự như tập dữ liệu huấn luyện, bộ benchmark bắt buộc phải áp dụng tỷ lệ **2/3 dữ liệu cần gọi ngữ cảnh (retrieve) và 1/3 dữ liệu giữ nguyên**. Bài test này nhằm kiểm tra chính xác xem mô hình có học được khả năng phân biệt lúc nào cần dùng lịch sử hội thoại và lúc nào phải bỏ qua hay không.
- **Phân loại theo ý định (Intent) và Pattern:** Dữ liệu cần được chia nhỏ theo các ý định khác nhau (ví dụ: 14 intents như nghe nhạc online/offline, tìm đường, v.v.). Các dữ liệu này được chia thành nhiều mẫu (pattern), mỗi pattern có thể gồm khoảng 125 sample.
- **Tuân thủ Format:** Dữ liệu đầu ra của benchmark phải chuẩn hóa theo đúng định dạng (format) của hệ thống yêu cầu, bao gồm các trường nội dung thô (row content) và các thẻ phân cách (tab) rõ ràng.

**2. Chiến lược sinh dữ liệu chất lượng cao bằng LLM**

- **Tận dụng các mô hình hàng đầu thế giới:** Vì bộ benchmark cần chất lượng "vàng" nhưng số lượng không quá lớn, cách tối ưu nhất là sử dụng trực tiếp các mô hình lớn như **ChatGPT, Claude hoặc Gemini** trên giao diện web thay vì tự chạy mô hình nhỏ. 
- **Kỹ thuật Prompting và sinh hàng loạt:** Yếu tố quyết định là phải học cách thiết kế prompt thật chuẩn để mô hình sinh ra dữ liệu tốt. Bạn có thể áp dụng chiến lược mở cùng lúc nhiều tab (nhiều mô hình khác nhau) và yêu cầu chúng sinh dữ liệu theo từng đợt (ví dụ: 10 sample mỗi lần, lặp lại khoảng 100 lần) để gom được tập dữ liệu chất lượng nhất.

**3. Kiểm tra chéo chất lượng Benchmark (Quality Check)**
Sau khi tạo xong, bản thân bộ benchmark cũng cần được rà soát lại thông qua một hệ thống kiểm tra hai lớp:

- **Lớp thứ nhất:** Chuyên rà soát từ V1 đến V10 để đánh giá chất lượng về mặt ngữ nghĩa của câu.
- **Lớp thứ hai:** Chuyên phát hiện và loại bỏ các lỗi ảo giác (hallucinate) để đảm bảo độ tin cậy tuyệt đối trước khi đem đi chấm điểm mô hình.

---

## Phụ lục A — Prompt SINH benchmark (dán vào ChatGPT / Claude / Gemini web)

> Mỗi lần chạy ra 10 mẫu. Lặp lại "tiếp 10 mẫu, đổi pattern + domain + scenario" cho tới khi đủ. Mở nhiều tab/model song song để tăng đa dạng. Output JSON ăn thẳng vào `src/data/ingest_browser_bench.py`.

```text
# VAI TRÒ
Bạn là chuyên gia tạo dữ liệu BENCHMARK tiếng Việt cho task "rewrite hội thoại trong xe ô tô".
Bộ này CHỈ dùng để ĐÁNH GIÁ một model offline nhỏ (single-turn) — KHÔNG dùng để train —
nên gold phải chuẩn 100% và ca phải đủ khó.

# NHIỆM VỤ
Model nhỏ phải "retrieve" đủ ngữ cảnh từ lịch sử hội thoại để viết lại lượt user CUỐI thành
MỘT câu lệnh độc lập, đầy đủ slot, đúng intent. Bạn sinh hội thoại nhiều lượt (user/bot xen
kẽ, KẾT THÚC ở lượt user) + câu "rewrite" chuẩn cho lượt user cuối.

# TỶ LỆ NGỮ CẢNH (BẮT BUỘC ~2/3 : 1/3 trong mỗi đợt 10 mẫu)
- context_required=true  (~7/10): lượt cuối KHÔNG tự đủ nghĩa → phải dùng lịch sử mới rewrite được.
- context_required=false (~3/10): lượt cuối ĐÃ tự đủ thông tin (vd "Chỉ đường tới Hồ Gươm",
  "Bật đèn pha như tôi yêu cầu đi") → rewrite gần như giữ nguyên, KHÔNG lôi thêm gì từ lịch sử.
- KHÔNG LẶP HÀNH ĐỘNG: nếu bot đã làm xong hành động A, lượt cuối user yêu cầu B mới thì
  rewrite CHỈ chứa B, không nhắc lại A.

# 8 PATTERN KHÓ (mỗi đợt trộn ≥5 pattern)
1. pronoun_resolution — đại từ "cái đó/bài đó/người thứ hai/chỗ gần hơn" → resolve về slot bot nêu.
2. irrelevant_context — bot nêu số nhiễu (nhiệt độ ngoài, % pin, giá xăng), user xin slot KHÁC;
   rewrite KHÔNG kéo nhầm số của bot.
3. multi_turn_slot    — slot rải rác 2–4 lượt user, lượt cuối rất ngắn; rewrite GỘP đủ mọi slot.
4. correction         — user đổi X→Y hoặc huỷ; rewrite phản ánh intent CUỐI.
5. code_switching     — token tiếng Anh/brand (Spotify, Bluetooth, auto pilot, FM 99.9);
   rewrite GIỮ NGUYÊN token, không dịch.
6. implicit_reference — "nhà tôi/công ty/trường con"; có info trước thì resolve, không có thì
   GIỮ NGUYÊN cụm ngầm (KHÔNG bịa địa chỉ/SĐT).
7. negation           — "đừng đi cầu X/trừ bài Y/không bật điều hoà"; rewrite giữ rõ phủ định.
8. compound_intent    — lượt cuối 2 intent liền ("đổi bài rồi giảm volume"); rewrite giữ CẢ 2.

# INTENT & DOMAIN (gán đúng, ưu tiên đa dạng — ĐỪNG dồn navigation)
domain: navigation, climate, music, calling, messaging, charging, smart_home, driver_assist, vehicle
intent: ví dụ play_music_online, play_music_offline, navigate, find_charging, call, send_message,
  set_climate, control_window, smart_home_control, set_alarm, cruise_control, lane_assist,
  toggle_light, cancel_action  (chọn intent sát nhất với hành động ở lượt cuối).
→ Ưu tiên domain ít gặp: messaging, charging, smart_home, driver_assist, vehicle.

# RÀNG BUỘC CHẤT LƯỢNG (vi phạm = mẫu hỏng)
- Slot THẬT & cụ thể: tên người VN, địa danh HN/SG có thật, bài Vpop thật, hãng EV thật, nhiệt độ/giờ hợp lý.
- KHÔNG bịa số: mọi con số trong rewrite phải có trong hội thoại.
- Văn nói tự nhiên trong xe (lược chủ ngữ, nói nhanh, slang nhẹ).
- BIẾN TẤU opener rewrite — KHÔNG luôn "Tôi muốn..."; dùng "Hãy.../Làm ơn.../Cho tôi.../Mình muốn..."/mệnh lệnh ngắn.
- Lượt user cuối ≤ 25 từ. rewrite là MỘT câu, 5–240 ký tự.
- Không trùng scenario/từ vựng giữa các mẫu trong đợt.

# ĐẦU RA — CHỈ in JSON, không giải thích
{
  "samples": [
    {
      "pattern": "<1 trong 8>",
      "domain": "<1 trong 9>",
      "intent": "<intent sát nhất>",
      "context_required": true,
      "turns": [
        {"role":"user","content":"..."},
        {"role":"bot","content":"..."},
        {"role":"user","content":"..."}
      ],
      "rewrite": "câu rewrite chuẩn cho lượt user cuối",
      "rationale": "1 câu vì sao ca này khó"
    }
  ]
}

# YÊU CẦU LẦN NÀY
Sinh 10 mẫu: trộn ≥5 pattern, ≥5 domain; ~7 mẫu context_required=true, ~3 mẫu false;
đa dạng số lượt (2,3,4 lượt user). Sáng tạo, không lặp khuôn.
```

  # YÊU CẦU LẦN NÀY (đợt cân bằng)

  - CHỈ sinh các pattern: negation, irrelevant_context, compound_intent, implicit_reference, code_switching.

    TUYỆT ĐỐI KHÔNG sinh: multi_turn_slot, pronoun_resolution, correction (đã đủ).

  - Đặt context_required=false cho ÍT NHẤT 7/10 mẫu (đang thiếu mẫu giữ-nguyên).

  - Nhắc lại 2 luật hay bị sai:

    * negation: câu rewrite BẮT BUỘC còn nguyên từ phủ định "đừng/trừ/không/tránh".

    * KHÔNG bịa số — mọi con số trong rewrite phải có trong hội thoại.

## Phụ lục B — Prompt QC 2 lớp (chạy sau khi gom data — temperature thấp)

```text
# VAI TRÒ
Bạn là giám khảo rà soát chất lượng bộ benchmark rewrite hội thoại xe ô tô. Hãy chấm CHẶT, deterministic.

# INPUT: một list mẫu, mỗi mẫu có turns / rewrite / pattern / domain / context_required.

# LỚP 1 — Ngữ nghĩa (cho điểm V1..V10, mỗi tiêu chí 0/1)
V1 rewrite là MỘT câu, độc lập, đủ nghĩa
V2 giữ ĐÚNG intent của lượt user cuối
V3 chứa ĐỦ mọi slot cần thiết (gộp đủ nếu multi_turn_slot)
V4 resolve đúng đại từ / cụm ngầm (nếu có)
V5 giữ đúng phủ định "đừng/trừ/không" (nếu negation)
V6 giữ cả 2 intent (nếu compound_intent)
V7 nếu context_required=false: KHÔNG lôi thêm thông tin thừa từ lịch sử
V8 KHÔNG lặp lại hành động bot đã làm xong
V9 token tiếng Anh/brand giữ nguyên (nếu code_switching)
V10 văn phong tự nhiên, opener không rập khuôn

# LỚP 2 — Hallucination (loại bỏ tuyệt đối)
H1 mọi số trong rewrite PHẢI có trong hội thoại (sai → loại)
H2 không bịa địa chỉ/tên/SĐT không có trong hội thoại (sai → loại)
H3 với irrelevant_context: rewrite KHÔNG được chứa số nhiễu mà chỉ bot nêu

# ĐẦU RA — chỉ JSON
{"results":[{"index":0,"v_score":"x/10","fail":["V3","H1"],"verdict":"keep|fix|drop","note":"1 câu"}]}
Quy tắc verdict: bất kỳ H nào fail → drop; V<8 → fix; còn lại keep.
```

## Phụ lục C — Nạp data browser về

JSON model trả về (đợt 10 mẫu) lưu vào file bất kỳ trong `data/bench/browser_raw/` (chấp nhận
`{"samples":[...]}`, mảng `[...]`, jsonl, hoặc có rào ```json). Sau đó:

```bash
python -m src.data.ingest_browser_bench data/bench/browser_raw/*.json --report
```

Mặc định ghi vào `data/bench/dialogues_bench_browser.jsonl` (file RIÊNG — KHÔNG đụng tới bộ
`dialogues_bench.jsonl` sinh từ API). Script reuse `validate_sample` + `to_lf_record` của
`build_benchmark.py`, kiểm hallucinate số, dedup, và xuất đúng format Llama Factory
(`<REWRITE>` + `{"rewrite_message": ...}`), giữ thêm `intent` và `context_required` vào `meta`.
Mỗi lần chạy đọc lại toàn bộ `browser_raw/*.json` nên chạy lại an toàn, không nhân đôi.