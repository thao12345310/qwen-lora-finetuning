# Báo cáo Benchmark — vi-rewriter (LoRA) vs Qwen2.5-1.5B baseline

## 1. Thiết lập

| Hạng mục | Giá trị |
|---|---|
| Bench | `data/bench/dialogues_bench_browser.jsonl` — **1044 mẫu** |
| Model train | `vi-rewriter` (LoRA trên Qwen2.5-1.5B-Instruct), serve qua vLLM, gửi record AS-IS theo format train |
| Baseline | `Qwen/Qwen2.5-1.5B-Instruct` (chưa train) — prompt chi tiết `BASELINE_SYSTEM_PROMPT` + 6 few-shot |
| Judge | **`gpt-oss:120b-cloud`** (Ollama), đóng băng cho cả 2 model, `judge_error_rate = 0%` |
| Phương pháp | predict (vLLM) và judge (local) tách 2 bước; judge chấm 4 cờ → `score=1` chỉ khi cả 4 = 1 |

## 2. Kết quả tổng (n=1044 mỗi model)

| Metric | LoRA | Baseline | Δ |
|---|---:|---:|---:|
| Output Validity | 100.0% | 100.0% | +0.0 |
| **Command Accuracy** ★ | **65.9%** | 52.8% | **+13.1** |
| Intent Accuracy | 88.5% | 81.0% | +7.5 |
| Slot Completeness | 70.9% | 58.7% | +12.2 |
| Halluc. Slot Rate ↓ | 7.9% | 11.8% | −3.9 ✅ |
| Negation Preservation | 74.8% | 84.9% | **−10.1** ⚠️ |
| Copy Rate ↓ | 6.2% | 6.9% | −0.7 |
| Degenerate Rate ↓ | 0.2% | 2.7% | −2.5 ✅ |
| Restoration-F1 | 0.554 | 0.404 | +0.150 |

## 3. Phân tích theo lát cắt

### 3.1 context_required (mẫu có cần khôi phục thông tin từ ngữ cảnh hay không)

| | acc LoRA | acc base | Δ | halluc LoRA | halluc base |
|---|---:|---:|---:|---:|---:|
| False (n=326) | 75.8% | 74.2% | +1.6 | 4.9% | 12.3% |
| True (n=718) | **61.4%** | **43.0%** | **+18.4** | 9.2% | 11.6% |

→ Khi **không** cần ngữ cảnh, hai model gần ngang nhau (cả base cũng làm tốt nhờ few-shot). **Toàn bộ giá trị của fine-tune nằm ở nhóm cần ngữ cảnh (+18.4đ)** — đúng mục tiêu của task rewriter.

### 3.2 Theo pattern (LoRA vs base, command-acc)

| pattern | LoRA | base | Δ |
|---|---:|---:|---:|
| multi_turn_slot | 52.8% | 25.2% | **+27.6** |
| implicit_reference | 65.9% | 43.7% | +22.2 |
| pronoun_resolution | 54.7% | 40.9% | +13.9 |
| code_switching | 74.8% | 60.3% | +14.5 |
| irrelevant_context | 89.8% | 76.9% | +13.0 |
| compound_intent | 76.0% | 67.2% | +8.8 |
| correction | 59.3% | 57.0% | +2.2 |
| **negation** | **62.2%** | **63.9%** | **−1.7** |

### 3.3 Theo domain — LoRA thắng mọi domain trừ `vehicle` (−1.6); cách biệt lớn ở `messaging` (+23.2), `navigation` (+21.4), `calling` (+17.9), `charging` (+16.8), `smart_home` (+16.0).

## 4. Nhận xét

**Điểm mạnh (fine-tune ăn tiền ở đúng chỗ):**
- +13.1đ command-acc tổng, nhưng quan trọng hơn là **+18.4đ ở nhóm context_required** và **+27.6đ ở multi_turn_slot** — fine-tune dạy được model khôi phục slot bắc cầu / tham chiếu ngầm / đại từ, thứ base không tự làm được dù có few-shot.
- **Ít bịa hơn** (7.9% vs 11.8%) và **gần như không degenerate** (0.2% vs 2.7%): output ngắn gọn, đúng format, sạch.
- Restoration-F1 0.554 vs 0.404 xác nhận model đưa được đúng thông tin "phải khôi phục" vào câu lệnh.

**Điểm yếu — hồi quy negation (headline −10.1đ, thực −6.7đ sau khi tha đúng no_op — xem A4):**
- Cơ chế: LoRA **trim quá tay, bỏ hẳn vế phủ định** trong câu compound dạng *"làm X **nhưng không** làm Y"*. Base copy literal hơn nên vô tình giữ vế "không Y". Lỗi **không phải đảo nghĩa** ("không bật"→"bật") mà là **rụng mệnh đề** → trừ kép cả `slots_complete` lẫn `negation_ok`. (`no_hallucination` của LoRA còn *cao hơn* base: 97.5% vs 88.2% — model rất sạch, chỉ là cắt quá.)

- **Hiệu chỉnh quan trọng — không phải vế phủ định nào rớt cũng sai.** Câu lệnh rewrite được đưa cho model nhỏ gọi tool; một thiết bị chỉ đổi khi có **lệnh DƯƠNG**. Nên *"đừng bật B"* với B độc lập, đang tắt = **no-op** (bỏ đi cũng không gọi tool B → cùng end-state). Đã phân loại 119 mẫu negation bằng judge gpt-oss (cache: `negation_types.jsonl`):

  | Loại vế phủ định | n | Ý nghĩa |
  |---|---:|---|
  | **no_op** | 30 | phủ định hành động bật/mở độc lập, off-by-default → bỏ vô hại |
  | **load_bearing** | 84 | param/mode ("đừng dùng sạc nhanh"), ràng buộc ("không qua cầu X"), huỷ ("đừng mở nữa") → **bắt buộc giữ** |
  | none | 5 | gold không thực sự có phủ định |

- **Negation preservation sau khi tha no_op-drop (áp cho cả 2 model):**
  > ⚠️ Cập nhật A4 (`A4_negation_corrected.md`): luật tha *blanket* dưới đây over-forgive 3 ca **inversion** (model đảo `"Không bật X"`→`"Bật X"`, không phải bỏ vô hại). Tha **đúng** (chỉ clean-drop) cho **LoRA 81.5% / base 88.2% / Δ −6.7** — phần hồi quy thật LỚN hơn, không nhỏ hơn. Dùng số này về sau.

  | | LoRA | base | Δ |
  |---|---:|---:|---:|
  | Gốc (phạt mọi drop) | 74.8% | 84.9% | −10.1 |
  | Tha clean-drop (đúng) | **81.5%** | **88.2%** | **−6.7** |
  | ~~Tha blanket (over-forgive)~~ | ~~83.2%~~ | ~~89.1%~~ | ~~−5.9~~ |

- Trong 30 ca LoRA bị trừ negation: **10 no_op (tha được)** + **20 load_bearing (sai thật)**. Tha no-op xoá ~4đ khoảng cách, nhưng **~6đ còn lại là thật** — LoRA rớt đúng loại phủ định làm-sai-tool-call:
  - load_bearing: *"Dừng sạc... **nhưng đừng dùng sạc nhanh**"* → LoRA bỏ mode sạc nhanh.
  - load_bearing: *"...tuyến nhanh, **không đi qua cầu Nhật Tân**"* → ràng buộc tuyến.
  - load_bearing: cửa vừa mở → *"**đừng mở nữa**"* → lệnh huỷ, LoRA không thực hiện.
  - no_op (LoRA thực ra ổn): *"bật giữ làn, **không bật auto pilot**"* → "Bật giữ làn" (auto pilot vốn tắt, độc lập).
- `correction` (+2.2đ) cùng họ lỗi rút gọn làm rơi vế ràng buộc.

**Điểm yếu — multi_turn_slot là pattern YẾU NHẤT về tuyệt đối (52.8%), dù LoRA thắng base +27.6đ:**
- Đây vừa là chỗ fine-tune ăn tiền nhất (base chỉ 25.2%) vừa là trần thấp nhất còn lại. n=163, LoRA sai **77 ca (47%)**.
- **Bản chất: rớt slot, không phải sai intent.** 72/77 ca sai (**94%**) là `slots_complete=0`; chỉ 21 ca sai intent.
- **Điểm chung của lỗi**: lượt cuối user là **câu tham chiếu ngắn** ("chỉ đường đến đó", "gọi luôn đi"); gold gom **mọi qualifier rải khắp hội thoại**. LoRA bắt được intent + referent gần nhất nhưng **đánh rơi qualifier được nói MỘT LẦN ở lượt giữa, không lặp lại**.

  | Domain (số fail) | Slot bị rớt điển hình |
  |---|---|
  | **charging (16)** ← nặng nhất | qualifier loại sạc: *"sạc nhanh / DC / 150kW"* |
  | messaging (12) | nội dung tin nhắn sai — **đổi ngôi** "Họp dời…" → "**Anh ấy** họp dời…" |
  | smart_home (11) | vị trí *"phòng ngủ"*; hoặc **carry nhầm** hành động cũ vào (kèm halluc) |
  | navigation (9) | ràng buộc tuyến: *"tránh đường Võ Chí Công"*, *"gần Hồ Tây"* |
  | vehicle/calling (14) | hạng mục *"đảo lốp, kiểm tra phanh"*; phòng ban *"bộ phận giao xe"* |

- **3 biến thể**: (1) rớt qualifier ở xa — chủ đạo; (2) sai nội dung/đổi ngôi tin nhắn — messaging; (3) carry nhầm hành động lượt trước — smart_home (kèm `no_hallucination=0`).
- **Không phải do hội thoại dài**: 3-lượt 48.6% còn ≥4-lượt 59.6% — độ dài không phải thủ phạm; thủ phạm là slot nhắc đúng một lần ở lượt giữa rồi bị quên khi tổng hợp.
- 📄 Toàn bộ 77 ca (dialogue + gold + pred + lý do judge): **`errors_multi_turn_slot.txt`**.

## 5. Hướng cải thiện đề xuất

**A. Vá data huấn luyện — nhắm ĐÚNG load_bearing, KHÔNG ép giữ no_op (ưu tiên cao, rẻ nhất):**
1. **Oversample riêng loại load_bearing**: phủ định = param/mode ("đừng dùng sạc nhanh", "không bật điều hoà ngoài"), ràng buộc ("không đi qua cầu X", "tránh đường Y"), huỷ/undo ("đừng mở nữa"). Đây là 20/30 lỗi thật của LoRA. Tập trung domain `charging`/`navigation`/`vehicle`/`driver_assist`.
2. Bổ sung **hard negatives**: cặp (gold giữ đúng param/ràng buộc/huỷ) vs (biến thể sai rớt vế) để model học vế load_bearing là **slot bắt buộc**.
3. **Không** oversample no_op-negation — nếu ép model giữ cả phủ định trang trí sẽ kéo lùi về thói copy literal, mất lợi thế gọn-sạch (degenerate 0.2%, halluc 7.9%) mà LoRA đang có. Lý tưởng: chuẩn hoá gold để lược no_op (rewrite gọn vẫn đúng), chỉ giữ load_bearing.
4. **Sửa rubric judge / gold để đo công bằng** — ✅ ĐÃ LÀM (A4, `A4_negation_corrected.md`): tha `negation_ok` cho no_op-clean-drop (không tha inversion), tính lại post-hoc không cần re-judge → **LoRA 81.5% vs base 88.2% (gap −6.7đ)**. Command-acc không đổi (65.9%). Kết luận: hồi quy negation là THẬT (~6.7đ, gồm cả ca đảo `"không bật"`→`"bật"`), không phải artifact.

**B. Vá data cho multi_turn_slot — slot ở lượt xa (ưu tiên cao, đây là trần thấp nhất 52.8%):**
1. **Oversample cấu trúc "qualifier nói 1 lần ở lượt giữa → lượt cuối tham chiếu ngắn"**, ép model gom qualifier xa vào lệnh cuối. Nặng nhất: `charging` (loại sạc/cổng/kW), `navigation` (ràng buộc tuyến).
2. **messaging**: thêm mẫu giữ NGUYÊN ngôi + nội dung tin nhắn; hard negative cấm đổi "tôi/anh ấy" hay thêm chủ ngữ (lỗi "Anh ấy họp dời…").
3. **smart_home**: dạy chỉ lấy hành động ở lượt cuối, KHÔNG carry trạng thái/hành động lượt trước (lỗi "giữ nguyên đã bật máy lọc…" + halluc).

**C. Kiểm tra lại phân phối train hiện tại:** thống kê tỉ lệ mẫu chứa phủ định / chứa ≥2 mệnh đề / có slot ở lượt không-phải-cuối trong `data/processed/train.jsonl`. Nghi vấn: data nghiêng về câu đơn ngắn, slot ở lượt cuối → model học "rewrite = làm gọn quanh lượt cuối", nên rớt vế xa. Nếu đúng, đây là gốc rễ chung cho cả negation lẫn multi_turn_slot.

**D. Guardrail lúc inference (vá nhanh, bổ trợ):** detector phủ định đơn giản — nếu câu cuối của user chứa *"không/đừng/chớ/tránh/khỏi"* mà rewrite không còn từ phủ định nào → cờ cảnh báo / fallback. Không sửa được gốc nhưng chặn được lỗi rụng vế hiển nhiên.

**E. Mở rộng đánh giá để củng cố kết luận:**
1. Bật lại **20 hard case** đang tắt (server `max_model_len=2048` cắt mất) khi nâng context vLLM ≥4096 — đây đúng là nhóm khó hay sai.
2. Sau khi vá data, re-train rồi re-bench **cùng judge gpt-oss:120b-cloud** (đóng băng) — mục tiêu kéo Negation ≥ baseline (≥85%) mà **không tụt** các pattern context (giữ multi_turn_slot, implicit_reference).
3. (Tuỳ chọn) đối chứng 1 lần với judge thứ hai (Gemini-2.5-flash có billing) trên ~150 mẫu để kiểm tra độ ổn định xếp hạng, **không** đổi judge giữa đợt.

**F. Hướng nâng cấp model (nếu cần trần cao hơn):** thử LoRA trên **Qwen2.5-3B/7B-Instruct** (7B cần T4×2 TP=2, fp16) — kỳ vọng cải thiện các pattern suy luận (pronoun_resolution 54.7%, multi_turn_slot 52.8% vẫn còn thấp tuyệt đối).

---
*Nguồn: `data/bench/eval_results/full_gptoss/preds_*.jsonl`, `metrics_summary.json`. Judge gpt-oss:120b-cloud, 0% judge error.*
