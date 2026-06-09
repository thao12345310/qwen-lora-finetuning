# Improvement reports — index

Mỗi lần cải tiến model/eval → **1 file riêng** `reports/vMAJOR.MINOR_<slug>.md`.

**Version scheme**
- **MAJOR** tăng khi **model thay đổi** (re-train, vá data rồi train lại, đổi base model).
- **MINOR** tăng cho mỗi bước cải tiến trong cùng một model (chẩn đoán, sửa thước đo, thí nghiệm nhỏ).
- Mỗi báo cáo tự chứa, theo template ở cuối file này, và link tới chi tiết/sản phẩm.

## Changelog

| Version | Ngày | Loại | Tiêu đề | Kết quả chính |
|---|---|---|---|---|
| [v1.0](v1.0_baseline_full_bench.md) | 2026-06-04 | bench | Full bench LoRA vs baseline (n=1044) | Command-acc LoRA 65.9% vs base 52.8% (+13.1) |
| [v1.1](v1.1_diagnostic_train_distribution.md) | 2026-06-05 | diagnostic | Chẩn đoán phân phối train.jsonl (C) | Data SẠCH; weakness không do thiếu/hỏng data |
| [v1.2](v1.2_negation_measurement_fix.md) | 2026-06-05 | eval-fix | Sửa thước đo negation, tha no_op (A4) | Hồi quy negation THẬT −6.7đ (LoRA 81.5 / base 88.2) |
| [v1.3](v1.3_data_patch_b2b3.md) | 2026-06-05 | data | Vá data hẹp B2/B3 cho multi_turn_slot | +1000 mẫu nhắm 5 cấu trúc lỗi, 0 leak → train_v2 (17001) |
| [v2.0](v2.0_retrain_patch_b2b3_bench.md) | 2026-06-06 | model | Re-train data vá B2/B3 + re-bench (n=1044) | multi_turn_slot 52.8→57.1 (+4.3), implicit_reference +3.1; command-acc tổng phẳng (65.9→65.6); negation LoRA A4-corrected 81.5 (= v1.0, chưa giải quyết) |
| [v2.1](v2.1_error_analysis_next.md) | 2026-06-06 | diagnostic | Error-analysis bench v2.0 + thiết kế patch B4/slot-retention/confirm | Lỗi chính: slot-incomplete 302/84% + clause-drop/negation 81 (xuyên pattern: correction 27); gốc = SFT over-compression. Demo phát hiện lỗ hổng xác-nhận-đề-xuất (user "ờ"/"ok" đồng ý → LoRA bịa tool-call) mà bench mù → thêm gen n4_confirm_proposed + mở rộng bench |
| [v2.2](v2.2_data_rebalance_brainstorm.md) | 2026-06-08 | data | Cân lại tỉ lệ data train (severity-weighted) + brainstorm case thực tế | Bỏ `--per` đồng đều → WEIGHTS nghịch-acc (dồn pronoun/correction/code_switching/slot, trim multi_turn/compound, ~13.5%); thêm 2 gen lấp lỗ hổng p1_pronoun_resolution (52.6%, tệ nhất) + c1_language_control (regress+CJK); catalog 12 case thực tế cho backlog |
| [v2.3](v2.3_confirmation_benchmark.md) | 2026-06-08 | bench | Mở rộng benchmark slice confirmation bằng Ollama Cloud | Thêm 50 mẫu `confirmation` vào `dialogues_bench_browser.jsonl` (1044→1094), sinh bằng `gpt-oss:120b-cloud` + few-shot + validator; 0 malformed, 0 duplicate, 0 overlap gold train/patch |

---

## Template cho báo cáo mới

```markdown
# vX.Y — <Tiêu đề> (<loại: bench|diagnostic|eval-fix|data|model>)

- **Ngày:** YYYY-MM-DD
- **Model:** <vi-rewriter LoRA / Qwen base / …>  | **Bench:** <file, n>
- **Động cơ:** <vá điểm yếu nào / câu hỏi gì>

## Đã làm
- <bước 1, script/lệnh dùng>
- <bước 2>

## Kết quả
| Metric | Trước | Sau | Δ |
|---|---:|---:|---:|

## Kết luận
<1-3 câu chốt>

## File tạo/sửa
- `path` — mô tả

## Bước tiếp
- <next>
```
