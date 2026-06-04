# Báo cáo: mô hình sau fine-tune (`vi-rewriter`)

> Lập ngày 2026-06-04. Số liệu lấy từ các file đã judge (gpt-oss:120b-cloud, judge error 0%).
> **Lưu ý trạng thái:** base model trên bộ full 1044 CHƯA judge (để dành quota). So sánh
> head-to-head dưới đây mượn bộ pilot 500 (cùng judge, cùng mẫu = công bằng). Khi có quota,
> judge nốt base trên 1044 rồi mới có so sánh 1044-vs-1044 sạch.

## 0. Nguồn số liệu

| Bộ | Mẫu | Judge | Dùng cho |
|---|---|---|---|
| `full_gptoss/preds_vi-rewriter.jsonl` | **1044** | gpt-oss:120b-cloud, lỗi 0% | **Số chuẩn của bản fine-tune** |
| `pilot500_gptoss/` (cả 2 model) | 500 | gpt-oss, lỗi 0% | **So sánh trực tiếp base vs fine-tune** |

Bản fine-tune trên 500 (command_acc 64.6%) và trên 1044 (65.9%) gần như trùng → kết quả **ổn định**.

## 1. Kết quả tổng quan & mức cải thiện so với base

| Chỉ số | Base (trước) | **Fine-tune (sau)** | Δ |
|---|---|---|---|
| **Command Accuracy** (headline) | 48.6% | **65.9%** | **+17.3** |
| Intent Accuracy | 79.8% | 88.5% | +8.7 |
| Slot Completeness | 56.4% | 70.9% | +14.5 |
| Hallucination Rate ↓ | 12.8% | **7.9%** | **−4.9** (tốt) |
| Restoration-F1 | 0.412 | 0.554 | +0.142 |
| Output Validity | 100% | 100% | — |
| Copy Rate ↓ | 3.0% | 6.2% | +3.2 (xấu nhẹ) |
| Degenerate Rate ↓ | 2.0% | 0.2% | −1.8 (tốt) |
| **Negation Preservation** | **81.8%** | **74.8%** | **−7.0 ⚠️ thụt lùi** |

**Đọc nhanh:** fine-tune thắng rõ rệt gần như mọi mặt — đặc biệt headline `command_acc`
**+17 điểm** và giảm bịa đặt (hallucination) gần 5 điểm. Mô hình học được cách **khôi phục
thông tin từ lịch sử hội thoại** (Restoration-F1 nhảy từ 0.41 → 0.55).

Số liệu đầy đủ bản fine-tune (bộ 1044):
- n = 1044, judge_error_rate = 0%
- command_acc 0.6590 · intent_acc 0.8851 · slot_completeness 0.7088
- halluc_rate 0.0785 · negation_preservation 0.7479
- copy_rate 0.0623 · degenerate_rate 0.0019 · output_validity 1.0000
- restoration_f1 0.5538 · restoration_f2 0.5594

## 2. Hai vấn đề cần chú ý

**a) Phủ định bị thụt lùi (81.8% → 74.8%).** Đây là chỉ số DUY NHẤT fine-tune làm tệ hơn
base. Mô hình đang đánh rơi/đảo vế phủ định ("đừng bật…", "không đi qua…", "…nhưng KHÔNG
bật auto pilot"). Nghi vấn: dữ liệu train under-represent case phủ định, hoặc thay đổi
`mask_history` gần đây khiến vế phủ định ở turn trước bị che mất tín hiệu.
(Chỉ số này chỉ tính trên các dòng pattern = `negation`.)

**b) Khoảng cách context.** Khi câu cần lấy thông tin từ ngữ cảnh trước
(`context_required=True`): **61.4%**, còn khi không cần (`False`): **75.8%** → chênh **14 điểm**.
Đây chính là phần khó và cũng là lý do tồn tại của sản phẩm.

## 3. Điểm yếu cụ thể (theo slice, bộ 1044)

**Theo pattern (yếu → mạnh):**

| Pattern | Acc | Halluc | n | Nhận xét |
|---|---|---|---|---|
| `multi_turn_slot` | **52.8%** | 6.7% | 163 | Yếu nhất — bắc cầu slot qua nhiều lượt |
| `pronoun_resolution` | 54.7% | **14.6%** | 137 | Giải đại từ + **bịa referent nhiều nhất** |
| `correction` | 59.3% | 8.9% | 135 | "từ X xuống Y" — hay lấy nhầm X |
| `negation` | 62.2% | 2.5% | 119 | Khớp với vấn đề (a) |
| `implicit_reference` | 65.9% | 8.7% | 126 | |
| `code_switching` | 74.8% | 12.2% | 131 | Acc ổn nhưng **hallucinate cao** |
| `compound_intent` | 76.0% | 3.2% | 125 | |
| `irrelevant_context` | **89.8%** | 4.6% | 108 | Mạnh nhất — biết bỏ nhiễu tốt |

**Theo domain (yếu → mạnh):** `navigation` 60.0% · `charging` 61.9% · `calling` 62.1% ·
`messaging` 62.5% · `music` 63.2% · `smart_home` 64.8% · `vehicle` 66.7% ·
`driver_assist` 76.7% · `climate` 79.0%.

**Theo context_required:** True 61.4% (n=718, halluc 9.2%) vs False 75.8% (n=326, halluc 4.9%).

---

# Hướng điều chỉnh & nâng cấp

**Ưu tiên 1 — Sửa hồi quy phủ định (rõ ràng nhất, dễ đo):**
- Tăng mạnh tỉ lệ mẫu train có phủ định (mục tiêu ngang/hơn các pattern khác — hiện ≈11%).
- Soi lại `mask_history`: nếu nó che turn chứa vế phủ định thì mô hình mất tín hiệu —
  cân nhắc KHÔNG mask các turn mang ràng buộc phủ định.
- Thêm hard-case phủ định vào `fewshot_hard.jsonl` cho cả train lẫn baseline.

**Ưu tiên 2 — Multi-turn slot & pronoun (2 pattern đáy):**
- Sinh thêm data có mục tiêu cho `multi_turn_slot` và `pronoun_resolution`. Dùng chính các
  mẫu bench bị sai để mine hard-case (đối chiếu `pred` vs `gold` trong file preds).
- `pronoun_resolution` halluc 14.6% → thêm contrastive/negative example: dạy mô hình chỉ
  điền referent đã xuất hiện rõ, không suy diễn.

**Ưu tiên 3 — Đẩy headline bằng preference tuning:**
- Đã có judge pass/fail trên 1044 mẫu → có sẵn cặp dữ liệu. Cân nhắc DPO/ORPO trên tín hiệu
  judge (hoặc self-distill: giữ pred `score=1` làm positive) để nâng `command_acc` mà không
  cần data mới.

**Ưu tiên 4 — Kiểm tra under-training / cấu hình LoRA:**
- Intent đã 88.5% nhưng slot mới 70.9% → mô hình "hiểu đúng việc, thiếu chi tiết". Thử tăng
  epoch / LoRA rank, hoặc tăng trọng số loss ở phần slot.

**Việc cần làm để báo cáo sạch (khi có quota judge lại):**
- Judge nốt base trên 1044 (lệnh resume trong `RUN.md` đã sẵn) để có so sánh 1044-vs-1044.
- Sau đó mới xoá `pilot500/`, `pilot500_gptoss/`, `smoke_gptoss/`.

## Lệnh judge base còn dở (chạy lại khi có quota)

```bash
JUDGE_API_KEY=ollama /opt/homebrew/bin/python3.11 -m src.eval.eval_bench --resume \
  --models vi-rewriter Qwen/Qwen2.5-1.5B-Instruct \
  --judge-model gpt-oss:120b-cloud --judge-base-url http://localhost:11434/v1 \
  --judge-concurrency 4 \
  --output-dir data/bench/eval_results/full_gptoss
```
Resume chỉ judge các dòng `__JUDGE_ERROR__ pending` của base (1044), giữ nguyên judge của
vi-rewriter (đóng băng judge), rồi tạo lại `metrics_summary.json` đầy đủ cả 2 model.
