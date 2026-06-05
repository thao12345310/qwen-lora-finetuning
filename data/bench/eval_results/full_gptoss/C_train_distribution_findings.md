# C — Chẩn đoán phân phối train.jsonl (kiểm chứng giả thuyết gốc)

Script: `src/eval/analyze_train_distribution.py`, `src/eval/dump_dropped_negation.py`
Mẫu: 16,001 train. Không judge, thuần đếm local.

## Giả thuyết ban đầu (REPORT §C)
Train nghiêng về "câu ngắn, slot ở/gần lượt cuối" → model học *rewrite = làm gọn quanh lượt cuối* → rụng (a) vế phủ định và (b) qualifier nói-một-lần ở lượt giữa. Đây bị nghi là gốc chung của hồi quy **negation** và điểm yếu **multi_turn_slot**.

## Kết quả — giả thuyết phần lớn BỊ BÁC

### 1. Khôi phục từ lượt xa: ĐƯỢC PHỦ tốt, không thiếu
| max restore distance | n | % của context_required (11,068) |
|---|---:|---:|
| 0 (không khôi phục) | 420 | 3.8% |
| 1 (liền kề) | 4,343 | 39.2% |
| 2 | 2,292 | 20.7% |
| **≥3 (xa)** | **4,013** | **36.3%** |

→ **57% mẫu khôi phục slot từ lượt KHÔNG liền kề; 36% từ lượt xa (≥3).** Train **không** thiếu cấu trúc "qualifier ở lượt giữa". `multi_turn_slot` yếu **không phải** do thiếu mẫu far-restore → **đổ thêm data far-restore chung chung sẽ ít tác dụng**.

### 2. Gold negation: SẠCH, không phải thủ phạm
Sau khi lọc 3 loại false-positive của "không" (tiểu từ nghi vấn `...không?`, từ chối `Không, cảm ơn`, danh từ `không gian/không khí`):

| | n | % train |
|---|---:|---:|
| final-turn có phủ định THẬT | 641 | 4.0% |
| gold có phủ định | 595 | 3.7% |
| final có phủ định → gold giữ từ phủ định | 306 | 47.7% |
| final có phủ định → gold rụng từ phủ định | 335 | 52.3% |

"Rụng từ phủ định" nghe to nhưng **soi tay 95 ca nghi nặng nhất → gần như KHÔNG có ca rụng load_bearing thật**. Tất cả thuộc:
- **diễn giải tích cực** (đúng): `"đừng tắt máy"`→`"giữ cuộc gọi"`, `"đừng remix"`→`"bản gốc"`, `"không video nữa"`→`"gọi điện thoại thường"`, `"đừng chuyển làn"`→`"giữ nguyên làn"`.
- **no_op** (đúng khi bỏ): `"bật playlist trữ tình, không phải bài này"`, `"gửi luôn, không cần chờ xác nhận"`.
- **lý do/trang trí**: `"tắt đi, tôi không muốn bị làm phiền"` → positive đã đủ.

→ **Hồi quy negation KHÔNG do gold train sai.** Model đã được học các ví dụ load_bearing đúng (dạng paraphrase dương). Phần regression thật ≈ artifact đo lường ở bench (judge phạt no_op-drop) + một khe generalization nhỏ, **không** phải data hỏng.

### 3. Phân phối phụ
- context_required: 69.2% True / 30.8% False.
- user_turns ≤ 2 (hội thoại ngắn): 63.3% — có nghiêng ngắn, nhưng (do mục 1) không gây thiếu far-restore.
- gold đa mệnh đề (cue nhưng/rồi/`,`): 11.6%.
- Phủ định tập trung domain `navigation` (cao nhất), `calling`, `messaging`.
- source: 100% `mimo-v2.5-pro` (gold sinh từ một nguồn duy nhất).

## Hệ quả cho thứ tự ưu tiên (cập nhật so với REPORT §5)

| Đề xuất | Đánh giá lại sau C |
|---|---|
| **A4** sửa rubric/gold không phạt no_op-drop | **VẪN ưu tiên #1** — rẻ nhất, thuần đo lường, gỡ ~4đ "hồi quy ảo". Đã có nhãn `negation_types.jsonl`. |
| A1-A3 oversample load_bearing hard-neg | **HẠ ưu tiên** — gold không sai, model đã thấy ví dụ đúng. Lợi ích kỳ vọng thấp. Chỉ làm lượng nhỏ, nhắm `charging/navigation` param/ràng buộc. |
| B1 oversample far-restore chung | **BỎ/HẠ mạnh** — data đã phủ 36% far-restore. |
| B2/B3 messaging giữ ngôi + smart_home không carry | **GIỮ** — đây là micro-structure hẹp, đáng thêm vài chục hard-neg. |
| **C** | XONG (tài liệu này). |
| D guardrail inference | Bổ trợ, để sau. |
| E2 re-bench cùng judge | Sau khi áp A4. |
| **F** lên 3B/7B | **TĂNG ưu tiên tương đối** — vì data đủ & sạch, trần thấp (`multi_turn_slot` 52.8%, `pronoun` 54.7%) nhiều khả năng là **giới hạn dung lượng 1.5B**, không phải lỗ hổng data. |

**Một dòng:** data không phải thủ phạm như giả thuyết. Việc nên làm ngay là **A4 (sửa thước đo)**; phần data chỉ cần vá hẹp (B2/B3); nếu muốn nâng trần thật thì hướng có khả năng cao nhất là **F (model lớn hơn)**, không phải đổ thêm data.
