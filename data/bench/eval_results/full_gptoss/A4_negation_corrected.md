# A4 — Sửa thước đo negation (tha no_op-drop), tính lại post-hoc

Script: `src/eval/recompute_no_op_forgiven.py` (KHÔNG judge lại — chỉ tái-tổng-hợp cờ
`negation_ok` đã có trong `preds_*.jsonl`; judge gpt-oss đóng băng, áp cùng luật cho cả 2 model).
Nguồn nhãn: `negation_types.jsonl` (no_op=30, load_bearing=84, none=5).
Output máy đọc: `A4_corrected_metrics.json`; audit: `no_op_forgiven_flips.jsonl`.

## Luật tha (đã siết chặt so với report)
Một mẫu negation gắn nhãn **no_op** chỉ được tha (`negation_ok:=1`) khi model **BỎ SẠCH** vế phủ định — tức pred **không nhắc** token phân biệt của mục tiêu bị cấm. **KHÔNG tha** khi model *đụng* mục tiêu đó (đảo `"Không bật Lane Assist"`→`"Bật Lane Assist"`, hay thêm lệnh `"tắt khóa trẻ em"`) — đó là gọi tool sai, không phải bỏ vô hại.

Token phân biệt = token vế-sau-phủ-định, trừ động từ chung (`bật/tắt/mở…`) và trừ token đã có ở vế dương (nên `đèn/bản/cảnh báo` dùng chung không tính; chỉ `trần/gốc/sau/lane assist…` mới tính).

## Kết quả (Negation Preservation = `negation_ok` trên negation pattern, n=119)

| | gốc | **tha strict (clean-drop)** | tha blanket (report) |
|---|---:|---:|---:|
| LoRA | 74.8% | **81.5%** | 83.2% |
| Baseline | 84.9% | **88.2%** | 84.9→89.1% |
| **Δ (base − LoRA)** | **10.1** | **6.7** | 5.9 |

- Tha hợp lệ cho LoRA: **8/10** ca no_op-fail là clean-drop (tha), **2 ca là inversion thật** (giữ): idx 840 `"Không bật Lane Assist"`→`"Bật Lane Assist"`; idx 956 `"đừng bật khóa trẻ em"`→thêm `"tắt khóa trẻ em"`. Base: 4 tha, 1 inversion (idx 970 `"đừng bật cảnh báo…"`→`"tắt cảnh báo…"`).
- **Report over-forgive**: blanket tha cả 3 ca inversion → báo gap 5.9đ. Con số **đúng là 6.7đ** — phần hồi quy thật **lớn hơn** report nói, không nhỏ hơn.

## Hai kết luận quan trọng

1. **Command Accuracy KHÔNG đổi** sau khi tha: LoRA 65.9%→65.9%, base 52.8%→52.8%. Tha `negation_ok` đơn lẻ không cứu được `score` (các ca đó vẫn rớt cờ khác như `slots_complete`). → việc sửa thước đo này **chỉ làm trung thực sub-metric negation**, KHÔNG đụng tới headline command-acc. Lợi thế tổng của LoRA (+13.1đ) vẫn nguyên.

2. **Hồi quy negation là THẬT (~6.7đ), không phải artifact đo lường** như nghi ban đầu. Sau khi tha đúng phần vô hại, LoRA vẫn kém base 6.7đ, và trong đó có **ca đảo nghĩa thẳng** (`"Không bật X"`→`"Bật X"`). Khớp với chẩn đoán C (gold train sạch) → đây là **lỗi hành vi/dung lượng của model 1.5B**, không phải data hỏng.

## Hệ quả ưu tiên (cập nhật lần 2)
- A4 **xong**: thước đo negation trung thực = LoRA **81.5%** vs base **88.2%** (gap **6.7đ**). Dùng con số này thay cho 74.8/84.9 ở mọi báo cáo về sau.
- Vì hồi quy là thật và gồm inversion: **A1-A2 (hard-negative chống đảo/giữ load_bearing) lấy lại chút giá trị** — nhưng nhắm hẹp vào *inversion* và *param/ràng buộc*, không phải oversample tràn lan.
- Củng cố **F (model lớn hơn)**: 1.5B còn đảo `"không bật"`→`"bật"`, dấu hiệu giới hạn dung lượng/suy luận phủ định.
