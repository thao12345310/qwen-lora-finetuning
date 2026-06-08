# Sprint Report — Fine-tune & Host LLM (bản rút gọn)

**Người thực hiện:** Dương Phương Thảo · **Thời gian:** 2 tuần (chốt 06/06/2026)
**Dự án:** Fine-tune Qwen2.5-1.5B viết lại câu lệnh hội thoại tiếng Việt cho trợ lý trong xe

---

## Tổng quan
Dạy mô hình nhỏ (**Qwen2.5-1.5B**) đọc hội thoại nhiều lượt rồi **viết lại thành một câu lệnh
ngắn gọn, đủ ý**. Đây là **lần đầu mình làm trọn vòng đời fine-tune**: chuẩn bị dữ liệu →
huấn luyện → host vLLM → đo lường → cải thiện.

## Mình đã làm gì
- **Dữ liệu:** sinh **20.000 mẫu** bằng template, chia **train 16.001 / valid 2.000 /
  test 1.999**; vá thêm 1.000 mẫu thành train_v2 (17.001). Dùng Frontier Models (ChatGPT, Gemini, Claude, Grok trên browser) tạo riêng một
  **benchmark khó 1.044 mẫu** để chấm điểm (8 dạng dễ sai: đại từ, ngữ cảnh nhiễu, slot rải
  nhiều lượt, phủ định…) — bộ này tách biệt, không cho mô hình thấy lúc train.
- **Fine-tune:** dùng **LoRA/QLoRA (4-bit)** — như dán một "miếng vá" ~50MB lên mô hình gốc,
  chạy được trên 1 GPU T4 16GB miễn phí của Kaggle.
- **Host & demo:** dựng **vLLM** (API chuẩn OpenAI) + **ngrok** để truy cập từ xa; làm UI
  **Gradio** bấm-là-chạy, nạp sẵn mẫu test thật.
- **Đo lường:** pipeline benchmark dùng **LLM-judge** chấm 4 tiêu chí (ý định, đủ slot,
  không bịa, giữ phủ định); tách rời bước predict/judge và giữ một giám khảo cố định.
- **Cải thiện:** lặp chu trình **đo → tìm điểm yếu → vá đúng chỗ → train lại → đo lại** (2 vòng).

## Kết quả chính

| Chỉ số (trên benchmark 1.044 mẫu) | Base | LoRA v1.0 | LoRA v2.0 |
|---|---:|---:|---:|
| **Command Accuracy** (chính) | 52.8% | 65.9% | 65.6% |
| Intent Accuracy | 81.0% | 88.5% | 88.7% |
| Slot Completeness | 58.7% | 70.9% | 71.1% |
| Multi-turn slot (điểm yếu nhất) | 13.5% | 52.8% | **57.1%** |
| Phủ định (đã hiệu chỉnh) | 88.2% | 81.5% | 81.5% |

- Fine-tune giúp hơn base **~+13 điểm** ở chỉ số chính; cách biệt lớn nhất ở **multi-turn slot**
  (13.5% → 52.8%).

**Đã cải thiện gì sau khi vá dữ liệu** (LoRA v1.0 → v2.0, thêm ~1.000 mẫu):

| Dạng câu khó | Trước vá | Sau vá | Thay đổi |
|---|---:|---:|---:|
| **multi_turn_slot** (mục tiêu) | 52.8% | 57.1% | **+4.3** ✓ |
| **implicit_reference** (hưởng lợi) | 65.9% | 69.0% | **+3.1** ✓ |
| Command Accuracy tổng | 65.9% | 65.6% | −0.3 (phẳng) |

- Bản vá **đánh trúng đích** mà tổng thể không hỏng.
- Mô hình 1.5B hay **"nén" quá tay** → rớt mệnh đề phủ định; xác nhận đây là **lỗi thật
  (~−6.7đ), không phải lỗi đo**.

## Mình học được gì
- **Kỹ năng mới (lần đầu làm):** fine-tune model trên **Kaggle**; fine-tune bằng **LoRA với
  LLaMA Factory**; **research để chọn model phù hợp** cho tác vụ; và **research các metric**
  để biết đo cái gì cho đúng.
- **Prompt phải thật chi tiết:** viết prompt cho **judge model** và **base model** càng rõ
  ràng, càng nhiều **few-shot** thì kết quả càng chuẩn — đây là đòn bẩy lớn mà ít tốn công.
- **Mượn sức model lớn để nâng model nhỏ:** dùng **model mạnh tạo few-shot/dữ liệu chất lượng**
  rồi cho **model nhỏ học và làm theo** — cách hiệu quả để model nhỏ làm tốt hơn nhiều.
- **Đo lường đúng quan trọng ngang việc train** 
- **Chẩn đoán trước khi vá:** xác nhận dữ liệu sạch trước, nên biết lỗi nằm ở *hành vi mô hình*.
- **Vá hẹp, đúng chỗ thắng nhồi data bừa** — chất lượng hơn số lượng.
- **Hạ tầng cũng là việc:** nâng context 2048→8192, tách predict/judge, cho resume khi timeout,
  giữ giám khảo cố định để so sánh công bằng.
- **Mô hình nhỏ có "trần" năng lực** — đôi khi giải pháp là mô hình lớn hơn, không phải thêm data.

## Bước tiếp theo
Thêm **patch B4** dạy mô hình giữ mệnh đề phủ định + chi tiết quan trọng, train lại; nếu phủ
định vẫn kẹt thì cân nhắc nâng lên **Qwen2.5-3B**.
