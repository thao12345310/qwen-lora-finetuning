Dưới đây là **blueprint dự án** cho bài toán của bạn. Có thể dùng làm định hướng triển khai, viết README, hoặc đưa vào báo cáo/project proposal.

# Blueprint: Vietnamese Contextual Query Rewriter for Virtual Assistant

## 1. Tên dự án

**Vietnamese Contextual Query Rewriter for Multi-turn Virtual Assistant Conversations**

Tên ngắn gọn hơn:

```text
Vietnamese Dialogue Rewriter
```

Hoặc nếu muốn gắn với domain xe/thiết bị thông minh:

```text
In-car Virtual Assistant Query Rewriter
```

---

## 2. Mục tiêu dự án

Xây dựng một model nhỏ có khả năng **rewrite câu nói cuối của người dùng trong hội thoại nhiều lượt** thành một câu yêu cầu độc lập, đầy đủ ngữ cảnh và dễ xử lý bởi hệ thống phía sau.

Ví dụ:

```text
user: hi vinfast
bot: chào bạn
user: nhiệt độ hôm nay là bao nhiêu?
bot: 40 độ
user: bật điều hoà 27 độ
```

Output mong muốn:

```text
Tôi muốn bật điều hoà 27 độ.
```

Một ví dụ cần ngữ cảnh hơn:

```text
user: bật điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ
```

Output:

```text
Tôi muốn bật điều hoà ở 27 độ.
```

---

## 3. Bài toán cần giải quyết

Trong chatbot hoặc trợ lý ảo, câu nói của người dùng thường không đầy đủ nếu chỉ nhìn riêng câu cuối.

Ví dụ:

```text
user: tìm trạm sạc gần nhất
bot: Trạm sạc VinFast Times City cách bạn 2km
user: dẫn đường tới đó
```

Nếu chỉ nhìn câu cuối:

```text
dẫn đường tới đó
```

Hệ thống phía sau sẽ không biết “đó” là đâu.

Sau khi rewrite:

```text
Tôi muốn điều hướng đến trạm sạc VinFast Times City.
```

Câu này có thể được đưa tiếp vào:

```text
Intent Classification
Slot Extraction
Tool Calling
Action Execution
```

---

## 4. Phạm vi dự án

### In scope

Dự án sẽ tập trung vào:

```text
1. Rewrite câu nói cuối của user thành yêu cầu độc lập
2. Xử lý hội thoại tiếng Việt
3. Xử lý ngữ cảnh nhiều lượt
4. Giữ nguyên intent chính của user
5. Bổ sung slot bị thiếu từ ngữ cảnh
6. Không thêm thông tin không có trong hội thoại
7. So sánh base model và fine-tuned model
8. Serve model qua API demo
```

### Out of scope

Dự án không tập trung vào:

```text
1. Xây dựng chatbot hoàn chỉnh
2. Tự động thực thi hành động thật như bật điều hòa
3. Xây hệ thống ASR/TTS
4. Huấn luyện model từ đầu
5. Full fine-tune model lớn
```

---

## 5. Đầu vào và đầu ra

### Input

Một đoạn hội thoại nhiều lượt:

```text
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 26 độ
```

### Output

Một câu rewrite độc lập:

```text
Tôi muốn bật điều hoà ở 26 độ.
```

---

## 6. Kiến trúc tổng thể

```text
Conversation History
        ↓
Preprocessing
        ↓
Fine-tuned Qwen Rewriter
        ↓
Standalone User Request
        ↓
Evaluation / Intent-Slot Extraction
        ↓
API / Demo UI
```

Chi tiết hơn:

```text
Raw Dialogue
    ↓
Format conversation
    ↓
Qwen2.5-1.5B-Instruct + LoRA/QLoRA
    ↓
Rewritten Query
    ↓
Post-processing
    ↓
Intent & Slot Evaluation
```

---

## 7. Model sử dụng

### Base model đề xuất

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Lý do chọn:

```text
1. Model nhỏ, phù hợp GPU T4 16GB
2. Hỗ trợ tiếng Việt tương đối tốt
3. Dễ fine-tune bằng LLaMA Factory
4. Phù hợp task output ngắn
5. Có thể serve được trên máy cấu hình vừa phải
```

### Phương pháp fine-tune

```text
QLoRA 4-bit
```

Lý do:

```text
1. Tiết kiệm VRAM
2. Phù hợp GPU cloud như Kaggle/Colab
3. Không cần full fine-tune
4. Dễ thử nghiệm nhiều phiên bản
```

---

## 8. Dataset

### Format dữ liệu

Nên dùng dạng `messages` để phù hợp instruction tuning.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Bạn là model rewrite hội thoại. Nhiệm vụ của bạn là biến câu nói cuối của user thành một yêu cầu độc lập, rõ ràng, giữ nguyên ý định, không thêm thông tin không chắc chắn. Chỉ trả về câu rewrite."
    },
    {
      "role": "user",
      "content": "user: mở điều hoà\nbot: bạn muốn đặt bao nhiêu độ?\nuser: 27 độ"
    },
    {
      "role": "assistant",
      "content": "Tôi muốn bật điều hoà ở 27 độ."
    }
  ]
}
```

---

## 9. Các nhóm dữ liệu cần có

Dataset nên chia thành nhiều nhóm tình huống.

### Nhóm 1: Câu cuối đã đầy đủ

```text
Input:
user: bật điều hoà 27 độ

Output:
Tôi muốn bật điều hoà 27 độ.
```

Mục tiêu: model không rewrite quá đà.

---

### Nhóm 2: Câu cuối thiếu intent

```text
Input:
user: bật điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 26 độ

Output:
Tôi muốn bật điều hoà ở 26 độ.
```

Mục tiêu: lấy intent từ lượt trước, slot từ câu cuối.

---

### Nhóm 3: Câu cuối dùng đại từ

```text
Input:
user: tìm trạm sạc gần nhất
bot: Trạm sạc VinFast Times City cách bạn 2km
user: dẫn đường tới đó

Output:
Tôi muốn điều hướng đến trạm sạc VinFast Times City.
```

Mục tiêu: resolve “đó”, “nó”, “người đó”, “bài đó”.

---

### Nhóm 4: Câu xác nhận

```text
Input:
user: gọi cho mẹ
bot: Bạn muốn gọi cho Mẹ đúng không?
user: đúng rồi

Output:
Tôi muốn gọi cho Mẹ.
```

Mục tiêu: hiểu câu xác nhận dựa trên bot message.

---

### Nhóm 5: Câu phủ định / hủy bỏ

```text
Input:
user: bật điều hoà 27 độ
bot: Tôi sẽ bật điều hoà 27 độ
user: thôi hủy đi

Output:
Tôi muốn hủy yêu cầu bật điều hoà 27 độ.
```

Mục tiêu: xử lý cancel intent.

---

### Nhóm 6: Thay đổi thông số

```text
Input:
user: bật điều hoà 27 độ
bot: Đã bật điều hoà 27 độ
user: giảm thêm 2 độ

Output:
Tôi muốn giảm nhiệt độ điều hoà thêm 2 độ.
```

Mục tiêu: hiểu thao tác chỉnh sửa sau hành động trước đó.

---

### Nhóm 7: Ngữ cảnh không liên quan

```text
Input:
user: thời tiết hôm nay thế nào?
bot: Hôm nay khoảng 40 độ
user: bật điều hoà 27 độ

Output:
Tôi muốn bật điều hoà 27 độ.
```

Mục tiêu: không đưa thông tin “40 độ” vào output nếu không cần.

---

## 10. Domain nên bao phủ

Vì đây là virtual assistant, dataset nên có nhiều domain nhỏ.

```text
1. Điều hòa
2. Âm lượng
3. Âm nhạc
4. Điều hướng
5. Gọi điện
6. Tin nhắn
7. Trạm sạc
8. Thời tiết
9. Lịch trình
10. Đèn / cửa / thiết bị xe
11. Hủy yêu cầu
12. Xác nhận yêu cầu
```

Ví dụ intent:

```text
turn_on_ac
turn_off_ac
set_ac_temperature
increase_temperature
decrease_temperature
play_music
navigate_to_location
call_contact
send_message
find_charging_station
ask_weather
cancel_action
confirm_action
```

---

## 11. Quy mô dataset

### Giai đoạn prototype

```text
Train: 500 samples
Validation: 100 samples
Test: 100 samples
```

Mục tiêu: kiểm tra pipeline chạy được.

### Giai đoạn demo tốt

```text
Train: 3,000 samples
Validation: 300 samples
Test: 500 samples
```

Mục tiêu: thấy khác biệt rõ giữa base model và fine-tuned model.

### Giai đoạn nghiêm túc hơn

```text
Train: 10,000 samples
Validation: 1,000 samples
Test: 1,000 samples
```

Mục tiêu: đánh giá ổn định hơn, đủ tốt để đưa vào portfolio/CV.

---

## 12. Cách tạo dataset

### Nguồn 1: Template-based synthetic data

Tạo các template hội thoại theo intent.

Ví dụ:

```text
user: {intent phrase}
bot: {clarification question}
user: {slot value}
```

Sinh ra output:

```text
Tôi muốn {intent} với {slot value}.
```

Ví dụ:

```text
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ
```

Output:

```text
Tôi muốn bật điều hoà ở 27 độ.
```

---

### Nguồn 2: Paraphrase tiếng Việt

Tạo nhiều cách nói tự nhiên:

```text
bật điều hoà
mở điều hoà
cho điều hoà chạy đi
nóng quá bật điều hoà đi
bật máy lạnh
mở máy lạnh
```

Slot temperature:

```text
27 độ
để 27
mức 27
cho 27 độ
tầm 27 độ
đặt 27 độ C
```

---

### Nguồn 3: Hard negative examples

Thêm các case dễ gây nhầm:

```text
user: nhiệt độ ngoài trời là bao nhiêu?
bot: 40 độ
user: bật điều hoà 27 độ
```

Output không được chứa 40 độ.

```text
Tôi muốn bật điều hoà 27 độ.
```

---

### Nguồn 4: Human-reviewed test set

Test set nên được kiểm tra thủ công.

Không nên để toàn bộ test set là synthetic tự động, vì model có thể học pattern quá dễ.

---

## 13. Training pipeline

```text
1. Collect raw dialogue templates
2. Generate synthetic samples
3. Normalize format
4. Split train/validation/test
5. Fine-tune Qwen with QLoRA
6. Save LoRA adapter
7. Merge adapter hoặc serve adapter
8. Run evaluation
9. Compare base vs fine-tuned model
10. Build API demo
```

---

## 14. Config fine-tune đề xuất

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4

template: qwen
dataset: dialogue_rewrite_vi
cutoff_len: 1024
max_samples: 3000
overwrite_cache: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1

lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: all

fp16: true
output_dir: outputs/qwen-dialogue-rewriter-lora
logging_steps: 10
save_steps: 200
plot_loss: true
```

Nếu bị OOM:

```text
1. Giảm cutoff_len từ 1024 xuống 512
2. Giảm batch_size về 1
3. Giữ gradient_accumulation_steps = 8 hoặc 16
4. Dùng quantization_bit = 4
5. Giảm lora_rank từ 16 xuống 8
```

---

## 15. Tài nguyên cần chuẩn bị

### Cấu hình tối thiểu

```text
GPU: 8GB VRAM
RAM: 16GB
Storage: 30GB
Method: QLoRA
Dataset: <= 3,000 samples
```

### Cấu hình khuyên dùng

```text
GPU: T4 16GB / RTX 3060 12GB / L4 24GB
RAM: 32GB
Storage: 50GB
Method: QLoRA 4-bit
Dataset: 3,000–10,000 samples
```

---

## 16. Evaluation

Không nên chỉ dùng exact match, vì nhiều câu rewrite khác nhau vẫn đúng.

Ví dụ các output này đều đúng:

```text
Tôi muốn bật điều hoà 27 độ.
Tôi muốn bật điều hoà ở mức 27 độ.
Đặt điều hoà ở 27 độ giúp tôi.
```

### Metric nên dùng

```text
1. Intent Accuracy
2. Slot Accuracy
3. Context Resolution Accuracy
4. Hallucination Rate
5. Rewrite Fluency
```

### Ví dụ đánh giá

Input:

```text
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ
```

Gold output:

```text
Tôi muốn bật điều hoà ở 27 độ.
```

Prediction:

```text
Tôi muốn bật điều hoà 27 độ.
```

Đánh giá:

```json
{
  "intent_correct": true,
  "slot_correct": true,
  "context_correct": true,
  "hallucination": false,
  "pass": true
}
```

---

## 17. Baseline cần so sánh

Nên so sánh ít nhất 3 phiên bản:

```text
1. Rule-based baseline
2. Base Qwen2.5-1.5B-Instruct
3. Fine-tuned Qwen2.5-1.5B-Instruct
```

### Rule-based baseline

Ví dụ rule đơn giản:

```text
Nếu câu cuối chỉ chứa nhiệt độ và context trước có điều hòa
→ rewrite thành: Tôi muốn bật điều hoà ở {temperature}.
```

Mục tiêu không phải để thắng fine-tuned model, mà để có baseline rõ ràng.

---

## 18. API demo

Có thể build bằng FastAPI.

### Endpoint đề xuất

```text
POST /rewrite
```

Request:

```json
{
  "conversation": [
    {"role": "user", "content": "mở điều hoà"},
    {"role": "bot", "content": "bạn muốn đặt bao nhiêu độ?"},
    {"role": "user", "content": "27 độ"}
  ]
}
```

Response:

```json
{
  "rewritten_query": "Tôi muốn bật điều hoà ở 27 độ."
}
```

Endpoint đánh giá thử:

```text
POST /compare
```

Response:

```json
{
  "base_model_output": "27 độ",
  "fine_tuned_output": "Tôi muốn bật điều hoà ở 27 độ."
}
```

---

## 19. Demo UI

UI có thể đơn giản:

```text
Left panel:
- Conversation input

Right panel:
- Base model output
- Fine-tuned model output
- Intent
- Slots
- Pass/Fail evaluation
```

Ví dụ hiển thị:

```text
Conversation:
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ

Base model:
27 độ

Fine-tuned model:
Tôi muốn bật điều hoà ở 27 độ.

Intent:
set_ac_temperature

Slots:
temperature = 27

Result:
PASS
```

---

## 20. Cấu trúc thư mục project

```text
dialogue-rewriter/
├── README.md
├── requirements.txt
├── configs/
│   ├── qwen_lora_sft.yaml
│   └── inference.yaml
├── data/
│   ├── raw/
│   │   └── templates.json
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   └── samples/
│       └── demo_cases.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_error_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── generate_dataset.py
│   │   ├── preprocess.py
│   │   └── split_data.py
│   ├── train/
│   │   └── train_llamafactory.sh
│   ├── inference/
│   │   ├── predict.py
│   │   └── compare.py
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── error_analysis.py
│   └── api/
│       ├── main.py
│       └── schemas.py
├── outputs/
│   ├── qwen-dialogue-rewriter-lora/
│   └── eval_results/
├── demo/
│   └── app.py
└── docs/
    ├── project_blueprint.md
    ├── dataset_card.md
    └── experiment_report.md
```

---

## 21. Milestones triển khai

### Milestone 1: Problem setup

Mục tiêu:

```text
1. Định nghĩa task
2. Định nghĩa input/output
3. Chọn model
4. Chọn metric
```

Deliverables:

```text
README.md
project_blueprint.md
sample input/output
```

---

### Milestone 2: Dataset v1

Mục tiêu:

```text
1. Tạo 500–1,000 samples đầu tiên
2. Chia train/valid/test
3. Kiểm tra format cho LLaMA Factory
```

Deliverables:

```text
train.jsonl
valid.jsonl
test.jsonl
dataset_card.md
```

---

### Milestone 3: Baseline evaluation

Mục tiêu:

```text
1. Chạy base Qwen chưa fine-tune
2. Lưu output
3. Đánh giá lỗi thường gặp
```

Deliverables:

```text
base_model_predictions.jsonl
baseline_eval_report.md
```

---

### Milestone 4: Fine-tune v1

Mục tiêu:

```text
1. Fine-tune Qwen bằng QLoRA
2. Lưu LoRA adapter
3. Plot training loss
4. Chạy inference thử
```

Deliverables:

```text
LoRA adapter
training logs
loss curve
sample predictions
```

---

### Milestone 5: Evaluation v1

Mục tiêu:

```text
1. So sánh base model vs fine-tuned model
2. Đánh giá intent accuracy
3. Đánh giá slot accuracy
4. Đánh giá hallucination
```

Deliverables:

```text
eval_results.json
error_analysis.md
comparison_table.md
```

---

### Milestone 6: Improve dataset

Mục tiêu:

```text
1. Tìm lỗi model hay gặp
2. Thêm hard cases
3. Fine-tune version 2
```

Deliverables:

```text
dataset_v2
model_v2
eval_report_v2
```

---

### Milestone 7: Serve model

Mục tiêu:

```text
1. Tạo API /rewrite
2. Tạo API /compare
3. Test latency
4. Viết hướng dẫn chạy local
```

Deliverables:

```text
FastAPI server
API docs
latency report
```

---

### Milestone 8: Demo và báo cáo

Mục tiêu:

```text
1. Làm demo UI đơn giản
2. Viết README hoàn chỉnh
3. Viết báo cáo before/after fine-tune
4. Chuẩn bị video demo nếu cần
```

Deliverables:

```text
Demo app
Final README
Experiment report
Demo screenshots
```

---

## 22. Tiêu chí thành công

Project được coi là thành công nếu:

```text
1. Fine-tuned model rewrite tốt hơn base model rõ ràng
2. Intent accuracy đạt khoảng 85–90% trên test set nhỏ
3. Slot accuracy đạt khoảng 85–90%
4. Hallucination rate thấp
5. Có demo API hoặc UI chạy được
6. Có báo cáo lỗi và so sánh before/after
```

Với project học tập, chỉ cần đạt:

```text
Base model thường trả lời mơ hồ
Fine-tuned model trả về câu rewrite rõ ràng hơn
```

là đã chứng minh fine-tune có tác dụng.

---

## 23. Rủi ro và cách xử lý

| Rủi ro                                        | Cách xử lý                                      |
| --------------------------------------------- | ----------------------------------------------- |
| Model thêm thông tin không có trong hội thoại | Thêm negative samples và system prompt chặt hơn |
| Model chỉ copy câu cuối                       | Thêm nhiều case câu cuối thiếu context          |
| Model rewrite quá dài                         | Giới hạn output: chỉ 1 câu                      |
| Dataset synthetic quá máy móc                 | Thêm paraphrase tự nhiên                        |
| Eval khó vì output đa dạng                    | Dùng intent/slot accuracy thay vì exact match   |
| GPU OOM                                       | Giảm cutoff_len, batch size, LoRA rank          |

---

## 24. README nên mô tả project như sau

```text
This project fine-tunes a small Qwen language model for Vietnamese contextual query rewriting in multi-turn virtual assistant conversations.

Given a dialogue history, the model rewrites the latest user utterance into a standalone, context-aware request. This rewritten query can then be used by downstream NLU modules such as intent classification, slot extraction, or tool calling.

The project includes dataset generation, QLoRA fine-tuning, baseline comparison, evaluation by intent/slot correctness, and API serving.
```

Bản tiếng Việt:

```text
Dự án này fine-tune một mô hình Qwen nhỏ cho bài toán rewrite câu truy vấn tiếng Việt trong hội thoại nhiều lượt với trợ lý ảo.

Đầu vào là lịch sử hội thoại giữa người dùng và chatbot. Đầu ra là câu yêu cầu cuối cùng của người dùng đã được viết lại thành một câu độc lập, đầy đủ ngữ cảnh và dễ xử lý bởi các module phía sau như intent classification, slot extraction hoặc tool calling.
```

---

## 25. Công nghệ sử dụng

```text
Model: Qwen2.5-1.5B-Instruct
Fine-tuning: LoRA / QLoRA
Framework: LLaMA Factory
Training environment: Kaggle GPU / Google Colab / Local GPU
Serving: FastAPI hoặc vLLM
Evaluation: Python scripts
Demo UI: Streamlit hoặc React
Data format: JSONL
```

---

## 26. Phiên bản MVP nên làm trước

MVP không cần quá phức tạp. Làm theo thứ tự này là đủ:

```text
1. Tạo 500 samples về điều hòa, nhạc, điều hướng, gọi điện
2. Chạy base Qwen trên 100 test cases
3. Fine-tune Qwen bằng QLoRA
4. Chạy lại 100 test cases
5. Làm bảng so sánh before/after
6. Tạo FastAPI endpoint /rewrite
```

MVP output cần chứng minh được:

```text
Input:
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ

Base model:
27 độ

Fine-tuned model:
Tôi muốn bật điều hoà ở 27 độ.
```

Đây là điểm ăn tiền nhất của project.
