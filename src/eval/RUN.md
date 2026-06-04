# Cách chạy benchmark eval

Quy tắc vàng: **PREDICT và JUDGE là 2 bước riêng.** Predict chạy trên Kaggle (nơi vLLM
sống) vì tunnel ngrok hay rớt → chạy chung sẽ ra `__ERROR__` hàng loạt và mất hết. Judge
chạy ở máy local, lặp lại tùy ý, không đụng vLLM.

Interpreter: dùng `/opt/homebrew/bin/python3.11` cho lệnh nền/non-interactive (alias
`python3` chỉ có trong shell tương tác).

---

## Bước 1 — PREDICT (chạy TRÊN Kaggle, lúc vLLM còn sống)

```bash
python -m src.eval.eval_bench --no-judge \
  --vllm-url http://localhost:8000 \
  --models vi-rewriter Qwen/Qwen2.5-1.5B-Instruct \
  --baseline-models Qwen/Qwen2.5-1.5B-Instruct \
  --baseline-fewshot-n 6 --baseline-fewshot-hard-file /nonexistent \
  --output-dir data/bench/eval_results/full_gptoss
```

- Lưu `preds_<model>.jsonl` (chỉ prediction; judge để placeholder `__JUDGE_ERROR__ pending`).
- **Không cần judge key.** Predict đúng 1 lần. Xong thì **tải `preds_*.jsonl` về máy**.
- `--baseline-fewshot-hard-file /nonexistent`: tắt 20 hard case vì chúng tràn `max_model_len=2048`
  của server. Nếu server nâng context ≥4096 thì mới bật lại hard.

## Bước 2 — JUDGE (chạy ở MÁY LOCAL, lặp lại được, KHÔNG cần vLLM)

```bash
JUDGE_API_KEY=ollama /opt/homebrew/bin/python3.11 -m src.eval.eval_bench --resume \
  --models vi-rewriter Qwen/Qwen2.5-1.5B-Instruct \
  --judge-model gpt-oss:120b-cloud --judge-base-url http://localhost:11434/v1 \
  --judge-concurrency 4 \
  --output-dir data/bench/eval_results/full_gptoss
```

- Chỉ chấm dòng pending/lỗi, giữ nguyên cái đã thành công, **không gọi vLLM** (khỏi truyền `--vllm-url`).
- Quota judge cạn / lỗi giữa chừng → **chạy lại y lệnh này**, nó bù tiếp. Predict không bao giờ chạy lại.
- Cần `ollama serve` đang chạy ở local (`gpt-oss:120b-cloud` đã pull). `JUDGE_API_KEY=ollama` chỉ là giá trị giả để qua guard.

Lưu ý: dùng **cùng `--bench`** (mặc định `data/bench/dialogues_bench_browser.jsonl`, 1044 mẫu) và
cùng `--limit` ở cả 2 bước — các dòng khớp nhau theo thứ tự.

---

## Judge đã validate

- **`gpt-oss:120b-cloud`** (Ollama Cloud): 10/11 probe, JSON sạch. Code tự set `reasoning_effort=low`
  + `max_tokens=512` cho gpt-oss (nếu không sẽ cụt JSON).
- **`gemini-2.5-flash`** (Google direct, `--judge-base-url https://generativelanguage.googleapis.com/v1beta/openai/`):
  11/11 probe; code tự tắt thinking. Nhưng **free tier RPD quá nhỏ** (~vài chục call/ngày) → chỉ hợp khi bật billing.
- **Đóng băng 1 judge** cho cả đợt so sánh. Không đổi judge giữa chừng (vi-rewriter & baseline phải cùng judge).

## Xem kết quả

```bash
# bảng so sánh + slice
/opt/homebrew/bin/python3.11 - <<'PY'
import json
from src.eval.eval_bench import aggregate, _slice
for m in ["vi-rewriter","Qwen_Qwen2.5-1.5B-Instruct"]:
    rows=[json.loads(l) for l in open(f"data/bench/eval_results/full_gptoss/preds_{m}.jsonl",encoding="utf-8")]
    a=aggregate(rows)
    print(m, "| command_acc", round(a["command_acc"]*100,1), "| halluc", round(a["halluc_rate"]*100,1),
          "| judge_err", round(a["judge_error_rate"]*100,1))
PY
```

`metrics_summary.json` trong `--output-dir` là bản máy-đọc-được cho plot/so sánh.
