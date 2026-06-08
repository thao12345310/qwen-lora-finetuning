# Vietnamese Dialogue Rewriter

Fine-tune **Qwen2.5-1.5B-Instruct** with **QLoRA** to rewrite the last user
utterance in a multi-turn Vietnamese conversation into a standalone,
context-resolved request.

```text
user: mở điều hoà
bot: bạn muốn đặt bao nhiêu độ?
user: 27 độ
        ↓
Tôi muốn bật điều hoà ở 27 độ.
```

See `blue_print.md` for the full design doc.

## Layout

```
configs/                 # Training + inference YAML configs
data/raw/                # Generated dialogues + intent templates
data/processed/          # train/valid/test jsonl splits
src/data/                # Dataset generator, splitter
src/train/               # QLoRA training (python + LLaMA-Factory shell)
src/inference/           # predict.py (single), compare.py (test set)
src/api/                 # FastAPI /rewrite, /compare
outputs/                 # LoRA adapters + eval results
```

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

`bitsandbytes` is CUDA-only. On Apple Silicon, edit
`configs/qwen_lora_sft.yaml` and `configs/inference.yaml` and set
`load_in_4bit: false`.

### 2. Generate the dataset

```bash
python3 src/data/generate_dataset.py --target 500
python3 src/data/split_data.py
```

Produces `data/processed/{train,valid,test}.jsonl`. The generator covers all
seven scenario groups from the blueprint (complete utterance, missing intent,
pronoun resolution, confirmation, cancellation, parameter adjustment,
irrelevant context) across four domains (AC, music, navigation, calling).

For the current recipe-managed train mix:

```bash
python3 -m src.data.patch_b2b3
python3 -m src.data.patch_b4
python3 -m src.data.build_train_mix --recipe data/recipes/v3.yaml
```

The recipe writes `train.jsonl`, `valid.jsonl`, `test.jsonl`,
`dataset_info.json`, `dataset_manifest.json`, and `dataset_report.md` under its
`output_dir` (`data/processed` by default).

### 3. Fine-tune on T4 16GB (Kaggle / Colab)

```bash
python src/train/train.py --config configs/qwen_lora_sft.yaml
```

Adapter is saved to `outputs/qwen-dialogue-rewriter-lora/`. Defaults:
3 epochs, batch size 1 × 8 grad-accum, LoRA r=8, 4-bit NF4.

### 4. Compare base vs fine-tuned on the test set

```bash
python -m src.inference.compare
```

Writes `outputs/eval_results/comparison.jsonl` with one row per test case.

### 5. Single-shot inference

```bash
python -m src.inference.predict \
  --conversation '[{"role":"user","content":"mở điều hoà"},
                    {"role":"bot","content":"bạn muốn đặt bao nhiêu độ?"},
                    {"role":"user","content":"27 độ"}]'
```

### 6. Serve the API

```bash
uvicorn src.api.main:app --port 8000
```

```bash
curl -X POST localhost:8000/rewrite -H 'Content-Type: application/json' -d '{
  "conversation": [
    {"role": "user", "content": "mở điều hoà"},
    {"role": "bot", "content": "bạn muốn đặt bao nhiêu độ?"},
    {"role": "user", "content": "27 độ"}
  ]
}'
```

## Notes

- The rewrite system prompt is defined once in `src/data/prompts.py`. Generated
  Llama-Factory records embed that prompt in their `system` turn, and local
  inference imports the same constant when formatting API requests.
- `lru_cache` in `predict.py` keeps a loaded model in memory across calls; the
  first request after server start is slow (model load), subsequent ones are
  fast.
- For OOM, reduce `cutoff_len`, `lora_rank`, or `per_device_train_batch_size`
  in `configs/qwen_lora_sft.yaml`.
