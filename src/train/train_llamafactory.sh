#!/usr/bin/env bash
# Train the rewrite LoRA via LLaMA Factory CLI.
# Requires: pip install llamafactory[torch,bitsandbytes]
#
# Pipeline:
#   1. split the generated jsonl → data/processed/{train,valid,test}.jsonl
#      (stratified by domain × context_required)
#   2. llamafactory-cli reads those via data/processed/dataset_info.json
#
# Override the input file with arg 1 or the DATA_INPUT env var, e.g.:
#   ./src/train/train_llamafactory.sh data/train/dialogues_train_merged.jsonl
set -euo pipefail

# Resolve repo root so the script works from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Generated training file (Phase-2 vLLM output by default).
DATA_INPUT="${1:-${DATA_INPUT:-data/train/dialogues_train_vllm.jsonl}}"
DATASET_DIR="data/processed"

if [[ ! -f "$DATA_INPUT" ]]; then
  echo "ERROR: input not found: $DATA_INPUT" >&2
  echo "Pass the generated jsonl as arg 1 or set DATA_INPUT." >&2
  exit 1
fi

echo ">> Splitting $DATA_INPUT -> $DATASET_DIR/{train,valid,test}.jsonl"
python -m src.data.split_data \
  --input "$DATA_INPUT" \
  --output_dir "$DATASET_DIR"

echo ">> Training"
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --quantization_bit 4 \
  --template qwen \
  --dataset_dir "$DATASET_DIR" \
  --dataset train \
  --eval_dataset valid \
  --cutoff_len 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fp16 \
  --output_dir outputs/qwen-dialogue-rewriter-lora \
  --logging_steps 10 \
  --save_steps 200 \
  --plot_loss
