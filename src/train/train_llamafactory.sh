#!/usr/bin/env bash
# Optional alternative: train via LLaMA Factory CLI.
# Requires: pip install llamafactory[torch,bitsandbytes]
# Edit configs/qwen_lora_sft.yaml first.
set -euo pipefail

llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --quantization_bit 4 \
  --template qwen \
  --dataset_dir data/processed \
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
