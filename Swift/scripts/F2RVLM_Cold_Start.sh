
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=602112
export MASTER_PORT=33647

swift sft \
  --model Qwen2-VL-7B-Instruct \
  --dataset 'MLDR/MLDR_train_cold_start.json' \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_accumulation_steps 4 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 100 \
  --logging_steps 5 \
  --max_length 15536 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --deepspeed zero2 \
  --report_to wandb \
  --save_only_model true