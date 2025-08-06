

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=602112  
export MASTER_PORT=33651


swift rlhf \
    --rlhf_type grpo \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_retrieval_f1 external_retrieval_sim external_retrieval_format \
    --reward_weights 1 1 0.3 \
    --model Cold_start_model \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --dataset 'MLDR/MLDR_train_GRPO.json' \
    --dataset_shuffle false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --vllm_limit_mm_per_prompt '{"image": 8}' \
    --torch_dtype bfloat16 \
    --system examples/train/grpo/prompt.txt \
    --max_length 15536 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-6 \
    --save_total_limit 200 \
    --eval_steps 2000 \
    --save_steps 100 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --beta 0.02 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --sleep_level 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --gc_collect_after_offload true \
    --log_completions true \
    --report_to wandb \
    --deepspeed zero2 
