export MAX_PIXELS=602112
export MASTER_PORT=33647

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
  --model F2RVLM_Qwen2_VL_7B_Checkpoint \
  --val_dataset MLDR/MLDR_val.json \
  --infer_backend pt \
  --temperature 0 \
  --max_new_tokens 15536 \
  --max_batch_size 1 