
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift deploy \
    --model /mnt/gemininjceph3/geminicephfs/wx-mm-spr-xxxx/hanbobi/pretrain_models/Kwai-Keye/Keye-VL-8B-Preview \
    --infer_backend vllm \
    --served_model_name Keye-VL-8B-Preview

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen2.5-7B-Instruct",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
