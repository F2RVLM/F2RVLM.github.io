# please use python=3.10, cuda12.*
# sh requirements/install_all.sh
pip install vllm==0.8.5post1 -U
pip install lmdeploy==0.8 -U --no-deps
pip install transformers==4.51.3 -U
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes -U
# pip install git+https://github.com/modelscope/ms-swift.git
pip install timm -U
pip install deepspeed -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
