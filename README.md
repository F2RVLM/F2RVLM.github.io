
# 📌 F²RVLM: Boosting Fine-grained Fragment Retrieval for Multi-Modal Long-form Dialogue with Vision Language Model

This repository provides the implementation of **F²RVLM**, a vision-language model designed to retrieve semantically coherent fragments (text and images) from multi-modal long-form dialogues. The model is built on top of the [ms-swift](https://github.com/modelscope/swift) framework and supports both **supervised fine-tuning (SFT)** and **reinforcement tuning (GRPO)**.

---

## 🔧 Environment Requirements

To ensure smooth training and inference of **F²RVLM**, please set up the following environment:

- **OS**: Ubuntu 22.04  
- **CUDA**: 12.4.0  
- **Python**: 3.11  
- **PyTorch**: 2.6.0  
- **vllm**: 0.8.5.post1  
- **modelscope**: 1.26.0

To install the project dependencies, use:

```bash
pip install -e Swift/
```

## 📁 Dataset Format

We provide two datasets for training and evaluation:

- **MLDR**: A synthetic multi-modal long-form dialogue dataset annotated for fine-grained fragment-level retrieval.
- **WeChat-MDR**: A real-world test set collected from 12 volunteers. It has **no affiliation with the WeChat platform**.

✅ Both datasets are pre-processed into **ms-swift-compatible format** and ready to be used in training and inference pipelines.


### 📦 Download Links

You can download the datasets using the following URLs:

- [MLDR Dataset](https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/MLDR.tar)
- [WeChat Test Set](https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/WeChat_test_set.tar)

```bash
# Example: Download and extract MLDR dataset
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/MLDR.tar
tar -xf MLDR.tar

# Example: Download and extract WeChat test set
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/WeChat_test_set.tar
tar -xf WeChat_test_set.tar
```

---

## 🏋️‍♂️ Model Training

We provide two training scripts under the `scripts/` directory:

### 🔹 1. Cold Start Supervised Fine-tuning (SFT)

```bash
bash F2RVLM/Swift/scripts/F2RVLM_Cold_Start.sh
```

### 🔹 2. GRPO-based Reinforcement Tuning

```bash
bash F2RVLM/Swift/scripts/F2RVLM_GRPO.sh
```

---

## 🔍 Inference

Run inference on the validation and test sets using:

```bash
bash F2RVLM/Swift/scripts/infer.sh
```

Supported evaluation sets:

- MLDR validation set
- WeChat-MDR test set

---

## 📊 Evaluation

We provide a script to compute F1 and MCC:

```bash
python F2RVLM/Swift/tools/eval_multi_modal_retrieval_F1.py
```

---

## 📦 Model Weights

The trained checkpoints for:

- **F²RVLM**

- (https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/F2RVLM_Qwen2_VL_7B_Checkpoint/)

```bash
wget https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/hanbobi/F2RVLM_Qwen2_VL_7B_Checkpoint/
```


---

## 📄 License

Please refer to the [LICENSE.md](./LICENSE.md) for dataset usage terms.

---
