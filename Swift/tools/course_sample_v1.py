import json
import re
import sys
from tqdm import tqdm
import numpy as np
import argparse

def extract_ids(field_str):
    utt_match = re.search(r"<\|utt_ids_start\|>(.*?)<\|utt_ids_end\|>", field_str)
    img_match = re.search(r"<\|img_ids_start\|>(.*?)<\|img_ids_end\|>", field_str)

    utt_ids = parse_id_list(utt_match.group(1).strip()) if utt_match else []
    img_ids = parse_id_list(img_match.group(1).strip()) if img_match else []

    return {"sentence_ids": utt_ids, "image_ids": img_ids}

def parse_id_list(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"警告: 无法解析ID列表: {json_str}，将使用空列表", file=sys.stderr)
        return []

def calculate_entropy(logprobs):
    probs = np.exp(np.array(logprobs))
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return entropy

def calculate_prf(gt, pred):
    gt_set = set(gt)
    pred_set = set(pred)

    if not gt_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)

    intersection = gt_set & pred_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def extract_logprobs_from_data(data):
    logprobs_info = data.get("logprobs", {})
    content = logprobs_info.get("content", [])
    logprobs = [token_info.get("logprob", 0.0) for token_info in content]
    return logprobs

def extract_query_from_messages(messages):
    """从messages列表中找到role=user的content，提取query"""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"<\|query_start\|>(.*?)<\|query_end\|>", content, re.S)
            if match:
                return match.group(1).strip()
    return ""  # 找不到就空串

def categorize_sample(f1, entropy, f1_q25, f1_q50, f1_q75, entropy_q25, entropy_q50, entropy_q75):
    if f1 >= f1_q75 and entropy <= entropy_q25:
        return "easy"
    elif f1 >= f1_q50 and entropy <= entropy_q50:
        return "middle"
    elif f1 >= f1_q25 and entropy <= entropy_q75:
        return "confusing"
    else:
        return "hard"

def filter_and_categorize_samples(input_path, output_path):
    samples = []
    f1_scores = []
    entropies = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取样本"):
            try:
                data = json.loads(line)

                # 提取dialogue_id
                dialogue_id = data.get("dialogue_id", "")

                # 提取query
                messages = data.get("messages", [])
                query = extract_query_from_messages(messages)

                gt = extract_ids(data['labels'])
                pred = extract_ids(data['response'])

                _, _, f1_utt = calculate_prf(gt.get("sentence_ids", []), pred.get("sentence_ids", []))
                _, _, f1_img = calculate_prf(gt.get("image_ids", []), pred.get("image_ids", []))
                avg_f1 = (f1_utt + f1_img) / 2

                logprobs = extract_logprobs_from_data(data)
                entropy = calculate_entropy(logprobs) if logprobs else 10.0

                samples.append((data, avg_f1, entropy, dialogue_id, query))
                f1_scores.append(avg_f1)
                entropies.append(entropy)

            except Exception as e:
                print(f"警告: 处理失败: {e}", file=sys.stderr)
                continue

    f1_q25, f1_q50, f1_q75 = np.percentile(f1_scores, [25, 50, 75])
    entropy_q25, entropy_q50, entropy_q75 = np.percentile(entropies, [25, 50, 75])

    print(f"F1分位数：25%={f1_q25:.4f}, 50%={f1_q50:.4f}, 75%={f1_q75:.4f}")
    print(f"熵分位数：25%={entropy_q25:.4f}, 50%={entropy_q50:.4f}, 75%={entropy_q75:.4f}")

    counts = {"easy": 0, "middle": 0, "confusing": 0, "hard": 0}

    with open(output_path, 'w', encoding='utf-8') as fw:
        for data, f1_val, entropy_val, dialogue_id, query in tqdm(samples, desc="分类样本并写入"):
            level = categorize_sample(f1_val, entropy_val, f1_q25, f1_q50, f1_q75, entropy_q25, entropy_q50, entropy_q75)
            
            # 跳过 easy 且 labels 为空的样本
            if level == "easy":
                label_info = extract_ids(data.get("labels", ""))
                if not label_info["sentence_ids"] and not label_info["image_ids"]:
                    continue  # 不写入该样本

            if level == "middle":
                label_info = extract_ids(data.get("labels", ""))
                if not label_info["sentence_ids"] and not label_info["image_ids"]:
                    continue  # 不写入该样本            
            
            data["difficulty"] = level
            data["dialogue_id"] = dialogue_id
            data["query"] = query
            data["avg_f1"] = f1_val
            counts[level] += 1
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("样本难度分类统计:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于F1和logprob熵分位数划分难度')
    parser.add_argument('--input', default = '', help='输入jsonl文件路径')
    parser.add_argument('--output', default='', help='输出带难度标签的jsonl文件路径')
    args = parser.parse_args()

    filter_and_categorize_samples(args.input, args.output)