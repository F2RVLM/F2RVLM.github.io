import json
import re
import sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse

# 你的原始辅助函数
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

def extract_query_from_messages(messages):
    """从messages列表中找到role=user的content，提取query"""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"<\|query_start\|>(.*?)<\|query_end\|>", content, re.S)
            if match:
                return match.group(1).strip()
    return ""

# 按难度分组样本
def group_samples_by_difficulty(samples):
    groups = defaultdict(list)
    for s in samples:
        difficulty = s.get("difficulty", "unknown")
        groups[difficulty].append(s)
    return groups

# 按难度对easy类抽10%，其他类全部保留
def select_samples(groups):
    selected = []
    for diff, items in groups.items():
        if diff == "middle":
            n = max(1, int(len(items) * 0.1))
            # 按F1排序，假设样本里有"f1"字段，如果没有可以按其他标准排序，这里示例按query字母序
            items_sorted = sorted(items, key=lambda x: x.get("avg_f1", 0), reverse=True)
            selected.extend(items_sorted[:n])
        elif diff == "confusing":
            n = max(1, int(len(items) * 0.5))
            # 按F1排序，假设样本里有"f1"字段，如果没有可以按其他标准排序，这里示例按query字母序
            items_sorted = sorted(items, key=lambda x: x.get("avg_f1", 0), reverse=True)
            selected.extend(items_sorted[:n])
        else:
            selected.extend(items)
    return selected

# 加载带难度的样本（jsonl）
def load_samples_with_difficulty(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)
    return samples

# 读取真实数据，建立dialogue_id->list[样本]的映射
def load_real_data_grouped_by_dialogue_id(real_data_path):
    real_data_map = defaultdict(list)
    with open(real_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            did = data.get("dialogue_id", "")
            real_data_map[did].append(data)
    return real_data_map

# query匹配函数（字符串相等或包含）
def query_match(real_sample, query):
    messages = real_sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"<\|query_start\|>(.*?)<\|query_end\|>", content, re.S)
            if match:
                real_query = match.group(1).strip()
                if real_query == query or query in real_query or real_query in query:
                    return True
    return False

# 根据dialogue_id和query匹配真实样本
def match_samples_with_real_data(samples, real_data_map):
    matched_samples = []
    missed_count = 0
    for s in samples:
        did = s.get("dialogue_id", "")
        query = s.get("query", "")

        candidates = real_data_map.get(did, [])
        matched_real_sample = None
        for real_sample in candidates:
            if query_match(real_sample, query):
                matched_real_sample = real_sample
                break

        if matched_real_sample is not None:
            matched_samples.append({
                "sample_meta": s,
                "real_data": matched_real_sample
            })
        else:
            missed_count += 1
            print(f"未找到dialogue_id={did} 且 query匹配的真实数据，query={query}", file=sys.stderr)

    print(f"匹配失败样本数: {missed_count}/{len(samples)}", file=sys.stderr)
    return matched_samples

# 按难度排序，写出匹配的真实样本
def output_sorted_real_samples(matched_samples, output_path):
    # 先按难度排序
    difficulty_order = {"easy": 0, "middle": 1, "confusing": 2, "hard": 3}
    matched_samples.sort(key=lambda x: difficulty_order.get(x["sample_meta"].get("difficulty", "hard"), 3))

    with open(output_path, "w", encoding="utf-8") as fw:
        for item in matched_samples:
            real_data = item["real_data"]
            fw.write(json.dumps(real_data, ensure_ascii=False) + "\n")

def load_real_data_grouped_by_dialogue_id_from_json(real_data_path):
    real_data_map = defaultdict(list)
    with open(real_data_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)  # 一次性加载整个json
        for data in data_all:
            did = data.get("dialogue_id", "")
            if did:
                real_data_map[did].append(data)
    return real_data_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于难度分组匹配真实数据并排序输出')
    parser.add_argument('--samples', default = '', help='带难度的样本jsonl文件路径')
    parser.add_argument('--real_data', default='', help='真实数据jsonl文件路径')
    parser.add_argument('--output', default='', help='输出排序后的真实样本jsonl路径')
    args = parser.parse_args()

    print("加载带难度的样本...", file=sys.stderr)
    samples = load_samples_with_difficulty(args.samples)
    print(f"样本总数: {len(samples)}", file=sys.stderr)

    print("按难度分组...", file=sys.stderr)
    groups = group_samples_by_difficulty(samples)

    print("选择样本（easy抽10%，其余全部）...", file=sys.stderr)
    selected_samples = select_samples(groups)
    print(f"选中样本数: {len(selected_samples)}", file=sys.stderr)

    print("加载真实数据并建立索引...", file=sys.stderr)
    real_data_map = load_real_data_grouped_by_dialogue_id_from_json(args.real_data)

    print("开始匹配样本与真实数据...", file=sys.stderr)
    matched_samples = match_samples_with_real_data(selected_samples, real_data_map)

    print("写入排序后的真实样本...", file=sys.stderr)
    output_sorted_real_samples(matched_samples, args.output)

    print("完成。", file=sys.stderr)