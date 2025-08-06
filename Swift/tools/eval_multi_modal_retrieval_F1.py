import json
import re
import sys
from tqdm import tqdm
import numpy as np
import argparse

def extract_ids(field_str):
    """从指定字符串中提取句子ID和图像ID"""
    utt_match = re.search(r"<\|utt_ids_start\|>(.*?)<\|utt_ids_end\|>", field_str)
    img_match = re.search(r"<\|img_ids_start\|>(.*?)<\|img_ids_end\|>", field_str)
    
    utt_ids = parse_id_list(utt_match.group(1).strip()) if utt_match else []
    img_ids = parse_id_list(img_match.group(1).strip()) if img_match else []
    
    return {"sentence_ids": utt_ids, "image_ids": img_ids}

def parse_id_list(json_str):
    """安全解析ID列表"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"警告: 无法解析ID列表: {json_str}，将使用空列表", file=sys.stderr)
        return []

def extract_max_ids_from_message(message_content):
    """从用户消息中提取最大句子ID和图像ID"""
    # 匹配<|utt_id_start|>数字<|utt_id_end|>格式
    utt_ids = re.findall(r"<\|utt_id_start\|>(\d+)<\|utt_id_end\|>", message_content)
    # 匹配<|img_id_start|>数字<|img_id_end|>格式
    img_ids = re.findall(r"<\|img_id_start\|>(\d+)<\|img_id_end\|>", message_content)
    
    # 转换为整数并找到最大值，默认值为20
    max_utt_id = max(int(id) for id in utt_ids) if utt_ids else 20
    max_img_id = max(int(id) for id in img_ids) if img_ids else 20
    
    return max_utt_id, max_img_id

def load_and_prepare_samples(jsonl_path):
    """加载样本数据并转换为评估格式，同时提取max_id"""
    samples = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="加载样本"):
                try:
                    data = json.loads(line)
                    label_ids = extract_ids(data['labels'])
                    pred_ids = extract_ids(data['response'])
                    
                    # 从message中提取max_id
                    max_utt_id, max_img_id = 20, 20  # 默认值
                    for msg in data.get("messages", []):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            max_utt_id, max_img_id = extract_max_ids_from_message(content)
                            break
                    
                    samples.append({
                        "ground_truth": label_ids,
                        "model_prediction": pred_ids,
                        "max_utt_id": max_utt_id,
                        "max_img_id": max_img_id
                    })
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"警告: 解析样本失败: {e}，跳过此样本", file=sys.stderr)
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载文件时发生异常: {e}", file=sys.stderr)
        sys.exit(1)
    
    return samples

def calculate_mcc(gt, pred, max_id=20):
    """计算Matthews相关系数(MCC)"""
    gt_set = set(gt)
    pred_set = set(pred)
    all_ids = set(range(max_id + 1))  # 构建全集
    
    # 构建正负样本集合
    pred_positive = pred_set
    pred_negative = all_ids - pred_positive
    gt_positive = gt_set
    gt_negative = all_ids - gt_set
    
    # 计算混淆矩阵
    tp = len(pred_positive & gt_positive)
    fp = len(pred_positive & gt_negative)
    tn = len(pred_negative & gt_negative)
    fn = len(pred_negative & gt_positive)
    
    # 处理分母为0的情况
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0  # 所有样本为同一类时，MCC为0
    
    return (tp * tn - fp * fn) / denominator

def calculate_prf(gt, pred):
    """计算精确率、召回率和F1分数"""
    gt_set = set(gt)
    pred_set = set(pred)
    
    if not gt_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 0.0, 0.0)
    
    intersection = gt_set & pred_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1 

def evaluate_multimodal_retrieval(samples):
    """评估多模态检索结果，包含MCC指标"""
    sentence_precision, sentence_recall, sentence_f1, sentence_mcc = [], [], [], []
    image_precision, image_recall, image_f1, image_mcc = [], [], [], []
    joint_precision, joint_recall, joint_f1, joint_mcc = [], [], [], []

    for sample in tqdm(samples, desc="计算指标"):
        try: 
            gt = sample["ground_truth"]
            pred = sample["model_prediction"]
            max_utt_id = sample.get("max_utt_id", 20)
            max_img_id = sample.get("max_img_id", 20)

            # 句子检索指标
            gt_utt, pred_utt = gt.get("sentence_ids", []), pred.get("sentence_ids", [])
            sp, sr, sf1 = calculate_prf(gt_utt, pred_utt)
            smcc = calculate_mcc(gt_utt, pred_utt, max_utt_id)

            # 图像检索指标
            gt_img, pred_img = gt.get("image_ids", []), pred.get("image_ids", [])
            ip, ir, if1 = calculate_prf(gt_img, pred_img)
            imcc = calculate_mcc(gt_img, pred_img, max_img_id)

            # 联合指标（文本+图像）
            jp = 2 * sp * ip / (sp + ip) if (sp + ip) > 0 else 0
            jr = 2 * sr * ir / (sr + ir) if (sr + ir) > 0 else 0
            jf = (sf1 + if1) / 2
            jmcc = (smcc + imcc) / 2

            # 存储指标
            sentence_precision.append(sp)
            sentence_recall.append(sr)
            sentence_f1.append(sf1)
            sentence_mcc.append(smcc)

            image_precision.append(ip)
            image_recall.append(ir)
            image_f1.append(if1)
            image_mcc.append(imcc)
            
            joint_precision.append(jp)
            joint_recall.append(jr)
            joint_f1.append(jf)
            joint_mcc.append(jmcc)

        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue
            pass

    def average(metrics):
        return sum(metrics) / len(metrics) if metrics else 0.0

    return {
        "sentence_metrics": {
            "avg_precision": average(sentence_precision),
            "avg_recall": average(sentence_recall),
            "avg_f1": average(sentence_f1),
            "avg_mcc": average(sentence_mcc)
        },
        "image_metrics": {
            "avg_precision": average(image_precision),
            "avg_recall": average(image_recall),
            "avg_f1": average(image_f1),
            "avg_mcc": average(image_mcc)
        },
        "joint_metrics": {
            "avg_precision": average(joint_precision),
            "avg_recall": average(joint_recall),
            "avg_f1": average(joint_f1),
            "avg_mcc": average(joint_mcc)
        },
        "num_samples": len(samples)
    }

def evaluate_multimodal_retrieval_global(samples):
    """全局评估多模态检索性能：累积 TP/FP/FN/TN 并统一计算指标"""

    # 累积句子指标
    tp_utt, fp_utt, fn_utt, tn_utt = 0, 0, 0, 0
    # 累积图像指标
    tp_img, fp_img, fn_img, tn_img = 0, 0, 0, 0

    for sample in tqdm(samples, desc="累计 TP/FP/FN/TN"):
        try: 
            gt = sample["ground_truth"]
            pred = sample["model_prediction"]

            max_utt_id = sample.get("max_utt_id", 20)
            max_img_id = sample.get("max_img_id", 20)

            # ----- Sentence-Level -----
            gt_utt = set(gt.get("sentence_ids", []))
            pred_utt = set(pred.get("sentence_ids", []))
            all_utt_ids = set(range(max_utt_id + 1))

            tp_utt += len(gt_utt & pred_utt)
            fp_utt += len(pred_utt - gt_utt)
            fn_utt += len(gt_utt - pred_utt)
            tn_utt += len(all_utt_ids - (gt_utt | pred_utt))

            # ----- Image-Level -----
            gt_img = set(gt.get("image_ids", []))
            pred_img = set(pred.get("image_ids", []))
            all_img_ids = set(range(max_img_id + 1))

            tp_img += len(gt_img & pred_img)
            fp_img += len(pred_img - gt_img)
            fn_img += len(gt_img - pred_img)
            tn_img += len(all_img_ids - (gt_img | pred_img))
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue
            pass

    def compute_metrics(tp, fp, fn, tn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mcc_numerator = tp * tn - fp * fn
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0
        return precision, recall, f1, mcc

    # 分别计算指标
    sp, sr, sf1, smcc = compute_metrics(tp_utt, fp_utt, fn_utt, tn_utt)
    ip, ir, if1, imcc = compute_metrics(tp_img, fp_img, fn_img, tn_img)

    # 联合指标（可以改为 harmonic mean、平均或其他定义）
    jp = 2 * sp * ip / (sp + ip) if (sp + ip) > 0 else 0.0
    jr = 2 * sr * ir / (sr + ir) if (sr + ir) > 0 else 0.0
    jf = (sf1 + if1) / 2
    jmcc = (smcc + imcc) / 2

    return {
        "sentence_metrics": {
            "global_precision": round(sp, 4),
            "global_recall": round(sr, 4),
            "global_f1": round(sf1, 4),
            "global_mcc": round(smcc, 4)
        },
        "image_metrics": {
            "global_precision": round(ip, 4),
            "global_recall": round(ir, 4),
            "global_f1": round(if1, 4),
            "global_mcc": round(imcc, 4)
        },
        "joint_metrics": {
            "global_precision": round(jp, 4),
            "global_recall": round(jr, 4),
            "global_f1": round(jf, 4),
            "global_mcc": round(jmcc, 4)
        },
        "num_samples": len(samples)
    }





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多模态中文长对话检索任务评估工具（动态max_id）')
    parser.add_argument('--input', required=True, help='输入的JSONL格式结果文件路径')
    parser.add_argument('--output', help='输出结果的JSON文件路径，可选')
    
    args = parser.parse_args()
    samples = load_and_prepare_samples(args.input)
    
    if not samples:
        print("错误: 没有加载到有效样本", file=sys.stderr)
        sys.exit(1)
    
    # metrics = evaluate_multimodal_retrieval(samples)
    metrics = evaluate_multimodal_retrieval_global(samples)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"评估结果已保存到 {args.output}")
        except Exception as e:
            print(f"警告: 无法保存结果到文件: {e}，将打印到控制台", file=sys.stderr)
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))