import json
import re
from collections import defaultdict

def extract_ids(text):
    """提取utt_ids和img_ids（保持原逻辑）"""
    utt_pattern = r"<\|utt_ids_start\|>(.*?)<\|utt_ids_end\|>"
    img_pattern = r"<\|img_ids_start\|>(.*?)<\|img_ids_end\|>"
    
    utt_match = re.search(utt_pattern, text, re.DOTALL)
    utt_content = utt_match.group(1).strip() if utt_match else "[]"
    
    img_match = re.search(img_pattern, text, re.DOTALL)
    img_content = img_match.group(1).strip() if img_match else "[]"
    
    def parse_ids(content):
        if content == "[]":
            return []
        return list(map(int, re.findall(r"\d+", content)))
    
    return {
        "utt_ids": parse_ids(utt_content),
        "img_ids": parse_ids(img_content)
    }

def merge_ids(ids_set):
    """将集合转换为排序后的列表"""
    return sorted(ids_set) if ids_set else []

def load_original_dialogues(original_json_path):
    """
    加载原始对话数据：
    - 键：dialogue_id
    - 值：{
        "solution": 原始标签（labels）,
        "img_path_to_id": 图片路径→原始ID的映射（用于校准）
      }
    """
    original = {}
    with open(original_json_path, "r", encoding="utf-8") as f:
        for dialog in json.load(f):
            dialog_id = dialog["dialogue_id"]
            # 提取原始标签（solution）
            solution = dialog.get("solution", "")  # 假设标签存在于solution字段
            
            # 提取原始图片路径，建立路径→原始ID映射（按出现顺序）
            img_paths = dialog.get("images", [])
           
            img_path_to_id = {path: i for i, path in enumerate(img_paths)}
            
            original[dialog_id] = {
                "solution": solution,
                "img_path_to_id": img_path_to_id,
                "images":  dialog.get("images", []),
                "messages": dialog.get("messages", [])
            }
    return original

def process_json(input_file, original_json_path, output_file):
    # 1. 加载原始对话数据（含solution和图片映射）
    original_data = load_original_dialogues(original_json_path)

    # 2. 分组处理拆解片段的response（仅合并预测结果）
    groups = defaultdict(lambda: {
        "response_utt": set(),
        "response_img_paths": set()  # 预测的图片路径（去重）
    })
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            base_id = data["dialogue_id"].split("_split_")[0]  # 原始对话ID
            
            # 解析response中的ids
            response = data["response"]
            retrieval = extract_ids(response)
            response_utt = retrieval["utt_ids"]
            response_img_ids = retrieval["img_ids"]

            # 片段内图片ID→路径映射
            images = data.get("images", [])
            img_id_to_path = {i: img["path"] for i, img in enumerate(images)}

            # 收集预测的utt_ids和图片路径
            groups[base_id]["response_utt"].update(response_utt)
            for img_id in response_img_ids:
                if img_id in img_id_to_path:
                    groups[base_id]["response_img_paths"].add(img_id_to_path[img_id])
    
    # 3. 生成最终结果（关联原始solution，校准img_ids）
    merged = []
    for base_id, info in groups.items():
        # 检查原始对话是否存在
        if base_id not in original_data:
            print(f"警告：原始对话中未找到dialogue_id={base_id}，跳过")
            continue
        
        # 从原始数据中获取标签（solution）
        labels = original_data[base_id]["solution"]
        
        # 校准response的img_ids（映射为原始对话中的真实ID）
        original_img_map = original_data[base_id]["img_path_to_id"]
        response_img_ids = []
        for path in info["response_img_paths"]:
            if path in original_img_map:
                response_img_ids.append(original_img_map[path])
        response_img_ids = sorted(list(set(response_img_ids)))  # 去重排序
        
        # 处理response的utt_ids（直接合并排序）
        response_utt = merge_ids(info["response_utt"])
        
        # 构建response字符串
        response = (
            f"<|utt_ids_start|>{response_utt}<|utt_ids_end|>;"
            f"<|img_ids_start|>{response_img_ids}<|img_ids_end|>"
        )
        
        merged.append({
            "dialogue_id": base_id,
            "response": response,
            "labels": labels,  # 直接使用原始solution
            "messages": original_data[base_id]["messages"],
            "images": original_data[base_id]["images"]
        })
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 使用示例
if __name__ == "__main__":
    process_json(
        input_file="",
        original_json_path="",  # 含dialogue_id和solution字段
        output_file=""
    )