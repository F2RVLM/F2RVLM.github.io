

import base64
import json
import time
import io
import re
import os
from typing import List, Dict, Union, Optional, Any, Set
from openai import OpenAI
from PIL import Image

MAX_PIXELS = 28*28*768  # 默认限制为100万像素


client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.environ.get("ARK_API_KEY"),
)




def load_processed_ids(jsonl_path: str) -> Set[str]:
    """从JSONL文件中加载已处理成功的样本ID"""
    processed_ids = set()
    
    if not os.path.exists(jsonl_path):
        return processed_ids
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                dialogue_id = data.get('dialogue_id', '')
                if dialogue_id and data.get('response'):  # 只考虑有响应的样本为成功
                    processed_ids.add(dialogue_id)
    except Exception as e:
        print(f"加载已处理样本ID时出错: {e}")
    
    return processed_ids

def parse_json_data(json_input: Union[str, Dict]) -> Dict[str, Any]:
    """解析JSON格式的对话数据（支持文件路径或JSON字符串）"""
    try:
        if isinstance(json_input, str):
            if os.path.isfile(json_input):
                with open(json_input, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return json.loads(json_input)
        elif isinstance(json_input, dict):
            return json_input
        else:
            print("错误：输入必须是文件路径、JSON字符串或字典")
            return {}
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return {}


def extract_image_paths(json_data: Dict[str, Any], img_ids: List[int]) -> List[str]:
    """根据图片ID提取对应的文件路径"""
    images = json_data.get("images", [])
    return [images[i] for i in img_ids if 0 <= i < len(images)]


def replace_image_tags(content: str) -> str:
    """将<image>标签替换为模型所需的视觉标记"""
    return content.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")


def extract_user_message(json_data: Dict[str, Any]) -> str:
    """提取role="user"的消息内容"""
    messages = json_data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def resize_image(img: Image.Image, max_pixels: int) -> Image.Image:
    """调整图片大小，保持宽高比，确保总像素数不超过限制"""
    width, height = img.size
    current_pixels = width * height
    
    if current_pixels <= max_pixels:
        return img  # 图片已经符合要求，无需调整
    
    # 计算需要的缩放比例
    scale_factor = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 调整图片大小，使用高质量的抗锯齿方法
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    print(f"图片已从 {width}x{height} 调整为 {new_width}x{new_height}")
    return resized_img


def call_vlm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    user_img_bin_list: List[bytes]
) -> str:
    """调用多模态模型API并返回响应"""
    try:
        b64_img_list = []
        for img_bin in user_img_bin_list:
            # 打开图片并检查/调整大小
            try:
                img = Image.open(io.BytesIO(img_bin))
                img = resize_image(img, MAX_PIXELS)  # 调整图片大小
                
                # 将调整后的图片转换回二进制
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)  # 使用JPEG格式和90%质量
                resized_img_bin = buffer.getvalue()
                
                # 编码为Base64
                b64_img = base64.b64encode(resized_img_bin).decode("utf-8")
                b64_img = f"data:image/jpeg;base64,{b64_img}"
                b64_img_list.append(b64_img)
            except Exception as e:
                print(f"处理图片时出错: {e}")
                continue
        
        if not b64_img_list:
            print("警告：没有成功处理的图片，可能导致模型调用失败")
        
        client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key=os.environ.get("ARK_API_KEY"),
        )
        content = [{"type": "image_url", "image_url": {"url": b64_img}} for b64_img in b64_img_list] + [
            {"type": "text", "text": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.0,
            timeout=200
        )
        print(response.choices[0])
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"API调用错误: {e}")
        return ""


def process_multimodal_task(json_data: Dict[str, Any]) -> str:
    """处理单个多模态检索任务并返回模型响应"""
    user_message = extract_user_message(json_data)
    if not user_message:
        print("错误：样本中未找到用户消息")
        return ""
    
    img_id_pattern = re.compile(r'<\|img_id_start\|>(\d+)<\|img_id_end\|>')
    img_ids = [int(i) for i in img_id_pattern.findall(user_message) if i.isdigit()]
    
    image_paths = extract_image_paths(json_data, img_ids)
    if not image_paths:
        print("警告：样本中未找到有效图片路径")
        return ""
    
    user_img_bin_list = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                img_bin = f.read()
            user_img_bin_list.append(img_bin)
        except Exception as e:
            print(f"读取图片失败 {path}: {e}")
    
    try:
        result = call_vlm(
            model_name="doubao-seed-1-6-flash-250615",
            system_prompt="你是一个多模态信息检索助手。",
            user_prompt=user_message,
            user_img_bin_list=user_img_bin_list
        )
        return result
    except Exception as e:
        print(f"处理样本失败: {e}")
        return ""


def create_result_dict(result_str: str, sample_json: dict) -> dict:
    """生成包含响应、标签、消息、对话ID和图片信息的结果字典"""
    result_dict = {
        "response": result_str,  # 直接使用原始字符串，不做任何过滤
        "labels": sample_json.get("solution", ""),
        "messages": sample_json.get("messages", []),
        "dialogue_id": sample_json.get("dialogue_id", ""),
        "images": [{"bytes": None, "path": path} for path in sample_json.get("images", [])]
    }
    return result_dict

def save_to_jsonl(data_list: List[dict], output_file_path: str) -> None:
    """将结果列表保存为JSONL格式（每行一个JSON对象）"""
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                json_line = json.dumps(data, ensure_ascii=False) + "\n"
                f.write(json_line)
        print(f"已保存 {len(data_list)} 个样本到 {output_file_path}")
    except Exception as e:
        print(f"保存JSONL文件失败: {e}")

def batch_process_samples_with_retry(
    input_json_path: str, 
    output_jsonl_path: str,
    retry_only: bool = True
) -> None:
    """批量处理JSON文件中的样本，跳过已成功处理的样本"""
    # 加载原始样本数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # 获取已成功处理的样本ID
    processed_ids = load_processed_ids(output_jsonl_path)
    print(f"已发现 {len(processed_ids)} 个成功处理的样本")
    
    # 准备处理的样本
    samples_to_process = []
    for sample in samples:
        sample_id = sample.get('dialogue_id', '')
        if not sample_id:
            print(f"警告: 样本缺少dialogue_id，将被处理")
            samples_to_process.append(sample)
        elif sample_id not in processed_ids:
            samples_to_process.append(sample)
    
    print(f"需要处理的样本数量: {len(samples_to_process)}")
    
    # 如果不只是重试，而是追加处理，先将原文件内容复制到新文件
    if not retry_only and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        with open(output_jsonl_path + ".temp", 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines)
        
        os.replace(output_jsonl_path + ".temp", output_jsonl_path)
    
    # 处理剩余样本
    success_count = 0
    for i, sample in enumerate(samples_to_process):
        sample_id = sample.get('dialogue_id', f"样本{i+1}")
        print(f"处理样本 {i+1}/{len(samples_to_process)}: {sample_id}")
        
        # 调用原有的处理函数
        result = process_multimodal_task(sample)
        result_dict = create_result_dict(result, sample)
        
        # 将结果追加到JSONL文件
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(result_dict, ensure_ascii=False) + "\n"
            f.write(json_line)
        
        if result:
            success_count += 1
            print(f"样本处理成功: {sample_id}")
        else:
            print(f"样本处理失败: {sample_id}")
    
    print(f"处理完成! 成功: {success_count}/{len(samples_to_process)}")

def batch_process_samples(json_file_path: str, output_file_path: str = "results.json") -> List[dict]:
    """批量处理JSON文件中的多个样本并保存结果"""
    samples = parse_json_data(json_file_path)
    if not samples:
        print("错误：未解析到样本数据")
        return []
    
    results = []
    print(f"开始处理 {len(samples)} 个样本...")
    
    for i, sample in enumerate(samples):
        print(f"处理样本 {i+1}/{len(samples)}: {sample.get('dialogue_id', '无ID')}")
        result = process_multimodal_task(sample)
        result_dict = create_result_dict(result, sample)
        
        results.append(result_dict)  # 无论result是否为空，都添加到结果中
        status = "成功" if result else "失败（无响应）"
        print(f"样本处理{status}: {sample.get('dialogue_id', '无ID')}")
    

    save_to_jsonl(results, output_file_path)
    return results

# 示例用法
if __name__ == "__main__":
    input_json_path = ""
    output_jsonl_path = ""
    # 设置retry_only=True表示只处理失败的样本
    batch_process_samples_with_retry(input_json_path, output_jsonl_path, retry_only=True)