import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards

import re
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
import sys
from tqdm import tqdm

class RetrievalOrderSim(ORM):
    def __init__(self, 
                 sentence_base: str = "./MLDR/dialogue_ID_embeddings",
                 embedding_dir_map: Dict[str, str] = None):
        """
        初始化跨模态顺序相似度奖励函数
        
        Args:
            sentence_base: 句子嵌入文件的基础目录
            embedding_dir_map: 图片路径到嵌入目录的映射
            weight: 奖励权重，范围[0,1]
        """
        self.sentence_base = Path(sentence_base)
        self.embedding_dir_map = embedding_dir_map or {
            './MLDR/MLDR_train_images/img0/': './MLDR/MLDR_train_images/img0_embeddings/',
            './MLDR/MLDR_train_images/img1/': './MLDR/MLDR_train_images/img1_embeddings/'
        }
        
        # 正则表达式定义
        self.id_pattern = re.compile(
            r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;'
            r'<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>(?![\s\S])')
    
    def __call__(self, completions, solution, messages, dialogue_id, images, **kwargs) -> List[float]:
        """
        计算跨模态顺序相似度奖励
        
        Args:
            completions: 模型生成的检索结果列表
            solution: 真值检索结果列表
            messages: 对话消息列表，包含对话历史
            
        Returns:
            奖励分数列表，每个元素为样本的跨模态顺序相似度（范围[-1, 1]）
        """
        rewards = []
        for content, gt_content, sample, sample_id, imgs in zip(completions, solution, messages, dialogue_id, images):
            try:
                # 1. 解析ID
                pred_ids = self._parse_retrieval_ids(content)
                gt_ids = self._parse_retrieval_ids(gt_content)
                                
                # 2. 检查边界条件
                is_pred_empty = not pred_ids['sentence_ids'] and not pred_ids['image_ids']
                is_gt_empty = not gt_ids['sentence_ids'] and not gt_ids['image_ids']
                
                # 特殊情况处理
                if is_pred_empty and is_gt_empty:
                    # 预测和真值都为空集，完全匹配
                    rewards.append(1.0)
                    continue
                
                if is_pred_empty and not is_gt_empty:
                    # 预测为空，真值非空，完全不匹配
                    rewards.append(0.0)
                    continue
                
                if not is_pred_empty and is_gt_empty:
                    # 预测非空，真值为空，完全不匹配
                    rewards.append(0.0)
                    continue
                
                # 3. 计算跨模态顺序相似度
                reward = self._calculate_cross_modal_order(
                    pred_ids, gt_ids, sample, sample_id, imgs
                )
                
                rewards.append(reward)
                
            except Exception as e:
                print(f"奖励计算异常: {str(e)}", file=sys.stderr)
                rewards.append(0.0)
                
        return rewards
    
    def _parse_retrieval_ids(self, text: str) -> Dict[str, List[int]]:
        """解析检索结果中的句子和图片ID"""
        match = self.id_pattern.match(text)
        if not match:
            return {"sentence_ids": [], "image_ids": []}
        
        utt_ids = self._parse_id_list(match.group("utt"))
        img_ids = self._parse_id_list(match.group("img"))
        return {"sentence_ids": utt_ids, "image_ids": img_ids}
    
    def _parse_id_list(self, id_str: str) -> List[int]:
        """安全解析ID列表"""
        if not id_str.strip():
            return []
        return [int(x.strip()) for x in id_str.split(',') if x.strip().isdigit()]
        
    def _calculate_cross_modal_order(self, pred_ids: Dict[str, List[int]], gt_ids: Dict[str, List[int]], 
                                    sample: Dict, dialog_id: str, sample_images: List[Dict]) -> float:
        """计算跨模态顺序相似度（指标3）"""
        pred_sent_ids, pred_img_ids = pred_ids['sentence_ids'], pred_ids['image_ids']
                
        # 解析图片-句子映射
        mapping = self._get_image_sentence_mapping(sample)
        
        if not mapping and pred_img_ids:
            print("警告: 未找到图片-句子映射关系，跨模态指标返回0", file=sys.stderr)
            return 0.0
        
        all_utts = list(dict.fromkeys(pred_sent_ids))

        # 加载句子嵌入
        utt_emb_dir = self.sentence_base / dialog_id if dialog_id else self.sentence_base
        utt_id_to_emb = self._load_embeddings(all_utts, utt_emb_dir)
        valid_sent_ids = [uid for uid in all_utts if uid in utt_id_to_emb]
        
        # 加载图片嵌入
        img_id_to_emb, _ = self._load_image_embeddings(pred_img_ids, sample_images)
        valid_img_ids = list(img_id_to_emb.keys())
        
        # 构建混合嵌入列表
        mixed_embs = []
        if valid_sent_ids and mapping and valid_img_ids:
            # 构建句子到图片嵌入的映射
            sent_to_img_embs = {}
            for img_id in valid_img_ids:
                if img_id not in mapping:
                    continue
                    
                assoc_utt = mapping[img_id][0]
                if assoc_utt in sent_to_img_embs:
                    sent_to_img_embs[assoc_utt].append(img_id_to_emb[img_id])
                else:
                    sent_to_img_embs[assoc_utt] = [img_id_to_emb[img_id]]
            
            # 合并并排序句子ID
            all_sents = sorted(set(valid_sent_ids + list(sent_to_img_embs.keys()))) 
            for sent_id in all_sents:
                if sent_id in utt_id_to_emb:
                    mixed_embs.append(utt_id_to_emb[sent_id].reshape(1, -1))
                
                if sent_id in sent_to_img_embs:
                    img_embs = sent_to_img_embs[sent_id]
                    avg_img_emb = np.mean(np.vstack(img_embs), axis=0).reshape(1, -1)
                    mixed_embs.append(avg_img_emb)
        
        # 计算相邻相似度
        if not mixed_embs:
            # 没有有效的嵌入，返回0
            return 0.0
            
        if len(mixed_embs) < 2:
            # 只有一个元素，无法计算顺序相似度
            # 可以根据业务需求选择返回0或0.5
            # 这里选择返回0.5，表示不确定
            print("警告: 混合嵌入列表长度小于2，返回0.5作为默认值", file=sys.stderr)
            return 0.5
            
        mixed_seq_scores = []
        for i in range(len(mixed_embs) - 1):
            sim = self._cosine_similarity(mixed_embs[i], mixed_embs[i+1])
            mixed_seq_scores.append(sim)
        
        return np.mean(mixed_seq_scores) if mixed_seq_scores else 0.0
    
    def _get_image_sentence_mapping(self, sample: Dict) -> Dict[int, List[int]]:
        """解析图片ID与句子ID的对应关系"""
        user_content = next((msg['content'] for msg in sample if msg.get('role') == 'user'), '')
        image_mapping = {}
        
        utt_pattern = re.compile(r'<\|utt_id_start\|>(\d+)<\|utt_id_end\|>')
        img_pattern = re.compile(r'<\|img_id_start\|>(\d+)<\|img_id_end\|>')
        
        for line in user_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            utt_match = utt_pattern.search(line)
            if not utt_match:
                continue
                
            utt_id = int(utt_match.group(1))
            img_matches = img_pattern.findall(line)
            
            for img_id_str in img_matches:
                img_id = int(img_id_str)
                image_mapping.setdefault(img_id, []).append(utt_id)
        
        return image_mapping
    
    def _load_embeddings(self, ids: List[int], base_dir: Path) -> Dict[int, np.ndarray]:
        """加载ID的嵌入向量"""
        embeddings = {}
        for idx in ids:
            file_path = base_dir / f"{idx}.npy"
            if file_path.exists():
                try:
                    embeddings[idx] = np.load(file_path).squeeze()
                except Exception as e:
                    print(f"警告: 加载句子嵌入失败 - {file_path}: {e}", file=sys.stderr)
        return embeddings
    
    def _load_image_embeddings(self, image_ids: List[int], sample_images: List[Dict]) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """加载图片嵌入向量"""
        img_emb_map = {}
        failed_ids = []
        
        for img_id in image_ids:
            if img_id >= len(sample_images):
                failed_ids.append(img_id)
                continue
                
            img_path = sample_images[img_id].get("path", "")
            if not img_path:
                failed_ids.append(img_id)
                continue
                
            # 替换路径为嵌入目录
            new_path = img_path
            for old_dir, new_dir in self.embedding_dir_map.items():
                if old_dir in new_path:
                    new_path = new_path.replace(old_dir, new_dir)
                    break
                    
            embedding_path = Path(new_path).with_suffix('.npy')
            try:
                emb = np.load(embedding_path).squeeze()
                img_emb_map[img_id] = emb
            except Exception as e:
                failed_ids.append(img_id)
                print(f"警告: 加载图片嵌入失败 - {embedding_path}: {e}", file=sys.stderr)
        
        return img_emb_map, failed_ids
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
            
        emb1 = emb1.reshape(1, -1) if emb1.ndim == 1 else emb1
        emb2 = emb2.reshape(1, -1) if emb2.ndim == 1 else emb2
        
        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True).clip(min=1e-10)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True).clip(min=1e-10)
        norm_emb1, norm_emb2 = emb1 / norm1, emb2 / norm2
        
        sim = np.dot(norm_emb1, norm_emb2.T)[0, 0]
        return max(min(sim, 1.0), -1.0)  # 确保在[-1,1]范围内


class RetrievalMcc(ORM):
    def __init__(self):
        """初始化仅计算MCC的检索奖励函数"""
        # 正则表达式定义
        self.id_pattern = re.compile(
            r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;'
            r'<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>(?![\s\S])'
        )
        self.max_utt_id_pattern = re.compile(r'<\|utt_id_start\|>(\d+)<\|utt_id_end\|>')
        self.max_img_id_pattern = re.compile(r'<\|img_id_start\|>(\d+)<\|img_id_end\|>')
    
    def __call__(self, completions: List[str], solution: List[str], messages: List[Dict]) -> List[float]:
        """
        计算基于MCC指标的奖励分数
        
        Args:
            completions: 模型生成的检索结果列表
            solution: 真值检索结果列表
            messages: 对话消息列表，包含对话历史
            
        Returns:
            奖励分数列表，每个元素为样本的MCC值（范围[-1, 1]）
        """
        rewards = []
        for content, gt_content, sample in zip(completions, solution, messages):
            try:
                # 1. 提取最大ID
                max_utt_id, max_img_id = self._extract_max_ids(sample)
                
                # 2. 解析预测和真值ID
                gen_ids = self._parse_retrieval_ids(content)
                gt_ids = self._parse_retrieval_ids(gt_content)
                
                if not gen_ids or not gt_ids:
                    rewards.append(0.0)
                    continue
                
                # 3. 计算句子和图片的MCC
                utt_mcc = self._calculate_mcc(
                    gen_ids["sentence_ids"], 
                    gt_ids["sentence_ids"], 
                    max_utt_id
                )
                img_mcc = self._calculate_mcc(
                    gen_ids["image_ids"], 
                    gt_ids["image_ids"], 
                    max_img_id
                )
                
                # 4. 合并MCC（简单平均）
                final_reward = (utt_mcc + img_mcc) / 2
                rewards.append(final_reward)
                
            except Exception:
                rewards.append(0.0)
                
        return rewards
    
    def _extract_max_ids(self, sample: Dict) -> Tuple[int, int]:
        """从对话中提取最大句子ID和图片ID"""
        user_content = ""
        for msg in sample:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        # 提取句子ID
        utt_matches = self.max_utt_id_pattern.findall(user_content)
        max_utt_id = max(map(int, utt_matches)) if utt_matches else 20
        
        # 提取图片ID
        img_matches = self.max_img_id_pattern.findall(user_content)
        max_img_id = max(map(int, img_matches)) if img_matches else 20
        
        return max_utt_id, max_img_id
    
    def _parse_retrieval_ids(self, text: str) -> Dict[str, List[int]]:
        """解析检索结果中的句子和图片ID"""
        match = self.id_pattern.match(text)
        if not match:
            return {"sentence_ids": [], "image_ids": []}
        
        utt_ids = self._parse_id_list(match.group("utt"))
        img_ids = self._parse_id_list(match.group("img"))
        return {"sentence_ids": utt_ids, "image_ids": img_ids}
    
    def _calculate_mcc(self, pred_ids: List[int], gt_ids: List[int], max_id: int) -> float:
        """计算Matthews相关系数(MCC)"""
        # 处理空集情况
        if not pred_ids and not gt_ids:
            return 1.0  # 全匹配
        if not pred_ids or not gt_ids:
            return 0.0  # 部分空集
        
        # 构建集合
        all_ids = set(range(max_id + 1))
        pred_pos = set(pred_ids)
        gt_pos = set(gt_ids)
        pred_neg = all_ids - pred_pos
        gt_neg = all_ids - gt_pos
        
        # 计算混淆矩阵
        tp = len(pred_pos & gt_pos)    # 真正例
        fp = len(pred_pos & gt_neg)    # 假正例
        tn = len(pred_neg & gt_neg)    # 真反例
        fn = len(pred_neg & gt_pos)    # 假反例
        
        # 计算MCC（处理分母为0的情况）
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    
    def _parse_id_list(self, id_str: str) -> List[int]:
        """安全解析ID列表"""
        if not id_str.strip():
            return []
        return [int(x.strip()) for x in id_str.split(',') if x.strip().isdigit()]
    
class RetrievalF1(ORM):
    def __init__(self, 
                 utt_weight: float = 0.5,
                 img_weight: float = 0.5,
                 beta: float = 1.0,
                 base_length_penalty: float = 0.95):  # 新增长度惩罚参数
        """
        初始化检索准确率奖励函数
        
        Args:
            utt_weight: utt_ids的权重
            img_weight: img_ids的权重
            beta: F1分数中的beta参数
            base_length_penalty: 长度惩罚基准值（默认0.95，值越小惩罚越严格）
        """
        self.utt_weight = utt_weight
        self.img_weight = img_weight
        self.beta = beta
        self.base_length_penalty = base_length_penalty  # 保存长度惩罚参数
        
        # 预编译正则表达式
        self.pattern = re.compile(
            r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;'
            r'<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>(?![\s\S])'
        )
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """计算奖励分数（包含长度惩罚）"""
        rewards = []
        # print('completions', completions)
        # print('solution', solution)
        # print('kwargs_keys', kwargs.keys() )
        # print('kwargs', kwargs)

        for content, sol in zip(completions, solution):
            try:
                # 解析生成内容和真值
                gen_match = self.pattern.match(content)
                if not gen_match:
                    rewards.append(0.0)
                    continue
                true_match = self.pattern.match(sol)
                if not true_match:
                    rewards.append(0.0)
                    continue
                
                # 提取ID列表
                utt_gen = self._parse_id_list(gen_match.group('utt'))
                img_gen = self._parse_id_list(gen_match.group('img'))
                utt_true = self._parse_id_list(true_match.group('utt'))
                img_true = self._parse_id_list(true_match.group('img'))
                
                # 计算F1分数（含长度惩罚）
                utt_score = self._calculate_scored_with_penalty(utt_gen, utt_true)
                img_score = self._calculate_scored_with_penalty(img_gen, img_true)
                
                # 加权总得分
                reward = utt_score * self.utt_weight + img_score * self.img_weight
                rewards.append(reward)
                
            except Exception as e:
                print(f"Error processing example: {e}")
                rewards.append(0.0)
                
        return rewards
    
    def _calculate_scored_with_penalty(self, predicted: List[int], ground_truth: List[int]) -> float:
        """计算带长度惩罚的F1分数"""
        # 计算F1分数
        f1 = self._calculate_f1(predicted, ground_truth)
        # 计算长度惩罚
        length_penalty = self._length_penalty(predicted, ground_truth)
        # 总得分 = F1分数 × 长度惩罚
        return f1 * length_penalty
    
    def _length_penalty(self, pred: List[int], gt: List[int]) -> float:
        """计算长度惩罚（差异越大，惩罚越重）"""
        delta = abs(len(pred) - len(gt))
        return self.base_length_penalty ** delta  # 指数级惩罚，例如差异1则乘0.95，差异2则乘0.9025
    
    def _parse_id_list(self, id_str: str) -> List[int]:
        """解析ID列表字符串"""
        if not id_str.strip():
            return []
        return [int(x.strip()) for x in id_str.split(',') if x.strip()]
    
    def _calculate_f1(self, predicted: List[int], ground_truth: List[int]) -> float:
        """计算F1分数"""
        if not predicted and not ground_truth:
            return 1.0
        
        intersection = set(predicted).intersection(set(ground_truth))
        intersection_size = len(intersection)
        precision = intersection_size / len(predicted) if predicted else 0.0
        recall = intersection_size / len(ground_truth) if ground_truth else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        beta_squared = self.beta ** 2
        return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    

class RetrievalFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        奖励函数：检查完成的内容是否符合指定格式模板，并且两个列表中不含重复项。
        格式要求：
        <|utt_ids_start|>[数字列表或空]<|utt_ids_end|>;<|img_ids_start|>[数字列表或空]<|img_ids_end|>
        """
        pattern = r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>(?![\s\S])'
        
        rewards = []
        for content in completions:
            try:
                match = re.match(pattern, content)
                if not match:
                    rewards.append(0.0)
                    continue

                # 提取列表内容
                utt_ids_str = match.group("utt").strip()
                img_ids_str = match.group("img").strip()

                # 空列表处理
                utt_ids = [int(x) for x in utt_ids_str.split(",") if x.strip()] if utt_ids_str else []
                img_ids = [int(x) for x in img_ids_str.split(",") if x.strip()] if img_ids_str else []

                # 检查是否有重复
                if len(set(utt_ids)) != len(utt_ids) or len(set(img_ids)) != len(img_ids):
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)

            except Exception as e:
                rewards.append(0.0)  # 异常时添加0.0

        return rewards 


    def __call__(self, completions, **kwargs) -> List[float]:
        """
        奖励函数：检查是否符合如下格式：
        <think>推理过程</think> <answer><|utt_ids_start|>[数字列表]<|utt_ids_end|>;<|img_ids_start|>[数字列表]<|img_ids_end|></answer>
        并验证两个列表中无重复项。
        """
        outer_pattern = re.compile(
            r'^<think>.*?</think>\s*<answer>(.*?)</answer>(?![\s\S])',
            re.DOTALL  # 允许跨行匹配
        )
        inner_pattern = re.compile(
            r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;'
            r'<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>$'
        )

        rewards = []
        for content in completions:
            try:
                outer_match = outer_pattern.match(content)
                if not outer_match:
                    rewards.append(0.0)
                    continue

                answer_body = outer_match.group(1).strip()
                inner_match = inner_pattern.match(answer_body)
                if not inner_match:
                    rewards.append(0.0)
                    continue

                utt_str = inner_match.group("utt").strip()
                img_str = inner_match.group("img").strip()

                utt_ids = [int(x) for x in utt_str.split(",") if x.strip()] if utt_str else []
                img_ids = [int(x) for x in img_str.split(",") if x.strip()] if img_str else []

                if len(set(utt_ids)) != len(utt_ids) or len(set(img_ids)) != len(img_ids):
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)

            except Exception:
                rewards.append(0.0)

        return rewards


class RetrievalAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        奖励函数：检查生成的utt_id和img_id列表是否与真值完全匹配（显式添加分数版）
        Args:
            completions (list[str]): 生成的输出
            solution (list[str]): 真实标签，格式与生成内容一致
        Returns:
            list[float]: 长度与输入一致的奖励分数列表
        """
        rewards = []
        pattern = r'^<\|utt_ids_start\|>(\[(?P<utt>[\d,\s]*)\])<\|utt_ids_end\|>;<\|img_ids_start\|>(\[(?P<img>[\d,\s]*)\])<\|img_ids_end\|>(?![\s\S])'
        
        if len(completions) != len(solution):
            raise ValueError("completions和solution的长度必须一致")
        
        for content, sol in zip(completions, solution):
            try:
                # 解析生成的内容和真实标签
                # print(content, sol)
                gen_match = re.match(pattern, content)
                true_match = re.match(pattern, sol)
                
                # 情况1：格式不匹配，直接添加0.0
                if not gen_match or not true_match:
                    rewards.append(0.0)
                    continue  # 跳过后续匹配逻辑
                
                # 情况2：格式匹配，提取并检查ID列表
                utt_gen = self._parse_id_list(gen_match.group('utt'))
                img_gen = self._parse_id_list(gen_match.group('img'))
                utt_true = self._parse_id_list(true_match.group('utt'))
                img_true = self._parse_id_list(true_match.group('img'))
                
                # 列表完全匹配，添加1.0；否则添加0.0
                if utt_gen == utt_true and img_gen == img_true:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                    
            except Exception as e:
                print(f"Error processing example: {e}")
                rewards.append(0.0)  # 异常时添加0.0
        
        return rewards
    
    def _parse_id_list(self, id_str: str) -> List[int]:
        """将ID字符串解析为整数列表（支持空列表）"""
        if not id_str.strip():  # 处理空列表 []
            return []
        return [int(x.strip()) for x in id_str.split(',') if x.strip()]
    
    
class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        import os
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards

orms['external_retrieval_sim'] = RetrievalOrderSim
orms['external_retrieval_mcc'] = RetrievalMcc
orms['external_retrieval_f1'] = RetrievalF1
orms['external_retrieval_acc'] = RetrievalAccuracy
orms['external_retrieval_format'] = RetrievalFormat


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# For genrm you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
