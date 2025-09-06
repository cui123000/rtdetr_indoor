# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 场景分类器模块
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
import clip

class SceneClassifier:
    """场景分类器，用于识别室内场景类型"""
    
    def __init__(self, clip_model, clip_preprocess=None, device="cuda"):
        """初始化场景分类器"""
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 场景类别定义（添加更多场景类型）
        self.scene_types = [
            "bathroom", "bedroom", "kitchen", "living room", 
            "dining room", "office", "hallway", "laundry room",
            "person", "portrait", "selfie",  # 人物相关场景
            "food", "meal", "dish", "cuisine",  # 食物相关场景
            "restaurant", "cafe", "tableware",  # 餐饮场所
            "cat", "dog", "pet", "animal"  # 动物相关场景
        ]
        
        # 场景描述模板
        self.scene_templates = [
            "a {}", "an indoor {}", "a typical {}", 
            "a photo of a {}", "a picture of a {}"
        ]
        
        # 缓存场景文本特征
        self._cache_scene_features()
        
        print("✅ 场景分类器初始化完成")
    
    def _cache_scene_features(self):
        """缓存场景文本特征"""
        all_features = []
        
        for scene in self.scene_types:
            scene_features = []
            
            for template in self.scene_templates:
                text = template.format(scene)
                tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    features = self.clip_model.encode_text(tokens)
                    normalized = F.normalize(features, p=2, dim=1)
                    scene_features.append(normalized)
            
            # 平均所有模板的特征
            scene_avg = torch.cat(scene_features).mean(dim=0, keepdim=True)
            all_features.append(scene_avg)
        
        # 合并所有场景特征
        self.scene_text_features = torch.cat(all_features, dim=0)
    
    def classify_scene(self, image_path, top_k=1):
        """分类图像场景类型"""
        try:
            start_time = time.time()
            
            # 打开图像
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path  # 假设已经是PIL图像
            
            # 预处理图像
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # 提取图像特征
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # 计算相似度
            similarity = torch.mm(image_features, self.scene_text_features.t())
            
            # 获取结果
            values, indices = similarity.topk(min(top_k, len(self.scene_types)))
            
            results = []
            for i in range(values.size(1)):
                score = values[0][i].item()
                scene_idx = indices[0][i].item()
                scene_type = self.scene_types[scene_idx]
                results.append((scene_type, score))
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            if top_k == 1:
                print(f"🏠 场景识别结果: {results[0][0]} (置信度: {results[0][1]:.3f}, 用时: {processing_time:.2f}秒)")
                return results[0][0], results[0][1]
            else:
                print(f"🏠 场景识别结果:")
                for scene, score in results:
                    print(f"   - {scene}: {score:.3f}")
                print(f"⏱️ 场景识别用时: {processing_time:.2f}秒")
                return results
                
        except Exception as e:
            import traceback
            print(f"❌ 场景分类失败: {e}")
            traceback.print_exc()
            return None, 0.0
    
    def get_scene_type(self, scene_name):
        """将场景名称映射到类型"""
        # 场景类型映射
        scene_mapping = {
            # 浴室相关
            "bathroom": "bathroom",
            "washroom": "bathroom",
            "toilet": "bathroom",
            "powder room": "bathroom",
            "shower room": "bathroom",
            
            # 卧室相关
            "bedroom": "bedroom",
            "master bedroom": "bedroom",
            "children room": "bedroom",
            "guest room": "bedroom",
            
            # 厨房相关
            "kitchen": "kitchen",
            "kitchenette": "kitchen",
            
            # 客厅相关
            "living room": "living_room",
            "lounge": "living_room",
            "family room": "living_room",
            "sitting room": "living_room",
            
            # 餐厅相关
            "dining room": "dining_room",
            
            # 办公室相关
            "office": "office",
            "study room": "office",
            "home office": "office",
            "computer room": "office",
            
            # 走廊相关
            "hallway": "hallway",
            "corridor": "hallway",
            "entrance": "hallway",
            
            # 洗衣房相关
            "laundry room": "laundry_room",
            "utility room": "laundry_room",
            
            # 人物相关
            "person": "person",
            "portrait": "person",
            "selfie": "person",
            "people": "person",
            "human": "person",
            
            # 食物相关（新增）
            "food": "food",
            "meal": "food",
            "dish": "food",
            "cuisine": "food",
            "plate": "food",
            "dinner": "food",
            "lunch": "food",
            "breakfast": "food",
            "restaurant": "food",
            "cafe": "food",
            "tableware": "food",
            
            # 动物相关场景
            "cat": "animal",
            "dog": "animal",
            "pet": "animal",
            "animal": "animal",
            "kitten": "animal",
            "puppy": "animal",
            "kitty": "animal",
            "feline": "animal",
            "canine": "animal",
        }
        
        # 转换为小写并尝试匹配
        scene_lower = scene_name.lower().strip()
        if scene_lower in scene_mapping:
            return scene_mapping[scene_lower]
        
        # 部分匹配
        for key, value in scene_mapping.items():
            if key in scene_lower or scene_lower in key:
                return value
        
        # 默认返回None表示未知场景
        return None
