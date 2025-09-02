# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 检测器模块
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import cv2
from PIL import Image
import time
import traceback

class FixedViLDDetector:
    """优化版ViLD检测器 - 支持从训练好的模型加载"""
    
    def __init__(self, clip_model, detector_model=None, image_processor=None, clip_preprocess=None, device="cuda", projector_path=None, config=None):
        self.clip_model = clip_model
        self.detector_model = detector_model
        self.image_processor = image_processor
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # 加载配置（如果提供）
        if config is None:
            try:
                from config import MODEL_CONFIG, INFERENCE_CONFIG, MACRO_CATEGORIES
                config = {
                    'model': MODEL_CONFIG,
                    'inference': INFERENCE_CONFIG
                }
                self.macro_categories = MACRO_CATEGORIES
                print(f"✅ 已从配置文件加载参数和大类别映射")
            except ImportError:
                config = {}
                self.macro_categories = {}
                print(f"⚠️ 无法导入配置文件，使用默认参数")
        else:
            try:
                from config import MACRO_CATEGORIES
                self.macro_categories = MACRO_CATEGORIES
            except ImportError:
                self.macro_categories = {}
        
        # 默认使用大类别模式
        self.use_macro_categories = True
        
        # 创建简化版投影器
        self.visual_projector = self.create_identity_projector()
        self.text_projector = self.create_identity_projector()
        
        # 如果提供了模型路径，加载保存的投影器
        if projector_path is None and 'model' in config and 'projector_path' in config['model']:
            projector_path = config['model']['projector_path']
            
        if projector_path and os.path.exists(projector_path):
            try:
                print(f"📥 正在加载模型投影器: {projector_path}")
                # 在PyTorch 2.6中，需要显式设置weights_only=False
                checkpoint = torch.load(projector_path, map_location=self.device, weights_only=False)
                self.visual_projector.load_state_dict(checkpoint['visual_projector'])
                self.text_projector.load_state_dict(checkpoint['text_projector'])
                print(f"✅ 成功加载投影器模型 (Epoch {checkpoint['epoch']+1}, 验证损失: {checkpoint['val_loss']:.6f})")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print(f"🔄 尝试备用加载方法...")
                try:
                    # 尝试使用PyTorch安全上下文来加载模型
                    import numpy as np
                    # 添加numpy标量类型到安全全局变量
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                    checkpoint = torch.load(projector_path, map_location=self.device)
                    self.visual_projector.load_state_dict(checkpoint['visual_projector'])
                    self.text_projector.load_state_dict(checkpoint['text_projector'])
                    print(f"✅ 成功通过备用方法加载模型")
                except Exception as e2:
                    print(f"❌ 备用加载方法也失败: {e2}")
                    print(f"⚠️ 将使用默认投影器")
    
        # 设置为评估模式
        self.visual_projector.eval()
        self.text_projector.eval()
        
        # 检测参数 - 优先使用配置文件中的参数
        if 'model' in config and 'vild_similarity_threshold' in config['model']:
            self.similarity_threshold = config['model']['vild_similarity_threshold']
        else:
            self.similarity_threshold = 0.25
            
        if 'inference' in config and 'score_threshold' in config['inference']:
            self.detection_threshold = config['inference']['score_threshold']
        else:
            self.detection_threshold = 0.05
            
        if 'inference' in config and 'max_detections' in config['inference']:
            self.max_detections = config['inference']['max_detections']
        else:
            self.max_detections = 15
            
        print(f"🔧 检测器参数: 相似度阈值={self.similarity_threshold:.2f}, 检测阈值={self.detection_threshold:.2f}")
        
        # 室内类别集合（基础类别）
        self.base_categories = [
            'chair', 'table', 'bed', 'sofa', 'lamp', 'cabinet', 'door', 'window',
            'mirror', 'picture', 'book', 'bottle', 'cup', 'bowl', 'clock',
            'plant', 'television', 'refrigerator', 'microwave', 'toilet', 'sink',
            'towel', 'pillow', 'curtains', 'rug', 'shower', 'bathtub', 'shelf',
            'counter', 'desk', 'wardrobe', 'nightstand', 'computer', 'monitor',
            'glass', 'plate', 'tree', 'person', 'wine glass', 'fork', 'knife', 'spoon'
        ]
        
        # 使用基础类别初始化当前活动类别
        self.categories = self.base_categories.copy()
        
        # 场景特定类别（用于场景上下文优化）
        self.scene_categories = {
            'bathroom': ['toilet', 'sink', 'towel', 'bathtub', 'shower', 'mirror'],
            'kitchen': ['refrigerator', 'microwave', 'sink', 'cabinet', 'counter', 'table', 'bottle', 'cup', 'bowl'],
            'bedroom': ['bed', 'pillow', 'lamp', 'nightstand', 'wardrobe', 'mirror', 'clock'],
            'living_room': ['sofa', 'table', 'television', 'lamp', 'rug', 'curtains', 'picture'],
            'dining_room': ['table', 'chair', 'bottle', 'cup', 'glass', 'plate', 'fork', 'knife', 'spoon', 'bowl'],
            'outdoor': ['tree', 'chair', 'table', 'bottle', 'glass', 'cup', 'person', 'plant'],
            'person': ['person'],  # 人物场景主要识别人
            'food': ['bowl', 'plate', 'fork', 'knife', 'spoon', 'cup', 'glass', 'bottle', 'food']  # 食物场景
        }
        
        # 开放词汇支持
        self.clip_vocabulary = []
        self.custom_categories = []
        self.enable_open_vocabulary = True
        self.open_vocabulary_threshold = 0.35  # 提高阈值，减少误检测
        self.max_open_vocabulary_results = 3
        
        # 从CLIP加载大量词汇
        self._load_clip_vocabulary()
        
        print("🔧 ViLD检测器初始化完成")
    
    def create_identity_projector(self):
        """创建接近恒等映射的投影器"""
        projector = torch.nn.Sequential(
            torch.nn.Linear(512, 512, bias=True, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias=True, dtype=torch.float32)
        ).to(self.device)
        
        # 初始化为恒等映射
        with torch.no_grad():
            # 第一层：恒等映射
            torch.nn.init.eye_(projector[0].weight)
            if projector[0].bias is not None:
                torch.nn.init.zeros_(projector[0].bias)
            
            # 第三层：恒等映射
            torch.nn.init.eye_(projector[2].weight)
            if projector[2].bias is not None:
                torch.nn.init.zeros_(projector[2].bias)
            
            # 确保所有权重都是float32
            for param in projector.parameters():
                param.data = param.data.float()
        
        projector.eval()  # 设置为评估模式
        return projector
    
    def map_to_macro_category(self, label):
        """将细粒度标签映射到大类别"""
        # 查找标签所属的大类别
        for macro_cat, items in self.macro_categories.items():
            if label.lower() in [item.lower() for item in items]:
                return macro_cat
                
        # 如果找不到映射，使用原始标签
        return label
    
    def _load_clip_vocabulary(self):
        """加载CLIP词汇表"""
        # 常见室内物体的扩展词汇表
        extended_vocabulary = [
            # 家具类
            "armchair", "bench", "bookshelf", "bunk bed", "coffee table", "dining table",
            "dresser", "end table", "filing cabinet", "footstool", "futon", "loveseat",
            "ottoman", "recliner", "rocking chair", "sideboard", "stool", "tv stand",
            
            # 电器类
            "air conditioner", "blender", "coffee maker", "dishwasher", "electric fan", 
            "food processor", "hair dryer", "heater", "humidifier", "iron", "juicer",
            "kettle", "microwave oven", "mixer", "oven", "rice cooker", "toaster", 
            "vacuum cleaner", "washing machine", "water heater",
            
            # 卫浴类
            "bathroom cabinet", "bathroom mirror", "bathroom shelf", "bath mat",
            "faucet", "hand towel", "medicine cabinet", "shower curtain", "shower door",
            "shower head", "soap dish", "toilet brush", "toilet paper holder", "towel rack",
            
            # 食物类
            "food", "meal", "dish", "cuisine", "lunch", "dinner", "breakfast", 
            "appetizer", "entree", "dessert", "vegetable", "fruit", "meat", "beef",
            "chicken", "pork", "salad", "stew", "soup", "rice", "noodles", "pasta"
        ]
        
        # 加载基本类别和扩展词汇表
        self.clip_vocabulary = self.base_categories + extended_vocabulary
        print(f"✅ 加载了 {len(self.clip_vocabulary)} 个词汇项")
    
    def detect_objects(self, image_path, scene_type=None, custom_categories=None, enable_open_vocabulary=True, use_macro_categories=True):
        """检测图像中的物体"""
        try:
            start_time = time.time()
            
            # 打开图像
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                # 如果已经是PIL图像，直接使用
                image = image_path if hasattr(image_path, 'convert') else Image.fromarray(image_path)
                
            # 设置使用大类别的模式
            self.use_macro_categories = use_macro_categories
            print(f"🔍 检测模式: {'大类别分组' if self.use_macro_categories else '细粒度类别'}")
            
            # 处理自定义类别
            if custom_categories:
                self.set_custom_categories(custom_categories)
            
            # 设置开放词汇检测
            self.enable_open_vocabulary = enable_open_vocabulary
            
            # 1. 提取候选区域
            boxes, detection_scores = self.extract_regions(image)
            if len(boxes) == 0:
                print(f"❌ 没有找到候选区域")
                return {'boxes': [], 'scores': [], 'labels': []}
            
            print(f"📦 找到 {len(boxes)} 个候选区域")
            
            # 2. 提取视觉特征
            visual_features = self.extract_visual_features(image, boxes)
            if visual_features.size(0) == 0:
                return {'boxes': [], 'scores': [], 'labels': []}
            
            # 3. 编码文本特征
            text_features = self.encode_text_features()
            
            # 4. 计算相似度
            similarity_matrix = torch.mm(visual_features, text_features.t())
            
            # 场景优化
            if scene_type is not None:
                similarity_matrix = self.apply_scene_context(scene_type, similarity_matrix)
            
            max_similarities, best_category_indices = similarity_matrix.max(dim=1)
            
            # 5. 过滤
            # 动态阈值 - 更保守的阈值设置
            similarity_threshold = self.similarity_threshold
            if max_similarities.max() > 0.4:
                adaptive_threshold = max(max_similarities.max() * 0.65, self.similarity_threshold)
                similarity_threshold = min(adaptive_threshold, 0.4)
            
            # 对于户外/餐厅场景，使用更高的阈值筛选
            if scene_type in ["outdoor", "dining_room"]:
                similarity_threshold += 0.05  # 增加5%的阈值
                print(f"📊 场景特化: 为 {scene_type} 场景增加阈值至 {similarity_threshold:.2f}")
            
            valid_mask = max_similarities >= similarity_threshold
            valid_count = valid_mask.sum().item()
            
            # 如果没有匹配，降低阈值
            if valid_count == 0:
                low_threshold = 0.05
                valid_mask = max_similarities >= low_threshold
                valid_count = valid_mask.sum().item()
                
                if valid_count == 0 and not self.enable_open_vocabulary:
                    return {'boxes': [], 'scores': [], 'labels': []}
            
            # 处理基础类别检测
            if valid_count > 0:
                valid_boxes = boxes[:len(valid_mask)][valid_mask.cpu().numpy()]
                valid_detection_scores = detection_scores[:len(valid_mask)][valid_mask.cpu().numpy()]
                valid_similarities = max_similarities[valid_mask].cpu().numpy()
                valid_category_indices = best_category_indices[valid_mask].cpu().numpy()
                valid_labels = [self.categories[idx] for idx in valid_category_indices]
                
                # 组合分数
                combined_scores = valid_detection_scores * 0.3 + valid_similarities * 0.7
                
                # 排序
                sorted_indices = np.argsort(combined_scores)[::-1][:self.max_detections]
                
                final_boxes = valid_boxes[sorted_indices]
                final_scores = combined_scores[sorted_indices]
                final_labels = [valid_labels[i] for i in sorted_indices]
                
                # 如果使用大类别，进行映射
                if self.use_macro_categories and self.macro_categories:
                    final_labels = [self.map_to_macro_category(label) for label in final_labels]
                    print(f"✓ 已将检测结果映射到大类别")
                
                result = {
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels,
                    'open_vocab_results': {}
                }
            else:
                result = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': [],
                    'open_vocab_results': {}
                }
            
            # 开放词汇检测
            if self.enable_open_vocabulary:
                open_vocab_results = self.perform_open_vocabulary_detection(
                    visual_features, boxes, detection_scores
                )
                
                if open_vocab_results:
                    result['open_vocab_results'] = open_vocab_results
                    
                    if len(result['boxes']) == 0 and len(open_vocab_results['boxes']) > 0:
                        result['boxes'] = open_vocab_results['boxes']
                        result['scores'] = open_vocab_results['scores']
                        result['labels'] = open_vocab_results['labels']
            
            # 计算检测时间
            detection_time = time.time() - start_time
            result['detection_time'] = detection_time
            
            print(f"⏱️ 检测完成，用时: {detection_time:.2f}秒")
            return result
            
        except Exception as e:
            import traceback
            print(f"❌ 检测失败: {e}")
            traceback.print_exc()
            return {'boxes': [], 'scores': [], 'labels': []}
    
    def extract_regions(self, image):
        """提取候选区域"""
        if self.image_processor is None or self.detector_model is None:
            # 如果没有检测器，使用简单的网格区域
            print("⚠️ 没有检测器模型，使用网格区域")
            width, height = image.size
            boxes = []
            scores = []
            
            # 创建3x3网格
            for i in range(3):
                for j in range(3):
                    x1 = j * width // 3
                    y1 = i * height // 3
                    x2 = (j + 1) * width // 3
                    y2 = (i + 1) * height // 3
                    boxes.append([x1, y1, x2, y2])
                    scores.append(0.9)  # 使用较高的置信度
            
            # 添加整个图像
            boxes.append([0, 0, width, height])
            scores.append(1.0)
            
            return np.array(boxes), np.array(scores)
        
        # 使用RT-DETR模型提取区域
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.detection_threshold
        )[0]
        
        return results['boxes'].cpu().numpy(), results['scores'].cpu().numpy()
    
    def extract_visual_features(self, image, boxes):
        """提取视觉特征"""
        if len(boxes) == 0:
            return torch.empty(0, 512).to(self.device)
        
        features = []
        img_array = np.array(image)
        max_regions = min(len(boxes), 50)  # 限制处理数量
        
        for i, box in enumerate(boxes[:max_regions]):
            try:
                x1, y1, x2, y2 = box.astype(int)
                
                # 边界检查
                x1 = max(0, min(x1, img_array.shape[1]-1))
                y1 = max(0, min(y1, img_array.shape[0]-1))
                x2 = max(x1+1, min(x2, img_array.shape[1]))
                y2 = max(y1+1, min(y2, img_array.shape[0]))
                
                # 提取区域
                region = img_array[y1:y2, x1:x2]
                region_image = Image.fromarray(region)
                
                # 使用CLIP处理
                region_input = self.clip_preprocess(region_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    region_feature = self.clip_model.encode_image(region_input).float()
                    projected_feature = self.visual_projector(region_feature)
                    normalized_feature = F.normalize(projected_feature, p=2, dim=1)
                    features.append(normalized_feature)
                    
            except Exception as e:
                print(f"⚠️ 区域特征提取错误 (box={box}): {e}")
                # 跳过问题区域
                continue
        
        if not features:
            return torch.empty(0, 512).to(self.device)
        
        return torch.cat(features, dim=0)
    
    def encode_text_features(self):
        """编码文本特征"""
        all_text_features = []
        templates = ["a {}", "indoor {}", "a {} in a room"]
        
        for category in self.categories:
            category_features = []
            
            for template in templates:
                text = template.format(category)
                text_tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens).float()
                    projected_text = self.text_projector(text_features)
                    normalized_feature = F.normalize(projected_text, p=2, dim=1)
                    category_features.append(normalized_feature)
            
            # 平均多个模板的特征
            if category_features:
                avg_features = torch.stack(category_features).mean(dim=0)
                all_text_features.append(avg_features)
        
        if all_text_features:
            return torch.cat(all_text_features, dim=0)
        else:
            return torch.empty(0, 512, dtype=torch.float32).to(self.device)
    
    def set_custom_categories(self, categories):
        """设置用户自定义类别列表"""
        if not categories:
            return
            
        self.categories = self.base_categories.copy()
        self.custom_categories = [c for c in categories if c not in self.categories]
        self.categories.extend(self.custom_categories)
        
    def apply_scene_context(self, scene_type, similarity_matrix):
        """应用场景上下文优化"""
        # 如果是"dining room"，也尝试匹配"dining_room"
        if scene_type == "dining room":
            scene_type = "dining_room"
            
        # 如果是户外场景，使用户外优化
        if scene_type.lower() in ["outdoor", "garden", "patio", "yard", "terrace"]:
            scene_type = "outdoor"
            
        # 如果是人物场景
        if scene_type.lower() in ["person", "portrait", "selfie", "people", "human"]:
            scene_type = "person"
        
        if scene_type not in self.scene_categories:
            print(f"⚠️ 未找到场景类型 '{scene_type}' 的特定优化，使用通用检测")
            return similarity_matrix
            
        # 获取场景相关类别
        relevant_categories = self.scene_categories[scene_type]
        relevant_indices = [i for i, cat in enumerate(self.categories) if cat in relevant_categories]
        
        # 修改相似度分数
        modified_matrix = similarity_matrix.clone()
        boost_factor = 0.20  # 提高到20%提升
        
        # 人物场景特殊处理 - 更高的提升因子
        if scene_type == "person":
            boost_factor = 0.40  # 对人物的检测提升40%
            
            # 对person类别进行强化
            person_indices = [i for i, cat in enumerate(self.categories) if cat == "person"]
            for i in range(similarity_matrix.size(0)):
                for idx in person_indices:
                    modified_matrix[i, idx] *= (1 + boost_factor)
            
            # 强烈抑制不太可能在人物肖像中出现的物体
            highly_unlikely_categories = ['toilet', 'bathtub', 'shower', 'refrigerator', 
                                         'microwave', 'oven', 'sink', 'bed']
                                         
            # 不那么强烈地抑制可能错误检测的物体
            unlikely_categories = ['chair', 'table', 'cabinet', 'sofa']
            
            # 获取高度不可能的类别索引
            highly_unlikely_indices = [i for i, cat in enumerate(self.categories) 
                                      if cat in highly_unlikely_categories]
            
            # 获取不太可能的类别索引
            unlikely_indices = [i for i, cat in enumerate(self.categories) 
                               if cat in unlikely_categories]
            
            # 应用强抑制
            for i in range(similarity_matrix.size(0)):
                for idx in highly_unlikely_indices:
                    modified_matrix[i, idx] *= 0.3  # 降低70%
                
                for idx in unlikely_indices:
                    modified_matrix[i, idx] *= 0.5  # 降低50%
                    
            print(f"✅ 已应用人物场景特殊优化: 人物 +{boost_factor*100:.0f}%, 抑制不相关物体")
            return modified_matrix
            
        # 食物场景特殊处理
        if scene_type == "food":
            boost_factor = 0.40  # 对食物相关类别提升40%
            
            # 食物相关类别
            food_categories = ['bowl', 'plate', 'fork', 'knife', 'spoon', 'food']
            if self.use_macro_categories:
                food_categories.extend(['tableware', 'food'])
                
            # 加强食物相关类别
            food_indices = [i for i, cat in enumerate(self.categories) 
                            if any(food_cat in cat.lower() for food_cat in food_categories)]
            
            for i in range(similarity_matrix.size(0)):
                for idx in food_indices:
                    modified_matrix[i, idx] *= (1 + boost_factor)
            
            # 强烈抑制不太可能在食物场景中出现的物体
            highly_unlikely_categories = ['toilet', 'bathtub', 'shower', 'bed', 'person']
            
            # 获取高度不可能的类别索引
            highly_unlikely_indices = [i for i, cat in enumerate(self.categories) 
                                      if cat in highly_unlikely_categories]
            
            # 应用强抑制
            for i in range(similarity_matrix.size(0)):
                for idx in highly_unlikely_indices:
                    modified_matrix[i, idx] *= 0.3  # 降低70%
                    
            print(f"✅ 已应用食物场景特殊优化: 食物相关物品 +{boost_factor*100:.0f}%, 抑制不相关物体")
            return modified_matrix
        
        # 其他场景的处理
        for i in range(similarity_matrix.size(0)):
            for idx in relevant_indices:
                modified_matrix[i, idx] *= (1 + boost_factor)
                
        # 不相关类别降低分数 - 更强的惩罚
        highly_unlikely_categories = []
        if scene_type == "outdoor":
            # 户外不太可能出现的物体
            highly_unlikely_categories = ['toilet', 'bathtub', 'shower', 'refrigerator', 'microwave', 
                                         'wardrobe', 'curtains', 'bed', 'nightstand']
        elif scene_type == "dining_room":
            # 餐厅不太可能出现的物体
            highly_unlikely_categories = ['toilet', 'bathtub', 'shower', 'bed', 'pillow']
            
        # 获取高度不可能的类别索引
        highly_unlikely_indices = [i for i, cat in enumerate(self.categories) if cat in highly_unlikely_categories]
        
        # 常规不相关类别
        non_relevant_indices = [i for i, cat in enumerate(self.categories) 
                              if cat not in relevant_categories and cat not in highly_unlikely_categories]
        
        # 常规不相关类别的小惩罚
        penalty_factor = 0.10  # 提高到10%惩罚
        for i in range(similarity_matrix.size(0)):
            for idx in non_relevant_indices:
                modified_matrix[i, idx] *= (1 - penalty_factor)
        
        # 高度不可能类别的强惩罚
        strong_penalty = 0.50  # 50%惩罚
        for i in range(similarity_matrix.size(0)):
            for idx in highly_unlikely_indices:
                modified_matrix[i, idx] *= (1 - strong_penalty)
                
        print(f"✅ 已应用 '{scene_type}' 场景优化: 相关物体 +{boost_factor*100:.0f}%, 不相关物体 -{penalty_factor*100:.0f}%, 极不可能物体 -{strong_penalty*100:.0f}%")
        return modified_matrix
    
    def perform_open_vocabulary_detection(self, visual_features, boxes, detection_scores):
        """执行开放词汇检测"""
        try:
            if not self.clip_vocabulary:
                return {}
                
            open_vocab_results = {
                'boxes': [],
                'scores': [],
                'labels': [],
                'alternative_labels': []
            }
            
            # 批量处理词汇
            batch_size = 200
            all_text_features = []
            
            for i in range(0, len(self.clip_vocabulary), batch_size):
                batch = self.clip_vocabulary[i:i+batch_size]
                
                texts = [f"a {word}" for word in batch]
                text_tokens = clip.tokenize(texts).to(self.device)
                
                with torch.no_grad():
                    batch_text_features = self.clip_model.encode_text(text_tokens).float()
                    batch_text_features = self.text_projector(batch_text_features)
                    batch_text_features = F.normalize(batch_text_features, p=2, dim=1)
                    all_text_features.append(batch_text_features)
            
            text_features = torch.cat(all_text_features, dim=0)
            
            # 计算相似度
            similarity_matrix = torch.mm(visual_features, text_features.t())
            
            # 找到最佳匹配
            for i in range(similarity_matrix.size(0)):
                # 获取前K个最佳匹配
                similarities, indices = torch.topk(similarity_matrix[i], k=self.max_open_vocabulary_results)
                
                # 检查相似度是否高于阈值
                if similarities[0] >= self.open_vocabulary_threshold:
                    # 最佳匹配作为主标签
                    best_idx = indices[0].item()
                    best_score = similarities[0].item()
                    best_label = self.clip_vocabulary[best_idx]
                    
                    # 其他候选项
                    alt_indices = indices[1:].cpu().numpy()
                    alt_scores = similarities[1:].cpu().numpy()
                    alt_labels = [(self.clip_vocabulary[idx], score) for idx, score in zip(alt_indices, alt_scores)]
                    
                    # 添加结果
                    open_vocab_results['boxes'].append(boxes[i])
                    open_vocab_results['scores'].append(best_score)
                    open_vocab_results['labels'].append(best_label)
                    open_vocab_results['alternative_labels'].append(alt_labels)
                    
            # 转换为numpy数组
            if open_vocab_results['boxes']:
                open_vocab_results['boxes'] = np.array(open_vocab_results['boxes'])
                open_vocab_results['scores'] = np.array(open_vocab_results['scores'])
                
                # 保留最佳结果
                if len(open_vocab_results['boxes']) > self.max_detections:
                    # 排序
                    sorted_indices = np.argsort(open_vocab_results['scores'])[::-1][:self.max_detections]
                    open_vocab_results['boxes'] = open_vocab_results['boxes'][sorted_indices]
                    open_vocab_results['scores'] = open_vocab_results['scores'][sorted_indices]
                    open_vocab_results['labels'] = [open_vocab_results['labels'][i] for i in sorted_indices]
                    open_vocab_results['alternative_labels'] = [open_vocab_results['alternative_labels'][i] for i in sorted_indices]
                    
            return open_vocab_results
            
        except Exception as e:
            print(f"❌ 开放词汇检测失败: {e}")
            return {}
