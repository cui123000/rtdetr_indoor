# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 模型定义
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class ViLDModel(nn.Module):
    """ViLD模型：结合CLIP和RT-DETR"""
    
    def __init__(self, clip_model, detector_model):
        super().__init__()
        self.clip_model = clip_model
        self.detector_model = detector_model
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 特征融合层
        self.fusion_layer = nn.Linear(512, 256)  # CLIP输出512维，检测器特征256维
        
        # 多尺度特征投影器
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, 512)
            ) for _ in range(4)  # 对应RT-DETR的4个特征尺度
        ])
        
    def forward(self, images, image_processor):
        """前向传播"""
        # 使用检测器获取区域特征
        detector_inputs = image_processor(images=images, return_tensors="pt").to(next(self.parameters()).device)
        detector_outputs = self.detector_model(**detector_inputs, output_hidden_states=True)
        
        # 获取多尺度特征（取最后4层的[CLS] token）
        features = [h[:, 0] for h in detector_outputs.hidden_states[-4:]]
        
        # 投影特征
        projected_features = [proj(feat) for proj, feat in zip(self.projectors, features)]
        
        # 使用CLIP获取全局特征
        device = next(self.parameters()).device
        clip_inputs = torch.stack(images).to(device)
        clip_features = self.clip_model.encode_image(clip_inputs)
        
        # 特征融合
        fused_features = self.fusion_layer(clip_features)
        
        return {
            "detector_outputs": detector_outputs,
            "clip_features": clip_features,
            "fused_features": fused_features,
            "projected_features": projected_features
        }

def load_models(rtdetr_path=None, clip_name='ViT-B/32', device="cuda"):
    """加载所需的模型
    
    参数:
        rtdetr_path: RT-DETR模型路径，如果为None则从Hugging Face加载
        clip_name: CLIP模型名称
        device: 设备名称
        
    返回:
        rtdetr_model: RT-DETR检测模型
        clip_model: CLIP模型
        image_processor: RT-DETR图像处理器
        clip_preprocess: CLIP图像预处理器
    """
    import clip
    import traceback
    
    try:
        # 加载CLIP模型
        clip_model, clip_preprocess = clip.load(clip_name, device)
        clip_model.eval()
        print(f"✅ 成功加载CLIP模型: {clip_name}")
        
        # 加载RT-DETR检测器
        try:
            if rtdetr_path and os.path.exists(rtdetr_path):
                # 从本地加载
                print(f"正在从本地加载RT-DETR: {rtdetr_path}")
                rtdetr_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rtdetr_path)
                image_processor = None  # YOLOv5不需要单独的处理器
            else:
                # 从Hugging Face加载
                print(f"正在从Hugging Face加载RT-DETR模型")
                image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
                rtdetr_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)
            
            rtdetr_model.eval()
            print("✅ 成功加载RT-DETR模型")
        except Exception as e:
            print(f"⚠️ RT-DETR加载失败: {e}")
            print("将使用简化的网格区域作为候选框")
            rtdetr_model = None
            image_processor = None
        
        return rtdetr_model, clip_model, image_processor, clip_preprocess
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return None, None, None, None
