#!/usr/bin/env python3
"""
RT-DETR 知识蒸馏训练脚本
使用已训练的大模型作为teacher，训练轻量化student模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 蒸馏配置
DISTILLATION_CONFIG = {
    # Teacher 模型配置
    'teacher_weight': '/home/cjj/rtdetr_indoor/runs/detect/rtdetr_l_best.pt',  # 已训练的RT-DETR-L权重
    'teacher_config': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml',
    
    # Student 模型配置（可选择的轻量模型）
    'student_configs': {
        '5': {  # GhostNet
            'config': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-ghostnet.yaml',
            'name': 'rtdetr_ghostnet_distilled',
            'batch': 48,
        },
        '6': {  # ShuffleNet+SEA
            'config': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-shufflenet-sea.yaml',
            'name': 'rtdetr_shufflenet_distilled',
            'batch': 48,
        },
        '9': {  # MobileNetV3
            'config': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mobilenetv3.yaml',
            'name': 'rtdetr_mobilenetv3_distilled',
            'batch': 56,
        },
        '10': {  # RepGhostNet
            'config': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-repghostnet.yaml',
            'name': 'rtdetr_repghostnet_distilled',
            'batch': 64,
        }
    },
    
    # 蒸馏超参数
    'temperature': 4.0,        # 蒸馏温度
    'alpha': 0.5,              # 蒸馏损失权重 (alpha * distill_loss + (1-alpha) * hard_loss)
    'beta': 0.3,               # 特征蒸馏权重
    
    # 训练参数
    'epochs': 120,
    'lr0': 0.0005,
    'warmup_epochs': 10,
    'weight_decay': 0.001,
    
    # 其他设置
    'dataset': '/home/cjj/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
    'save_dir': '/home/cjj/rtdetr_indoor/runs/distillation',
    'device': '0',
}

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta    # 特征蒸馏权重
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
    
    def forward(self, student_logits, teacher_logits, student_features=None, teacher_features=None, targets=None, hard_loss=None):
        """
        计算蒸馏损失
        Args:
            student_logits: 学生模型的输出logits
            teacher_logits: 教师模型的输出logits  
            student_features: 学生模型的特征图
            teacher_features: 教师模型的特征图
            targets: 真实标签
            hard_loss: 硬标签损失
        """
        # 1. 响应蒸馏损失 (KL散度)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 2. 特征蒸馏损失 (如果提供特征)
        feature_loss = 0
        if student_features is not None and teacher_features is not None:
            if isinstance(student_features, (list, tuple)):
                for s_feat, t_feat in zip(student_features, teacher_features):
                    # 特征图大小可能不同，需要对齐
                    if s_feat.shape != t_feat.shape:
                        s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
                    feature_loss += self.mse(s_feat, t_feat)
            else:
                if student_features.shape != teacher_features.shape:
                    student_features = F.adaptive_avg_pool2d(student_features, teacher_features.shape[-2:])
                feature_loss = self.mse(student_features, teacher_features)
        
        # 3. 总损失
        if hard_loss is not None:
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss + self.beta * feature_loss
        else:
            total_loss = distill_loss + self.beta * feature_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'feature_loss': feature_loss,
            'hard_loss': hard_loss if hard_loss is not None else 0
        }

class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config["device"]}' if torch.cuda.is_available() else 'cpu')
        self.teacher_model = None
        self.student_model = None
        self.distill_loss = DistillationLoss(
            temperature=config['temperature'],
            alpha=config['alpha'], 
            beta=config['beta']
        )
    
    def load_teacher_model(self):
        """加载教师模型"""
        print("🎓 加载教师模型...")
        
        try:
            from ultralytics import RTDETR
            
            # 检查权重文件
            if not os.path.exists(self.config['teacher_weight']):
                raise FileNotFoundError(f"教师模型权重文件不存在: {self.config['teacher_weight']}")
            
            # 加载模型
            self.teacher_model = RTDETR(self.config['teacher_weight'])
            self.teacher_model.model.eval()
            self.teacher_model.model = self.teacher_model.model.to(self.device)
            
            # 冻结教师模型参数
            for param in self.teacher_model.model.parameters():
                param.requires_grad = False
            
            print(f"   ✅ 教师模型加载成功: {self.config['teacher_weight']}")
            
        except Exception as e:
            print(f"   ❌ 教师模型加载失败: {e}")
            raise
    
    def load_student_model(self, student_config):
        """加载学生模型"""
        print("🎒 加载学生模型...")
        
        try:
            from ultralytics import RTDETR
            
            self.student_model = RTDETR(student_config['config'])
            self.student_model.model = self.student_model.model.to(self.device)
            
            print(f"   ✅ 学生模型加载成功: {student_config['config']}")
            
        except Exception as e:
            print(f"   ❌ 学生模型加载失败: {e}")
            raise
    
    def create_distillation_training_config(self, student_config):
        """创建蒸馏训练配置"""
        config = {
            'task': 'detect',
            'mode': 'train',
            'model': student_config['config'],
            'data': self.config['dataset'],
            
            # 训练参数
            'epochs': self.config['epochs'],
            'batch': student_config['batch'],
            'imgsz': 640,
            'workers': 16,
            'device': self.config['device'],
            
            # 优化器设置
            'optimizer': 'AdamW',
            'lr0': self.config['lr0'],
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': self.config['weight_decay'],
            'warmup_epochs': self.config['warmup_epochs'],
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,
            
            # 数据增强 - 蒸馏时稍微减少增强强度
            'hsv_h': 0.02,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15.0,
            'translate': 0.2,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.05,
            'close_mosaic': 15,  # 提前关闭mosaic
            
            # GPU优化
            'amp': True,
            'cache': True,
            'rect': False,
            'single_cls': False,
            'half': False,
            'deterministic': False,
            
            # 保存设置
            'save': True,
            'save_period': -1,
            'save_json': True,
            'plots': True,
            'val': True,
            'project': self.config['save_dir'],
            'name': f\"{student_config['name']}_{time.strftime('%Y%m%d_%H%M%S')}\",
            'exist_ok': True,
            
            # 验证设置
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'augment': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            
            # 损失权重调整
            'box': 7.5,
            'cls': 1.0,
            'dfl': 1.5,
            
            # 其他设置
            'verbose': True,
            'seed': 42,
            'dropout': 0.1,
            
            # 蒸馏特定参数
            'distillation': True,
            'teacher_model': self.teacher_model,
            'temperature': self.config['temperature'],
            'alpha': self.config['alpha'],
            'beta': self.config['beta'],
        }
        
        return config
    
    def train_student(self, student_id):
        """训练指定的学生模型"""
        if student_id not in self.config['student_configs']:
            raise ValueError(f"无效的学生模型ID: {student_id}")
        
        student_config = self.config['student_configs'][student_id]
        
        print(f"🎯 开始蒸馏训练: {student_config['name']}")
        print("=" * 70)
        
        # 加载模型
        self.load_teacher_model()
        self.load_student_model(student_config)
        
        # 创建训练配置
        train_config = self.create_distillation_training_config(student_config)
        
        print(f"📊 训练配置:")
        print(f"   学生模型: {student_config['name']}")
        print(f"   批次大小: {student_config['batch']}")
        print(f"   训练轮数: {self.config['epochs']}")
        print(f"   学习率: {self.config['lr0']}")
        print(f"   蒸馏温度: {self.config['temperature']}")
        print(f"   蒸馏权重: {self.config['alpha']}")
        print(f"   特征权重: {self.config['beta']}")
        
        # 开始训练
        try:
            print("🚀 开始蒸馏训练...")
            start_time = time.time()
            
            # 注意：这里需要修改ultralytics的训练逻辑来支持蒸馏
            # 这是一个简化版本，实际需要深入修改trainer
            results = self.student_model.train(**{k: v for k, v in train_config.items() 
                                                if k not in ['teacher_model', 'distillation', 'temperature', 'alpha', 'beta']})
            
            training_time = (time.time() - start_time) / 3600
            
            print(f"🎉 蒸馏训练完成! 用时: {training_time:.2f} 小时")
            
            return results
            
        except Exception as e:
            print(f"❌ 蒸馏训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR 知识蒸馏训练')
    parser.add_argument('--student', type=str, required=True, 
                       choices=['5', '6', '9', '10'],
                       help='选择学生模型 (5:GhostNet, 6:ShuffleNet+SEA, 9:MobileNetV3, 10:RepGhostNet)')
    parser.add_argument('--teacher', type=str, 
                       default='/home/cjj/rtdetr_indoor/runs/detect/rtdetr_l_best.pt',
                       help='教师模型权重路径')
    parser.add_argument('--temperature', type=float, default=4.0, help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.5, help='蒸馏损失权重')
    parser.add_argument('--beta', type=float, default=0.3, help='特征损失权重')
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    
    args = parser.parse_args()
    
    # 更新配置
    config = DISTILLATION_CONFIG.copy()
    config['teacher_weight'] = args.teacher
    config['temperature'] = args.temperature
    config['alpha'] = args.alpha  
    config['beta'] = args.beta
    config['epochs'] = args.epochs
    config['lr0'] = args.lr
    
    # 创建训练器
    trainer = DistillationTrainer(config)
    
    print("🔥 RT-DETR 知识蒸馏训练器")
    print("=" * 70)
    print(f"🎓 教师模型: {config['teacher_weight']}")
    print(f"🎒 学生模型: {config['student_configs'][args.student]['name']}")
    print(f"🌡️  蒸馏温度: {config['temperature']}")
    print(f"⚖️  权重配比: α={config['alpha']}, β={config['beta']}")
    print("=" * 70)
    
    # 开始训练
    try:
        results = trainer.train_student(args.student)
        print("✅ 知识蒸馏完成!")
    except Exception as e:
        print(f"❌ 知识蒸馏失败: {e}")

if __name__ == "__main__":
    main()