#!/usr/bin/env python3
"""
COCO 室内子数据集训练脚本
使用筛选后的 COCO 室内场景数据（阈值30：3,015 train / 151 val）
支持 RT-DETR 模型训练
过滤条件：对象数 > 30，保留75.4%数据，优化显存使用
"""

import os
import sys
import torch
import gc
from pathlib import Path
import argparse
import time

# 全局模型设置 - 在这里修改要训练的模型
GLOBAL_MODEL = 'rtdetr-l'  # 可选: rtdetr-l, rtdetr-mnv4, rtdetr-mnv4-sea

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def setup_environment():
    """设置训练环境"""
    print("🔧 配置训练环境...")
    
    # CUDA内存管理 - 最优策略：允许扩展但控制碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.7,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # PyTorch优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置线程数
    torch.set_num_threads(12)  # 匹配workers数量
    os.environ['OMP_NUM_THREADS'] = '12'
    os.environ['MKL_NUM_THREADS'] = '12'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # 不限制显存，让PyTorch自动管理（你的GPU显存充足）
        
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA: {torch.version.cuda}")
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ 显存管理: 自动（24GB RTX 4090）")
        print(f"   ✓ Tensor Cores: 已启用 (AMP + TF32)")
    else:
        print("   ⚠️  警告: 未检测到GPU，将使用CPU训练（速度会很慢）")

def get_model_config(model_name):
    """
    获取模型配置
    
    参数:
        model_name: 模型名称
            - rtdetr-l: RT-DETR-L 官方模型
            - rtdetr-mnv4: RT-DETR with MobileNetV4 backbone
            - rtdetr-mnv4-sea: RT-DETR with MobileNetV4 + SEA attention
    """
    configs = {
        'rtdetr-l': {
            'pretrained': None,  # ❌ 不使用预训练权重，从零训练
            'config_file': 'rtdetr-l.yaml',  # 使用 YAML 配置文件
            'batch': 4,  # 降低到4以适配阈值30数据集（21.8G -> 预计15G）
            'lr0': 0.0015,
            'name': 'rtdetr_l_coco_indoor_scratch',  # 修改名称标识从零训练
        },
        'rtdetr-mnv4': {
            'pretrained': None,  # ❌ 不使用预训练权重，从零训练
            'config_file': 'rtdetr-mnv4-hybrid-m.yaml',
            'batch': 4,  # 降低到4以适配COCO室内高密度数据（阈值30，21.3G→预计14G）
            'lr0': 0.0018,
            'name': 'rtdetr_mnv4_coco_indoor_scratch',  # 修改名称标识从零训练
        },
        'rtdetr-mnv4-sea': {
            'pretrained': None,
            'config_file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
            'batch': 4,  # 参考variants
            'lr0': 0.0015,
            'name': 'rtdetr_mnv4_sea_coco_indoor',
        },
    }
    
    if model_name not in configs:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选模型: {', '.join(configs.keys())}"
        )
    
    return configs[model_name]

def get_training_config(model_cfg, args):
    """获取训练配置"""
    
    # 数据集配置 - 
    data_path = '/home/cui/rtdetr_indoor/datasets/coco_indoor_4k/coco_indoor_4k.yaml'
    
    # 项目输出目录（使用绝对路径）
    if not args.project.startswith('/'):
        project_path = str(project_root / args.project)
    else:
        project_path = args.project
    
    config = {
        'task': 'detect',
        'mode': 'train',
        'data': data_path,
        
        # 基础训练参数
        'epochs': args.epochs,
        'batch': args.batch if args.batch > 0 else model_cfg['batch'],
        'imgsz': args.imgsz,
        'patience': 20,
        
        # 设备设置 - 参考 train_mnv4_variants.py 的成功配置
        'device': args.device,
        'workers': 4,  # 和variants保持一致
        'amp': True,
        'cache': 'ram',  # 重新启用RAM缓存（variants用的）
        'rect': True,  # 🔥 关键！矩形训练大幅节省显存
        'single_cls': False,
        
        # 优化器 - 参考 train_mnv4_variants.py 的配置
        'optimizer': 'AdamW',
        'lr0': model_cfg['lr0'],
        'lrf': 0.0015,  # 参考 variants
        'momentum': 0.94,  # 参考 variants
        'weight_decay': 0.00045,  # 参考 variants
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,
        
        # 数据增强 - 参考 train_mnv4_variants.py（关闭 mosaic）
        'hsv_h': 0.015,
        'hsv_s': 0.65,  # 参考 variants
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.0,  # 完全关闭以节省显存
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # 损失权重 - 参考 train_mnv4_variants.py
        'box': 7.5,
        'cls': 0.55,  # 参考 variants
        'dfl': 1.5,
        
        # 验证 - 参考variants配置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 400,  # 恢复到variants的值
        
        # 保存
        'save': True,
        'save_period': args.save_period,
        'project': project_path,
        'name': model_cfg['name'],
        'exist_ok': True,
        
        # 其他
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'close_mosaic': 10,
    }
    
    return config

def train(args):
    """执行训练"""
    setup_environment()
    
    # 使用全局模型或命令行参数
    model_name = args.model if args.model else GLOBAL_MODEL
    
    # 获取模型配置
    model_cfg = get_model_config(model_name)
    
    print("\n" + "="*70)
    print(f"🎯 训练配置")
    print("="*70)
    print(f"模型: {model_name}")
    print(f"模型名称: {model_cfg['name']}")
    print(f"训练方式: 从零训练 (不使用预训练权重)")
    print(f"数据集: COCO 室内过滤版 (3,015 train / 151 val)")
    print(f"过滤条件: 对象数 ≤ 30 (移除1,034张高密度图，保留75.4%)")
    print(f"平均对象数: ~20个/图 (vs 原始25.7个)")
    print(f"批次大小: {args.batch if args.batch > 0 else model_cfg['batch']}")
    print(f"训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"学习率: {model_cfg['lr0']}")
    print("="*70 + "\n")
    
    # 导入 RT-DETR
    from ultralytics import RTDETR
    
    # 判断是使用预训练模型还是配置文件
    if model_cfg['pretrained']:
        print(f"📦 加载预训练模型: {model_cfg['pretrained']}")
        model = RTDETR(model_cfg['pretrained'])
    elif model_cfg['config_file']:
        config_path = project_root / 'ultralytics' / 'ultralytics' / 'cfg' / 'models' / 'rt-detr' / model_cfg['config_file']
        if not config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
        print(f"📄 使用配置文件: {model_cfg['config_file']}")
        model = RTDETR(str(config_path))
    else:
        raise ValueError("模型配置错误：必须指定 pretrained 或 config_file")
    
    # 获取训练配置
    train_config = get_training_config(model_cfg, args)
    
    # 添加显存清理回调（Ultralytics正确方式）
    batch_counter = {'count': 0}  # 使用字典保持可变引用
    
    def on_train_batch_end(trainer):
        """每100个batch清理一次显存"""
        batch_counter['count'] += 1
        if batch_counter['count'] % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_train_epoch_end(trainer):
        """每个epoch结束清理显存，为验证腾出空间"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def on_val_start(trainer):
        """验证开始前强制清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    model.add_callback('on_train_batch_end', on_train_batch_end)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_val_start', on_val_start)
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚠️  从零训练模式: 无预训练权重，纯粹学习")
    print(f"✅ 使用成功配置: rect=True, cache=ram, workers=4")
    print(f"🎯 数据集优化: 过滤了1,034张高密度图(>30对象)")
    print(f"💾 预期显存: 10-14G (batch={args.batch if args.batch > 0 else model_cfg['batch']})")
    print(f"⚡ 预期速度: ~2-3 it/s")
    print(f"🔄 三重清理: epoch结束、验证开始、每100 batch")
    print(f"📊 公平对比: L 和 MNV4 都从零训练，无预训练优势\n")
    
    try:
        results = model.train(**train_config)
        
        print("\n" + "="*70)
        print("✅ 训练完成！")
        print("="*70)
        print(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 输出目录: {train_config['project']}/{train_config['name']}")
        print("="*70)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='COCO 室内子数据集训练脚本 (RT-DETR)')
    
    # 模型选择（可选，默认使用全局设置）
    parser.add_argument('--model', type=str, default=None,
                       help=f'模型名称 (默认: {GLOBAL_MODEL}), 可选: rtdetr-l, rtdetr-mnv4, rtdetr-mnv4-sea')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=-1,
                       help='批次大小 (-1 使用默认值)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                       help='训练设备 (如: 0, 0,1, cpu)')
    
    # 保存设置
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='项目保存目录')
    parser.add_argument('--save-period', type=int, default=10,
                       help='保存checkpoint的间隔轮数')
    
    args = parser.parse_args()
    
    # 执行训练
    train(args)

if __name__ == '__main__':
    main()
