#!/usr/bin/env python3
"""
为RTX 3080Ti优化的RT-DETR训练脚本
解决训练慢和内存泄漏问题 + 文件描述符问题
"""

import os
import sys
import yaml
import torch
import gc
from pathlib import Path
import threading
import time
import resource
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 全局设置训练模式版本和关机选项
DEFAULT_TRAIN_MODE =1  # 1: RT-DETR-L, 2: RT-DETR+MNV4, 3: RT-DETR+MNV4+SEA
SHUTDOWN_AFTER_TRAIN = True  # 设置为 True 表示训练完成后自动关机

# 修复文件描述符限制问题
def fix_file_descriptor_limit():
    print("🔧 修复文件描述符限制...")

    try:
        # 获取当前限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"   当前文件描述符限制: {soft} (软限制) / {hard} (硬限制)")

        # 设置软限制为 65536
        new_soft = min(65536, hard)  # 设置为65536或硬限制的较小值
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

        print(f"   ✅ 新的文件描述符限制: {new_soft}")

        # 设置环境变量限制 workers
        os.environ['TORCH_NUM_WORKERS'] = '2'  # 强制限制 workers 数量

    except Exception as e:
        print(f"   ⚠️ 无法修改文件描述符限制: {e}")
        print("   💡 建议在系统级别增加文件描述符限制")

# 为RTX 3080Ti设置专门的优化
def setup_rtx3080ti_optimization():
    print("🚀 为RTX 3080Ti设置专门优化...")

    # 首先修复文件描述符问题
    fix_file_descriptor_limit()

    # RTX 3080Ti专用CUDA设置 - 保守配置避免驱动错误
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 移除 expandable_segments
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

    # 启用RTX 3080Ti的优化特性
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 设置合理的线程数
    torch.set_num_threads(4)  # 减少线程数避免资源竞争
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'

    if torch.cuda.is_available():
        # RTX 3080Ti显存较少，使用80%避免OOM
        torch.cuda.set_per_process_memory_fraction(0.8)

        # 清理初始缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ 显存限制: 80% (~10GB)")

        # 检查CUDA状态
        try:
            test_tensor = torch.randn(100, 100).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("   ✅ CUDA状态正常")
        except Exception as e:
            print(f"   ❌ CUDA测试失败: {e}")
            raise
        print(f"   ✅ TF32加速: 启用")

# RTX 3080Ti专用内存监控
def memory_monitor_rtx3080ti():
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                
                # RTX 3080Ti显存阈值较低
                if allocated > 9.0:  # 9GB以上时清理
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"🧹 自动清理GPU内存: {allocated:.1f}GB -> {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
            
            time.sleep(15)  # 每15秒检查一次
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

# RTX 3080Ti优化的训练配置
def get_rtx3080ti_config(model_choice):
    model_configs = {
        '1': {
            'file': 'rtdetr-l.yaml',
            'name': 'rtdetr_l_rtx3080ti',
            'batch': 8,       # 降低batch避免内存泄漏
            'lr0': 0.001,     # 相应调整学习率
            'workers': 8,     # 增加数据加载器线程数
        },
        '2': {
            'file': 'rtdetr-mnv4-hybrid-m.yaml', 
            'name': 'rtdetr_mnv4_hybrid_rtx3080ti',
            'batch': 8,       # 增加批次大小
            'lr0': 0.0008,
            'workers': 8,     # 增加数据加载器线程数
        },
        '3': {
            'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
            'name': 'rtdetr_mnv4_sea_rtx3080ti',
            'batch': 8,       # 增加批次大小
            'lr0': 0.0006,
            'workers': 8,     # 增加数据加载器线程数
        }   
    }

    if model_choice not in model_configs:
        raise ValueError(f"无效的模型选择: {model_choice}")

    model_config = model_configs[model_choice]

    # RTX 3080Ti专用配置
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': str(project_root / f"ultralytics/ultralytics/cfg/models/rt-detr/{model_config['file']}"),
        'data': str(project_root / "datasets/homeobjects-3K/HomeObjects-3K.yaml"),

        # RTX 3080Ti优化的核心参数
        'epochs': 100,
        'batch': model_config['batch'],
        'imgsz': 640,
        'patience': 20,

        # 稳定性优化设置 - 防止内存泄漏
        'device': '0',
        'workers': model_config['workers'],
        'amp': True,            # 混合精度训练
        'cache': False,         # 关闭缓存避免文件描述符问题
        'rect': True,           # 矩形训练
        'single_cls': False,

        # RTX 3080Ti优化的学习率设置
        'optimizer': 'AdamW',
        'lr0': model_config['lr0'],
        'lrf': 0.001,           # 更激进的学习率衰减
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,

        # 内存安全的数据增强设置
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,         # 关闭旋转减少计算
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,           # 关闭剪切减少计算
        'perspective': 0.0,     # 关闭透视变换防止内存泄漏
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.0,          # 关闭mosaic防止内存泄漏
        'mixup': 0.0,           # 关闭mixup防止内存泄漏
        'copy_paste': 0.0,      # 关闭copy_paste防止内存泄漏

        # 损失权重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,

        # 验证设置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,

        # 保存设置
        'save': True,
        'save_period': 20,  # 每 20 个 epoch 保存一次
        'project': '/root/autodl-tmp/runs/detect',
        'name': model_config['name'],
        'exist_ok': True,

        # RTX 3080Ti专用设置
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'close_mosaic': 10,
        'overlap_mask': True,   # RTX 3080Ti可以处理重叠mask
        'mask_ratio': 4,

        # 高级优化设置
        'profile': False,       # 关闭性能分析以提高速度
        'half': False,          # RTX 3080Ti用FP16可能不稳定，用AMP就够了
        'dnn': False,           # 不使用OpenCV DNN
    }

    return config

# RTX 3080Ti优化训练
def train_with_rtx3080ti_optimization(model_choice):
    try:
        # 设置环境
        setup_rtx3080ti_optimization()
        memory_monitor_rtx3080ti()

        # 导入ultralytics
        print("📦 导入Ultralytics...")
        from ultralytics import RTDETR

        # 获取配置
        config = get_rtx3080ti_config(model_choice)

        print(f"\n🚀 RTX 3080Ti优化训练开始")
        print(f"📄 模型: {config['model'].split('/')[-1]}")
        print(f"📊 批次大小: {config['batch']}")
        print(f"🎯 学习率: {config['lr0']}")
        print(f"👥 Workers: {config['workers']}")
        print(f"🧠 内存缓存: {config['cache']}")
        print("=" * 60)

        # 显存预热
        print("🔥 GPU预热中...")
        dummy_data = torch.randn(1, 3, 640, 640).cuda()
        for _ in range(10):
            _ = torch.nn.functional.conv2d(dummy_data, torch.randn(64, 3, 3, 3).cuda())
        torch.cuda.synchronize()
        del dummy_data
        torch.cuda.empty_cache()
        print("✅ GPU预热完成")

        # 创建模型
        model = RTDETR(config['model'])

        # 开始训练
        results = model.train(**{k: v for k, v in config.items() if k not in ['model']})

        print("\n🎉 训练完成!")
        print(f"📊 最佳mAP50: {results.mean_results()[2]}")  # 使用 mean_results 方法获取 mAP50

        # 最终清理
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # 自动关机
        if SHUTDOWN_AFTER_TRAIN:
            print("👋 训练完成，系统将在 1 分钟后关机...")
            os.system("shutdown -h +1")
        else:
            print("👋 训练完成，自动退出程序...")
            sys.exit(0)

    except Exception as e:
        print(f"❌ 训练出错: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="RTX 3080Ti 专用 RT-DETR 训练脚本")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_TRAIN_MODE,  # 使用全局默认模式
        help="选择训练模式: 1 (RT-DETR-L), 2 (RT-DETR+MNV4), 3 (RT-DETR+MNV4+SEA)"
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="训练完成后自动关机"
    )
    return parser.parse_args()

# 主函数
def main():
    print("🏎️  RTX 3080Ti专用RT-DETR训练优化器")
    print("=" * 50)

    args = parse_arguments()
    model_choice = str(args.mode)

    try:
        print(f"🚀 开始训练模式 {model_choice}...")
        train_with_rtx3080ti_optimization(model_choice)
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()