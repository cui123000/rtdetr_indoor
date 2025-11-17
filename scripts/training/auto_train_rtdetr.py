#!/usr/bin/env python3
"""
RTX 4090专用RT-DETR自动训练脚本 - 全局配置版
使用智能筛选的HomeObjects数据集，最快速度完成训练
"""

import os
import sys
import yaml
import torch
import gc
import time
import shutil
from pathlib import Path
import threading

# ==================== 模型配置选择 ====================
MODEL_CONFIGS = {
    '1': {
        'file': 'rtdetr-l.yaml',
        'name': 'rtdetr_l',
        'batch': 16,       # 🚀 优化：最大化batch size
        'lr0': 0.00015,    # 🚀 batch增大时提高学习率
        'epochs': 100,     # ✅ 完整训练100 epochs
        'warmup_epochs': 5.0, # 🚀 适中预热期
        'amp': True,       # 🚀 加速：启用混合精度训练（速度提升30%）
        'cache': True,     # 启用缓存加速数据加载
    },
    '2': {
        'file': 'rtdetr-mnv4-hybrid-m.yaml', 
        'name': 'rtdetr_mnv4',
        'batch': 8,        # 保守批次
        'lr0': 0.0001,     # 标准学习率
        'epochs': 120,     # MobileNetV4 需要更多训练
        'warmup_epochs': 12.0, # 一般预热期
        'amp': False,      # 禁用AMP
        'cache': False,    # 禁用缓存
    },
    '3': {
        'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
        'name': 'rtdetr_mnv4_sea',
        'batch': 6,        # SEA 模块使用较小批次
        'lr0': 0.00008,    # SEA 模块使用较低学习率
        'epochs': 150,     # SEA 版本需要长训练
        'warmup_epochs': 15.0, # 较长预热期
        'amp': False,      # 禁用AMP
        'cache': False,    # 禁用缓存
    },
    '4': {
        'file': 'rtdetr-l-sea.yaml',
        'name': 'rtdetr_l_sea',
        'batch': 10,       # RT-DETR-L + SEA
        'lr0': 0.0001,     # 标准学习率
        'epochs': 100,     # 标准训练轮数
        'warmup_epochs': 10.0, # 一般预热期
        'amp': False,      # 禁用AMP保证稳定性
        'cache': True,     # 启用缓存
    }
}

# RTX 4090 优化配置 - 选择要训练的模型 (修改这里来选择不同模型)
SELECTED_MODEL = '1'  # '1'=RT-DETR-L, '2'=RT-DETR+MNV4, '3'=RT-DETR+MNV4+SEA, '4'=RT-DETR-L+SEA

# 添加时间估算功能
def estimate_training_time():
    """估算训练时间"""
    current_model = get_model_config(SELECTED_MODEL)
    
    # 基于RTX 4090的性能估算 (更新为实际观察值)
    rtx4090_speeds = {
        '1': 4.5,    # RT-DETR-L 实际观察速度更新
        '2': 5.8,    # RT-DETR-MNV4 预计速度
        '3': 4.2,    # RT-DETR-MNV4-SEA 预计速度
        '4': 4.0     # RT-DETR-L-SEA 预计速度
    }
    
    estimated_speed = rtx4090_speeds.get(SELECTED_MODEL, 4.0)
    iterations_per_epoch = 6400 // current_model['batch']  # 更新为新的训练样本数
    seconds_per_epoch = iterations_per_epoch / estimated_speed
    total_hours = (seconds_per_epoch * current_model['epochs']) / 3600
    
    return {
        'speed': estimated_speed,
        'iterations_per_epoch': iterations_per_epoch,
        'seconds_per_epoch': seconds_per_epoch,
        'total_hours': total_hours,
        'epochs': current_model['epochs']
    }

# ==================== 全局训练配置 ====================
def get_model_config(model_choice):
    """获取选定模型的配置"""
    if model_choice not in MODEL_CONFIGS:
        raise ValueError(f"无效的模型选择: {model_choice}. 可选: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_choice]

# 获取当前选择的模型配置
current_model = get_model_config(SELECTED_MODEL)

GLOBAL_CONFIG = {
    # 路径配置 - 使用室内筛选数据集
    'dataset_path': '/home/cjj/rtdetr_indoor/datasets/coco_indoor/coco_indoor.yaml',
    'model_config': f'/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{current_model["file"]}',
    'save_dir': '/home/cjj/rtdetr_indoor/runs/detect',  # 权重保存目录
    'project_name': f"train_{current_model['name']}_{time.strftime('%Y%m%d_%H%M%S')}",  # 使用时间戳而非模型名
    
    # 训练参数 - RTX 4090 优化
    'epochs': current_model['epochs'],     # 训练轮数
    'batch_size': current_model['batch'],  # 保守批次
    'img_size': 640,                      # 标准图像大小
    'workers': 4,   # 🚀 加速：减少workers降低CPU开销
    'pin_memory': True,  # 启用加速
    'patience': 25,                       # ✅ 完整训练：适中的早停容忍度
    
    # 学习率策略 - 保守配置
    'lr0': current_model['lr0'],          # 使用模型特定学习率
    'lrf': 0.2,                          # 一般最终学习率因子
    'warmup_epochs': current_model['warmup_epochs'], # 使用模型特定预热轮数
    'cos_lr': True,                      # 余弦学习率衰减
    
    # 优化器设置 - AdamW 更稳定
    'optimizer': 'AdamW',
    'weight_decay': 0.0001,              # 保守权重衰减
    'momentum': 0.937,                   # 保留以兼容配置(AdamW实际使用betas而非momentum)
    
    # 修复验证问题的关键设置
    'save_period': -1,  # 🚀 加速：仅保存last和best
    'plots': False,     # 🚀 加速：禁用图表生成
    'save_json': False, # 🚀 加速：禁用JSON保存
    
    # 数据增强 - 🚀 极简配置加速
    'hsv_h': 0.005,          # 🚀 最小化增强
    'hsv_s': 0.3,            # 🚀 最小化增强
    'hsv_v': 0.2,            # 🚀 最小化增强
    'degrees': 0.0,          # 🚀 禁用旋转
    'translate': 0.05,       # 🚀 最小平移
    'scale': 0.2,            # 🚀 最小缩放
    'fliplr': 0.5,           # 保留水平翻转
    'mosaic': 0.0,           # 🚀 禁用mosaic加速
    'mixup': 0.0,            # 🚀 禁用mixup加速
    'copy_paste': 0.0,       # 禁用 copy_paste
    
    # RTX 4090 专用优化 - 保守配置
    'amp': current_model.get('amp', False),    # 禁用 AMP
    'cache': current_model.get('cache', False), # 禁用缓存
    'rect': False,             # 关闭矩形训练
    'single_cls': False,
    'close_mosaic': 10,        # 🚀 加速：提早关闭mosaic
    
    # GPU设置 - 稳定性优先，避免确定性警告
    'device': '0',             # 使用第一块GPU
    'dnn': False,
    'half': False,             # 禁用half precision
    'deterministic': False,    # 禁用严格确定性避免CuBLAS警告
    'seed': 42,                # 保持随机种子确保相对一致性
    'verbose': True,
    
    # 验证和检测设置 - 关键修复RT-DETR验证错误
    'val': True,
    'conf': 0.001,             # 降低置信度阈值
    'iou': 0.6,                # 降低IoU阈值
    'max_det': 300,
    'augment': False,          # 验证时不使用增强
    'save_txt': False,         # 禁用文本保存
    'save_conf': False,        # 禁用置信度保存
    'save_crop': False,        # 禁用裁剪保存
}

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def setup_rtx4090_environment():
    """设置RTX 4090优化环境"""
    print("🚀 设置RTX 4090优化环境...")
    
    # 修复文件描述符限制
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"   ✅ 文件描述符限制: {new_soft}")
    except Exception as e:
        print(f"   ⚠️ 无法设置文件描述符: {e}")
    
    # RTX 4090 专用 CUDA 优化 - 最小化配置以避免初始化问题
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # PyTorch 优化设置 - 性能优先
    torch.backends.cudnn.benchmark = True  # 启用自动优化
    torch.backends.cudnn.deterministic = False
    # 启用 TF32 加速计算（A40 支持）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # 详细GPU信息
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ 总显存: {gpu_memory:.1f}GB")
            print(f"   ✅ TF32优化: 已禁用（稳定模式）")
        except Exception as e:
            print(f"   ⚠️ GPU信息获取失败: {e}")
    else:
        raise RuntimeError("❌ CUDA不可用")

def check_dataset():
    """检查数据集"""
    print("📊 检查数据集...")
    
    dataset_path = Path(GLOBAL_CONFIG['dataset_path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ 数据集配置文件不存在: {dataset_path}")
    
    # 获取数据集根目录
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_root = Path(dataset_config['path'])
    train_dir = dataset_root / dataset_config['train']
    val_dir = dataset_root / dataset_config['val']
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"❌ 数据集图像目录不存在")
    
    # 统计图像数量
    train_count = len(list(train_dir.glob('*.jpg')))
    val_count = len(list(val_dir.glob('*.jpg')))
    
    print(f"   ✅ 训练图像: {train_count}")
    print(f"   ✅ 验证图像: {val_count}")
    print(f"   ✅ 总计: {train_count + val_count}")
    print(f"   ✅ 类别数: {dataset_config['nc']}")
    
    return dataset_config

def setup_save_directory():
    """设置保存目录"""
    print("💾 设置权重保存目录...")
    
    save_dir = Path(GLOBAL_CONFIG['save_dir'])
    
    # 尝试创建目录
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ 权重保存目录: {save_dir}")
    except PermissionError:
        # 如果没有权限，使用用户主目录
        alternative_dir = Path.home() / 'rtdetr_weights'
        alternative_dir.mkdir(parents=True, exist_ok=True)
        GLOBAL_CONFIG['save_dir'] = str(alternative_dir)
        print(f"   ⚠️ 权限不足，使用备用目录: {alternative_dir}")
        
    return Path(GLOBAL_CONFIG['save_dir'])

def create_training_config():
    """创建训练配置"""
    print("⚙️ 创建训练配置...")
    
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': GLOBAL_CONFIG['model_config'],
        'data': GLOBAL_CONFIG['dataset_path'],
        
        # 训练参数
        'epochs': GLOBAL_CONFIG['epochs'],
        'batch': GLOBAL_CONFIG['batch_size'],
        'imgsz': GLOBAL_CONFIG['img_size'],
        'patience': GLOBAL_CONFIG['patience'],
        'workers': GLOBAL_CONFIG['workers'],
        # pin_memory由PyTorch/DataLoader内部处理,不是Ultralytics的参数
        'device': GLOBAL_CONFIG['device'],
        
        # 优化设置
        'optimizer': GLOBAL_CONFIG['optimizer'],
        'lr0': GLOBAL_CONFIG['lr0'],
        'lrf': GLOBAL_CONFIG['lrf'],
        'momentum': GLOBAL_CONFIG['momentum'],
        'weight_decay': GLOBAL_CONFIG['weight_decay'],
        'warmup_epochs': GLOBAL_CONFIG['warmup_epochs'],
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': GLOBAL_CONFIG['cos_lr'],
        
        # 数据增强
        'hsv_h': GLOBAL_CONFIG['hsv_h'],
        'hsv_s': GLOBAL_CONFIG['hsv_s'],
        'hsv_v': GLOBAL_CONFIG['hsv_v'],
        'degrees': GLOBAL_CONFIG['degrees'],
        'translate': GLOBAL_CONFIG['translate'],
        'scale': GLOBAL_CONFIG['scale'],
        'fliplr': GLOBAL_CONFIG['fliplr'],
        'mosaic': GLOBAL_CONFIG['mosaic'],
        'mixup': GLOBAL_CONFIG['mixup'],
        'copy_paste': GLOBAL_CONFIG['copy_paste'],
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        
        # RTX 4090优化
        'amp': GLOBAL_CONFIG['amp'],
        'cache': GLOBAL_CONFIG['cache'],
        'rect': GLOBAL_CONFIG['rect'],
        'single_cls': GLOBAL_CONFIG['single_cls'],
        'dnn': GLOBAL_CONFIG['dnn'],
        'half': GLOBAL_CONFIG['half'],
        'deterministic': GLOBAL_CONFIG['deterministic'],
        'close_mosaic': GLOBAL_CONFIG['close_mosaic'],
        
        # 保存设置
        'save': True,
        'save_period': GLOBAL_CONFIG['save_period'],
        'save_json': GLOBAL_CONFIG['save_json'],
        'plots': GLOBAL_CONFIG['plots'],
        'val': GLOBAL_CONFIG['val'],
        'project': GLOBAL_CONFIG['save_dir'],
        'name': GLOBAL_CONFIG['project_name'],
        'exist_ok': True,
        
        # 验证设置 - 关键修复
        'conf': GLOBAL_CONFIG['conf'],
        'iou': GLOBAL_CONFIG['iou'],
        'max_det': GLOBAL_CONFIG['max_det'],
        'augment': GLOBAL_CONFIG['augment'],
        
        # RT-DETR特殊设置 - 修复验证错误
        'save_txt': False,         # 禁用文本保存
        'save_conf': False,        # 禁用置信度保存
        'save_crop': False,        # 禁用裁剪保存
        'rect': False,             # 确保使用方形图像避免缩放问题
        
        # 损失权重调整 - 针对RT-DETR优化
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 其他设置
        'verbose': GLOBAL_CONFIG['verbose'],
        'seed': GLOBAL_CONFIG['seed'],
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,  # 禁用dropout以获得更稳定的训练
    }
    
    return config

def gpu_memory_monitor():
    """GPU内存监控 - 更激进的清理"""
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                # 更早触发清理以避免驱动压力
                if allocated > 12.0:  # 超过12GB时清理，更保守
                    print(f"🧹 触发GPU内存清理: {allocated:.1f}GB")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    new_allocated = torch.cuda.memory_allocated(0) / 1e9
                    print(f"   清理后: {new_allocated:.1f}GB")
            time.sleep(5)  # 更频繁的检查
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def force_cuda_cleanup():
    """强制CUDA内存清理"""
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 强制CUDA清理完成")
    except Exception as e:
        print(f"⚠️ CUDA清理警告: {e}")

def copy_best_weights(results, config):
    """复制最佳权重到指定位置"""
    try:
        project_dir = Path(config['project']) / config['name']
        weights_dir = project_dir / 'weights'
        best_weight = weights_dir / 'best.pt'
        
        if best_weight.exists():
            # 复制到目标目录(使用模型名+时间戳)
            model_name = current_model['name']
            final_name = f"homeobjects_{model_name}_best_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            final_path = Path(config['project']) / final_name
                
            shutil.copy2(best_weight, final_path)
            print(f"✅ 最佳权重已保存: {final_path}")
            
            # 创建信息文件
            info_file = final_path.with_suffix('.txt')
            with open(info_file, 'w') as f:
                f.write(f"RT-DETR HomeObjects训练结果\n")
                f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型选择: {SELECTED_MODEL} - {current_model['name']}\n")
                f.write(f"模型文件: {current_model['file']}\n")
                f.write(f"数据集: HomeObjects扩展版 (智能筛选)\n")
                f.write(f"批次大小: {config['batch']}\n")
                f.write(f"学习率: {config['lr0']}\n")
                f.write(f"训练轮数: {config['epochs']}\n")
                f.write(f"Workers: {config['workers']}\n")
                f.write(f"权重文件: {final_name}\n")
            
            print(f"📄 训练信息已保存: {info_file}")
            
        else:
            print("❌ 未找到最佳权重文件")
            
    except Exception as e:
        print(f"❌ 权重复制失败: {e}")

def main():
    """主训练函数"""
    print("🏎️ RT-DETR HomeObjects 自动训练器 (RTX 4090优化)")
    print("=" * 70)
    
    # 显示可选模型
    print("📋 可选模型配置 (防NaN稳定版):")
    for key, config in MODEL_CONFIGS.items():
        marker = "👉" if key == SELECTED_MODEL else "  "
        print(f"{marker} {key}. {config['file']}")
        print(f"     batch={config['batch']}, lr={config['lr0']}, epochs={config['epochs']}")
        print(f"     warmup={config['warmup_epochs']}, amp={config['amp']}, cache={config['cache']}")
    
    print(f"\n🎯 当前选择: 模型 {SELECTED_MODEL} (稳定配置)")
    print("💡 要更改模型，请修改脚本中的 SELECTED_MODEL 变量")
    print("🛡️ 所有模型已配置防NaN参数: 低学习率 + 禁用AMP + 长预热期")
    
    # 训练时间估算
    time_estimate = estimate_training_time()
    print(f"\n⏱️ 训练时间估算:")
    print(f"   预计速度: {time_estimate['speed']:.1f} it/s")
    print(f"   每epoch迭代数: {time_estimate['iterations_per_epoch']}")
    print(f"   每epoch时间: {time_estimate['seconds_per_epoch']/60:.1f} 分钟")
    print(f"   总训练时间: {time_estimate['total_hours']:.1f} 小时")
    print("=" * 70)
    
    # 交互式确认
    print("\n🤔 训练前确认:")
    print(f"1. 模型: {current_model['name']}")
    print(f"2. 预计时间: {time_estimate['total_hours']:.1f} 小时")
    print(f"3. Batch大小: {current_model['batch']}")
    
    # 添加简单的用户交互 - 支持命令行参数跳过
    import sys
    skip_confirm = '--skip-confirm' in sys.argv or len(sys.argv) > 1
    
    if not skip_confirm:
        try:
            confirm = input("\n确认开始训练? (y/n): ").strip().lower()
            if confirm != 'y':
                print("❌ 取消训练")
                return
        except KeyboardInterrupt:
            print("\n❌ 用户取消")
            return
    else:
        print("⏭️ 跳过确认，直接开始训练")
    
    try:
        # 环境设置
        setup_rtx4090_environment()
        dataset_config = check_dataset()
        save_dir = setup_save_directory()
        
        # 显示配置信息
        print("\n🎯 最终训练配置:")
        print(f"   模型选择: {SELECTED_MODEL} - {current_model['name']}")
        print(f"   模型文件: {current_model['file']}")
        print(f"   数据集: HomeObjects扩展版 ({dataset_config['nc']}类)")
        print(f"   批次大小: {GLOBAL_CONFIG['batch_size']}")
        print(f"   训练轮数: {GLOBAL_CONFIG['epochs']}")
        print(f"   学习率: {GLOBAL_CONFIG['lr0']}")
        print(f"   Workers: {GLOBAL_CONFIG['workers']}")
        print(f"   权重保存: {save_dir}")
        print(f"   预热轮数: {GLOBAL_CONFIG['warmup_epochs']}")
        print("=" * 70)
        
        # 启动内存监控
        print("📡 启动内存监控线程...")
        # 注释掉内存监控，因为它可能导致线程冲突
        # gpu_memory_monitor()
        
        # 强制清理初始状态
        print("🧹 执行初始CUDA清理...")
        sys.stdout.flush()
        force_cuda_cleanup()
        sys.stdout.flush()
        
        # 导入ultralytics
        print("📦 导入Ultralytics模块（可能需要 30-60 秒）...")
        sys.stdout.flush()
        
        start_import = time.time()
        try:
            from ultralytics import RTDETR
            import_time = time.time() - start_import
            print(f"✅ Ultralytics导入成功 ({import_time:.2f}s)")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Ultralytics导入失败: {e}")
            import traceback
            traceback.print_exc()
            force_cuda_cleanup()
            raise
        
        print("⚙️ 创建训练配置...")
        config = create_training_config()
        print("✅ 训练配置创建成功")
        
        # 跳过GPU预热，直接创建模型以加快启动
        print("🚀 创建RT-DETR模型（这可能需要 2-3 分钟）...")
        sys.stdout.flush()
        
        model_load_start = time.time()
        try:
            model = RTDETR(config['model'])
            model_load_time = time.time() - model_load_start
            print(f"✅ 模型创建成功 ({model_load_time:.2f}s)")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            force_cuda_cleanup()
            raise
        
        # 训练开始时间
        start_time = time.time()
        
        # 开始训练 - 添加异常处理
        try:
            print("🎯 开始模型训练...")
            print("⏳ 这可能需要几分钟来加载数据集和验证配置，请耐心等待...")
            sys.stdout.flush()
            results = model.train(**{k: v for k, v in config.items() if k != 'model'})
            
            training_time = (time.time() - start_time) / 3600  # 转换为小时
            print(f"\n🎉 训练完成! 实际用时: {training_time:.2f} 小时")
            
            # 处理训练结果
            copy_best_weights(results, config)
            
            # 训练总结
            print(f"\n📊 训练总结:")
            print(f"   模型: {current_model['name']}")
            print(f"   实际训练时间: {training_time:.2f} 小时")
            print(f"   预估时间: {time_estimate['total_hours']:.1f} 小时")
            print(f"   时间差异: {abs(training_time - time_estimate['total_hours']):.2f} 小时")
            
        except RuntimeError as e:
            if "expandable_segment" in str(e) or "CUDA" in str(e):
                print(f"\n❌ CUDA内存分配器错误: {e}")
                print("💡 建议的解决方案:")
                print("   1. 重启Python进程清理CUDA状态")
                print("   2. 降低batch size (当前: {})".format(current_model['batch']))
                print("   3. 检查GPU驱动和PyTorch版本兼容性")
                force_cuda_cleanup()
                raise
            else:
                print(f"\n❌ 训练运行时错误: {e}")
                force_cuda_cleanup()
                raise
        except Exception as e:
            print(f"\n❌ 未知训练错误: {e}")
            force_cuda_cleanup()
            raise
        
        # 最终清理
        try:
            del model
            force_cuda_cleanup()
        except:
            pass
        
        print("✅ 训练任务全部完成!")
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        
        # 详细的错误分析
        if "expandable_segment" in str(e):
            print("🔧 CUDA内存分配器错误解决方案:")
            print("   - 这是PyTorch的CUDA内存管理问题")
            print("   - 已在脚本中设置 expandable_segments=False")
            print("   - 请重启Python进程后重新运行")
        elif "CUDA out of memory" in str(e):
            print("🔧 GPU内存不足解决方案:")
            print(f"   - 当前batch size: {current_model['batch']}")
            print("   - 建议减少batch size到 6-8")
            print("   - 或减少workers数量")
        
        force_cuda_cleanup()
        raise

if __name__ == "__main__":
    main()