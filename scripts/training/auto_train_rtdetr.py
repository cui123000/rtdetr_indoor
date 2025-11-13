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
        'name': 'rtdetr_l_homeobjects_smart_optimized',
        'batch': 10,        # 降低批次避免NaN
        'lr0': 0.0001,     # 大幅降低学习率防止NaN
        'workers': 4,      # 减少workers提升稳定性
        'epochs': 100,     # 减少epochs，数据集较小
        'warmup_epochs': 10.0, # 增加预热期
        'amp': False,      # 禁用AMP提升稳定性
        'cache': False,    # 禁用缓存避免内存问题
    },
    '2': {
        'file': 'rtdetr-mnv4-hybrid-m.yaml', 
        'name': 'rtdetr_mnv4_hybrid_rtx4090_safe',
        'batch': 8,        # 更保守的batch size
        'lr0': 0.00008,    # 更低的学习率
        'workers': 4,      # 减少workers
        'epochs': 100,     # MNV4需要更多训练轮数
        'warmup_epochs': 12.0, # 更长预热期
        'amp': False,      # 禁用AMP
        'cache': False,    # 禁用缓存
    },
    '3': {
        'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
        'name': 'rtdetr_mnv4_sea_rtx4090_safe',
        'batch': 6,        # 最保守的batch size
        'lr0': 0.00006,    # 最低学习率
        'workers': 4,      # 减少workers
        'epochs': 100,     # SEA版本需要最多训练轮数
        'warmup_epochs': 15.0, # 最长预热期
        'amp': False,      # 禁用AMP
        'cache': False,    # 禁用缓存
    }   
}

# 选择要训练的模型 (修改这里来选择不同模型)
SELECTED_MODEL = '1'  # '1'=RT-DETR-L, '2'=RT-DETR+MNV4, '3'=RT-DETR+MNV4+SEA

# 添加时间估算功能
def estimate_training_time():
    """估算训练时间"""
    current_model = get_model_config(SELECTED_MODEL)
    
    # 基于RTX 4090的性能估算 (更新为实际观察值)
    rtx4090_speeds = {
        '1': 4.5,    # RT-DETR-L 实际观察速度更新
        '2': 5.8,    # RT-DETR-MNV4 预计速度
        '3': 4.2     # RT-DETR-MNV4-SEA 预计速度
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
    # 路径配置
    'dataset_path': '/home/cui/rtdetr_indoor/datasets/homeobjects_extended_yolo_smart/homeobjects_extended_smart.yaml',
    'model_config': f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{current_model["file"]}',
    'save_dir': '/root/autodl-tmp/rtdetr_weights',  # 权重保存目录
    'project_name': current_model['name'],
    
    # 训练参数 - 使用模型特定配置
    'epochs': current_model['epochs'],     # 训练轮数
    'batch_size': current_model['batch'],  # 使用模型特定批次大小
    'img_size': 640,                      # 输入图像尺寸
    'workers': current_model['workers'],   # 使用模型特定workers数
    'patience': 40,                       # 进一步增加patience
    
    # 学习率策略 - 使用模型特定配置
    'lr0': current_model['lr0'],          # 使用模型特定学习率
    'lrf': 0.2,                          # 提高最终学习率因子
    'warmup_epochs': current_model['warmup_epochs'], # 使用模型特定预热轮数
    'cos_lr': True,                      # 余弦学习率衰减
    
    # 优化器设置 - 更保守参数
    'optimizer': 'AdamW',
    'weight_decay': 0.00005,             # 大幅降低权重衰减
    'momentum': 0.8,                     # 降低momentum
    
    # 修复验证问题的关键设置
    'save_period': 10, 
    'plots': True,
    'save_json': True,         # 保存验证结果JSON用于分析
    
    # 数据增强 - 极度保守防止训练不稳定
    'hsv_h': 0.005,          # 极小色调变化
    'hsv_s': 0.1,            # 极小饱和度变化
    'hsv_v': 0.1,            # 极小明度变化
    'degrees': 1.0,          # 极小旋转
    'translate': 0.02,       # 极小平移
    'scale': 0.1,            # 极小缩放
    'fliplr': 0.3,           # 减少翻转
    'mosaic': 0.1,           # 大幅减少mosaic
    'mixup': 0.0,            # 完全禁用mixup
    'copy_paste': 0.0,       # 完全禁用copy_paste
    
    # RTX 4090专用优化 - 使用模型特定稳定性设置
    'amp': current_model.get('amp', False),    # 使用模型特定AMP设置
    'cache': current_model.get('cache', False), # 使用模型特定缓存设置
    'rect': False,             # 关闭矩形训练，使用标准正方形训练 - 重要!
    'single_cls': False,
    'close_mosaic': 30,        # 提前更多关闭mosaic
    
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
    
    # RTX 4090专用CUDA优化 - 修复内存分配器错误
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:False'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '6'   # 减少线程数避免冲突
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['TORCH_NUM_WORKERS'] = str(current_model['workers'])
    
    # 禁用有问题的CUDA功能
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''  # 让PyTorch自动检测
    
    # PyTorch优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(6)  # 减少线程数避免冲突
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # 使用85%显存，更安全
        torch.cuda.empty_cache()
        
        # 详细GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"   ✅ GPU: {gpu_name}")
        print(f"   ✅ 总显存: {gpu_memory:.1f}GB")
        print(f"   ✅ 可用显存: {gpu_memory * 0.85:.1f}GB")
        print(f"   ✅ TF32优化: 已启用")
        print(f"   ⚠️ 安全模式: expandable_segments=False")
        
        # GPU性能测试
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            start_time = time.time()
            for _ in range(100):
                _ = torch.mm(test_tensor, test_tensor)
            torch.cuda.synchronize()
            gpu_test_time = time.time() - start_time
            del test_tensor
            torch.cuda.empty_cache()
            print(f"   ✅ GPU性能测试: {gpu_test_time:.3f}s (正常 < 1.0s)")
        except Exception as e:
            print(f"   ❌ GPU测试失败: {e}")
            raise
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
                if allocated > 16.0:  # 超过16GB时清理，更保守
                    print(f"🧹 触发GPU内存清理: {allocated:.1f}GB")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    new_allocated = torch.cuda.memory_allocated(0) / 1e9
                    print(f"   清理后: {new_allocated:.1f}GB")
            time.sleep(10)  # 更频繁的检查
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def force_cuda_cleanup():
    """强制CUDA内存清理"""
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.synchronize()
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
            # 复制到目标目录
            final_name = f"homeobjects_rtdetr_best_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            if config['project'].startswith('/root/autodl-tmp'):
                final_path = Path('/root/autodl-tmp') / final_name
            else:
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
    
    # 添加简单的用户交互
    try:
        confirm = input("\n确认开始训练? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消训练")
            return
    except KeyboardInterrupt:
        print("\n❌ 用户取消")
        return
    
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
        gpu_memory_monitor()
        
        # 强制清理初始状态
        force_cuda_cleanup()
        
        # 导入ultralytics
        print("📦 导入Ultralytics...")
        from ultralytics import RTDETR
        
        # 创建配置
        config = create_training_config()
        
        # 安全的GPU预热 - 避免内存分配器错误
        print("🔥 安全GPU预热...")
        try:
            # 使用小批次预热避免内存问题
            small_data = torch.randn(2, 3, 640, 640).cuda()
            small_conv = torch.nn.Conv2d(3, 32, 3, padding=1).cuda()
            
            # 预热循环
            with torch.no_grad():
                for i in range(5):
                    _ = small_conv(small_data)
                    if i % 2 == 0:
                        torch.cuda.empty_cache()
            
            torch.cuda.synchronize()
            del small_data, small_conv
            force_cuda_cleanup()
            print("✅ GPU预热完成")
        except Exception as e:
            print(f"⚠️ GPU预热警告: {e}")
            force_cuda_cleanup()
        
        # 创建模型并开始训练
        print("🚀 开始训练...")
        try:
            model = RTDETR(config['model'])
            print("✅ 模型创建成功")
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            force_cuda_cleanup()
            raise
        
        # 训练开始时间
        start_time = time.time()
        
        # 开始训练 - 添加异常处理
        try:
            print("🎯 开始模型训练...")
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