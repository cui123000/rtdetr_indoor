#!/usr/bin/env python3
"""
RT-DETR自动训练脚本 - 全局配置版
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
        'batch': 24,
        'lr0': 0.0004,
        'epochs': 100,
        'warmup_epochs': 8.0,
        'amp': True,
        'cache': True,
    },
    '2': {
        'file': 'rtdetr-mnv4-hybrid-m.yaml',
        'name': 'rtdetr_mnv4',
        'batch': 32,
        'lr0': 0.0004,
        'epochs': 100,
        'warmup_epochs': 8.0,
        'amp': True,
        'cache': True,
    },
    '3': {
        'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
        'name': 'rtdetr_mnv4_sea',
        'batch': 32,
        'lr0': 0.0004,
        'epochs': 100,
        'warmup_epochs': 8.0,
        'amp': True,
        'cache': True,
    },
    '4': {
        'file': 'rtdetr-l-sea.yaml',
        'name': 'rtdetr_l_sea',
        'batch': 24,
        'lr0': 0.0004,
        'epochs': 100,
        'warmup_epochs': 8.0,
        'amp': True,
        'cache': True,
    },
    '11': {
        'file': 'ert-detr.yaml',
        'name': 'ert_detr',
        'batch': 48,
        'lr0': 0.0005,
        'epochs': 120,
        'warmup_epochs': 10.0,
        'amp': False,
        'cache': True,
    },
}

# A40 GPU 优化配置 - 选择要训练的模型 (可通过环境变量临时覆盖)
# 默认使用脚本内定义的选择，但可通过环境变量 `SELECTED_MODEL` 临时覆盖，方便批量实验。
SELECTED_MODEL = os.environ.get('SELECTED_MODEL', '11')  # '1'=RT-DETR-L, '2'=RT-DETR+MNV4, '3'=RT-DETR+MNV4+SEA, '4'=RT-DETR-L+SEA

# 添加时间估算功能
def estimate_training_time():
    """估算训练时间 - A40 GPU优化版"""
    current_model = get_model_config(SELECTED_MODEL)
    
    # 基于A40 GPU的性能估算 (40GB内存，大batch优化)
    a40_speeds = {
        '1': 4.5,    # RT-DETR-L A40预计速度
        '2': 7.2,    # RT-DETR-MNV4 A40预计速度
        '3': 5.8,    # RT-DETR-MNV4-SEA A40预计速度
        '4': 5.2,    # RT-DETR-L-SEA A40预计速度
        '5': 9.5,    # RT-DETR-GhostNet A40预计速度
        '6': 10.2,   # RT-DETR-ShuffleNet-SEA A40预计速度
        '7': 8.0,    # RT-DETR-EfficientNet-CBAM A40预计速度
        '8': 4.8,    # RT-DETR-L-CBAM A40预计速度
        '9': 12.0,   # RT-DETR-MobileNetV3 A40预计速度
        '10': 15.0,  # RT-DETR-RepGhostNet A40预计速度
        '11': 18.0   # ERT-DETR A40预计速度(最轻量)
    }
    
    estimated_speed = a40_speeds.get(SELECTED_MODEL, 5.0)
    iterations_per_epoch = 2285 // current_model['batch']  # HomeObjects-3K训练集样本数 (2285 train images)，batch=24
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
    # 路径配置 - 使用 HomeObjects-3K 数据集 ✨
    'dataset_path': '/home/cjj/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
    'model_config': f'/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{current_model["file"]}',
    'save_dir': '/home/cjj/rtdetr_indoor/runs/detect',  # 权重保存目录
    'project_name': f"train_{current_model['name']}_balanced_{time.strftime('%Y%m%d_%H%M%S')}",  # 使用时间戳而非模型名
    
    # 训练参数 - A40 GPU 优化 ⚡
    'epochs': current_model['epochs'],     # 训练轮数
    'batch_size': current_model['batch'],  # 优化批次
    'img_size': 640,                      # 标准图像大小
    'workers': 16,  # ⚡ A40优化: 增加workers到16加速数据加载
    'pin_memory': True,  # 启用加速
    'patience': 15,     # 🚀 A40优化: 提前停止以节省时间
    
    # 学习率策略 - A40优化配置 ✨
    'lr0': current_model['lr0'],         # 🚀 直接使用模型特定lr0 (不再乘以2倍)
    'lrf': 0.01,                         # 最终学习率因子
    'warmup_epochs': current_model['warmup_epochs'], # 使用模型特定预热轮数
    'cos_lr': True,                      # 余弦学习率衰减
    
    # 优化器设置 - AdamW 稳定性好（针对RT-DETR优化）
    'optimizer': 'AdamW',
    'weight_decay': 0.001,              # ⚡ 增加正则化到0.001防止小数据集过拟合
    'momentum': 0.937,                   # 保留以兼容配置
    
    # 修复验证问题的关键设置
    'save_period': -1,  # 🚀 加速：仅保存last和best
    'plots': True,     # 🚀 启用图表生成
    'save_json': True, # 🚀 启用JSON保存
    
    # 数据增强 - ✨ 超强增强提升泛化能力（针对RT-DETR优化）
    'hsv_h': 0.025,          # ✨ 增加色调增强到0.025
    'hsv_s': 0.85,           # ✨ 增加饱和度增强到0.85
    'hsv_v': 0.55,           # ✨ 增加明度增强到0.55
    'degrees': 20.0,         # ✨ 增加旋转到±20度
    'translate': 0.25,       # ✨ 增加平移到25%
    'scale': 0.6,            # ✨ 调整缩放到0.4-1.6倍
    'fliplr': 0.5,           # 保留水平翻转
    'mosaic': 1.0,           # ✨ 启用Mosaic增强
    'mixup': 0.2,            # ✨ 增加MixUp到0.2
    'copy_paste': 0.1,       # ✨ 轻度启用copy_paste增强
    
    # A40 GPU 专用优化 - 快速稳定 ⚡
    'amp': current_model.get('amp', True),     # ⚡ 启用 AMP 混合精度
    'cache': current_model.get('cache', True), # ⚡ 启用缓存加速
    'rect': False,             # 关闭矩形训练保持稳定性
    'single_cls': False,
        'close_mosaic': 10,        # 🚀 提前关闭mosaic到第10轮，减少过拟合风险    # GPU设置 - 稳定性优先，避免确定性警告
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

def setup_a40_environment():
    """设置A40 GPU优化环境"""
    print("🚀 设置A40 GPU优化环境...")
    
    # 修复文件描述符限制
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"   ✅ 文件描述符限制: {new_soft}")
    except Exception as e:
        print(f"   ⚠️ 无法设置文件描述符: {e}")
    
    # A40 GPU 专用 CUDA 优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '8'   # A40优化: 增加到8
    os.environ['MKL_NUM_THREADS'] = '8'   # A40优化: 增加到8
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    # PyTorch 优化设置 - A40性能优先
    torch.backends.cudnn.benchmark = True  # 启用自动优化
    torch.backends.cudnn.deterministic = False
    # 启用 TF32 加速计算（A40 和 H100 都支持）✨ 关键优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    
    # 获取当前模型配置以应用模型特定的覆盖
    model_cfg = get_model_config(SELECTED_MODEL)
    
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
        
        # 损失权重调整 - 针对RT-DETR优化（增强分类权重）
        'box': 7.5,
        'cls': 1.0,    # 🚀 提高分类损失权重从0.5到1.0
        'dfl': 1.5,
        
        # 其他设置
        'verbose': GLOBAL_CONFIG['verbose'],
        'seed': GLOBAL_CONFIG['seed'],
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,  # 启用dropout防止过拟合
    }
    
    # 应用模型特定的配置覆盖（用于调试和微调）
    if 'mosaic' in model_cfg:
        config['mosaic'] = model_cfg['mosaic']
        print(f"   📌 模型特定覆盖: mosaic = {model_cfg['mosaic']}")
    if 'mixup' in model_cfg:
        config['mixup'] = model_cfg['mixup']
        print(f"   📌 模型特定覆盖: mixup = {model_cfg['mixup']}")
    if 'weight_decay' in model_cfg:
        config['weight_decay'] = model_cfg['weight_decay']
        print(f"   📌 模型特定覆盖: weight_decay = {model_cfg['weight_decay']}")
    if model_cfg.get('amp') is not None:
        config['amp'] = model_cfg['amp']
        print(f"   📌 模型特定覆盖: amp = {model_cfg['amp']}")
    if model_cfg.get('cache') is not None:
        config['cache'] = model_cfg['cache']
        print(f"   📌 模型特定覆盖: cache = {model_cfg['cache']}")
    
    return config

def gpu_memory_monitor():
    """GPU内存监控 - A40专用优化"""
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                # A40有40GB显存，在25GB时触发清理以保持稳定
                if allocated > 25.0:  # ⚡ A40优化: 提升阈值到25GB
                    print(f"🧹 触发GPU内存清理: {allocated:.1f}GB")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    new_allocated = torch.cuda.memory_allocated(0) / 1e9
                    print(f"   清理后: {new_allocated:.1f}GB")
            time.sleep(10)  # ⚡ 优化: 增加检查间隔到10秒
    
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
    print("🏎️ RT-DETR HomeObjects 自动训练器 (A40 GPU优化)")
    print("=" * 70)
    
    # 显示可选模型
    print("📋 可选模型配置 (A40 GPU极速版):")
    for key, config in MODEL_CONFIGS.items():
        marker = "👉" if key == SELECTED_MODEL else "  "
        print(f"{marker} {key}. {config['file']}")
        print(f"     batch={config['batch']}, lr={config['lr0']}, epochs={config['epochs']}")
        print(f"     AMP={config['amp']}, cache={config['cache']}")
    
    print(f"\n🎯 当前选择: 模型 {SELECTED_MODEL}")
    print("💡 要更改模型，请修改脚本中的 SELECTED_MODEL 变量")
    print("🚀 所有模型已配置A40 GPU加速: 超大batch + AMP + TF32 + 快速预热")
    
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
        setup_a40_environment()
        dataset_config = check_dataset()
        save_dir = setup_save_directory()
        
        # 显示配置信息
        print("\n🎯 最终训练配置:")
        print(f"   模型选择: {SELECTED_MODEL} - {current_model['name']}")
        print(f"   模型文件: {current_model['file']}")
        print(f"   数据集: HomeObjects-3K ({dataset_config['nc']}类) ✨")
        print(f"   批次大小: {GLOBAL_CONFIG['batch_size']}")
        print(f"   训练轮数: {GLOBAL_CONFIG['epochs']}")
        print(f"   学习率: {GLOBAL_CONFIG['lr0']:.6f} ✨")
        print(f"   Workers: {GLOBAL_CONFIG['workers']}")
        print(f"   权重保存: {save_dir}")
        print(f"   预热轮数: {GLOBAL_CONFIG['warmup_epochs']}")
        print(f"   数据增强: Mosaic={GLOBAL_CONFIG['mosaic']}, MixUp={GLOBAL_CONFIG['mixup']} ✨")
        print(f"   正则化: weight_decay={GLOBAL_CONFIG['weight_decay']} ✨")
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
        # Allow short-term fine-tune overrides via environment variables for quick experiments
        # e.g. FT_EPOCHS=5 FT_LR0=0.0002 FT_BATCH=24 FT_EMA=True
        try:
            ft_epochs = int(os.environ.get('FT_EPOCHS')) if os.environ.get('FT_EPOCHS') else None
        except Exception:
            ft_epochs = None
        try:
            ft_lr0 = float(os.environ.get('FT_LR0')) if os.environ.get('FT_LR0') else None
        except Exception:
            ft_lr0 = None
        try:
            ft_batch = int(os.environ.get('FT_BATCH')) if os.environ.get('FT_BATCH') else None
        except Exception:
            ft_batch = None
        ft_ema = os.environ.get('FT_EMA')
        if ft_epochs is not None:
            config['epochs'] = ft_epochs
            print(f"   📌 FT override: epochs = {ft_epochs}")
        if ft_lr0 is not None:
            config['lr0'] = ft_lr0
            print(f"   📌 FT override: lr0 = {ft_lr0}")
        if ft_batch is not None:
            config['batch'] = ft_batch
            print(f"   📌 FT override: batch = {ft_batch}")
        if ft_ema is not None:
            # ultralytics train accepts 'ema' arg; convert common truthy strings to bool
            if ft_ema.lower() in ('1', 'true', 'yes', 'on'):
                config['ema'] = True
            elif ft_ema.lower() in ('0', 'false', 'no', 'off'):
                config['ema'] = False
            else:
                config['ema'] = True
            print(f"   📌 FT override: ema = {config['ema']}")
        # 支持从指定权重微调：FT_WEIGHTS=/path/to/weights.pt
        ft_weights = os.environ.get('FT_WEIGHTS')
        if ft_weights:
            # Ultralytics accepts 'model' override as a path to a .pt weights file.
            config['model'] = ft_weights
            # Ensure resume is False so it will load provided weights as initialization
            config['resume'] = False
            print(f"   📌 FT override: model (weights) = {ft_weights}")
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
            # Ultralytics 会校验传入的参数字典，某些内部参数（如 'ema'）可能不被接受。
            # 从 overrides 中移除未知参数以避免语法错误。
            overrides = {k: v for k, v in config.items() if k != 'model' and k != 'ema'}
            results = model.train(**overrides)
            
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