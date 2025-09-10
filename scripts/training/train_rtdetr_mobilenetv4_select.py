#!/usr/bin/env python3
"""
RT-DETR with MobileNetV4 Training Script
统一的训练配置，支持三个主要版本：原始、MobileNetV4混合、MobileNetV4+SEA优化
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent.parent  # 从scripts/training/回到项目根目录
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 定义可用的模型版本 - 简化为三个主要版本
MODEL_VERSIONS = {
    '1': {
        'name': 'Original RT-DETR (原始版本)',
        'file': 'rtdetr-l.yaml',
        'description': '原始RT-DETR-L模型，标准基准',
        'modules': ['HGStem', 'HGBlock', 'RepC3', 'AIFI'],
        'status': '📊 基准模型'
    },
    '2': {
        'name': 'RT-DETR + MobileNetV4 Hybrid (混合版本)',
        'file': 'rtdetr-mnv4-hybrid-m.yaml',
        'description': '集成MobileNetV4混合架构的高效版本',
        'modules': ['EdgeResidual', 'UniversalInvertedResidual', 'C2f', 'RepC3', 'AIFI'],
        'status': '🚀 移动优化'
    },
    '3': {
        'name': 'RT-DETR + MobileNetV4 + SEA Attention (SEA优化版本)',
        'file': 'rtdetr-mnv4-hybrid-m-sea.yaml', 
        'description': 'MobileNetV4 + 优化SEA注意力机制的最强版本',
        'modules': ['EdgeResidual', 'UniversalInvertedResidual', 'Sea_Attention_Simplified', 'OptimizedSEA_Attention', 'TransformerEnhancedSEA', 'C2f', 'RepC3'],
        'status': '🌟 SEA增强'
    }
}

def select_model_version():
    """选择模型版本"""
    print("\n📋 可用的RT-DETR模型版本:")
    print("=" * 80)
    
    for key, version in MODEL_VERSIONS.items():
        print(f"{key}. {version['name']}")
        print(f"   📄 配置文件: {version['file']}")
        print(f"   📝 描述: {version['description']}")
        print(f"   🧩 核心模块: {', '.join(version['modules'][:3])}{'...' if len(version['modules']) > 3 else ''}")
        print(f"   📊 状态: {version['status']}")
        print()
    
    while True:
        try:
            choice = input("请选择版本 (1-3): ").strip()
            if choice in MODEL_VERSIONS:
                selected = MODEL_VERSIONS[choice]
                print(f"\n✅ 已选择: {selected['name']}")
                print(f"📄 配置文件: {selected['file']}")
                return selected['file'], choice
            else:
                print("❌ 无效选择，请输入 1-3")
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 输入错误: {e}")

def create_training_config(model_file, version_choice):
    """创建统一的训练配置"""
    model_path = f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_file}'
    
    # 统一的基础训练配置
    config = {
        # 基本设置
        'task': 'detect',
        'mode': 'train',
        
        # 模型和数据
        'model': model_path,
        'data': '/home/cui/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
        
        # 统一训练参数 - 内存优化版本
        'epochs': 100,
        'batch': 4,              # 大幅减少batch size以节省显存
        'imgsz': 640,
        'patience': 20,          # 早停耐心
        
        # 保存设置
        'save': True,
        'save_period': 5,        # 每5轮保存一次
        'project': 'runs/detect',
        'name': f'rtdetr_{model_file.replace(".yaml", "").replace("-", "_")}',
        'exist_ok': True,
        
        # 设备设置 - 内存优化
        'device': '0',
        'workers': 4,            # 减少workers以节省CPU内存
        'amp': True,             # 混合精度训练
        
        # 验证设置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        
        # 优化器设置 - 根据版本优化
        'optimizer': 'AdamW',
        'lr0': 0.001,            # 初始学习率
        'lrf': 0.01,             # 最终学习率比例
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'cos_lr': True,          # 余弦学习率调度
        
        # 数据增强策略 - 减少内存消耗的增强
        'hsv_h': 0.015,          # 色调变化
        'hsv_s': 0.7,            # 饱和度变化
        'hsv_v': 0.4,            # 亮度变化
        'degrees': 5.0,          # 旋转角度
        'translate': 0.1,        # 平移
        'scale': 0.5,            # 缩放
        'shear': 2.0,            # 剪切
        'perspective': 0.0,      # 透视变换
        'flipud': 0.0,           # 垂直翻转
        'fliplr': 0.5,           # 水平翻转
        'mosaic': 0.5,           # 减少Mosaic增强以节省内存
        'mixup': 0.0,            # 关闭Mixup以节省内存
        'copy_paste': 0.0,       # 关闭Copy-paste以节省内存
        
        # 损失权重
        'box': 7.5,              # 边界框损失权重
        'cls': 0.5,              # 分类损失权重
        'dfl': 1.5,              # 分布焦点损失权重
        
        # 其他设置 - 内存优化
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'cache': False,          # 关闭缓存以节省内存
        'close_mosaic': 10,      # 最后10轮关闭mosaic
    }
    
    # 根据不同版本进行微调
    if version_choice == '1':  # 原始RT-DETR
        print("🎯 使用原始RT-DETR配置...")
        config.update({
            'lr0': 0.001,          # 标准学习率
            'batch': 4,            # 原始模型batch size
            'warmup_epochs': 5.0,  # 更长预热
        })
        
    elif version_choice == '2':  # MobileNetV4混合版本
        print("🚀 使用MobileNetV4混合版本配置...")
        config.update({
            'lr0': 0.0008,         # 稍微降低学习率
            'batch': 4,            # 适中batch size
            'weight_decay': 0.0008, # 稍微增加权重衰减
        })
        
    elif version_choice == '3':  # SEA优化版本
        print("🌟 使用SEA注意力优化配置...")
        config.update({
            'lr0': 0.0005,         # 更保守的学习率
            'batch': 4,            # 最小batch size以适应复杂模型
            'warmup_epochs': 5.0,  # 更长预热期
            'weight_decay': 0.001, # 更强的正则化
            'patience': 25,        # 更多耐心
            'cos_lr': True,        # 确保使用余弦学习率
            'workers': 4,          # 进一步减少workers
        })
    
    return config

def setup_environment():
    """设置训练环境"""
    print("🔧 设置训练环境...")
    
    # 设置CUDA内存管理
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用CUDA阻塞模式以便调试
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 限制CUDA内存分配
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"🔥 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 CUDA version: {torch.version.cuda}")
        
        # 获取GPU内存信息
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU总内存: {gpu_memory:.1f}GB")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 获取当前可用内存
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"   GPU可用内存: {available_memory / 1e9:.1f}GB")
        
        # 设置内存分数以防止OOM
        torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
        print("   ⚙️ 设置GPU内存使用限制: 90%")
        
    else:
        print("💻 CUDA is not available. Using CPU.")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{project_root}/ultralytics"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_HOME'] = '/home/cui/.cache/torch'
    
    # 设置PyTorch数值稳定性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    # 验证核心模块导入
    try:
        # 添加ultralytics路径
        ultralytics_path = "/home/cui/rtdetr_indoor/ultralytics"
        if ultralytics_path not in sys.path:
            sys.path.insert(0, ultralytics_path)
        
        from ultralytics import RTDETR
        print("✅ RTDETR模块导入成功")
        
        # 验证SEA注意力模块
        from ultralytics.nn.modules.sea_attention import (
            Sea_Attention_Simplified, OptimizedSEA_Attention, TransformerEnhancedSEA
        )
        print("✅ SEA注意力模块导入成功")
        print("  � Sea_Attention_Simplified - 简化版SEA注意力")
        print("  ⚡ OptimizedSEA_Attention - 优化版SEA注意力") 
        print("  🌟 TransformerEnhancedSEA - Transformer增强SEA")
        
        from ultralytics.nn.modules import Conv
        print("✅ 基础模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        print("� 请检查以下配置:")
        print("   1. ultralytics路径是否正确")
        print("   2. SEA注意力模块是否正确注册")
        print("   3. 相关依赖是否安装")
        return False
        print("📝 增强模块不可用，使用标准模块")

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        cached_memory = torch.cuda.memory_reserved(device) / 1e9
        free_memory = total_memory - cached_memory
        
        print(f"\n📊 GPU内存使用情况:")
        print(f"   总内存: {total_memory:.1f}GB")
        print(f"   已分配: {allocated_memory:.1f}GB")
        print(f"   已缓存: {cached_memory:.1f}GB") 
        print(f"   可用内存: {free_memory:.1f}GB")
        
        if free_memory < 2.0:
            print("⚠️ 可用GPU内存不足2GB，建议:")
            print("   1. 进一步减少batch size")
            print("   2. 降低图像分辨率")
            print("   3. 关闭更多数据增强")
            return False
        return True
    return True

def test_model_loading(model_path):
    """测试模型是否能正常加载"""
    try:
        print(f"\n🧪 测试模型加载: {os.path.basename(model_path)}")
        
        from ultralytics import RTDETR
        model = RTDETR(model_path)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✅ 模型加载成功!")
        print(f"📊 总参数量: {total_params:,}")
        
        # 简单的前向传播测试
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            x = torch.randn(1, 3, 640, 640).cuda()
            model.model.cuda()
        else:
            device = 'cpu'
            x = torch.randn(1, 3, 640, 640)
            
        model.model.eval()
        with torch.no_grad():
            output = model.model(x)
        print(f"✅ 前向传播测试通过! (设备: {device})")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 建议检查模型配置或选择其他版本")
        return False

def check_model_config(model_file):
    """检查模型配置文件是否存在"""
    model_config_path = Path(f"/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_file}")
    if not model_config_path.exists():
        print(f"❌ Model config file not found: {model_config_path}")
        return False
    
    print(f"✅ Model config file found: {model_config_path}")
    return True

def check_dataset_config():
    """检查数据集配置文件是否存在"""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    # 检查homeobjects-3K数据集
    dataset_config_path = Path("/home/cui/rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml")
    if dataset_config_path.exists():
        print(f"✅ HomeObjects-3K dataset config found: {dataset_config_path}")
        return True
    
    print(f"❌ Dataset config file not found in both locations")
    print("Please make sure your dataset is properly configured.")
    return False

def train_model(config):
    """训练模型"""
    try:
        # 导入Ultralytics YOLO
        from ultralytics import RTDETR
        
        print("\n🚀 Starting RT-DETR with MobileNetV4 training...")
        print(f"📊 Configuration:")
        for key, value in config.items():
            if key != 'model':  # 不打印完整路径
                print(f"  {key}: {value}")
        print(f"  model: {Path(config['model']).name}")
        
        # 初始化模型
        print("\n📦 Loading RT-DETR with MobileNetV4 model...")
        model = RTDETR(config['model'])
        
        # 开始训练
        print("\n🏃 Starting training...")
        results = model.train(**config)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config['project']}/{config['name']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 RT-DETR with MobileNetV4 Training Script")
    print("🎯 支持多版本选择和模型测试")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 选择模型版本
    selected_file, version_choice = select_model_version()
    
    # 检查配置文件
    if not check_model_config(selected_file):
        sys.exit(1)
    
    # 创建训练配置
    model_path = f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{selected_file}'
    
    # 测试模型加载
    if not test_model_loading(model_path):
        print("\n❌ 模型加载测试失败!")
        print("💡 建议:")
        print("  1. 检查配置文件语法")
        print("  2. 确保所有自定义模块已正确实现")
        print("  3. 选择其他稳定版本")
        print("  4. 或者跳过测试直接开始训练")
        
        print("\n❓ 是否跳过测试直接开始训练? (y/N)")
        choice = input().strip().lower()
        if choice != 'y':
            print("👋 退出程序")
            sys.exit(1)
        else:
            print("⚠️ 跳过模型测试，直接开始训练...")
    
    # 检查数据集
    if not check_dataset_config():
        print("⚠️  Dataset config not found, but continuing with training...")
    
    # 创建训练配置
    config = create_training_config(selected_file, version_choice)
    
    # 检查GPU内存
    print("\n🔍 检查GPU内存使用情况...")
    if not check_gpu_memory():
        print("\n❓ GPU内存可能不足，是否继续训练? (y/N)")
        choice = input().strip().lower()
        if choice != 'y':
            print("👋 退出程序")
            sys.exit(1)
    
    # 显示配置信息
    print("\n📋 训练配置摘要:")
    print(f"  🎯 模型: {os.path.basename(config['model'])}")
    print(f"  📊 数据集: {config['data']}")
    print(f"  🔄 训练轮次: {config['epochs']}")
    print(f"  📦 批次大小: {config['batch']} (内存优化)")
    print(f"  📏 图像尺寸: {config['imgsz']}")
    print(f"  🎓 学习率: {config['lr0']}")
    print(f"  👥 Workers: {config.get('workers', 4)}")
    print(f"  💾 保存路径: {config['project']}/{config['name']}")
    
    # 根据版本显示特点
    if version_choice == '1':
        print("\n📊 原始RT-DETR配置:")
        print("  ✅ 标准Transformer架构")
        print("  📈 基准性能参考")
    elif version_choice == '2':
        print("\n🚀 MobileNetV4混合版本:")
        print("  ⚡ 移动端优化架构")
        print("  🔧 EdgeResidual + UniversalInvertedResidual")
        print("  📈 平衡性能与效率")
    elif version_choice == '3':
        print("\n🌟 SEA注意力优化版本:")
        print("  🧠 Squeeze-enhanced Axial Attention")
        print("  🎯 检测感知的特征提取")
        print("  📈 最高性能预期")
        print("  ⚠️ 使用最小batch size以适应复杂模型")
    
    # 训练模型
    print(f"\n🎯 开始训练 {selected_file} 版本...")
    if version_choice in ['8', '9']:
        if version_choice == '8':
            print("� 使用原始SEA注意力综合优化策略训练!")
            print("🔥 SeaFormer原始实现: 轴向注意力+细节增强+门控机制")
        else:
            print("🎯 使用标准模块综合优化策略训练!")
        print("📈 目标: mAP50从基准线提升12-15%")
        print("🚀 集成策略1+2+4: 特征融合+注意力+架构微调")
        print("⚠️  训练时间: 10-12小时 (最高质量训练)")
    elif version_choice == '7':
        print("📊 使用原始RT-DETR训练!")
        print("📈 基准性能参考")
    else:
        print("🔧 使用标准配置训练!")
        print("📈 标准性能预期")
    
    # 开始训练
    print("=" * 60)
    
    try:
        from ultralytics import RTDETR
        
        # 创建模型
        model = RTDETR(config['model'])
        
        # 开始训练
        results = model.train(**config)
        
        print("\n🎉 训练完成!")
        print(f"📊 最佳结果保存在: {results.save_dir}")
        print("✅ RT-DETR MobileNetV4 训练脚本成功完成!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        print("💡 请检查配置和数据集设置")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
