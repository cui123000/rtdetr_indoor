#!/usr/bin/env python3
"""
RT-DETR with MobileNetV4 Training Script with Version Selection
使用Ultralytics框架训练RT-DETR with MobileNetV4模型 - 支持版本选择
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

# 定义可用的模型版本
MODEL_VERSIONS = {
    '1': {
        'name': 'Basic Version (基础版本)',
        'file': 'rtdetr-mnv4-basic.yaml',
        'description': '使用基础模块，最稳定',
        'modules': ['Conv', 'C2f', 'SPPF'],
        'status': '✅ 稳定'
    },
    '2': {
        'name': 'Stable Version (稳定版本)',
        'file': 'rtdetr-mnv4-stable.yaml',
        'description': '添加轻量级模块，平衡性能',
        'modules': ['Conv', 'C2f', 'SPPF', 'GhostBottleneck', 'RepC3'],
        'status': '✅ 推荐'
    },
    '3': {
        'name': 'Advanced Version (高级版本)',
        'file': 'rtdetr-mnv4-advanced.yaml',
        'description': '集成注意力机制，高性能',
        'modules': ['Conv', 'C2f', 'SPPF', 'CBAM', 'GhostBottleneck', 'RepC3'],
        'status': '⚠️ 需要CBAM支持'
    },
    '4': {
        'name': 'Hybrid Version (混合版本)',
        'file': 'rtdetr-mnv4-hybrid-m.yaml',
        'description': '原生MobileNetV4模块，最完整',
        'modules': ['EdgeResidual', 'UniversalInvertedResidual', 'C2f', 'RepC3'],
        'status': '🚀 最新'
    }
}

def select_model_version():
    """选择模型版本"""
    print("\n📋 可用的RT-DETR + MobileNetV4版本:")
    print("=" * 60)
    
    for key, version in MODEL_VERSIONS.items():
        print(f"{key}. {version['name']}")
        print(f"   📄 文件: {version['file']}")
        print(f"   📝 描述: {version['description']}")
        print(f"   🧩 模块: {', '.join(version['modules'])}")
        print(f"   📊 状态: {version['status']}")
        print()
    
    while True:
        try:
            choice = input("请选择版本 (1-4): ").strip()
            if choice in MODEL_VERSIONS:
                selected = MODEL_VERSIONS[choice]
                print(f"\n✅ 已选择: {selected['name']}")
                print(f"📄 配置文件: {selected['file']}")
                return selected['file']
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 输入错误: {e}")

def create_training_config(model_file):
    """创建训练配置文件"""
    model_path = f'/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_file}'
    
    config = {
        # 基本设置
        'task': 'detect',
        'mode': 'train',
        
        # 模型和数据
        'model': model_path,
        'data': '/home/cui/vild_rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml',
        
        # 训练参数
        'epochs': 100,
        'batch': 4,
        'imgsz': 640,
        'patience': 50,
        
        # 保存设置
        'save': True,
        'save_period': 10,
        'project': str(project_root),
        'name': f'rtdetr_mobilenetv4_{model_file.replace(".yaml", "").replace("-", "_")}',
        'exist_ok': True,
        
        # 优化器和学习率
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # 设备设置
        'device': '0',
        'workers': 4,
        'amp': True,
        
        # 验证设置
        'val': True,
        'conf': 0.25,
        'iou': 0.7,
        'max_det': 300,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # 其他设置
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'plots': True,
        'cache': False,
        'pretrained': False,  # 不使用预训练，从头开始训练
    }
    
    return config

def setup_environment():
    """设置训练环境"""
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"🔥 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 CUDA version: {torch.version.cuda}")
    else:
        print("💻 CUDA is not available. Using CPU.")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{project_root}/ultralytics"

def test_model_loading(model_path):
    """测试模型是否能正常加载"""
    try:
        print(f"\n🧪 测试模型加载: {model_path}")
        
        from ultralytics import RTDETR
        model = RTDETR(model_path)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✅ 模型加载成功!")
        print(f"📊 总参数量: {total_params:,}")
        
        # 简单的前向传播测试
        import torch
        x = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            output = model.model(x)
        print(f"✅ 前向传播测试通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 建议选择其他版本或检查模块实现")
        return False

def check_model_config(model_file):
    """检查模型配置文件是否存在"""
    model_config_path = Path(f"/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_file}")
    if not model_config_path.exists():
        print(f"❌ Model config file not found: {model_config_path}")
        return False
    
    print(f"✅ Model config file found: {model_config_path}")
    return True

def check_dataset_config():
    """检查数据集配置文件是否存在"""
    dataset_config_path = Path("/home/cui/vild_rtdetr_indoor/datasets/homeobjects-3K/HomeObjects-3K.yaml")
    if not dataset_config_path.exists():
        print(f"❌ Dataset config file not found: {dataset_config_path}")
        print("Please make sure your dataset is properly configured.")
        return False
    
    print(f"✅ Dataset config file found: {dataset_config_path}")
    return True

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
    selected_file = select_model_version()
    
    # 检查配置文件
    if not check_model_config(selected_file):
        sys.exit(1)
    
    # 创建训练配置
    model_path = f'/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{selected_file}'
    
    # 测试模型加载
    if not test_model_loading(model_path):
        print("\n❌ 模型加载测试失败，是否继续训练? (y/N)")
        choice = input().strip().lower()
        if choice != 'y':
            print("👋 退出程序")
            sys.exit(1)
    
    # 检查数据集
    if not check_dataset_config():
        print("⚠️  Dataset config not found, but continuing with training...")
    
    # 创建训练配置
    config = create_training_config(selected_file)
    
    # 训练模型
    print(f"\n🎯 开始训练 {selected_file} 版本...")
    results = train_model(config)
    
    if results is not None:
        print("\n✅ Training script completed successfully!")
        print(f"🎊 恭喜! RT-DETR + MobileNetV4 融合模型训练完成!")
        sys.exit(0)
    else:
        print("\n❌ Training script failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
