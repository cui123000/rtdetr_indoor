#!/usr/bin/env python3
"""
测试训练各个阶段，诊断卡顿位置
"""

import os
import sys
import time
import torch
from pathlib import Path

print("=" * 70)
print("🔧 RT-DETR 训练阶段诊断工具")
print("=" * 70)

# 第一阶段：环境设置
print("\n[1/5] 阶段 1: 设置环境...")
start = time.time()
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

print(f"✅ 环境设置完成 ({time.time() - start:.2f}s)")

# 第二阶段：GPU 检查
print("\n[2/5] 阶段 2: 检查 GPU...")
start = time.time()
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ 显存: {gpu_memory:.1f}GB")
else:
    print("❌ CUDA 不可用")
    sys.exit(1)
print(f"✅ GPU 检查完成 ({time.time() - start:.2f}s)")

# 第三阶段：导入 Ultralytics
print("\n[3/5] 阶段 3: 导入 Ultralytics（这可能需要 30-60 秒）...")
start = time.time()
try:
    from ultralytics import RTDETR
    print(f"✅ Ultralytics 导入完成 ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"❌ Ultralytics 导入失败: {e}")
    sys.exit(1)

# 第四阶段：加载模型
print("\n[4/5] 阶段 4: 加载 RT-DETR 模型（这可能需要 1-2 分钟）...")
model_path = "/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"
start = time.time()
try:
    print(f"   加载模型: {model_path}")
    model = RTDETR(model_path)
    print(f"✅ 模型加载完成 ({time.time() - start:.2f}s)")
    print(f"   模型类型: {type(model)}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第五阶段：数据集验证
print("\n[5/5] 阶段 5: 验证数据集配置...")
start = time.time()
try:
    import yaml
    dataset_path = "/home/cjj/rtdetr_indoor/datasets/coco_indoor/coco_indoor.yaml"
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    dataset_root = Path(dataset_config['path'])
    train_dir = dataset_root / dataset_config['train']
    val_dir = dataset_root / dataset_config['val']
    
    train_count = len(list(train_dir.glob('*.jpg')))
    val_count = len(list(val_dir.glob('*.jpg')))
    
    print(f"✅ 训练图像: {train_count}")
    print(f"✅ 验证图像: {val_count}")
    print(f"✅ 类别数: {dataset_config['nc']}")
    print(f"✅ 数据集验证完成 ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"❌ 数据集验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有诊断阶段完成！训练环境就绪。")
print("=" * 70)
print("\n💡 现在可以运行完整训练脚本:")
print("   python3 scripts/training/auto_train_rtdetr.py")
print("\n📝 如果上面某个阶段卡住超过 2 分钟，说明有问题。")
print("   最常见的是模型加载和数据集加载，这是正常的长时间操作。")
