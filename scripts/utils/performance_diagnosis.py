#!/usr/bin/env python3
"""
训练性能诊断工具
分析训练速度瓶颈和内存问题
"""

import torch
import time
import os
import psutil
import subprocess
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def diagnose_system():
    """诊断系统性能"""
    print("🔍 系统性能诊断")
    print("=" * 50)
    
    # CPU信息
    print(f"💻 CPU: {psutil.cpu_count()} cores @ {psutil.cpu_freq().max:.0f}MHz")
    print(f"   使用率: {psutil.cpu_percent(interval=1):.1f}%")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"🧠 内存: {memory.total/1e9:.1f}GB 总量")
    print(f"   使用率: {memory.percent:.1f}%")
    print(f"   可用: {memory.available/1e9:.1f}GB")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU内存: {props.total_memory/1e9:.1f}GB")
        print(f"   计算能力: {props.major}.{props.minor}")
        try:
            print(f"   多处理器: {props.multi_processor_count}")
        except AttributeError:
            print(f"   多处理器: 无法获取")
    else:
        print("❌ GPU不可用")
    
    # 磁盘I/O
    disk = psutil.disk_usage('/')
    print(f"💾 磁盘: {disk.free/1e9:.1f}GB 可用 / {disk.total/1e9:.1f}GB 总量")

def test_gpu_performance():
    """测试GPU性能"""
    print("\n🚀 GPU性能测试")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，跳过测试")
        return
    
    device = torch.device('cuda:0')
    
    # 测试不同batch size的性能
    batch_sizes = [1, 2, 4, 8]
    img_size = 640
    
    for batch_size in batch_sizes:
        try:
            print(f"\n📊 测试 batch_size={batch_size}")
            
            # 创建随机数据
            data = torch.randn(batch_size, 3, img_size, img_size).to(device)
            
            # 预热
            for _ in range(5):
                _ = torch.nn.functional.conv2d(data, torch.randn(64, 3, 3, 3).to(device))
            
            torch.cuda.synchronize()
            
            # 计时测试
            start_time = time.time()
            iterations = 20
            
            for _ in range(iterations):
                _ = torch.nn.functional.conv2d(data, torch.randn(64, 3, 3, 3).to(device))
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            fps = iterations * batch_size / (end_time - start_time)
            memory_used = torch.cuda.memory_allocated() / 1e9
            
            print(f"   FPS: {fps:.2f}")
            print(f"   GPU内存: {memory_used:.2f}GB")
            
            # 清理内存
            del data
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"   ❌ OOM - batch_size={batch_size} 超出内存限制")
            torch.cuda.empty_cache()

def test_dataloader_performance():
    """测试数据加载性能"""
    print("\n📦 数据加载性能测试") 
    print("=" * 30)
    
    dataset_path = "/home/cui/rtdetr_indoor/datasets/homeobjects-3K"
    
    if not Path(dataset_path).exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    try:
        from ultralytics.data import YOLODataset
        from torch.utils.data import DataLoader
        
        # 测试不同worker数量
        worker_counts = [0, 2, 4, 8]
        batch_size = 4
        
        for num_workers in worker_counts:
            try:
                print(f"\n🔄 测试 workers={num_workers}")
                
                # 创建数据集
                dataset = YOLODataset(
                    img_path=f"{dataset_path}/images/train",
                    imgsz=640,
                    augment=False,
                    cache=False
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    pin_memory=True
                )
                
                # 计时
                start_time = time.time()
                batch_count = 0
                
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 20:  # 只测试前20个batch
                        break
                
                end_time = time.time()
                
                if batch_count > 0:
                    speed = batch_count / (end_time - start_time)
                    print(f"   速度: {speed:.2f} batches/sec")
                    print(f"   图片/秒: {speed * batch_size:.2f}")
                
            except Exception as e:
                print(f"   ❌ 错误: {e}")
    
    except ImportError as e:
        print(f"❌ 无法导入ultralytics: {e}")

def diagnose_model_complexity():
    """诊断模型复杂度"""
    print("\n🧠 模型复杂度分析")
    print("=" * 30)
    
    models = {
        'RT-DETR-L': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml',
        'RT-DETR+MNV4': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml',
        'RT-DETR+MNV4+SEA': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml'
    }
    
    try:
        from ultralytics import RTDETR
        
        for name, config_path in models.items():
            if not Path(config_path).exists():
                print(f"❌ {name}: 配置文件不存在 {config_path}")
                continue
                
            try:
                print(f"\n📋 {name}:")
                
                # 创建模型
                model = RTDETR(config_path)
                
                # 计算参数量
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                
                print(f"   总参数: {total_params/1e6:.2f}M")
                print(f"   可训练参数: {trainable_params/1e6:.2f}M")
                
                # 测试前向传播速度
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    model.model.to(device)
                    
                    # 创建测试输入
                    test_input = torch.randn(1, 3, 640, 640).to(device)
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model.model(test_input)
                    
                    torch.cuda.synchronize()
                    
                    # 计时
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(20):
                            _ = model.model(test_input)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) / 20 * 1000  # ms
                    fps = 1000 / inference_time
                    
                    print(f"   推理时间: {inference_time:.2f}ms")
                    print(f"   FPS: {fps:.2f}")
                    
                    # 内存使用
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"   GPU内存: {memory_used:.2f}GB")
                
                # 清理
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ 错误: {e}")
    
    except ImportError as e:
        print(f"❌ 无法导入RTDETR: {e}")

def get_optimization_suggestions():
    """获取优化建议"""
    print("\n💡 优化建议")
    print("=" * 30)
    
    suggestions = []
    
    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:
            suggestions.append("🔸 GPU内存较小(<8GB), 建议使用batch_size=2-3")
        elif gpu_memory < 12:
            suggestions.append("🔸 GPU内存中等(<12GB), 建议使用batch_size=4-6")
        else:
            suggestions.append("🔸 GPU内存充足(≥12GB), 可以使用batch_size=6-8")
    
    # 检查CPU
    cpu_count = psutil.cpu_count()
    if cpu_count < 8:
        suggestions.append("🔸 CPU核心较少, 建议workers=2-4")
    else:
        suggestions.append("🔸 CPU核心充足, 可以使用workers=4-8")
    
    # 检查系统内存
    memory = psutil.virtual_memory()
    if memory.total < 16e9:
        suggestions.append("🔸 系统内存较小(<16GB), 建议关闭cache, 减少workers")
    
    # 通用优化建议
    suggestions.extend([
        "🔸 使用混合精度训练 (amp=True)",
        "🔸 启用cuDNN benchmark (torch.backends.cudnn.benchmark=True)",
        "🔸 关闭不必要的数据增强 (mixup=0, copy_paste=0)",
        "🔸 使用矩形训练 (rect=True)",
        "🔸 定期清理GPU缓存",
        "🔸 对于SEA模型, 使用更小的batch_size和学习率",
        "🔸 监控内存使用, 避免内存泄漏",
    ])
    
    for suggestion in suggestions:
        print(suggestion)

def main():
    """主函数"""
    print("🩺 RT-DETR训练性能诊断工具")
    print("=" * 40)
    
    while True:
        print("\n📋 诊断选项:")
        print("1. 系统性能诊断")
        print("2. GPU性能测试")
        print("3. 数据加载性能测试")
        print("4. 模型复杂度分析") 
        print("5. 获取优化建议")
        print("6. 完整诊断")
        print("7. 退出")
        
        try:
            choice = input("\n请选择 (1-7): ").strip()
            
            if choice == '1':
                diagnose_system()
            elif choice == '2':
                test_gpu_performance()
            elif choice == '3':
                test_dataloader_performance()
            elif choice == '4':
                diagnose_model_complexity()
            elif choice == '5':
                get_optimization_suggestions()
            elif choice == '6':
                print("🔍 执行完整诊断...")
                diagnose_system()
                test_gpu_performance()
                test_dataloader_performance()
                diagnose_model_complexity()
                get_optimization_suggestions()
            elif choice == '7':
                print("👋 退出")
                break
            else:
                print("❌ 请输入 1-7")
                
        except KeyboardInterrupt:
            print("\n👋 退出")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
