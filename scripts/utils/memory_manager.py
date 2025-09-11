#!/usr/bin/env python3
"""
GPU内存监控和清理工具
"""

import torch
import gc
import os
import time
import psutil
from pathlib import Path

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    total_memory = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    free = total_memory - allocated
    
    print(f"🔥 GPU: {props.name}")
    print(f"   总内存: {total_memory:.2f}GB")
    print(f"   已分配: {allocated:.2f}GB ({allocated/total_memory*100:.1f}%)")
    print(f"   已缓存: {cached:.2f}GB ({cached/total_memory*100:.1f}%)")
    print(f"   可用内存: {free:.2f}GB ({free/total_memory*100:.1f}%)")
    
    # 内存使用警告
    if allocated/total_memory > 0.8:
        print("⚠️  警告: GPU内存使用率超过80%")
    if allocated/total_memory > 0.9:
        print("🚨 危险: GPU内存使用率超过90%，建议清理")
    
    return {
        'total': total_memory,
        'allocated': allocated, 
        'cached': cached,
        'free': free,
        'usage_percent': allocated/total_memory*100
    }

def check_system_memory():
    """检查系统内存"""
    memory = psutil.virtual_memory()
    
    print(f"💻 系统内存:")
    print(f"   总内存: {memory.total / 1e9:.2f}GB")
    print(f"   已使用: {memory.used / 1e9:.2f}GB ({memory.percent:.1f}%)")
    print(f"   可用内存: {memory.available / 1e9:.2f}GB")
    
    if memory.percent > 80:
        print("⚠️  警告: 系统内存使用率超过80%")

def cleanup_gpu_memory():
    """清理GPU内存"""
    if not torch.cuda.is_available():
        return
    
    print("🧹 清理GPU内存...")
    
    # 清理PyTorch缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 强制垃圾回收
    gc.collect()
    
    # 等待一秒让清理完成
    time.sleep(1)
    
    print("✅ GPU内存清理完成")

def set_memory_optimization():
    """设置内存优化参数"""
    print("⚙️ 设置内存优化参数...")
    
    # CUDA内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    
    # PyTorch优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if torch.cuda.is_available():
        # 设置内存分数
        torch.cuda.set_per_process_memory_fraction(0.85)
        print("   GPU内存限制: 85%")
    
    # 设置线程数
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    print("✅ 内存优化设置完成")

def monitor_training_memory(duration_minutes=60):
    """监控训练期间的内存使用"""
    print(f"📊 开始监控内存使用 ({duration_minutes}分钟)...")
    
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    
    max_gpu_usage = 0
    max_system_usage = 0
    
    try:
        while time.time() < end_time:
            os.system('clear')  # 清屏
            
            print("📊 实时内存监控")
            print("=" * 50)
            
            # GPU内存
            gpu_info = check_gpu_memory()
            if gpu_info:
                max_gpu_usage = max(max_gpu_usage, gpu_info['usage_percent'])
            
            print()
            
            # 系统内存
            check_system_memory()
            system_memory = psutil.virtual_memory()
            max_system_usage = max(max_system_usage, system_memory.percent)
            
            print()
            print(f"📈 峰值使用率:")
            print(f"   GPU最高: {max_gpu_usage:.1f}%")
            print(f"   系统最高: {max_system_usage:.1f}%")
            
            print(f"\n⏰ 监控时间: {(time.time() - start_time)/60:.1f}/{duration_minutes}分钟")
            print("按 Ctrl+C 停止监控")
            
            # 自动清理高内存使用
            if gpu_info and gpu_info['usage_percent'] > 85:
                print("\n🧹 自动清理GPU内存...")
                cleanup_gpu_memory()
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n⏹️ 监控停止")

def optimize_for_training():
    """为训练优化系统"""
    print("🚀 为训练优化系统...")
    
    # 设置内存优化
    set_memory_optimization()
    
    # 清理初始内存
    cleanup_gpu_memory()
    
    # 检查当前状态
    print("\n📊 优化后的系统状态:")
    check_gpu_memory()
    print()
    check_system_memory()
    
    print("\n✅ 系统优化完成，可以开始训练")

def main():
    """主菜单"""
    while True:
        print("\n🔧 GPU内存管理工具")
        print("=" * 30)
        print("1. 检查内存状态")
        print("2. 清理GPU内存")
        print("3. 设置内存优化")
        print("4. 优化系统用于训练")
        print("5. 监控训练内存(实时)")
        print("6. 退出")
        
        try:
            choice = input("\n请选择操作 (1-6): ").strip()
            
            if choice == '1':
                print("\n📊 检查内存状态:")
                check_gpu_memory()
                print()
                check_system_memory()
                
            elif choice == '2':
                cleanup_gpu_memory()
                print("\n当前状态:")
                check_gpu_memory()
                
            elif choice == '3':
                set_memory_optimization()
                
            elif choice == '4':
                optimize_for_training()
                
            elif choice == '5':
                try:
                    duration = int(input("监控时长(分钟, 默认60): ") or "60")
                    monitor_training_memory(duration)
                except ValueError:
                    print("❌ 请输入有效的数字")
                    
            elif choice == '6':
                print("👋 退出")
                break
                
            else:
                print("❌ 请输入 1-6")
                
        except KeyboardInterrupt:
            print("\n👋 退出")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
