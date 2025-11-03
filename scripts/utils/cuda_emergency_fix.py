#!/usr/bin/env python3
"""
CUDA错误紧急修复 - 专门处理RTX 4090的CUDA内存访问错误
"""

import os
import sys
import torch
import gc
import warnings

def setup_cuda_debug_mode():
    """设置CUDA调试模式"""
    print("🔧 设置CUDA调试模式...")
    
    # 启用同步CUDA调用以准确定位错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 启用设备端断言
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # 严格的内存检查
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
    
    # 禁用可能导致问题的优化
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
    
    print("✅ CUDA调试模式已启用")

def emergency_gpu_cleanup():
    """紧急GPU清理"""
    print("🧹 执行紧急GPU清理...")
    
    if torch.cuda.is_available():
        # 清空所有CUDA缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 重置CUDA上下文（如果可能）
        try:
            torch.cuda.reset_peak_memory_stats()
            print("✅ CUDA统计已重置")
        except:
            print("⚠️  无法重置CUDA统计")
        
        # 显示当前显存状态
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"📊 显存状态: {allocated:.2f}GB/{total:.1f}GB (缓存:{cached:.2f}GB)")
        
        if allocated > 0.1:
            print("⚠️  检测到未释放的显存")
    
    # Python垃圾回收
    gc.collect()
    print("✅ 紧急清理完成")

def create_safe_training_config():
    """创建超安全的训练配置"""
    config = {
        # 模型配置
        'model': '/home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml',
        'data': '/home/cui/rtdetr_indoor/datasets/indoor_training/data.yaml',
        'project': '/home/cui/rtdetr_indoor',
        'name': 'rtdetr_safe_training',
        
        # 超保守设置
        'epochs': 100,
        'batch': 1,          # 最小批次
        'imgsz': 640,
        'patience': 30,
        
        # 最安全的设备设置
        'device': '0',
        'workers': 0,        # 禁用多进程
        'amp': False,        # 禁用混合精度
        'cache': False,      # 禁用缓存
        'rect': False,       # 禁用矩形训练
        'single_cls': False,
        'save_period': 10,   # 频繁保存
        
        # 保守的优化器设置
        'optimizer': 'SGD',  # 使用更稳定的SGD
        'lr0': 0.001,        # 更小的学习率
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'warmup_epochs': 1.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.01,
        'cos_lr': False,     # 禁用余弦学习率
        
        # 禁用所有数据增强
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # 验证设置
        'val': True,
        'plots': False,      # 禁用绘图
        'save': True,
        'save_txt': False,
        'save_conf': False,
        'save_json': False,
        'half': False,       # 禁用FP16
        'dnn': False,
        'verbose': True,
    }
    
    return config

def safe_train():
    """安全训练模式"""
    try:
        # 1. 设置调试模式
        setup_cuda_debug_mode()
        
        # 2. 紧急清理
        emergency_gpu_cleanup()
        
        # 3. 导入必要的库
        print("📚 导入训练库...")
        from ultralytics import RTDETR
        
        # 4. 设置最保守的PyTorch设置
        print("⚙️  设置PyTorch...")
        torch.backends.cudnn.benchmark = False  # 禁用benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32
        torch.backends.cudnn.allow_tf32 = False
        
        # 设置显存限制为50%
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5)
            print("📊 显存限制设置为50%")
        
        # 5. 创建配置
        config = create_safe_training_config()
        
        print("🚀 开始超安全模式训练...")
        print("⚠️  注意：此模式训练速度较慢，但稳定性最高")
        
        # 6. 初始化模型
        model = RTDETR('rtdetr-l.pt')
        
        # 7. 开始训练
        results = model.train(**config)
        
        print("🎉 安全训练完成!")
        if hasattr(results, 'fitness'):
            fitness_score = results.fitness()
            print(f"📊 最终fitness评分: {fitness_score}")
        
        return results
        
    except Exception as e:
        print(f"❌ 安全训练也失败了: {e}")
        print("🔍 建议检查:")
        print("1. GPU硬件是否正常")
        print("2. CUDA驱动是否需要更新")
        print("3. 是否存在硬件过热问题")
        raise

def cuda_memory_test():
    """CUDA内存测试"""
    print("🧪 执行CUDA内存测试...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 测试小张量
        print("📝 测试小张量...")
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x + y
        del x, y, z
        torch.cuda.empty_cache()
        print("✅ 小张量测试通过")
        
        # 测试中等张量
        print("📝 测试中等张量...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        del x, y, z
        torch.cuda.empty_cache()
        print("✅ 中等张量测试通过")
        
        # 测试大张量（模拟训练）
        print("📝 测试大张量...")
        x = torch.randn(4, 3, 640, 640).cuda()  # 模拟batch
        y = torch.randn(4, 256, 20, 20).cuda()  # 模拟特征图
        del x, y
        torch.cuda.empty_cache()
        print("✅ 大张量测试通过")
        
        print("🎉 CUDA内存测试全部通过")
        return True
        
    except Exception as e:
        print(f"❌ CUDA内存测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚨 CUDA错误紧急修复工具")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            cuda_memory_test()
        elif sys.argv[1] == "clean":
            emergency_gpu_cleanup()
        elif sys.argv[1] == "train":
            safe_train()
    else:
        print("使用方法:")
        print("python cuda_emergency_fix.py test   # 测试CUDA内存")
        print("python cuda_emergency_fix.py clean  # 紧急清理GPU")
        print("python cuda_emergency_fix.py train  # 超安全训练")
        
        choice = input("选择操作 (test/clean/train): ").strip().lower()
        
        if choice == "test":
            cuda_memory_test()
        elif choice == "clean":
            emergency_gpu_cleanup()
        elif choice == "train":
            safe_train()
        else:
            print("无效选择")
