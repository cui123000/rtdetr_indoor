#!/usr/bin/env python3
"""
为RTX 4090优化的RT-DETR训练脚本
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
import multiprocessing

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def fix_file_descriptor_limit():
    """修复文件描述符限制问题"""
    print("🔧 修复文件描述符限制...")
    
    try:
        # 获取当前限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"   当前文件描述符限制: {soft} (软限制) / {hard} (硬限制)")
        
        # 设置更高的软限制
        new_soft = min(65536, hard)  # 设置为65536或硬限制的较小值
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        
        print(f"   ✅ 新的文件描述符限制: {new_soft}")
        
        # 设置环境变量限制workers
        os.environ['TORCH_NUM_WORKERS'] = '2'  # 强制限制workers数量
        
    except Exception as e:
        print(f"   ⚠️ 无法修改文件描述符限制: {e}")
        print("   💡 建议在系统级别增加文件描述符限制")

def setup_rtx4090_optimization():
    """为RTX 4090设置专门的优化"""
    print("🚀 为RTX 4090设置专门优化...")
    
    # 首先修复文件描述符问题
    fix_file_descriptor_limit()
    
    # RTX 4090专用CUDA设置 - 保守配置避免驱动错误
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:False'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # 启用RTX 4090的优化特性，但更保守
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置合理的线程数
    torch.set_num_threads(4)  # 减少线程数避免资源竞争
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    if torch.cuda.is_available():
        # RTX 4090显存充足，但保守使用85%避免OOM
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # 清理初始缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ 显存限制: 85% (~22GB)")
        
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
        print(f"   ✅ Flash Attention: 启用")

def memory_monitor_rtx4090():
    """RTX 4090专用内存监控"""
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                
                # RTX 4090显存阈值更高
                if allocated > 20.0:  # 20GB以上时清理
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"🧹 自动清理GPU内存: {allocated:.1f}GB -> {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
            
            time.sleep(15)  # 每15秒检查一次
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def get_rtx4090_config(model_choice):
    """RTX 4090优化的训练配置"""

    model_configs = {
        '1': {
            'file': 'rtdetr-l.yaml',
            'name': 'rtdetr_l_homeobjects_smart',
            'batch': 16,
            'lr0': 0.003,
            'workers': 6,
        },
        '2': {
            'file': 'rtdetr-mnv4-hybrid-m.yaml',
            'name': 'rtdetr_mnv4_hybrid_rtx4090',
            'batch': 8,
            'lr0': 0.0015,
            'workers': 4,
        },
        '3': {
            'file': 'rtdetr-mnv4-hybrid-m-sea.yaml',
            'name': 'rtdetr_mnv4_sea_rtx4090',
            'batch': 6,
            'lr0': 0.0012,
            'workers': 4,
        },
        '4': {
            'file': 'rtdetr-l-sea.yaml',
            'name': 'rtdetr_l_sea_rtx4090',
            'batch': 12,
            'lr0': 0.0025,
            'workers': 5,
        },
        '11': {
            'file': 'ert-detr.yaml',
            'name': 'ert_detr_rtx4090',
            'batch': 8,
            'lr0': 0.0008,
            'workers': 4,
        }
    }

    if model_choice not in model_configs:
        raise ValueError(f"无效的模型选择: {model_choice}")

    model_config = model_configs[model_choice]

    # RTX 4090专用配置
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': f'/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/{model_config["file"]}',
                'data': '/home/cui/rtdetr_indoor/datasets/homeobjects_extended_yolo_smart/homeobjects_extended_smart.yaml',

        # RTX 4090优化的核心参数
        'epochs': 150,          # 增加训练轮数以充分训练
        'batch': model_config['batch'],
        'imgsz': 640,
        'patience': 25,         # 增加patience避免过早停止

        # 稳定性优化设置 - 防止内存泄漏
        'device': '0',
        'workers': model_config['workers'],
        'amp': True,            # 启用混合精度训练
        'cache': 'ram',         # 缓存到内存以加速数据加载
        'rect': True,           # 矩形训练
        'single_cls': False,

        # RTX 4090优化的学习率设置
        'optimizer': 'AdamW',
        'lr0': model_config['lr0'],
        'lrf': 0.001,           # 更激进的学习率衰减
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,

        # 内存安全的数据增强设置 - 为了最快速度关闭复杂增强
        'hsv_h': 0.01,          # 减少HSV增强以加速
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 0.0,         # 关闭旋转减少计算
        'translate': 0.05,      # 减少平移增强
        'scale': 0.3,           # 减少缩放增强
        'shear': 0.0,           # 关闭剪切减少计算
        'perspective': 0.0,     # 关闭透视变换防止内存泄漏
        'flipud': 0.0,
        'fliplr': 0.3,          # 减少水平翻转
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

        # 保存设置 - 只保存最佳权重到指定位置
                # 保存设置 - 只保存最佳权重到指定位置
        'save': True,
        'save_period': -1,      # 禁用定期保存
        'save_json': False,     # 不保存JSON结果
        'save_hybrid': False,   # 不保存混合格式
        'save_crop': False,     # 不保存裁剪图像
        'save_txt': False,      # 不保存txt结果
        'project': '/root/autodl-tmp/rtdetr_weights',  # 现在cui用户有权限了
        'name': model_config['name'],
        'exist_ok': True,

        # RTX 4090专用设置
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'plots': True,
        'close_mosaic': 10,
        'overlap_mask': True,   # RTX 4090可以处理重叠mask
        'mask_ratio': 4,

        # 高级优化设置
        'profile': False,       # 关闭性能分析以提高速度
        'half': False,          # RTX 4090用FP16可能不稳定，用AMP就够了
        'dnn': False,           # 不使用OpenCV DNN
    }

    return config

def train_with_rtx4090_optimization(model_choice):
    """RTX 4090优化训练"""
    try:
        # 设置环境
        setup_rtx4090_optimization()
        memory_monitor_rtx4090()
        
        # 导入ultralytics
        print("📦 导入Ultralytics...")
        from ultralytics import RTDETR
        
        # 获取配置
        config = get_rtx4090_config(model_choice)
        
        print(f"\n🚀 RTX 4090优化训练开始")
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
        
        # 复制最佳权重到指定位置
        try:
            import shutil
            from pathlib import Path
            
            # 查找最佳权重文件
            project_dir = Path(config['project']) / config['name']
            best_weight = project_dir / 'weights' / 'best.pt'
            last_weight = project_dir / 'weights' / 'last.pt'
            
            if best_weight.exists():
                # 复制最佳权重
                final_path = Path('/root/autodl-tmp') / f"{config['name']}_best.pt"
                shutil.copy2(best_weight, final_path)
                print(f"✅ 最佳权重已保存到: {final_path}")
                
                # 创建权重信息文件
                info_file = final_path.with_suffix('.txt')
                with open(info_file, 'w') as f:
                    f.write(f"模型: {config['model'].split('/')[-1]}\n")
                    f.write(f"数据集: HomeObjects扩展版 (智能筛选)\n")
                    f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"批次大小: {config['batch']}\n")
                    f.write(f"学习率: {config['lr0']}\n")
                    f.write(f"训练轮数: {config['epochs']}\n")
                
                print(f"📝 权重信息已保存到: {info_file}")
            else:
                print("❌ 未找到最佳权重文件")
                
        except Exception as e:
            print(f"❌ 权重复制失败: {e}")
        
        # 使用正确的属性获取训练结果
        if hasattr(results, 'fitness'):
            fitness_score = results.fitness()
            print(f"📊 最终fitness评分: {fitness_score}")
        elif hasattr(results, 'mean_results'):
            mean_results = results.mean_results()
            print(f"📊 平均结果: P={mean_results[0]:.3f}, R={mean_results[1]:.3f}, mAP50={mean_results[2]:.3f}, mAP50-95={mean_results[3]:.3f}")
        
        # 最终清理
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

def quick_speed_test():
    """快速速度测试"""
    print("⚡ RTX 4090速度测试")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    # 测试不同batch size的实际速度
    batch_sizes = [4, 6, 8, 12, 16]
    img_size = 640
    
    for batch_size in batch_sizes:
        try:
            print(f"\n📊 测试 batch_size={batch_size}")
            
            # 模拟RT-DETR输入
            data = torch.randn(batch_size, 3, img_size, img_size).cuda()
            
            # 模拟一个简单的前向传播
            conv1 = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            conv2 = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    x = conv1(data)
                    x = torch.relu(x)
                    x = conv2(x)
            
            torch.cuda.synchronize()
            
            # 计时
            start_time = time.time()
            iterations = 50
            
            with torch.no_grad():
                for _ in range(iterations):
                    x = conv1(data)
                    x = torch.relu(x)
                    x = conv2(x)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            fps = iterations * batch_size / total_time
            memory_used = torch.cuda.memory_allocated() / 1e9
            
            print(f"   处理速度: {fps:.2f} imgs/sec")
            print(f"   GPU内存: {memory_used:.2f}GB")
            print(f"   每批次时间: {total_time/iterations*1000:.2f}ms")
            
            # 清理
            del data, conv1, conv2, x
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"   ❌ OOM - batch_size={batch_size}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            torch.cuda.empty_cache()

def main():
    """主函数"""
    print("🏎️  RTX 4090专用RT-DETR训练优化器")
    print("=" * 50)

    while True:
        print("\n📋 选项:")
        print("1. RTX 4090速度测试")
        print("2. 开始优化训练 - RT-DETR-L")
        print("3. 开始优化训练 - RT-DETR+MNV4")
        print("4. 开始优化训练 - RT-DETR+MNV4+SEA")
        print("5. 内存状态检查")
        print("6. 退出")

        try:
            choice = input("\n请选择 (1-6): ").strip()

            if choice == '1':
                quick_speed_test()

            elif choice in ['2', '3', '4']:
                model_map = {'2': '1', '3': '2', '4': '3'}
                model_choice = model_map[choice]

                confirm = input(f"确认开始训练? (y/n): ").strip().lower()
                if confirm == 'y':
                    train_with_rtx4090_optimization(model_choice)
                else:
                    print("❌ 取消训练")

            elif choice == '5':
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    cached = torch.cuda.memory_reserved() / 1e9
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9

                    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
                    print(f"   总显存: {total:.1f}GB")
                    print(f"   已使用: {allocated:.1f}GB ({allocated/total*100:.1f}%)")
                    print(f"   已缓存: {cached:.1f}GB ({cached/total*100:.1f}%)")
                    print(f"   可用: {total-allocated:.1f}GB")
                else:
                    print("❌ CUDA不可用")

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
