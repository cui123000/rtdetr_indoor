#!/usr/bin/env python3
"""
RT-DETR 模型分析工具
计算模型参数量、FLOPs、推理速度等指标
"""

import torch
import time
import os
import sys
from pathlib import Path
from thop import profile, clever_format
import numpy as np
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

def analyze_model(model_path, input_size=(640, 640), device='cuda', warmup_runs=10, test_runs=50):
    """
    分析模型性能指标
    Args:
        model_path: 模型配置文件路径或权重路径
        input_size: 输入图像大小
        device: 设备
        warmup_runs: 预热运行次数
        test_runs: 测试运行次数
    """
    try:
        from ultralytics import RTDETR
        
        # 加载模型
        print(f"📦 加载模型: {model_path}")
        if model_path.endswith('.pt'):
            # 加载训练好的权重
            model = RTDETR(model_path)
        else:
            # 加载配置文件
            model = RTDETR(model_path)
        
        model.model.eval()
        if device == 'cuda' and torch.cuda.is_available():
            model.model = model.model.cuda()
        
        # 创建输入张量
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, input_size[0], input_size[1])
        if device == 'cuda' and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        results = {}
        
        # 1. 计算参数量
        print("🔢 计算参数量...")
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        results['total_params'] = total_params
        results['trainable_params'] = trainable_params
        results['total_params_M'] = total_params / 1e6
        results['trainable_params_M'] = trainable_params / 1e6
        
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 2. 计算FLOPs
        print("⚡ 计算FLOPs...")
        try:
            # 创建模型副本用于FLOPs计算（避免修改原模型）
            flops_model = model.model.cpu()
            flops_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            flops, params = profile(flops_model, inputs=(flops_input,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            
            results['flops'] = flops
            results['flops_G'] = flops / 1e9
            results['flops_formatted'] = flops_formatted
            
            print(f"   FLOPs: {flops_formatted}")
            
        except Exception as e:
            print(f"   ⚠️ FLOPs计算失败: {e}")
            results['flops'] = None
            results['flops_formatted'] = "N/A"
        
        # 3. 计算推理速度
        if device == 'cuda' and torch.cuda.is_available():
            print("🚀 测试推理速度...")
            model.model = model.model.cuda()
            input_tensor = input_tensor.cuda()
            
            # 预热
            print(f"   预热 {warmup_runs} 次...")
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model.model(input_tensor)
                    torch.cuda.synchronize()
            
            # 正式测试
            print(f"   测试 {test_runs} 次...")
            inference_times = []
            with torch.no_grad():
                for _ in range(test_runs):
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    _ = model.model(input_tensor)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    inference_times.append(end_time - start_time)
            
            # 计算统计指标
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            fps = 1.0 / avg_time
            
            results['avg_inference_time'] = avg_time
            results['std_inference_time'] = std_time
            results['min_inference_time'] = min_time
            results['max_inference_time'] = max_time
            results['fps'] = fps
            
            print(f"   平均推理时间: {avg_time*1000:.2f} ms (±{std_time*1000:.2f})")
            print(f"   FPS: {fps:.1f}")
            print(f"   最快: {min_time*1000:.2f} ms, 最慢: {max_time*1000:.2f} ms")
        else:
            print("   ⚠️ CUDA不可用，跳过推理速度测试")
            results['fps'] = None
        
        # 4. 计算模型大小
        print("💾 计算模型大小...")
        if model_path.endswith('.pt') and os.path.exists(model_path):
            model_size_bytes = os.path.getsize(model_path)
            model_size_mb = model_size_bytes / (1024 * 1024)
            results['model_size_bytes'] = model_size_bytes
            results['model_size_mb'] = model_size_mb
            print(f"   模型大小: {model_size_mb:.2f} MB")
        else:
            results['model_size_mb'] = None
            print("   ⚠️ 无法计算模型大小（配置文件）")
        
        return results
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(model_configs, save_path=None):
    """
    对比多个模型的性能
    Args:
        model_configs: 模型配置字典，格式: {'name': 'path'}
        save_path: 保存结果的路径
    """
    print("📊 开始模型对比分析...")
    results = {}
    
    for name, path in model_configs.items():
        print(f"\n🔍 分析模型: {name}")
        print("=" * 50)
        
        result = analyze_model(path)
        if result:
            results[name] = result
            results[name]['model_path'] = path
    
    # 生成对比表格
    print("\n📋 模型对比表格:")
    print("=" * 100)
    print(f"{'模型名称':<20} {'参数量(M)':<12} {'FLOPs':<15} {'FPS':<10} {'大小(MB)':<12}")
    print("=" * 100)
    
    for name, result in results.items():
        params_m = f"{result['total_params_M']:.2f}" if result['total_params_M'] else "N/A"
        flops_str = result.get('flops_formatted', 'N/A')
        fps_str = f"{result['fps']:.1f}" if result.get('fps') else "N/A"
        size_str = f"{result['model_size_mb']:.2f}" if result.get('model_size_mb') else "N/A"
        
        print(f"{name:<20} {params_m:<12} {flops_str:<15} {fps_str:<10} {size_str:<12}")
    
    # 保存结果
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 结果已保存到: {save_path}")
    
    return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR 模型分析工具')
    parser.add_argument('--model', type=str, help='单个模型路径')
    parser.add_argument('--config', type=str, help='模型配置文件')
    parser.add_argument('--compare', action='store_true', help='对比所有配置模型')
    parser.add_argument('--save', type=str, default='analysis_results.json', help='保存结果路径')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备选择')
    
    args = parser.parse_args()
    
    if args.model:
        # 分析单个模型
        result = analyze_model(args.model, device=args.device)
        if result and args.save:
            with open(args.save, 'w') as f:
                json.dump({Path(args.model).stem: result}, f, indent=2, default=str)
            print(f"💾 结果已保存到: {args.save}")
    
    elif args.compare:
        # 对比所有配置模型
        model_configs = {
            'RT-DETR-L': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml',
            'RT-DETR+MNV4': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml',
            'RT-DETR+MNV4+SEA': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml',
            'RT-DETR-L+SEA': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l-sea.yaml',
            'RT-DETR+GhostNet': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-ghostnet.yaml',
            'RT-DETR+ShuffleNet+SEA': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-shufflenet-sea.yaml',
            'RT-DETR+EfficientNet+CBAM': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-efficientnet-cbam.yaml',
            'RT-DETR-L+CBAM': '/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l-cbam.yaml',
        }
        
        compare_models(model_configs, args.save)
    
    else:
        print("请指定 --model 或 --compare 选项")
        parser.print_help()

if __name__ == "__main__":
    main()