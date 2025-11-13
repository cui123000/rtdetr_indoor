#!/usr/bin/env python3
"""
RT-DETR模型详细参数分析工具
精确统计各层参数分布和计算复杂度
"""

import time
import torch
import torch.nn as nn
import sys
import os
from collections import defaultdict

# 添加正确的模块路径
sys.path.insert(0, '/home/cui/rtdetr_indoor/ultralytics')
os.chdir('/home/cui/rtdetr_indoor')

from ultralytics import RTDETR

def analyze_model(model_path, model_name):
    """详细分析单个模型"""
    print(f"\n{'='*60}")
    print(f"🔍 分析模型: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 创建模型
        model = RTDETR(model_path)
        net = model.model
        
        # 基本信息
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        print(f"📊 参数统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   不可训练参数: {total_params-trainable_params:,}")
        
        # 层类型统计
        layer_stats = defaultdict(lambda: {'count': 0, 'params': 0})
        
        for name, module in net.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                module_type = type(module).__name__
                layer_params = sum(p.numel() for p in module.parameters())
                layer_stats[module_type]['count'] += 1
                layer_stats[module_type]['params'] += layer_params
        
        print(f"\n📋 层类型分布:")
        print(f"{'层类型':<25} {'数量':<8} {'参数量':<12} {'占比'}")
        print("-" * 55)
        
        sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]['params'], reverse=True)
        for layer_type, stats in sorted_layers:
            if stats['params'] > 0:
                ratio = stats['params'] / total_params * 100
                print(f"{layer_type:<25} {stats['count']:<8} {stats['params']:<12,} {ratio:>5.1f}%")
        
        # 推理性能测试
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net.to(device)
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        # 预热
        print(f"\n🔥 性能测试 (预热中...):")
        with torch.no_grad():
            for _ in range(5):
                _ = net(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        iterations = 20
        times = []
        
        with torch.no_grad():
            for i in range(iterations):
                start_time = time.time()
                _ = net(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        print(f"   平均推理时间: {avg_time:.2f} ms")
        print(f"   最快推理时间: {min_time:.2f} ms")
        print(f"   最慢推理时间: {max_time:.2f} ms")
        print(f"   标准差: {std_time:.2f} ms")
        
        # FLOPs估算 (简化版)
        # 注意: 这是一个粗略估算
        input_size = 640 * 640 * 3
        estimated_flops = total_params * input_size * 2  # 粗略估算
        
        print(f"\n💾 内存和计算:")
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            print(f"   GPU内存占用: {memory_allocated:.2f} GB")
        
        print(f"   模型大小: {total_params * 4 / 1e6:.2f} MB (FP32)")
        print(f"   估算FLOPs: {estimated_flops / 1e9:.2f} GFLOPs")
        
        # 网络结构概要
        backbone_params = 0
        head_params = 0
        
        for name, module in net.named_modules():
            if 'backbone' in name or any(x in name for x in ['conv', 'stem', 'stage', 'block']):
                backbone_params += sum(p.numel() for p in module.parameters() if len(list(module.children())) == 0)
            elif 'head' in name or any(x in name for x in ['detect', 'decoder', 'cls', 'bbox']):
                head_params += sum(p.numel() for p in module.parameters() if len(list(module.children())) == 0)
        
        other_params = total_params - backbone_params - head_params
        
        print(f"\n🏗️ 网络结构分布:")
        print(f"   骨干网络: {backbone_params:,} ({backbone_params/1e6:.2f}M, {backbone_params/total_params*100:.1f}%)")
        print(f"   检测头: {head_params:,} ({head_params/1e6:.2f}M, {head_params/total_params*100:.1f}%)")
        print(f"   其他组件: {other_params:,} ({other_params/1e6:.2f}M, {other_params/total_params*100:.1f}%)")
        
        return {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'backbone_params': backbone_params,
            'head_params': head_params,
            'other_params': other_params,
            'layer_stats': dict(layer_stats)
        }
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理
        if 'net' in locals():
            del net
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def main():
    """主函数"""
    print("🚀 RT-DETR 模型详细分析工具")
    print("=" * 60)
    
    models = {
        'RT-DETR-L': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml',
        'RT-DETR-MNV4': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml',
        'RT-DETR-MNV4-SEA': '/home/cui/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml'
    }
    
    results = []
    
    for model_name, model_path in models.items():
        result = analyze_model(model_path, model_name)
        if result:
            results.append(result)
        time.sleep(2)  # 间隔2秒避免GPU冲突
    
    # 生成对比总结
    if results:
        print(f"\n{'='*80}")
        print("📊 三模型对比总结")
        print(f"{'='*80}")
        
        print(f"\n{'模型':<20} {'参数(M)':<12} {'推理时间(ms)':<15} {'骨干网络(M)':<15} {'检测头(M)':<12}")
        print("-" * 80)
        
        for result in results:
            name = result['name']
            params = result['total_params'] / 1e6
            time_ms = result['avg_time']
            backbone = result['backbone_params'] / 1e6
            head = result['head_params'] / 1e6
            
            print(f"{name:<20} {params:<12.2f} {time_ms:<15.2f} {backbone:<15.2f} {head:<12.2f}")
        
        # 效率分析
        print(f"\n🏆 性能排名:")
        
        # 按参数量排序 (越少越好)
        by_params = sorted(results, key=lambda x: x['total_params'])
        print(f"   参数效率: {' > '.join([r['name'] for r in by_params])}")
        
        # 按推理时间排序 (越快越好)
        by_speed = sorted(results, key=lambda x: x['avg_time'])
        print(f"   推理速度: {' > '.join([r['name'] for r in by_speed])}")
        
        # 计算参数效率比 (相对于RT-DETR-L)
        baseline = next(r for r in results if 'RT-DETR-L' in r['name'])
        
        print(f"\n📈 相对RT-DETR-L的效率:")
        for result in results:
            if result['name'] != baseline['name']:
                param_ratio = result['total_params'] / baseline['total_params']
                speed_ratio = result['avg_time'] / baseline['avg_time']
                efficiency = (1 / param_ratio) * (1 / speed_ratio)  # 综合效率指标
                
                print(f"   {result['name']}: 参数比={param_ratio:.2f}, 速度比={speed_ratio:.2f}, 效率比={efficiency:.2f}")

if __name__ == "__main__":
    main()