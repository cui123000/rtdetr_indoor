#!/usr/bin/env python3
"""
模型性能评估脚本 - 轻量化对比分析
用于对比RT-DETR和ERT-DETR的性能指标
"""

import os
import sys
import torch
import time
import json
import numpy as np
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ultralytics"))

class ModelEvaluator:
    """
    模型性能评估器
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
    
    def count_parameters(self, model):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        }
    
    def calculate_flops(self, model, input_size=(640, 640)):
        """计算FLOPs"""
        try:
            from thop import profile, clever_format
            
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            model.eval()
            
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            return {
                'flops': flops,
                'flops_G': flops / 1e9,
                'flops_str': flops_str,
            }
        except Exception as e:
            print(f"FLOPs计算失败: {e}")
            return {'flops': None, 'flops_G': None, 'flops_str': 'N/A'}
    
    def benchmark_inference(self, model, batch_size=1, input_size=(640, 640), 
                            warmup_runs=10, test_runs=100):
        """基准推理性能测试"""
        dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(self.device)
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
                torch.cuda.synchronize()
        
        # 测试
        inference_times = []
        with torch.no_grad():
            for _ in range(test_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
                inference_times.append(end - start)
        
        inference_times = np.array(inference_times)
        
        return {
            'avg_time_ms': np.mean(inference_times) * 1000,
            'std_time_ms': np.std(inference_times) * 1000,
            'min_time_ms': np.min(inference_times) * 1000,
            'max_time_ms': np.max(inference_times) * 1000,
            'fps': batch_size / np.mean(inference_times),
            'fps_std': batch_size / np.std(inference_times) if np.std(inference_times) > 0 else 0
        }
    
    def evaluate_model(self, model_config_path, model_name="Model"):
        """完整的模型评估"""
        print(f"\n{'='*70}")
        print(f"评估模型: {model_name}")
        print(f"配置文件: {model_config_path}")
        print(f"{'='*70}")
        
        try:
            from ultralytics import RTDETR
            
            # 加载模型
            print("📦 加载模型...")
            model = RTDETR(model_config_path)
            model = model.model.to(self.device)
            
            results = {
                'model_name': model_name,
                'model_path': model_config_path
            }
            
            # 参数统计
            print("🔢 计算参数量...")
            params = self.count_parameters(model)
            results['params'] = params
            print(f"   总参数: {params['total_M']:.2f}M")
            print(f"   可训练参数: {params['trainable_M']:.2f}M")
            
            # FLOPs计算
            print("⚡ 计算FLOPs...")
            flops = self.calculate_flops(model)
            results['flops'] = flops
            print(f"   FLOPs: {flops['flops_str']}")
            
            # 推理速度测试
            print("🚀 测试推理速度...")
            speed = self.benchmark_inference(model, batch_size=1)
            results['speed'] = speed
            print(f"   平均推理时间: {speed['avg_time_ms']:.2f} ms")
            print(f"   FPS: {speed['fps']:.1f}")
            
            # 模型大小
            model_size = 0
            for param in model.parameters():
                model_size += param.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
            results['model_size_mb'] = model_size
            print(f"   模型大小: {model_size:.2f} MB")
            
            return results
            
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_models(self, model_configs):
        """对比多个模型"""
        print("\n🔄 开始模型对比分析...")
        print("="*80)
        
        results_list = []
        
        for name, config_path in model_configs.items():
            result = self.evaluate_model(config_path, name)
            if result:
                results_list.append(result)
        
        # 生成对比表格
        print("\n📊 模型对比表格:")
        print("="*120)
        
        table_data = []
        for result in results_list:
            row = [
                result['model_name'],
                f"{result['params']['total_M']:.1f}",
                f"{result['flops']['flops_G']:.1f}" if result['flops']['flops_G'] else "N/A",
                f"{result['speed']['fps']:.1f}",
                f"{result['model_size_mb']:.1f}",
                f"{result['speed']['avg_time_ms']:.2f}"
            ]
            table_data.append(row)
        
        headers = ["模型名称", "参数(M)", "FLOPs(G)", "FPS", "大小(MB)", "推理时间(ms)"]
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 效率分析
        print("\n📈 效率分析:")
        print("-"*80)
        
        if len(results_list) >= 2:
            baseline = results_list[0]
            print(f"基线模型: {baseline['model_name']}")
            print()
            
            for result in results_list[1:]:
                param_reduction = (1 - result['params']['total_M'] / baseline['params']['total_M']) * 100
                flops_reduction = (1 - (result['flops']['flops_G'] or 0) / (baseline['flops']['flops_G'] or 1)) * 100
                fps_improvement = (result['speed']['fps'] / baseline['speed']['fps'] - 1) * 100
                
                print(f"{result['model_name']}:")
                print(f"  参数减少: {param_reduction:+.1f}%")
                print(f"  FLOPs减少: {flops_reduction:+.1f}%")
                print(f"  速度提升: {fps_improvement:+.1f}%")
                print()
        
        # 保存结果
        results_dict = {r['model_name']: {
            'params_M': r['params']['total_M'],
            'flops_G': r['flops']['flops_G'],
            'fps': r['speed']['fps'],
            'model_size_mb': r['model_size_mb'],
            'inference_time_ms': r['speed']['avg_time_ms']
        } for r in results_list}
        
        return results_dict, results_list
    
    def plot_comparison(self, results_dict, save_path='model_comparison.png'):
        """绘制对比图表"""
        import matplotlib.pyplot as plt
        
        names = list(results_dict.keys())
        params = [results_dict[n]['params_M'] for n in names]
        fps = [results_dict[n]['fps'] for n in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 参数量对比
        colors = ['#FF6B6B' if 'ERT' not in n else '#4ECDC4' for n in names]
        ax1.bar(names, params, color=colors, alpha=0.7)
        ax1.set_ylabel('Parameters (M)', fontsize=12)
        ax1.set_title('Model Parameters Comparison', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(params):
            ax1.text(i, v + 0.5, f'{v:.1f}M', ha='center', fontsize=10)
        
        # 推理速度对比
        ax2.bar(names, fps, color=colors, alpha=0.7)
        ax2.set_ylabel('FPS', fontsize=12)
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(fps):
            ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 对比图表已保存: {save_path}")
        plt.close()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR 轻量化性能评估')
    parser.add_argument('--models', nargs='+', default=['1', '11'],
                       help='模型编号 (1:RT-DETR-L, 11:ERT-DETR)')
    parser.add_argument('--config_dir', type=str, 
                       default='/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr',
                       help='模型配置文件目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save', type=str, default='model_comparison.json', help='保存结果路径')
    parser.add_argument('--plot', action='store_true', help='生成对比图表')
    
    args = parser.parse_args()
    
    # 模型映射
    model_mapping = {
        '1': ('rtdetr-l.yaml', 'RT-DETR-L (基线)'),
        '2': ('rtdetr-mnv4-hybrid-m.yaml', 'RT-DETR+MNV4'),
        '3': ('rtdetr-mnv4-hybrid-m-sea.yaml', 'RT-DETR+MNV4+SEA'),
        '4': ('rtdetr-l-sea.yaml', 'RT-DETR-L+SEA'),
        '5': ('rtdetr-ghostnet.yaml', 'RT-DETR+GhostNet'),
        '6': ('rtdetr-shufflenet-sea.yaml', 'RT-DETR+ShuffleNet+SEA'),
        '7': ('rtdetr-efficientnet-cbam.yaml', 'RT-DETR+EfficientNet+CBAM'),
        '8': ('rtdetr-l-cbam.yaml', 'RT-DETR-L+CBAM'),
        '9': ('rtdetr-mobilenetv3.yaml', 'RT-DETR+MobileNetV3'),
        '10': ('rtdetr-repghostnet.yaml', 'RT-DETR+RepGhostNet'),
        '11': ('ert-detr.yaml', 'ERT-DETR (创新轻量化)'),
    }
    
    # 准备模型配置
    model_configs = {}
    config_dir = Path(args.config_dir)
    
    for model_id in args.models:
        if model_id in model_mapping:
            config_file, model_name = model_mapping[model_id]
            config_path = config_dir / config_file
            if config_path.exists():
                model_configs[model_name] = str(config_path)
            else:
                print(f"⚠️  配置文件不存在: {config_path}")
    
    if not model_configs:
        print("❌ 没有有效的模型配置")
        return
    
    # 执行评估
    evaluator = ModelEvaluator(device=args.device)
    results_dict, results_list = evaluator.compare_models(model_configs)
    
    # 保存结果
    with open(args.save, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✅ 结果已保存到: {args.save}")
    
    # 生成图表
    if args.plot:
        evaluator.plot_comparison(results_dict)

if __name__ == "__main__":
    main()