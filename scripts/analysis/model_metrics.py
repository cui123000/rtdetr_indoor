#!/usr/bin/env python3
"""
提取模型参数量和性能指标
"""

import sys
import yaml
import pandas as pd
from pathlib import Path

def get_model_info(train_dir):
    """从训练目录提取模型信息"""
    train_path = Path(train_dir)
    
    # 读取配置
    args_file = train_path / 'args.yaml'
    results_file = train_path / 'results.csv'
    
    if not args_file.exists() or not results_file.exists():
        return None
    
    # 读取参数配置
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)
    
    # 读取训练结果
    df = pd.read_csv(results_file)
    df.columns = df.columns.str.strip()
    
    # 获取模型名称
    model_name = args.get('model', 'unknown')
    if isinstance(model_name, str) and '/' in model_name:
        model_name = Path(model_name).stem
    
    # 获取最佳性能
    best_map50_95_idx = df['metrics/mAP50-95(B)'].idxmax()
    best_map50_idx = df['metrics/mAP50(B)'].idxmax()
    
    # 计算模型参数量(从权重文件)
    best_weights = train_path / 'weights' / 'best.pt'
    params_millions = 0
    weight_size_mb = 0
    
    if best_weights.exists():
        import torch
        try:
            ckpt = torch.load(best_weights, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model']
                # 尝试多种方式获取参数量
                if hasattr(model, 'parameters'):
                    # PyTorch模型对象
                    params_millions = sum(p.numel() for p in model.parameters()) / 1e6
                elif isinstance(model, dict):
                    # state_dict字典
                    params_millions = sum(p.numel() for p in model.values() if torch.is_tensor(p)) / 1e6
            weight_size_mb = best_weights.stat().st_size / (1024 * 1024)
        except Exception as e:
            pass  # 静默失败，使用配置文件参数量
    
    info = {
        '模型名称': model_name,
        '训练目录': train_path.name,
        '参数量(M)': f"{params_millions:.2f}" if params_millions > 0 else "N/A",
        '权重大小(MB)': f"{weight_size_mb:.2f}" if weight_size_mb > 0 else "N/A",
        '训练轮数': len(df),
        '最佳mAP50-95': f"{df.loc[best_map50_95_idx, 'metrics/mAP50-95(B)']:.4f}",
        '最佳mAP50-95轮数': best_map50_95_idx + 1,
        '最佳mAP50': f"{df.loc[best_map50_idx, 'metrics/mAP50(B)']:.4f}",
        '最佳mAP50轮数': best_map50_idx + 1,
        '最终Precision': f"{df.iloc[-1]['metrics/precision(B)']:.4f}",
        '最终Recall': f"{df.iloc[-1]['metrics/recall(B)']:.4f}",
        '最终mAP50-95': f"{df.iloc[-1]['metrics/mAP50-95(B)']:.4f}",
        '最终mAP50': f"{df.iloc[-1]['metrics/mAP50(B)']:.4f}",
        '训练时间(h)': f"{df.iloc[-1]['time'] / 3600:.2f}",
    }
    
    return info

def load_model_params_from_yaml(model_yaml):
    """从模型配置文件加载并计算参数量"""
    try:
        sys.path.insert(0, '/home/cjj/rtdetr_indoor/ultralytics')
        from ultralytics import RTDETR
        
        model = RTDETR(model_yaml)
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        return total_params / 1e6, trainable_params / 1e6
    except Exception as e:
        print(f"⚠️  加载模型失败: {e}")
        return None, None

def main():
    # 扫描所有训练目录
    runs_dir = Path('/home/cjj/rtdetr_indoor/runs/detect')
    train_dirs = sorted([d for d in runs_dir.glob('train_*') if d.is_dir()], 
                       key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not train_dirs:
        print("❌ 未找到训练目录")
        return
    
    print("=" * 120)
    print("🎯 RT-DETR 模型参数量与性能汇总")
    print("=" * 120)
    print()
    
    all_info = []
    
    for train_dir in train_dirs:
        info = get_model_info(train_dir)
        if info:
            all_info.append(info)
    
    if not all_info:
        print("❌ 未找到有效的训练结果")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_info)
    
    # 打印表格
    print("📊 详细信息:")
    print("-" * 120)
    
    # 打印每个模型的详细信息
    for idx, row in df.iterrows():
        print(f"\n{'='*120}")
        print(f"模型 #{idx+1}: {row['模型名称']}")
        print(f"{'='*120}")
        print(f"  📁 训练目录: {row['训练目录']}")
        print(f"  🔢 参数量: {row['参数量(M)']} M")
        print(f"  💾 权重大小: {row['权重大小(MB)']} MB")
        print(f"  📈 训练轮数: {row['训练轮数']} epochs")
        print(f"  ⏱️  训练时间: {row['训练时间(h)']} 小时")
        print()
        print(f"  🎯 最佳性能:")
        print(f"     • mAP50-95: {row['最佳mAP50-95']} (第{row['最佳mAP50-95轮数']}轮)")
        print(f"     • mAP50: {row['最佳mAP50']} (第{row['最佳mAP50轮数']}轮)")
        print()
        print(f"  📊 最终性能:")
        print(f"     • mAP50-95: {row['最终mAP50-95']}")
        print(f"     • mAP50: {row['最终mAP50']}")
        print(f"     • Precision: {row['最终Precision']}")
        print(f"     • Recall: {row['最终Recall']}")
    
    # 保存CSV
    output_file = runs_dir / 'model_metrics_summary.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n{'='*120}")
    print(f"✅ 详细数据已保存: {output_file}")
    print(f"{'='*120}")
    
    # 简化对比表
    print("\n\n" + "="*120)
    print("📋 模型对比表")
    print("="*120)
    
    compare_cols = ['模型名称', '参数量(M)', '权重大小(MB)', '最佳mAP50-95', '最佳mAP50', '最终Precision', '最终Recall']
    print()
    
    # 打印表头
    header = f"{'模型':<30} {'参数(M)':<12} {'大小(MB)':<12} {'mAP50-95':<12} {'mAP50':<12} {'Precision':<12} {'Recall':<12}"
    print(header)
    print("-" * 120)
    
    # 打印数据行
    for _, row in df.iterrows():
        model_name = row['模型名称'][:28]  # 截断过长名称
        print(f"{model_name:<30} {row['参数量(M)']:<12} {row['权重大小(MB)']:<12} "
              f"{row['最佳mAP50-95']:<12} {row['最佳mAP50']:<12} "
              f"{row['最终Precision']:<12} {row['最终Recall']:<12}")
    
    print("\n" + "="*120)
    
    # 模型配置参考
    print("\n\n📚 各模型理论参数量 (从配置文件):")
    print("="*120)
    
    model_configs = [
        ('rtdetr-l', 'RT-DETR-L 基线'),
        ('rtdetr-l-sea', 'RT-DETR-L + SEA注意力'),
        ('rtdetr-l-cbam', 'RT-DETR-L + CBAM注意力'),
        ('rtdetr-ghostnet', 'RT-DETR + GhostNet'),
        ('rtdetr-shufflenet-sea', 'RT-DETR + ShuffleNet + SEA'),
        ('rtdetr-efficientnet-cbam', 'RT-DETR + EfficientNet + CBAM'),
    ]
    
    models_dir = Path('/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr')
    
    for yaml_file, desc in model_configs:
        yaml_path = models_dir / f"{yaml_file}.yaml"
        if yaml_path.exists():
            total, trainable = load_model_params_from_yaml(str(yaml_path))
            if total:
                print(f"  • {desc:<40} {total:>8.2f}M 参数 ({trainable:>8.2f}M 可训练)")
            else:
                print(f"  • {desc:<40} 加载失败")
        else:
            print(f"  • {desc:<40} 配置文件不存在")
    
    print("\n" + "="*120)

if __name__ == '__main__':
    main()
