#!/usr/bin/env python3
"""
SEA注意力版本性能下降诊断分析
深度分析SEA集成后性能下降的原因
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_sea_performance_degradation():
    """分析SEA版本性能下降原因"""
    
    print("🔍 SEA注意力版本性能下降诊断分析")
    print("=" * 60)
    
    # 读取训练结果
    base_path = Path("/home/cui/vild_rtdetr_indoor")
    hybrid_path = base_path / "rtdetr_mobilenetv4_rtdetr_mnv4_hybrid_m" / "results.csv"
    sea_path = base_path / "rtdetr_mobilenetv4_rtdetr_mnv4_hybrid_m_sea" / "results.csv"
    
    # 读取数据
    def read_csv_data(filepath):
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # 转换数值
                    for key, value in row.items():
                        if key != 'epoch':
                            try:
                                row[key] = float(value) if value.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit() else value
                            except:
                                pass
                    data.append(row)
                except:
                    continue
        return data
    
    hybrid_data = read_csv_data(hybrid_path)
    sea_data = read_csv_data(sea_path)
    
    print(f"✅ Hybrid-M数据: {len(hybrid_data)} epochs")
    print(f"✅ SEA数据: {len(sea_data)} epochs")
    print()
    
    # 1. 训练损失对比分析
    print("📉 训练损失对比分析:")
    print("-" * 40)
    
    loss_metrics = ['train/giou_loss', 'train/cls_loss', 'train/l1_loss']
    
    for epoch_idx in [4, 9, 19, 49, 99]:  # 检查几个关键epoch
        if epoch_idx < len(hybrid_data) and epoch_idx < len(sea_data):
            print(f"\nEpoch {epoch_idx + 1}:")
            for loss in loss_metrics:
                if loss in hybrid_data[epoch_idx] and loss in sea_data[epoch_idx]:
                    hybrid_val = hybrid_data[epoch_idx][loss]
                    sea_val = sea_data[epoch_idx][loss]
                    
                    try:
                        diff = ((sea_val - hybrid_val) / hybrid_val * 100) if hybrid_val > 0 else 0
                        status = "🔴" if diff > 10 else "🟡" if diff > 0 else "🟢"
                        print(f"  {loss.split('/')[-1]:12}: Hybrid={hybrid_val:.4f}, SEA={sea_val:.4f} {status} ({diff:+.1f}%)")
                    except:
                        print(f"  {loss.split('/')[-1]:12}: 数据解析错误")
    
    print()
    
    # 2. 验证损失分析
    print("📊 验证损失分析:")
    print("-" * 40)
    
    val_metrics = ['val/giou_loss', 'val/cls_loss', 'val/l1_loss']
    
    # 检查是否有NaN值
    sea_nan_count = 0
    total_count = 0
    
    for epoch_data in sea_data[:10]:  # 检查前10个epoch
        for metric in val_metrics:
            if metric in epoch_data:
                val = epoch_data[metric]
                total_count += 1
                if str(val).lower() == 'nan' or val == 'nan':
                    sea_nan_count += 1
    
    if sea_nan_count > 0:
        print(f"⚠️ SEA版本前10epochs中有{sea_nan_count}/{total_count}个验证损失为NaN")
        print("  这表明训练初期不稳定，可能原因:")
        print("    1. SEA注意力导致梯度爆炸/消失")
        print("    2. 学习率对SEA模块过高")
        print("    3. SEA模块初始化不当")
    else:
        print("✅ 验证损失无NaN值")
    
    print()
    
    # 3. 收敛模式分析
    print("📈 收敛模式分析:")
    print("-" * 40)
    
    # 分析mAP50的收敛模式
    hybrid_map50 = [float(row['metrics/mAP50(B)']) for row in hybrid_data if 'metrics/mAP50(B)' in row and str(row['metrics/mAP50(B)']).replace('.', '').isdigit()]
    sea_map50 = [float(row['metrics/mAP50(B)']) for row in sea_data if 'metrics/mAP50(B)' in row and str(row['metrics/mAP50(B)']).replace('.', '').isdigit()]
    
    # 找到mAP50的峰值和对应epoch
    if hybrid_map50 and sea_map50:
        hybrid_peak = max(hybrid_map50)
        sea_peak = max(sea_map50)
        
        hybrid_peak_epoch = hybrid_map50.index(hybrid_peak) + 1
        sea_peak_epoch = sea_map50.index(sea_peak) + 1
        
        print(f"Hybrid-M峰值: mAP50={hybrid_peak:.4f} at epoch {hybrid_peak_epoch}")
        print(f"SEA版本峰值: mAP50={sea_peak:.4f} at epoch {sea_peak_epoch}")
        print(f"性能差距: {((sea_peak - hybrid_peak) / hybrid_peak * 100):+.1f}%")
        
        # 检查是否有性能退化
        if len(sea_map50) >= 50:
            early_avg = sum(sea_map50[10:20]) / 10  # epoch 11-20平均
            late_avg = sum(sea_map50[40:50]) / 10   # epoch 41-50平均
            
            if late_avg < early_avg:
                print(f"⚠️ SEA版本存在性能退化: 早期平均{early_avg:.4f} → 后期平均{late_avg:.4f}")
            else:
                print(f"✅ SEA版本持续改进: 早期平均{early_avg:.4f} → 后期平均{late_avg:.4f}")
    
    print()
    
    # 4. 模型复杂度分析
    print("🧠 模型复杂度分析:")
    print("-" * 40)
    
    # 测试模型参数量
    try:
        import sys
        sys.path.insert(0, '/home/cui/vild_rtdetr_indoor/ultralytics')
        from ultralytics import RTDETR
        
        hybrid_model = RTDETR('/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m.yaml')
        sea_model = RTDETR('/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea.yaml')
        
        hybrid_params = sum(p.numel() for p in hybrid_model.model.parameters())
        sea_params = sum(p.numel() for p in sea_model.model.parameters())
        
        param_increase = ((sea_params - hybrid_params) / hybrid_params * 100)
        
        print(f"Hybrid-M参数量: {hybrid_params:,}")
        print(f"SEA版本参数量: {sea_params:,}")
        print(f"参数增加: {param_increase:+.1f}%")
        
        if param_increase > 50:
            print("⚠️ 参数量增加过多，可能导致:")
            print("  1. 过拟合风险增加")
            print("  2. 训练困难")
            print("  3. 需要更多数据和更长训练时间")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    
    print()
    
    return hybrid_data, sea_data

def generate_improvement_recommendations():
    """生成改进建议"""
    print("💡 SEA版本改进建议:")
    print("=" * 60)
    
    print("🎯 立即可行的改进:")
    print("-" * 30)
    improvements = [
        "1. 减少SEA模块使用量 (5个→2个)",
        "2. 只在关键层使用SEA_Attention_Light",
        "3. 调整SEA模块的学习率 (使用更小的lr)",
        "4. 增加warmup时间 (3→10 epochs)",
        "5. 使用梯度裁剪防止梯度爆炸",
        "6. 检查SEA模块的权重初始化"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print()
    
    print("🔬 深度优化策略:")
    print("-" * 30)
    deep_strategies = [
        "1. 渐进式SEA集成 (先训练50epochs无SEA，再加入SEA)",
        "2. 自适应SEA启用 (根据训练进度动态开启SEA)",
        "3. SEA模块的专用正则化策略",
        "4. 混合精度训练优化SEA计算",
        "5. 不同SEA变体的A/B测试",
        "6. SEA与现有注意力(C2f)的协调机制"
    ]
    
    for strategy in deep_strategies:
        print(f"  {strategy}")
    
    print()
    
    print("📝 建议的实验方案:")
    print("-" * 30)
    print("  方案1: 最小SEA集成 (仅1个SEA_Attention_Light)")
    print("  方案2: 渐进式训练 (先Hybrid-M 50epochs，再加SEA 50epochs)")
    print("  方案3: 专用训练策略 (lr=0.0001, warmup=10, gradient_clip=5.0)")
    
    print()

def test_minimal_sea_version():
    """测试最小SEA版本"""
    print("🧪 测试最小SEA版本:")
    print("-" * 40)
    
    try:
        import sys
        sys.path.insert(0, '/home/cui/vild_rtdetr_indoor/ultralytics')
        from ultralytics import RTDETR
        import torch
        
        # 测试新的轻量SEA配置
        lite_model_path = '/home/cui/vild_rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-mnv4-hybrid-m-sea-lite.yaml'
        
        print("加载轻量SEA配置...")
        model = RTDETR(lite_model_path)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✅ 轻量SEA模型加载成功")
        print(f"📊 参数量: {total_params:,}")
        
        # 测试前向传播
        x = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            output = model.model(x)
        print(f"✅ 前向传播测试通过")
        
        print("\n🎯 轻量SEA配置特点:")
        print("  • 仅使用2个SEA_Attention_Light模块")
        print("  • 位置：Stage3后期 + Stage4中期")
        print("  • 移除了过度的SEA使用")
        print("  • 保持原有的C2f注意力机制")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    hybrid_data, sea_data = analyze_sea_performance_degradation()
    generate_improvement_recommendations()
    test_minimal_sea_version()
    
    print("\n" + "=" * 60)
    print("🔍 诊断总结:")
    print("  SEA注意力机制本身是有价值的，但需要:")
    print("  1. 更谨慎的集成策略")
    print("  2. 专门的训练配置")
    print("  3. 渐进式的优化方法")
    print("=" * 60)

if __name__ == "__main__":
    main()
