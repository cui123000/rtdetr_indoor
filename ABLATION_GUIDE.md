# MobileNetV4+SEA 消融实验指南

## 📖 概述

本指南将帮助您系统地进行MobileNetV4+SEA模型的消融实验，验证各种优化策略的独立贡献。

## 🎯 实验目标

通过消融实验回答以下关键问题：
1. **学习率策略** 对性能的贡献是多少？
2. **数据增强** 能带来多大改进？
3. **EMA** 的效果如何？
4. **损失权重调整** 是否有效？
5. **正则化技术** 的作用？
6. **架构优化** vs **训练优化** 哪个更重要？
7. **完整优化** 的最终效果？

## 🔬 实验设计

### 实验序列
```
基准实验 → Exp1 → Exp2 → Exp3 → Exp4 → Exp5 → Exp6 → Exp7
   ↓        ↓      ↓      ↓      ↓      ↓      ↓      ↓
 SEA-Lite  +LR   +Aug   +EMA   +Loss  +Reg   Arch   Full
```

### 实验详情

| 实验 | 名称 | 优化策略 | 预期改进 |
|------|------|----------|----------|
| Baseline | SEA-Lite基准 | 原始配置 | - |
| Exp1 | 学习率优化 | Cosine Annealing | +1-2% |
| Exp2 | + 数据增强 | Mixup, CutMix, Mosaic | +1-3% |
| Exp3 | + EMA | 指数移动平均 | +0.5-1% |
| Exp4 | + 损失权重 | 分类/回归权重调整 | +0.5-1.5% |
| Exp5 | + 正则化 | Weight Decay, Dropout | +0.5-1% |
| Exp6 | 架构优化 | FPN + Skip Connections | +2-4% |
| Exp7 | 完整优化 | 所有策略组合 | +4-8% |

## 🚀 快速开始

### 1. 环境检查
```bash
python scripts/ablation/check_ablation_environment.py
```

确保所有检查项都通过：
- ✅ Python环境 (PyTorch, Ultralytics等)
- ✅ GPU可用性 (推荐8GB+显存)
- ✅ 数据集完整性
- ✅ 配置文件
- ✅ 训练脚本
- ✅ 磁盘空间 (需要10GB+)

### 2. 运行完整实验
```bash
python scripts/ablation/run_ablation_experiments.py
```

**注意事项：**
- ⏱️ 预计总时间：4-8小时
- 💾 磁盘需求：5-10GB
- 🔋 建议在稳定的环境中运行
- 📊 实验会自动保存结果和日志

### 3. 分析结果
```bash
python scripts/ablation/analyze_ablation_results.py
```

生成文件：
- `ablation_results.png` - 性能对比图表
- `ablation_report.md` - 详细分析报告
- `ablation_experiment_log.json` - 实验日志

## 🔧 手动运行单个实验

如果需要手动运行或重新运行特定实验：

### 基准实验
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name baseline_sea_lite
```

### 实验1：学习率优化
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp1_lr_strategy \
  --optimization_version 8
```

### 实验2：+ 数据增强
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp2_data_augmentation \
  --optimization_version 8 --enhanced_augmentation
```

### 实验3：+ EMA
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp3_ema \
  --optimization_version 8 --enhanced_augmentation \
  --ema_decay 0.9999
```

### 实验4：+ 损失权重
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp4_loss_weights \
  --optimization_version 8 --enhanced_augmentation \
  --ema_decay 0.9999 --loss_weights cls:2.0,box:5.0,dfl:1.5
```

### 实验5：+ 正则化
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-hybrid-m-sea-lite.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp5_regularization \
  --optimization_version 8 --enhanced_augmentation \
  --ema_decay 0.9999 --loss_weights cls:2.0,box:5.0,dfl:1.5 \
  --weight_decay 0.0005 --dropout 0.1
```

### 实验6：架构优化
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-phase2-enhanced.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp6_architecture
```

### 实验7：完整优化
```bash
python scripts/training/train_rtdetr_mobilenetv4_select.py \
  --config rtdetr-mnv4-phase2-enhanced.yaml \
  --epochs 100 --batch 8 \
  --name ablation_exp7_full_optimization \
  --optimization_version 9
```

## 📊 结果解读

### 关键指标
- **mAP50**: 主要评估指标
- **mAP50-95**: 严格评估指标  
- **Precision**: 精确率
- **Recall**: 召回率

### 预期结果模式
1. **渐进改进**: 每个策略应有正向贡献
2. **累积效应**: 组合策略效果 ≈ 单独策略累加
3. **架构 vs 训练**: 架构优化通常比训练优化更显著

### 异常情况处理
- **负向贡献**: 某策略降低性能，需要调整参数
- **饱和效应**: 后期改进递减，属正常现象
- **训练不稳定**: 考虑降低学习率或增加正则化

## 🔍 调试指南

### 实验失败处理
1. **GPU内存不足**
   ```bash
   # 减小batch size
   --batch 4
   ```

2. **训练发散**
   ```bash
   # 降低学习率
   --lr 0.0005
   ```

3. **数据加载错误**
   ```bash
   # 检查数据路径
   python -c "from ultralytics import YOLO; print('数据集路径正确')"
   ```

### 监控训练进度
```bash
# 查看训练日志
tail -f runs/detect/ablation_exp1_lr_strategy/train.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

## 📁 文件结构

```
rtdetr_indoor/
├── 配置文件
│   ├── rtdetr-mnv4-hybrid-m-sea-lite.yaml
│   └── rtdetr-mnv4-phase2-enhanced.yaml
├── 训练脚本
│   ├── train_rtdetr_mobilenetv4_select.py
│   └── ablation_study.py
├── 分析脚本
│   ├── analyze_ablation_results.py
│   └── check_ablation_environment.py
├── 自动化脚本
│   └── run_ablation_experiments.py
└── 结果目录
    ├── runs/detect/ablation_*/
    ├── ablation_results.png
    ├── ablation_report.md
    └── ablation_experiment_log.json
```

## 💡 最佳实践

### 实验前准备
1. 🔋 确保电源稳定，避免训练中断
2. 💾 检查磁盘空间充足
3. 🌡️ 监控GPU温度，避免过热
4. 📝 记录实验环境和参数

### 实验期间
1. ⏰ 定期检查训练进度
2. 📊 观察loss曲线变化
3. 🔍 及时发现异常情况
4. 💾 确保结果正确保存

### 实验后分析
1. 📈 对比性能指标
2. 🔍 分析策略贡献
3. 📝 记录关键发现
4. 🎯 规划后续改进

## ❓ 常见问题

### Q: 实验需要多长时间？
A: 单个实验约45分钟，总共8个实验需要4-8小时。

### Q: 可以并行运行多个实验吗？
A: 不建议，GPU内存和计算资源有限。

### Q: 实验失败怎么办？
A: 可以单独重新运行失败的实验，不影响其他实验。

### Q: 如何解读负向贡献？
A: 某些策略可能不适合当前模型/数据，需要调整参数或跳过。

### Q: 结果不如预期怎么办？
A: 检查数据质量、模型配置、超参数设置，可能需要更多epochs。

## 📞 支持

如遇到问题，请检查：
1. 📋 环境检查脚本输出
2. 📝 训练日志文件
3. 🔍 错误信息详情
4. 💾 磁盘空间和GPU状态

---
*消融实验指南 v1.0*
