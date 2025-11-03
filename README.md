# RT-DETR 室内物体检测

基于 **RT-DETR + MobileNetV4 + SEA注意力** 的室内物体检测项目。

## 🏆 项目状态

**✅ 项目已完成** | 最佳模型: **MNV4-SEA** (mAP50: 0.4566)

## 📊 快速概览

| 指标 | 值 |
|------|-----|
| **最佳mAP50** | 0.4566 |
| **最佳mAP50-95** | 0.2973 |
| **参数量** | 29.06M |
| **训练数据集** | HomeObjects-3K (67类) |
| **训练时长** | 120 epochs (~15h on RTX 4090) |

## 🚀 快速开始

### 推理使用最佳模型

```python
from ultralytics import RTDETR

# 加载最佳模型
model = RTDETR("runs/detect/rtdetr_mnv4_sea_single_bs6/weights/best.pt")

# 推理
results = model.predict("your_image.jpg")
results[0].show()
```

### 查看训练结果

```bash
# 查看详细对比分析
python scripts/analysis/performance_summary.py

# 查看版本对比
python scripts/analysis/detailed_version_comparison.py
```

## 📈 训练结果汇总

| 排名 | 模型 | mAP50 | 参数量 | vs 最佳 | 状态 |
|------|------|-------|--------|---------|------|
| 🥇 | **MNV4-SEA** | **0.4566** | 29.06M | - | ✅ 最佳 |
| 🥈 | MNV4-SEA-BiFPN | 0.4167 | 29.26M | -8.7% | ❌ 失败 |
| 🥉 | RT-DETR-MNV4 | 0.3990 | 24.98M | -12.6% | 基线 |
| 4 | ASFF-v1 | 0.3889 | 27.75M | -14.8% | ❌ 失败 |
| 5 | ASFF-v3 | 0.3593 | 25.23M | -21.3% | ❌ 失败 |
| 6 | RT-DETR-L | 0.3144 | 32.97M | -31.1% | 基线 |

## 💡 核心发现

### ✅ 成功
- **MNV4-SEA** 是唯一成功的改进
- SEA注意力机制显著提升性能 (+14.4% vs MNV4基线)
- 29.06M参数达到最佳性能平衡

### ❌ 失败教训
- **所有融合网络**(BiFPN, ASFF)均失败
- 过度轻量化严重损害性能
- ASFF不适合RT-DETR的Transformer架构

## 📁 项目结构

```
rtdetr_indoor/
├── FINAL_REPORT.md              # 📄 完整项目报告
├── README.md                    # 本文件
├── docs/                        # 📚 文档
│   ├── 训练结果对比分析.md      # 详细分析
│   └── ASFF配置版本说明.md      # ASFF失败原因
├── scripts/
│   ├── training/                # 训练脚本
│   ├── analysis/                # 分析脚本
│   └── evaluation/              # 评估脚本
├── runs/detect/                 # 训练结果
│   └── rtdetr_mnv4_sea_single_bs6/  # ✅ 最佳模型
└── ultralytics/                 # 框架与配置
```

## 📖 详细文档

- **[FINAL_REPORT.md](FINAL_REPORT.md)** - 完整项目报告和分析
- **[docs/训练结果对比分析.md](docs/训练结果对比分析.md)** - 各版本详细对比
- **[docs/ASFF配置版本说明.md](docs/ASFF配置版本说明.md)** - ASFF失败原因分析

## 🎯 推荐与建议

### ✅ 推荐
- 使用 **MNV4-SEA** 作为生产模型
- 性能优化方向：知识蒸馏、数据增强、超参数调优

### ❌ 不推荐
- 继续尝试融合网络(ASFF/BiFPN)
- 过度减少模型参数量
- 训练ASFF v2 (大概率失败)

## 🔧 环境配置

```bash
# 1. 激活环境
conda activate uRTDETR

# 2. 安装依赖（已完成）
# pip install ultralytics

# 3. 数据集
数据集路径: datasets/homeobjects-3K/
```

## 📊 性能指标

**MNV4-SEA (最佳模型):**
- mAP50: **0.4566**
- mAP50-95: **0.2973**
- Precision: 0.4980
- Recall: 0.4928
- 参数量: 29.06M
- 推理速度: ~XX FPS (RTX 4090)

## 🤝 贡献

项目由 CUI 完成。

## 📝 更新日志

- **2025-10-23**: 项目完成，所有模型训练完毕
- **2025-10-23**: 生成最终报告和对比分析
- **2025-10-23**: 确认MNV4-SEA为最佳模型

---

**状态**: ✅ 项目完成  
**最后更新**: 2025-10-23  
**推荐模型**: MNV4-SEA (mAP50: 0.4566)
