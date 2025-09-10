# 脚本目录结构说明

## 📁 scripts/ 目录组织

### 🏋️ training/ - 训练脚本
| 文件 | 说明 | 用途 |
|------|------|------|
| `train_rtdetr_mobilenetv4.py` | 基础训练脚本 | 标准MobileNetV4+RT-DETR训练 |
| `train_rtdetr_mobilenetv4_select.py` | 高级训练脚本 | 支持多种优化策略的训练 |
| `train_sea_lite_optimized.py` | SEA-Lite优化训练 | 针对SEA-Lite的专门优化 |

**使用示例:**
```bash
# 基础训练
python scripts/training/train_rtdetr_mobilenetv4.py

# 高级训练（支持消融实验）
python scripts/training/train_rtdetr_mobilenetv4_select.py --config rtdetr-mnv4-hybrid-m-sea-lite.yaml --optimization_version 8
```

### 🧪 evaluation/ - 模型评估
| 文件 | 说明 | 用途 |
|------|------|------|
| `test_hybrid_model.py` | 混合模型测试 | 测试MobileNetV4混合架构 |
| `test_all_versions.py` | 批量模型测试 | 对比测试多个模型版本 |

**使用示例:**
```bash
# 测试混合模型
python scripts/evaluation/test_hybrid_model.py

# 批量测试所有版本
python scripts/evaluation/test_all_versions.py
```

### 📊 analysis/ - 分析可视化
| 文件 | 说明 | 用途 |
|------|------|------|
| `diagnose_sea_performance.py` | SEA性能诊断 | 分析SEA注意力机制性能问题 |
| `compare_configurations.py` | 配置对比 | 对比不同模型配置 |
| `visualize_fusion_architecture.py` | 架构可视化 | 可视化融合架构设计 |
| `fusion_implementation_summary.py` | 融合实现总结 | 生成融合实现报告 |

**使用示例:**
```bash
# SEA性能诊断
python scripts/analysis/diagnose_sea_performance.py

# 可视化架构
python scripts/analysis/visualize_fusion_architecture.py
```

### 🔬 ablation/ - 消融实验
| 文件 | 说明 | 用途 |
|------|------|------|
| `check_ablation_environment.py` | 环境检查 | 验证消融实验环境 |
| `ablation_study.py` | 消融实验脚本 | 单独运行消融实验 |
| `run_ablation_experiments.py` | 批量消融实验 | 自动运行完整消融实验序列 |
| `analyze_ablation_results.py` | 结果分析 | 分析消融实验结果 |

**使用示例:**
```bash
# 1. 检查环境
python scripts/ablation/check_ablation_environment.py

# 2. 运行完整消融实验
python scripts/ablation/run_ablation_experiments.py

# 3. 分析结果
python scripts/ablation/analyze_ablation_results.py
```

### 🛠️ utils/ - 工具脚本
| 文件 | 说明 | 用途 |
|------|------|------|
| `quick_start_optimization.py` | 快速优化启动 | 快速应用优化策略 |
| `optimize_sea_training.py` | SEA训练优化 | SEA模型训练优化工具 |

**使用示例:**
```bash
# 快速开始优化
python scripts/utils/quick_start_optimization.py
```

## 🚀 常用工作流程

### 1. 新模型训练
```bash
# 检查环境
python scripts/ablation/check_ablation_environment.py

# 开始训练
python scripts/training/train_rtdetr_mobilenetv4_select.py --config your_config.yaml
```

### 2. 性能诊断
```bash
# 分析性能问题
python scripts/analysis/diagnose_sea_performance.py

# 对比配置
python scripts/analysis/compare_configurations.py
```

### 3. 消融实验
```bash
# 完整消融实验流程
python scripts/ablation/run_ablation_experiments.py
```

### 4. 模型评估
```bash
# 测试模型
python scripts/evaluation/test_all_versions.py
```

## 📝 注意事项

1. **路径依赖**: 所有脚本都假设从项目根目录 `/home/cui/rtdetr_indoor/` 运行
2. **配置文件**: 确保配置文件在正确位置（根目录或ultralytics/cfg/models/rt-detr/）
3. **数据集**: 确保数据集路径正确配置在 `datasets/indoor_enhanced/`
4. **环境**: 确保激活了正确的conda环境 `uRTDETR`

## 🔄 迁移注意

由于脚本路径发生变化，如果有其他脚本引用了这些文件，需要更新路径。主要影响：

- 消融实验脚本中的路径引用
- 训练脚本的导入路径
- 配置文件中的脚本路径

## 📞 使用帮助

如果遇到路径问题，可以：
1. 检查当前工作目录是否为项目根目录
2. 确认相对路径是否正确
3. 查看具体错误信息进行调试
