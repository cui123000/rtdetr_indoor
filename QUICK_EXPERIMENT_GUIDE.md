# 快速实验指南

## 当前可用模型

### 已实现
1. ✅ **rtdetr-l.yaml** - 基线模型
2. ✅ **rtdetr-l-sea.yaml** - 添加SEA注意力
3. ✅ **rtdetr-l-cbam.yaml** - 添加CBAM注意力
4. ✅ **rtdetr-mnv4-hybrid-m.yaml** - MobileNetV4轻量级
5. ✅ **rtdetr-mnv4-hybrid-m-sea.yaml** - MNV4+SEA

### 待实现（优先级排序）

#### 高优先级（立即可做）
1. **数据增强对比**
   - 修改训练脚本参数即可
   - 无需改模型结构
   - 预计时间：2小时/实验

2. **损失函数对比**
   - 修改 loss 权重
   - 测试 GIoU vs CIoU
   - 预计时间：1小时配置

3. **优化器对比**
   - AdamW vs SGD
   - 添加 EMA
   - 预计时间：1小时配置

#### 中优先级（需要少量代码）
4. **ECA 注意力** (Efficient Channel Attention)
   - 代码已有
   - 复制 SEA 配置模式
   - 预计时间：30分钟

5. **CA 注意力** (Coordinate Attention)
   - 需要检查是否已实现
   - 预计时间：1-2小时

6. **不同 Backbone**
   - ResNet50/101 (Ultralytics 已有)
   - 预计时间：30分钟/模型

#### 低优先级（需要较多开发）
7. **BiFPN**
   - 需要实现新模块
   - 预计时间：4-6小时

8. **ASFF**
   - 项目中有代码但需要集成
   - 预计时间：2-3小时

## 实验执行方案

### 方案A: 顺序执行（稳妥但慢）

```bash
# 1. 基线模型（已完成或进行中）
SELECTED_MODEL='1' python3 scripts/training/auto_train_rtdetr.py

# 2. SEA注意力
SELECTED_MODEL='4' python3 scripts/training/auto_train_rtdetr.py

# 3. 数据增强对比
# 修改脚本中的 mosaic, mixup, copy_paste 参数

# 4. 损失函数对比  
# 修改 box, cls, dfl 权重
```

**优点**: 稳定，容易管理
**缺点**: 耗时长（每个24小时，共3-5天）

### 方案B: 批量执行（快速但占资源）

```bash
# 使用批量实验脚本
python3 scripts/experiments/batch_experiments.py
```

**优点**: 一次性运行多个实验
**缺点**: 需要独占GPU，每个实验串行执行

### 方案C: 分布式执行（推荐）

如果有多个GPU或机器：
```bash
# GPU 0: 基线模型
CUDA_VISIBLE_DEVICES=0 python3 scripts/training/auto_train_rtdetr.py &

# GPU 1: SEA模型  
CUDA_VISIBLE_DEVICES=1 SELECTED_MODEL='4' python3 scripts/training/auto_train_rtdetr.py &
```

## 立即可做的 3 个实验

### 实验1: 数据增强对比（最快）

修改 `scripts/training/auto_train_rtdetr.py`:

```python
# 强增强版
'mosaic': 0.8,      # 原 0.5
'mixup': 0.2,       # 原 0.1
'copy_paste': 0.15, # 原 0.0
'hsv_h': 0.02,      # 原 0.015
'hsv_s': 0.8,       # 原 0.7
```

**预期**: +1-2% mAP
**时间**: 12-24小时
**难度**: ⭐

### 实验2: 损失权重优化

```python
# 在 create_training_config() 中修改
'box': 10.0,   # 增加 box loss 权重（原7.5）
'cls': 0.5,
'dfl': 1.5,
```

或测试:
```python
'box': 5.0,    # 减少 box loss 权重
'cls': 1.0,    # 增加 cls loss 权重
'dfl': 2.0,    # 增加 dfl loss 权重
```

**预期**: +0.5-1% mAP
**时间**: 12-24小时
**难度**: ⭐

### 实验3: 学习率策略

```python
# 测试更大学习率
'lr0': 0.0002,     # 原 0.0001
'lrf': 0.1,        # 原 0.2
'warmup_epochs': 5.0, # 原 10.0
```

**预期**: 可能加快收敛
**时间**: 12-24小时
**难度**: ⭐

## 结果分析流程

### 1. 训练完成后

```bash
# 运行分析脚本
python3 scripts/analysis/analyze_results.py
```

### 2. 查看结果

```bash
cd runs/experiments/analysis
cat analysis_report.md  # Markdown报告
open map_comparison.png  # mAP对比图
open loss_comparison.png # 损失对比图
```

### 3. 提取最佳权重

```bash
# 找到最佳模型目录
cd runs/experiments/

# 复制最佳权重
cp rtdetr_l_sea_20251117/weights/best.pt ../best_models/rtdetr_l_sea.pt
```

## 论文写作检查清单

### 必须包含的实验
- [ ] 基线模型结果
- [ ] 至少2种改进方法对比
- [ ] 消融实验（证明每个组件有效）
- [ ] 与SOTA对比（YOLOv8, DETR等）

### 必须包含的图表
- [ ] mAP曲线对比
- [ ] PR曲线
- [ ] 混淆矩阵
- [ ] 可视化检测结果
- [ ] 注意力图可视化

### 必须包含的表格
- [ ] 各模型性能汇总
- [ ] 消融实验结果
- [ ] 各类别AP对比
- [ ] 速度/精度权衡

## 时间规划建议

### 第1周
- Day 1-2: 基线模型训练
- Day 3-4: SEA注意力模型
- Day 5-7: 数据增强/损失函数对比

### 第2周  
- Day 1-3: CBAM/CA注意力对比
- Day 4-5: 不同backbone对比
- Day 6-7: 结果分析与图表生成

### 第3周
- Day 1-2: 消融实验
- Day 3-4: 最优配置训练
- Day 5-7: 论文写作与完善

## 常见问题

### Q: GPU被占用怎么办？
A: 
1. 减小batch size (12→8→6)
2. 使用workers=0 (单进程)
3. 联系管理员协调GPU使用

### Q: 训练太慢怎么办？
A:
1. 减少epochs (100→50)
2. 启用cache=True
3. 增加workers
4. 启用AMP (如果稳定)

### Q: 如何快速验证想法？
A:
1. 用少量数据训练10 epochs
2. 观察loss下降趋势
3. 确认有效后完整训练

## 下一步行动

### 立即执行
```bash
# 1. 确保当前训练正常运行
tail -f training.log

# 2. 准备下一个实验配置
# 编辑 auto_train_rtdetr.py

# 3. 创建实验记录文件
mkdir -p experiment_logs
echo "Experiment 1: Baseline" > experiment_logs/exp1.txt
```

### 本周目标
- 完成基线模型训练
- 完成SEA注意力对比
- 完成1个数据增强实验

### 本月目标
- 完成所有核心对比实验
- 生成完整分析报告
- 准备论文实验章节
