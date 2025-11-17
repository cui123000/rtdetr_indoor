# RT-DETR 室内检测改进与实验方案

## 一、现有模型基线

### 已实现的模型
1. **RT-DETR-L** (标准基线)
2. **RT-DETR-L-SEA** (新增SEA注意力)
3. **RT-DETR-MNV4** (轻量级backbone)
4. **RT-DETR-MNV4-SEA** (轻量级+注意力)

## 二、建议的改进方向

### A. 注意力机制改进（高优先级）

#### 1. CBAM 注意力模块
**原理**: Channel + Spatial 双重注意力
**位置**: Backbone 各阶段输出
**预期**: 提升 2-3% mAP
**实现难度**: ⭐⭐

```yaml
# rtdetr-l-cbam.yaml
backbone:
  # Stage 3
  - [-1, 1, CBAM, [1024]]
  # Stage 4  
  - [-1, 1, CBAM, [2048]]
```

#### 2. 多尺度注意力融合
**原理**: 在 FPN 各层添加注意力
**预期**: 提升小物体检测
**实现难度**: ⭐⭐⭐

#### 3. Deformable Attention
**原理**: 可变形注意力，适应不同形状物体
**预期**: 提升 3-5% mAP
**实现难度**: ⭐⭐⭐⭐

### B. Neck 结构改进（中优先级）

#### 1. BiFPN 替换标准 FPN
**优势**: 
- 双向特征流动
- 加权特征融合
- 更好的多尺度融合

**配置**:
```python
# 替换 head 中的 FPN 结构
- [-1, 1, BiFPN, [256, 3]]  # num_layers=3
```

#### 2. ASFF (Adaptively Spatial Feature Fusion)
**优势**: 自适应融合不同尺度特征
**已有基础**: 项目中有 ASFF 实现
**实现**: 在 FPN 输出层添加

#### 3. PANet 增强
**优势**: 增强路径聚合
**改进点**: 添加额外的 bottom-up 路径

### C. 损失函数优化（中优先级）

#### 1. Focal Loss 替换 BCE
**适用**: 类别不平衡问题
**实现**:
```python
'loss_cls': 'focal',  # BCE -> Focal
'focal_alpha': 0.25,
'focal_gamma': 2.0,
```

#### 2. GIoU/CIoU/EIoU 对比
**当前**: GIoU
**建议测试**: CIoU, EIoU, Alpha-IoU

#### 3. 多任务损失权重调整
```python
'box': 7.5,  # 当前值
'cls': 0.5,
'dfl': 1.5,
# 尝试不同权重组合
```

### D. 数据增强改进（高优先级）

#### 1. Mosaic 增强优化
**当前**: mosaic=0.5
**建议**:
- 动态 mosaic (前期1.0, 后期0.0)
- close_mosaic=30 → 20
- 添加 mosaic9 (9宫格)

#### 2. MixUp 比例调整
**当前**: mixup=0.1
**建议**: 0.15-0.2 (室内场景复杂)

#### 3. Copy-Paste 增强
**当前**: 禁用
**建议**: 启用 0.1-0.2 (适合室内物体)

#### 4. 颜色增强
**HSV 参数优化**:
```python
'hsv_h': 0.015,  # → 0.02 (室内光照变化)
'hsv_s': 0.7,    # → 0.8
'hsv_v': 0.4,    # → 0.5
```

### E. 训练策略优化（中优先级）

#### 1. 学习率调度
**当前**: Cosine
**建议测试**:
- Warmup Cosine (已使用)
- OneCycle
- Step decay

#### 2. 优化器对比
**当前**: AdamW
**建议测试**:
- SGD with momentum
- AdamW vs Adam
- Lion optimizer

#### 3. EMA (Exponential Moving Average)
**建议**: 启用 EMA，decay=0.9999
**预期**: 提升 0.5-1% mAP

### F. 后处理优化（低优先级）

#### 1. NMS 参数调整
**当前**: iou=0.6
**建议**: 0.5-0.7 对比

#### 2. Soft-NMS
**优势**: 减少漏检
**实现难度**: ⭐⭐

#### 3. 置信度阈值优化
**当前**: conf=0.001
**建议**: 根据 PR 曲线调整

## 三、对比实验设计

### 实验组 A: 注意力机制对比
```
1. RT-DETR-L (baseline)
2. RT-DETR-L-SEA
3. RT-DETR-L-CBAM (新增)
4. RT-DETR-L-CA (Coordinate Attention，新增)
5. RT-DETR-L-ECA (Efficient Channel Attention，新增)
```

**对比指标**:
- mAP@0.5, mAP@0.5:0.95
- 参数量
- FPS
- 各类别 AP

### 实验组 B: Backbone 对比
```
1. RT-DETR-L (HGNetv2)
2. RT-DETR-ResNet50
3. RT-DETR-ResNet101
4. RT-DETR-MNV4
5. RT-DETR-EfficientNet
```

### 实验组 C: Neck 结构对比
```
1. RT-DETR-L-FPN (baseline)
2. RT-DETR-L-BiFPN (新增)
3. RT-DETR-L-PANet (新增)
4. RT-DETR-L-ASFF (新增)
```

### 实验组 D: 损失函数对比
```
1. GIoU + BCE (baseline)
2. CIoU + BCE
3. EIoU + BCE
4. GIoU + Focal
5. CIoU + Focal
```

### 实验组 E: 数据增强对比
```
1. Baseline (当前配置)
2. Strong Aug (增强所有参数)
3. No Mosaic (禁用mosaic)
4. Heavy MixUp (mixup=0.3)
5. Copy-Paste (copy_paste=0.2)
```

### 实验组 F: 训练策略对比
```
1. AdamW (baseline)
2. SGD + Momentum
3. AdamW + EMA
4. Lion optimizer
5. OneCycle LR
```

## 四、消融实验设计

### 4.1 SEA 注意力消融
```
1. No attention (baseline)
2. SEA at Stage 3 only
3. SEA at Stage 4 only
4. SEA at both stages (完整版)
```

### 4.2 多尺度特征消融
```
1. P3 only
2. P4 only
3. P5 only
4. P3+P4
5. P4+P5
6. P3+P4+P5 (完整版)
```

### 4.3 Decoder 层数消融
```
1. 1 layer
2. 3 layers
3. 6 layers (标准)
4. 9 layers
```

## 五、评估指标体系

### 5.1 精度指标
- **mAP@0.5**: 主要指标
- **mAP@0.5:0.95**: COCO 标准
- **各类别 AP**: 找出弱势类别
- **小/中/大物体 AP**: 多尺度性能

### 5.2 效率指标
- **参数量**: 模型大小
- **FLOPs**: 计算复杂度
- **FPS**: 推理速度
- **训练时间**: 每 epoch 时间

### 5.3 鲁棒性指标
- **不同光照**: 测试集分组
- **遮挡情况**: 部分遮挡检测
- **小物体**: AP_small 指标
- **密集场景**: 重叠物体检测

## 六、实施优先级

### 阶段 1: 快速改进（1-2天）
✅ 1. RT-DETR-L-SEA (已完成)
🔲 2. 添加 CBAM 注意力版本
🔲 3. 数据增强参数调优
🔲 4. 损失函数对比实验

### 阶段 2: 深度优化（3-5天）
🔲 5. BiFPN/ASFF Neck 改进
🔲 6. 多种注意力机制对比
🔲 7. EMA + 优化器对比
🔲 8. 不同 backbone 对比

### 阶段 3: 消融分析（2-3天）
🔲 9. SEA 位置消融实验
🔲 10. 多尺度特征消融
🔲 11. Decoder 层数消融
🔲 12. 综合最优配置

## 七、论文实验章节结构建议

### 7.1 实验设置
- 数据集描述
- 训练配置
- 评估指标
- 实验环境

### 7.2 与 SOTA 对比
- RT-DETR vs YOLOv8/v9
- RT-DETR vs DETR 系列
- RT-DETR vs Faster R-CNN

### 7.3 注意力机制对比
- SEA vs CBAM vs CA
- 可视化注意力图
- 定量分析

### 7.4 消融实验
- 各组件贡献度
- 参数敏感性分析

### 7.5 定性分析
- 可视化检测结果
- 失败案例分析
- 改进方向讨论

## 八、快速开始

### 立即可做的实验
```bash
# 1. 训练 RT-DETR-L-SEA
SELECTED_MODEL='4' python3 scripts/training/auto_train_rtdetr.py

# 2. 数据增强对比（修改脚本参数）
# 创建不同增强策略的配置

# 3. 损失函数对比（代码修改 loss 权重）
# box: 7.5 vs 5.0 vs 10.0
```

## 九、预期贡献点

1. **SEA 在 RT-DETR 的首次应用**
2. **室内场景优化策略**
3. **多种注意力机制系统对比**
4. **完整的消融实验**
5. **轻量级与精度的权衡分析**

## 十、需要的资源

### 计算资源
- GPU: A40 (47GB) 或 RTX 4090 (24GB)
- 训练时间: 每个模型 12-24 小时
- 总实验时间: 约 2-3 周

### 数据资源
- COCO Indoor 子集 (已有)
- 可选: 扩展到完整 COCO
- 可选: 增加自定义室内数据

### 对比基线
- YOLOv8
- YOLOv9
- DETR
- Faster R-CNN
