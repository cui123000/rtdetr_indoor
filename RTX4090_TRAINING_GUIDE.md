# RTX 4090 RT-DETR 训练配置指南

## 📋 配置概览

### 硬件配置
- **GPU**: RTX 4090 (24GB 显存)
- **显存使用**: 95% 优化利用
- **计算优化**: TF32 启用 (~5x 加速)

### 模型配置

#### 1️⃣ RT-DETR-L (推荐)
```yaml
batch: 16              # 大批次加快训练
lr0: 0.001            # 较高初始学习率
epochs: 100           # 标准训练轮数
warmup_epochs: 5.0    # 缩短预热期
optimizer: SGD        # 对 RT-DETR 更稳定
```
**预计时间**: ~2-3 小时 (118K 图像)
**性能**: 最佳精度与速度平衡

#### 2️⃣ RT-DETR+MobileNetV4
```yaml
batch: 16             # 同样批次
lr0: 0.001           # 较高学习率
epochs: 100          # 减少轮数提高效率
optimizer: SGD
```
**预计时间**: ~2.5-3.5 小时
**性能**: 轻量级，推理快

#### 3️⃣ RT-DETR+MobileNetV4+SEA
```yaml
batch: 12            # 相对较小 (SEA 模块复杂)
lr0: 0.0005          # 较低学习率 (稳定)
epochs: 120          # 更多训练轮数
optimizer: SGD
```
**预计时间**: ~4-5 小时
**性能**: 融合架构，高精度

## 🚀 快速启动

### 方式 1: 使用启动脚本
```bash
bash start_training_4090.sh
```

### 方式 2: 直接运行训练
```bash
python scripts/training/auto_train_rtdetr.py
```

### 方式 3: 修改配置后训练
编辑 `scripts/training/auto_train_rtdetr.py` 的 `SELECTED_MODEL` 变量:
- `SELECTED_MODEL = '1'` (RT-DETR-L)
- `SELECTED_MODEL = '2'` (RT-DETR+MNV4)
- `SELECTED_MODEL = '3'` (RT-DETR+MNV4+SEA)

## 📊 数据集配置

```yaml
path: /home/cjj/rtdetr_indoor/datasets/coco_indoor
train: images/train2017  # 训练集
val: images/val2017      # 验证集
nc: 21                   # 室内类别数
```

### 室内类别 (21 个)
- **基础家具**: bed, sofa, chair, table
- **电器**: tv, laptop, microwave, refrigerator, clock
- **室内物品**: plant, vase, book, bottle, cup, bowl, glass
- **配件**: keyboard, phone, remote, toilet
- **标准**: person

## ⚙️ 优化参数说明

### 批次大小 (Batch Size)
- RTX 4090 24GB 显存支持 batch=16
- 较大批次加快收敛，需要充足 GPU 显存

### 学习率 (Learning Rate)
- `lr0=0.001`: 较高初始学习率快速学习
- `lrf=0.01`: 低最终学习率因子精细调整
- 余弦学习率衰减确保平稳下降

### Workers (数据加载线程)
- `workers=12`: 充分利用 CPU 加速数据加载
- Linux 原生支持，不需要 WSL2 兼容处理

### GPU 优化
```python
# TF32 启用 (Tensor Float 32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 提升 ~5x 性能，保持精度

# 显存占比
torch.cuda.set_per_process_memory_fraction(0.95)
# 使用 95% 显存，避免浪费
```

### 数据增强
- **mosaic**: 1.0 (完整增强)
- **mixup**: 0.1 (轻度混合)
- **旋转**: degrees=10.0
- **缩放**: scale=0.5

## 📈 训练监控

### 预期指标
1. **Loss 下降**
   - Epoch 1-10: 快速下降
   - Epoch 10-50: 缓慢下降
   - Epoch 50+: 平台期

2. **mAP 提升**
   - Epoch 1-20: 快速提升
   - Epoch 20-80: 稳定提升
   - Epoch 80+: 微调优化

### 检查点保存
- 每 10 个 epoch 保存一次
- 最佳模型自动保存为 `best.pt`
- 最后模型保存为 `last.pt`

## 🛠️ 故障排除

### GPU 内存溢出
**症状**: CUDA out of memory

**解决方案**:
```python
# 降低批次大小
batch_size = 8  # 从 16 改为 8

# 减少 workers
workers = 8     # 从 12 改为 8
```

### 训练速度慢
**原因**: TF32 未启用或 workers 不足

**检查**:
```bash
python -c "import torch; print(torch.backends.cuda.matmul.allow_tf32)"
# 应输出: True
```

### 验证错误
**原因**: 数据集路径配置错误

**检查**:
```bash
ls -la datasets/coco_indoor/images/train2017 | head
ls -la datasets/coco_indoor/images/val2017 | head
```

## 📁 输出文件位置

```
runs/detect/
├── train_rtdetr_l_YYYYMMDD_HHMMSS/
│   ├── weights/
│   │   ├── best.pt       # 最佳模型
│   │   └── last.pt       # 最后模型
│   ├── results.csv       # 训练结果
│   ├── args.yaml         # 训练参数
│   └── labels_*.jpg      # 可视化标签
```

## 🎯 推荐用法

### 快速测试
```python
SELECTED_MODEL = '1'   # RT-DETR-L
epochs = 20            # 快速验证
```

### 完整训练
```python
SELECTED_MODEL = '1'   # RT-DETR-L
epochs = 100           # 标准训练
```

### 高精度训练
```python
SELECTED_MODEL = '3'   # RT-DETR+MNV4+SEA
epochs = 120           # 完整训练
```

## 📝 常用命令

```bash
# 启动训练
python scripts/training/auto_train_rtdetr.py

# 查看最佳模型
ls -lh runs/detect/*/weights/best.pt

# 验证模型
python -m ultralytics.yolo detect predict model=runs/detect/.../weights/best.pt source=datasets/coco_indoor/images/val2017

# 清理旧权重
rm -rf runs/detect/train_*_old
```

## 🔍 性能基准

在 RTX 4090 + COCO Indoor (118K 图像) 上:

| 模型 | 批次 | 学习率 | 预计时间 | mAP |
|------|------|--------|---------|-----|
| RT-DETR-L | 16 | 0.001 | 2-3h | ~55-60 |
| RT-DETR-MNV4 | 16 | 0.001 | 2.5-3.5h | ~50-55 |
| RT-DETR+SEA | 12 | 0.0005 | 4-5h | ~58-62 |

*注: mAP 值为参考，实际结果取决于数据质量*

## ✅ 快速检查清单

- [x] GPU 显存充足 (24GB RTX 4090)
- [x] CUDA/PyTorch 已安装
- [x] 数据集配置正确
- [x] 模型权重文件可访问
- [x] 输出目录有写权限
- [x] TF32 已启用
- [x] Workers 设置正确

现在可以开始训练了! 🚀
