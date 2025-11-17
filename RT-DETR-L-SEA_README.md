# RT-DETR-L-SEA 模型说明

## 模型概述

**RT-DETR-L-SEA** 是在标准 RT-DETR-L 基础上添加了 SEA (Squeeze-enhanced Axial) 注意力模块的增强版本。

### 架构特点

1. **基础架构**: RT-DETR-L (HGNetv2 backbone + DETR head)
2. **增强模块**: OptimizedSEA_Attention
3. **插入位置**: 
   - Backbone Stage 3 之后 (P3/16 特征层)
   - Backbone Stage 4 之后 (P4/32 特征层)

### 模型对比

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| **RT-DETR-L** | 32.8M | 标准版本，速度快 | 实时检测 |
| **RT-DETR-L-SEA** | 77.1M | 加强版，精度高 | 高精度需求 |
| **RT-DETR+MNV4** | - | 轻量级 | 移动端 |
| **RT-DETR+MNV4+SEA** | - | 轻量增强 | 移动端高精度 |

### SEA 注意力优势

1. **轴向注意力**: 分别在 H 和 W 维度计算注意力，降低计算复杂度
2. **通道挤压**: SE 模块增强通道特征表达
3. **空间感知**: 保持空间结构信息

### 配置文件

**位置**: `/home/cjj/rtdetr_indoor/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l-sea.yaml`

**关键修改**:
```yaml
backbone:
  # ... Stage 3
  - [-1, 1, OptimizedSEA_Attention, [1024]]  # 添加 SEA
  
  # ... Stage 4
  - [-1, 1, OptimizedSEA_Attention, [2048]]  # 添加 SEA
```

### 训练配置

在 `scripts/training/auto_train_rtdetr.py` 中选择模型 4:

```python
SELECTED_MODEL = '4'  # RT-DETR-L+SEA
```

**推荐参数**:
- Batch size: 10
- Learning rate: 0.0001
- Epochs: 100
- Cache: True
- AMP: False (稳定性优先)

### 使用方法

#### 1. 测试模型加载

```bash
python3 test_rtdetr_l_sea.py
```

#### 2. 开始训练

```bash
# 修改脚本中的 SELECTED_MODEL = '4'
python3 scripts/training/auto_train_rtdetr.py
```

#### 3. 推理使用

```python
from ultralytics import RTDETR

# 加载训练好的模型
model = RTDETR('runs/detect/train_rtdetr_l_sea_xxx/weights/best.pt')

# 推理
results = model('image.jpg')
```

### 性能预估

#### 训练速度 (A40 GPU 共享环境)
- **速度**: ~1.0 it/s
- **每 epoch**: ~13-15 分钟
- **100 epochs**: ~22-25 小时

#### 训练速度 (A40 GPU 独占)
- **速度**: ~2.0 it/s  
- **每 epoch**: ~7-8 分钟
- **100 epochs**: ~12-14 小时

### 预期效果

相比 RT-DETR-L:
- **精度提升**: +2-5% mAP (理论值)
- **速度下降**: 15-20% (参数增加导致)
- **显存增加**: +2-3GB

### 注意事项

1. **参数量**: 77M 参数，比标准版大 2.4 倍
2. **显存需求**: 建议至少 12GB VRAM
3. **训练时间**: 比标准版长 20-30%
4. **适用场景**: 精度优先，速度要求不太严格的场景

### 文件清单

```
ultralytics/ultralytics/cfg/models/rt-detr/
├── rtdetr-l.yaml              # 标准版
├── rtdetr-l-sea.yaml          # ← 新增 SEA 版本
├── rtdetr-mnv4-hybrid-m.yaml
└── rtdetr-mnv4-hybrid-m-sea.yaml

scripts/training/
└── auto_train_rtdetr.py       # 训练脚本（已添加模型4选项）

test_rtdetr_l_sea.py           # 模型测试脚本
```

### 下一步

选择使用 RT-DETR-L-SEA 进行训练：

```bash
# 在脚本中修改
SELECTED_MODEL = '4'

# 运行训练
python3 scripts/training/auto_train_rtdetr.py
```
