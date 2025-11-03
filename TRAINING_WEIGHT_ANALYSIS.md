# 🔍 训练权重配置分析报告

## 问题发现

### 两个脚本的预训练权重配置对比

| 脚本 | RT-DETR-L | RT-DETR-MNV4 | 问题 |
|------|-----------|--------------|------|
| **train_coco_indoor.py** (旧) | ✅ `rtdetr-l.pt` | ❌ `rtdetr-mnv4-hybrid-m.yaml` | MNV4从零训练 |
| **train_mnv4_variants.py** | ❌ `rtdetr-l.yaml` | ❌ `rtdetr-mnv4-hybrid-m.yaml` | 全部从零训练 |

### 训练结果对比

#### COCO Indoor 数据集 (阈值30: 3,015张训练图)

| 模型 | 预训练权重 | mAP50-95 | Precision | Recall | cls_loss变化 |
|------|----------|----------|-----------|--------|--------------|
| **RT-DETR-L** | ✅ COCO 80类 | **0.235** | 0.623 | 0.309 | 0.78→0.44 ✅ |
| **RT-DETR-MNV4** | ❌ 无 | **0.053** | 0.336 | 0.125 | 0.38→0.77 ❌ |

**性能差距：L 是 MNV4 的 4.5倍！**

#### HomeObjects 数据集 (3K 简单家居物品)

| 模型 | 预训练权重 | mAP50-95 | Precision | 训练轮数 |
|------|----------|----------|-----------|----------|
| **RT-DETR-L** | ❌ 无 | 0.216 | 0.326 | 120 |
| **RT-DETR-MNV4** | ❌ 无 | **0.269** | **0.491** | 120 |

**MNV4 反而更好！因为数据集简单，从零训练也能学好。**

---

## 根本原因

### 1️⃣ **MNV4 在 COCO Indoor 从零训练失败的证据**

```
训练崩溃标志：
- cls_loss 不降反升: 0.376 → 0.769 (+104%)
- 前3个epoch完全无检测: mAP50-95 = 0
- 最终性能仅为L的22%
```

### 2️⃣ **数据集复杂度差异**

| 特征 | HomeObjects | COCO Indoor |
|------|-------------|-------------|
| 场景类型 | 简单家居 | 复杂室内场景 |
| 平均对象数 | ~10个/图 | ~20个/图 |
| 对象密度 | 低 | 高 |
| 从零训练可行性 | ✅ 可行 | ❌ 失败 |

### 3️⃣ **为什么 train_mnv4_variants.py 在 HomeObjects 成功？**

- ✅ 数据集简单（10个对象/图 vs 20个）
- ✅ 训练轮数足够（120 epochs）
- ✅ 场景单一（只有家居物品）
- ✅ 轻量模型对简单任务够用

---

## 🛠️ 改进方案

### 已实施：使用 HomeObjects 预训练权重微调

```python
# 修改后的配置 (train_coco_indoor.py)
'rtdetr-mnv4': {
    'pretrained': '/home/cui/rtdetr_indoor/runs/detect/rtdetr_mnv4_single_bs8/weights/best.pt',
    'config_file': None,
    'batch': 4,
    'lr0': 0.001,  # 微调用较低学习率
    'name': 'rtdetr_mnv4_coco_indoor_finetuned',
}
```

### 预期效果

| 指标 | 从零训练 (旧) | 预训练微调 (新) | 预期提升 |
|------|--------------|----------------|----------|
| **mAP50-95** | 0.053 | ~0.18-0.22 | **+240-315%** |
| **Precision** | 0.336 | ~0.45-0.52 | **+34-55%** |
| **Recall** | 0.125 | ~0.25-0.30 | **+100-140%** |
| **收敛速度** | 慢/不收敛 | 快速收敛 | 30-50 epochs 即可 |

### 为什么这个方案有效？

1. ✅ **迁移学习**: MNV4 已在 HomeObjects 学会了基础物体检测能力
2. ✅ **领域相似**: 两个数据集都是室内场景
3. ✅ **类别重叠**: 很多物体类别相同（杯子、瓶子、家具等）
4. ✅ **微调效率高**: 只需调整高层特征，backbone特征提取能力保留

---

## 📊 训练命令

### 重新训练 MNV4（使用预训练权重）

```bash
cd /home/cui/rtdetr_indoor
python scripts/training/train_coco_indoor.py \
    --model rtdetr-mnv4 \
    --epochs 100 \
    --batch 4
```

### 对比训练 (L vs MNV4)

```bash
# RT-DETR-L (已完成，作为baseline)
# mAP50-95: 0.235

# RT-DETR-MNV4 (新配置，带预训练)
python scripts/training/train_coco_indoor.py --model rtdetr-mnv4 --epochs 100
```

---

## 🎯 关键发现总结

1. **train_mnv4_variants.py 所有模型都从零训练** - YAML文件而非.pt权重
2. **简单数据集(HomeObjects)从零训练可行** - MNV4甚至超过L
3. **复杂数据集(COCO)从零训练失败** - MNV4性能仅为L的22%
4. **预训练权重是关键** - RT-DETR-L使用COCO预训练，MNV4没有
5. **迁移学习是最优方案** - 用HomeObjects权重微调到COCO Indoor

---

## 📝 后续计划

- [ ] 使用新配置重新训练 MNV4
- [ ] 对比 L vs MNV4-finetuned 性能
- [ ] 如果效果好，更新 train_mnv4_variants.py 添加预训练选项
- [ ] 考虑训练 MNV4+SEA 的预训练版本

