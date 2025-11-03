# RT-DETR 训练配置对比

## 📊 配置版本对比

| 配置项 | 安全模式 (safe) | 最优模式 (推荐) | 性能提升 |
|--------|----------------|----------------|----------|
| **Batch Size** | 2 | 6 (rtdetr-l) | **3x** |
| **Workers** | 2 | 8 | **4x** |
| **Cache** | False | ram | **显著加速** |
| **CPU Threads** | 2 | 8 | **4x** |
| **显存限制** | 85% | 无限制 | 充分利用 |
| **Split Size** | 64MB | 512MB | 减少碎片 |
| **显存清理** | 每50 batch | 每100 batch | 平衡 |

## ⚡ 预期性能

### 安全模式 (train_coco_indoor_safe.py)
```
速度: ~2.4 it/s
单epoch: ~13分钟
100 epochs: ~22小时
显存: ~2-4G (非常保守)
```

### 最优模式 (train_coco_indoor.py) ⭐ 推荐
```
速度: ~7-8 it/s (预估)
单epoch: ~4-5分钟
100 epochs: ~7-8小时
显存: ~12-16G (稳定，有自动清理)
```

## 🎯 选择建议

### 使用最优模式，如果你：
- ✅ GPU显存 ≥ 16GB
- ✅ 想要快速完成训练
- ✅ 可以监控训练过程

**命令:**
```bash
python scripts/training/train_coco_indoor.py --model rtdetr-l --epochs 100
```

### 使用安全模式，如果你：
- ⚠️ GPU显存 < 12GB
- ⚠️ 需要绝对稳定（无人值守）
- ⚠️ 遇到最优模式OOM

**命令:**
```bash
python scripts/training/train_coco_indoor_safe.py --model rtdetr-l --epochs 100
```

## 🔧 最优模式关键优化

1. **RAM缓存**: 4K数据集(~2GB)全部缓存到内存
   - 消除磁盘I/O瓶颈
   - 需要确保系统内存充足

2. **自动显存清理**: 每100个batch清理一次
   - 防止显存泄漏
   - 最小化性能影响

3. **动态显存管理**: 
   - `max_split_size_mb:512` - 减少碎片化
   - `garbage_collection_threshold:0.7` - 适度回收
   - `expandable_segments:True` - 允许扩展

4. **并行加载**: 8 workers + 8 threads
   - 充分利用CPU多核
   - 数据预处理并行化

## 📈 监控命令

```bash
# 监控GPU使用（另一个终端）
watch -n 2 nvidia-smi

# 查看训练日志
tail -f runs/detect/rtdetr_l_coco_indoor/train.log

# 实时查看显存变化
watch -n 1 "nvidia-smi | grep python"
```

## ⚠️ 如果最优模式出现OOM

**快速降级方案:**
```bash
# 方案1: 降低batch size
python scripts/training/train_coco_indoor.py --model rtdetr-l --batch 4 --epochs 100

# 方案2: 切换到MNV4（更小的模型）
python scripts/training/train_coco_indoor.py --model rtdetr-mnv4 --epochs 150

# 方案3: 使用安全模式
python scripts/training/train_coco_indoor_safe.py --model rtdetr-l --epochs 100
```

## 🎓 技术细节

### 为什么之前会OOM？
- ❌ 不是显存不足（你的GPU>24G）
- ✅ 是**显存泄漏** - 中间结果未及时释放
- ✅ 是**碎片化** - 小块内存分配累积

### 最优配置如何解决？
1. **定期清理**: callback每100 batch清空缓存
2. **大块分配**: max_split_size提升到512MB
3. **RAM缓存**: 数据在内存而非显存累积
4. **关闭rect**: 避免动态shape导致的内存问题

