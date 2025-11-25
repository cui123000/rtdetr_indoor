# 🚀 RT-DETR 训练快速参考

## 📋 使用命令速查表

### 基础命令

```bash
# 显示所有可用模型（8个模型详情）
./train.sh list

# 查看训练状态
./train.sh status

# 显示实时日志
./train.sh log 1        # 查看模型1
./train.sh log 2        # 查看模型2
./train.sh log <1-8>    # 查看任意模型

# 停止所有训练
./train.sh stop
```

### 训练命令

```bash
# ✅ 训练单个模型
./train.sh 1            # 训练模型1 (RT-DETR-L)
./train.sh 2            # 训练模型2 (RT-DETR+MNV4)
./train.sh 5            # 训练模型5 (RT-DETR+GhostNet)

# ✅ 依次训练多个模型
./train.sh 1 2          # 先训练模型1，再训练模型2
./train.sh 1 2 3 4      # 依次训练模型1、2、3、4
./train.sh 5 6 7 8      # 依次训练所有轻量模型

# ✅ 训练所有8个模型
./train.sh 1 2 3 4 5 6 7 8
```

### 实时监控

```bash
# 查看模型1的训练进度（实时滚动）
tail -f training_model_1.log

# 查看最后20行日志
tail -20 training_model_1.log

# 检查是否完成
grep "🎉 训练完成" training_model_1.log

# 监控所有日志
watch -n 5 ./train.sh status
```

---

## 🎯 模型选择建议

### 按性能排序
1. **模型1** (RT-DETR-L) - ⭐⭐⭐⭐⭐ 最强（32.8M参数）
2. **模型4** (RT-DETR-L+SEA) - ⭐⭐⭐⭐⭐ 次强
3. **模型7** (RT-DETR+EfficientNet+CBAM) - ⭐⭐⭐⭐
4. **模型3** (RT-DETR+MNV4+SEA) - ⭐⭐⭐⭐
5. **模型2** (RT-DETR+MNV4) - ⭐⭐⭐

### 按速度排序（快到慢）
1. **模型5** (RT-DETR+GhostNet) - ⚡ 最快 1.5-2小时
2. **模型6** (RT-DETR+ShuffleNet+SEA) - ⚡ 1.5-2小时
3. **模型2** (RT-DETR+MNV4) - ⚡ 2-3小时
4. **模型7** (RT-DETR+EfficientNet+CBAM) - 2-3小时
5. **模型1** (RT-DETR-L) - ⏱️ 2-3小时

### 按显存需求排序（少到多）
1. **模型5、6** - 💾 18GB （最节省）
2. **模型2、7** - 💾 19-20GB
3. **模型3** - 💾 20GB
4. **模型1、4、8** - 💾 21GB （最多）

---

## 📊 推荐训练计划

### 方案A：快速验证（推荐首选）
```bash
# 优先训练最强的3个模型
./train.sh 1 2 4

# 预计总时间: ~6-9小时
```

### 方案B：全面评估
```bash
# 训练所有8个模型
./train.sh 1 2 3 4 5 6 7 8

# 预计总时间: ~16-20小时（晚上启动，早上完成）
```

### 方案C：轻量优化（适合实际部署）
```bash
# 训练所有轻量模型
./train.sh 2 5 6 7

# 预计总时间: ~8-10小时
```

---

## 📝 输出说明

训练日志文件位置：
- `training_model_1.log` - 模型1日志
- `training_model_2.log` - 模型2日志
- ...
- `training_model_8.log` - 模型8日志

训练结果保存位置：
- `/home/cjj/rtdetr_indoor/runs/detect/` - 模型权重
- `homeobjects_rtdetr_l_best_*.pt` - 模型1最佳权重
- `homeobjects_rtdetr_mnv4_best_*.pt` - 模型2最佳权重
- ...

---

## 🔧 常见操作

```bash
# 1️⃣ 启动并监控模型1
./train.sh 1 && tail -f training_model_1.log

# 2️⃣ 检查模型1是否完成
tail -20 training_model_1.log | grep -E "完成|mAP50"

# 3️⃣ 查看所有已完成的模型
for i in {1..8}; do 
  if grep -q "✅ 训练任务全部完成" training_model_$i.log 2>/dev/null; then
    echo "✅ 模型 $i 已完成"
  fi
done

# 4️⃣ 比较所有模型的最终性能
for i in {1..8}; do
  echo -n "模型 $i: "
  grep "mAP50=" training_model_$i.log | tail -1 || echo "未完成"
done
```

---

## 💡 提示

- ✅ **推荐做法**: 先运行方案A快速验证，再根据结果选择最有潜力的模型继续优化
- ✅ **GPU优化**: 模型顺序启动，每个模型2-3小时，显存充足
- ✅ **监控方式**: 使用 `./train.sh status` 随时检查所有进程
- ✅ **日志查看**: `tail -f training_model_X.log` 查看实时进度

---

## ❌ 常见问题

**Q: 能同时运行多个模型吗？**
A: 不建议！这样会导致显存不足。建议单个运行，自动顺序执行。

**Q: 如何中途换模型？**
A: 先 `./train.sh stop` 停止当前，再启动新模型。

**Q: 日志太多怎么清理？**
A: `rm training_model_*.log` 删除所有日志（谨慎！）

**Q: 如何在已有日志的基础上继续？**
A: 直接运行新命令，会自动覆盖旧日志。

---
