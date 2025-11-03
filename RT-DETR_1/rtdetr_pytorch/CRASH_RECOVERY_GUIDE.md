# RT-DETR 训练崩溃问题解决方案

## 🔴 问题：训练过程中服务器崩溃重启

### 可能原因（WSL2环境）

1. **GPU负载过高触发Windows保护机制**
2. **系统内存不足**
3. **WSL2内存限制**
4. **电源管理问题**

## ✅ 已采取的稳定性措施

### 1. 降低资源占用
```yaml
# configs/rtdetr/include/dataloader.yml
batch_size: 6        # 从8降到6（GPU显存约8-9G）
num_workers: 2       # 从4降到2（减少CPU和内存压力）
```

### 2. 创建稳定训练脚本
```bash
# 使用 train_stable.sh 启动训练
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch
./train_stable.sh
```

**功能：**
- ✅ 自动检测并恢复检查点
- ✅ 后台运行（使用nohup）
- ✅ 日志持久化保存
- ✅ 显示GPU信息

### 3. 实时监控工具
```bash
# 另开一个终端监控训练状态
python monitor_training.py
```

**监控内容：**
- GPU利用率、显存、温度、功耗
- 训练epoch进度
- 实时loss和mAP
- 自动警告异常状态

## 🛠️ 进一步稳定性建议

### 方案A：限制WSL2内存（推荐）

创建 `C:\Users\你的用户名\.wslconfig`：
```ini
[wsl2]
memory=16GB          # 限制WSL2最大内存
processors=8         # 限制CPU核心数
swap=8GB            # 交换空间
localhostForwarding=true
```

保存后重启WSL2：
```powershell
# Windows PowerShell (管理员)
wsl --shutdown
```

### 方案B：使用梯度累积（减少显存）

如果batch_size=6还是不稳定，可以改为：
```yaml
batch_size: 4        # 更保守
accumulation_steps: 2  # 梯度累积，等效batch_size=8
```

### 方案C：使用tmux保持会话

```bash
# 安装tmux（如果没有）
sudo apt install tmux

# 创建训练会话
tmux new -s rtdetr_train

# 在tmux中启动训练
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch
./train_stable.sh

# 分离会话: Ctrl+B 然后按 D
# 重新连接: tmux attach -t rtdetr_train
```

## 📋 训练恢复步骤

如果训练中断，按以下步骤恢复：

### 1. 检查检查点文件
```bash
ls -lh /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/checkpoint*.pth
```

### 2. 查看最后训练的epoch
```bash
tail -n 1 /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/log.txt | jq '.epoch'
```

### 3. 重启训练（自动恢复）
```bash
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch
./train_stable.sh  # 会自动检测checkpoint并恢复
```

或手动指定检查点：
```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_coco_indoor_4k.yml \
  --resume /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/checkpoint.pth \
  --amp --seed 42
```

## 🔍 诊断命令

### 检查GPU状态
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 查看GPU温度和功耗
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv
```

### 检查系统资源
```bash
# CPU和内存
htop

# 磁盘空间
df -h

# WSL2内存使用
free -h
```

### 检查训练日志
```bash
# 查看最新训练日志
tail -f /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/train_*.log

# 查看训练历史
cat /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/log.txt | jq '.epoch, .train_loss, .test_coco_eval_bbox[0]'
```

## 📊 当前配置总结

| 配置项 | 值 | 说明 |
|--------|-----|------|
| batch_size | 6 | 降低显存使用 |
| num_workers | 2 | 减少CPU/内存压力 |
| epochs | 100 | 总训练轮数 |
| 预期显存 | ~8-9GB | 在安全范围内 |
| 每epoch步数 | ~667 | 4000/6 |

## ⚠️ 警告信号

如果看到以下情况，需要立即调整：

- ❌ GPU温度 > 85°C
- ❌ GPU显存使用 > 95%
- ❌ 系统内存使用 > 90%
- ❌ 训练速度突然变慢 (<1 it/s)
- ❌ 频繁的CUDA OOM错误

## 🎯 推荐训练流程

```bash
# 1. 启动训练（后台）
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch
./train_stable.sh

# 2. 开启监控（另一个终端）
python monitor_training.py

# 3. 定期检查（每小时）
tail /home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k/log.txt

# 4. 如果崩溃，重启即可自动恢复
./train_stable.sh
```

---

**更新时间**: 2025-10-31  
**配置版本**: Stable v1.0
