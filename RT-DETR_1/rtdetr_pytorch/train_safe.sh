#!/bin/bash
# 极度保守的训练启动 - 逐步加载以避免崩溃

set -e

CONFIG="configs/rtdetr/rtdetr_r50vd_coco_indoor_4k.yml"
OUTPUT_DIR="/home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k"
CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch

echo "=========================================="
echo "🐢 极度保守模式训练启动"
echo "=========================================="

# 步骤1: GPU预热（避免冷启动冲击）
echo -e "\n🔥 步骤1: GPU预热（5秒）..."
python << 'EOF'
import torch
import time
model = torch.nn.Linear(100, 100).cuda()
for i in range(3):
    x = torch.randn(10, 100).cuda()
    y = model(x)
    time.sleep(1)
    print(f"  预热 {i+1}/3")
del model, x, y
torch.cuda.empty_cache()
print("✅ 预热完成")
EOF

sleep 2

# 步骤2: 等待GPU温度稳定
echo -e "\n🌡️  步骤2: 等待GPU温度稳定（3秒）..."
sleep 3
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
echo "  当前GPU温度: ${TEMP}°C"

# 步骤3: 检查检查点
if [ -f "$CHECKPOINT" ]; then
    echo -e "\n💾 步骤3: 发现检查点，将从上次恢复"
    RESUME_FLAG="--resume $CHECKPOINT"
else
    echo -e "\n🆕 步骤3: 从头开始训练"
    RESUME_FLAG=""
fi

# 步骤4: 逐步启动训练
echo -e "\n🚀 步骤4: 启动训练..."
echo "=========================================="
echo "⚙️  配置: batch_size=4, num_workers=2"
echo "⚙️  混合精度: 启用"
echo "⚙️  Epochs: 100"
echo "=========================================="

sleep 1

# 启动训练
python tools/train.py \
    -c $CONFIG \
    $RESUME_FLAG \
    --amp \
    --seed 42

echo -e "\n=========================================="
echo "训练结束或中断"
echo "=========================================="
