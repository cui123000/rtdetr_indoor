#!/bin/bash
# RT-DETR 稳定训练脚本
# 用于WSL2环境，添加了自动恢复和监控功能

set -e

# 配置
CONFIG="configs/rtdetr/rtdetr_r50vd_coco_indoor_4k.yml"
OUTPUT_DIR="/home/cui/rtdetr_indoor/output/rtdetr_r50vd_coco_indoor_4k"
CHECKPOINT="$OUTPUT_DIR/checkpoint.pth"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr

# 进入工作目录
cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch

echo "=========================================="
echo "RT-DETR 稳定训练启动"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

# 检查是否存在检查点
if [ -f "$CHECKPOINT" ]; then
    echo "✅ 找到检查点文件，从上次训练恢复..."
    RESUME_FLAG="--resume $CHECKPOINT"
else
    echo "ℹ️  未找到检查点，从头开始训练..."
    RESUME_FLAG=""
fi

# 启动训练
echo "🚀 开始训练..."
echo "=========================================="

# 前台直接运行
python tools/train.py \
    -c $CONFIG \
    $RESUME_FLAG \
    --amp \
    --seed 42
