#!/bin/bash
# RT-DETR 后台训练启动脚本

cd /home/cjj/rtdetr_indoor

# 检查是否已有训练进程在运行
if pgrep -f "auto_train_rtdetr" > /dev/null; then
    echo "⚠️ 训练进程已在运行，退出..."
    exit 1
fi

# 获取模型选择参数（默认为模型1）
MODEL=${1:-1}

echo "🚀 启动 RT-DETR 训练..."
echo "📌 模型选择: 模型 $MODEL"
echo "📝 日志文件: training_model_${MODEL}.log"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 根据模型号修改脚本
sed -i "s/^SELECTED_MODEL = '[0-9]'/SELECTED_MODEL = '$MODEL'/" /home/cjj/rtdetr_indoor/scripts/training/auto_train_rtdetr.py

# 后台运行训练，使用nohup和disown避免进程重复
nohup python3 scripts/training/auto_train_rtdetr.py --skip-confirm > training_model_${MODEL}.log 2>&1 &
TRAIN_PID=$!

echo "✅ 训练已启动"
echo "📊 进程ID: $TRAIN_PID"
echo "🔍 监控命令: tail -f training_model_${MODEL}.log"
echo ""
echo "💾 GPU显存需求:"
echo "   - 模型1 (RT-DETR-L): ~21GB"
echo "   - 模型2 (RT-DETR+MNV4): ~20GB"
echo ""

# 显示启动时间
disown $TRAIN_PID
echo "⏰ 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "💡 提示: 使用 'jobs -l' 或 'ps aux | grep auto_train' 查看进程状态"
