#!/bin/bash

# RT-DETR自动训练启动脚本 (解决权限问题)

echo "🚀 RT-DETR HomeObjects 自动训练启动器"
echo "================================================="

# 初始化conda (解决conda命令找不到的问题)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ conda命令不可用，尝试手动初始化..."
    # 尝试常见的conda路径
    if [ -f ~/miniconda3/bin/conda ]; then
        export PATH="~/miniconda3/bin:$PATH"
    elif [ -f ~/anaconda3/bin/conda ]; then
        export PATH="~/anaconda3/bin:$PATH"
    elif [ -f /opt/conda/bin/conda ]; then
        export PATH="/opt/conda/bin:$PATH"
    else
        echo "❌ 找不到conda，请确保conda已安装"
        exit 1
    fi
fi

# 检查conda环境
if ! conda info --envs | grep -q "rtdetr"; then
    echo "❌ conda环境 'rtdetr' 不存在"
    echo "请先创建conda环境: conda create -n rtdetr python=3.8"
    exit 1
else
    echo "✅ 发现conda环境 'rtdetr'"
fi

# 检查权重保存目录
echo "📁 检查权重保存目录..."
if [ -d "/root/autodl-tmp/rtdetr_weights" ] && [ -w "/root/autodl-tmp/rtdetr_weights" ]; then
    echo "✅ 权重目录已就绪: /root/autodl-tmp/rtdetr_weights"
else
    echo "⚠️ /root/autodl-tmp/rtdetr_weights 不可写，将使用备用目录"
fi

# 启动tmux会话进行训练
echo "🖥️  启动tmux训练会话..."

# 检查是否已有训练会话
if tmux has-session -t rtdetr_training 2>/dev/null; then
    echo "⚠️ 发现已存在的训练会话"
    echo "1. 查看现有会话: tmux attach -t rtdetr_training"
    echo "2. 强制重新开始: 按任意键继续..."
    read -n 1 -s
    tmux kill-session -t rtdetr_training
fi

# 创建新的训练会话
tmux new-session -d -s rtdetr_training -c /home/cui/rtdetr_indoor

# 在tmux中激活环境并开始训练
tmux send-keys -t rtdetr_training "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null" Enter
tmux send-keys -t rtdetr_training "conda activate rtdetr" Enter
tmux send-keys -t rtdetr_training "export CUDA_VISIBLE_DEVICES=0" Enter
tmux send-keys -t rtdetr_training "export CUBLAS_WORKSPACE_CONFIG=:4096:8" Enter
tmux send-keys -t rtdetr_training "export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256,expandable_segments:False'" Enter
tmux send-keys -t rtdetr_training "cd /home/cui/rtdetr_indoor" Enter
tmux send-keys -t rtdetr_training "python /home/cui/rtdetr_indoor/scripts/training/auto_train_rtdetr.py" Enter

echo "✅ 训练已在tmux会话中启动"
echo ""
echo "📋 有用的命令:"
echo "   查看训练进度: tmux attach -t rtdetr_training"
echo "   从会话分离: Ctrl+B, 然后按 D"
echo "   停止训练: tmux kill-session -t rtdetr_training"
echo "   查看所有tmux会话: tmux ls"
echo ""
echo "🎯 训练配置信息:"
echo "   默认选择: 模型 1 (RT-DETR-L)"
echo "   修改模型: 编辑 auto_train_rtdetr.py 中的 SELECTED_MODEL"
echo "   权重保存: /root/autodl-tmp/rtdetr_weights/"
echo ""
echo "🔥 正在后台训练中... 使用 tmux attach -t rtdetr_training 查看"