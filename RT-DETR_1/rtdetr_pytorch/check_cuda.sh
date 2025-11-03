#!/bin/bash
# 快速检查CUDA是否可用

echo "检查CUDA驱动..."

if nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA驱动正常"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    # 测试PyTorch CUDA
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate rtdetr
    python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
else
    echo "❌ NVIDIA驱动不可用"
    echo ""
    echo "解决方法："
    echo "1. 在Windows PowerShell（管理员）运行: wsl --shutdown"
    echo "2. 等待10秒"
    echo "3. 重新打开WSL2"
    echo "4. 如果还不行，重启Windows系统"
fi
