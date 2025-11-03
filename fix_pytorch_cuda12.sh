#!/bin/bash
# 修复 PyTorch CUDA 12.1 安装问题
# 解决 "undefined symbol: iJIT_NotifyEvent" 错误

set -e

echo "========================================================================"
echo "🔧 修复 PyTorch CUDA 12.1 安装"
echo "========================================================================"
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr

echo "📦 当前 PyTorch 版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" 2>/dev/null || echo "无法导入 PyTorch（需要修复）"
echo ""

echo "🗑️  步骤 1/4: 完全卸载现有 PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
conda uninstall -y pytorch torchvision torchaudio pytorch-cuda --force 2>/dev/null || true
echo "✅ 卸载完成"
echo ""

echo "🧹 步骤 2/4: 清理缓存和冲突的包..."
pip cache purge
conda clean -a -y
# 清理可能的残留文件
rm -rf ~/.cache/torch
rm -rf ~/miniconda3/envs/rtdetr/lib/python3.10/site-packages/torch*
# 卸载可能冲突的 transformers（在 ~/.local 中）
pip uninstall -y transformers 2>/dev/null || true
echo "✅ 缓存清理完成"
echo ""

echo "📥 步骤 3/4: 重新安装 PyTorch 2.1.2 + CUDA 12.1..."
# 使用 pip 安装（比 conda 更可靠）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 如果需要 transformers，安装兼容版本
if pip list | grep -q transformers; then
    echo "📥 安装兼容的 transformers..."
    pip install --upgrade transformers>=4.36.0
fi
echo "✅ 安装完成"
echo ""

echo "🔍 步骤 4/4: 验证安装..."
python -c "
import torch
import torchvision
import torchaudio

print('=' * 72)
print('✅ PyTorch 安装成功！')
print('=' * 72)
print(f'PyTorch 版本: {torch.__version__}')
print(f'TorchVision 版本: {torchvision.__version__}')
print(f'TorchAudio 版本: {torchaudio.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'cuDNN 版本: {torch.backends.cudnn.version()}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
print('=' * 72)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 修复成功！现在可以重新运行训练："
    echo ""
    echo "   cd /home/cui/rtdetr_indoor/RT-DETR/rtdetr_pytorch"
    echo "   python train_coco_indoor_4k.py"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ 验证失败，请检查错误信息"
    echo "========================================================================"
    exit 1
fi
