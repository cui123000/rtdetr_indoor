#!/bin/bash
# 训练RT-DETR-L在平衡数据集上（9094张图片，30个类别）
# 针对A40 GPU优化

cd "$(dirname "$0")"
cd rtdetr_pytorch

echo "============================================"
echo "RT-DETR-L训练 - 平衡数据集"
echo "============================================"
echo "数据集: coco_indoor_balanced"
echo "图片数: 9094"
echo "类别数: 30"
echo "模型: RT-DETR-L"
echo "GPU: A40 (40GB)"
echo "批次大小: 48"
echo "Epochs: 80"
echo "============================================"

python tools/train.py \
  --data ../datasets/coco_indoor_balanced/data.yaml \
  --conf configs/rtdetr/rtdetr_l_6x_coco.yml \
  --epochs 80 \
  --batch-size 48 \
  --device 0 \
  --workers 16 \
  --imgsz 640 \
  --amp \
  --dropout 0.0 \
  --clip-grad 100 \
  --warmup-epochs 3 \
  --close-mosaic 10 \
  --output-dir ../runs/detect/rtdetr_l_coco_indoor_balanced

echo ""
echo "训练完成！"
echo "结果保存在: runs/detect/rtdetr_l_coco_indoor_balanced/"
