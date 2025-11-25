#!/bin/bash
# 使用清理后的大规模室内数据集训练 RT-DETR-L
# 配置针对 A40 GPU 优化

cd /home/cjj/rtdetr_indoor/rtdetr_pytorch

python tools/train.py \
  --data /home/cjj/rtdetr_indoor/datasets/coco_indoor_auto/data.yaml \
  --conf configs/rtdetr/rtdetr_l_6x_coco.yml \
  --batch-size 48 \
  --epochs 80 \
  --device 0 \
  --output-dir ../runs/detect/rtdetr_l_coco_indoor_clean \
  --num-workers 16 \
  --img-size 640 \
  --warmup-epochs 3 \
  --close-mosaic 10 \
  --amp \
  --use-aux-loss

echo "✅ Training complete!"
