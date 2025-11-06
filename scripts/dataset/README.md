# COCO-Indoor 构建（基于 Places365 室内/室外判别）

本工具通过 Places365 预训练分类器对 COCO 图像进行室内/室外判别，筛出“室内”图像并裁剪原 COCO 标注，生成 coco-indoor 的 train/val 标注文件。

## 脚本位置

- `scripts/dataset/build_coco_indoor_places365.py`

## 依赖

- Python 3.8+
- PyTorch, torchvision, Pillow, tqdm
- 可选（建议）：GPU 环境（显著加速推理）

安装建议（示例，仅供参考）：

```bash
pip install torch torchvision pillow tqdm
```

> 说明：脚本会在首次运行时自动下载 Places365 权重与类别/IO 映射（默认缓存到 `~/.cache/places365`）。若无法联网，请提前手动下载到该目录：
>
> - `resnet50_places365.pth.tar`
> - `categories_places365.txt`
> - `IO_places365.txt`

## 使用方法

假设 COCO 目录结构为：
```
/path/to/coco
  ├── train2017/
  ├── val2017/
  └── annotations/
      ├── instances_train2017.json
      └── instances_val2017.json
```

运行示例：

```bash
python scripts/dataset/build_coco_indoor_places365.py \
  --train-images /path/to/coco/train2017 \
  --train-anno   /path/to/coco/annotations/instances_train2017.json \
  --val-images   /path/to/coco/val2017 \
  --val-anno     /path/to/coco/annotations/instances_val2017.json \
  --out-train    datasets/coco_indoor/annotations/instances_train2017_coco_indoor.json \
  --out-val      datasets/coco_indoor/annotations/instances_val2017_coco_indoor.json \
  --batch-size 128 --topk 5
```

参数说明：
- `--batch-size`：推理批大小（默认 128）。如果显存不足，适当调小。
- `--topk`：按 Top-K 类别的概率加权投票决定 Indoor/Outdoor（默认 5）。
- `--limit`：调试用，仅处理前 N 张图。
- `--device`：`cuda`/`cpu`（默认自动检测）。
- `--cache-dir`：Places365 资源缓存目录（默认 `~/.cache/places365`）。

## 判定逻辑

- 使用 Places365 ResNet-50 预测 365 个场景类别概率。
- 取 Top-K 类别，根据官方 `IO_places365.txt`（0=Indoor, 1=Outdoor）做概率加权平均；加权值 < 0.5 判定为 Indoor。

## 输出

- 生成与 COCO 原始格式一致的 `instances_*.json`，仅保留室内图像及其对应标注；`categories` 等元信息原样保留。

## 常见问题

- 无法下载权重/映射文件：将对应文件手动放置到 `--cache-dir` 指定目录。
- 速度慢：
  - 使用 GPU (`--device cuda`)
  - 增大 `--batch-size`
  - 同时只处理一个 split（先 train 再 val）

## 后续

- 若需导出 Ultralytics 风格的数据 yaml，可基于输出的 `instances_*.json` 自行撰写 yaml 指向对应 images 目录与 annotations。
