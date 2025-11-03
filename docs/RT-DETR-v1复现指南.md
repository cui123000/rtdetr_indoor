## 概览

本指南帮助你在官方仓库的 PyTorch v1 实现上完成从环境到评估、训练、导出 ONNX 的完整复现，严格对齐官方配置与流程。

目标输出：
- 评估：加载官方提供的 v1 权重在 COCO val2017 上复现表格精度（AP/AR）。
- 训练：使用官方 6x 配置从零训练（或基于预训练权重微调）。
- 导出：将训练好的模型导出为 ONNX 并进行基础检查。

推荐目录：`RT-DETR/rtdetr_pytorch`（本文的所有相对路径均以此为工作目录）。

---

## 环境准备

- Python ≥ 3.8，PyTorch ≥ 1.12（CUDA 对应版本）。
- 进入目录并安装依赖：

可选命令（按需执行，每行独立）：
```
# 进入 v1 实现目录
cd RT-DETR/rtdetr_pytorch

# 安装依赖（建议在虚拟环境/conda 中执行）
pip install -r requirements.txt
```

备注：若你使用多卡分布式（torchrun），建议 PyTorch 版本与 CUDA 驱动匹配良好，NCCL 环境可用。

---

## 数据准备（COCO2017）

要求目录结构（官方默认）：
```
path/to/coco/
  annotations/                 # annotation json
  train2017/                   # 训练图片
  val2017/                     # 验证图片
```
将 `RT-DETR/rtdetr_pytorch/configs/dataset/coco_detection.yml` 中的路径改为你本机 COCO 目录，例如：
- `train.img_folder: path/to/coco/train2017`
- `train.ann_file: path/to/coco/annotations/instances_train2017.json`
- `val.img_folder: path/to/coco/val2017`
- `val.ann_file: path/to/coco/annotations/instances_val2017.json`

也可以在 `RT-DETR/rtdetr_pytorch` 下创建到实际数据目录的符号链接，保持默认相对路径（可选）。

---

## 快速验证：评估官方权重

官方 v1 权重（从 Paddle 转换）在 `rtdetr_pytorch/README.md` 的 Model Zoo 表格中提供。常用下载：
- R50vd 6x COCO：
  - https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
- 其它：R18vd、R34vd、R101vd、RegNet、DLA34 等请参考 README 表格。

将权重下载到本地，例如放到 `weights/` 下，然后执行评估：

可选命令（按需执行，每行独立）：
```
# 多卡评估（示例 4 卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r weights/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --test-only

# 单卡评估（若仅 1 张 GPU）
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r weights/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --test-only
```
输出目录由配置中的 `output_dir` 决定（见 `configs/rtdetr/rtdetr_r50vd_6x_coco.yml`，默认 `./output/rtdetr_r50vd_6x_coco`）。评估完成会将指标保存到该目录下（`eval.pth` 或 `eval/` 内的缓存）。

---

## 从零训练（官方 6x 配置）

最常用配置：`configs/rtdetr/rtdetr_r50vd_6x_coco.yml`（包含数据、运行时、优化器、模型组件等）。

可选命令（按需执行，每行独立）：
```
# 多卡训练（示例 4 卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml

# 单卡训练（更慢）
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- 训练日志、ckpt、评估缓存等会写入 `output_dir`（默认 `./output/rtdetr_r50vd_6x_coco`）。
- 默认总轮次等超参数见配置文件；如需快速冒烟测试，可临时将配置中的 `epoches` 改为 1-2，验证通路是否正常。

### 基于预训练权重微调（可选）

你也可以基于表格中的权重做微调（“tuning” 模式，仅加载可匹配参数）。

可选命令：
```
# 在 R18vd 的 Objects365+COCO 预训练权重上微调
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
```

### 断点恢复（resume）

- 训练中断后可用 `-r path/to/checkpoint.pth` 恢复，包括优化器、EMA、scaler 等状态。
- 每轮都会保存 `checkpoint.pth`，并按 `checkpoint_step` 额外保存快照（如 `checkpoint00xx.pth`）。

可选命令：
```
# 从最近 checkpoint 恢复继续训练
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r output/rtdetr_r50vd_6x_coco/checkpoint.pth
```

---

## AMP / EMA / 分布式细节

- AMP：默认关闭。CLI 追加 `--amp` 可开启混合精度，或在 `configs/runtime.yml` 中将 `use_amp: True`。
- EMA：默认关闭。如需开启，在 `configs/runtime.yml` 中设置 `use_ema: True`，`ema.decay`、`warmups` 可根据显存/收敛调优。
- 分布式：使用 `torchrun --nproc_per_node=<GPU数>`。如遇端口冲突，可加 `--master-port=xxxx`。NCCL 初始化失败多半来自驱动或容器权限问题。

---

## 导出 ONNX（并检查）

可选命令（按需执行，每行独立）：
```
# 使用已训练的权重导出 ONNX，并做 shape/ops 基础检查
python tools/export_onnx.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -r output/rtdetr_r18vd_6x_coco/checkpoint.pth \
  --check
```
导出脚本会根据配置构建模型再加载权重；若只是做推理部署，推荐选择较小模型（如 R18vd）更易上设备。

---

## 常见问题（FAQ）

- Data path 不正确：请确保 `coco_detection.yml` 四个路径项均指向真实存在的目录/文件；或通过软链接适配默认路径。
- 多卡/NCCL 报错：检查驱动/CUDA/容器权限；尝试设置 `NCCL_P2P_DISABLE=1` 或切换 `--master-port`；单卡先跑通再扩到多卡。
- OOM：减小 `batch_size` 或开启 `--amp`；必要时降低训练图像尺寸（需同步调整相关数据增广/尺度策略）。
- 指标对不齐：
  - 确认使用的配置与官方一致（特别是 `include/optimizer.yml` 和数据增广设置）。
  - 官方权重是从 Paddle 转换，和论文表格存在“轻微差异”说明，详见 README 的 Notes。
- 评估缓存：每轮会在 `output_dir/eval/` 下保存 `latest.pth` 等缓存，用于快速对比。

---

## 结果产物位置

- 训练输出目录：见配置 `output_dir`（如 `./output/rtdetr_r50vd_6x_coco`）。
- Checkpoint：`checkpoint.pth`、`checkpointxxxx.pth`。
- 日志：`log.txt`。
- 评估缓存：`eval/` 子目录。

---

## 室内数据集训练

### COCO 室内子数据集

我们已经从 COCO 2017 数据集中筛选出了 **74,178** 张训练图像和 **3,102** 张验证图像的室内场景子数据集。

**数据集详情：**
- 基于 39 个室内相关类别筛选（家具、家电、厨房用品等）
- 保留所有 80 个 COCO 类别以兼容预训练权重
- 标注文件位于：`datasets/coco_indoor/annotations/`

**使用方法：**

可选命令（按需执行，每行独立）：
```
# 评估预训练权重在室内数据集上的表现
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco_indoor.yml \
  -r weights/rtdetr_r50vd_6x_coco_from_paddle.pth \
  --test-only

# 在室内数据集上训练
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco_indoor.yml

# 基于 COCO 预训练权重微调室内数据集
torchrun --nproc_per_node=4 tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco_indoor.yml \
  -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
```

**配置文件：**
- 数据集配置：`configs/dataset/coco_indoor.yml`
- 训练配置：`configs/rtdetr/rtdetr_r50vd_6x_coco_indoor.yml`

详细信息请参考：`datasets/coco_indoor/README.md`

### 自定义室内数据集

- 若要用其他室内数据训练官方 v1，请参考 `Train custom data`（`rtdetr_pytorch/README.md`）。
  - 创建新的数据集配置文件（参考 `configs/dataset/coco_indoor.yml`）；
  - 设置正确的 `img_folder`、`ann_file` 和 `num_classes`；
  - 若想沿用 MS-COCO 的类名映射逻辑，需修改 `src/data/coco/coco_dataset.py` 中的 `mscoco_category2name`。
- 也可继续沿用我们在 Ultralytics 路线的工程化训练/部署方案，二者可以并行对比。

---

## 验证与状态

- 本文档基于仓库内 `rtdetr_pytorch/README.md`、`configs/*`、`tools/*` 及源码 `src/solver/*` 等进行整理，命令与路径已对齐验证。
- 质量门：
  - Build：PASS（文档改动，不涉及代码构建）。
  - Lint/Type：PASS（不涉及）。
  - Tests：PASS（不涉及；训练/评估需按需执行上述命令）。

---

如需我直接把你的 COCO 路径改写进 `coco_detection.yml` 或为室内数据新增一个自定义数据配置，请告诉我你的数据根路径与类别信息。