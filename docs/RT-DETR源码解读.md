# RT-DETR 源码解读（v1 / v2，PyTorch 重点）

本文档解读你工作区中的官方仓库 `RT-DETR/`（已包含 v1 与 v2 的 Paddle 与 PyTorch 实现），重点聚焦 `rtdetrv2_pytorch/` 的结构、数据流、关键模块、配置系统与可扩展点，帮助快速二开与排错。

---

## 一、快速结论（给决策者）
- 生产建议：优先使用 `rtdetrv2_pytorch` 作为基线，v2 在不增推理开销或轻微开销下提供更稳健的训练“免费午餐（BoF）”。
- 扩展建议：
  - 替换主干（如 MobileNet 系）时，务必对齐 `HybridEncoder.in_channels` 与 `feat_strides`。
  - 小目标/TRT 兼容：可用 `cross_attn_method: discrete` 与 `num_points` 列表做推理兼容与速度/精度权衡。
  - 收敛与性能：可迁移 DINO/DAB/DN 等 DETR 家族增强策略（已部分体现在 v2）。

---

## 二、仓库结构总览

顶层：
- `README.md / README_cn.md`：论文与更新日志
- `benchmark/`：TensorRT/ONNX 等加速与基准
- 实现分支：
  - `rtdetr_pytorch/`：RT-DETR v1 PyTorch
  - `rtdetrv2_pytorch/`：RT-DETR v2 PyTorch（本解读重点）
  - `rtdetr_paddle/`、`rtdetrv2_paddle/`：Paddle 实现

v2（PyTorch）目录关键路径：
- `rtdetrv2_pytorch/configs/`：YAML 配置（模型/数据/运行时/优化器）
- `rtdetrv2_pytorch/src/`：源码
  - `core/`：注册器、配置装配与工作空间
  - `nn/backbone/`：主干网络（`presnet.py`、`csp_resnet.py`、`hgnetv2.py`、`timm_model.py` 等）
  - `zoo/rtdetr/`：RT-DETR 关键模块（编码器/解码器/损失/后处理）
- `rtdetrv2_pytorch/tools/`：训练与导出脚本（`train.py`、`export_onnx.py`、`export_trt.py`、`run_profile.py`）
- `rtdetrv2_pytorch/references/deploy/`：PyTorch/ONNXRuntime/TensorRT/OpenVINO 推理脚本

---

## 三、数据流与输入输出“契约”

1) `RTDETR` 顶层装配（`src/zoo/rtdetr/rtdetr.py`）
- 依赖注入：`__inject__ = ['backbone','encoder','decoder']`
- 前向：`x -> backbone -> encoder -> decoder`

2) I/O 约定
- 输入图像：`(B,3,H,W)`
- backbone 输出：多尺度特征（通常 3 路，strides 8/16/32）
- encoder 输出：统一到 `hidden_dim`（默认 256）的三尺度特征
- decoder 输出字典：
  - `pred_logits`: `(B, Q, num_classes)`（未 Sigmoid）
  - `pred_boxes`: `(B, Q, 4)`（归一化 `cx,cy,w,h`）
  - `aux_outputs`: 中间层监督

---

## 四、关键模块详解（v2）

### 4.1 HybridEncoder（`src/zoo/rtdetr/hybrid_encoder.py`）
- 作用：整合多尺度特征，通道统一为 `hidden_dim`；包含 `CSPRepLayer` + `RepVggBlock`，支持 re-parameterization（`convert_to_deploy`）。
- 关键配置：
  - `in_channels`: 与 backbone 输出通道必须一致（例如 `[512,1024,2048]` 对应 ResNet C3/C4/C5）
  - `feat_strides`: `[8,16,32]` 等；与解码器一致
  - `hidden_dim`、`expansion`、`num_encoder_layers`、`nhead`、`dim_feedforward`

### 4.2 解码器 RTDETRTransformerv2（`src/zoo/rtdetr/rtdetrv2_decoder.py`）
- 核心：`MSDeformableAttention`（多尺度可变形注意，按层可设采样点 `num_points`）
  - `cross_attn_method`: `default` 或 `discrete`（TRT 8.4 及以下兼容）
  - `num_points`: 可为单值或列表（如 `[4,4,4]`）
  - 提供 `offset_scale` 与按层缩放，提升数值稳定性与可移植性
- 去噪训练：`get_contrastive_denoising_training_group`（DINO/DN 思想，v2 的 BoF 之一）
- 常用超参：`num_layers`、`num_queries`、`num_denoising`、`label_noise_ratio`、`box_noise_scale`

### 4.3 损失与匹配（`src/zoo/rtdetr/rtdetrv2_criterion.py`）
- 匹配：`HungarianMatcher`（`cost_class/cost_bbox/cost_giou` 可配）
- 分类损失：`sigmoid_focal_loss` 或 `VFL（Varifocal Loss）`
- 回归损失：`L1 + GIoU`，权重在 YAML `weight_dict` 中设定
- 多卡与归一化：内部已处理 `num_boxes` 归一化与分布式一致性

### 4.4 后处理（`src/zoo/rtdetr/rtdetr_postprocessor.py`）
- 选择 top-k 查询，Sigmoid 分类并还原框至原图尺寸；`num_top_queries` 可控

### 4.5 主干（`src/nn/backbone/`）
- 典型：`PResNet`（`presnet.py`），变体 `variant=d`，按 `return_idx` 选取 C3/C4/C5；也有 `hgnetv2`、CSP、timm/torchvision 封装
- 如需接入自定义 MobileNet：保证输出三层（C3/C4/C5）通道与步幅可预知

---

## 五、配置系统与注册机制

- YAML `__include__` 组合：
  - 例：`configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml` 引入 `../dataset/coco_detection.yml`、`../runtime.yml`、`include/rtdetrv2_r50vd.yml`
- 组件装配：
  - `RTDETR` 节指定 `backbone/encoder/decoder` 名称 → 对应 `@register()` 的类
- 关键配置字段（以 `include/rtdetrv2_r50vd.yml` 为例）：
  - `PResNet`：`depth/variant/return_idx/pretrained`
  - `HybridEncoder`：`in_channels/feat_strides/hidden_dim/num_encoder_layers` 等
  - `RTDETRTransformerv2`：`num_layers/num_queries/num_denoising/num_points/cross_attn_method`
  - `RTDETRCriterionv2`：`losses/weight_dict/matcher`
  - `RTDETRPostProcessor`：`num_top_queries`

---

## 六、训练/验证/导出入口

- 训练：`rtdetrv2_pytorch/tools/train.py`
- 导出：`export_onnx.py`、`export_trt.py`
- 性能/FLOPs/Params：`run_profile.py`
- 部署参考：`references/deploy/*.py`（PyTorch/ONNXRuntime/TensorRT/OpenVINO）

---

## 七、扩展点与工程建议

1) 接入 MobileNet（或其他轻量主干）
- 核心约束：
  - 输出 3 个尺度（通常 strides 8/16/32）
  - 在 YAML 中对齐 `HybridEncoder.in_channels` 与 `feat_strides`
- 方式：
  - 在 `src/nn/backbone/` 新增实现，或用 `timm_model.py` 调用 timm 版本
  - 如需 SE/激活不同，确保数值稳定（BN/同步 BN、初始化）

2) 小目标与 TRT 兼容
- `cross_attn_method: discrete` 可避免 `grid_sample` 依赖；配合 `num_points`（如 `[4,4,4]`）
- v2 提供 `*_dsp` 配置与权重对齐策略

3) 收敛/精度增强
- DN（去噪）/DAB（动态 anchor）/DINO（更强的去噪与 query 策略）均可作为参考
- v2 已集成 BoF；在你场景可逐项启停做 ablation

---

## 八、常见问题与排查

- 形状不匹配（most common）：
  - `HybridEncoder.in_channels` 与 backbone 实际输出不一致 → 检查主干通道数
  - `feat_strides` 与实际下采样率不一致 → 严格核对 C3/C4/C5 的步幅
- 类别数错误：
  - 检查 `num_classes` 传递链（数据集、criterion/postprocessor 构造）
- TensorRT 版本问题：
  - 低版本对 `grid_sample` 支持有限 → 使用 `discrete` 采样与对应模型
- 预训练权重加载失败：
  - 核对 `variant`（如 ResNet vd/a/b/d）、`return_idx` 与通道数

---

## 九、最小验证（可选）

```bash
# 进入 v2 目录并安装依赖
cd RT-DETR/rtdetrv2_pytorch
pip install -r requirements.txt

# 统计参数与 FLOPs（确认配置与前向 OK）
python tools/run_profile.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml

# 单卡训练 smoke test（建议先跑 1~2 epoch 验证链路）
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml --use-amp --seed=0
```

---

## 十、与你当前工程的对接提示

- 你已有 `ultralytics` 分支与自研模块（SEA/ASFF 等）。若要切换到 v2：
  - 优先在 `RT-DETR/rtdetrv2_pytorch` 上建立 indoor 数据集配置，跑出新基线
  - 如果要继续使用自研 MobileNetV4 主干：
    - 对齐 `in_channels/feat_strides`；保持 `hidden_dim=256` 不变最稳妥
    - 先做纯替换，不引入额外融合/注意力，得到干净对比
  - 需要我可提供一份最小 MobileNet 适配补丁与 YAML 模板

---

## 参考文件索引（便于你跳转）
- 顶层装配：`rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr.py`
- 编码器：`rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder.py`
- 解码器（v2）：`rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py`
- 损失：`rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py`
- 主干示例：`rtdetrv2_pytorch/src/nn/backbone/presnet.py`
- YAML 示例：`rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml`
- 训练脚本：`rtdetrv2_pytorch/tools/train.py`
