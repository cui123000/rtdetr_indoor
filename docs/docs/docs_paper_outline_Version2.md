# Efficient Attention-Augmented MobileNetV4 for Indoor Object Detection (Short Paper Outline)

## 1. Title & Authors
- Title: Efficient Attention-Augmented MobileNetV4 for Real-Time Indoor Object Detection
- Authors, Affiliations, Emails

## 2. Abstract (150–200 words)
Problem: Real-time indoor detection needs better accuracy–efficiency trade-off on mobile-friendly backbones.
Method: We insert a lightweight SEA attention into MobileNetV4 within RT-DETR.
Results: On HomeObjects-3K, our MNV4-SEA improves mAP50 by +14.4% over MNV4 baseline with 29.06M params; shows favorable Pareto vs. popular real-time baselines. Cross-dataset on COCO-Indoor subset validates generalization.
Takeaway: A simple, reusable attention insertion strategy for indoor scenarios; future: CLIP-instruction-driven detection.

## 3. Introduction (0.5 page)
- Motivation: Indoor scenes are cluttered, small objects, occlusions; mobile deployment constraints.
- Gap: Fusion-heavy necks (ASFF/BiFPN) harm RT-DETR balance; lightweight channel/spatial reweighting may help.
- Contributions:
  1) A mobile-friendly attention insertion scheme for MobileNetV4 within RT-DETR.
  2) Extensive ablations isolating structural vs. training-strategy gains.
  3) Efficiency study (Params/FLOPs/Latency/Memory) and cross-dataset generalization to COCO-Indoor.
  4) Reproducible code, configs, and dataset-generation pipeline.

## 4. Related Work (0.3 page)
- Efficient backbones (MobileNetV3/V4, ShuffleNet, ConvNeXt-Tiny)
- Attention in efficient CNNs (SE, ECA, CBAM, SEA variants)
- Real-time detectors (YOLOv8/11, RT-DETR)
- Open-vocabulary (Detic, OWL-ViT) — brief for context and future direction

## 5. Method (0.8 page)
- 5.1 Backbone recap: MobileNetV4 stages and RT-DETR encoder–decoder summary
- 5.2 SEA module: structure, compute cost, where inserted (stage/block indices), ratios
- 5.3 Complexity analysis: Params/FLOPs delta; theoretical vs. measured latency
- 5.4 Design choices: why attention over fusion-necks for RT-DETR; pitfalls of ASFF/BiFPN observed

Figure 1: Architecture diagram of MNV4-SEA within RT-DETR (with insertion points and tensor shapes).

## 6. Experiments (1.5 pages)
Datasets:
- HomeObjects-3K (67 classes, indoor household items)
- COCO-Indoor subset (derived via Places365 I/O filtering; script provided)

Metrics:
- COCO-style mAP50, mAP50-95; APs/APm/APl; Precision/Recall; Latency (GPU/CPU/Edge), Memory

Implementation details:
- Hardware (RTX 4090), Pytorch/Ultralytics versions, imgsz, batch, LR/scheduler, EMA, epochs, seeds
- Reproducibility: config files and training commands; 3 seeds mean±std

Baselines:
- RT-DETR-MNV4 (no SEA), RT-DETR-L/R50, YOLO11n/s, YOLOv8n/s, MobileNetV3 backbone variant (if available)

Main results (HomeObjects-3K):
Table 1: mAP50 / mAP50-95 vs Params / FLOPs / Latency. Bold best. Mark relative gains (+%).

Cross-dataset generalization (COCO-Indoor):
Table 2: Validation mAP50-95 and delta vs. HomeObjects; highlight robustness.

Ablations:
- Table 3: Insertion depth/ratio, with/without EMA, augmentation/loss tweaks — isolate structural gains.
- Table 4: Resolution sweep (512/640/800) and MNV4 scales (T/S/L).

Efficiency:
- Table 5: Inference latency (FP32/FP16 TensorRT; CPU), memory peak; Pareto curve (mAP vs latency/params).

Qualitative:
- Figure 2: PR curves; confusion matrix; small/medium/large AP breakdown
- Figure 3: Visual examples and typical failure cases (clutter/occlusion/long-tail)

## 7. Discussion (0.3 page)
- Why fusion-necks underperform on RT-DETR here; when SEA helps/doesn’t
- Limitations: dataset simplicity, domain coverage, edge-device diversity

## 8. Conclusion & Future Work (0.2 page)
- Summary of improvements and practical impact for indoor detection
- Future: CLIP-instruction-driven detection
  - Minimal pipeline: text prompts → vision-text alignment → zero/low-shot novel indoor categories
  - Planned OVD splits: COCO/LVIS base–novel protocols

## 9. Reproducibility Checklist (bullets)
- Code: repo URL and commit
- Pretrained weights + config YAMLs
- COCO-Indoor generation script + command
- Training/val splits, seeds, hardware
- Logs/CSV and model cards with latency profiles

Appendix (if allowed): More ablations, extra qualitative results, OVD pilot.