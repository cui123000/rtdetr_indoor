# 推荐架构
1. Backbone: MobileNetV4 + SEA (保持，性能最好)
2. FPN上采样: 替换为 DySample (轻量+快速)
3. 特征融合: 替换为 ASFF (自适应权重)
4. 移除: BiFPN双向路径 (导致性能下降)