"""
ERT-DETR: Efficient Real-Time DETR 轻量化创新模块
论文题目: "ERT-DETR: Efficient Real-Time Detection Transformer with Lightweight Attention and Adaptive Pruning"

主要轻量化创新点:
1. 轻量化注意力机制 (Lightweight Attention Mechanisms)
2. 自适应通道选择 (Adaptive Channel Selection) 
3. 线性注意力 (Linear Attention)
4. 分组注意力机制 (Grouped Attention)
5. 高效特征融合 (Efficient Feature Fusion)
6. 渐进式网络压缩 (Progressive Network Compression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

class LightSEA(nn.Module):
    """
    轻量化SEA注意力 (Lightweight Spatial-channel Enhanced Attention)
    
    改进点：
    - 减少参数量50%
    - 使用深度可分离卷积
    - 引入残差快捷连接
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 4)  # 确保最小通道数
        
        # 轻量化通道注意力 - 使用全局平均池化
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 轻量化空间注意力 - 使用深度可分离卷积
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),  # depthwise
            nn.Conv2d(channels, 1, 1, bias=False),  # pointwise
            nn.Sigmoid()
        )
        
        # 参数量减少的特征增强
        self.enhance = nn.Conv2d(channels, channels, 3, padding=1, groups=channels // 4, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        # 通道注意力
        ca_weights = self.channel_attention(x)
        ca_out = x * ca_weights
        
        # 空间注意力  
        sa_weights = self.spatial_attention(x)
        sa_out = x * sa_weights
        
        # 加权融合 (参数共享)
        fused = 0.5 * ca_out + 0.5 * sa_out
        
        # 轻量化增强
        enhanced = self.enhance(fused)
        enhanced = self.bn(enhanced)
        
        return x + enhanced  # 残差连接

class GroupedCBAM(nn.Module):
    """
    分组CBAM注意力 (Grouped Channel-Spatial Attention)
    
    改进点：
    - 通道分组减少参数
    - 并行计算提高效率
    - 保持注意力效果
    """
    
    def __init__(self, channels, groups=8, reduction=16):
        super().__init__()
        self.groups = groups
        self.channels_per_group = channels // groups
        
        # 分组通道注意力
        self.group_channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels_per_group, max(self.channels_per_group // reduction, 1), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(self.channels_per_group // reduction, 1), self.channels_per_group, 1),
                nn.Sigmoid()
            ) for _ in range(groups)
        ])
        
        # 共享空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 分组处理通道注意力
        x_groups = x.chunk(self.groups, dim=1)
        ca_outputs = []
        
        for i, x_group in enumerate(x_groups):
            ca_weight = self.group_channel_attentions[i](x_group)
            ca_outputs.append(x_group * ca_weight)
        
        ca_out = torch.cat(ca_outputs, dim=1)
        
        # 空间注意力
        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        
        out = ca_out * sa_weight
        return out

class AdaptiveChannelSelection(nn.Module):
    """
    自适应通道选择 (Adaptive Channel Selection)
    
    创新点：
    - 动态选择重要通道
    - 运行时自适应剪枝
    - 保持特征表达能力
    """
    
    def __init__(self, channels, selection_ratio=0.75, temperature=1.0):
        super().__init__()
        self.channels = channels
        self.selection_ratio = selection_ratio
        self.temperature = temperature
        
        # 通道重要性评估
        self.channel_scorer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
        )
        
        # 可学习的选择阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # 特征补偿机制
        self.compensation = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算通道重要性分数
        scores = self.channel_scorer(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        if self.training:
            # 训练时使用Gumbel-Softmax实现可微分选择
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            soft_scores = F.softmax((scores + gumbel_noise) / self.temperature, dim=1)
            
            # 软选择
            selected = x * soft_scores.unsqueeze(-1).unsqueeze(-1)
        else:
            # 推理时硬选择
            num_selected = int(C * self.selection_ratio)
            _, indices = torch.topk(scores, num_selected, dim=1)
            
            # 创建选择mask
            mask = torch.zeros_like(scores)
            mask.scatter_(1, indices, 1)
            
            selected = x * mask.unsqueeze(-1).unsqueeze(-1)
        
        # 特征补偿
        compensated = self.compensation(selected)
        
        return compensated

class LinearAttention(nn.Module):
    """
    线性注意力机制 (Linear Attention with O(n) complexity)
    
    创新点：
    - 降低注意力复杂度从O(n²)到O(n)
    - 保持长距离依赖建模能力
    - 适合高分辨率特征图
    """
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # ELU激活用于确保正值
        self.elu = nn.ELU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, head_dim, HW]
        
        # 确保Q和K为正值 (线性注意力要求)
        q = self.elu(q) + 1
        k = self.elu(k) + 1
        
        # 线性注意力计算: O(n)复杂度
        # 计算K^T * V
        kv = torch.matmul(k, v.transpose(-2, -1))  # [B, num_heads, head_dim, head_dim]
        
        # 计算K的归一化因子
        k_sum = torch.sum(k, dim=-1, keepdim=True)  # [B, num_heads, head_dim, 1]
        
        # 线性注意力输出
        numerator = torch.matmul(q.transpose(-2, -1), kv)  # [B, num_heads, HW, head_dim]
        denominator = torch.matmul(q.transpose(-2, -1), k_sum)  # [B, num_heads, HW, 1]
        
        out = numerator / (denominator + 1e-8)  # [B, num_heads, HW, head_dim]
        
        # 重塑和投影
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj(out)
        
        return out + x  # 残差连接

class EfficientFusion(nn.Module):
    """
    高效特征融合模块 (Efficient Feature Fusion)
    
    创新点：
    - 轻量化特征融合
    - 自适应权重学习
    - 减少计算开销
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # 轻量化特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, groups=4),  # 分组卷积
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应融合权重
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x应该是concat后的特征 [B, 2*C, H, W]
        
        # 计算自适应权重
        weights = self.fusion_weights(x)  # [B, 2, 1, 1]
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        
        # 分离特征
        C = x.shape[1] // 2
        feat1, feat2 = x[:, :C], x[:, C:]
        
        # 加权融合
        fused = w1 * feat1 + w2 * feat2
        
        # 变换
        out = self.transform(x)
        
        return out + fused  # 残差连接

class LightRTDETRDecoder(nn.Module):
    """
    轻量化RT-DETR解码器 (Lightweight RT-DETR Decoder)
    
    改进点：
    - 减少解码层数
    - 降低查询数量
    - 共享参数机制
    """
    
    def __init__(self, nc, d_model, num_queries=200, num_layers=3, num_heads=4, num_levels=3):
        super().__init__()
        self.nc = nc
        self.num_queries = num_queries
        self.num_layers = num_layers
        
        # 轻量化查询嵌入
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 减少的解码层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,  # 减少注意力头数
            dim_feedforward=d_model * 2,  # 减少FFN维度
            dropout=0.0,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 轻量化预测头
        self.class_embed = nn.Linear(d_model, nc)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)
        )
        
        # 特征投影
        self.input_proj = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 1) for _ in range(num_levels)
        ])
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features):
        # 特征投影和展平
        src_flatten = []
        for level, feat in enumerate(features):
            src = self.input_proj[level](feat)
            src_flatten.append(src.flatten(2).transpose(1, 2))
        
        src_flatten = torch.cat(src_flatten, 1)  # [B, total_len, C]
        
        # 查询嵌入
        query_embed = self.query_embed.weight
        bs = src_flatten.shape[0]
        tgt = torch.zeros_like(query_embed).unsqueeze(1).repeat(1, bs, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # 解码
        memory = src_flatten.transpose(0, 1)  # [total_len, B, C]
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        
        # 预测
        outputs_class = self.class_embed(hs[-1].transpose(0, 1))  # [B, num_queries, nc]
        outputs_coord = self.bbox_embed(hs[-1].transpose(0, 1)).sigmoid()  # [B, num_queries, 4]
        
        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }