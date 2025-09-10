"""
Optimized SEA (Squeeze-enhanced Axial) Attention for RT-DETR Object Detection
Enhan    def forward(self, x):
        # Multi-path squeeze operations
        global_feat = self.global_squeeze(x)  # (B, C, 1, 1)
        local_feat = self.local_squeeze(x)    # (B, C, H, W)
        boundary_feat = self.boundary_enhance(x)  # (B, C, H, W)
        
        # Broadcast global features to match spatial dimensions
        global_feat = global_feat.expand_as(local_feat)  # (B, C, H, W)
        
        # Adaptive fusion
        combined = torch.cat([global_feat, local_feat, boundary_feat], dim=1)
        gate = self.fusion_gate(combined)
        
        return x * gatelti-scale feature extraction and detection-specific optimizations.

Key Improvements:
1. Multi-scale squeeze operations for better object detection
2. Enhanced detail preservation with detection-aware kernels  
3. Adaptive attention scaling based on feature pyramid levels
4. Optimized memory usage for real-time deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class AdaptiveSqueezeAxialPositionalEmbedding(nn.Module):
    """
    Adaptive axial positional embedding optimized for object detection.
    Supports multiple scales and interpolation for different FPN levels.
    """
    def __init__(self, dim, max_shape=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # Multi-scale positional embeddings for different FPN levels
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn([1, dim, max_shape]) * 0.02) 
            for _ in range(num_scales)
        ])
        self.scale_selector = nn.Linear(dim, num_scales)
        
    def forward(self, x, scale_level=None):
        B, C, N = x.shape
        
        if scale_level is not None:
            # Use specified scale level
            pos_embed = self.pos_embeds[scale_level]
        else:
            # Adaptive scale selection
            scale_weights = F.softmax(self.scale_selector(x.mean(-1)), dim=-1)  # B, num_scales
            pos_embed = sum(w.unsqueeze(-1) * embed for w, embed in 
                          zip(scale_weights.unbind(-1), self.pos_embeds))
            pos_embed = pos_embed.mean(0, keepdim=True)  # Average across batch
            
        pos_embed = F.interpolate(pos_embed, size=(N), mode='linear', align_corners=False)
        return x + pos_embed


class DetectionAwareSqueezeModule(nn.Module):
    """
    Detection-aware squeeze module that preserves object boundaries
    and enhances feature discrimination for detection tasks.
    """
    def __init__(self, dim, squeeze_ratio=4):
        super().__init__()
        self.dim = dim
        hidden_dim = max(dim // squeeze_ratio, 8)
        
        # Multi-path squeeze for different semantic levels
        self.global_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
        self.local_squeeze = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
        # Object boundary enhancement
        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1),
            nn.SiLU(),
            nn.Conv2d(dim//4, dim//4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim//4, dim, 1)  # 确保输出通道数是dim
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-path squeeze operations
        global_feat = self.global_squeeze(x)  # (B, C, 1, 1)
        local_feat = self.local_squeeze(x)    # (B, C, H, W)
        boundary_feat = self.boundary_enhance(x)  # (B, C, H, W)
        
        # Broadcast global features to match spatial dimensions
        global_feat = global_feat.expand_as(local_feat)  # (B, C, H, W)
        
        # Adaptive fusion
        combined = torch.cat([global_feat, local_feat, boundary_feat], dim=1)
        gate = self.fusion_gate(combined)
        
        return x * gate


class OptimizedSEA_Attention(nn.Module):
    """
    Optimized Squeeze-enhanced Axial Attention for RT-DETR Object Detection.
    
    Enhancements for detection:
    - Multi-scale aware attention computation
    - Detection-specific detail enhancement
    - Adaptive complexity scaling
    - Memory-efficient implementation
    """
    
    def __init__(self, dim, key_dim=16, num_heads=4, attn_ratio=2, max_shape=64, 
                 detection_mode=True, fpn_level='auto'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        self.detection_mode = detection_mode
        self.fpn_level = fpn_level
        
        self.nh_kd = nh_kd = key_dim * num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        
        # Adaptive parameters based on feature map size
        if dim <= 128:  # Early stage (P3)
            self.complexity_scale = 0.75
            self.detail_enhancement_strength = 1.2
        elif dim <= 256:  # Middle stage (P4)
            self.complexity_scale = 1.0
            self.detail_enhancement_strength = 1.0
        else:  # Late stage (P5)
            self.complexity_scale = 1.25
            self.detail_enhancement_strength = 0.8
            
        # Core projections with adaptive dimensions
        effective_nh_kd = int(nh_kd * self.complexity_scale)
        effective_dh = int(self.dh * self.complexity_scale)
        
        self.to_q = Conv(dim, effective_nh_kd, 1, act=False)
        self.to_k = Conv(dim, effective_nh_kd, 1, act=False)
        self.to_v = Conv(dim, effective_dh, 1, act=False)
        
        # Enhanced positional embeddings
        self.pos_emb_rowq = AdaptiveSqueezeAxialPositionalEmbedding(effective_nh_kd, max_shape)
        self.pos_emb_rowk = AdaptiveSqueezeAxialPositionalEmbedding(effective_nh_kd, max_shape)
        self.pos_emb_colq = AdaptiveSqueezeAxialPositionalEmbedding(effective_nh_kd, max_shape)
        self.pos_emb_colk = AdaptiveSqueezeAxialPositionalEmbedding(effective_nh_kd, max_shape)
        
        # Detection-aware squeeze modules
        self.detection_squeeze = DetectionAwareSqueezeModule(dim)
        
        # Row and column processing
        self.row_processor = nn.Sequential(
            nn.SiLU(),
            Conv(effective_dh, effective_dh, 1, act=False)
        )
        
        self.col_processor = nn.Sequential(
            nn.SiLU(), 
            Conv(effective_dh, effective_dh, 1, act=False)
        )
        
        # Enhanced detail preservation network
        detail_dim = effective_dh + 2 * effective_nh_kd
        self.detail_enhance = nn.Sequential(
            # Multi-scale detail extraction
            Conv(detail_dim, detail_dim, 3, 1, 1, g=detail_dim, act=True),
            # Feature compression and enhancement - 修复通道数
            Conv(detail_dim, dim, 1, act=False)  # 直接从detail_dim到dim
        )
        
        # Multi-scale output projection
        self.output_proj = nn.Sequential(
            Conv(effective_dh, dim, 1, act=False),
            nn.Dropout2d(0.1) if detection_mode else nn.Identity()
        )
        
        # Detection-specific gating
        self.detection_gate = nn.Sequential(
            Conv(dim * 2, dim, 1, act=True),
            Conv(dim, dim, 1, act=False),
            nn.Sigmoid()
        ) if detection_mode else None
        
        # 统一的通道数
        effective_dh = self.dh
        effective_nh_kd = self.nh_kd
        
        # 确保行和列输出通道一致
        channel_unify_dim = min(effective_dh, effective_nh_kd * 2)  # 选择较小的维度
        
        # Adaptive normalization
        self.norm = nn.LayerNorm(dim)
        
        # 通道适配器 - 预先定义避免动态创建
        self.channel_adapter = nn.Conv2d(channel_unify_dim, dim, 1) if channel_unify_dim != dim else nn.Identity()
        
    def _get_scale_level(self, H, W):
        """Determine FPN scale level based on feature map size"""
        if self.fpn_level != 'auto':
            return 0
        
        size = H * W
        if size >= 1600:  # >= 40x40
            return 0  # P3 level
        elif size >= 400:  # >= 20x20  
            return 1  # P4 level
        else:
            return 2  # P5 level
    
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Detection-aware preprocessing
        if self.detection_mode:
            x = self.detection_squeeze(x)
        
        # Generate Q, K, V with adaptive scaling
        q = self.to_q(x)  # B, nh_kd, H, W
        k = self.to_k(x)  # B, nh_kd, H, W
        v = self.to_v(x)  # B, dh, H, W
        
        scale_level = self._get_scale_level(H, W)
        
        # Enhanced detail branch - 修复维度匹配问题
        qkv = torch.cat([q, k, v], dim=1)
        # 直接使用完整的detail_enhance网络
        detail_enhanced = self.detail_enhance(qkv)
        
        # Row attention with enhanced positional encoding
        qrow = self.pos_emb_rowq(q.mean(-1), scale_level)  # B, nh_kd, H
        krow = self.pos_emb_rowk(k.mean(-1), scale_level)  # B, nh_kd, H
        vrow = v.mean(-1)  # B, dh, H
        
        # Reshape for multi-head attention
        qrow = qrow.reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = krow.reshape(B, self.num_heads, -1, H)
        vrow = vrow.reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        
        # Enhanced attention computation with temperature scaling
        temperature = 1.0 + 0.1 * scale_level  # Adaptive temperature
        attn_row = torch.matmul(qrow, krow) * (self.scale / temperature)
        attn_row = F.softmax(attn_row, dim=-1)
        
        # Apply attention with dropout for regularization
        if self.training and self.detection_mode:
            attn_row = F.dropout(attn_row, p=0.1)
            
        out_row = torch.matmul(attn_row, vrow)
        out_row = out_row.permute(0, 1, 3, 2).reshape(B, -1, H, 1)
        out_row = self.row_processor(out_row)
        
        # Column attention with enhanced positional encoding
        qcol = self.pos_emb_colq(q.mean(-2), scale_level)  # B, nh_kd, W
        kcol = self.pos_emb_colk(k.mean(-2), scale_level)  # B, nh_kd, W
        vcol = v.mean(-2)  # B, dh, W
        
        # Reshape for multi-head attention
        qcol = qcol.reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcol = kcol.reshape(B, self.num_heads, -1, W)
        vcol = vcol.reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        # Enhanced attention computation
        attn_col = torch.matmul(qcol, kcol) * (self.scale / temperature)
        attn_col = F.softmax(attn_col, dim=-1)
        
        if self.training and self.detection_mode:
            attn_col = F.dropout(attn_col, p=0.1)
            
        out_col = torch.matmul(attn_col, vcol)
        out_col = out_col.permute(0, 1, 3, 2).reshape(B, -1, 1, W)
        out_col = self.col_processor(out_col)
        
        # Adaptive fusion of row and column attention - 修复维度问题
        # 确保维度匹配
        if out_row.shape != out_col.shape:
            # 将行和列输出扩展到相同维度
            out_row_expanded = out_row.expand(-1, -1, H, W)
            out_col_expanded = out_col.expand(-1, -1, H, W)
        else:
            out_row_expanded = out_row
            out_col_expanded = out_col
            
        fusion_weight = torch.sigmoid(out_row_expanded.mean([2,3], keepdim=True))
        fused_attention = fusion_weight * out_row_expanded + (1 - fusion_weight) * out_col_expanded
        
        # Combine with value features - 简化处理
        # 确保所有张量都是相同的空间尺寸
        if fused_attention.shape[2:] != v.shape[2:]:
            fused_attention = F.interpolate(fused_attention, size=v.shape[2:], mode='bilinear', align_corners=False)
        
        # 通道数对齐
        if fused_attention.size(1) != v.size(1):
            fused_attention = self.channel_adapter(fused_attention)
        
        enhanced_features = v + fused_attention
        
        # Output projection
        output = self.output_proj(enhanced_features)
        
        # Detection-specific gating mechanism
        if self.detection_mode and self.detection_gate is not None:
            gate_input = torch.cat([output, detail_enhanced], dim=1)
            gate = self.detection_gate(gate_input)
            output = output * gate + detail_enhanced * (1 - gate)
        
        # Residual connection with adaptive weighting
        residual_weight = 0.1 if self.detection_mode else 0.2
        output = output + identity * residual_weight
        
        # Layer normalization for stability
        output = output.permute(0, 2, 3, 1)  # B, H, W, C
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # B, C, H, W
        
        return output


class EfficientSEA_Attention_Light(nn.Module):
    """
    超轻量级SEA注意力模块，专为移动端RT-DETR优化
    保持检测精度的同时显著降低计算复杂度
    """
    
    def __init__(self, dim, key_dim=8, num_heads=2, max_shape=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        
        nh_kd = key_dim * num_heads
        
        # 轻量级投影 - 使用分组卷积减少参数
        self.qkv_proj = Conv(dim, nh_kd * 3, 1, act=False)
        
        # 简化的位置编码
        self.pos_h = nn.Parameter(torch.randn(1, nh_kd, max_shape) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, nh_kd, max_shape) * 0.02)
        
        # 轻量级融合模块
        self.fusion = Conv(nh_kd * 2, dim, 1, act=False)
        
        # 检测增强门控
        self.detection_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(dim, dim//4, 1, act=True),
            Conv(dim//4, dim, 1, act=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # 一次性生成 Q, K, V
        qkv = self.qkv_proj(x)  # B, nh_kd*3, H, W
        nh_kd = self.key_dim * self.num_heads
        q, k, v = qkv.chunk(3, dim=1)  # 每个都是 B, nh_kd, H, W
        
        # 行注意力 - 简化版本
        q_h = q.mean(-1)  # B, nh_kd, H
        k_h = k.mean(-1)
        v_h = v.mean(-1)
        
        # 添加位置编码
        pos_h = F.interpolate(self.pos_h, size=H, mode='linear', align_corners=False)
        q_h = q_h + pos_h
        k_h = k_h + pos_h
        
        # 多头注意力 - 行方向
        q_h = q_h.reshape(B, self.num_heads, -1, H).transpose(-1, -2)
        k_h = k_h.reshape(B, self.num_heads, -1, H)
        v_h = v_h.reshape(B, self.num_heads, -1, H).transpose(-1, -2)
        
        attn_h = F.softmax(torch.matmul(q_h, k_h) * self.scale, dim=-1)
        out_h = torch.matmul(attn_h, v_h).transpose(-1, -2).reshape(B, nh_kd, H, 1)
        out_h = out_h.expand(-1, -1, -1, W)
        
        # 列注意力 - 简化版本  
        q_w = q.mean(-2)  # B, nh_kd, W
        k_w = k.mean(-2)
        v_w = v.mean(-2)
        
        # 添加位置编码
        pos_w = F.interpolate(self.pos_w, size=W, mode='linear', align_corners=False)
        q_w = q_w + pos_w
        k_w = k_w + pos_w
        
        # 多头注意力 - 列方向
        q_w = q_w.reshape(B, self.num_heads, -1, W).transpose(-1, -2)
        k_w = k_w.reshape(B, self.num_heads, -1, W)
        v_w = v_w.reshape(B, self.num_heads, -1, W).transpose(-1, -2)
        
        attn_w = F.softmax(torch.matmul(q_w, k_w) * self.scale, dim=-1)
        out_w = torch.matmul(attn_w, v_w).transpose(-1, -2).reshape(B, nh_kd, 1, W)
        out_w = out_w.expand(-1, -1, H, -1)
        
        # 融合行列注意力
        fused = torch.cat([out_h, out_w], dim=1)
        output = self.fusion(fused)
        
        # 检测门控增强
        gate = self.detection_gate(output)
        output = output * gate
        
        # 残差连接
        return identity + output * 0.1


class TransformerEnhancedSEA(nn.Module):
    """
    Transformer增强的SEA模块，优化RT-DETR的特征提取能力
    集成多头自注意力和SEA的优势
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # SEA注意力分支
        self.sea_attention = OptimizedSEA_Attention(
            dim, key_dim=max(16, dim//16), num_heads=min(num_heads, 4),
            detection_mode=True
        )
        
        # Transformer自注意力分支（用于全局特征）
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        
        # FFN模块
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # 分支融合门控
        self.branch_gate = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # SEA分支 - 保持空间结构的轴向注意力
        sea_out = self.sea_attention(x)  # B, C, H, W
        
        # Transformer分支 - 全局自注意力
        # 将特征图转换为序列
        x_seq = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x_seq = self.norm1(x_seq)
        
        # 自注意力
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        x_seq = x_seq + attn_out
        
        # FFN
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        # 转换回特征图格式
        transformer_out = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        # 自适应分支融合
        # 使用全局平均池化来决定分支权重
        global_feat = F.adaptive_avg_pool2d(identity, 1).flatten(1)  # B, C
        branch_weights = self.branch_gate(global_feat)  # B, 2
        
        # 加权融合两个分支
        w_sea, w_transformer = branch_weights.chunk(2, dim=1)
        w_sea = w_sea.unsqueeze(-1).unsqueeze(-1)  # B, 1, 1, 1
        w_transformer = w_transformer.unsqueeze(-1).unsqueeze(-1)
        
        fused_output = w_sea * sea_out + w_transformer * transformer_out
        
        return identity + fused_output * 0.2


# 保持原有的轻量版本以确保兼容性
class SEA_Attention_Light(EfficientSEA_Attention_Light):
    """保持向后兼容性的别名"""
    pass


class Sea_Attention_Simplified(nn.Module):
    """
    简化版SEA注意力 - 针对RT-DETR检测任务优化
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        # 检测导向的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 轴向空间注意力 - 行方向
        self.row_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 轴向空间注意力 - 列方向
        self.col_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # 检测增强模块
        self.detection_enhance = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels//2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 轴向空间注意力
        ra = self.row_attention(x)
        ca_out = self.col_attention(x)
        
        # 组合空间注意力
        spatial_att = ra * ca_out
        x = x * spatial_att
        
        # 检测增强
        det_gate = self.detection_enhance(x)
        x = x * det_gate
        
        # 残差连接
        return identity + x * 0.1


# 工厂函数 - 根据需求自动选择最佳版本
def create_sea_attention(dim, variant='auto', detection_mode=True, **kwargs):
    """
    SEA注意力模块工厂函数
    
    Args:
        dim: 输入维度
        variant: 'auto', 'optimized', 'light', 'transformer', 'simplified'
        detection_mode: 是否启用检测优化
        **kwargs: 其他参数
    """
    if variant == 'auto':
        # 根据维度自动选择最佳版本
        if dim <= 64:
            variant = 'light'
        elif dim <= 256:
            variant = 'optimized'
        else:
            variant = 'transformer'
    
    if variant == 'optimized':
        return OptimizedSEA_Attention(dim, detection_mode=detection_mode, **kwargs)
    elif variant == 'light':
        return EfficientSEA_Attention_Light(dim, **kwargs)
    elif variant == 'transformer':
        return TransformerEnhancedSEA(dim, **kwargs)
    elif variant == 'simplified':
        return Sea_Attention_Simplified(dim, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# 别名以保持兼容性
SEA_Attention = OptimizedSEA_Attention
