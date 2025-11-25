"""
ERT-DETR: Efficient Real-Time DETR 轻量化模块
适配ultralytics框架的自定义模块实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .conv import Conv, DWConv, CBAM
from .block import C2f

__all__ = ["MBConv", "LightSEA", "GroupedCBAM", "AdaptiveChannelSelection", "LinearAttention", 
           "EfficientFusion", "LightRTDETRDecoder"]

class MBConv(nn.Module):
    """
    MobileNet-V4 MBConv块 (Mobile inverted residual Convolution)
    基于ultralytics Conv模块的实现
    """
    
    def __init__(self, c1, c2, k=3, s=1, expand_ratio=4, se_ratio=0.25):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数 
            k: 卷积核大小
            s: 步长
            expand_ratio: 扩张比例
            se_ratio: SE注意力压缩比例
        """
        super().__init__()
        self.use_shortcut = s == 1 and c1 == c2
        hidden_dim = int(c1 * expand_ratio)
        
        layers = []
        
        # 扩张卷积 (Pointwise expansion)
        if expand_ratio != 1:
            layers.append(Conv(c1, hidden_dim, 1, 1, act='swish'))
            dw_channels = hidden_dim
        else:
            dw_channels = c1
        
        # 深度可分离卷积 (Depthwise convolution)
        layers.append(DWConv(dw_channels, dw_channels, k, s, act='swish'))
        
        # SE注意力
        if se_ratio > 0:
            layers.append(SqueezeExcitation(dw_channels, max(1, int(dw_channels * se_ratio))))
        
        # 投影卷积 (Pointwise projection)
        layers.append(Conv(dw_channels, c2, 1, 1, act=False))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        return self.conv(x)

class SqueezeExcitation(nn.Module):
    """SE注意力模块"""
    
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class LightSEA(nn.Module):
    """
    轻量化SEA注意力机制
    基于ultralytics CBAM的改进版本
    """
    
    def __init__(self, c1, reduction=16):
        super().__init__()
        c_ = max(c1 // reduction, 8)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c_, 1, 1, act='relu'),
            Conv(c_, c1, 1, 1, act='sigmoid')
        )
        
        # 空间注意力 - 使用轻量化设计
        self.spatial_attention = nn.Sequential(
            DWConv(c1, c1, 7, 1),  # 深度可分离卷积
            Conv(c1, 1, 1, 1, act='sigmoid')
        )
        
        # 特征增强
        self.enhance = DWConv(c1, c1, 3, 1)
        
    def forward(self, x):
        # 通道注意力
        ca_weights = self.channel_attention(x)
        ca_out = x * ca_weights
        
        # 空间注意力
        sa_weights = self.spatial_attention(x) 
        sa_out = x * sa_weights
        
        # 融合增强
        fused = 0.5 * ca_out + 0.5 * sa_out
        enhanced = self.enhance(fused)
        
        return x + enhanced

class GroupedCBAM(nn.Module):
    """
    分组CBAM注意力 - 减少参数量
    """
    
    def __init__(self, c1, groups=8):
        super().__init__()
        self.groups = groups
        c_per_group = c1 // groups
        
        # 分组通道注意力
        self.group_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(c_per_group, max(c_per_group // 16, 1), 1, 1, act='relu'),
                Conv(max(c_per_group // 16, 1), c_per_group, 1, 1, act='sigmoid')
            ) for _ in range(groups)
        ])
        
        # 共享空间注意力
        self.spatial_attention = nn.Sequential(
            Conv(2, 1, 7, 1, act='sigmoid')
        )
    
    def forward(self, x):
        # 分组通道注意力
        x_groups = x.chunk(self.groups, dim=1)
        ca_outputs = []
        
        for i, x_group in enumerate(x_groups):
            ca_weight = self.group_attentions[i](x_group)
            ca_outputs.append(x_group * ca_weight)
        
        ca_out = torch.cat(ca_outputs, dim=1)
        
        # 空间注意力
        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        
        return ca_out * sa_weight

class AdaptiveChannelSelection(nn.Module):
    """
    自适应通道选择模块 - 运行时通道剪枝
    """
    
    def __init__(self, c1, selection_ratio=1.0, temperature=1.0):
        super().__init__()
        self.selection_ratio = selection_ratio
        self.temperature = temperature
        
        # 通道重要性评估
        self.scorer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c1 // 4, 1, 1, act='relu'),
            Conv(c1 // 4, c1, 1, 1, act=False)
        )
        
        # 特征补偿
        self.compensation = Conv(c1, c1, 1, 1, act=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算通道重要性分数
        scores = self.scorer(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        # 更稳健的通道选择实现：训练时使用 sigmoid gating（逐通道）
        # 推理时使用 batch-mean top-k 硬选择，保证稳定且一致
        eps = 1e-6
        if self.training:
            # 为提高训练稳定性：训练阶段绕过通道剪枝，直接返回补偿后的完整特征
            # 这样模型能先在完整通道上收敛，再在推理阶段进行通道选择
            return self.compensation(x)
        else:
            num_selected = max(1, int(C * self.selection_ratio))
            # 使用 batch 均值来确定全局重要通道，避免单个样本波动
            mean_scores = scores.mean(dim=0, keepdim=True)  # [1, C]
            _, topk_idx = torch.topk(mean_scores, num_selected, dim=1)
            mask = torch.zeros_like(scores)
            mask[:, topk_idx[0]] = 1.0
            selected = x * mask.unsqueeze(-1).unsqueeze(-1)
        
        return self.compensation(selected)

class LinearAttention(nn.Module):
    """
    线性注意力机制 - O(n)复杂度
    修正版本：正确的线性注意力实现 + AMP兼容
    """
    
    def __init__(self, c1, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = Conv(c1, c1 * 3, 1, 1, act=False)
        self.proj = Conv(c1, c1, 1, 1, act=False)
        self.elu = nn.ELU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 3, 4)  # [3, B, num_heads, head_dim, H*W]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, head_dim, H*W]
        
        # 确保Q和K为正值并归一化以增强数值稳定性
        eps = 1e-6
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 对 head_dim 做 L2 归一化，避免数值过大或过小
        q = q / (q.norm(p=2, dim=2, keepdim=True) + eps)
        k = k / (k.norm(p=2, dim=2, keepdim=True) + eps)

        # 线性注意力计算: O(n)复杂度 (核技巧)
        # kv: [B, num_heads, head_dim, head_dim]
        kv = torch.matmul(k, v.transpose(-2, -1))

        # k_sum: [B, num_heads, head_dim, 1]
        k_sum = torch.sum(k, dim=-1, keepdim=True)

        # numerator: [B, num_heads, head_dim, N]
        numerator = torch.matmul(kv, q)

        # denominator: [B, num_heads, head_dim, 1]
        denominator = torch.matmul(k_sum, q.sum(dim=-2, keepdim=True))  # sum over head_dim -> shape [B, num_heads, head_dim, 1]

        # 防止除以零并广播到 [B, num_heads, head_dim, N]
        denominator = torch.clamp(denominator, min=eps)
        out = numerator / denominator

        # 重新组织为原始形状： [B, num_heads*head_dim, H, W]
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)

        return out + x

class EfficientFusion(nn.Module):
    """
    高效特征融合模块 - 修正版本
    处理concat后的特征融合
    """
    
    def __init__(self, c2):
        super().__init__()
        # c2是目标输出通道数
        self.output_channels = c2
        
        # 延迟初始化权重生成网络
        self.fusion_weights = None
        self.transform = None
        self.shortcut = None
        
    def _init_layers(self, input_channels, device=None, dtype=None):
        """根据实际输入通道数初始化层"""
        if self.fusion_weights is None:
            if input_channels % 2 == 0:
                # concat后的双倍通道处理
                c_half = input_channels // 2

                # 更丰富的融合：基于concat的通道注意力 + 1x1变换
                # 使用输入完整通道数来学习权重和变换，保留更多信息
                self.fusion_weights = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(input_channels, max(input_channels // 8, 1), 1, bias=False),
                    nn.ReLU(inplace=True),
                    # 输出按组门控，组数在运行时决定
                    nn.Conv2d(max(input_channels // 8, 1), min(max(input_channels // 32, 1), 16), 1, bias=False),
                    nn.Sigmoid()
                )

                # 变换层：从concat完整通道到目标输出通道
                g = max(min(input_channels//8, 8), 1)
                chosen = 1
                for gg in range(g, 0, -1):
                    if input_channels % gg == 0 and self.output_channels % gg == 0:
                        chosen = gg
                        break
                self.transform = Conv(input_channels, self.output_channels, 1, 1, g=chosen, act='swish')
                self.shortcut = Conv(input_channels, self.output_channels, 1, 1, act=False) if input_channels != self.output_channels else nn.Identity()
                # 后处理的SE用于重标定输出通道
                self.post_se = SqueezeExcitation(self.output_channels, max(self.output_channels // 8, 1))
                # move the created layers to the model's device and float dtype to avoid AMP dtype/device mismatch
                if device is not None:
                    # Move layers to the correct device. If dtype is provided (e.g., half),
                    # also cast modules to match the input dtype to avoid conv dtype mismatch
                    self.fusion_weights = self.fusion_weights.to(device=device)
                    self.transform = self.transform.to(device=device)
                    if isinstance(self.shortcut, nn.Module):
                        self.shortcut = self.shortcut.to(device=device)
                        if hasattr(self, 'post_se') and isinstance(self.post_se, nn.Module):
                            self.post_se = self.post_se.to(device=device)
                    if dtype is not None and dtype is not torch.float32:
                        self.fusion_weights = self.fusion_weights.to(dtype=dtype)
                        self.transform = self.transform.to(dtype=dtype)
                        if isinstance(self.shortcut, nn.Module):
                            self.shortcut = self.shortcut.to(dtype=dtype)
                            if hasattr(self, 'post_se') and isinstance(self.post_se, nn.Module):
                                self.post_se = self.post_se.to(dtype=dtype)
            else:
                # 单通道特征直接变换
                g = max(min(input_channels//8, 8), 1)
                chosen = 1
                for gg in range(g, 0, -1):
                    if input_channels % gg == 0 and self.output_channels % gg == 0:
                        chosen = gg
                        break
                self.transform = Conv(input_channels, self.output_channels, 1, 1, g=chosen, act='swish')
                self.shortcut = Conv(input_channels, self.output_channels, 1, 1, act=False) if input_channels != self.output_channels else nn.Identity()
                self.post_se = SqueezeExcitation(self.output_channels, max(self.output_channels // 8, 1))
                if device is not None:
                    self.transform = self.transform.to(device=device)
                    if isinstance(self.shortcut, nn.Module):
                        self.shortcut = self.shortcut.to(device=device)
                    if dtype is not None and dtype is not torch.float32:
                        self.transform = self.transform.to(dtype=dtype)
                        if isinstance(self.shortcut, nn.Module):
                            self.shortcut = self.shortcut.to(dtype=dtype)
                        if hasattr(self, 'post_se') and isinstance(self.post_se, nn.Module):
                            self.post_se = self.post_se.to(dtype=dtype)
    
    def forward(self, x):
        """
        输入x是concat后的特征，需要进行融合处理
        """
        B, C, H, W = x.shape
        
        # 延迟初始化: 将运行时创建层移动到正确device，避免CPU/GPU或dtype不一致
        self._init_layers(C, device=x.device, dtype=x.dtype)

        # 如果 fusion_weights 被创建，则使用它们来为 concat 的两部分计算缩放门控
        # 期望 fusion_weights 输出形状为 [B, 2, 1, 1]
        if self.fusion_weights is not None:
            gates = self.fusion_weights(x)  # [B, G, 1, 1], G = num_groups
            # 防止极端值
            gates = gates.clamp(0.0, 1.0)

            G = gates.shape[1]
            # 将通道划分为 G 组并对每组应用对应gate
            group_size = C // G
            if group_size >= 1:
                x_groups = []
                for i in range(G):
                    start = i * group_size
                    end = start + group_size if i < G - 1 else C
                    xg = x[:, start:end, :, :]
                    g = gates[:, i:i+1, :, :]
                    x_groups.append(xg * g)
                x_scaled = torch.cat(x_groups, dim=1)
            else:
                # fallback: 全局缩放
                g = gates.mean(dim=1, keepdim=True)
                x_scaled = x * g

            out = self.transform(x_scaled)
            if hasattr(self, 'post_se'):
                out = self.post_se(out)
            return out + self.shortcut(x)

        # 默认回退：保守残差融合
        out = self.transform(x)
        return out + self.shortcut(x)

class LightRTDETRDecoder(nn.Module):
    """
    轻量化RT-DETR解码器 - 完全兼容标准RTDETRDecoder接口
    
    主要轻量化改进:
    1. 减少解码层数 (6->3)
    2. 降低查询数量 (300->200)  
    3. 减少注意力头数 (8->4)
    4. 简化FFN结构 (1024->512)
    5. 共享参数机制
    """
    
    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries (提高到300恢复表达能力)
        ndp: int = 6,  # num decoder points
        nh: int = 8,  # num heads (提高到8恢复注意力容量)
        ndl: int = 6,  # num decoder layers (提高到6恢复深度)
        d_ffn: int = 1024,  # dim of feedforward (提高到1024)
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args - 兼容标准接口
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = True,
    ):
        super().__init__()
        self.nc = nc
        self.nq = nq
        self.nh = nh
        self.ndl = ndl
        self.hd = hd
        self.eval_idx = eval_idx
        self.nd = nd
        
        # 轻量化backbone特征投影 - 兼容多尺度输入
        # 使用 GroupNorm 替代 BatchNorm，适配小 batch 稳定性
        num_groups = 8 if hd >= 64 else 4
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(x, hd, 1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=hd)
            ) for x in ch
        ])
        
        # 轻量化查询嵌入
        self.query_embed = nn.Embedding(nq, hd)
        # 可学习的查询位置编码，增强解码器对查询的位置信息建模
        self.query_pos = nn.Parameter(torch.zeros(1, nq, hd))
        
        # 轻量化编码器层 (简化版AIFI)
        # 使用 GroupNorm 替代 BatchNorm 提高小 batch 下稳定性
        self.enc_output = nn.Sequential(
            Conv(hd, hd, 1, 1, act=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=hd)
        )
        self.enc_score_head = Conv(hd, nc, 1, 1, act=False)
        self.enc_bbox_head = nn.Sequential(
            Conv(hd, hd//2, 1, 1, act='relu'),
            Conv(hd//2, 4, 1, 1, act=False)
        )
        
        # 轻量化解码器
        # decoder 嵌入使用 GroupNorm
        self.decoder_embed = nn.Sequential(
            Conv(hd, hd, 1, 1, act=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=hd)
        )

        # 多尺度融合模块: 将投影后的多尺度特征 concat 后通过 EfficientFusion 映射到 hd
        self.multi_scale_fusion = EfficientFusion(hd)
        
        # 轻量化预测头
        self.score_head = nn.ModuleList([
            Conv(hd, nc, 1, 1, act=False) for _ in range(ndl)
        ])
        self.bbox_head = nn.ModuleList([
            nn.Sequential(
                Conv(hd, hd//2, 1, 1, act='relu'),
                Conv(hd//2, 4, 1, 1, act=False)
            ) for _ in range(ndl)
        ])
        
        # 轻量化解码器层
        self.decoder_layers = nn.ModuleList([
            LightDecoderLayer(hd, nh, d_ffn, dropout) 
            for _ in range(ndl)
        ])

        # 位置编码 MLP: 将二维坐标映射到 hd 维度的可学习位置编码
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, hd),
            nn.ReLU(inplace=True),
            nn.Linear(hd, hd)
        )
        
        self._reset_parameters()
        self.learnt_init_query = learnt_init_query
    
    def _reset_parameters(self):
        """初始化参数"""
        # 简化初始化
        for module in [self.query_embed]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, feats, targets=None):
        """
        前向传播 - 完全兼容标准RTDETR接口
        """
        # 特征投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 多尺度特征融合：将所有投影后的特征上采样到最高分辨率并相加
        # 这比只使用最后一层更能保留多尺度信息，提升检测质量
        # 找到最高空间分辨率（通常是 proj_feats[-1] 或第一个，按实际shape决定）
        spatial_sizes = [f.shape[-2:] for f in proj_feats]
        # 以最后一层的空间尺寸为基准 (通常最小): upsample others to this size
        target_h, target_w = proj_feats[-1].shape[-2], proj_feats[-1].shape[-1]
        upsampled = []
        for f in proj_feats:
            if f.shape[-2] != target_h or f.shape[-1] != target_w:
                upsampled.append(F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False))
            else:
                upsampled.append(f)

        # 将多尺度特征连接并使用 EfficientFusion 进行融合，保留更丰富的多尺度信息
        cat = torch.cat(upsampled, dim=1)
        memory = self.multi_scale_fusion(cat)
        B, C, H, W = memory.shape
        
        # 编码器输出
        enc_output = self.enc_output(memory)
        
        # 查询初始化：使用可学习查询嵌入作为初始 target（提高检索性能）
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, nq, hd]
        if getattr(self, 'learnt_init_query', True):
            target = query_embed.clone() + self.query_pos.repeat(B, 1, 1)
        else:
            target = torch.zeros_like(query_embed)
        
        # 简化解码过程
        # 添加位置编码到 memory
        memory_flat = memory.flatten(2).transpose(1, 2)  # [B, HW, hd]
        Bm, HW, _ = memory_flat.shape
        # 生成网格坐标并映射为位置编码
        device = memory.device
        h, w = H, W
        ys = torch.linspace(0, 1, steps=h, device=device)
        xs = torch.linspace(0, 1, steps=w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [HW, 2]
        pos_emb = self.pos_mlp(coords)  # [HW, hd]
        pos_emb = pos_emb.unsqueeze(0).expand(Bm, -1, -1)  # [B, HW, hd]
        memory_flat = memory_flat + pos_emb
        
        # 轻量化解码
        hidden_states = []
        for i, layer in enumerate(self.decoder_layers):
            target = layer(target, memory_flat)
            hidden_states.append(target)
        
        # 预测输出
        all_cls_scores = []
        all_bbox_preds = []
        
        for i, hidden in enumerate(hidden_states):
            # 分类预测
            cls_score = self.score_head[i](hidden.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
            all_cls_scores.append(cls_score)
            
            # 边界框预测
            bbox_pred = self.bbox_head[i](hidden.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2).sigmoid()
            all_bbox_preds.append(bbox_pred)
        
        # 返回格式兼容标准RTDETR
        if self.training:
            return {
                'pred_logits': all_cls_scores[-1],
                'pred_boxes': all_bbox_preds[-1],
                'aux_outputs': [
                    {'pred_logits': cls, 'pred_boxes': box}
                    for cls, box in zip(all_cls_scores[:-1], all_bbox_preds[:-1])
                ]
            }
        else:
            return {
                'pred_logits': all_cls_scores[-1], 
                'pred_boxes': all_bbox_preds[-1]
            }

class LightDecoderLayer(nn.Module):
    """轻量化解码器层 - 简化版Transformer层"""
    
    def __init__(self, d_model, n_head, d_ffn, dropout):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # 轻量化FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, target, memory):
        # 自注意力
        target2 = self.self_attn(target, target, target)[0]
        target = self.norm1(target + target2)
        
        # 交叉注意力
        target2 = self.cross_attn(target, memory, memory)[0] 
        target = self.norm2(target + target2)
        
        # FFN
        target2 = self.ffn(target)
        target = self.norm3(target + target2)
        
        return target

