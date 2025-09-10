"""
Sea Attention Module for RT-DETR
基于SeaFormer的原始SEA注意力实现，适配ultralytics框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SqueezeAxialPositionalEmbedding(nn.Module):
    """轴向位置编码"""
    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x


class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm组合模块"""
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, bias=False):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Sea_Attention(nn.Module):
    """
    SEA注意力机制 - SeaFormer原始实现
    
    Args:
        dim: 输入特征通道数
        key_dim: 键的维度
        num_heads: 注意力头数
        attn_ratio: 注意力比例
    """
    def __init__(self, dim, key_dim=32, num_heads=8, attn_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        # Q, K, V 投影层
        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)
        
        # 输出投影
        self.proj = nn.Sequential(
            nn.ReLU6(),
            Conv2d_BN(self.dh, dim, bn_weight_init=0)
        )
        
        # 行编码
        self.proj_encode_row = nn.Sequential(
            nn.ReLU6(),
            Conv2d_BN(self.dh, self.dh, bn_weight_init=0)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        # 列编码
        self.proj_encode_column = nn.Sequential(
            nn.ReLU6(),
            Conv2d_BN(self.dh, self.dh, bn_weight_init=0)
        )
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        # 细节增强
        self.dwconv = Conv2d_BN(
            self.dh + 2 * self.nh_kd, 
            2 * self.nh_kd + self.dh, 
            ks=3, stride=1, pad=1, dilation=1,
            groups=2 * self.nh_kd + self.dh
        )
        self.act = nn.ReLU6()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1)
        self.sigmoid = h_sigmoid()

    def forward(self, x):  
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # 细节增强
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # 挤压轴向注意力
        ## 挤压行
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## 挤压列
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        # 融合行列注意力
        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        
        # 门控机制
        xx = self.sigmoid(xx) * qkv
        return xx


class Sea_Attention_Simplified(nn.Module):
    """
    简化版SEA注意力 - 适用于RT-DETR
    减少参数量和计算复杂度
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        # 简化的通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            h_sigmoid()
        )
        
        # 简化的空间注意力 - 分别处理行和列
        self.row_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # (B, C, H, 1)
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            h_sigmoid()
        )
        
        self.col_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # (B, C, 1, W)
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            h_sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 行注意力
        ra = self.row_attention(x)
        x = x * ra
        
        # 列注意力
        ca = self.col_attention(x)
        x = x * ca
        
        return x


# 为了向后兼容，提供不同的接口
def Sea_Attention_Adaptive(channels, spatial_size=None):
    """自适应SEA注意力 - 根据通道数选择合适的版本"""
    if channels <= 128:
        return Sea_Attention_Simplified(channels, reduction=4)
    else:
        return Sea_Attention_Simplified(channels, reduction=8)


# 注册到模块系统
import sys
current_module = sys.modules[__name__]
setattr(current_module, 'Sea_Attention', Sea_Attention)
setattr(current_module, 'Sea_Attention_Simplified', Sea_Attention_Simplified)
setattr(current_module, 'Sea_Attention_Adaptive', Sea_Attention_Adaptive)
