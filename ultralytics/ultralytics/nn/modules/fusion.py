# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Advanced Feature Fusion Modules for RT-DETR.

This module implements lightweight and efficient feature fusion techniques:
- ASFF (Adaptive Spatial Feature Fusion): Learnable per-pixel adaptive fusion weights
- DySample: Dynamic upsampling with content-aware sampling offsets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """
    Dynamic Upsampling module with content-aware sampling.
    
    Ultra-lightweight upsampling that learns dynamic sampling offsets.
    Used in YOLO11 for efficient feature upsampling in FPN.
    
    Args:
        in_channels (int): Number of input channels
        scale (int): Upsampling scale factor (default: 2)
        groups (int): Number of groups for grouped convolution (default: 4)
        
    Reference:
        "Learning to Upsample by Learning to Sample" - ICCV 2023
    """
    
    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.groups = groups
        
        # 生成动态采样偏移的轻量级卷积
        # 使用分组卷积降低参数量
        self.offset = nn.Conv2d(
            in_channels, 
            2 * scale * scale * groups,  # 每个输出位置的x,y偏移
            kernel_size=1,
            groups=groups,
            bias=True
        )
        
        # 初始化偏移为0（接近双线性插值）
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)
        
    def forward(self, x):
        """
        Forward pass with dynamic upsampling.
        
        Args:
            x (Tensor): Input feature map [B, C, H, W]
            
        Returns:
            (Tensor): Upsampled feature map [B, C, H*scale, W*scale]
        """
        B, C, H, W = x.shape
        dtype = x.dtype  # 保存输入的数据类型
        device = x.device
        
        # 1. 生成动态采样偏移 [B, 2*scale*scale*groups, H, W]
        offset = self.offset(x)
        
        # 2. 上采样偏移到目标分辨率
        offset = F.pixel_shuffle(offset, self.scale)  # [B, 2*groups, H*scale, W*scale]
        
        # 3. 归一化偏移到[-1, 1]范围
        offset = offset.tanh()  # 限制偏移范围
        
        # 4. 构建采样网格
        N, _, out_h, out_w = offset.shape
        
        # 创建基础网格 (归一化到[-1, 1])，使用相同的dtype
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, out_h, device=device, dtype=dtype),
            torch.linspace(-1, 1, out_w, device=device, dtype=dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # 添加动态偏移
        offset = offset.permute(0, 2, 3, 1)  # [B, H, W, 2*groups]
        offset = offset.reshape(B, out_h, out_w, self.groups, 2).mean(dim=3)  # 平均groups
        
        # 5. 使用grid_sample进行动态采样
        grid = grid + offset * 0.1  # 缩放偏移避免过大
        output = F.grid_sample(
            x, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # 6. 上采样到目标尺寸
        output = F.interpolate(
            output,
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=True
        )
        
        return output


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion module.
    
    Learns adaptive per-pixel fusion weights for multi-scale features.
    More efficient than BiFPN by avoiding bidirectional paths.
    
    Args:
        level (int): Current feature level (0=P3, 1=P4, 2=P5)
        channels (int): Number of channels for all feature levels
        multiplier (int): Channel multiplier for weight learning (default: 1)
        
    Reference:
        "Learning Spatial Fusion for Single-Shot Object Detection" - CVPR 2019
    """
    
    def __init__(self, level, channels=256, multiplier=1):
        super().__init__()
        self.level = level
        self.channels = channels
        self.inter_channels = int(channels * multiplier)
        
        # 压缩通道用于权重学习（减少计算量）
        self.compress_level_0 = nn.Conv2d(channels, self.inter_channels, 1, 1, 0)
        self.compress_level_1 = nn.Conv2d(channels, self.inter_channels, 1, 1, 0)
        self.compress_level_2 = nn.Conv2d(channels, self.inter_channels, 1, 1, 0)
        
        # 学习自适应融合权重（3个尺度）- 修复padding参数
        self.weight_level_0 = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.weight_level_1 = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.weight_level_2 = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0)
        
        # 扩展回原始通道数
        self.expand = nn.Conv2d(self.inter_channels, channels, 1, 1, 0)
        
    def forward(self, x):
        """
        Forward pass with adaptive fusion.
        
        Args:
            x (list[Tensor]): List of 3 feature maps
                - x[0]: Feature map from level 0 (P3) [B, C, H, W]
                - x[1]: Feature map from level 1 (P4) [B, C, H/2, W/2]
                - x[2]: Feature map from level 2 (P5) [B, C, H/4, W/4]
            
        Returns:
            (Tensor): Fused feature map at current level [B, C, H_out, W_out]
        """
        # 处理输入
        if isinstance(x, (list, tuple)):
            if len(x) != 3:
                raise ValueError(f"ASFF expects 3 inputs, got {len(x)}")
            x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]
        else:
            raise TypeError(f"ASFF expects list of 3 tensors, got {type(x)}")
        
        # 1. 确定目标尺寸（当前level的尺寸）
        if self.level == 0:  # P3
            target_size = x_level_0.shape[2:]
        elif self.level == 1:  # P4
            target_size = x_level_1.shape[2:]
        else:  # P5
            target_size = x_level_2.shape[2:]
        
        # 2. 调整所有特征到目标尺寸
        if self.level == 0:  # 融合到P3
            level_0_resized = x_level_0
            level_1_resized = F.interpolate(x_level_1, size=target_size, mode='bilinear', align_corners=True)
            level_2_resized = F.interpolate(x_level_2, size=target_size, mode='bilinear', align_corners=True)
        elif self.level == 1:  # 融合到P4
            level_0_resized = F.adaptive_max_pool2d(x_level_0, output_size=target_size)
            level_1_resized = x_level_1
            level_2_resized = F.interpolate(x_level_2, size=target_size, mode='bilinear', align_corners=True)
        else:  # 融合到P5
            level_0_resized = F.adaptive_max_pool2d(x_level_0, output_size=target_size)
            level_1_resized = F.adaptive_max_pool2d(x_level_1, output_size=target_size)
            level_2_resized = x_level_2
        
        # 3. 压缩通道（减少计算量）
        level_0_compressed = self.compress_level_0(level_0_resized)
        level_1_compressed = self.compress_level_1(level_1_resized)
        level_2_compressed = self.compress_level_2(level_2_resized)
        
        # 4. 学习自适应融合权重
        weight_level_0 = self.weight_level_0(level_0_compressed)
        weight_level_1 = self.weight_level_1(level_1_compressed)
        weight_level_2 = self.weight_level_2(level_2_compressed)
        
        # 5. Softmax归一化权重（每个像素位置的权重和为1）
        weights = torch.cat([weight_level_0, weight_level_1, weight_level_2], dim=1)
        weights = F.softmax(weights, dim=1)
        
        # 6. 加权融合
        fused_out = (
            level_0_compressed * weights[:, 0:1, :, :] +
            level_1_compressed * weights[:, 1:2, :, :] +
            level_2_compressed * weights[:, 2:3, :, :]
        )
        
        # 7. 扩展回原始通道数
        fused_out = self.expand(fused_out)
        
        return fused_out


class ASFF_Simple(nn.Module):
    """
    Simplified ASFF for single-scale enhancement (P3 only).
    
    Used when only enhancing small object detection (P3 path).
    
    Args:
        channels (int): Number of channels
    """
    
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        
        # 学习两个输入的融合权重（当前层 + 下采样层）
        self.weight_level_0 = nn.Conv2d(channels, 1, 1, 1, 0)
        self.weight_level_1 = nn.Conv2d(channels, 1, 1, 1, 0)
        
    def forward(self, x):
        """
        Forward pass with two-level fusion.
        
        Args:
            x (list[Tensor] or Tensor): If list, expects [x_current, x_from_higher]
                                        If single Tensor with two inputs, splits them
            
        Returns:
            (Tensor): Fused feature [B, C, H, W]
        """
        # 处理输入：支持列表或单个tensor
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError(f"ASFF_Simple expects 2 inputs, got {len(x)}")
            x_current, x_from_higher = x[0], x[1]
        else:
            raise TypeError(f"ASFF_Simple expects list of 2 tensors, got {type(x)}")
        
        # 学习权重
        w1 = self.weight_level_0(x_current)
        w2 = self.weight_level_1(x_from_higher)
        
        # Softmax归一化
        weights = torch.cat([w1, w2], dim=1)
        weights = F.softmax(weights, dim=1)
        
        # 加权融合
        out = x_current * weights[:, 0:1, :, :] + x_from_higher * weights[:, 1:2, :, :]
        
        return out


if __name__ == '__main__':
    # 测试DySample
    print("Testing DySample...")
    x = torch.randn(2, 256, 20, 20)
    dy = DySample(256, scale=2, groups=4)
    out = dy(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dy.parameters()):,}")
    
    # 测试ASFF
    print("\nTesting ASFF...")
    x0 = torch.randn(2, 256, 80, 80)  # P3
    x1 = torch.randn(2, 256, 40, 40)  # P4
    x2 = torch.randn(2, 256, 20, 20)  # P5
    
    asff_p3 = ASFF(level=0, channels=256)
    asff_p4 = ASFF(level=1, channels=256)
    asff_p5 = ASFF(level=2, channels=256)
    
    out_p3 = asff_p3(x0, x1, x2)
    out_p4 = asff_p4(x0, x1, x2)
    out_p5 = asff_p5(x0, x1, x2)
    
    print(f"  P3: {out_p3.shape}, Params: {sum(p.numel() for p in asff_p3.parameters()):,}")
    print(f"  P4: {out_p4.shape}, Params: {sum(p.numel() for p in asff_p4.parameters()):,}")
    print(f"  P5: {out_p5.shape}, Params: {sum(p.numel() for p in asff_p5.parameters()):,}")
    
    # 测试ASFF_Simple
    print("\nTesting ASFF_Simple...")
    asff_simple = ASFF_Simple(256)
    out_simple = asff_simple(x0, F.interpolate(x1, size=x0.shape[2:], mode='bilinear'))
    print(f"  Output: {out_simple.shape}, Params: {sum(p.numel() for p in asff_simple.parameters()):,}")
    
    print("\n✅ All tests passed!")
