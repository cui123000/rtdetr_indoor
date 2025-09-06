# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MobileNetV4 modules for Ultralytics models."""

import torch
import torch.nn as nn
from .conv import Conv, DWConv


__all__ = ("MobileViTBlock", "EdgeResidual", "UniversalInvertedResidual")


class EdgeResidual(nn.Module):
    """EdgeResidual block from MobileNetV4."""
    
    def __init__(self, c1, c2, s=1, e=4):
        """Initialize EdgeResidual block.
        
        Args:
            c1: Input channels
            c2: Output channels  
            s: Stride
            e: Expansion ratio
        """
        super().__init__()
        self.stride = s
        self.use_residual = s == 1 and c1 == c2
        
        # Expansion convolution
        c_expand = int(c1 * e)
        self.conv_expand = Conv(c1, c_expand, 1) if e != 1 else nn.Identity()
        
        # Depthwise convolution
        self.conv_dw = DWConv(c_expand, c_expand, 3, s)
        
        # Point-wise projection
        self.conv_project = nn.Conv2d(c_expand, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        """Forward pass through EdgeResidual block."""
        identity = x
        
        if isinstance(self.conv_expand, Conv):
            x = self.conv_expand(x)
        
        x = self.conv_dw(x)
        x = self.conv_project(x)
        x = self.bn(x)
        
        if self.use_residual:
            x = x + identity
            
        return x


class UniversalInvertedResidual(nn.Module):
    """Universal Inverted Residual block from MobileNetV4."""
    
    def __init__(self, c1, c2, s=1, e=4, k=3):
        """Initialize UniversalInvertedResidual block.
        
        Args:
            c1: Input channels
            c2: Output channels
            s: Stride
            e: Expansion ratio
            k: Kernel size
        """
        super().__init__()
        self.stride = s
        self.use_residual = s == 1 and c1 == c2
        
        # Expansion
        c_expand = int(c1 * e)
        self.conv1 = Conv(c1, c_expand, 1) if e != 1 else nn.Identity()
        
        # Depthwise
        self.conv2 = DWConv(c_expand, c_expand, k, s)
        
        # Squeeze-and-Excite (optional)
        self.se = nn.Identity()  # Simplified for now
        
        # Point-wise projection
        self.conv3 = nn.Conv2d(c_expand, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        """Forward pass through UniversalInvertedResidual block."""
        identity = x
        
        if isinstance(self.conv1, Conv):
            x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.bn(x)
        
        if self.use_residual:
            x = x + identity
            
        return x


class MobileViTBlock(nn.Module):
    """MobileViT block combining convolution and vision transformer."""
    
    def __init__(self, c1, c2, s=1):
        """Initialize MobileViT block.
        
        Args:
            c1: Input channels
            c2: Output channels
            s: Stride
        """
        super().__init__()
        self.conv1 = Conv(c1, c1, 3, s)
        self.conv2 = Conv(c1, c2, 1)
        # Simplified version - can be expanded with actual ViT blocks

    def forward(self, x):
        """Forward pass through MobileViT block."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x
