"""
CityViMD-Net 骨干网络
基于 YOLOv8 的 CSPDarknet 改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    """自动填充"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块: Conv + BN + SiLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            autopad(kernel_size, padding, dilation), groups=groups, 
            dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """融合卷积和BN用于推理加速"""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """标准瓶颈模块"""
    
    def __init__(self, in_channels, out_channels, shortcut=True, 
                 groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f 模块 (YOLOv8)
    CSP Bottleneck with 2 convolutions and cross-stage partial connections
    """
    
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, 
                 groups=1, expansion=0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv((2 + n) * self.hidden_channels, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, 
                      groups, expansion=1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    SPPF 模块 (Spatial Pyramid Pooling - Fast)
    空间金字塔池化，YOLOv8版本
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, 
                              padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet 骨干网络 (YOLOv8 风格)
    
    输出特征层级:
        - P2: stride=4,  channels=64*width
        - P3: stride=8,  channels=128*width
        - P4: stride=16, channels=256*width
        - P5: stride=32, channels=512*width
    """
    
    def __init__(self, in_channels=3, depth_multiple=0.67, width_multiple=0.75):
        super().__init__()
        
        # 基础通道数
        base_channels = int(64 * width_multiple)
        base_depth = max(round(3 * depth_multiple), 1)
        
        # Stem
        self.stem = Conv(in_channels, base_channels, 3, 2)  # P1/2
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),  # P2/4
            C2f(base_channels * 2, base_channels * 2, n=base_depth),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),  # P3/8
            C2f(base_channels * 4, base_channels * 4, n=base_depth * 2),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),  # P4/16
            C2f(base_channels * 8, base_channels * 8, n=base_depth * 2),
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),  # P5/32
            C2f(base_channels * 16, base_channels * 16, n=base_depth),
            SPPF(base_channels * 16, base_channels * 16, 5),
        )
        
        # 输出通道数
        self.out_channels = [
            base_channels * 2,   # P2
            base_channels * 4,   # P3
            base_channels * 8,   # P4
            base_channels * 16,  # P5
        ]
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            features: list of feature maps [P2, P3, P4, P5]
        """
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        
        return [p2, p3, p4, p5]


if __name__ == '__main__':
    sample = torch.randn(1, 5, 640, 640)
    backbone = CSPDarknet(in_channels=5)
    features = backbone(sample)
    for i, f in enumerate(features):
        print(f"  P{i+2}: {f.shape}")
