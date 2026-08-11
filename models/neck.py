"""
CityViMD-Net Neck 特征金字塔
基于 YOLOv8 PAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Conv, C2f


class Upsample(nn.Module):
    """上采样模块"""
    
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Concat(nn.Module):
    """通道拼接模块"""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class YOLOv8Neck(nn.Module):
    """
    YOLOv8 风格的 Neck (PANet)
    
    输入: [P2, P3, P4, P5]
    输出: [P3, P4, P5] 用于检测
    """
    
    def __init__(self, in_channels, depth_multiple=0.67):
        super().__init__()
        
        c2, c3, c4, c5 = in_channels
        base_depth = max(round(3 * depth_multiple), 1)
        
        # 自顶向下
        self.cv1 = Conv(c5, c4, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f1 = C2f(c4 * 2, c4, n=base_depth, shortcut=False)
        
        self.cv2 = Conv(c4, c3, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f2 = C2f(c3 * 2, c3, n=base_depth, shortcut=False)
        
        # 自底向上
        self.cv3 = Conv(c3, c3, 3, 2)
        self.c2f3 = C2f(c4 + c3, c4, n=base_depth, shortcut=False)
        
        self.cv4 = Conv(c4, c4, 3, 2)
        self.c2f4 = C2f(c5 + c4, c5, n=base_depth, shortcut=False)
        
        self.out_channels = [c3, c4, c5]
    
    def forward(self, features):
        """
        Args:
            features: [P2, P3, P4, P5]
        Returns:
            outputs: [P3_out, P4_out, P5_out]
        """
        p2, p3, p4, p5 = features
        
        # 自顶向下
        p5_up = self.cv1(p5)
        p4_cat = torch.cat([p4, self.upsample1(p5_up)], dim=1)
        p4_up = self.c2f1(p4_cat)
        
        p4_up_2 = self.cv2(p4_up)
        p3_cat = torch.cat([p3, self.upsample2(p4_up_2)], dim=1)
        p3_out = self.c2f2(p3_cat)
        
        # 自底向上
        p3_down = self.cv3(p3_out)
        p4_cat2 = torch.cat([p4_up, p3_down], dim=1)
        p4_out = self.c2f3(p4_cat2)
        
        p4_down = self.cv4(p4_out)
        p5_cat = torch.cat([p5, p4_down], dim=1)
        p5_out = self.c2f4(p5_cat)
        
        return [p3_out, p4_out, p5_out]


def build_neck(cfg, in_channels):
    """构建 Neck"""
    model_cfg = cfg['model']
    neck_cfg = model_cfg['neck']
    
    if neck_cfg['type'] == 'pan':
        neck = YOLOv8Neck(in_channels, model_cfg['backbone']['depth_multiple'])
    else:
        raise ValueError(f"Unsupported neck type: {neck_cfg['type']}")
    
    return neck
