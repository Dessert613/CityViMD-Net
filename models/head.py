"""
CityViMD-Net 检测头
基于 YOLOv8 解耦检测头 + DFL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import Conv


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL)
    将离散分布转换为连续坐标
    """
    
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, reg_max, reg_max + 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 4*(reg_max+1), H, W]
        Returns:
            [B, 4, H, W]
        """
        B, C, H, W = x.shape
        x = x.view(B, 4, self.reg_max + 1, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, 4, H, W, reg_max+1]
        x = F.softmax(x, dim=-1)
        x = F.linear(x, self.project.type_as(x))  # [B, 4, H, W]
        return x


class YOLOv8Head(nn.Module):
    """
    YOLOv8 解耦检测头
    
    包含:
    - 分类分支 (cls)
    - 回归分支 (reg) - 使用 DFL
    """
    
    def __init__(self, num_classes=12, in_channels=None, reg_max=16):
        """
        Args:
            num_classes: 类别数
            in_channels: 输入通道数列表 [c3, c4, c5]
            reg_max: DFL 回归范围
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + 4 * (reg_max + 1)
        self.num_layers = len(in_channels)  # 检测层数
        
        # 分类分支
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        
        # 回归分支
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # 为每个检测层构建头部
        for ch in in_channels:
            # 分类分支
            self.cls_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1),
            ))
            self.cls_preds.append(
                nn.Conv2d(ch, num_classes, 1)
            )
            
            # 回归分支
            self.reg_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1),
            ))
            self.reg_preds.append(
                nn.Conv2d(ch, 4 * (reg_max + 1), 1)
            )
        
        # DFL 模块
        self.dfl = DFL(reg_max)
        
        # 初始化偏置
        self._initialize_biases()
    
    def _initialize_biases(self):
        """初始化偏置，参考 YOLOv8"""
        for cls_pred, reg_pred in zip(self.cls_preds, self.reg_preds):
            nn.init.constant_(cls_pred.bias, -4.5)
            nn.init.constant_(reg_pred.bias, 1.0)
    
    def forward(self, features):
        """
        Args:
            features: list of feature maps [P3, P4, P5]
        Returns:
            predictions: list of predictions for each layer
                each: [B, num_outputs, H, W]
        """
        predictions = []
        
        for i, feat in enumerate(features):
            # 分类分支
            cls_feat = self.cls_convs[i](feat)
            cls_pred = self.cls_preds[i](cls_feat)
            
            # 回归分支
            reg_feat = self.reg_convs[i](feat)
            reg_pred = self.reg_preds[i](reg_feat)
            
            # 拼接
            pred = torch.cat([cls_pred, reg_pred], dim=1)
            predictions.append(pred)
        
        return predictions
    
    def decode(self, predictions, img_size=(640, 640)):
        """
        将预测解码为检测框
        
        Args:
            predictions: list of predictions [P3, P4, P5]
            img_size: 图像尺寸 (h, w)
        Returns:
            boxes: [B, N, 4]  xyxy format
            scores: [B, N, num_classes]
        """
        all_boxes = []
        all_scores = []
        
        for i, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            
            # 分离分类和回归
            cls_pred = pred[:, :self.num_classes, :, :]  # [B, nc, H, W]
            reg_pred = pred[:, self.num_classes:, :, :]  # [B, 4*(reg_max+1), H, W]
            
            # DFL 解码
            reg_decoded = self.dfl(reg_pred)  # [B, 4, H, W]
            
            # 生成以特征格为单位的锚点坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred.device, dtype=pred.dtype),
                torch.arange(W, device=pred.device, dtype=pred.dtype),
                indexing='ij'
            )
            anchor = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1)
            anchor = anchor.unsqueeze(0)  # [1, H, W, 2]
            
            stride_y = img_size[0] / H
            stride_x = img_size[1] / W
            
            # DFL 的四个分布分别表示锚点到框四边的 l/t/r/b 距离
            reg_decoded = reg_decoded.permute(0, 2, 3, 1)  # [B, H, W, 4]
            x1 = (anchor[..., 0] - reg_decoded[..., 0]) * stride_x
            y1 = (anchor[..., 1] - reg_decoded[..., 1]) * stride_y
            x2 = (anchor[..., 0] + reg_decoded[..., 2]) * stride_x
            y2 = (anchor[..., 1] + reg_decoded[..., 3]) * stride_y
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            boxes[..., 0::2].clamp_(0, img_size[1])
            boxes[..., 1::2].clamp_(0, img_size[0])
            
            # 分类得分
            scores = cls_pred.permute(0, 2, 3, 1).sigmoid()  # [B, H, W, nc]
            
            # reshape
            boxes = boxes.view(B, -1, 4)
            scores = scores.view(B, -1, self.num_classes)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        # 拼接所有检测层
        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        
        return all_boxes, all_scores


def build_head(cfg, in_channels):
    """构建检测头"""
    model_cfg = cfg['model']
    head_cfg = model_cfg['head']
    data_cfg = cfg['data']
    if head_cfg.get('type', 'yolov8') != 'yolov8':
        raise ValueError(f"Unsupported detection head: {head_cfg['type']}")
    
    head = YOLOv8Head(
        num_classes=data_cfg['num_classes'],
        in_channels=in_channels,
        reg_max=head_cfg['reg_max']
    )
    
    return head


if __name__ == '__main__':
    # 测试检测头
    batch_size = 2
    num_classes = 12
    in_channels = [128, 256, 512]
    sizes = [80, 40, 20]
    
    features = [
        torch.randn(batch_size, c, s, s)
        for c, s in zip(in_channels, sizes)
    ]
    
    head = YOLOv8Head(num_classes=num_classes, in_channels=in_channels, reg_max=16)
    
    print("Input shapes:")
    for i, f in enumerate(features):
        print(f"  P{i+3}: {f.shape}")
    
    predictions = head(features)
    
    print("\nOutput shapes:")
    for i, p in enumerate(predictions):
        print(f"  P{i+3}: {p.shape}")
    
    # 解码测试
    boxes, scores = head.decode(predictions, img_size=(640, 640))
    print(f"\nDecoded boxes: {boxes.shape}")
    print(f"Decoded scores: {scores.shape}")
    
    params = sum(p.numel() for p in head.parameters())
    print(f"\nHead Params: {params / 1e6:.2f}M")
