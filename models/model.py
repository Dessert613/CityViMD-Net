"""CityViMD-Net 五通道早期融合目标检测模型。"""

import torch
import torch.nn as nn
import yaml

from .backbone import CSPDarknet
from .neck import build_neck
from .head import build_head


class CityViMDNet(nn.Module):
    """
    RGB、红外与深度在输入端直接拼接，使用一个共享 CSPDarknet 骨干。

    输入通道顺序由配置的 modalities 与 in_channels 共同确定，默认是
    RGB(3) + Infrared(1) + Depth(1)。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg['model']
        data_cfg = cfg['data']
        backbone_cfg = model_cfg['backbone']

        self.modalities = data_cfg['modalities']
        self.num_classes = data_cfg['num_classes']
        self.input_channels = [model_cfg['in_channels'][m] for m in self.modalities]
        total_channels = sum(self.input_channels)

        self.backbone = CSPDarknet(
            in_channels=total_channels,
            depth_multiple=backbone_cfg['depth_multiple'],
            width_multiple=backbone_cfg['width_multiple'],
        )
        self.neck = build_neck(cfg, self.backbone.out_channels)
        self.head = build_head(cfg, self.neck.out_channels)
        self._initialize_weights()
        self.head._initialize_biases()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        expected_channels = sum(self.input_channels)
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} early-fusion channels, got {x.shape[1]}"
            )
        features = self.backbone(x)
        predictions = self.head(self.neck(features))
        uniform_weights = x.new_full(
            (x.shape[0], len(self.modalities)), 1.0 / len(self.modalities)
        )
        modality_weights = [uniform_weights for _ in features]
        return predictions, modality_weights

    def predict(self, x, conf_thres=0.25, iou_thres=0.45, max_det=100):
        """返回每张图的 [x1, y1, x2, y2, confidence, class]。"""
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(x)
            boxes, scores = self.head.decode(
                predictions, img_size=(x.shape[2], x.shape[3])
            )
            results = []
            for index in range(x.shape[0]):
                image_boxes = boxes[index]
                max_scores, max_classes = scores[index].max(dim=1)
                valid = max_scores > conf_thres
                image_boxes = image_boxes[valid]
                max_scores = max_scores[valid]
                max_classes = max_classes[valid]

                if len(image_boxes) == 0:
                    results.append(torch.zeros((0, 6), device=x.device))
                    continue

                keep_per_class = []
                for class_id in max_classes.unique():
                    class_indices = torch.where(max_classes == class_id)[0]
                    class_keep = self._nms(
                        image_boxes[class_indices],
                        max_scores[class_indices],
                        iou_thres,
                    )
                    keep_per_class.append(class_indices[class_keep])
                keep = torch.cat(keep_per_class)
                keep = keep[max_scores[keep].argsort(descending=True)][:max_det]
                results.append(torch.cat([
                    image_boxes[keep],
                    max_scores[keep].unsqueeze(1),
                    max_classes[keep].float().unsqueeze(1),
                ], dim=1))
        return results

    def _nms(self, boxes, scores, iou_thres):
        indices = scores.argsort(descending=True)
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())
            if len(indices) == 1:
                break
            iou = self._box_iou(boxes[current:current + 1], boxes[indices[1:]])
            indices = indices[1:][iou[0] <= iou_thres]
        return torch.tensor(keep, device=boxes.device, dtype=torch.long)

    @staticmethod
    def _box_iou(box1, box2):
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        left_top = torch.maximum(box1[:, None, :2], box2[:, :2])
        right_bottom = torch.minimum(box1[:, None, 2:], box2[:, 2:])
        size = (right_bottom - left_top).clamp(min=0)
        intersection = size[:, :, 0] * size[:, :, 1]
        return intersection / (area1[:, None] + area2 - intersection + 1e-7)


def build_model(cfg):
    return CityViMDNet(cfg)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    config = load_config('configs/default.yaml')
    model = build_model(config).eval()
    sample = torch.randn(1, 5, 640, 640)
    with torch.no_grad():
        outputs, _ = model(sample)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print([tuple(output.shape) for output in outputs])
