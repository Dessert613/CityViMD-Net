"""
CityViMD-Net 评估指标
mAP@50-95 计算
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou(box1, box2):
    """
    计算两组框的 IoU
    
    Args:
        box1: [N, 4] xyxy
        box2: [M, 4] xyxy
    Returns:
        iou: [N, M]
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    
    area1 = box_area(box1)
    area2 = box_area(box2)
    
    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.maximum(union, np.finfo(np.float64).eps)
    return iou


def compute_ap(recall, precision):
    """
    计算 Average Precision (101点插值法)
    
    Args:
        recall: [N] 召回率
        precision: [N] 精确率
    Returns:
        ap: Average Precision
        recall_interp: 插值后的召回率
        precision_interp: 插值后的精确率
    """
    # 赛事文档定义：对每个召回率采样点，取 recall >= r 的最大 precision
    recall_points = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_points)
    for index, recall_point in enumerate(recall_points):
        valid = recall >= recall_point
        precision_interp[index] = precision[valid].max() if valid.any() else 0.0
    
    ap = np.mean(precision_interp)
    
    return ap, recall_points, precision_interp


def compute_map(predictions, targets, num_classes=12, iou_thresholds=None):
    """
    计算 mAP@50-95

    协议与 pycocotools（COCO 官方实现）对齐：贪心匹配在未占用的 GT 中取
    最大 IoU，101 点插值，无 GT 的类别不参与平均。
    一致性由 tests/test_metrics_pycoco_parity.py 保证。
    
    Args:
        predictions: list of predictions for each image
            each: [N, 6] (x1, y1, x2, y2, conf, cls)
        targets: list of targets for each image
            each: [M, 5] (cls, x1, y1, x2, y2)
        num_classes: 类别数
        iou_thresholds: IoU 阈值列表
    Returns:
        results: dict with mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    else:
        iou_thresholds = np.round(np.asarray(iou_thresholds), 2)
    
    num_images = len(predictions)
    
    # 按类别收集所有预测和真实框
    class_preds = defaultdict(list)  # {cls: [(conf, image_idx, box)]}
    class_gts = defaultdict(list)    # {cls: [(image_idx, box)]}
    
    for img_idx in range(num_images):
        # 预测
        pred = predictions[img_idx]
        if len(pred) > 0:
            for p in pred:
                cls = int(p[5])
                conf = p[4]
                box = p[:4]
                class_preds[cls].append((conf, img_idx, box))
        
        # 真实框
        gt = targets[img_idx]
        if len(gt) > 0:
            for g in gt:
                cls = int(g[0])
                box = g[1:]
                class_gts[cls].append((img_idx, box))
    
    # 计算每个类别的 AP
    # 口径与 pycocotools 对齐：数据集中没有 GT 的类别不参与 mAP 平均；
    # 有 GT 但没有预测的类别按 AP=0 计入。
    ap_per_iou = defaultdict(list)  # {iou_thresh: [ap_per_class_with_gt]}
    ap50_per_class = [0.0] * num_classes

    for cls in range(num_classes):
        # 获取该类别的所有预测
        cls_pred = class_preds.get(cls, [])
        cls_gt = class_gts.get(cls, [])

        if len(cls_gt) == 0:
            # 无 GT 类别：排除在平均之外（COCO 协议）
            continue

        if len(cls_pred) == 0:
            # 有 GT 无预测：AP=0 计入平均
            for iou_thresh in iou_thresholds:
                ap_per_iou[iou_thresh].append(0.0)
            continue
        
        # 按置信度排序
        cls_pred.sort(key=lambda x: x[0], reverse=True)
        
        # 统计每个图像的真实框数量
        gt_per_image = defaultdict(int)
        for img_idx, _ in cls_gt:
            gt_per_image[img_idx] += 1
        
        total_gt = len(cls_gt)
        
        # 为每个 IoU 阈值计算 AP
        for iou_thresh in iou_thresholds:
            # 记录每个真实框是否被匹配
            gt_matched = {}  # {img_idx: [matched_flags]}
            
            # 初始化
            for img_idx, _ in cls_gt:
                if img_idx not in gt_matched:
                    gt_matched[img_idx] = [False] * gt_per_image[img_idx]
            
            # 按图像组织真实框
            gt_boxes_per_image = defaultdict(list)
            for img_idx, box in cls_gt:
                gt_boxes_per_image[img_idx].append(box)
            for img_idx in gt_boxes_per_image:
                gt_boxes_per_image[img_idx] = np.array(gt_boxes_per_image[img_idx])
            
            # 计算 TP 和 FP
            tp = np.zeros(len(cls_pred))
            fp = np.zeros(len(cls_pred))
            
            for pred_idx, (conf, img_idx, pred_box) in enumerate(cls_pred):
                if img_idx not in gt_boxes_per_image:
                    fp[pred_idx] = 1
                    continue
                
                gt_boxes = gt_boxes_per_image[img_idx]
                
                if len(gt_boxes) == 0:
                    fp[pred_idx] = 1
                    continue
                
                # 计算 IoU
                pred_box_np = np.array(pred_box).reshape(1, 4)
                ious = box_iou(pred_box_np, gt_boxes)[0]

                # COCO 协议：在「尚未匹配」的 GT 中取 IoU 最大者，达阈值记 TP。
                # 已匹配的 GT 不可重复占用，但不阻止该预测匹配其余空闲 GT。
                candidates = ious.copy()
                candidates[np.array(gt_matched[img_idx], dtype=bool)] = -1.0
                best_gt_idx = int(np.argmax(candidates))

                if candidates[best_gt_idx] >= iou_thresh:
                    tp[pred_idx] = 1
                    gt_matched[img_idx][best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # 累积 TP 和 FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # 计算召回率和精确率
            recall = tp_cumsum / total_gt
            precision = tp_cumsum / np.maximum(
                tp_cumsum + fp_cumsum, np.finfo(np.float64).eps
            )
            
            # 计算 AP
            ap, _, _ = compute_ap(recall, precision)
            ap_per_iou[iou_thresh].append(ap)
            if np.isclose(iou_thresh, 0.5):
                ap50_per_class[cls] = float(ap)
    
    # 计算 mAP（整个数据集没有任何 GT 时全部指标记 0）
    results = {}
    has_gt = len(ap_per_iou[iou_thresholds[0]]) > 0

    results['map50'] = float(np.mean(ap_per_iou[0.50])) if has_gt else 0.0
    results['map75'] = float(np.mean(ap_per_iou[0.75])) if has_gt else 0.0

    if has_gt:
        all_aps = [np.mean(ap_per_iou[iou_thresh]) for iou_thresh in iou_thresholds]
        results['map50_95'] = float(np.mean(all_aps))
    else:
        results['map50_95'] = 0.0

    # 每类 AP@50（无 GT 的类别显示 0.0，但不参与上面的平均）
    results['ap_per_class_50'] = ap50_per_class
    
    return results


def format_metric_summary(results):
    """Format key metrics for human-readable logging."""
    return (
        f"mAP@50: {results.get('map50', 0.0):.4f} | "
        f"mAP@75: {results.get('map75', 0.0):.4f} | "
        f"mAP@50-95: {results.get('map50_95', 0.0):.4f}"
    )


def format_class_summary(results, class_names=None, topk=3):
    """Format the strongest per-class AP@50 entries."""
    ap_values = results.get('ap_per_class_50', [])
    if not ap_values:
        return "per-class AP@50: unavailable"

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(len(ap_values))]

    ranked = sorted(
        enumerate(ap_values),
        key=lambda item: item[1],
        reverse=True,
    )[:max(topk, 1)]
    parts = [
        f"{class_names[idx]}={score:.4f}"
        for idx, score in ranked
    ]
    return "per-class AP@50 (top): " + ", ".join(parts)


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, num_classes=12):
        self.num_classes = num_classes
        self.predictions = []
        self.targets = []
    
    def update(self, preds, gts):
        """
        更新指标
        
        Args:
            preds: list of predictions
                each: [N, 6] (x1, y1, x2, y2, conf, cls)
            gts: list of targets
                each: [M, 5] (cls, x1, y1, x2, y2)
        """
        self.predictions.extend(preds)
        self.targets.extend(gts)
    
    def compute(self):
        """计算所有指标"""
        if len(self.predictions) == 0:
            return {
                'map50': 0.0,
                'map75': 0.0,
                'map50_95': 0.0,
            }
        
        results = compute_map(
            self.predictions, 
            self.targets, 
            self.num_classes
        )
        
        return results
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


def evaluate(model, dataloader, device, conf_thres=0.001, iou_thres=0.7, 
             num_classes=12, max_det=100, img_size=(640, 640)):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        num_classes: 类别数
        max_det: 最大检测数
        img_size: 图像尺寸
    Returns:
        results: 评估结果
    """
    model.eval()
    metrics = MetricsCalculator(num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['labels']
            
            # 推理
            predictions = model.predict(images, conf_thres=conf_thres, 
                                        iou_thres=iou_thres, max_det=max_det)
            
            # 处理预测结果
            preds_np = []
            for pred in predictions:
                preds_np.append(pred.cpu().numpy())
            
            # 处理真实标签
            batch_size = images.shape[0]
            gts_np = []
            for b in range(batch_size):
                mask = labels[:, 0] == b
                gt = labels[mask]
                if len(gt) > 0:
                    # labels: [batch_idx, cls, cx, cy, w, h] 归一化
                    cls = gt[:, 1:2]
                    cx, cy, w, h = gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5]
                    
                    img_h, img_w = img_size
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    
                    gt_boxes = torch.cat([cls, 
                                         x1.unsqueeze(1), y1.unsqueeze(1),
                                         x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)
                    gts_np.append(gt_boxes.numpy())
                else:
                    gts_np.append(np.zeros((0, 5)))
            
            metrics.update(preds_np, gts_np)
    
    results = metrics.compute()
    return results
