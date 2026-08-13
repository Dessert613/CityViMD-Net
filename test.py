"""
CityViMD-Net 测试/推理脚本
生成比赛要求的 TXT 格式预测结果
"""

import os
import sys
import argparse
import yaml
import numpy as np
import cv2
import torch
from tqdm import tqdm
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import build_model
from datasets.multimodal_dataset import encode_depth, encode_infrared, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='CityViMD-Net Testing')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--input', type=str, default='data/test',
                        help='测试数据目录')
    parser.add_argument('--output', type=str, default='runs/test/predictions',
                        help='输出目录')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.7,
                        help='NMS IoU 阈值')
    parser.add_argument('--max-det', type=int, default=100,
                        help='每张图最大检测数')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU ID')
    parser.add_argument('--zip', action='store_true',
                        help='打包结果为 zip')
    parser.add_argument('--tta', action='store_true',
                        help='启用水平翻转 TTA')
    parser.add_argument('--tta-scales', type=str, default='',
                        help='多尺度 TTA，如 "0.75,1.0,1.25"（留空 = 单尺度）')
    parser.add_argument('--tta-iou', type=float, default=0.55,
                        help='TTA 视角间 WBF 融合的 IoU 阈值')
    return parser.parse_args()


def load_image(modality, path, img_size):
    """加载单模态图像"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {modality} image: {path}")
    
    if modality == 'rgb':
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif modality == 'infrared':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif modality == 'depth':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((img_size[0], img_size[1]), dtype=np.uint16)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    if img is None:
        raise ValueError(f"Failed to decode {modality} image: {path}")
    return img


def preprocess(images, img_size, depth_validity_mask=False,
               depth_encoding='linear', ir_encoding='raw'):
    """图像预处理（与训练侧 datasets.multimodal_dataset 共享编码实现）"""
    h, w = img_size
    modality_shapes = {name: image.shape[:2] for name, image in images.items()}
    if len(set(modality_shapes.values())) != 1:
        raise ValueError(f"Input modalities are not spatially aligned: {modality_shapes}")
    
    processed = {}
    
    # RGB
    rgb = images['rgb']
    rgb_h, rgb_w = rgb.shape[:2]
    scale = min(w / rgb_w, h / rgb_h)
    new_w = int(rgb_w * scale)
    new_h = int(rgb_h * scale)
    
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb_padded = np.zeros((h, w, 3), dtype=np.uint8)
    pad_h = (h - new_h) // 2
    pad_w = (w - new_w) // 2
    rgb_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = rgb_resized
    
    # 归一化
    rgb_norm = rgb_padded.astype(np.float32) / 255.0
    processed['rgb'] = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    
    # Infrared
    ir = images['infrared']
    ir_resized = cv2.resize(ir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ir_padded = np.zeros((h, w), dtype=np.uint8)
    ir_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = ir_resized
    ir_norm = encode_infrared(ir_padded, ir_encoding)
    processed['infrared'] = torch.from_numpy(ir_norm).unsqueeze(0).float()
    
    # Depth
    depth = images['depth']
    depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    depth_padded = np.zeros((h, w), dtype=np.uint16)
    depth_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = depth_resized
    depth_norm = encode_depth(depth_padded.astype(np.float32), depth_encoding)
    depth_channels = [torch.from_numpy(depth_norm)]
    if depth_validity_mask:
        # 与训练侧一致：0 深度（含 letterbox 填充）视为无效
        depth_channels.append(
            torch.from_numpy((depth_padded > 0).astype(np.float32))
        )
    processed['depth'] = torch.stack(depth_channels, dim=0).float()
    
    # 拼接
    input_tensor = torch.cat([processed['rgb'], 
                              processed['infrared'], 
                              processed['depth']], dim=0)
    
    # 记录缩放信息用于后处理
    scale_info = {
        'scale': scale,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'orig_h': rgb_h,
        'orig_w': rgb_w,
    }
    
    return input_tensor, scale_info


def postprocess(detections, scale_info):
    """后处理：将检测框映射回原图坐标"""
    if len(detections) == 0:
        return detections
    
    scale = scale_info['scale']
    pad_h = scale_info['pad_h']
    pad_w = scale_info['pad_w']
    
    # 坐标映射回原图
    detections = detections.copy()
    detections[:, 0] = (detections[:, 0] - pad_w) / scale  # x1
    detections[:, 1] = (detections[:, 1] - pad_h) / scale  # y1
    detections[:, 2] = (detections[:, 2] - pad_w) / scale  # x2
    detections[:, 3] = (detections[:, 3] - pad_h) / scale  # y2
    
    # 裁剪到图像范围内
    orig_w = scale_info['orig_w']
    orig_h = scale_info['orig_h']
    detections[:, 0] = np.clip(detections[:, 0], 0, orig_w)
    detections[:, 1] = np.clip(detections[:, 1], 0, orig_h)
    detections[:, 2] = np.clip(detections[:, 2], 0, orig_w)
    detections[:, 3] = np.clip(detections[:, 3], 0, orig_h)
    
    return detections


def xyxy_to_yolo(boxes, img_w, img_h):
    """xyxy 转 YOLO 格式 (cx, cy, w, h) 归一化"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h
    w = (boxes[:, 2] - boxes[:, 0]) / img_w
    h = (boxes[:, 3] - boxes[:, 1]) / img_h
    
    return np.stack([cx, cy, w, h], axis=1)


def nms_per_class(detections, iou_thres, max_det):
    if len(detections) == 0:
        return detections
    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5].astype(int)
    keep_all = []
    for cls_id in np.unique(classes):
        cls_idx = np.where(classes == cls_id)[0]
        cls_boxes = boxes[cls_idx]
        cls_scores = scores[cls_idx]
        order = np.argsort(-cls_scores)
        while len(order) > 0:
            current = order[0]
            keep_all.append(cls_idx[current])
            if len(order) == 1:
                break
            rest = order[1:]
            ious = _box_iou_np(cls_boxes[current], cls_boxes[rest])
            order = rest[ious <= iou_thres]
    keep_all = np.array(keep_all, dtype=int)
    keep_all = keep_all[np.argsort(-scores[keep_all])]
    return detections[keep_all[:max_det]]


def _box_iou_np(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = np.clip(box[2] - box[0], 0, None) * np.clip(box[3] - box[1], 0, None)
    area2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / np.maximum(area1 + area2 - inter, 1e-7)


def flip_boxes_horizontally(detections, width):
    if len(detections) == 0:
        return detections
    flipped = detections.copy()
    x1 = detections[:, 0].copy()
    x2 = detections[:, 2].copy()
    flipped[:, 0] = width - x2
    flipped[:, 2] = width - x1
    return flipped


def _round_to_stride(value, stride=32):
    """TTA 尺度对齐到骨干步长的整数倍。"""
    return max(stride, int(round(value / stride)) * stride)


def weighted_box_fusion(detections, iou_thres, num_views, max_det):
    """单模型 TTA 的逐类加权框融合（WBF 简化实现）。

    Args:
        detections: [N, 6] (x1, y1, x2, y2, conf, cls)，各视角结果
            已映射回原图坐标后拼接
        num_views: TTA 视角总数；簇分数按 min(簇大小, num_views)/num_views
            缩放，奖励跨视角共识
    """
    if len(detections) == 0:
        return detections
    fused_all = []
    classes = detections[:, 5].astype(int)
    for cls_id in np.unique(classes):
        cls_det = detections[classes == cls_id]
        cls_det = cls_det[np.argsort(-cls_det[:, 4])]
        clusters = []
        fused = []
        for row in cls_det:
            matched = False
            for index, box in enumerate(fused):
                if _box_iou_np(box[:4], row[None, :4])[0] > iou_thres:
                    clusters[index].append(row)
                    members = np.stack(clusters[index])
                    weights = np.maximum(members[:, 4], 1e-7)
                    box[:4] = (
                        (members[:, :4] * weights[:, None]).sum(axis=0)
                        / weights.sum()
                    )
                    box[4] = members[:, 4].mean()
                    matched = True
                    break
            if not matched:
                clusters.append([row])
                fused.append(row.copy())
        for index, box in enumerate(fused):
            box[4] *= min(len(clusters[index]), num_views) / num_views
            box[5] = cls_id
            fused_all.append(box)
    fused_all = np.stack(fused_all)
    fused_all = fused_all[np.argsort(-fused_all[:, 4])]
    return fused_all[:max_det]


def main():
    args = parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 设置设备
    use_cuda = torch.cuda.is_available() and args.device.lower() != 'cpu'
    device = torch.device(f'cuda:{args.device}' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 构建模型
    print("Building model...")
    model = build_model(cfg)
    
    # 加载权重
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    use_half = bool(cfg['test'].get('half', False) and device.type == 'cuda')
    if use_half:
        model.half()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取测试样本
    input_dir = args.input
    rgb_dir = os.path.join(input_dir, 'rgb')
    
    if not os.path.exists(rgb_dir):
        print(f"Error: RGB directory not found: {rgb_dir}")
        return
    
    sample_ids = [
        os.path.splitext(f)[0] for f in sorted(os.listdir(rgb_dir))
        if f.lower().endswith('.png')
    ]
    if not sample_ids:
        raise RuntimeError(f"No PNG test samples found in: {rgb_dir}")
    
    print(f"Found {len(sample_ids)} test samples")
    
    img_size = tuple(cfg['data']['img_size'])
    modalities = cfg['data']['modalities']
    depth_validity_mask = bool(cfg['data'].get('depth_validity_mask', False))
    depth_encoding = cfg['data'].get('depth_encoding', 'linear')
    ir_encoding = cfg['data'].get('ir_encoding', 'raw')

    # TTA 视角集合：尺度 × 水平翻转（均为同一模型的多次前向）
    tta_scales = [1.0]
    if args.tta_scales:
        tta_scales = [
            float(value) for value in args.tta_scales.split(',') if value.strip()
        ]
    flips = [False, True] if args.tta else [False]
    view_sizes = [
        (_round_to_stride(img_size[0] * scale), _round_to_stride(img_size[1] * scale))
        for scale in tta_scales
    ]
    num_views = len(view_sizes) * len(flips)
    if num_views > 1:
        print(f"TTA enabled: sizes={view_sizes}, flips={flips} "
              f"({num_views} views, WBF iou={args.tta_iou})")
    
    # 推理
    print("\nRunning inference...")
    for sample_id in tqdm(sample_ids, desc='Testing'):
        # 加载图像
        images = {}
        for mod in modalities:
            img_path = os.path.join(input_dir, mod, f"{sample_id}.png")
            images[mod] = load_image(mod, img_path, img_size)
        
        # 逐视角推理：每个视角单独预处理并映射回原图坐标，再统一融合
        view_detections = []
        scale_info = None
        for view_size in view_sizes:
            input_tensor, view_scale_info = preprocess(
                images, view_size,
                depth_validity_mask=depth_validity_mask,
                depth_encoding=depth_encoding,
                ir_encoding=ir_encoding,
            )
            if scale_info is None:
                scale_info = view_scale_info  # 各视角 orig_w/h 相同
            base_tensor = input_tensor.unsqueeze(0).to(device)
            if use_half:
                base_tensor = base_tensor.half()
            for flip in flips:
                tensor = torch.flip(base_tensor, dims=[3]) if flip else base_tensor
                with torch.no_grad():
                    results = model.predict(
                        tensor,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det
                    )
                det = results[0].cpu().numpy()
                if flip and len(det):
                    det[:, :4] = flip_boxes_horizontally(det[:, :4], view_size[1])
                det = postprocess(det, view_scale_info)
                view_detections.append(det)

        if num_views > 1:
            nonempty = [det for det in view_detections if len(det)]
            if nonempty:
                stacked = np.concatenate(nonempty, axis=0)
                detections = weighted_box_fusion(
                    stacked, args.tta_iou, num_views, args.max_det
                )
            else:
                detections = np.zeros((0, 6), dtype=np.float32)
        else:
            detections = view_detections[0]
        
        # 转换为 YOLO 格式
        if len(detections) > 0:
            boxes = detections[:, :4]
            confs = detections[:, 4]
            cls_ids = detections[:, 5].astype(int)
            
            orig_w = scale_info['orig_w']
            orig_h = scale_info['orig_h']
            yolo_boxes = xyxy_to_yolo(boxes, orig_w, orig_h)
            valid = (
                np.isfinite(yolo_boxes).all(axis=1) &
                np.isfinite(confs) &
                (yolo_boxes[:, 2] > 0) &
                (yolo_boxes[:, 3] > 0) &
                (cls_ids >= 0) &
                (cls_ids < cfg['data']['num_classes'])
            )
            yolo_boxes = np.clip(yolo_boxes[valid], 0.0, 1.0)
            confs = np.clip(confs[valid], 0.0, 1.0)
            cls_ids = cls_ids[valid]
            
            # 按置信度排序
            sorted_idx = np.argsort(-confs)
            yolo_boxes = yolo_boxes[sorted_idx]
            confs = confs[sorted_idx]
            cls_ids = cls_ids[sorted_idx]
            
            # 截断到 max_det
            if len(yolo_boxes) > args.max_det:
                yolo_boxes = yolo_boxes[:args.max_det]
                confs = confs[:args.max_det]
                cls_ids = cls_ids[:args.max_det]
        else:
            yolo_boxes = np.zeros((0, 4))
            confs = np.zeros(0)
            cls_ids = np.zeros(0, dtype=int)
        
        # 保存结果
        output_path = os.path.join(args.output, f"{sample_id}.txt")
        with open(output_path, 'w') as f:
            for i in range(len(cls_ids)):
                # 格式: class_id cx cy w h confidence
                line = f"{cls_ids[i]} {yolo_boxes[i, 0]:.6f} {yolo_boxes[i, 1]:.6f} " \
                       f"{yolo_boxes[i, 2]:.6f} {yolo_boxes[i, 3]:.6f} {confs[i]:.6f}\n"
                f.write(line)
    
    print(f"\nPredictions saved to {args.output}")
    print(f"Total samples: {len(sample_ids)}")
    
    # 打包 zip
    if args.zip:
        zip_path = os.path.join(os.path.dirname(args.output), 'predictions.zip')
        print(f"\nCreating zip file: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for sample_id in sample_ids:
                txt_path = os.path.join(args.output, f"{sample_id}.txt")
                zipf.write(txt_path, arcname=f"{sample_id}.txt")
        
        print(f"Zip file created: {zip_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
