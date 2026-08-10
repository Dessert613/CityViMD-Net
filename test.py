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
from datasets.multimodal_dataset import load_config


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
                        help='启用简单翻转 TTA')
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


def preprocess(images, img_size):
    """图像预处理"""
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
    ir_norm = ir_padded.astype(np.float32) / 255.0
    processed['infrared'] = torch.from_numpy(ir_norm).unsqueeze(0).float()
    
    # Depth
    depth = images['depth']
    depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    depth_padded = np.zeros((h, w), dtype=np.uint16)
    depth_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = depth_resized
    depth_norm = np.clip(depth_padded.astype(np.float32) / 20000.0, 0, 1)
    processed['depth'] = torch.from_numpy(depth_norm).unsqueeze(0).float()
    
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
    
    # 推理
    print("\nRunning inference...")
    for sample_id in tqdm(sample_ids, desc='Testing'):
        # 加载图像
        images = {}
        for mod in modalities:
            img_path = os.path.join(input_dir, mod, f"{sample_id}.png")
            images[mod] = load_image(mod, img_path, img_size)
        
        # 预处理
        input_tensor, scale_info = preprocess(images, img_size)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        if use_half:
            input_tensor = input_tensor.half()
        
        # 推理
        with torch.no_grad():
            results = model.predict(
                input_tensor,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det
            )
            detections = results[0].cpu().numpy()
            if args.tta:
                flipped_tensor = torch.flip(input_tensor, dims=[3])
                tta_results = model.predict(
                    flipped_tensor,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_det=args.max_det
                )
                tta_det = tta_results[0].cpu().numpy()
                tta_det[:, :4] = flip_boxes_horizontally(tta_det[:, :4], img_size[1])
                detections = np.concatenate([detections, tta_det], axis=0) if len(tta_det) else detections
                detections = nms_per_class(detections, args.iou_thres, args.max_det)

        # 后处理
        detections = postprocess(detections, scale_info)
        
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
