"""
CityViMD-Net 多模态数据集加载模块
支持 RGB + Infrared + Depth 三模态目标检测
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class MultimodalDataset(Dataset):
    """
    多模态目标检测数据集
    
    数据目录结构:
        root/
            train/
                rgb/      *.png
                infrared/ *.png
                depth/    *.png
                labels/   *.txt
            val/
                ...
    """
    
    def __init__(self, root_dir, split='train', img_size=(640, 640), 
                 num_classes=12, augment=True, modalities=None,
                 augment_cfg=None, strict_modalities=True):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train' / 'val' / 'test'
            img_size: 输入图像尺寸 (h, w)
            num_classes: 类别数
            augment: 是否数据增强
            modalities: 使用的模态列表
        """
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')
        self.modalities = modalities or ['rgb', 'infrared', 'depth']
        self.augment_cfg = augment_cfg or {}
        self.strict_modalities = strict_modalities
        
        # 获取所有样本ID
        self.sample_ids = self._get_sample_ids()
    
    def _get_sample_ids(self):
        """获取所有样本ID（不含扩展名）"""
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
        if not sample_ids:
            raise RuntimeError(f"No PNG samples found in: {rgb_dir}")

        if self.strict_modalities:
            rgb_ids = set(sample_ids)
            for modality in self.modalities:
                modality_dir = os.path.join(self.root_dir, modality)
                if not os.path.isdir(modality_dir):
                    raise FileNotFoundError(
                        f"Required modality directory not found: {modality_dir}"
                    )
                modality_ids = {
                    os.path.splitext(os.path.basename(path))[0]
                    for path in glob.glob(os.path.join(modality_dir, '*.png'))
                }
                missing = sorted(rgb_ids - modality_ids)
                extra = sorted(modality_ids - rgb_ids)
                if missing or extra:
                    raise RuntimeError(
                        f"Modality '{modality}' is not aligned with RGB: "
                        f"{len(missing)} missing, {len(extra)} extra files. "
                        f"Examples missing={missing[:3]}, extra={extra[:3]}"
                    )
        return sample_ids
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 加载三模态图像
        images = {}
        for mod in self.modalities:
            img_path = os.path.join(self.root_dir, mod, f"{sample_id}.png")
            images[mod] = self._load_image(mod, img_path)
        self._validate_alignment(images, sample_id)
        
        # 加载标签
        label_path = os.path.join(self.root_dir, 'labels', f"{sample_id}.txt")
        labels = self._load_labels(label_path)
        
        # 数据增强
        if self.augment:
            images, labels = self._augment(images, labels)
        
        # 调整尺寸
        images, labels = self._resize(images, labels)
        
        # 归一化
        images = self._normalize(images)
        
        # 转换为张量
        images_tensor = torch.cat([images[m] for m in self.modalities], dim=0)
        
        # 标签格式转换
        if len(labels) > 0:
            labels_tensor = torch.zeros((len(labels), 6))
            labels_tensor[:, 0] = 0  # batch index (collate时填充)
            labels_tensor[:, 1:] = torch.from_numpy(labels)
        else:
            labels_tensor = torch.zeros((0, 6))
        
        return {
            'images': images_tensor,
            'labels': labels_tensor,
            'sample_id': sample_id,
            'modalities': self.modalities
        }
    
    def _load_image(self, modality, path):
        """加载单模态图像"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {modality} image: {path}")
        
        if modality == 'rgb':
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif modality == 'infrared':
            # 红外图像：三通道但实际是灰度，取第一通道
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif modality == 'depth':
            # 深度图像：16位单通道
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint16)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if img is None:
            raise ValueError(f"Failed to decode {modality} image: {path}")

        if modality == 'rgb' and img.ndim != 3:
            raise ValueError(f"Expected RGB image to have 3 channels: {path}")
        if modality in {'infrared', 'depth'} and img.ndim != 2:
            img = img[..., 0]
        
        return img

    def _validate_alignment(self, images, sample_id):
        """空间对齐数据必须具有完全相同的原始高宽。"""
        shapes = {mod: image.shape[:2] for mod, image in images.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(
                f"Unaligned modality sizes for sample '{sample_id}': {shapes}"
            )
    
    def _load_labels(self, path):
        """加载YOLO格式标签 [class_id, cx, cy, w, h]"""
        labels = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) != 5:
                            raise ValueError(
                                f"Expected 5 label fields at {path}:{line_number}, "
                                f"got {len(parts)}"
                            )
                        if len(parts) == 5:
                            cls_id = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            if not 0 <= cls_id < self.num_classes:
                                raise ValueError(
                                    f"Invalid class id at {path}:{line_number}: {cls_id}"
                                )
                            if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
                                raise ValueError(
                                    f"Non-normalized box at {path}:{line_number}"
                                )
                            if w <= 0 or h <= 0:
                                raise ValueError(
                                    f"Non-positive box size at {path}:{line_number}"
                                )
                            labels.append([cls_id, cx, cy, w, h])
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
    
    def _resize(self, images, labels):
        """调整图像尺寸，保持比例并填充"""
        h, w = self.img_size
        
        resized_images = {}
        for mod, img in images.items():
            if mod == 'rgb':
                img_h, img_w = img.shape[:2]
            else:
                img_h, img_w = img.shape[:2]
            
            # 计算缩放比例
            scale = min(w / img_w, h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # 调整大小
            if mod == 'rgb':
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # 填充
                padded = np.zeros((h, w, 3), dtype=img.dtype)
            elif mod == 'depth':
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                padded = np.zeros((h, w), dtype=img.dtype)
            else:  # infrared
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                padded = np.zeros((h, w), dtype=img.dtype)
            
            # 居中填充
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            
            if mod == 'rgb':
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
            else:
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            resized_images[mod] = padded
            
            # 调整标签坐标（只需要计算一次）
            if mod == 'rgb' and len(labels) > 0:
                labels = labels.copy()
                labels[:, 1] = (labels[:, 1] * img_w * scale + pad_w) / w
                labels[:, 2] = (labels[:, 2] * img_h * scale + pad_h) / h
                labels[:, 3] = labels[:, 3] * img_w * scale / w
                labels[:, 4] = labels[:, 4] * img_h * scale / h
        
        return resized_images, labels
    
    def _normalize(self, images):
        """图像归一化"""
        normalized = {}
        
        for modality, image in images.items():
            if modality == 'rgb':
                value = image.astype(np.float32) / 255.0
                normalized[modality] = torch.from_numpy(
                    value.transpose(2, 0, 1)
                ).float()
            elif modality == 'infrared':
                value = image.astype(np.float32) / 255.0
                normalized[modality] = torch.from_numpy(value).unsqueeze(0).float()
            elif modality == 'depth':
                value = np.clip(image.astype(np.float32) / 20000.0, 0, 1)
                normalized[modality] = torch.from_numpy(value).unsqueeze(0).float()
            else:
                raise ValueError(f"Unknown modality: {modality}")
        
        return normalized
    
    def _augment(self, images, labels):
        """数据增强"""
        # 水平翻转
        if random.random() < self.augment_cfg.get('fliplr', 0.5):
            for mod in images:
                images[mod] = np.fliplr(images[mod]).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
        
        # HSV 增强（仅RGB）
        if 'rgb' in images and random.random() < self.augment_cfg.get('color_prob', 0.5):
            rgb = images['rgb']
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            h_gain = random.uniform(
                -self.augment_cfg.get('hsv_h', 0.015),
                self.augment_cfg.get('hsv_h', 0.015)
            )
            s_gain = random.uniform(
                -self.augment_cfg.get('hsv_s', 0.7),
                self.augment_cfg.get('hsv_s', 0.7)
            )
            v_gain = random.uniform(
                -self.augment_cfg.get('hsv_v', 0.4),
                self.augment_cfg.get('hsv_v', 0.4)
            )
            
            hsv[..., 0] = (hsv[..., 0] + h_gain * 180) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * (1 + s_gain), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * (1 + v_gain), 0, 255)
            
            images['rgb'] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 红外和深度也做一些亮度调整
        if 'infrared' in images and random.random() < self.augment_cfg.get('ir_gamma_prob', 0.3):
            gamma = random.uniform(0.8, 1.2)
            ir = images['infrared'].astype(np.float32)
            ir = np.clip(255 * (ir / 255) ** gamma, 0, 255).astype(np.uint8)
            images['infrared'] = ir
        
        return images, labels


def collate_fn(batch):
    """自定义批处理函数"""
    images = torch.stack([item['images'] for item in batch], dim=0)
    
    # 拼接标签，添加batch index
    labels_list = []
    for i, item in enumerate(batch):
        if len(item['labels']) > 0:
            item['labels'][:, 0] = i
            labels_list.append(item['labels'])
    
    if labels_list:
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)
    
    sample_ids = [item['sample_id'] for item in batch]
    modalities = batch[0]['modalities']
    
    return {
        'images': images,
        'labels': labels,
        'sample_ids': sample_ids,
        'modalities': modalities
    }


def build_dataloader(cfg, split='train'):
    """构建数据加载器"""
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    
    split_dir = data_cfg.get(f'{split}_dir', split)
    dataset = MultimodalDataset(
        root_dir=data_cfg['root'],
        split=split_dir,
        img_size=tuple(data_cfg['img_size']),
        num_classes=data_cfg['num_classes'],
        augment=(split == 'train'),
        modalities=data_cfg['modalities'],
        augment_cfg=train_cfg.get('augment', {}),
        strict_modalities=data_cfg.get('strict_modalities', True),
    )
    
    batch_size = train_cfg['batch_size'] if split == 'train' else cfg['test']['batch_size']
    shuffle = (split == 'train')
    num_workers = train_cfg['workers']
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg
