"""
CityViMD-Net 多模态数据集加载模块
支持 RGB + Infrared + Depth 三模态目标检测
"""

import os
import glob
import hashlib
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


DEPTH_MAX_MM = 20000.0
DEPTH_ENCODINGS = ('linear', 'inverse', 'log', 'minmax')
IR_ENCODINGS = ('raw', 'clahe', 'percentile')


def file_sha256(path, chunk_size=1 << 20):
    """计算文件 SHA-256（用于测试集哈希黑名单守卫）。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def encode_depth(depth, encoding='linear'):
    """深度编码（输入毫米 float32，输出 [0,1] float32）。

    官方约定深度 0 或过小为无效；所有编码下无效像素统一输出 0，
    「无效」与「极近」的区分交给可选的有效性掩码通道。
    """
    depth = np.clip(depth.astype(np.float32), 0.0, DEPTH_MAX_MM)
    valid = depth > 0
    if encoding == 'linear':
        value = depth / DEPTH_MAX_MM
    elif encoding == 'inverse':
        # 逆深度：近处分辨率高，1000mm 处约 0.5
        value = np.where(valid, 1000.0 / (1000.0 + depth), 0.0)
    elif encoding == 'log':
        value = np.where(valid, np.log1p(depth) / np.log1p(DEPTH_MAX_MM), 0.0)
    elif encoding == 'minmax':
        if valid.any():
            valid_values = depth[valid]
            dmin = float(valid_values.min())
            dmax = float(valid_values.max())
            value = np.where(valid, (depth - dmin) / max(dmax - dmin, 1.0), 0.0)
        else:
            value = np.zeros_like(depth)
    else:
        raise ValueError(f"Unknown depth encoding: {encoding}")
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def encode_infrared(image, encoding='raw'):
    """红外编码（输入 uint8 灰度，输出 [0,1] float32）。"""
    if encoding == 'raw':
        value = image.astype(np.float32) / 255.0
    elif encoding == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        value = clahe.apply(image).astype(np.float32) / 255.0
    elif encoding == 'percentile':
        low, high = np.percentile(image, (1.0, 99.0))
        value = (image.astype(np.float32) - low) / max(float(high - low), 1.0)
    else:
        raise ValueError(f"Unknown infrared encoding: {encoding}")
    return np.clip(value, 0.0, 1.0).astype(np.float32)


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
                 augment_cfg=None, strict_modalities=True,
                 depth_validity_mask=False, depth_encoding='linear',
                 ir_encoding='raw', sample_ids=None,
                 forbidden_hashes_path=None):
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
        self.depth_validity_mask = depth_validity_mask
        if depth_encoding not in DEPTH_ENCODINGS:
            raise ValueError(f"Unknown depth encoding: {depth_encoding}")
        if ir_encoding not in IR_ENCODINGS:
            raise ValueError(f"Unknown infrared encoding: {ir_encoding}")
        self.depth_encoding = depth_encoding
        self.ir_encoding = ir_encoding
        self.requested_sample_ids = sample_ids
        self.forbidden_hashes_path = forbidden_hashes_path
        
        # 获取所有样本ID
        self.sample_ids = self._get_sample_ids()
        self._enforce_hash_blacklist(self.sample_ids)
    
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

        # 交叉验证等场景下按显式 ID 列表取子集
        if self.requested_sample_ids is not None:
            requested = sorted(set(self.requested_sample_ids))
            available = set(sample_ids)
            unknown = [sid for sid in requested if sid not in available]
            if unknown:
                raise RuntimeError(
                    f"Requested sample ids not found in split: {unknown[:5]}"
                )
            sample_ids = requested
        return sample_ids

    def _enforce_hash_blacklist(self, sample_ids):
        """合规守卫：阻止测试集图像进入训练/验证数据加载器。

        黑名单由 tools/build_test_blacklist.py 生成；文件不存在时守卫不生效。
        """
        if not self.forbidden_hashes_path:
            return
        if not os.path.exists(self.forbidden_hashes_path):
            print(
                "[compliance] test blacklist not found, guard inactive: "
                f"{self.forbidden_hashes_path}"
            )
            return
        with open(self.forbidden_hashes_path, encoding='utf-8') as file:
            payload = json.load(file)
        forbidden = set(payload.get('hashes', []))
        if not forbidden:
            return
        for sample_id in sample_ids:
            for modality in self.modalities:
                path = os.path.join(self.root_dir, modality, f"{sample_id}.png")
                if not os.path.exists(path):
                    continue
                if file_sha256(path) in forbidden:
                    raise RuntimeError(
                        "COMPLIANCE VIOLATION: test-set image detected in "
                        f"training data: {path}"
                    )
    
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
                value = encode_infrared(image, self.ir_encoding)
                normalized[modality] = torch.from_numpy(value).unsqueeze(0).float()
            elif modality == 'depth':
                depth = image.astype(np.float32)
                value = encode_depth(depth, self.depth_encoding)
                channels = [torch.from_numpy(value)]
                if self.depth_validity_mask:
                    # 官方约定深度 0 或过小为无效；掩码区分「无效」与「极近」
                    channels.append(
                        torch.from_numpy((depth > 0).astype(np.float32))
                    )
                normalized[modality] = torch.stack(channels, dim=0).float()
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

        # 模态 dropout：随机整幅置零红外或深度，模拟传感器失效/劣化
        dropout_prob = self.augment_cfg.get('modality_dropout', 0.0)
        if dropout_prob > 0 and random.random() < dropout_prob:
            candidates = [m for m in ('infrared', 'depth') if m in images]
            if candidates:
                dropped = random.choice(candidates)
                images[dropped] = np.zeros_like(images[dropped])
        
        return images, labels


def seed_worker(worker_id):
    """DataLoader worker 随机种子初始化。

    默认情况下各 worker 继承相同的 Python random 状态，增强序列会在
    worker 间重复；按 torch 官方配方为每个 worker 派生独立种子。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def build_dataloader(cfg, split='train', sample_ids=None, eval_mode=False):
    """构建数据加载器。

    Args:
        sample_ids: 显式样本 ID 子集（交叉验证折）
        eval_mode: 强制评估行为（关增强、评估 batch、不 shuffle），
            用于「从训练目录取验证折」的交叉验证场景
    """
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    
    split_dir = data_cfg.get(f'{split}_dir', split)
    is_train = (split == 'train') and not eval_mode
    dataset = MultimodalDataset(
        root_dir=data_cfg['root'],
        split=split_dir,
        img_size=tuple(data_cfg['img_size']),
        num_classes=data_cfg['num_classes'],
        augment=is_train,
        modalities=data_cfg['modalities'],
        augment_cfg=train_cfg.get('augment', {}),
        strict_modalities=data_cfg.get('strict_modalities', True),
        depth_validity_mask=data_cfg.get('depth_validity_mask', False),
        depth_encoding=data_cfg.get('depth_encoding', 'linear'),
        ir_encoding=data_cfg.get('ir_encoding', 'raw'),
        sample_ids=sample_ids,
        forbidden_hashes_path=data_cfg.get('test_blacklist'),
    )
    
    batch_size = train_cfg['batch_size'] if is_train else cfg['test']['batch_size']
    num_workers = train_cfg['workers']
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=seed_worker,
    )
    
    return dataloader


def load_fold_assignments(path):
    """读取 tools/make_folds.py 生成的折划分文件。"""
    with open(path, encoding='utf-8') as file:
        payload = json.load(file)
    return {str(key): int(value) for key, value in payload['assignments'].items()}


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg
