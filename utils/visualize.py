"""
CityViMD-Net 可视化工具
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 类别颜色
COLORS = [
    (255, 0, 0),      # 0: person - 红
    (0, 255, 0),      # 1: boat - 绿
    (0, 0, 255),      # 2: animal - 蓝
    (255, 255, 0),    # 3: seat - 黄
    (255, 0, 255),    # 4: sign - 品红
    (0, 255, 255),    # 5: bicycle - 青
    (128, 0, 128),    # 6: car - 紫
    (255, 165, 0),    # 7: ball - 橙
    (128, 128, 0),    # 8: light - 橄榄
    (0, 128, 128),    # 9: garbage_can - 水鸭
    (255, 192, 203),  # 10: uav - 粉
    (165, 42, 42),    # 11: tricycle - 棕
]

# 类别名称
CLASS_NAMES = [
    'person', 'boat', 'animal', 'seat', 'sign', 'bicycle',
    'car', 'ball', 'light', 'garbage_can', 'uav', 'tricycle'
]


def draw_detections(image, detections, class_names=None, colors=None, 
                    conf_thres=0.0, thickness=2, font_scale=0.5):
    """
    在图像上绘制检测框
    
    Args:
        image: 输入图像 [H, W, 3] BGR 或 RGB
        detections: [N, 6] (x1, y1, x2, y2, conf, cls)
        class_names: 类别名称列表
        colors: 颜色列表
        conf_thres: 置信度阈值
        thickness: 线宽
        font_scale: 字体大小
    Returns:
        image: 绘制后的图像
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if colors is None:
        colors = COLORS
    
    image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        
        if conf < conf_thres:
            continue
        
        color = colors[cls % len(colors)]
        label = f"{class_names[cls]}: {conf:.2f}"
        
        # 绘制框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), 
                     color, thickness)
        
        # 绘制标签背景
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                              font_scale, 1)
        cv2.rectangle(image, (int(x1), int(y1) - text_h - 4), 
                     (int(x1) + text_w, int(y1)), color, -1)
        
        # 绘制标签文字
        cv2.putText(image, label, (int(x1), int(y1) - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return image


def visualize_multimodal(rgb_img, ir_img, depth_img, detections=None, 
                         gt_boxes=None, save_path=None, show=False):
    """
    可视化三模态图像和检测结果
    
    Args:
        rgb_img: RGB 图像 [H, W, 3]
        ir_img: 红外图像 [H, W]
        depth_img: 深度图像 [H, W]
        detections: 检测结果 [N, 6]
        gt_boxes: 真实框 [M, 5] (cls, x1, y1, x2, y2)
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB')
    axes[0].axis('off')
    
    # 红外
    axes[1].imshow(ir_img, cmap='hot')
    axes[1].set_title('Infrared')
    axes[1].axis('off')
    
    # 深度
    depth_vis = np.clip(depth_img / 20000.0, 0, 1)
    axes[2].imshow(depth_vis, cmap='jet')
    axes[2].set_title('Depth')
    axes[2].axis('off')
    
    # 绘制检测框
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            color = COLORS[cls % len(COLORS)] / 255.0
            label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
            
            for ax in axes:
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
    
    # 绘制真实框
    if gt_boxes is not None and len(gt_boxes) > 0:
        for gt in gt_boxes:
            cls, x1, y1, x2, y2 = gt
            cls = int(cls)
            
            for ax in axes:
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='white', facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_training_curves(log_dir, save_path=None, show=False):
    """
    绘制训练曲线
    
    Args:
        log_dir: TensorBoard 日志目录
        save_path: 保存路径
        show: 是否显示
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not installed, cannot plot training curves")
        return
    
    # 加载事件文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out')]
    if not event_files:
        print("No event files found")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # 获取所有标量
    tags = ea.Tags()['scalars']
    
    # 分组
    train_tags = [t for t in tags if t.startswith('Train/')]
    val_tags = [t for t in tags if t.startswith('Val/')]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练损失
    ax = axes[0, 0]
    for tag in train_tags:
        if 'loss' in tag and 'box' not in tag and 'cls' not in tag and 'dfl' not in tag:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, label=tag.split('/')[-1])
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 训练各分量损失
    ax = axes[0, 1]
    for tag in train_tags:
        if 'box' in tag or 'cls' in tag or 'dfl' in tag:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, label=tag.split('/')[-1])
    ax.set_title('Training Loss Components')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证 mAP
    ax = axes[1, 0]
    for tag in val_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, label=tag.split('/')[-1])
    ax.set_title('Validation mAP')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 学习率
    ax = axes[1, 1]
    lr_tag = 'Train/lr'
    if lr_tag in tags:
        events = ea.Scalars(lr_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def plot_pr_curve(recall, precision, ap, class_name='', save_path=None, show=False):
    """
    绘制 P-R 曲线
    
    Args:
        recall: 召回率
        precision: 精确率
        ap: Average Precision
        class_name: 类别名称
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'P-R Curve - {class_name} (AP={ap:.4f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
