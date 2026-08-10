"""
CityViMD-Net 损失函数
包含: BCE分类损失 + CIoU定位损失 + DFL损失
以及目标分配策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算 IoU / GIoU / DIoU / CIoU
    
    Args:
        box1: [N, 4]
        box2: [M, 4]
        xywh: 是否是 xywh 格式
    Returns:
        iou: [N, M]
    """
    if xywh:
        # xywh -> xyxy
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # 交集
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # 并集
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + eps
    
    iou = inter_area / union_area
    
    if GIoU or DIoU or CIoU:
        # 最小外接矩形
        cw = torch.max(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0)) - \
             torch.min(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        ch = torch.max(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0)) - \
             torch.min(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        
        if CIoU or DIoU:
            # 中心点距离
            cx1 = (b1_x1 + b1_x2) / 2
            cy1 = (b1_y1 + b1_y2) / 2
            cx2 = (b2_x1 + b2_x2) / 2
            cy2 = (b2_y1 + b2_y2) / 2
            
            rho2 = (cx1.unsqueeze(1) - cx2.unsqueeze(0)) ** 2 + \
                   (cy1.unsqueeze(1) - cy2.unsqueeze(0)) ** 2
            c2 = cw ** 2 + ch ** 2 + eps
            
            if CIoU:
                # 宽高比
                w1 = b1_x2 - b1_x1
                h1 = b1_y2 - b1_y1
                w2 = b2_x2 - b2_x1
                h2 = b2_y2 - b2_y1
                
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - 
                                          torch.atan(w1.unsqueeze(1) / (h1.unsqueeze(1) + eps))) ** 2
                alpha = v / (1 - iou + v + eps)
                
                return iou - (rho2 / c2 + v * alpha)
            else:
                return iou - rho2 / c2
        else:
            # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union_area) / c_area
    
    return iou


def aligned_ciou(box1, box2, eps=1e-7):
    """逐元素计算两组 xyxy 框的 CIoU，输入形状均为 [N, 4]。"""
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((0,), device=box1.device, dtype=box1.dtype)
    inter_lt = torch.maximum(box1[:, :2], box2[:, :2])
    inter_rb = torch.minimum(box1[:, 2:], box2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    wh1 = (box1[:, 2:] - box1[:, :2]).clamp(min=eps)
    wh2 = (box2[:, 2:] - box2[:, :2]).clamp(min=eps)
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    union = area1 + area2 - inter + eps
    iou = inter / union

    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    center_dist = ((center1 - center2) ** 2).sum(dim=1)
    enclosing_lt = torch.minimum(box1[:, :2], box2[:, :2])
    enclosing_rb = torch.maximum(box1[:, 2:], box2[:, 2:])
    enclosing_diag = ((enclosing_rb - enclosing_lt) ** 2).sum(dim=1) + eps

    wh1 = wh1.clamp(min=eps)
    wh2 = wh2.clamp(min=eps)
    v = (4 / math.pi ** 2) * (
        torch.atan(wh2[:, 0] / wh2[:, 1]) -
        torch.atan(wh1[:, 0] / wh1[:, 1])
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_dist / enclosing_diag - alpha * v


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner (TOOD)
    动态目标分配策略
    """
    
    def __init__(self, topk=13, num_classes=12, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores: [B, N, num_classes] 预测分类得分
            pd_bboxes: [B, N, 4] 预测框 (xyxy)
            anc_points: [N, 2] 锚点坐标
            gt_labels: [B, M, 1] 真实类别
            gt_bboxes: [B, M, 4] 真实框 (xyxy)
            mask_gt: [B, M] 真实框掩码
        Returns:
            target_labels: [B, N]
            target_bboxes: [B, N, 4]
            target_scores: [B, N, num_classes]
            fg_mask: [B, N]
        """
        batch_size = pd_scores.shape[0]
        num_anchors = pd_scores.shape[1]
        
        target_labels = torch.zeros_like(pd_scores[..., 0])
        target_bboxes = torch.zeros_like(pd_bboxes)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = torch.zeros_like(pd_scores[..., 0]).bool()
        
        for b in range(batch_size):
            # 获取当前图像的 GT
            gt_b = gt_labels[b][mask_gt[b]]  # [M', 1]
            gb_b = gt_bboxes[b][mask_gt[b]]  # [M', 4]
            num_gt = len(gt_b)
            
            if num_gt == 0:
                continue
            
            # 计算对齐度量
            # alignment_metric = score^alpha * iou^beta
            iou = bbox_iou(gb_b, pd_bboxes[b], xywh=False)  # [M', N]
            scores = pd_scores[b, :, gt_b.squeeze(-1).long()].T  # [M', N]

            # 候选锚点必须位于真实框内部，避免给远离目标的位置分配正样本
            x, y = anc_points[:, 0], anc_points[:, 1]
            in_gts = (
                (x.unsqueeze(0) >= gb_b[:, 0:1]) &
                (y.unsqueeze(0) >= gb_b[:, 1:2]) &
                (x.unsqueeze(0) <= gb_b[:, 2:3]) &
                (y.unsqueeze(0) <= gb_b[:, 3:4])
            )
            alignment_metrics = scores.pow(self.alpha) * iou.clamp(min=0).pow(self.beta)
            candidate_metrics = alignment_metrics.masked_fill(~in_gts, -1.0)
            
            # TopK 选择
            topk = min(self.topk, num_anchors)
            topk_values, topk_idx = candidate_metrics.topk(topk, dim=1)
            
            # 构建目标
            is_in_topk = torch.zeros(num_gt, num_anchors, device=pd_scores.device).bool()
            for i in range(num_gt):
                valid = topk_values[i] >= 0
                is_in_topk[i, topk_idx[i, valid]] = True
            
            # 一个锚点只能分配给一个 GT（选 metric 最大的）
            fg_mask_b = is_in_topk.sum(0) > 0
            if fg_mask_b.sum() == 0:
                continue
            
            # 处理重叠
            overlap = is_in_topk.sum(0) > 1
            if overlap.sum() > 0:
                overlap_idx = torch.where(overlap)[0]
                for idx in overlap_idx:
                    best_gt = alignment_metrics[:, idx].argmax()
                    is_in_topk[:, idx] = False
                    is_in_topk[best_gt, idx] = True
            
            # 分配目标
            for i in range(num_gt):
                assigned = is_in_topk[i]
                if assigned.sum() == 0:
                    continue
                
                target_labels[b][assigned] = gt_b[i]
                target_bboxes[b][assigned] = gb_b[i]
                target_scores[b, assigned, gt_b[i].long()] = 1.0
                fg_mask[b][assigned] = True
        
        return target_labels, target_bboxes, target_scores, fg_mask


class BboxLoss(nn.Module):
    """边界框损失 (CIoU + DFL)"""
    
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, pred_dist, pred_bboxes, anchor_points, strides,
                target_bboxes, target_scores, fg_mask):
        """
        Args:
            pred_dist: [B, N, 4*(reg_max+1)] 预测分布
            pred_bboxes: [B, N, 4] 预测框 (xyxy)
            anchor_points: [N, 2] 锚点
            target_bboxes: [B, N, 4] 目标框 (xyxy)
            target_scores: [B, N, num_classes] 目标得分
            fg_mask: [B, N] 前景掩码
        Returns:
            loss_iou: CIoU 损失
            loss_dfl: DFL 损失
        """
        # 只计算前景
        if fg_mask.sum() == 0:
            zero = pred_bboxes.sum() * 0.0
            return zero, zero

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1).clamp(min=1e-6)
        
        # CIoU 损失
        iou = aligned_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou).unsqueeze(-1) * weight).sum() / weight.sum()
        
        # DFL 损失
        target_ltrb = self._bbox2dist(anchor_points, strides, target_bboxes)
        loss_dfl = self._df_loss(pred_dist[fg_mask], target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / weight.sum()
        
        return loss_iou, loss_dfl
    
    def _bbox2dist(self, anchor_points, strides, bboxes):
        """将 bbox 转换为 ltrb 分布目标"""
        # anchor_points 为像素坐标，DFL 目标需换算为特征格距离
        stride_xy = strides.unsqueeze(0)
        lt = (anchor_points.unsqueeze(0) - bboxes[..., :2]) / stride_xy
        rb = (bboxes[..., 2:] - anchor_points.unsqueeze(0)) / stride_xy
        dist = torch.cat([lt, rb], dim=-1)
        return dist.clamp(min=0, max=self.reg_max - 0.01)
    
    def _df_loss(self, pred_dist, target):
        """Distribution Focal Loss"""
        if pred_dist.numel() == 0:
            return pred_dist.new_zeros((0, 1))
        # pred_dist: [N, 4*(reg_max+1)]
        # target: [N, 4]
        target = target.clamp(min=0, max=self.reg_max - 1e-4)
        target_left = target.long()
        target_right = target_left + 1
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right
        
        pred_dist = pred_dist.view(-1, 4, self.reg_max + 1)
        
        left_loss = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_left.view(-1),
            reduction='none'
        ).view(-1, 4)
        right_loss = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_right.view(-1).clamp(max=self.reg_max),
            reduction='none'
        ).view(-1, 4)
        loss = left_loss * weight_left + right_loss * weight_right
        
        return loss.mean(dim=-1, keepdim=True)


class v8DetectionLoss:
    """
    YOLOv8 检测损失
    包含: 分类损失(BCE) + 定位损失(CIoU) + DFL损失
    """
    
    def __init__(self, model, cfg=None):
        device = next(model.parameters()).device
        h = cfg['train']['loss'] if cfg else {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
        
        self.num_classes = model.num_classes
        self.reg_max = model.head.reg_max
        
        # 损失权重
        self.box_weight = h.get('box', 7.5)
        self.cls_weight = h.get('cls', 0.5)
        self.dfl_weight = h.get('dfl', 1.5)
        
        # 目标分配器
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.num_classes,
            alpha=1.0,
            beta=6.0
        )
        
        # 损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max)
        
        self.device = device
        self.cfg = cfg or {}
    
    def __call__(self, preds, targets, img_size=(640, 640)):
        """
        Args:
            preds: list of predictions [P3, P4, P5]
            targets: [N, 6] (batch_idx, cls, cx, cy, w, h) 归一化坐标
            img_size: 图像尺寸
        Returns:
            loss: 总损失
            loss_items: 各损失分量
        """
        device = self.device
        batch_size = preds[0].shape[0]
        
        # 解码预测
        pred_bboxes_list = []
        pred_scores_list = []
        pred_dist_list = []
        anchor_points_list = []
        stride_list = []
        
        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            stride_y = img_size[0] / H
            stride_x = img_size[1] / W
            
            # 分离分类和回归
            cls_pred = pred[:, :self.num_classes, :, :]  # [B, nc, H, W]
            reg_pred = pred[:, self.num_classes:, :, :]  # [B, 4*(reg_max+1), H, W]
            
            # 生成锚点
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float() + 0.5
            anchor_grid = grid.view(-1, 2)
            stride_xy = torch.tensor(
                [stride_x, stride_y], device=device, dtype=grid.dtype
            ).view(1, 2).repeat(H * W, 1)
            anchor_points = anchor_grid * stride_xy
            
            # DFL 解码
            reg_decoded = self._decode_dfl(reg_pred)  # [B, 4, H, W]
            reg_decoded = reg_decoded.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4]
            
            anchor_view = anchor_grid.view(1, H, W, 2)
            x1 = (anchor_view[..., 0] - reg_decoded[..., 0]) * stride_x
            y1 = (anchor_view[..., 1] - reg_decoded[..., 1]) * stride_y
            x2 = (anchor_view[..., 0] + reg_decoded[..., 2]) * stride_x
            y2 = (anchor_view[..., 1] + reg_decoded[..., 3]) * stride_y
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            # reshape
            bboxes = bboxes.view(B, -1, 4)
            scores = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            dist = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4 * (self.reg_max + 1))
            
            pred_bboxes_list.append(bboxes)
            pred_scores_list.append(scores)
            pred_dist_list.append(dist)
            anchor_points_list.append(anchor_points)
            stride_list.append(stride_xy)
        
        # 拼接所有检测层
        pred_bboxes = torch.cat(pred_bboxes_list, dim=1)
        pred_scores = torch.cat(pred_scores_list, dim=1)
        pred_dist = torch.cat(pred_dist_list, dim=1)
        anchor_points = torch.cat(anchor_points_list, dim=0)
        strides = torch.cat(stride_list, dim=0)
        
        # 处理 targets
        # targets: [N, 6] (batch_idx, cls, cx, cy, w, h) 归一化
        gt_labels = []
        gt_bboxes = []
        mask_gt = []
        
        for b in range(batch_size):
            mask = targets[:, 0] == b
            t = targets[mask]
            
            if len(t) == 0:
                gt_labels.append(torch.zeros(0, 1, device=device))
                gt_bboxes.append(torch.zeros(0, 4, device=device))
                mask_gt.append(torch.zeros(1, dtype=torch.bool, device=device))
            else:
                cls = t[:, 1:2]
                cx, cy, w, h = t[:, 2], t[:, 3], t[:, 4], t[:, 5]
                
                # 归一化 -> 像素坐标
                img_h, img_w = img_size
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                
                gt_labels.append(cls)
                gt_bboxes.append(torch.stack([x1, y1, x2, y2], dim=1))
                mask_gt.append(torch.ones(len(t), dtype=torch.bool, device=device))
        
        # 填充到相同长度
        max_gt = max(len(g) for g in gt_labels)
        if max_gt == 0:
            max_gt = 1
        
        gt_labels_padded = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes_padded = torch.zeros(batch_size, max_gt, 4, device=device)
        mask_gt_padded = torch.zeros(batch_size, max_gt, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            n = len(gt_labels[b])
            if n > 0:
                gt_labels_padded[b, :n] = gt_labels[b]
                gt_bboxes_padded[b, :n] = gt_bboxes[b]
                mask_gt_padded[b, :n] = mask_gt[b]
        
        # 目标分配
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.sigmoid(),
            pred_bboxes,
            anchor_points,
            gt_labels_padded,
            gt_bboxes_padded,
            mask_gt_padded
        )
        
        # 计算损失
        # 分类损失
        normalizer = target_scores.sum().clamp(min=1.0)
        loss_cls = self.bce(pred_scores, target_scores).sum() / normalizer
        
        # 定位损失 + DFL
        if fg_mask.sum() > 0:
            loss_box, loss_dfl = self.bbox_loss(
                pred_dist, pred_bboxes, anchor_points, strides,
                target_bboxes, target_scores, fg_mask
            )
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_dfl = torch.tensor(0.0, device=device)
        
        # 加权求和
        loss = self.box_weight * loss_box + \
               self.cls_weight * loss_cls + \
               self.dfl_weight * loss_dfl
        
        loss_items = {
            'loss': loss.item(),
            'loss_box': loss_box.item(),
            'loss_cls': loss_cls.item(),
            'loss_dfl': loss_dfl.item(),
        }
        
        return loss, loss_items
    
    def _decode_dfl(self, reg_pred):
        """DFL 解码"""
        B, C, H, W = reg_pred.shape
        reg_pred = reg_pred.view(B, 4, self.reg_max + 1, H, W)
        reg_pred = reg_pred.permute(0, 1, 3, 4, 2).contiguous()
        reg_pred = F.softmax(reg_pred, dim=-1)
        
        project = torch.arange(self.reg_max + 1, device=reg_pred.device).float()
        reg_decoded = (reg_pred * project).sum(dim=-1)
        
        return reg_decoded


def build_loss(model, cfg):
    """构建损失函数"""
    return v8DetectionLoss(model, cfg)
