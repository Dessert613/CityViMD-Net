"""
CityViMD-Net 训练脚本
"""

import os
import sys
import argparse
import json
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import build_model
from datasets.multimodal_dataset import (
    build_dataloader,
    load_config,
    load_fold_assignments,
)
from utils.loss import build_loss
from utils.metrics import evaluate, format_metric_summary, format_class_summary
from utils.ema import ModelEMA


def parse_args():
    parser = argparse.ArgumentParser(description='CityViMD-Net Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集路径（覆盖配置文件）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备（覆盖配置文件）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（覆盖配置文件）')
    parser.add_argument('--val-interval', type=int, default=None,
                        help='验证间隔（覆盖配置文件）')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='保存间隔（覆盖配置文件）')
    parser.add_argument('--patience', type=int, default=None,
                        help='早停耐心值（覆盖配置文件）')
    parser.add_argument('--no-ema', action='store_true',
                        help='关闭 EMA')
    parser.add_argument('--folds', type=str, default=None,
                        help='交叉验证折划分文件（tools/make_folds.py 生成）')
    parser.add_argument('--fold', type=int, default=None,
                        help='当前验证折编号（与 --folds 联用）')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    """构建优化器"""
    opt_cfg = cfg['train']['optimizer']
    
    # 区分权重衰减参数
    params = [], [], []  # no_weight_decay, weight_decay, bias
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            params[2].append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            params[0].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            params[1].append(v.weight)
    
    if opt_cfg['type'] == 'SGD':
        optimizer = optim.SGD(
            params[1],
            lr=opt_cfg['lr'],
            momentum=opt_cfg['momentum'],
            weight_decay=opt_cfg['weight_decay'],
            nesterov=True
        )
    elif opt_cfg['type'] == 'AdamW':
        optimizer = optim.AdamW(
            params[1],
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    # 添加 bias 参数（无权重衰减）
    optimizer.add_param_group({'params': params[2], 'weight_decay': 0.0})
    # 添加 BN 参数（无权重衰减）
    optimizer.add_param_group({'params': params[0], 'weight_decay': 0.0})
    
    return optimizer


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """构建学习率调度器"""
    sched_cfg = cfg['train']['scheduler']
    train_cfg = cfg['train']
    epochs = train_cfg['epochs']
    
    if sched_cfg['type'] == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < sched_cfg['warmup_epochs']:
                # Warmup
                return (epoch + 1) / sched_cfg['warmup_epochs']
            else:
                # Cosine decay
                progress = (epoch - sched_cfg['warmup_epochs']) / \
                          (epochs - sched_cfg['warmup_epochs'])
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                lrf = sched_cfg.get('lrf', 0.01)
                return lrf + (1.0 - lrf) * cosine
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_cfg['type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=epochs // 3, 
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")
    
    return scheduler


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, 
                    img_size=(640, 640), scaler=None, use_amp=False):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    loss_box_sum = 0
    loss_cls_sum = 0
    loss_dfl_sum = 0
    num_batches = 0
    train_start = time.time()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(images)

            # 计算损失
            loss, loss_items = loss_fn(predictions, labels, img_size)

        # 反向传播
        optimizer.zero_grad()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        max_norm = (
            loss_fn.cfg.get('train', {}).get('grad_clip', 0.0)
            if hasattr(loss_fn, 'cfg') else 0.0
        )
        if max_norm > 0:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # 统计
        total_loss += loss_items['loss']
        loss_box_sum += loss_items['loss_box']
        loss_cls_sum += loss_items['loss_cls']
        loss_dfl_sum += loss_items['loss_dfl']
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_items["loss"]:.4f}',
            'box': f'{loss_items["loss_box"]:.4f}',
            'cls': f'{loss_items["loss_cls"]:.4f}',
            'dfl': f'{loss_items["loss_dfl"]:.4f}',
        })
    
    avg_loss = total_loss / num_batches
    avg_box = loss_box_sum / num_batches
    avg_cls = loss_cls_sum / num_batches
    avg_dfl = loss_dfl_sum / num_batches
    epoch_time = time.time() - train_start

    return {
        'loss': avg_loss,
        'loss_box': avg_box,
        'loss_cls': avg_cls,
        'loss_dfl': avg_dfl,
        'epoch_time': epoch_time,
    }


def validate(model, dataloader, loss_fn, device, img_size=(640, 640), 
             num_classes=12):
    """验证"""
    model.eval()
    validate_start = time.time()
    # 计算验证集 mAP
    results = evaluate(
        model, dataloader, device,
        conf_thres=0.001,
        iou_thres=0.7,
        num_classes=num_classes,
        max_det=100,
        img_size=img_size
    )
    results['val_time'] = time.time() - validate_start
    
    return results


def save_checkpoint(model, optimizer, scheduler, epoch, best_map, is_best,
                    save_dir, filename='last.pt', ema_state_dict=None,
                    model_state_dict=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict or model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_map': best_map,
    }
    if ema_state_dict is not None:
        checkpoint['ema_state_dict'] = ema_state_dict
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best.pt')
        torch.save(checkpoint, best_path)


def main():
    args = parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 覆盖配置
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if args.data is not None:
        cfg['data']['root'] = args.data
    if args.device is not None:
        cfg['device']['gpu_id'] = int(args.device)
    if args.output_dir is not None:
        cfg['paths']['output_dir'] = args.output_dir
    if args.val_interval is not None:
        cfg['train']['val_interval'] = args.val_interval
    if args.save_interval is not None:
        cfg['train']['save_interval'] = args.save_interval
    if args.no_ema:
        cfg['train']['use_ema'] = False
    if args.patience is not None:
        cfg['train'].setdefault('early_stopping', {})
        cfg['train']['early_stopping']['patience'] = args.patience
    
    # 设置设备
    device = torch.device(f'cuda:{cfg["device"]["gpu_id"]}' 
                         if torch.cuda.is_available() and cfg['device']['cuda'] 
                         else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    set_seed(cfg['device']['seed'])
    
    # 创建输出目录
    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")
    print(f"Weights dir: {weights_dir}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"Logs dir: {log_dir}")
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 构建数据加载器
    print("Building dataloaders...")
    if (args.folds is None) != (args.fold is None):
        raise SystemExit("--folds and --fold must be provided together")
    if args.folds is not None:
        # 交叉验证：训练/验证折均来自训练目录
        assignments = load_fold_assignments(args.folds)
        train_ids = sorted(
            sid for sid, fold in assignments.items() if fold != args.fold
        )
        val_ids = sorted(
            sid for sid, fold in assignments.items() if fold == args.fold
        )
        if not val_ids:
            raise SystemExit(f"Fold {args.fold} has no validation samples")
        print(f"Cross-validation fold {args.fold}: "
              f"train={len(train_ids)}, val={len(val_ids)}")
        train_loader = build_dataloader(cfg, split='train', sample_ids=train_ids)
        val_loader = build_dataloader(
            cfg, split='train', sample_ids=val_ids, eval_mode=True
        )
    else:
        train_loader = build_dataloader(cfg, split='train')
        val_loader = build_dataloader(cfg, split='val')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # 构建模型
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    use_amp = bool(cfg.get('train', {}).get('amp', device.type == 'cuda'))
    ema_decay = float(cfg.get('train', {}).get('ema_decay', 0.9999))
    model_ema = ModelEMA(model, decay=ema_decay, device=device) if cfg['train'].get('use_ema', True) else None
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    
    # 构建损失函数
    loss_fn = build_loss(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 构建优化器
    optimizer = build_optimizer(model, cfg)
    
    # 构建学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    
    # 恢复训练
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if model_ema is not None and 'ema_state_dict' in checkpoint:
                model_ema.load_state_dict(checkpoint['ema_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint.get('best_map', 0.0)
            print(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # 训练循环
    epochs = cfg['train']['epochs']
    val_interval = cfg['train']['val_interval']
    save_interval = cfg['train']['save_interval']
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"{'='*60}")
    print(
        f"Validation every {val_interval} epochs, "
        f"checkpoint save every {save_interval} epochs, "
        f"EMA={'on' if model_ema is not None else 'off'}"
    )
    
    img_size = tuple(cfg['data']['img_size'])
    num_classes = cfg['data']['num_classes']
    patience = cfg['train'].get('early_stopping', {}).get('patience', 0)
    epochs_without_improvement = 0
    best_epoch = -1
    last_epoch = start_epoch - 1
    early_stopped = False
    metrics_path = os.path.join(output_dir, 'metrics.jsonl')
    
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        epoch_start = time.time()
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, img_size,
            scaler=scaler, use_amp=use_amp
        )
        if model_ema is not None:
            model_ema.update(model)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练指标
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/box', train_metrics['loss_box'], epoch)
        writer.add_scalar('train/cls', train_metrics['loss_cls'], epoch)
        writer.add_scalar('train/dfl', train_metrics['loss_dfl'], epoch)
        writer.add_scalar('train/lr', current_lr, epoch)
        writer.add_scalar('train/epoch_time', train_metrics['epoch_time'], epoch)
        writer.add_scalar('train/samples_per_sec', len(train_loader.dataset) / max(train_metrics['epoch_time'], 1e-6), epoch)
        writer.add_text(
            'train/summary',
            (
                f"loss={train_metrics['loss']:.4f}, "
                f"box={train_metrics['loss_box']:.4f}, "
                f"cls={train_metrics['loss_cls']:.4f}, "
                f"dfl={train_metrics['loss_dfl']:.4f}, "
                f"epoch_time={train_metrics['epoch_time']:.2f}s"
            ),
            epoch,
        )
        
        print(f"\nEpoch {epoch}/{epochs} - "
              f"loss: {train_metrics['loss']:.4f} - "
              f"box: {train_metrics['loss_box']:.4f} - "
              f"cls: {train_metrics['loss_cls']:.4f} - "
              f"dfl: {train_metrics['loss_dfl']:.4f} - "
              f"lr: {current_lr:.6f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            print(f"\nValidating...")
            prev_best_map = best_map
            eval_model = model_ema.ema if model_ema is not None else model
            val_results = validate(
                eval_model, val_loader, loss_fn, device, img_size, num_classes
            )
            
            writer.add_scalar('val/map50', val_results['map50'], epoch)
            writer.add_scalar('val/map75', val_results['map75'], epoch)
            writer.add_scalar('val/map50_95', val_results['map50_95'], epoch)
            writer.add_scalar('val/best_map50_95', best_map, epoch)
            writer.add_scalar('val/val_time', val_results['val_time'], epoch)
            writer.add_scalar('val/best_delta', val_results['map50_95'] - prev_best_map, epoch)
            writer.add_text(
                'val/summary',
                f"{format_metric_summary(val_results)} | best_mAP50-95: {best_map:.4f}",
                epoch,
            )
            writer.add_text(
                'val/class_summary',
                format_class_summary(val_results, cfg['data'].get('class_names')),
                epoch,
            )
            
            print(f"Val summary: {format_metric_summary(val_results)}")
            print(format_class_summary(val_results, cfg['data'].get('class_names')))
            
            # 保存最佳模型
            is_best = val_results['map50_95'] > best_map
            if is_best:
                best_map = val_results['map50_95']
                best_epoch = epoch
                epochs_without_improvement = 0
                print(f"New best mAP@50-95: {best_map:.4f}")
            else:
                epochs_without_improvement += val_interval
                print(
                    f"No improvement for {epochs_without_improvement} epochs "
                    f"(patience={patience})"
                )

            writer.add_scalar('val/improvement', val_results['map50_95'] - prev_best_map, epoch)
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_map, is_best,
                weights_dir, filename='last.pt',
                ema_state_dict=model_ema.state_dict() if model_ema is not None else None
            )
            if is_best and model_ema is not None:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_map, True,
                    weights_dir, filename='best.pt',
                    ema_state_dict=model_ema.state_dict(),
                    model_state_dict=model_ema.state_dict(),
                )
            print(f"Checkpoint saved. best_map={best_map:.4f}, patience_counter={epochs_without_improvement}")

            # 机器可读指标（tools/aggregate_results.py 汇总用）
            with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
                metrics_file.write(json.dumps({
                    'epoch': epoch,
                    'loss': train_metrics['loss'],
                    'loss_box': train_metrics['loss_box'],
                    'loss_cls': train_metrics['loss_cls'],
                    'loss_dfl': train_metrics['loss_dfl'],
                    'lr': current_lr,
                    'map50': val_results['map50'],
                    'map75': val_results['map75'],
                    'map50_95': val_results['map50_95'],
                    'best_map50_95': best_map,
                }) + '\n')
        
        # 定期保存
        elif (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_map, False,
                checkpoints_dir, filename=f'epoch_{epoch}.pt',
                ema_state_dict=model_ema.state_dict() if model_ema is not None else None
            )
            print(f"Checkpoint saved: epoch_{epoch}.pt")
        
        print(f"{'='*60}")

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: no mAP improvement for {patience} epochs")
            early_stopped = True
            break
    
    # 训练结束
    print(f"\nTraining completed!")
    print(f"Best mAP@50-95: {best_map:.4f}")

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump({
            'best_map50_95': best_map,
            'best_epoch': best_epoch,
            'epochs_completed': last_epoch + 1,
            'early_stopped': early_stopped,
            'seed': cfg['device']['seed'],
        }, summary_file, ensure_ascii=False, indent=2)
    print(f"Summary written: {summary_path}")
    
    writer.close()
    
    return best_map


if __name__ == '__main__':
    main()
