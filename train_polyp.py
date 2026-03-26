import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

from mkunet_network import MODEL_REGISTRY, soft_cldice_loss
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, AvgMeter, cal_params_flops


# =============================================================================
# CHANGE 4: hybrid_loss  — 0.4*BCE + 0.3*IoU + 0.3*clDice
# clDice enforces polyp boundary topology/connectivity.
# BCE uses logits directly (numerically stable).
# IoU and clDice receive sigmoid-activated predictions.
# =============================================================================

def structure_loss(pred, mask):
    """Returns (bce_scalar, iou_scalar). pred = raw logits."""
    pred_sig = torch.sigmoid(pred)
    bce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    iou   = 1.0 - (inter + 1e-6) / (union - inter + 1e-6)
    return bce, iou.mean()

def hybrid_loss(pred, mask):
    """
    Combined loss for ClinicDB polyp segmentation.
    pred : raw logits  [B, 1, H, W]
    mask : binary GT   [B, 1, H, W]  float
    """
    bce, iou   = structure_loss(pred, mask)
    cldice     = soft_cldice_loss(mask, torch.sigmoid(pred))
    return 0.4 * bce + 0.3 * iou + 0.3 * cldice


# =============================================================================
# Metrics
# =============================================================================

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    p = predicted.contiguous().view(-1)
    g = labels.contiguous().view(-1)
    return (2.0 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)

def iou_metric(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    p = predicted.contiguous().view(-1)
    g = labels.contiguous().view(-1)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return (inter + smooth) / (union + smooth)


# =============================================================================
# Evaluation loop
# =============================================================================

def test(model, path, dataset, opt):
    data_path  = os.path.join(path, dataset)
    test_loader = get_loader(
        image_root=f'{data_path}/images/',
        gt_root=f'{data_path}/masks/',
        batchsize=opt.test_batchsize,
        trainsize=opt.img_size,
        shuffle=False, split='test',
        color_image=opt.color_image,
    )
    model.eval()
    DSC = IOU = total = 0.0

    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack
            images = images.cuda()
            gts    = gts.cuda().float()

            preds = model(images)
            pred  = preds[0] if isinstance(preds, list) else preds

            for i in range(len(images)):
                h_o = int(original_shapes[0][i])
                w_o = int(original_shapes[1][i])

                p = F.interpolate(pred[i].unsqueeze(0),
                                  size=(h_o, w_o),
                                  mode='bilinear',
                                  align_corners=False).sigmoid().squeeze()
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)

                g = F.interpolate(gts[i].unsqueeze(0),
                                  size=(h_o, w_o),
                                  mode='nearest').squeeze()

                p_bin = (p   >= 0.5).float()
                g_bin = (g   >= 0.2).float()

                DSC   += dice_coefficient(p_bin, g_bin).item()
                IOU   += iou_metric(p_bin, g_bin).item()
                total += 1

    return DSC / total, IOU / total, int(total)


# =============================================================================
# Training loop
# =============================================================================

def train(train_loader, model, optimizer, epoch, opt, run_id):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot

    t0          = time.time()
    loss_meter  = AvgMeter()
    size_rates  = [0.75, 1.0, 1.25]
    total_steps = len(train_loader)

    for step, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images_v = Variable(images).cuda()
            gts_v    = Variable(gts).float().cuda()

            if rate != 1.0:
                sz = int(round(opt.img_size * rate / 32) * 32)
                images_v = F.interpolate(images_v, size=(sz, sz),
                                         mode='bilinear', align_corners=True)
                gts_v    = F.interpolate(gts_v, size=(sz, sz), mode='nearest')

            out  = model(images_v)
            pred = out[0] if isinstance(out, list) else out
            loss = hybrid_loss(pred, gts_v)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1.0:
                loss_meter.update(loss.data, opt.batchsize)

        if step % 100 == 0 or step == total_steps:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}] '
                  f'Step [{step:04d}/{total_steps:04d}] '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} '
                  f'Loss: {loss_meter.show():.4f}')

    total_train_time += time.time() - t0

    # ── Save last checkpoint ─────────────────────────────────────────────────
    os.makedirs(opt.train_save, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(opt.train_save, f'{run_id}-last.pth'))

    # ── Validate + test ───────────────────────────────────────────────────────
    epoch_results = {}
    for split in ['val', 'test']:
        d, io, n = test(model, opt.test_path, split, opt)
        epoch_results[split] = d
        msg = (f'Epoch {epoch:03d} | {split:4s} | '
               f'Dice {d:.4f} | IoU {io:.4f} | N={n}')
        logging.info(msg)
        print(msg)
        dict_plot[split].append(d)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    if epoch_results['val'] > best:
        logging.info(
            f'Best model saved  val_dice {best:.4f} → {epoch_results["val"]:.4f}')
        print(
            f'### Best model saved  {best:.4f} → {epoch_results["val"]:.4f} ###')
        best                  = epoch_results['val']
        test_dice_at_best_val = epoch_results['test']
        torch.save(model.state_dict(),
                   os.path.join(opt.train_save, f'{run_id}-best.pth'))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    dataset_name = 'ClinicDB'

    parser = argparse.ArgumentParser()
    parser.add_argument('--network',       type=str,   default='MK_UNet',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--epoch',         type=int,   default=200)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--batchsize',     type=int,   default=8)
    parser.add_argument('--test_batchsize',type=int,   default=8)
    parser.add_argument('--img_size',      type=int,   default=352)
    parser.add_argument('--clip',          type=float, default=0.5)
    parser.add_argument('--color_image',   default=True)
    parser.add_argument('--augmentation',  default=True)
    parser.add_argument('--train_path',    type=str,
                        default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path',     type=str,
                        default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save',    type=str, default='')
    opt = parser.parse_args()

    NET_CONFIGS = {
        'MK_UNet_T': [4,  8,  16,  24,  32],
        'MK_UNet_S': [8,  16, 32,  48,  80],
        'MK_UNet':   [16, 32, 64,  96,  160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512],
    }

    chosen_net = opt.network if opt.network in MODEL_REGISTRY else 'MK_UNet'
    channels   = NET_CONFIGS[chosen_net]
    ModelClass = MODEL_REGISTRY[chosen_net]   # FIX B8: use chosen variant

    for run in range(1, 6):
        dict_plot            = {'val': [], 'test': []}
        best                 = 0.0
        test_dice_at_best_val = 0.0
        total_train_time     = 0.0

        ts     = time.strftime('%H%M%S')
        run_id = (f'{dataset_name}_{chosen_net}_bs{opt.batchsize}'
                  f'_lr{opt.lr}_e{opt.epoch}_aug{opt.augmentation}'
                  f'_run{run}_t{ts}')
        opt.train_save = f'./model_pth/{run_id}/'

        os.makedirs('logs',          exist_ok=True)
        os.makedirs(opt.train_save,  exist_ok=True)

        logging.basicConfig(
            filename=f'logs/train_log_{run_id}.log',
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            force=True,
        )

        # ── Build model ───────────────────────────────────────────────────────
        model = ModelClass(num_classes=1, in_channels=3, channels=channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f'Network: {chosen_net} | Channels: {channels}')
        cal_params_flops(model, opt.img_size, logging)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/',
            gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize,
            trainsize=opt.img_size,
            shuffle=True,
            augmentation=opt.augmentation,
            split='train',
            color_image=opt.color_image,
        )

        for epoch in range(1, opt.epoch + 1):
            train(train_loader, model, optimizer, epoch, opt, run_id)
            scheduler.step()

        summary = (
            f"\n{'='*50}\n"
            f"FINAL RESULTS: {run_id}\n"
            f"Best Val  Dice : {best:.4f}\n"
            f"Test Dice @ Best Val : {test_dice_at_best_val:.4f}\n"
            f"Total Train Time     : {total_train_time:.1f}s\n"
            f"{'='*50}"
        )
        print(summary)
        logging.info(summary)
