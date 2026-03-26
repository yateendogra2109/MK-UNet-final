import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

from mkunet_network import MODEL_REGISTRY
from utils.dataloader_polyp import get_loader
from medpy.metric.binary import hd95


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
    return (inter + smooth) / (p.sum() + g.sum() - inter + smooth)

def get_binary_metrics(pred, gt):
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    try:
        hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy()) \
                 if pred.sum() > 0 and gt.sum() > 0 else 100.0
    except Exception:
        hd_val = 100.0
    return sensitivity, specificity, precision, hd_val


def test(model, path, dataset, opt, save_base=None):
    data_path   = os.path.join(path, dataset)
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
    results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f'Inference on {dataset}'):
            images, gts, original_shapes, names = pack
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

                p_bin = (p >= 0.5).float()
                g_bin = (g >= 0.2).float()

                d   = dice_coefficient(p_bin, g_bin).item()
                io  = iou_metric(p_bin, g_bin).item()
                sens, spec, prec, hd = get_binary_metrics(p_bin, g_bin)

                DSC   += d
                IOU   += io
                total += 1

                results.append({
                    'Name':        names[i],
                    'Dice':        round(d,    4),
                    'IoU':         round(io,   4),
                    'Sensitivity': round(sens, 4),
                    'Specificity': round(spec, 4),
                    'Precision':   round(prec, 4),
                    'HD95':        round(hd,   4),
                })

                if save_base:
                    mask_img = (p_bin.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), mask_img)

    return DSC / total, IOU / total, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',        type=str, required=True)
    parser.add_argument('--network',       type=str, default='MK_UNet',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--dataset_name',  type=str, default='ClinicDB')
    parser.add_argument('--split',         type=str, default='test')
    parser.add_argument('--img_size',      type=int, default=352)
    parser.add_argument('--test_batchsize',type=int, default=1)
    parser.add_argument('--color_image',   default=True)
    parser.add_argument('--test_path',     type=str,
                        default='./data/polyp/target/')
    opt = parser.parse_args()

    NET_CONFIGS = {
        'MK_UNet_T': [4,  8,  16,  24,  32],
        'MK_UNet_S': [8,  16, 32,  48,  80],
        'MK_UNet':   [16, 32, 64,  96,  160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512],
    }

    save_base  = f'./predictions_polyp/{opt.run_id}/{opt.dataset_name}/{opt.split}'
    os.makedirs(save_base, exist_ok=True)
    model_path = f'/scratch/b23cs1001/MK-UNet-final/model_pth/ClinicDB_MK_UNet_bs8_lr0.0005_e200_augTrue_run2_t025048/ClinicDB_MK_UNet_bs8_lr0.0005_e200_augTrue_run2_t025048-best.pth'

    opt.test_path = f'{opt.test_path}/{opt.dataset_name}/'

    channels   = NET_CONFIGS.get(opt.network, NET_CONFIGS['MK_UNet'])
    ModelClass = MODEL_REGISTRY.get(opt.network, MODEL_REGISTRY['MK_UNet'])
    model      = ModelClass(num_classes=1, in_channels=3, channels=channels).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda'),
                          strict=False)
    model.eval()

    mean_dice, mean_iou, results = test(
        model, opt.test_path, opt.split, opt, save_base=save_base)

    # ── Per-image Excel report ────────────────────────────────────────────────
    os.makedirs('results_polyp', exist_ok=True)
    df = pd.DataFrame(results)
    avg_row = df.mean(numeric_only=True).to_dict()
    avg_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    excel_path = f'results_polyp/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx'
    df.to_excel(excel_path, index=False)
    print(f'\nMean Dice : {mean_dice:.4f}')
    print(f'Mean IoU  : {mean_iou:.4f}')
    print(f'Report    : {excel_path}')

    # ── Append to master summary ──────────────────────────────────────────────
    summary_file = 'All_Runs_Summary_Polyp.xlsx'
    new_row = pd.DataFrame([{
        'run_id':      opt.run_id,
        'network':     opt.network,
        'dataset':     opt.dataset_name,
        'split':       opt.split,
        'dice':        mean_dice,
        'iou':         mean_iou,
        'sensitivity': avg_row.get('Sensitivity', 0),
        'specificity': avg_row.get('Specificity', 0),
        'precision':   avg_row.get('Precision',   0),
        'HD95':        avg_row.get('HD95',         0),
    }])

    if os.path.exists(summary_file):
        existing = pd.read_excel(summary_file)
        pd.concat([existing, new_row], ignore_index=True).to_excel(
            summary_file, index=False)
    else:
        new_row.to_excel(summary_file, index=False)

    print(f'Summary   : {summary_file}')
