import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib not found. Install it with:\n"
        "  conda run -p /scratch/b23cs1001/mkunetenv pip install 'matplotlib>=3.5.0'")

from mkunet_gik_network import MK_UNet_GIK
try:
    from mkunet_network import MK_UNet
except ImportError:
    MK_UNet = None
from utils.dataloader_polyp import TestDataset
from vis_utils import save_test_vis

def dice_score(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0: return 1.0
    return 2.0 * inter / union

def iou_score(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    if union == 0: return 1.0
    return inter / union

def mae_score(pred, gt):
    return np.mean(np.abs(pred - gt))

def evaluate_dataset(model, data_path, test_size, run_id, save_path=None, dataset_name='', plot_root='./plots', vis_every_n=4):
    model.eval()
    image_root = os.path.join(data_path, 'images')
    gt_root = os.path.join(data_path, 'masks')
    test_data = TestDataset(image_root, gt_root, test_size)

    dice_list, iou_list, mae_list = [], [], []
    vis_images, vis_gts, vis_preds = [], [], []
    batch_idx = 0

    MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    with torch.no_grad():
        for i in range(test_data.size):
            image, gt, name = test_data.load_data()
            image = image.cuda()
            out = model(image)
            p1 = out[0] if isinstance(out, (list, tuple)) else out
            pred = torch.sigmoid(p1)

            gt_np = np.array(gt).squeeze()
            gt_np = (gt_np / 255.0 if gt_np.max() > 1.0 else gt_np).astype(np.float32)
            H, W = gt_np.shape[:2]

            if pred.shape[-2:] != (H, W):
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

            pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
            pred_bin = (pred_np > 0.5).astype(np.uint8)
            gt_bin = (gt_np > 0.5).astype(np.uint8)

            dice_list.append(dice_score(pred_bin, gt_bin))
            iou_list.append(iou_score(pred_bin, gt_bin))
            mae_list.append(mae_score(pred_np, gt_np))

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                out_img = Image.fromarray((pred_bin * 255).astype(np.uint8))
                if isinstance(name, bytes): name = name.decode('utf-8')
                out_img.save(os.path.join(save_path, name))

            img_tensor = image.squeeze(0).cpu()
            img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            img_denorm = np.clip(img_resized.numpy() * STD + MEAN, 0.0, 1.0)
            
            vis_images.append(torch.tensor(img_denorm.astype(np.float32)))
            vis_gts.append(gt_np)
            vis_preds.append(pred_np)

            if len(vis_images) == vis_every_n:
                try:
                    save_test_vis(vis_images, vis_gts, vis_preds, dataset_name, run_id, plot_root, batch_idx)
                except Exception as e:
                    print(f'[Vis] Test grid save failed: {e}')
                vis_images.clear(); vis_gts.clear(); vis_preds.clear()
                batch_idx += 1

    if len(vis_images) > 0:
        while len(vis_images) < 4:
            vis_images.append(vis_images[-1]); vis_gts.append(vis_gts[-1]); vis_preds.append(vis_preds[-1])
        try:
            save_test_vis(vis_images[:4], vis_gts[:4], vis_preds[:4], dataset_name, run_id, plot_root, batch_idx)
        except Exception as e:
            print(f'[Vis] Final test grid save failed: {e}')

    results = {'dice': np.mean(dice_list) * 100, 'iou': np.mean(iou_list) * 100, 'mae': np.mean(mae_list) * 100, 'n': len(dice_list)}
    print(f'[{dataset_name:>20s}]  Dice={results["dice"]:.2f}%  IoU={results["iou"]:.2f}%  MAE={results["mae"]:.4f}  (n={results["n"]})')
    return results

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet_GIK')
    parser.add_argument('--test_path', type=str, default='./data/polyp/TestDataset')
    parser.add_argument('--save_path', type=str, default='./snapshots/')
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='best.pth')
    parser.add_argument('--plot_root', default='./plots')
    parser.add_argument('--vis_every_n', type=int, default=4)
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--channels', nargs='+', type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument('--kernels', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--expand_ratio', type=float, default=2.0)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--pred_save_path', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', type=str, default=['CVC-ClinicDB', 'CVC-ColonDB', 'Kvasir', 'ETIS-LaribPolypDB', 'CVC-300', 'ETIS'])
    return parser.parse_args()

def main():
    args = get_args()

    if args.network == 'MK_UNet_GIK':
        model = MK_UNet_GIK(in_channels=args.in_channels, num_classes=args.num_classes, channels=tuple(args.channels), kernels=tuple(args.kernels), expand_ratio=args.expand_ratio)
    elif args.network == 'MK_UNet' and MK_UNet is not None:
        model = MK_UNet()
    else:
        raise ValueError(f'Unknown network: {args.network}')
    
    model.cuda()
    
    ckpt_path = os.path.join(args.save_path, args.network, args.run_id, args.ckpt_name)
    # HACK
    ckpt_path = './snapshots/MK_UNet_GIK/20260326_173805/best.pth'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
        
    print("="*60)
    print(f"  {args.network}  |  testsize={args.testsize}  |  run_id={args.run_id}")
    print("="*60)
    print(f"Test plots will be saved to: {args.plot_root}/{args.run_id}/test/")
    
    all_dice, all_iou = [], []
    for ds_name in args.datasets:
        ds_path = os.path.join(args.test_path, ds_name)
        if not os.path.isdir(ds_path):
            continue
            
        pred_save_path = os.path.join(args.pred_save_path, ds_name) if args.pred_save_path else None
        res = evaluate_dataset(model, ds_path, args.testsize, args.run_id, pred_save_path, ds_name, args.plot_root, args.vis_every_n)
        all_dice.append(res['dice'])
        all_iou.append(res['iou'])

    if all_dice:
        print("-" * 60)
        print(f"  Average Dice = {np.mean(all_dice):.2f}%   Average IoU = {np.mean(all_iou):.2f}%")
        print("=" * 60)
        print(f"Visualisations saved to: {args.plot_root}/{args.run_id}/test/")
        
if __name__ == '__main__':
    main()
