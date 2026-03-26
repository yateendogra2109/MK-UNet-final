import os, argparse, logging, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, AvgMeter
from vis_utils import save_training_vis, save_validation_vis

def structure_loss(pred, mask):
    """
    Weighted BCE + weighted IoU loss.
    """
    mask = mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_s = torch.sigmoid(pred)
    inter = ((pred_s * mask) * weit).sum(dim=(2, 3))
    union = ((pred_s + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, args):
    model.train()
    loss_record = AvgMeter()

    for step, (images, gts) in enumerate(train_loader, start=1):
        images, gts = images.cuda(), gts.cuda()
        out = model(images)
        p1, p2, p3, p4 = out[0], out[1], out[2], out[3]

        loss = (structure_loss(p1, gts)
              + 0.5 * structure_loss(p2, gts)
              + 0.25 * structure_loss(p3, gts)
              + 0.125 * structure_loss(p4, gts))

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, args.clip)
        optimizer.step()

        loss_record.update(loss.item(), args.batch_size)

        if step % 50 == 0 or step == len(train_loader):
            logging.info(f'[Train] Epoch [{epoch:03d}/{args.epoch:03d}] Step [{step:04d}/{len(train_loader):04d}] Loss: {loss_record.show():.4f}')

    try:
        plot_path = save_training_vis(model, train_loader, epoch, args.run_id, args.plot_root, args.num_vis_samples, 'cuda')
        logging.info(f'[Vis] Train grid saved → {plot_path}')
    except Exception as e:
        logging.warning(f'[Vis] Train visualisation failed: {e}')

    return loss_record.show()

def val(val_loader, model, epoch, args):
    model.eval()
    dice_list = []
    with torch.no_grad():
        for images, gts in val_loader:
            images, gts = images.cuda(), gts.cuda()
            out = model(images)
            p1 = out[0] if isinstance(out, (list, tuple)) else out
            pred = (torch.sigmoid(p1) > 0.5).float()
            inter = (pred * gts).sum((1, 2, 3))
            union = pred.sum((1, 2, 3)) + gts.sum((1, 2, 3))
            dice = (2 * inter / (union + 1e-8)).mean().item()
            dice_list.append(dice)

    try:
        plot_path = save_validation_vis(model, val_loader, epoch, args.run_id, args.plot_root, args.num_vis_samples, 'cuda')
        logging.info(f'[Vis] Val grid saved → {plot_path}')
    except Exception as e:
        logging.warning(f'[Vis] Val visualisation failed: {e}')

    return np.mean(dice_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet_GIK')
    parser.add_argument('--train_path', type=str, default='./data/polyp/target/ClinicDB/train')
    parser.add_argument('--test_path', type=str, default='./data/polyp/target/ClinicDB/test')
    parser.add_argument('--save_path', type=str, default='./snapshots/')
    parser.add_argument('--plot_root', default='./plots')
    parser.add_argument('--num_vis_samples', type=int, default=4)
    parser.add_argument('--ckpt_freq', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--channels', nargs='+', type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument('--kernels', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--expand_ratio', type=float, default=2.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)
    return parser.parse_args()

def main():
    args = get_args()
    run_id = time.strftime('%Y%m%d_%H%M%S')
    args.run_id = run_id

    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, f'{args.network}_{run_id}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info(f'Run ID        : {run_id}')
    logging.info(f'Args          : {vars(args)}')
    logging.info(f'Plot root     : {args.plot_root}')
    logging.info(f'Ckpt freq     : every {args.ckpt_freq} epochs')

    if args.network == 'MK_UNet_GIK':
        model = MK_UNet_GIK(in_channels=args.in_channels, num_classes=args.num_classes, channels=tuple(args.channels), kernels=tuple(args.kernels), expand_ratio=args.expand_ratio)
    elif args.network == 'MK_UNet' and MK_UNet is not None:
        model = MK_UNet()
    else:
        raise ValueError(f'Unknown network: {args.network}')
    
    model.cuda()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'#Params       : {n_params:,}  ({n_params/1e6:.3f} M)')

    train_loader = get_loader(os.path.join(args.train_path, 'images/'), os.path.join(args.train_path, 'masks/'), args.batch_size, args.trainsize, args.num_workers, True)
    val_loader = get_loader(os.path.join(args.train_path, 'images/'), os.path.join(args.train_path, 'masks/'), 1, args.trainsize, args.num_workers, False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr)

    save_dir = os.path.join(args.save_path, args.network, run_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.plot_root, run_id, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.plot_root, run_id, 'val'), exist_ok=True)

    best_dice = 0.0

    for epoch in range(1, args.epoch + 1):
        train_loss = train(train_loader, model, optimizer, epoch, args)
        scheduler.step()

        if epoch % args.ckpt_freq == 0:
            ckpt_path = os.path.join(save_dir, f'ckpt_epoch_{epoch:03d}.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'best_dice': best_dice, 'run_id': run_id, 'args': vars(args)}, ckpt_path)
            logging.info(f'[Ckpt] Periodic checkpoint → {ckpt_path}')

        if epoch % args.val_freq == 0:
            dice = val(val_loader, model, epoch, args)
            logging.info(f'[Val] Epoch [{epoch:03d}/{args.epoch:03d}] Dice = {dice:.4f}  (best = {best_dice:.4f})')
            if dice > best_dice:
                best_dice = dice
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save(model.state_dict(), best_path)
                logging.info(f'[Ckpt] Best model saved → {best_path}')

        torch.save(model.state_dict(), os.path.join(save_dir, 'latest.pth'))

    logging.info(f'Training complete. Best Val Dice = {best_dice:.4f}')
    logging.info(f'Run ID (use with test_polyp_gik.py): {run_id}')

if __name__ == '__main__':
    main()
