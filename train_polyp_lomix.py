import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from mkunet_lomix_network import MK_UNet_LoMix, LoMix, structure_loss

try:
    from utils.dataloader import get_loader
except ImportError:  # pragma: no cover - compatibility with this checkout
    try:
        from utils.dataloader_polyp import get_loader
    except ImportError:  # pragma: no cover - lightweight local fallback
        class _SimpleSegDataset(torch.utils.data.Dataset):
            def __init__(self, image_root, gt_root, trainsize):
                exts = (".jpg", ".jpeg", ".png", ".tif")
                self.images = sorted(
                    [
                        os.path.join(image_root, name)
                        for name in os.listdir(image_root)
                        if name.lower().endswith(exts)
                    ]
                )
                self.gts = sorted(
                    [
                        os.path.join(gt_root, name)
                        for name in os.listdir(gt_root)
                        if name.lower().endswith(exts)
                    ]
                )
                self.image_transform = transforms.Compose(
                    [
                        transforms.Resize((trainsize, trainsize)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                self.mask_transform = transforms.Compose(
                    [
                        transforms.Resize((trainsize, trainsize), interpolation=Image.NEAREST),
                        transforms.ToTensor(),
                    ]
                )

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                image = Image.open(self.images[index]).convert("RGB")
                mask = Image.open(self.gts[index]).convert("L")
                image = self.image_transform(image)
                mask = (self.mask_transform(mask) > 0.5).float()
                return image, mask

        def get_loader(
            image_root,
            gt_root,
            batchsize,
            trainsize,
            num_workers=4,
            augmentation=False,
            shuffle=False,
            split="train",
            **kwargs,
        ):
            del augmentation, split, kwargs
            dataset = _SimpleSegDataset(image_root, gt_root, trainsize)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

try:
    from utils.utils import clip_gradient, AvgMeter
except ImportError:  # pragma: no cover - lightweight local fallback
    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    class AvgMeter(object):
        def __init__(self, num=40):
            self.num = num
            self.reset()

        def reset(self):
            self.losses = []

        def update(self, val, n=1):
            del n
            self.losses.append(val if torch.is_tensor(val) else torch.tensor(val))

        def show(self):
            start = max(len(self.losses) - self.num, 0)
            return torch.mean(torch.stack(self.losses[start:]))


def train(train_loader, model, lomix, optimizer, epoch, args) -> float:
    """Run one training epoch and return the average total loss."""
    model.train()
    lomix.train()
    loss_record = AvgMeter()

    for step, (images, gts) in enumerate(train_loader, start=1):
        images = images.cuda()
        gts = gts.float().cuda()

        p1, p2, p3, p4 = model(images)

        loss_ds = (
            structure_loss(p1, gts)
            + 0.5 * structure_loss(
                p2,
                F.interpolate(gts, size=p2.shape[2:], mode="bilinear", align_corners=False),
            )
            + 0.25 * structure_loss(
                p3,
                F.interpolate(gts, size=p3.shape[2:], mode="bilinear", align_corners=False),
            )
            + 0.125 * structure_loss(
                p4,
                F.interpolate(gts, size=p4.shape[2:], mode="bilinear", align_corners=False),
            )
        )
        loss_lm = lomix([p1, p2, p3, p4], gts, structure_loss)
        loss = loss_ds + loss_lm

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, args.clip)
        optimizer.step()

        loss_record.update(loss.detach(), args.batch_size)

        if step % 50 == 0 or step == len(train_loader):
            logging.info(
                f"[Train] Epoch [{epoch:03d}/{args.epoch:03d}] "
                f"Step [{step:04d}/{len(train_loader):04d}] "
                f"Loss: {loss_record.show():.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return float(loss_record.show().item())


def val(val_loader, model, epoch, args) -> float:
    """Evaluate the model on the validation split and return mean Dice."""
    del epoch, args
    model.eval()
    dice_list = []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) >= 2:
                images, gts = batch[0], batch[1]
            else:
                raise ValueError("Validation loader must return images and masks")
            images = images.cuda()
            gts = gts.float().cuda()
            p1, _, _, _ = model(images)
            pred = (torch.sigmoid(p1) > 0.5).float()
            inter = (pred * gts).sum(dim=(1, 2, 3))
            union = pred.sum(dim=(1, 2, 3)) + gts.sum(dim=(1, 2, 3))
            dice = (2.0 * inter / (union + 1e-8)).mean().item()
            dice_list.append(dice)
    return float(np.mean(dice_list))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train MK-UNet with LoMix supervision on ClinicDB"
    )
    parser.add_argument(
        "--data_root",
        default="./data/polyp/target",
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--dataset",
        default="ClinicDB",
        help="Dataset folder name under data_root",
    )
    parser.add_argument(
        "--trainsize",
        type=int,
        default=352,
        help="Input image resize resolution (square)",
    )
    parser.add_argument("--network", default="MK_UNet_LoMix")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum LR (eta_min for scheduler)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.5, help="Gradient clip norm value")
    parser.add_argument("--val_freq", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument(
        "--T_0",
        type=int,
        default=10,
        help=(
            "CosineAnnealingWarmRestarts: initial restart period (epochs). "
            "Default 10: fast initial cycle for early convergence."
        ),
    )
    parser.add_argument(
        "--T_mult",
        type=int,
        default=2,
        help=(
            "CosineAnnealingWarmRestarts: cycle length multiplier. "
            "Default 2: each successive cycle is twice as long."
        ),
    )
    parser.add_argument(
        "--save_path",
        default="./snapshots/",
        help="Root directory for checkpoint saving",
    )
    parser.add_argument("--ckpt_freq", type=int, default=5, help="Save a full periodic checkpoint")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
        help="Encoder channel widths C1-C5",
    )
    parser.add_argument("--expand_ratio", type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id

    save_dir = os.path.join(args.save_path, args.network, run_id)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{args.network}_{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Run ID        : {run_id}")
    logging.info(f"Args          : {vars(args)}")

    model = MK_UNet_LoMix(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        channels=tuple(args.channels),
        expand_ratio=args.expand_ratio,
    ).cuda()
    lomix = LoMix(n_logits=4).cuda()

    n_net = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lomix = sum(p.numel() for p in lomix.parameters() if p.requires_grad)
    logging.info(f"Network params : {n_net:,}  ({n_net / 1e6:.3f} M)")
    logging.info(f"LoMix   params : {n_lomix:,}")

    ds_root = os.path.join(args.data_root, args.dataset)
    train_loader = get_loader(
        image_root=os.path.join(ds_root, "train", "images"),
        gt_root=os.path.join(ds_root, "train", "masks"),
        batchsize=args.batch_size,
        trainsize=args.trainsize,
        num_workers=args.num_workers,
        augmentation=True,
        shuffle=True,
        split="train",
    )
    val_loader = get_loader(
        image_root=os.path.join(ds_root, "val", "images"),
        gt_root=os.path.join(ds_root, "val", "masks"),
        batchsize=1,
        trainsize=args.trainsize,
        num_workers=args.num_workers,
        augmentation=False,
        shuffle=False,
        split="val",
    )
    logging.info(f"Train batches : {len(train_loader)}")
    logging.info(f"Val   batches : {len(val_loader)}")

    optimizer = optim.Adam(
        list(model.parameters()) + list(lomix.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.min_lr,
    )

    best_dice = 0.0

    for epoch in range(1, args.epoch + 1):
        train_loss = train(train_loader, model, lomix, optimizer, epoch, args)
        scheduler.step()
        logging.info(f"[Epoch] {epoch:03d} train_loss={train_loss:.4f}")

        if epoch % args.ckpt_freq == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "lomix": lomix.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "run_id": run_id,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logging.info(f"[Ckpt] Periodic -> {ckpt_path}")

        if epoch % args.val_freq == 0:
            dice = val(val_loader, model, epoch, args)
            logging.info(
                f"[Val] Epoch [{epoch:03d}/{args.epoch:03d}] "
                f"Dice = {dice:.4f}  (best = {best_dice:.4f})"
            )
            if dice > best_dice:
                best_dice = dice
                torch.save(
                    {"model": model.state_dict(), "lomix": lomix.state_dict()},
                    os.path.join(save_dir, "best.pth"),
                )
                logging.info(f"[Ckpt] Best -> {os.path.join(save_dir, 'best.pth')}")

        torch.save(
            {"model": model.state_dict(), "lomix": lomix.state_dict()},
            os.path.join(save_dir, "latest.pth"),
        )

    logging.info(f"Training complete. Best Val Dice = {best_dice:.4f}")
    logging.info(f"Run ID for evaluation: {run_id}")


if __name__ == "__main__":
    main()
