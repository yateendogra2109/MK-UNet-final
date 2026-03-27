import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mkunet_lomix_network import MK_UNet_LoMix, LoMix

try:
    from utils.dataloader import TestDataset
except ImportError:  # pragma: no cover - compatibility with this checkout
    from torchvision import transforms

    class TestDataset(object):
        """Compatibility test dataset wrapper matching the expected API.

        Args:
            image_root: str, directory containing test RGB images.
            gt_root: str, directory containing binary mask images.
            testsize: int, square resize resolution.
        """

        def __init__(self, image_root, gt_root, testsize):
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
            self.testsize = testsize
            self.index = 0
            self.size = len(self.images)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((testsize, testsize)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        def load_data(self):
            image_path = self.images[self.index]
            gt_path = self.gts[self.index]
            image = Image.open(image_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")
            tensor = self.transform(image).unsqueeze(0)
            name = os.path.basename(image_path)
            if name.lower().endswith(".jpg"):
                name = name.rsplit(".", 1)[0] + ".png"
            self.index += 1
            return tensor, gt, name


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Dice = (2*|P∩G| + 1) / (|P| + |G| + 1) with smoothing."""
    inter = (pred_bin & gt_bin).sum()
    return (2.0 * inter + 1.0) / (pred_bin.sum() + gt_bin.sum() + 1.0)


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """IoU = (|P∩G| + 1) / (|P∪G| + 1) with smoothing."""
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    return (inter + 1.0) / (union + 1.0)


def mae_score(pred_prob: np.ndarray, gt_prob: np.ndarray) -> float:
    """Mean absolute error between float predictions and float masks."""
    return float(np.mean(np.abs(pred_prob - gt_prob)))


def evaluate_dataset(
    model,
    data_path: str,
    testsize: int,
    save_path: str = None,
    dataset_name: str = "",
) -> dict:
    """Evaluate a test split, print summary metrics, and optionally save masks."""
    image_root = os.path.join(data_path, "images")
    gt_root = os.path.join(data_path, "masks")
    test_data = TestDataset(image_root, gt_root, testsize)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    dice_list, iou_list, mae_list = [], [], []

    with torch.no_grad():
        for _ in range(test_data.size):
            image, gt, name = test_data.load_data()
            image = image.cuda()

            p1, _, _, _ = model(image)
            pred = torch.sigmoid(p1)

            gt_np = np.array(gt, dtype=np.float32)
            if gt_np.max() > 1.0:
                gt_np = gt_np / 255.0
            height, width = gt_np.shape[:2]

            if pred.shape[-2:] != (height, width):
                pred = F.interpolate(
                    pred, size=(height, width), mode="bilinear", align_corners=False
                )

            pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
            pred_bin = (pred_np > 0.5).astype(np.uint8)
            gt_bin = (gt_np > 0.5).astype(np.uint8)

            dice_list.append(dice_score(pred_bin, gt_bin))
            iou_list.append(iou_score(pred_bin, gt_bin))
            mae_list.append(mae_score(pred_np, gt_np))

            if save_path is not None:
                Image.fromarray(pred_bin * 255).save(os.path.join(save_path, name))

    results = {
        "dice": float(np.mean(dice_list)) * 100.0,
        "iou": float(np.mean(iou_list)) * 100.0,
        "mae": float(np.mean(mae_list)),
        "n": len(dice_list),
    }
    print(
        f"[{dataset_name:>12s}]  "
        f"Dice={results['dice']:6.2f}%  "
        f"IoU={results['iou']:6.2f}%  "
        f"MAE={results['mae']:.4f}  "
        f"(n={results['n']})"
    )
    return results


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MK-UNet + LoMix on ClinicDB test split"
    )
    parser.add_argument("--run_id", required=True, help="Run ID from training")
    parser.add_argument("--network", default="MK_UNet_LoMix")
    parser.add_argument(
        "--ckpt_name",
        default="best.pth",
        help="Checkpoint file to load (best.pth | latest.pth | ckpt_epoch_NNN.pth)",
    )
    parser.add_argument(
        "--save_path",
        default="./snapshots/",
        help="Root directory where training checkpoints are stored",
    )
    parser.add_argument("--data_root", default="./data/polyp/target")
    parser.add_argument("--dataset", default="ClinicDB")
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument(
        "--pred_save_root",
        default="./results",
        help="Directory root for saved predicted binary masks",
    )
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--channels", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--expand_ratio", type=int, default=2)
    return parser.parse_args()


def main():
    args = get_args()
    model = MK_UNet_LoMix(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        channels=tuple(args.channels),
        expand_ratio=args.expand_ratio,
    ).cuda()
    lomix = LoMix(n_logits=4).cuda()

    ckpt_path = os.path.join(args.save_path, args.network, args.run_id, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    if "lomix" in ckpt:
        lomix.load_state_dict(ckpt["lomix"])

    model.eval()
    lomix.eval()

    print("=" * 60)
    print(f"  {args.network}  |  testsize={args.testsize}  |  run_id={args.run_id}")
    print("=" * 60)

    test_path = os.path.join(args.data_root, args.dataset, "test")
    if not os.path.isdir(test_path):
        raise NotADirectoryError(f"Test path not found: {test_path}")

    save_dir = None
    if args.pred_save_root is not None:
        save_dir = os.path.join(args.pred_save_root, args.run_id, args.dataset)

    evaluate_dataset(
        model=model,
        data_path=test_path,
        testsize=args.testsize,
        save_path=save_dir,
        dataset_name=args.dataset,
    )
    print("=" * 60)
    if save_dir is not None:
        print(f"Predicted masks saved to: {save_dir}")


if __name__ == "__main__":
    main()
