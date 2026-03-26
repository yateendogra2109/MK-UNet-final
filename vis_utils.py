"""
vis_utils.py
Visualisation utilities for MK-UNet-GIK training and evaluation.
Produces per-epoch PNG grids showing:
  Row 0: Input RGB image
  Row 1: Ground-truth binary mask (overlaid on image in green)
  Row 2: Predicted segmentation mask (sigmoid output, overlaid in red)
  Row 3: Difference map (|GT − Pred| in a diverging colormap)
Saved to: ./plots/<run_id>/train/epoch_<NNN>.png
           ./plots/<run_id>/val/epoch_<NNN>.png
"""

import os
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
except ImportError:
    raise ImportError(
        "matplotlib not found. Install it with:\n"
        "  conda run -p /scratch/b23cs1001/mkunetenv "
        "pip install 'matplotlib>=3.5.0'")

from PIL import Image

GT_OVERLAY_COLOR   = np.array([0, 255, 0],   dtype=np.uint8)   # green
PRED_OVERLAY_COLOR = np.array([255, 0, 0],   dtype=np.uint8)   # red
OVERLAY_ALPHA      = 0.45    # blending weight for mask overlay
DIFF_CMAP          = 'RdBu_r'  # diverging colormap for difference maps

def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW float tensor (values in [0,1]) to a HW3 uint8 numpy array.
    Handles both C=1 (mask) and C=3 (RGB) inputs.
    Clips values to [0, 1] before converting.
    """
    t = t.detach().cpu().float()
    if t.ndim == 4:
        t = t.squeeze(0)   # (1, C, H, W) → (C, H, W)
    if t.ndim == 2:
        t = t.unsqueeze(0)  # (H, W) → (1, H, W)
    t = t.permute(1, 2, 0).numpy()   # (H, W, C)
    t = np.clip(t, 0.0, 1.0)
    if t.shape[2] == 1:
        t = np.repeat(t, 3, axis=2)  # grayscale → RGB
    return (t * 255).astype(np.uint8)

def overlay_mask_on_image(
        image_np: np.ndarray,
        mask_np:  np.ndarray,
        color:    np.ndarray,
        alpha:    float = OVERLAY_ALPHA) -> np.ndarray:
    """
    Blend a binary mask onto an RGB image using additive colour blending.
    """
    mask_bool = (mask_np > 0.5).astype(np.float32)
    overlay   = np.zeros_like(image_np, dtype=np.float32)
    overlay[..., :] = color.astype(np.float32)
    mask_3d   = mask_bool[..., np.newaxis]
    blended   = (
        (1.0 - alpha * mask_3d) * image_np.astype(np.float32)
        + alpha * mask_3d * overlay
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

def make_difference_map(
        pred_np: np.ndarray,
        gt_np:   np.ndarray,
        cmap:    str = DIFF_CMAP) -> np.ndarray:
    """
    Compute absolute difference |pred − gt| and render as a coloured heatmap.
    """
    signed_diff  = pred_np.astype(np.float32) - gt_np.astype(np.float32)
    norm         = Normalize(vmin=-1.0, vmax=1.0)
    mapper       = get_cmap(cmap)
    rgba         = mapper(norm(signed_diff))   # (H, W, 4) float [0,1]
    rgb          = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

def save_epoch_grid(
        images:    list,
        gts:       list,
        preds:     list,
        epoch:     int,
        run_id:    str,
        phase:     str  = 'train',
        plot_root: str  = './plots',
        dpi:       int  = 120) -> str:
    """
    Save a (4 columns × 4 rows) PNG grid for one epoch.
    """
    assert len(images) == len(gts) == len(preds) == 4, (
        "save_epoch_grid requires exactly 4 samples per call")

    n_cols   = 4   # samples
    n_rows   = 4   # Input / GT overlay / Pred overlay / Diff map
    row_labels = ['Input', 'Ground Truth', 'Prediction', 'Difference\n(Pred−GT)']

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.2))
    fig.suptitle(
        f'Epoch {epoch:03d}  |  phase={phase}',
        fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(n_rows, n_cols,
                           figure=fig,
                           hspace=0.08, wspace=0.04)

    for col_idx in range(n_cols):
        img_np   = tensor_to_numpy_image(images[col_idx])
        gt_np    = gts[col_idx]
        pred_np  = preds[col_idx]

        gt_overlay   = overlay_mask_on_image(img_np, gt_np,   GT_OVERLAY_COLOR)
        pred_overlay = overlay_mask_on_image(img_np, pred_np, PRED_OVERLAY_COLOR)
        diff_map     = make_difference_map(pred_np, gt_np)

        panels = [img_np, gt_overlay, pred_overlay, diff_map]

        for row_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(panel)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'Sample {col_idx + 1}',
                             fontsize=9, pad=3)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx],
                              fontsize=9, rotation=90,
                              labelpad=6, va='center')

    save_dir = os.path.join(plot_root, run_id, phase)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path

def collect_vis_samples(
        model:       'nn.Module',
        data_loader: 'torch.utils.data.DataLoader',
        n_samples:   int = 4,
        device:      str = 'cuda') -> tuple:
    """
    Run the model in eval mode on the FIRST n_samples from data_loader
    and collect (images, gts, preds) for visualisation.
    """
    was_training = model.training
    model.eval()
    images_list, gts_list, preds_list = [], [], []

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to('cpu')
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to('cpu')

    try:
        with torch.no_grad():
            for images, gts in data_loader:
                images_dev = images.to(device)
                
                # Check what model returns
                out = model(images_dev)
                if isinstance(out, (list, tuple)):
                    p1 = out[0]
                else:
                    p1 = out
                    
                preds  = torch.sigmoid(p1)

                batch_sz = images.shape[0]
                for b in range(batch_sz):
                    if len(images_list) >= n_samples:
                        break
                    
                    img = images[b].cpu()
                    img = img * STD + MEAN
                    img = torch.clamp(img, 0.0, 1.0)
                    images_list.append(img)

                    # Extract the corresponding ground truth
                    # Handle both tensor and list formats
                    if isinstance(gts, list):
                        gt = gts[b] if b < len(gts) else gts[0]
                    else:
                        gt = gts[b]
                        
                    gt_np = np.array(gt).squeeze()
                    gt_np = gt_np / 255.0 if gt_np.max() > 1.0 else gt_np
                    gt_np = np.clip(gt_np.astype(np.float32), 0.0, 1.0)
                    gts_list.append(gt_np)

                    pred = preds[b].squeeze().cpu().numpy()
                    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
                    preds_list.append(pred)

                if len(images_list) >= n_samples:
                    break
    finally:
        if was_training:
            model.train()

    return images_list[:n_samples], gts_list[:n_samples], preds_list[:n_samples]

def save_training_vis(
        model:       'nn.Module',
        train_loader: 'torch.utils.data.DataLoader',
        epoch:       int,
        run_id:      str,
        plot_root:   str = './plots',
        n_samples:   int = 4,
        device:      str = 'cuda') -> str:
    """
    Convenience wrapper: collect 4 training samples + save grid.
    """
    images_list, gts_list, preds_list = collect_vis_samples(
        model, train_loader, n_samples=n_samples, device=device)
    return save_epoch_grid(
        images    = images_list,
        gts       = gts_list,
        preds     = preds_list,
        epoch     = epoch,
        run_id    = run_id,
        phase     = 'train',
        plot_root = plot_root)

def save_validation_vis(
        model:      'nn.Module',
        val_loader: 'torch.utils.data.DataLoader',
        epoch:      int,
        run_id:     str,
        plot_root:  str = './plots',
        n_samples:  int = 4,
        device:     str = 'cuda') -> str:
    """
    Convenience wrapper: collect 4 validation samples + save grid.
    """
    images_list, gts_list, preds_list = collect_vis_samples(
        model, val_loader, n_samples=n_samples, device=device)
    return save_epoch_grid(
        images    = images_list,
        gts       = gts_list,
        preds     = preds_list,
        epoch     = epoch,
        run_id    = run_id,
        phase     = 'val',
        plot_root = plot_root)

def save_test_vis(
        images:      list,
        gts:         list,
        preds:       list,
        dataset_name: str,
        run_id:      str,
        plot_root:   str = './plots',
        batch_idx:   int = 0,
        dpi:         int = 120) -> str:
    """
    Save a test-time grid for a given dataset and batch.
    """
    assert len(images) == len(gts) == len(preds) == 4, (
        "save_test_vis requires exactly 4 samples per call")

    n_cols   = 4   
    n_rows   = 4   
    row_labels = ['Input', 'Ground Truth', 'Prediction', 'Difference\n(Pred−GT)']

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.2))
    fig.suptitle(
        f'Dataset: {dataset_name}  |  Batch: {batch_idx:04d}',
        fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(n_rows, n_cols,
                           figure=fig,
                           hspace=0.08, wspace=0.04)

    for col_idx in range(n_cols):
        img_np   = tensor_to_numpy_image(images[col_idx])
        gt_np    = gts[col_idx]
        pred_np  = preds[col_idx]

        gt_overlay   = overlay_mask_on_image(img_np, gt_np,   GT_OVERLAY_COLOR)
        pred_overlay = overlay_mask_on_image(img_np, pred_np, PRED_OVERLAY_COLOR)
        diff_map     = make_difference_map(pred_np, gt_np)

        panels = [img_np, gt_overlay, pred_overlay, diff_map]

        for row_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(panel)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'Sample {col_idx + 1}',
                             fontsize=9, pad=3)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx],
                              fontsize=9, rotation=90,
                              labelpad=6, va='center')

    save_dir = os.path.join(plot_root, run_id, 'test', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path_override = os.path.join(save_dir, f'batch_{batch_idx:04d}.png')
    
    fig.savefig(save_path_override, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return save_path_override
