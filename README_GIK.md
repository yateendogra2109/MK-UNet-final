# MK-UNet-GIK extended implementation

Official Pytorch implementation of the paper [MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) published in ICCV 2025 CVAMD
[Md Mostafijur Rahman](https://mostafij-rahman.github.io/), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>

## Environment Setup

> **Note:** The conda environment at `/scratch/b23cs1001/mkunetenv` is assumed
> to already contain all packages from `requirements.txt` (torch 1.11.0+cu113,
> timm, etc.) as set up for the original MK-UNet repo.
> The only additional package required is `matplotlib`.

### Install matplotlib (one-time)

```bash
conda run -p /scratch/b23cs1001/mkunetenv pip install "matplotlib>=3.5.0"
```

### Verify environment

```bash
conda run -p /scratch/b23cs1001/mkunetenv python -c \
    "import torch; import timm; import matplotlib; \
     print('torch:', torch.__version__); \
     print('timm:', timm.__version__); \
     print('matplotlib:', matplotlib.__version__); \
     print('CUDA available:', torch.cuda.is_available())"
```

## Data preparation

- **ClinicDB dataset:**
Download the splited ClinicDB dataset from [Google Drive](https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P?usp=share_link) and move into './data/polyp/' folder.

- **ColonDB dataset:**
Download the splited ColonDB dataset from [Google Drive](https://drive.google.com/drive/folders/1u4_8dMztnEBUaX-w3XfUR3jXLBhpccPA?usp=share_link) and move into './data/polyp/' folder.

## Training

### Default training run

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore train_polyp_gik.py \
        --network     MK_UNet_GIK \
        --plot_root   ./plots \
        --ckpt_freq   5
```

### Custom run with larger model

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore train_polyp_gik.py \
        --network       MK_UNet_GIK \
        --channels      16 32 64 128 256 \
        --kernels       3 5 7 \
        --epoch         200 \
        --batch_size    8 \
        --ckpt_freq     10 \
        --plot_root     ./plots_large
```

### Training flags reference table

| Flag              | Default                        | Description                            |
|-------------------|--------------------------------|----------------------------------------|
| --network         | MK_UNet_GIK                    | Model class to use                     |
| --train_path      | ./data/polyp/TrainDataset      | Training dataset root                  |
| --test_path       | ./data/polyp/TestDataset       | Test dataset root                      |
| --save_path       | ./snapshots/                   | Checkpoint root                        |
| --plot_root       | ./plots                        | Visualisation PNG root                 |
| --epoch           | 100                            | Total training epochs                  |
| --batch_size      | 16                             | Training batch size                    |
| --lr              | 1e-4                           | Initial learning rate                  |
| --min_lr          | 1e-6                           | Minimum LR (cosine annealing)          |
| --weight_decay    | 1e-4                           | Adam weight decay                      |
| --clip            | 0.5                            | Gradient clip value                    |
| --trainsize       | 352                            | Input resize resolution                |
| --val_freq        | 5                              | Validate every N epochs                |
| --ckpt_freq       | 5                              | Save periodic checkpoint every N epochs|
| --num_vis_samples | 4                              | Samples per vis grid (keep at 4)       |
| --channels        | 8 16 32 64 128                 | Encoder channel widths (C1–C5)         |
| --kernels         | 3 5 7                          | Kernel sizes for multi-kernel branches |
| --expand_ratio    | 2                              | Inverted residual expansion ratio      |
| --num_workers     | 4                              | DataLoader worker threads              |

## Evaluation

### Standard evaluation

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore test_polyp_gik.py \
        --network   MK_UNet_GIK \
        --run_id    20250101_120000
```

### Evaluation with prediction mask saving

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore test_polyp_gik.py \
        --network         MK_UNet_GIK \
        --run_id          20250101_120000 \
        --pred_save_path  ./results/MK_UNet_GIK/ \
        --plot_root       ./plots \
        --vis_every_n     4
```

### Load a periodic checkpoint (e.g. epoch 50)

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore test_polyp_gik.py \
        --network    MK_UNet_GIK \
        --run_id     20250101_120000 \
        --ckpt_name  ckpt_epoch_050.pth
```

## Visualisation Output

### Directory layout

```
plots/
└── <run_id>/
    ├── train/
    │   ├── epoch_001.png    ← 4 training samples after epoch 1
    │   ├── epoch_002.png
    │   └── ...
    ├── val/
    │   ├── epoch_005.png    ← 4 validation samples at epoch 5 (val_freq)
    │   ├── epoch_010.png
    │   └── ...
    └── test/
        ├── CVC-ClinicDB/
        │   ├── batch_0000.png   ← samples 1–4
        │   ├── batch_0001.png   ← samples 5–8
        │   └── ...
        ├── Kvasir/
        └── ...
```

### Grid layout (4 columns × 4 rows per PNG)

Each column = one sample.

| Row | Content               | Description                                  |
|-----|-----------------------|----------------------------------------------|
| 0   | Input                 | Denormalised RGB image                       |
| 1   | Ground Truth          | GT binary mask overlaid in **green**         |
| 2   | Prediction            | Sigmoid prediction overlaid in **red**       |
| 3   | Difference (Pred−GT)  | Signed heatmap: red=FP, blue=FN, white=TP/TN |

### Interpreting the difference map

- **White/neutral** areas: prediction matches GT (correct)
- **Red** areas: model predicts polyp where none exists (false positives)
- **Blue** areas: model misses polyp that exists in GT (false negatives)

## Checkpoint Management

### Checkpoint types

| Filename                    | Saved when               | Contents                              |
|-----------------------------|--------------------------|---------------------------------------|
| `best.pth`                  | Val Dice improves        | `model.state_dict()` only             |
| `latest.pth`                | End of every epoch       | `model.state_dict()` only             |
| `ckpt_epoch_<NNN>.pth`      | Every 5 epochs (default) | Full state: model + optim + scheduler + metadata |

### Resuming training from a periodic checkpoint

The periodic checkpoints contain full training state. To resume:

```python
ckpt = torch.load('./snapshots/MK_UNet_GIK/<run_id>/ckpt_epoch_050.pth')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
scheduler.load_state_dict(ckpt['scheduler_state'])
start_epoch = ckpt['epoch'] + 1
best_dice   = ckpt['best_dice']
```

## Unit Tests

```bash
conda run -p /scratch/b23cs1001/mkunetenv python test_architecture.py
```

Expected output:

```
[TEST  1]  GIKDWConv output shape                    ✓
[TEST  2]  GIKDWConv C4 equivariance                 ✓
[TEST  3]  GIKDWConv gradient flow                   ✓
[TEST  4]  GMKDC output shape                        ✓
[TEST  5]  GMKDC parameter count                     ✓
[TEST  6]  MKIR_G channel divisibility fix           ✓
[TEST  7]  MKIR_G residual=True when in==out         ✓
[TEST  8]  MKIR_G residual=False when in!=out        ✓
[TEST  9]  Full forward pass shapes                  ✓
[TEST 10]  Non-square input                          ✓
[TEST 11]  Parameter count ≤ 1M                      ✓
[TEST 12]  Gradient flows to all GIKDWConv.weight    ✓
[TEST 13]  vis_utils import and Agg backend          ✓
[TEST 14]  save_epoch_grid produces valid PNG        ✓
[TEST 15]  tensor_to_numpy_image shape handling      ✓
All tests passed ✓
```

## Citations

``` 
@inproceedings{rahman2025mk,
  title     = {Mk-unet: Multi-kernel lightweight cnn for medical image segmentation},
  author    = {Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     = {1042--1051},
  year      = {2025}
}
```
