# MK-UNet + LoMix

## Overview

This branch keeps the LoMix implementation as the only active training and
evaluation path. The main files are:

- `mkunet_lomix_network.py`
- `train_polyp_lomix.py`
- `test_polyp_lomix.py`
- `test_architecture.py`

The implementation adds additive LoMix supervision on top of fixed-weight deep
supervision and uses `CosineAnnealingWarmRestarts` for optimization.

## Environment

The prepared environment is located at:

```bash
/scratch/b23cs1001/mkunetenv
```

No extra packages are required beyond the project environment. All commands use
`conda run`.

## Dataset Layout

```text
MK-UNet/
├── mkunet_lomix_network.py
├── train_polyp_lomix.py
├── test_polyp_lomix.py
├── test_architecture.py
├── requirements.txt
├── utils/
│   ├── dataloader.py
│   └── utils.py
└── data/
    └── polyp/
        └── target/
            └── ClinicDB/
                ├── train/
                │   ├── images/
                │   └── masks/
                ├── val/
                │   ├── images/
                │   └── masks/
                └── test/
                    ├── images/
                    └── masks/
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore train_polyp_lomix.py
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore train_polyp_lomix.py \
        --channels 16 32 64 128 256 \
        --epoch 200 \
        --batch_size 8 \
        --T_0 20 \
        --T_mult 2 \
        --ckpt_freq 10
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p /scratch/b23cs1001/mkunetenv \
    python -W ignore test_polyp_lomix.py \
        --run_id 20250101_120000 \
        --ckpt_name best.pth \
        --pred_save_root ./results
```

## Unit Tests

```bash
conda run -p /scratch/b23cs1001/mkunetenv python test_architecture.py
```

Expected ending:

```text
All tests passed ✓
```

## Citation

```bibtex
@inproceedings{rahman2025mk,
  title     = {Mk-unet: Multi-kernel lightweight cnn for medical image segmentation},
  author    = {Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     = {1042--1051},
  year      = {2025}
}
```
