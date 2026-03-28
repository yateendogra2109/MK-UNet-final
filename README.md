-----

# MK-UNet with Sobel-Enhanced GAG for Polyp Segmentation

This repository contains the code and findings for a targeted modification to MK-UNet, an ultra-lightweight multi-kernel CNN designed for medical image segmentation. Our work specifically tackles the ClinicDB polyp segmentation benchmark, focusing on improving boundary detection with zero additional learned parameters.

## Abstract

Medical image segmentation requires a balance of high accuracy and computational efficiency, especially for point-of-care and edge devices. While heavyweight models like TransUNet require tens of millions of parameters, lightweight alternatives often sacrifice accuracy on challenging tasks like polyp segmentation.

Our project introduces a structural bias to the MK-UNet architecture by injecting a fixed Sobel edge signal into the Grouped Attention Gate (GAG). This modification structurally biases the attention gate toward boundary-aware attention. It successfully raised the test Dice score on the ClinicDB dataset from 0.93 to 0.94 (+0.006) without adding any trainable parameters.

## Architecture
![Architecture](https://i.imgur.com/IVYcQrj.png)

## The Approach: Sobel Edge Signal in GAG

Our final and most successful architectural change modifies the Grouped Attention Gate (GAG).

  * **Motivation:** The original GAG suppresses background regions but lacks explicit knowledge of polyp boundaries, which are thin and diagnostically critical.
  * **Implementation:** We compute a single-channel L2 Sobel magnitude map on-the-fly.
  * **Integration:** This edge map is concatenated to the encoder skip feature before the W\_x branch processes it.
  * **Parameter Cost:** Zero. The Sobel kernels (3x3) are stored as fixed, non-learned register buffers.

### Investigated Ablations (Unsuccessful Attempts)

During our research, we also explored several other modifications that ultimately decreased accuracy due to training instability on a small dataset (612 images):

  * **Attempt 1 (Combined Modifications):** Simultaneous implementation of G\_MKDC (C4-equivariant depthwise convolutions), Conditional Positional Encoding (CPE), Squeeze-and-Excitation (SE) Gates, and a clDice boundary loss. This resulted in a Dice score decrease of -0.014.
  * **Attempt 2 (G\_MKDC Only):** Isolating the C4-equivariant depthwise convolutions introduced a representational bottleneck, resulting in a -0.010 decrease from the baseline.

## Dataset and Training Details

  * **Dataset:** ClinicDB polyp dataset (612 total training images).
  * **Split:** 80/10/10 (train/val/test).
  * **Resolution:** 352x352.
  * **Hyperparameters:** 200 epochs, AdamW optimizer (learning rate = 5e-4, weight decay = 1e-4), cosine annealing, and a batch size of 8.
  * **Augmentation:** Multi-scale training with scales {0.75, 1.0, 1.25}.
  * **Loss Function:** Weighted BCE + weighted IoU (1:1).

## Results

### Ablation Study on ClinicDB Test Set

All results are averaged over five independent runs.

| Model/Variant | Dice | Δ |
| :--- | :--- | :--- |
| MK-UNet (baseline) | 0.930 | - |
| + G\_MKDC + CPE + SE + clDice | 0.920 | -0.014 |
| + G\_MKDC only | 0.923 | -0.010 |
| **+ Sobel-enhanced GAG (Ours)** | **0.940** | **+0.006** |

### Comparison with Base Paper Models

Our modification allows the standard 0.316M parameter MK-UNet to match or exceed the performance of much heavier variants.

| Method | \#Params | FLOPS | Dice |
| :--- | :--- | :--- | :--- |
| UNet | 34.53M | 65.53G | 0.9143 |
| TransUNet | 105.32M | 38.52G | 0.9318 |
| MK-UNet | 0.316M | 0.314G | 0.9348 |
| MK-UNet-M | 1.15M | 0.951G | 0.9367 |
| MK-UNet-L | 3.76M | 3.19G | 0.9385 |
| **MK-UNet + Sobel GAG (Ours)** | **0.38M** | **\~0.742G** | **0.940** |

## Contributors

  * **Yateen Dogra:** Investigated architectural changes to the MKIR encoder block, implemented G\_MKDC, CPE, and SE Gate, managed the repository, and conducted ablations.
  * **Aaditya Biswas:** Led loss function investigation, implemented clDice loss, tuned the hybrid loss weighting, and produced qualitative segmentation maps.
  * **Sravanth:** Designed and implemented the successful Sobel-enhanced GAG module, prepared result tables, and constructed architecture diagrams.

-----

Would you like me to help format a `LICENSE` file or write up a `CONTRIBUTING.md` guide to go alongside this README?
