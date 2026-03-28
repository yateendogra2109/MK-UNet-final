
# MK-UNet with Sobel-Enhanced GAG for Polyp Segmentation

[cite_start]This repository contains the code and findings for a targeted modification to MK-UNet, an ultra-lightweight multi-kernel CNN designed for medical image segmentation[cite: 1, 6]. [cite_start]Our work specifically tackles the ClinicDB polyp segmentation benchmark, focusing on improving boundary detection with zero additional learned parameters[cite: 7, 11].

## Abstract
[cite_start]Medical image segmentation requires a balance of high accuracy and computational efficiency, especially for point-of-care and edge devices[cite: 14]. [cite_start]While heavyweight models like TransUNet require tens of millions of parameters, lightweight alternatives often sacrifice accuracy on challenging tasks like polyp segmentation[cite: 17, 18]. 

[cite_start]Our project introduces a structural bias to the MK-UNet architecture by injecting a fixed Sobel edge signal into the Grouped Attention Gate (GAG)[cite: 9, 73]. [cite_start]This modification structurally biases the attention gate toward boundary-aware attention[cite: 77]. [cite_start]It successfully raised the test Dice score on the ClinicDB dataset from 0.93 to 0.94 (+0.006) without adding any trainable parameters[cite: 10, 11].

## The Approach: Sobel Edge Signal in GAG
[cite_start]Our final and most successful architectural change modifies the Grouped Attention Gate (GAG)[cite: 73]. 

* [cite_start]**Motivation:** The original GAG suppresses background regions but lacks explicit knowledge of polyp boundaries, which are thin and diagnostically critical[cite: 75, 76].
* [cite_start]**Implementation:** We compute a single-channel L2 Sobel magnitude map on-the-fly[cite: 78].
* [cite_start]**Integration:** This edge map is concatenated to the encoder skip feature before the $W_x$ branch processes it[cite: 86].
* **Parameter Cost:** Zero. [cite_start]The Sobel kernels ($3\times3$) are stored as fixed, non-learned register buffers[cite: 89, 90].

### Investigated Ablations (Unsuccessful Attempts)
[cite_start]During our research, we also explored several other modifications that ultimately decreased accuracy due to training instability on a small dataset (612 images)[cite: 65]:
* [cite_start]**Attempt 1 (Combined Modifications):** Simultaneous implementation of G_MKDC (C4-equivariant depthwise convolutions), Conditional Positional Encoding (CPE), Squeeze-and-Excitation (SE) Gates, and a clDice boundary loss[cite: 8, 45, 61]. [cite_start]This resulted in a Dice score decrease of -0.014[cite: 64].
* [cite_start]**Attempt 2 (G_MKDC Only):** Isolating the C4-equivariant depthwise convolutions introduced a representational bottleneck, resulting in a -0.010 decrease from the baseline[cite: 66, 71, 72].

## Dataset and Training Details
* [cite_start]**Dataset:** ClinicDB polyp dataset (612 total training images)[cite: 39, 96].
* [cite_start]**Split:** 80/10/10 (train/val/test)[cite: 96].
* [cite_start]**Resolution:** 352x352[cite: 97].
* [cite_start]**Hyperparameters:** 200 epochs, AdamW optimizer ($lr=5\times10^{-4}$, weight decay of $10^{-4}$), cosine annealing, and a batch size of 8[cite: 42, 97].
* [cite_start]**Augmentation:** Multi-scale training with scales {0.75, 1.0, 1.25}[cite: 98].
* [cite_start]**Loss Function:** Weighted BCE + weighted IoU (1:1)[cite: 99].

## Results

### Ablation Study on ClinicDB Test Set
[cite_start]All results are averaged over five independent runs[cite: 102].

| Model/Variant | Dice | $\Delta$ |
| :--- | :--- | :--- |
| [cite_start]MK-UNet (baseline) [cite: 1] | [cite_start]0.930 [cite: 104] | [cite_start]- [cite: 104] |
| + G_MKDC + CPE + SE + clDice | [cite_start]0.92 [cite: 104] | [cite_start]-0.014 [cite: 104] |
| + G_MKDC only | [cite_start]0.9230 [cite: 104] | [cite_start]-0.010 [cite: 104] |
| + Sobel-enhanced GAG (Ours) | [cite_start]0.940 [cite: 104] | [cite_start]+0.0060 [cite: 104] |

### Comparison with Base Paper Models
[cite_start]Our modification allows the standard 0.316M parameter MK-UNet to match or exceed the performance of much heavier variants[cite: 139].

| Method | #Params | FLOPS | Dice |
| :--- | :--- | :--- | :--- |
| [cite_start]UNet [cite: 2] | [cite_start]34.53M [cite: 141] | [cite_start]65.53G [cite: 141] | [cite_start]0.9143 [cite: 141] |
| [cite_start]TransUNet [cite: 3] | [cite_start]105.32M [cite: 141] | [cite_start]38.52G [cite: 141] | [cite_start]0.9318 [cite: 141] |
| [cite_start]MK-UNet [cite: 1] | [cite_start]0.316M [cite: 141] | [cite_start]0.314G [cite: 141] | [cite_start]0.9348 [cite: 141] |
| [cite_start]MK-UNet-M [cite: 1] | [cite_start]1.15M [cite: 141] | [cite_start]0.951G [cite: 141] | [cite_start]0.9367 [cite: 141] |
| [cite_start]MK-UNet-L [cite: 1] | [cite_start]3.76M [cite: 141] | [cite_start]3.19G [cite: 141] | [cite_start]0.9385 [cite: 141] |
| **MK-UNet + Sobel GAG (Ours)** | [cite_start]**0.38M** [cite: 141] | [cite_start]$\approx0.742G$ [cite: 141] | [cite_start]**0.940** [cite: 141] |

## Contributors
* [cite_start]**Yateen Dogra:** Investigated architectural changes to the MKIR encoder block, implemented G_MKDC, CPE, and SE Gate, managed the repository, and conducted ablations[cite: 44, 45, 46, 47].
* [cite_start]**Aaditya Biswas:** Led loss function investigation, implemented clDice loss, tuned the hybrid loss weighting, and produced qualitative segmentation maps[cite: 48, 49, 50, 51].
* [cite_start]**Sravanth:** Designed and implemented the successful Sobel-enhanced GAG module, prepared result tables, and constructed architecture diagrams[cite: 52, 53, 54].

***

Would you like me to draft a skeleton `requirements.txt` or a `train.py` script structure to include alongside this README in your repository?
