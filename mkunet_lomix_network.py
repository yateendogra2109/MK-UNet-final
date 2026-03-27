import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels across groups. Torch 1.11 compatible."""
    batch_size, channels, height, width = x.shape
    assert channels % groups == 0, f"C={channels} not divisible by groups={groups}"
    x = x.view(batch_size, groups, channels // groups, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute weighted BCE plus weighted IoU loss for one logit map."""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_s = torch.sigmoid(pred)
    inter = ((pred_s * mask) * weit).sum(dim=(2, 3))
    union = ((pred_s + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class MKDC(nn.Module):
    """Apply three depth-wise kernels and shuffle the merged channels.

    Args:
        channels: int, total input/output channels; must be divisible by 3.
        kernels: tuple, three kernel sizes for the depth-wise branches.
    """

    def __init__(self, channels: int, kernels: tuple = (3, 5, 7)):
        super().__init__()
        assert channels % 3 == 0, "MKDC requires channels divisible by 3"
        assert len(kernels) == 3, "MKDC requires exactly 3 kernels"
        branch_channels = channels // 3
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=branch_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU6(inplace=True),
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — split channels into three equal groups.
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
        # Step 2 — apply one depth-wise branch per split.
        b1 = self.branches[0](x1)
        b2 = self.branches[1](x2)
        b3 = self.branches[2](x3)
        # Step 3 — merge and shuffle channels across the three groups.
        out = torch.cat([b1, b2, b3], dim=1)
        return channel_shuffle(out, groups=3)


class MKIR(nn.Module):
    """Apply expansion, multi-kernel depth-wise mixing, and projection.

    Args:
        in_ch: int, input channel count.
        out_ch: int, output channel count.
        expand_ratio: int, expansion ratio before MKDC.
        kernels: tuple, three kernel sizes for MKDC.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 2,
        kernels: tuple = (3, 5, 7),
    ):
        super().__init__()
        c_exp = int(math.ceil(float(in_ch * expand_ratio) / 3.0) * 3)
        self.residual = in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, c_exp, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_exp),
            nn.ReLU6(inplace=True),
        )
        self.mkdc = MKDC(channels=c_exp, kernels=kernels)
        self.project = nn.Sequential(
            nn.Conv2d(c_exp, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — expand channels with a point-wise projection.
        out = self.expand(x)
        # Step 2 — apply multi-kernel depth-wise processing.
        out = self.mkdc(out)
        # Step 3 — project back to the requested output width.
        out = self.project(out)
        # Step 4 — add the residual only when channel counts match.
        if self.residual:
            out = out + x
        return out


class ChannelAttention(nn.Module):
    """Apply squeeze-and-excitation style channel attention.

    Args:
        in_ch: int, number of input channels.
        reduction: int, bottleneck reduction ratio. Default is 16.
    """

    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        mid = max(in_ch // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — aggregate channel descriptors via average and max pooling.
        batch_size, channels, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(batch_size, channels))
        mx = self.mlp(self.max_pool(x).view(batch_size, channels))
        # Step 2 — produce channel weights and rescale the feature map.
        weight = self.sigmoid(avg + mx).view(batch_size, channels, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    """Apply channel-pooled spatial attention.

    Args:
        kernel_size: int, convolution kernel size. Default is 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — pool the feature map across channels with mean and max.
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        # Step 2 — build a spatial mask and apply it element-wise.
        weight = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class MKIRA(nn.Module):
    """Apply channel attention, spatial attention, then MKIR.

    Args:
        in_ch: int, input channel count.
        out_ch: int, output channel count.
        expand_ratio: int, expansion ratio for the embedded MKIR.
        kernels: tuple, three kernel sizes for MKDC.
        reduction: int, channel attention reduction ratio. Default is 16.
        spatial_kernel: int, spatial attention kernel size. Default is 7.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 2,
        kernels: tuple = (3, 5, 7),
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.ca = ChannelAttention(in_ch=in_ch, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)
        self.mkir = MKIR(
            in_ch=in_ch,
            out_ch=out_ch,
            expand_ratio=expand_ratio,
            kernels=kernels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — apply channel attention.
        x = self.ca(x)
        # Step 2 — refine the attended features spatially.
        x = self.sa(x)
        # Step 3 — process the result with MKIR.
        return self.mkir(x)


class GAG(nn.Module):
    """Gate encoder skip features using decoder context.

    Args:
        skip_ch: int, channels in the encoder skip feature.
        gate_ch: int, channels in the decoder gate feature.
        groups: int, grouped convolution factor for the skip path. Default is 1.
    """

    def __init__(self, skip_ch: int, gate_ch: int, groups: int = 1):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_ch, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Step 1 — upsample the low-resolution gate to the skip resolution.
        gate_up = F.interpolate(
            gate, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        # Step 2 — transform the gate into an attention mask.
        attn = self.gate_conv(gate_up)
        # Step 3 — filter the skip feature and modulate it with the mask.
        return self.skip_conv(skip) * attn


class SH(nn.Module):
    """Project a decoder feature map to a single raw logit channel.

    Args:
        in_ch: int, number of input channels.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CMM(nn.Module):
    """Generate pairwise logit mutations with four fusion operators.

    Args:
        n_logits: int, number of input logits. Default is 4.
    """

    def __init__(self, n_logits: int = 4):
        super().__init__()
        self.pairs = list(itertools.combinations(range(n_logits), 2))
        self.concat_convs = nn.ModuleList(
            [nn.Conv2d(2, 1, 1, bias=False) for _ in self.pairs]
        )
        for conv in self.concat_convs:
            nn.init.constant_(conv.weight, 0.5)

    def forward(self, logits: list) -> list:
        # Step 1 — begin with the original decoder logits.
        mutated = list(logits)
        # Step 2 — generate all four mutations for every logit pair.
        for index, (i, j) in enumerate(self.pairs):
            a, b = logits[i], logits[j]
            mutated.append((a + b) / 2.0)
            mutated.append(self.concat_convs[index](torch.cat([a, b], dim=1)))
            prod = (torch.sigmoid(a) * torch.sigmoid(b)).clamp(1e-6, 1.0 - 1e-6)
            mutated.append(torch.log(prod / (1.0 - prod)))
            mutated.append(torch.sigmoid(a) * b + torch.sigmoid(b) * a)
        return mutated


class LoMix(nn.Module):
    """Learn positive weights over original and mutated multi-scale logits.

    Args:
        n_logits: int, number of decoder logits. Default is 4.
    """

    def __init__(self, n_logits: int = 4):
        super().__init__()
        self.cmm = CMM(n_logits=n_logits)
        n_total = n_logits + len(self.cmm.pairs) * 4
        self.raw_weights = nn.Parameter(torch.zeros(n_total))

    @property
    def weights(self) -> torch.Tensor:
        """Return strictly positive LoMix weights normalized to sum to 1."""
        weight = F.softplus(self.raw_weights)
        return weight / (weight.sum() + 1e-8)

    def forward(self, logits: list, mask: torch.Tensor, loss_fn: callable) -> torch.Tensor:
        # Step 1 — upsample every decoder logit to the p1 resolution.
        height, width = logits[0].shape[2:]
        upsampled = [
            F.interpolate(p, size=(height, width), mode="bilinear", align_corners=False)
            if p.shape[2:] != (height, width)
            else p
            for p in logits
        ]
        # Step 2 — create the full list of original plus mutated logits.
        all_logits = self.cmm(upsampled)
        weight = self.weights
        # Step 3 — compute the weighted additive loss over all logits.
        total = sum(weight[k] * loss_fn(all_logits[k], mask) for k in range(len(all_logits)))
        return total


class MK_UNet_LoMix(nn.Module):
    """Build the MK-UNet backbone used with additive LoMix supervision.

    Args:
        in_channels: int, number of image input channels. Default is 3.
        num_classes: int, kept for API compatibility; expected to be 1.
        channels: tuple, encoder widths C1 through C5.
        kernels: tuple, three kernel sizes for MKDC.
        expand_ratio: int, inverted residual expansion ratio. Default is 2.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple = (8, 16, 32, 64, 128),
        kernels: tuple = (3, 5, 7),
        expand_ratio: int = 2,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("MK_UNet_LoMix expects num_classes=1")
        if len(channels) != 5:
            raise ValueError("channels must contain exactly 5 stage widths")

        c1, c2, c3, c4, c5 = channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e1 = MKIR(in_ch=in_channels, out_ch=c1, expand_ratio=expand_ratio, kernels=kernels)
        self.e2 = MKIR(in_ch=c1, out_ch=c2, expand_ratio=expand_ratio, kernels=kernels)
        self.e3 = MKIR(in_ch=c2, out_ch=c3, expand_ratio=expand_ratio, kernels=kernels)
        self.e4 = MKIR(in_ch=c3, out_ch=c4, expand_ratio=expand_ratio, kernels=kernels)
        self.e5 = MKIR(in_ch=c4, out_ch=c5, expand_ratio=expand_ratio, kernels=kernels)

        self.g4 = GAG(skip_ch=c4, gate_ch=c5)
        self.g3 = GAG(skip_ch=c3, gate_ch=c4)
        self.g2 = GAG(skip_ch=c2, gate_ch=c3)
        self.g1 = GAG(skip_ch=c1, gate_ch=c2)

        self.d4 = MKIRA(in_ch=c4, out_ch=c4, expand_ratio=expand_ratio, kernels=kernels)
        self.d3 = MKIRA(in_ch=c3, out_ch=c3, expand_ratio=expand_ratio, kernels=kernels)
        self.d2 = MKIRA(in_ch=c2, out_ch=c2, expand_ratio=expand_ratio, kernels=kernels)
        self.d1 = MKIRA(in_ch=c1, out_ch=c1, expand_ratio=expand_ratio, kernels=kernels)

        self.p4_head = SH(in_ch=c4)
        self.p3_head = SH(in_ch=c3)
        self.p2_head = SH(in_ch=c2)
        self.p1_head = SH(in_ch=c1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        # Step 1 — encode the image into five progressively coarser features.
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))

        # Step 2 — decode stage D4 and predict p4.
        d4 = self.d4(self.g4(e4, e5))
        p4 = self.p4_head(d4)

        # Step 3 — decode stage D3 and predict p3.
        d3 = self.d3(self.g3(e3, d4))
        p3 = self.p3_head(d3)

        # Step 4 — decode stage D2 and predict p2.
        d2 = self.d2(self.g2(e2, d3))
        p2 = self.p2_head(d2)

        # Step 5 — decode stage D1 and predict p1.
        d1 = self.d1(self.g1(e1, d2))
        p1 = self.p1_head(d1)

        return p1, p2, p3, p4
