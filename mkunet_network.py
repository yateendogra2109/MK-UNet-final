import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from functools import partial

import timm
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

__all__ = ['MK_UNet', 'MK_UNet_T', 'MK_UNet_S', 'MK_UNet_M', 'MK_UNet_L']

# =============================================================================
# CHANGE 1: G-DWCONV  — C4-equivariant depthwise convolution
# One base kernel W is learned. Three rotated copies (90,180,270) are derived
# from it via torch.rot90. All 4 share weights — gradient flows back to W only.
# Applied to the 3×3 and 5×5 DWC branches inside G_MKDC.
# The 1×1 branch is rotation-invariant by definition and is left unchanged.
# =============================================================================

class GroupEquivariantDepthwiseConv(nn.Module):
    """
    C4-equivariant depthwise convolution.
    - Learns one base kernel W of shape [C_in, 1, K, K]
    - Derives 4 rotations: W0, R90(W), R180(W), R270(W)
    - Applies each rotation as a separate depthwise conv on the SAME input x
    - Concatenates outputs → [B, 4*C_in, H, W]

    Parameter count = C_in * K * K  (same as one standard DWC kernel)
    Coverage         = 4 orientations  (0°, 90°, 180°, 270°)
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Single learned base kernel — shape [C_in, 1, K, K]
        self.weight = nn.Parameter(
            torch.empty(in_channels, 1, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Build 4 rotated copies — all sharing the same underlying parameters
        w0 = self.weight
        w1 = torch.rot90(w0, 1, [2, 3])
        w2 = torch.rot90(w0, 2, [2, 3])
        w3 = torch.rot90(w0, 3, [2, 3])

        # Stack into [4*C_in, 1, K, K]
        weights = torch.cat([w0, w1, w2, w3], dim=0)

        # Apply all 4 rotated kernels depthwise to the same input
        # Repeat x along channel dim so each rotation gets its own copy
        x4 = x.repeat(1, 4, 1, 1)   
        # print(x4.shape)
        # print(weights.shape)
        # print(4 * self.in_channels)                     # [B, 4*C_in, H, W]
        out = F.conv2d( x4,
                   weights,
                    bias=None,
                    stride=(self.stride, self.stride),
                    padding=(self.padding, self.padding),
                    dilation=(1, 1),
                    groups=4 * self.in_channels)
        return out                                        # [B, 4*C_in, H, W]


# =============================================================================
# CHANGE 1 (cont.): G_MKDC  — replaces original MKDC
#
# Input: x of shape [B, ex_c, H, W]   (ex_c = expanded channels from MKIR PWC1)
# Design:
#   Branch 1  —  1×1 DWC on (ex_c // 3) channels      → outputs (ex_c // 3) channels
#   Branch 2  —  G-DWC 3×3 on (ex_c_b2 // 4) channels → outputs ex_c_b2 channels
#   Branch 3  —  G-DWC 5×5 on (ex_c_b3 // 4) channels → outputs ex_c_b3 channels
#
# Channel split guarantees: c_b1 + c_b2 + c_b3 == ex_c  (cat output == input size)
# This ensures pconv2 in MKIR sees the correct number of input channels.
# =============================================================================

class G_MKDC(nn.Module):
    """
    Group-Equivariant Multi-Kernel Depthwise Convolution.
    Replaces original MKDC. Output channel count == input channel count (exactly).

    Channel split:
      c_b1 = in_channels // 3                  → Branch 1 (1×1 DWC)
      c_b2 = in_channels // 3                  → Branch 2 (G-DWC 3×3 + proj)
      c_b3 = in_channels - c_b1 - c_b2         → Branch 3 (G-DWC 5×5 + proj)

    Branches 2 & 3: G-DWC produces 4×base channels, then a 1×1 conv projects
    back to exactly c_b2 / c_b3 channels. This guarantees cat output == in_channels
    for every possible channel count without rounding errors.
    """
    def __init__(self, in_channels, stride=1, activation='relu6'):
        super().__init__()
        self.c_b1 = in_channels // 3
        self.c_b2 = in_channels // 3
        self.c_b3 = in_channels - self.c_b1 - self.c_b2  # absorbs remainder

        # ── Branch 1: 1×1 DWC (rotation-invariant) ───────────────────────────
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.c_b1, self.c_b1, kernel_size=1, stride=stride,
                      padding=0, groups=self.c_b1, bias=False),
            nn.BatchNorm2d(self.c_b1),
            act_layer(activation, inplace=True),
        )

        # ── Branch 2: G-DWC 3×3 + 1×1 projection ────────────────────────────
        self.c2_base = max(self.c_b2 // 4, 1)
        self.branch2 = nn.Sequential(
            GroupEquivariantDepthwiseConv(self.c2_base, kernel_size=3,
                                          stride=stride, padding=1),
            nn.BatchNorm2d(self.c2_base * 4),
            act_layer(activation, inplace=True),
            # project 4*c2_base → c_b2  (restores exact channel count)
            nn.Conv2d(self.c2_base * 4, self.c_b2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_b2),
        )

        # ── Branch 3: G-DWC 5×5 + 1×1 projection ────────────────────────────
        self.c3_base = max(self.c_b3 // 4, 1)
        self.branch3 = nn.Sequential(
            GroupEquivariantDepthwiseConv(self.c3_base, kernel_size=5,
                                          stride=stride, padding=2),
            nn.BatchNorm2d(self.c3_base * 4),
            act_layer(activation, inplace=True),
            nn.Conv2d(self.c3_base * 4, self.c_b3, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_b3),
        )

    def forward(self, x):
        # Slice input: branch 1 gets c_b1 channels, branches 2 & 3 get their
        # respective base widths (the G-DWC expands internally, then projects back)
        x1 = x[:, :self.c_b1, :, :]
        x2 = x[:, self.c_b1:self.c_b1 + self.c2_base, :, :]
        x3 = x[:, self.c_b1 + self.c2_base:
                   self.c_b1 + self.c2_base + self.c3_base, :, :]

        out1 = self.branch1(x1)   # [B, c_b1, H, W]
        out2 = self.branch2(x2)   # [B, c_b2, H, W]
        out3 = self.branch3(x3)   # [B, c_b3, H, W]

        # cat total = c_b1 + c_b2 + c_b3 = in_channels  ✓
        return torch.cat([out1, out2, out3], dim=1)



# =============================================================================
# Utility helpers (unchanged from original)
# =============================================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':       return nn.ReLU(inplace)
    elif act == 'relu6':    return nn.ReLU6(inplace)
    elif act == 'leakyrelu':return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':    return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':     return nn.GELU()
    elif act == 'hswish':   return nn.Hardswish(inplace)
    else: raise NotImplementedError(f'activation [{act}] not found')

def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    assert C % groups == 0, f"channel_shuffle: C={C} not divisible by groups={groups}"
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, -1, H, W)

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


# =============================================================================
# Attention blocks (unchanged from original)
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes  = in_planes
        ratio           = min(ratio, in_planes)          # guard for tiny channels
        self.reduced    = max(in_planes // ratio, 1)
        self.out_planes = out_planes or in_planes
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.act        = act_layer(activation, inplace=True)
        self.fc1        = nn.Conv2d(in_planes, self.reduced, 1, bias=False)
        self.fc2        = nn.Conv2d(self.reduced, self.out_planes, 1, bias=False)
        self.sigmoid    = nn.Sigmoid()
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        avg = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        mx  = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class GroupedAttentionGate(nn.Module):
    """GAG — unchanged from original."""
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = act_layer(activation, inplace=True)
        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, g, x):
        psi = self.act(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


# =============================================================================
# Core block: MultiKernelInvertedResidualBlock
#
# Forward sequence:
#   x → PWC1 → BN → ReLU6
#             → G_MKDC       [CHANGE 1 — equivariant depthwise conv]
#             → channel_shuffle(groups=3)
#             → PWC2 → BN
#             → + residual(x)
#
# Only G_MKDC is changed vs original MK-UNet. CPE and SE are NOT present.
# =============================================================================

class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride,
                 expansion_factor=2, activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride          = stride
        self.in_c            = in_c
        self.out_c           = out_c
        self.use_skip        = (stride == 1)
        self.ex_c            = int(in_c * expansion_factor)

        # PWC1: expand channels
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True),
        )

        # G_MKDC: group-equivariant multi-kernel DWC  [CHANGE 1 — only change]
        self.g_mkdc = G_MKDC(self.ex_c, stride=stride, activation=activation)

        # PWC2: project back to out_c
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.ex_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )

        # Projection for skip when channel mismatch
        if self.use_skip and (in_c != out_c):
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        named_apply(partial(_init_weights, scheme='normal'), self)

    def forward(self, x):
        out = self.pconv1(x)
        out = self.g_mkdc(out)
        # Shuffle across the 3 branches (1×1, 3×3-G, 5×5-G)
        def safe_channel_shuffle(x, max_groups=3):
           C = x.shape[1]
           for g in range(max_groups, 0, -1):
              if C % g == 0:
                return channel_shuffle(x, g)
           return x

        out = safe_channel_shuffle(out, 3)
        out = self.pconv2(out)

        if self.use_skip:
            res = self.proj(x) if self.in_c != self.out_c else x
            out = out + res

        return out


def mk_irb_bottleneck(in_c, out_c, n, s,
                      expansion_factor=2, activation='relu6'):
    """Stack n MultiKernelInvertedResidualBlocks."""
    blocks = [MultiKernelInvertedResidualBlock(
        in_c, out_c, s,
        expansion_factor=expansion_factor,
        activation=activation)]
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1,
            expansion_factor=expansion_factor,
            activation=activation))
    return nn.Sequential(*blocks)


# =============================================================================
# MK-UNet model family
# All variants share the same forward() logic through _MKUNetBase.
# Only change vs original: MKDC → G_MKDC (C4-equivariant depthwise conv).
# CPE and SE are NOT present in this variant.
# =============================================================================

class _MKUNetBase(nn.Module):
    """
    Shared encoder-decoder logic for all MK-UNet variants.
    Subclasses only differ in channel widths.
    """
    def __init__(self, num_classes, in_channels, channels,
                 depths, expansion_factor, gag_kernel):
        super().__init__()
        C = channels

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder1 = mk_irb_bottleneck(in_channels, C[0], depths[0], 1,
                                          expansion_factor)
        self.encoder2 = mk_irb_bottleneck(C[0], C[1], depths[1], 1,
                                          expansion_factor)
        self.encoder3 = mk_irb_bottleneck(C[1], C[2], depths[2], 1,
                                          expansion_factor)
        self.encoder4 = mk_irb_bottleneck(C[2], C[3], depths[3], 1,
                                          expansion_factor)
        self.encoder5 = mk_irb_bottleneck(C[3], C[4], depths[4], 1,
                                          expansion_factor)

        # ── Grouped Attention Gates ───────────────────────────────────────────
        self.AG1 = GroupedAttentionGate(C[3], C[3], C[3] // 2,
                                        gag_kernel, C[3] // 2)
        self.AG2 = GroupedAttentionGate(C[2], C[2], C[2] // 2,
                                        gag_kernel, C[2] // 2)
        self.AG3 = GroupedAttentionGate(C[1], C[1], C[1] // 2,
                                        gag_kernel, C[1] // 2)
        self.AG4 = GroupedAttentionGate(C[0], C[0], C[0] // 2,
                                        gag_kernel, C[0] // 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder1 = mk_irb_bottleneck(C[4], C[3], 1, 1,
                                          expansion_factor)
        self.decoder2 = mk_irb_bottleneck(C[3], C[2], 1, 1,
                                          expansion_factor)
        self.decoder3 = mk_irb_bottleneck(C[2], C[1], 1, 1,
                                          expansion_factor)
        self.decoder4 = mk_irb_bottleneck(C[1], C[0], 1, 1,
                                          expansion_factor)
        self.decoder5 = mk_irb_bottleneck(C[0], C[0], 1, 1,
                                          expansion_factor)

        # ── MKIRA attention in decoder (CA + SA) ─────────────────────────────
        # CA ratios clamped so reduced channels >= 1
        self.CA1 = ChannelAttention(C[4], ratio=16)
        self.CA2 = ChannelAttention(C[3], ratio=16)
        self.CA3 = ChannelAttention(C[2], ratio=16)
        self.CA4 = ChannelAttention(C[1], ratio=8)
        self.CA5 = ChannelAttention(C[0], ratio=4)
        self.SA  = SpatialAttention(kernel_size=7)

        # ── Segmentation heads ────────────────────────────────────────────────
        # p1 = highest-resolution final output  (up 2×)
        # p2, p3, p4 = intermediate auxiliary heads (computed but not used
        #              during ClinicDB training — only p1 is supervised)
        self.out_p4 = nn.Conv2d(C[2], num_classes, kernel_size=1)  # after dec2
        self.out_p3 = nn.Conv2d(C[1], num_classes, kernel_size=1)  # after dec3
        self.out_p2 = nn.Conv2d(C[0], num_classes, kernel_size=1)  # after dec4
        self.out_p1 = nn.Conv2d(C[0], num_classes, kernel_size=1)  # after dec5

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.encoder1(x)
        t1 = F.max_pool2d(e1, 2, 2)

        e2 = self.encoder2(t1)
        t2 = F.max_pool2d(e2, 2, 2)

        e3 = self.encoder3(t2)
        t3 = F.max_pool2d(e3, 2, 2)

        e4 = self.encoder4(t3)
        t4 = F.max_pool2d(e4, 2, 2)

        e5 = self.encoder5(t4)
        out = F.max_pool2d(e5, 2, 2)   # bottleneck

        # ── Decoder stage 4 (bottleneck → 22×22) ────────────────────────────
        out = self.CA1(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder1(out),
                                   scale_factor=2, mode='bilinear',
                                   align_corners=False))
        t4  = self.AG1(g=out, x=t4)
        out = out + t4

        # ── Decoder stage 3 (→ 44×44) ────────────────────────────────────────
        out = self.CA2(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder2(out),
                                   scale_factor=2, mode='bilinear',
                                   align_corners=False))
        t3  = self.AG2(g=out, x=t3)
        out = out + t3

        # ── Decoder stage 2 (→ 88×88) ────────────────────────────────────────
        out = self.CA3(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder3(out),
                                   scale_factor=2, mode='bilinear',
                                   align_corners=False))
        t2  = self.AG3(g=out, x=t2)
        out = out + t2

        # ── Decoder stage 1 (→ 176×176) ──────────────────────────────────────
        out = self.CA4(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder4(out),
                                   scale_factor=2, mode='bilinear',
                                   align_corners=False))
        t1  = self.AG4(g=out, x=t1)
        out = out + t1

        # ── Final decoder block (→ 352×352) ───────────────────────────────────
        out = self.CA5(out) * out
        out = self.SA(out)  * out
        out = F.relu(F.interpolate(self.decoder5(out),
                                   scale_factor=2, mode='bilinear',
                                   align_corners=False))

        # p1 is the final full-resolution prediction (supervised during training)
        p1 = self.out_p1(out)

        # Return [p1] — single output supervised by hybrid_loss (BCE+IoU+clDice)
        return [p1]


# ── Public model variants ──────────────────────────────────────────────────────
# channels = [16,32,64,96,160]    for MK_UNet   (standard, ~0.322M params)
# channels = [4,8,16,24,32]       for MK_UNet_T (tiny,     ~0.028M params)
# channels = [8,16,32,48,80]      for MK_UNet_S (small,    ~0.096M params)
# channels = [32,64,128,192,320]  for MK_UNet_M (medium,   ~1.18M  params)
# channels = [64,128,256,384,512] for MK_UNet_L (large,    ~3.85M  params)

class MK_UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=None, depths=None,
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 96, 160]
        if depths is None:
            depths = [1, 1, 1, 1, 1]
        self.net = _MKUNetBase(num_classes, in_channels, channels,
                               depths, expansion_factor, gag_kernel)

    def forward(self, x):
        return self.net(x)


class MK_UNet_T(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=None, depths=None,
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        if channels is None:
            channels = [4, 8, 16, 24, 32]
        if depths is None:
            depths = [1, 1, 1, 1, 1]
        self.net = _MKUNetBase(num_classes, in_channels, channels,
                               depths, expansion_factor, gag_kernel)

    def forward(self, x):
        return self.net(x)


class MK_UNet_S(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=None, depths=None,
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        if channels is None:
            channels = [8, 16, 32, 48, 80]
        if depths is None:
            depths = [1, 1, 1, 1, 1]
        self.net = _MKUNetBase(num_classes, in_channels, channels,
                               depths, expansion_factor, gag_kernel)

    def forward(self, x):
        return self.net(x)


class MK_UNet_M(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=None, depths=None,
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 192, 320]
        if depths is None:
            depths = [1, 1, 1, 1, 1]
        self.net = _MKUNetBase(num_classes, in_channels, channels,
                               depths, expansion_factor, gag_kernel)

    def forward(self, x):
        return self.net(x)


class MK_UNet_L(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=None, depths=None,
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 384, 512]
        if depths is None:
            depths = [1, 1, 1, 1, 1]
        self.net = _MKUNetBase(num_classes, in_channels, channels,
                               depths, expansion_factor, gag_kernel)

    def forward(self, x):
        return self.net(x)


# Factory for train/test scripts
MODEL_REGISTRY = {
    'MK_UNet':   MK_UNet,
    'MK_UNet_T': MK_UNet_T,
    'MK_UNet_S': MK_UNet_S,
    'MK_UNet_M': MK_UNet_M,
    'MK_UNet_L': MK_UNet_L,
}
