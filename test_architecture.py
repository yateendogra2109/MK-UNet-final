import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mkunet_lomix_network import (
    channel_shuffle,
    structure_loss,
    MKDC,
    MKIR,
    ChannelAttention,
    SpatialAttention,
    MKIRA,
    GAG,
    SH,
    CMM,
    LoMix,
    MK_UNet_LoMix,
)


def rand(*shape):
    return torch.rand(*shape)


def test_01_channel_shuffle():
    """channel_shuffle output shape and non-trivial reordering"""
    tensor = torch.arange(24).float().view(1, 6, 2, 2)
    out = channel_shuffle(tensor, groups=3)
    assert out.shape == (1, 6, 2, 2)
    assert not torch.equal(out, tensor)


def test_02_mkdc_output_shape():
    """MKDC output shape preserved"""
    out = MKDC(channels=12, kernels=(3, 5, 7))(rand(2, 12, 16, 16))
    assert out.shape == (2, 12, 16, 16)


def test_03_mkdc_channel_guard():
    """MKDC rejects channels not divisible by 3"""
    try:
        MKDC(channels=10)
    except AssertionError:
        return
    raise AssertionError("Expected MKDC(channels=10) to raise AssertionError")


def test_04_mkir_residual_true():
    """MKIR residual=True when in_ch == out_ch"""
    module = MKIR(in_ch=16, out_ch=16, expand_ratio=2)
    assert module.residual is True
    out = module(rand(2, 16, 32, 32))
    assert out.shape == (2, 16, 32, 32)


def test_05_mkir_residual_false():
    """MKIR residual=False when in_ch != out_ch"""
    module = MKIR(in_ch=8, out_ch=16, expand_ratio=2)
    assert module.residual is False
    out = module(rand(2, 8, 32, 32))
    assert out.shape == (2, 16, 32, 32)


def test_06_mkir_expand_divisibility():
    """MKIR expanded channels are divisible by 3"""
    module = MKIR(in_ch=8, out_ch=16, expand_ratio=2)
    c_exp = module.expand[0].out_channels
    assert c_exp % 3 == 0


def test_07_full_forward_shapes():
    """Full forward pass output shapes (default channels)"""
    model = MK_UNet_LoMix()
    p1, p2, p3, p4 = model(rand(1, 3, 352, 352))
    assert p1.shape == (1, 1, 352, 352)
    assert p2.shape == (1, 1, 176, 176)
    assert p3.shape == (1, 1, 88, 88)
    assert p4.shape == (1, 1, 44, 44)


def test_08_non_square_input():
    """Non-square input forward pass"""
    model = MK_UNet_LoMix()
    p1, _, _, _ = model(rand(1, 3, 256, 320))
    assert p1.shape == (1, 1, 256, 320)


def test_09_param_count():
    """Parameter count ≤ 1 M (default channels)"""
    model = MK_UNet_LoMix()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000


def test_10_gradient_flow_mkdc():
    """Gradients flow to all MKDC branch weights"""
    model = MK_UNet_LoMix()
    outputs = model(rand(1, 3, 64, 64))
    outputs[0].mean().backward()
    for module in model.modules():
        if isinstance(module, MKDC):
            for branch in module.branches:
                dw_conv = branch[0]
                assert dw_conv.weight.grad is not None
                assert dw_conv.weight.grad.abs().sum().item() > 0


def test_11_cmm_output_length():
    """CMM output length equals 28"""
    cmm = CMM(n_logits=4)
    inputs = [rand(1, 1, 32, 32) for _ in range(4)]
    output = cmm(inputs)
    assert len(output) == 28


def test_12_lomix_weights():
    """LoMix weights strictly positive and sum to 1"""
    lomix = LoMix(n_logits=4)
    weight = lomix.weights
    assert weight.shape == (28,)
    assert bool((weight > 0).all())
    assert abs(weight.sum().item() - 1.0) < 1e-5


def test_13_lomix_loss_scalar():
    """LoMix loss is scalar with gradient to raw_weights"""
    lomix = LoMix(n_logits=4)
    model = MK_UNet_LoMix()
    images = rand(1, 3, 64, 64)
    mask = rand(1, 1, 64, 64)
    p1, p2, p3, p4 = model(images)
    loss = lomix([p1, p2, p3, p4], mask, structure_loss)
    assert loss.shape == torch.Size([])
    loss.backward()
    assert lomix.raw_weights.grad is not None
    assert lomix.raw_weights.grad.abs().sum().item() > 0


def test_14_lomix_concat_grads():
    """LoMix CMM concat_convs receive gradients"""
    lomix = LoMix(n_logits=4)
    model = MK_UNet_LoMix()
    images = rand(1, 3, 64, 64)
    mask = rand(1, 1, 64, 64)
    p1, p2, p3, p4 = model(images)
    loss = lomix([p1, p2, p3, p4], mask, structure_loss)
    loss.backward()
    for conv in lomix.cmm.concat_convs:
        assert conv.weight.grad is not None
        assert conv.weight.grad.abs().sum().item() > 0


def run_all_tests():
    tests = [
        test_01_channel_shuffle,
        test_02_mkdc_output_shape,
        test_03_mkdc_channel_guard,
        test_04_mkir_residual_true,
        test_05_mkir_residual_false,
        test_06_mkir_expand_divisibility,
        test_07_full_forward_shapes,
        test_08_non_square_input,
        test_09_param_count,
        test_10_gradient_flow_mkdc,
        test_11_cmm_output_length,
        test_12_lomix_weights,
        test_13_lomix_loss_scalar,
        test_14_lomix_concat_grads,
    ]
    n_failed = 0
    for idx, fn in enumerate(tests, start=1):
        label = (fn.__doc__ or fn.__name__).strip().split("\n")[0]
        try:
            fn()
            print(f"[TEST {idx:2d}]  {label:<52s}  ✓")
        except Exception as exc:
            print(f"[TEST {idx:2d}]  {label:<52s}  ✗")
            print(f"          → {exc}")
            n_failed += 1
    print()
    if n_failed == 0:
        print("All tests passed ✓")
        sys.exit(0)
    print(f"{n_failed} test(s) failed ✗")
    sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
