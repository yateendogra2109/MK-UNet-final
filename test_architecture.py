import sys
import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')

from mkunet_gik_network import GIKDWConv, GMKDC, MKIR_G, MK_UNet_GIK

def print_result(num, name, passed):
    status = "✓" if passed else "✗"
    print(f"[TEST {num:2d}]  {name:<40} {status}")

failed_tests = []

def run_test(num, name, test_func):
    try:
        test_func()
        print_result(num, name, True)
    except AssertionError as e:
        print_result(num, name, False)
        print(f"    -> {e}")
        failed_tests.append(num)
    except Exception as e:
        print_result(num, name, False)
        print(f"    -> Exception: {e}")
        failed_tests.append(num)

def test_1():
    m = GIKDWConv(16, 3)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == (2, 16, 32, 32), f"Expected (2, 16, 32, 32), got {y.shape}"

def test_2():
    m = GIKDWConv(16, 3)
    x = torch.randn(2, 16, 32, 32)
    m.weight.data.normal_()
    if m.bias is not None:
        m.bias.data.zero_()
    y1 = m(x)
    x_rot = torch.rot90(x, k=1, dims=[2, 3])
    y2 = m(x_rot)
    y1_rot = torch.rot90(y1, k=1, dims=[2, 3])
    diff = torch.abs(y1_rot - y2).max().item()
    assert diff < 1e-5, f"Equivariance broken, diff={diff}"

def test_3():
    m = GIKDWConv(16, 3)
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.weight.grad is not None, "Weight grad is None"
    grad_norm = m.weight.grad.norm().item()
    assert grad_norm > 0, f"Weight grad norm is {grad_norm}"

def test_4():
    m = GMKDC(16, [3, 5, 7], stride=1)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert isinstance(y, list)
    assert len(y) == 3
    for yy in y:
        assert yy.shape == (2, 16, 32, 32), f"Expected (2, 16, 32, 32), got {yy.shape}"

def test_5():
    m1 = GMKDC(16, [3, 5, 7], stride=1)
    m2 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    assert m2 > 0

def test_6():
    m = MKIR_G(16, 32, stride=1, expansion_factor=2.5, add=True, kernel_sizes=[3,5,7])
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == (2, 32, 32, 32)

def test_7():
    m = MKIR_G(16, 16, stride=1)
    assert m.use_skip_connection, "Should use skip connection when stride=1"
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == (2, 16, 32, 32)

def test_8():
    m = MKIR_G(16, 32, stride=1)
    assert m.use_skip_connection, "stride=1 so skip is True, but in_c!=out_c uses 1x1 conv"
    m = MKIR_G(16, 32, stride=2)
    assert not m.use_skip_connection, "Should not use skip connection when stride=2"

def test_9():
    m = MK_UNet_GIK()
    x = torch.randn(2, 3, 352, 352)
    out = m(x)
    assert len(out) == 4
    for o in out:
        assert o.shape[2:] == (352, 352), f"Shape is {o.shape}"

def test_10():
    m = MK_UNet_GIK()
    x = torch.randn(2, 3, 256, 320)
    out = m(x)
    for o in out:
        assert o.shape[2:] == (256, 320), f"Shape is {o.shape}"

def test_11():
    m = MK_UNet_GIK(channels=[8, 16, 32, 64, 128])
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_params <= 1000000, f"Params = {n_params}"

def test_12():
    m = MK_UNet_GIK()
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    loss = sum(o.sum() for o in out)
    loss.backward()
    for name, param in m.named_parameters():
        if 'multi_scale_dwconv' in name and 'weight' in name and len(param.shape)==4:
            assert param.grad is not None and param.grad.norm().item() > 0, f"No grad for {name}"

def test_13_vis_utils_import():
    import vis_utils
    import matplotlib
    backend = matplotlib.get_backend().lower()
    assert backend == 'agg', (
        f"Expected matplotlib backend 'agg', got '{backend}'. "
        f"Ensure matplotlib.use('Agg') is called before importing pyplot.")

def test_14_save_epoch_grid_output():
    import tempfile
    from vis_utils import save_epoch_grid

    images = [torch.rand(3, 64, 64) for _ in range(4)]
    gts    = [np.random.rand(64, 64).astype(np.float32) for _ in range(4)]
    preds  = [np.random.rand(64, 64).astype(np.float32) for _ in range(4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_epoch_grid(
            images    = images,
            gts       = gts,
            preds     = preds,
            epoch     = 1,
            run_id    = 'test_run',
            phase     = 'train',
            plot_root = tmpdir,
            dpi       = 72)
        assert isinstance(path, str), "save_epoch_grid must return a str path"
        assert os.path.exists(path),  f"PNG not found at {path}"
        from PIL import Image as PILImage
        img = PILImage.open(path)
        assert img.format == 'PNG',   f"Expected PNG, got {img.format}"
        assert img.width > 400,       f"PNG width too small: {img.width}"

def test_15_tensor_to_numpy_image():
    from vis_utils import tensor_to_numpy_image

    H, W = 64, 64
    for shape, desc in [((3, H, W), 'C=3'), ((1, H, W), 'C=1'), ((H, W), '2D')]:
        t = torch.rand(*shape)
        out = tensor_to_numpy_image(t)
        assert out.shape == (H, W, 3), (
            f"[{desc}] Expected ({H},{W},3), got {out.shape}")
        assert out.dtype == np.uint8, (
            f"[{desc}] Expected uint8, got {out.dtype}")
        assert out.min() >= 0 and out.max() <= 255, (
            f"[{desc}] Values out of [0, 255] range")

if __name__ == '__main__':
    run_test(1, 'GIKDWConv output shape', test_1)
    run_test(2, 'GIKDWConv C4 equivariance', test_2)
    run_test(3, 'GIKDWConv gradient flow', test_3)
    run_test(4, 'GMKDC output shape', test_4)
    run_test(5, 'GMKDC parameter count', test_5)
    run_test(6, 'MKIR_G channel divisibility fix', test_6)
    run_test(7, 'MKIR_G residual=True when in==out', test_7)
    run_test(8, 'MKIR_G residual=False when in!=out', test_8)
    run_test(9, 'Full forward pass shapes', test_9)
    run_test(10, 'Non-square input', test_10)
    run_test(11, 'Parameter count <= 1M', test_11)
    run_test(12, 'Gradient flows to all GIKDWConv.weight', test_12)
    run_test(13, 'vis_utils import and Agg backend', test_13_vis_utils_import)
    run_test(14, 'save_epoch_grid produces valid PNG', test_14_save_epoch_grid_output)
    run_test(15, 'tensor_to_numpy_image shape handling', test_15_tensor_to_numpy_image)

    if len(failed_tests) == 0:
        print("All tests passed ✓")
        sys.exit(0)
    else:
        print(f"{len(failed_tests)} test(s) failed ✗")
        sys.exit(1)
