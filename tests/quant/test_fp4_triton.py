import pytest
import torch
from reference_mxfp4 import qdq_mxfp4_torch

from qatfactory.quant.formats.fp4_triton import (
    HAS_TRITON,
    triton_fake_quantize_mxfp4,
    triton_fake_quantize_mxfp4_activation,
    triton_fake_quantize_nvfp4,
)
from qatfactory.quant.formats.mxfp4 import (
    _fake_quantize_mxfp4_activation_torch,
    _fake_quantize_mxfp4_torch,
    fake_quantize_mxfp4,
    fake_quantize_mxfp4_activation,
)
from qatfactory.quant.formats.nvfp4 import (
    NVFP4_SCALE_DENOMINATOR,
    _fake_quantize_nvfp4_torch,
    fake_quantize_nvfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _assert_pair_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_quantized, actual_mask = actual
    expected_quantized, expected_mask = expected
    assert torch.equal(actual_quantized, expected_quantized)
    assert torch.equal(actual_mask, expected_mask)


def _nvfp4_triton_pair(
    tensor: torch.Tensor,
    tensor_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_amax = tensor.detach().abs().amax().float() if tensor_amax is None else tensor_amax.detach().float()
    expected = _fake_quantize_nvfp4_torch(tensor, tensor_amax=tensor_amax)
    if bool(tensor_amax == 0):
        return expected
    global_scale = (tensor_amax / NVFP4_SCALE_DENOMINATOR).to(tensor.device).reshape(())
    actual = triton_fake_quantize_nvfp4(tensor, global_scale)
    _assert_pair_equal(actual, expected)
    return actual


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("magnitude", (1e-4, 1.0, 1e4))
def test_random_nvfp4_matches_torch(dtype, magnitude):
    torch.manual_seed(10)
    tensor = (torch.randn(257, 4096, device="cuda") * magnitude).to(dtype)
    _nvfp4_triton_pair(tensor)
    _nvfp4_triton_pair(tensor, tensor_amax=tensor.float().abs().amax() * 1.25)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("magnitude", (1e-4, 1.0, 1e4))
def test_random_mxfp4_matches_torch(dtype, magnitude):
    torch.manual_seed(11)
    tensor = (torch.randn(257, 4096, device="cuda") * magnitude).to(dtype)
    _assert_pair_equal(triton_fake_quantize_mxfp4(tensor), _fake_quantize_mxfp4_torch(tensor))


@pytest.mark.parametrize(
    "shape",
    (
        (4096, 4096),
        (1024, 4096),
        (12288, 4096),
        (4096, 12288),
    ),
)
def test_production_weight_shapes_match_torch(shape):
    torch.manual_seed(shape[0] + shape[1])
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    _nvfp4_triton_pair(tensor)
    _assert_pair_equal(triton_fake_quantize_mxfp4(tensor), _fake_quantize_mxfp4_torch(tensor))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape",
    (
        (4096, 100),
        (4096, 24),
        (1, 16),
        (1, 32),
        (3, 48),
    ),
)
def test_padding_and_tiny_shapes_match_torch(dtype, shape):
    torch.manual_seed(shape[0] * shape[1])
    tensor = torch.randn(shape, dtype=dtype, device="cuda")
    _nvfp4_triton_pair(tensor)
    _assert_pair_equal(triton_fake_quantize_mxfp4(tensor), _fake_quantize_mxfp4_torch(tensor))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_activation_paths_match_torch_on_flattened_3d(dtype):
    torch.manual_seed(12)
    activation = torch.randn(8, 1024, 4096, dtype=dtype, device="cuda")
    flattened = activation.reshape(-1, activation.shape[-1])

    input_amax = activation.float().abs().amax() * 0.75
    expected_nv = _fake_quantize_nvfp4_torch(flattened, tensor_amax=input_amax)
    global_scale = (input_amax / NVFP4_SCALE_DENOMINATOR).reshape(())
    actual_nv = triton_fake_quantize_nvfp4(flattened, global_scale)
    _assert_pair_equal(actual_nv, expected_nv)

    expected_mx, expected_mask = _fake_quantize_mxfp4_activation_torch(activation)
    actual_mx = triton_fake_quantize_mxfp4_activation(flattened).reshape_as(activation)
    assert torch.equal(actual_mx, expected_mx)
    assert expected_mask.all()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_e2m1_ties_negative_values_and_signed_zero(dtype):
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    tie_values = torch.cat((boundaries, -boundaries))

    nv_tensor = torch.zeros(1, 32, device="cuda")
    nv_tensor[0, : tie_values.numel()] = tie_values
    nv_tensor[0, 14] = -0.0
    nv_tensor[0, 15] = 6.0  # global_scale=1 and block_scale=1 in the tie block
    nv_tensor = nv_tensor.to(dtype)
    _nvfp4_triton_pair(nv_tensor, tensor_amax=torch.tensor(2688.0, device="cuda"))

    mx_tensor = torch.zeros(1, 32, device="cuda")
    mx_tensor[0, : tie_values.numel()] = tie_values
    mx_tensor[0, 14] = -0.0
    mx_tensor[0, -1] = 5.0  # EVEN scale is exactly 1
    mx_tensor = mx_tensor.to(dtype)
    _assert_pair_equal(triton_fake_quantize_mxfp4(mx_tensor), _fake_quantize_mxfp4_torch(mx_tensor))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_blocks_and_single_huge_elements(dtype):
    tensor = torch.zeros(3, 64, dtype=dtype, device="cuda")
    tensor[0, 0] = 1e4
    tensor[1, 31] = -1e4
    tensor[2, 32:] = torch.linspace(-4.0, 4.0, 32, device="cuda").to(dtype)

    _nvfp4_triton_pair(tensor)
    _assert_pair_equal(triton_fake_quantize_mxfp4(tensor), _fake_quantize_mxfp4_torch(tensor))


def test_fp32_subnormal_range_matches_torch():
    tensor = torch.full((3, 32), 1e-40, dtype=torch.float32, device="cuda")
    tensor[1].neg_()
    _nvfp4_triton_pair(tensor)
    _assert_pair_equal(triton_fake_quantize_mxfp4(tensor), _fake_quantize_mxfp4_torch(tensor))


def test_nvfp4_e4m3_subnormal_rounding_matches_torch():
    # With global_scale=1, each block's E4M3 cast input is block_amax / 6.
    # Straddle E4M3 subnormal half-way points in exact 2^-11 units.
    scale_inputs = torch.tensor(
        [0.5, 1.0, 1.5, 2.5],
        dtype=torch.float32,
        device="cuda",
    ) * (2.0**-9)
    tensor = torch.zeros(1, 64, dtype=torch.float32, device="cuda")
    tensor.view(4, 16)[:, 0] = scale_inputs * 6.0
    _nvfp4_triton_pair(tensor, tensor_amax=torch.tensor(2688.0, device="cuda"))


def test_nvfp4_zero_amax_paths_are_unchanged():
    tensor = torch.zeros(3, 24, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    implicit = fake_quantize_nvfp4(tensor)
    explicit = fake_quantize_nvfp4(tensor, tensor_amax=torch.zeros((), device="cuda"))
    assert torch.equal(implicit, tensor)
    assert torch.equal(explicit, tensor)
    explicit.sum().backward()
    assert torch.equal(tensor.grad, torch.ones_like(tensor))


def test_mxfp4_activation_zero_block_and_shape_gate():
    activation = torch.zeros(2, 3, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with torch.no_grad():
        activation[0, 0, 32] = 1e-30
    expected, expected_mask = _fake_quantize_mxfp4_activation_torch(activation)
    actual = fake_quantize_mxfp4_activation(activation)
    assert torch.equal(actual, expected)
    assert expected_mask.all()
    actual.sum().backward()
    assert torch.equal(activation.grad, torch.ones_like(activation))

    with pytest.raises(ValueError, match="divisible by 32"):
        fake_quantize_mxfp4_activation(torch.randn(4, 33, device="cuda"))


def test_mxfp4_activation_ceil_boundaries_match_torch():
    tensor = torch.zeros(4, 32, dtype=torch.float32, device="cuda")
    tensor[:, 0] = torch.tensor([1.5, 3.0, 6.0, 100.0], device="cuda")
    expected, expected_mask = _fake_quantize_mxfp4_activation_torch(tensor)
    actual = triton_fake_quantize_mxfp4_activation(tensor)
    assert torch.equal(actual, expected)
    assert expected_mask.all()


@pytest.mark.parametrize("quantizer", ("nvfp4", "mxfp4"))
def test_public_triton_and_torch_paths_have_identical_gradients(monkeypatch, quantizer):
    torch.manual_seed(13)
    source = torch.randn(4, 96, dtype=torch.bfloat16, device="cuda")
    upstream = torch.randn_like(source)

    triton_input = source.clone().requires_grad_()
    monkeypatch.delenv("QATFACTORY_FP4_DISABLE_TRITON", raising=False)
    if quantizer == "nvfp4":
        triton_output = fake_quantize_nvfp4(triton_input)
    else:
        triton_output = fake_quantize_mxfp4(triton_input)
    triton_output.backward(upstream)

    torch_input = source.clone().requires_grad_()
    monkeypatch.setenv("QATFACTORY_FP4_DISABLE_TRITON", "1")
    if quantizer == "nvfp4":
        torch_output = fake_quantize_nvfp4(torch_input)
    else:
        torch_output = fake_quantize_mxfp4(torch_input)
    torch_output.backward(upstream)

    assert torch.equal(triton_output, torch_output)
    assert torch.equal(triton_input.grad, torch_input.grad)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_mxfp4_triton_matches_vllm_reference(dtype):
    torch.manual_seed(14)
    tensor = torch.randn(8, 128, dtype=dtype, device="cuda")
    quantized, _ = triton_fake_quantize_mxfp4(tensor)
    assert torch.equal(quantized, qdq_mxfp4_torch(tensor))


def test_noncontiguous_input_is_supported():
    source = torch.randn(100, 64, dtype=torch.bfloat16, device="cuda").T
    assert not source.is_contiguous()
    _nvfp4_triton_pair(source)
    _assert_pair_equal(triton_fake_quantize_mxfp4(source), _fake_quantize_mxfp4_torch(source))
