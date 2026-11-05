import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from qatfactory.quant import (
    Q4KLinear,
    apply_fake_quant,
    configure_q4k_reorders,
    dequantize_q4_k,
    fake_quantize_q4_k,
    get_quant_format,
    pack_deploy_weight,
    quantize_q4_k,
    v_head_reorder_perm,
)
from qatfactory.quant.formats.q4_k import (
    QK_K,
    Q4K_BLOCK_BYTES,
    _as_superblocks,
    _encode_superblocks,
    _encode_superblocks_torch,
    _fp16,
    _round,
)
from qatfactory.quant.export.q4_k_gguf import (
    _map_targets_to_ggml,
    _validate_q4k_metadata,
)


def test_round_matches_ggml_nearest_int_half_to_even():
    values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    torch.testing.assert_close(_round(values), torch.tensor([-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0]))


def test_fp16_round_trip_is_round_to_nearest_even():
    # 1 + 2^-11 is the midpoint between representable fp16 values 1.0 and 1 + 2^-10.
    x = torch.tensor([1.0 + 2**-11])
    torch.testing.assert_close(_fp16(x), torch.tensor([1.0]))


def test_quantize_dequantize_is_self_consistent_and_matches_fake_quant():
    torch.manual_seed(0)
    weight = (torch.randn(6, 512) * 0.1).to(torch.bfloat16)

    result = quantize_q4_k(weight)
    dequantized = dequantize_q4_k(result.packed_weight, weight.shape[1])

    assert result.packed_weight.dtype == torch.uint8
    assert result.packed_weight.shape == (6, (512 // QK_K) * Q4K_BLOCK_BYTES)
    torch.testing.assert_close(dequantized.to(result.reconstruction.dtype), result.reconstruction, atol=0, rtol=0)
    torch.testing.assert_close(fake_quantize_q4_k(weight), result.reconstruction, atol=0, rtol=0)


def test_q4_k_decoder_is_bit_exact_with_gguf_reference():
    gguf_quants = pytest.importorskip("gguf.quants")
    import numpy as np

    torch.manual_seed(1)
    weight = (torch.randn(4, 768) * 0.2).to(torch.bfloat16)
    result = quantize_q4_k(weight)
    ours = dequantize_q4_k(result.packed_weight, weight.shape[1]).numpy()

    blocks = result.packed_weight.reshape(-1, Q4K_BLOCK_BYTES).numpy().astype(np.uint8)
    reference = gguf_quants.Q4_K.dequantize_blocks(blocks).reshape(weight.shape[0], weight.shape[1])
    assert float(np.abs(ours - reference).max()) == 0.0


def test_q4_k_ste_passes_gradient_and_masks_saturation():
    weight = torch.randn(2, 256, requires_grad=True)
    fake_quantize_q4_k(weight).sum().backward()
    assert bool(torch.isfinite(weight.grad).all())
    assert set(weight.grad.unique().tolist()) <= {0.0, 1.0}


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton Q4_K encoder needs CUDA.")


@cuda_only
def test_triton_encoder_is_self_consistent_and_matches_fake_quant_on_cuda():
    # On CUDA both fake_quantize_q4_k and quantize_q4_k dispatch to the fused Triton kernel,
    # so the packed blocks must dequantize value-for-value to the training reconstruction.
    torch.manual_seed(0)
    weight = (torch.randn(64, 1024, device="cuda") * 0.1).to(torch.bfloat16)

    result = quantize_q4_k(weight)
    dequantized = dequantize_q4_k(result.packed_weight, weight.shape[1])
    torch.testing.assert_close(dequantized.to(result.reconstruction.dtype), result.reconstruction, atol=0, rtol=0)
    torch.testing.assert_close(fake_quantize_q4_k(weight), result.reconstruction, atol=0, rtol=0)


@cuda_only
def test_triton_encoder_matches_pytorch_reference_closely():
    # The Triton kernel runs the identical algorithm; it agrees with the reference bit-for-bit
    # on nearly all weights, differing by at most one code step where float reduction order
    # flips a grid-search tie. Both are valid Q4_K encodings.
    torch.manual_seed(1)
    weight = (torch.randn(256, 2048, device="cuda") * 0.05).to(torch.bfloat16)
    sub = _as_superblocks(weight)

    ref = _encode_superblocks_torch(sub.cpu()).reconstruction.to("cuda")
    tri = _encode_superblocks(sub, pack=False).reconstruction

    exact_frac = (ref == tri).float().mean().item()
    max_step = (ref - tri).abs().max().item()
    assert exact_frac > 0.99, exact_frac
    assert max_step < 0.05, max_step  # a single 4-bit code step, not a gross error


@cuda_only
def test_triton_q4_k_ste_passes_gradient_and_masks_saturation_on_cuda():
    weight = torch.randn(4, 512, device="cuda", requires_grad=True)
    fake_quantize_q4_k(weight).sum().backward()
    assert bool(torch.isfinite(weight.grad).all())
    assert set(weight.grad.unique().tolist()) <= {0.0, 1.0}


@cuda_only
def test_triton_minmax_encoder_is_self_consistent_on_cuda():
    torch.manual_seed(0)
    weight = (torch.randn(64, 1024, device="cuda") * 0.1).to(torch.bfloat16)

    result = quantize_q4_k(weight, method="minmax")
    dequantized = dequantize_q4_k(result.packed_weight, weight.shape[1])
    torch.testing.assert_close(dequantized.to(result.reconstruction.dtype), result.reconstruction, atol=0, rtol=0)
    torch.testing.assert_close(
        fake_quantize_q4_k(weight, method="minmax"), result.reconstruction, atol=0, rtol=0
    )


@cuda_only
def test_triton_minmax_matches_pytorch_reference_closely():
    # The naive min-max level-1 has no reduction-order-sensitive search, so the Triton kernel
    # matches the reference on nearly all weights; the few flips are round-half-to-even ties in
    # the final re-encode where the fp division differs by one ulp between backends.
    torch.manual_seed(1)
    weight = (torch.randn(256, 2048, device="cuda") * 0.05).to(torch.bfloat16)
    sub = _as_superblocks(weight)

    with_triton = _encode_superblocks(sub, pack=False, method="minmax").reconstruction
    reference = _encode_superblocks_torch(sub, method="minmax").reconstruction

    exact_frac = (with_triton == reference).float().mean().item()
    max_step = (with_triton - reference).abs().max().item()
    assert exact_frac > 0.99, exact_frac
    assert max_step < 0.05, max_step  # a single 4-bit code step, not a gross error


def test_q4_k_requires_in_features_multiple_of_256():
    with pytest.raises(ValueError, match="divisible by 256"):
        quantize_q4_k(torch.zeros(2, 300))
    with pytest.raises(ValueError, match="divisible by 256"):
        Q4KLinear.from_linear(nn.Linear(300, 4))


def test_q4_k_linear_preserves_activation_and_parameter_names():
    linear = nn.Linear(256, 4, bias=True)
    converted = Q4KLinear.from_linear(linear)
    output = converted(torch.randn(3, 256, dtype=linear.weight.dtype))

    assert output.shape == (3, 4)
    assert set(converted.state_dict()) == {"weight", "bias"}
    assert converted.weight is linear.weight


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(256, 512, bias=False)
        self.up_proj = nn.Linear(256, 512, bias=False)
        self.down_proj = nn.Linear(512, 256, bias=False)


class _AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256, bias=False)
        self.k_proj = nn.Linear(256, 256, bias=False)
        self.v_proj = nn.Linear(256, 256, bias=False)
        self.o_proj = nn.Linear(256, 256, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttnBlock()
        self.mlp = _MLPBlock()


class _InnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 256)
        self.layers = nn.ModuleList([_Layer()])
        self.norm = nn.LayerNorm(256)


class _Qwen3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            num_experts=None,
            tie_word_embeddings=False,
            get_text_config=lambda: SimpleNamespace(num_experts=None, tie_word_embeddings=False),
        )
        self.model = _InnerModel()
        self.lm_head = nn.Linear(256, 32)


_QAD_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_apply_fake_quant_q4_k_converts_targets_and_freezes_rest():
    model = _Qwen3Model()
    converted = apply_fake_quant(model, _QAD_TARGETS, quant_format="q4_k")

    assert set(converted) == {f"model.layers.0.{leaf}" for leaf in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )}
    assert isinstance(model.model.layers[0].self_attn.q_proj, Q4KLinear)
    assert not isinstance(model.lm_head, Q4KLinear)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {f"{name}.weight" for name in converted}


def test_q4_k_rejects_fused_runtime_scales():
    model = _Qwen3Model()
    with pytest.raises(ValueError, match="does not support fused_runtime_scales"):
        apply_fake_quant(model, _QAD_TARGETS, quant_format="q4_k", fused_runtime_scales=True)


def test_q4_k_format_metadata_and_pack_hook():
    fmt = get_quant_format("q4_k")
    assert fmt.quantizer_id == "q4_k_minmax"  # default q4_k is the naive min-max encoder
    assert fmt.supports_fused_runtime_scales is False
    assert fmt.pack_weight is not None
    packed = fmt.pack_weight(torch.randn(4, 256, dtype=torch.bfloat16))
    assert packed.dtype == torch.uint8 and packed.shape == (4, Q4K_BLOCK_BYTES)


def test_q4_k_grid_search_format_metadata():
    fmt = get_quant_format("q4_k_grid_search")
    assert fmt.quantizer_id == "q4_k_grid_search"
    assert fmt.supports_fused_runtime_scales is False
    assert fmt.pack_weight is not None
    packed = fmt.pack_weight(torch.randn(4, 256, dtype=torch.bfloat16))
    assert packed.dtype == torch.uint8 and packed.shape == (4, Q4K_BLOCK_BYTES)


def test_q4_k_minmax_encoder_self_consistent_and_matches_fake_quant():
    torch.manual_seed(0)
    weight = (torch.randn(6, 512) * 0.1).to(torch.bfloat16)

    result = quantize_q4_k(weight, method="minmax")
    dequantized = dequantize_q4_k(result.packed_weight, weight.shape[1])
    torch.testing.assert_close(dequantized.to(result.reconstruction.dtype), result.reconstruction, atol=0, rtol=0)
    torch.testing.assert_close(
        fake_quantize_q4_k(weight, method="minmax"), result.reconstruction, atol=0, rtol=0
    )


def test_q4_k_minmax_reconstruction_matches_hand_derived_affine():
    # A single sub-block with a known min/max: naive fit is scale=(max-vmin)/15, offset=-vmin,
    # so codes are round((x - vmin) / scale) and reconstruct on that affine lattice.
    x = torch.tensor([[-0.3] + [0.0] * 30 + [0.6]], dtype=torch.float32)  # 32 values, in one sub-block
    weight = x.repeat(1, 8)  # 256 in-features -> one super-block, all sub-blocks identical
    recon = fake_quantize_q4_k(weight, method="minmax")
    # vmin=-0.3, max=0.6 -> scale=0.9/15=0.06; endpoints reconstruct near-exactly (level-2 is
    # lossless here since all 8 sub-block scales/mins are identical).
    assert abs(recon[0, 0].item() - (-0.3)) < 1e-3
    assert abs(recon[0, 31].item() - 0.6) < 1e-3
    assert recon.min().item() >= -0.3 - 1e-3  # offset only subtracted, min bounded


def test_export_target_to_ggml_name_mapping():
    targets = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.3.mlp.down_proj",
        "model.layers.7.self_attn.o_proj",
    ]
    mapping = _map_targets_to_ggml(targets)
    assert mapping["model.layers.0.self_attn.q_proj"] == "blk.0.attn_q.weight"
    assert mapping["model.layers.3.mlp.down_proj"] == "blk.3.ffn_down.weight"
    assert mapping["model.layers.7.self_attn.o_proj"] == "blk.7.attn_output.weight"


def test_export_rejects_non_q4k_metadata():
    with pytest.raises(ValueError, match="missing qatfactory_qad"):
        _validate_q4k_metadata({"model_type": "qwen3"})
    with pytest.raises(ValueError, match="not a Q4_K QAD run"):
        _validate_q4k_metadata({"qatfactory_qad": {"quant_format": "nvfp4"}})


# ---- Qwen3.5 hybrid: GGUF reorder-aware Q4_K ----


def test_v_head_reorder_perm_is_grouped_to_tiled():
    # 2 K-heads, 3 V per K, head_dim 1: grouped [k0v0,k0v1,k0v2, k1v0,k1v1,k1v2]
    # -> tiled [k0v0,k1v0, k0v1,k1v1, k0v2,k1v2].
    perm = v_head_reorder_perm(num_k_heads=2, num_v_per_k=3, head_dim=1)
    assert perm.tolist() == [0, 3, 1, 4, 2, 5]


def test_out_proj_col_perm_fake_quant_matches_deploy_pack():
    torch.manual_seed(3)
    num_k, num_v_per_k, head_v = 2, 2, 128
    col_perm = v_head_reorder_perm(num_k, num_v_per_k, head_v)  # length 512
    weight = (torch.randn(64, col_perm.numel()) * 0.1).to(torch.bfloat16)

    fq = fake_quantize_q4_k(weight, col_perm=col_perm)
    deploy = pack_deploy_weight(weight, col_perm=col_perm)
    inverse = torch.empty_like(col_perm)
    inverse[col_perm] = torch.arange(col_perm.numel())
    un_permuted = dequantize_q4_k(deploy, col_perm.numel()).index_select(1, inverse)
    torch.testing.assert_close(un_permuted.to(fq.dtype), fq, atol=0, rtol=0)


def test_row_perm_deploy_pack_matches_row_permuted_fake_quant():
    torch.manual_seed(4)
    num_k, num_v_per_k, head_v = 2, 2, 128
    row_perm = v_head_reorder_perm(num_k, num_v_per_k, head_v)
    weight = (torch.randn(row_perm.numel(), 256) * 0.1).to(torch.bfloat16)

    deploy = pack_deploy_weight(weight, row_perm=row_perm)
    deployed_recon = dequantize_q4_k(deploy, 256)
    row_permuted_fq = fake_quantize_q4_k(weight).index_select(0, row_perm)
    torch.testing.assert_close(deployed_recon.to(row_permuted_fq.dtype), row_permuted_fq, atol=0, rtol=0)


def test_qwen35_linear_attn_ggml_name_mapping():
    prefix = "model.language_model.layers.5.linear_attn"
    targets = [f"{prefix}.{leaf}" for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")]
    mapping = _map_targets_to_ggml(targets)
    assert mapping[f"{prefix}.in_proj_qkv"] == "blk.5.attn_qkv.weight"
    assert mapping[f"{prefix}.in_proj_z"] == "blk.5.attn_gate.weight"
    assert mapping[f"{prefix}.in_proj_a"] == "blk.5.ssm_alpha.weight"
    assert mapping[f"{prefix}.in_proj_b"] == "blk.5.ssm_beta.weight"
    assert mapping[f"{prefix}.out_proj"] == "blk.5.ssm_out.weight"


class _LinearAttn(nn.Module):
    # head config: num_k=2, num_v=4, head_k=head_v=128 -> qk rows = 2*(128*2)=512, v rows = 512.
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(256, 1024, bias=False)  # q(256)+k(256)+v(512)... rows 1024
        self.in_proj_z = nn.Linear(256, 512, bias=False)     # num_v*head_v = 512
        self.in_proj_a = nn.Linear(256, 4, bias=False)       # num_v scalar rows
        self.in_proj_b = nn.Linear(256, 4, bias=False)
        self.out_proj = nn.Linear(512, 256, bias=False)      # in = num_v*head_v = 512


class _HybridLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _LinearAttn()
        self.mlp = _MLPBlock()


class _HybridInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 256)
        self.layers = nn.ModuleList([_HybridLayer()])
        self.norm = nn.LayerNorm(256)


class _HybridBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _HybridInner()


class _Qwen35LikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        text = SimpleNamespace(
            num_experts=None, tie_word_embeddings=False,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=128, linear_value_head_dim=128,
        )
        self.config = SimpleNamespace(
            model_type="qwen3_5", num_experts=None, tie_word_embeddings=False,
            get_text_config=lambda: text,
        )
        self.model = _HybridBackbone()
        self.lm_head = nn.Linear(256, 32)


def test_configure_q4k_reorders_sets_linear_attn_perms():
    model = _Qwen35LikeModel()
    apply_fake_quant(
        model,
        ["gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
        quant_format="q4_k",
    )
    configured = configure_q4k_reorders(model)
    assert configured == 5

    la = model.model.language_model.layers[0].linear_attn
    assert la.out_proj.q4k_col_perm is not None and la.out_proj.q4k_row_perm is None
    assert la.in_proj_z.q4k_row_perm is not None and la.in_proj_z.q4k_col_perm is None
    assert la.in_proj_a.q4k_row_perm.numel() == 4  # num_v_heads scalar rows
    # in_proj_qkv row perm keeps the q/k rows fixed and reorders only the V block.
    qk_rows = 128 * 2 + 128 * 2
    assert torch.equal(la.in_proj_qkv.q4k_row_perm[:qk_rows], torch.arange(qk_rows))
    # MLP targets carry no reorder.
    assert model.model.language_model.layers[0].mlp.gate_proj.q4k_col_perm is None


def test_q4_k_splice_into_synthetic_gguf_is_lossless(tmp_path):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    from qatfactory.quant.export.q4_k_gguf import _verify_lossless, splice_q4k_into_gguf

    scaffold = tmp_path / "scaffold.gguf"
    output = tmp_path / "out.gguf"
    torch.manual_seed(2)
    hidden, ffn, vocab = 512, 768, 64
    tensors = {
        "blk.0.attn_q.weight": torch.randn(hidden, hidden) * 0.1,
        "blk.0.ffn_down.weight": torch.randn(hidden, ffn) * 0.1,
        "blk.0.attn_norm.weight": torch.ones(hidden),
        "token_embd.weight": torch.randn(vocab, hidden) * 0.1,
        "output.weight": torch.randn(vocab, hidden) * 0.1,
    }

    def _write_scaffold(path):
        writer = gguf.GGUFWriter(str(path), "qwen3")
        writer.add_uint32("qwen3.block_count", 1)
        writer.add_string("general.name", "synthetic")
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor.to(torch.float32).numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    _write_scaffold(scaffold)

    ggml_packed = {
        name: quantize_q4_k(tensors[name].to(torch.bfloat16)).packed_weight.numpy()
        for name in ("blk.0.attn_q.weight", "blk.0.ffn_down.weight")
    }
    spliced = splice_q4k_into_gguf(scaffold, output, ggml_packed)
    _verify_lossless(output, ggml_packed, spliced)

    reader = gguf.GGUFReader(output)
    types = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
    assert types["blk.0.attn_q.weight"] == "Q4_K"
    assert types["blk.0.ffn_down.weight"] == "Q4_K"
    assert types["blk.0.attn_norm.weight"] == "F32"
    # Frozen vocab tensors default to a Q8_0 re-encode.
    assert types["token_embd.weight"] == "Q8_0"
    assert types["output.weight"] == "Q8_0"
    assert {field.name for field in reader.fields.values()} >= {"general.name", "qwen3.block_count"}

    # Opt-out: embed_output_dtype="bf16" copies the vocab tensors through unchanged (F32 here,
    # since the synthetic scaffold is written as F32).
    scaffold_bf16 = tmp_path / "scaffold_bf16.gguf"
    output_bf16 = tmp_path / "out_bf16.gguf"
    _write_scaffold(scaffold_bf16)
    spliced_bf16 = splice_q4k_into_gguf(scaffold_bf16, output_bf16, ggml_packed, embed_output_dtype="bf16")
    _verify_lossless(output_bf16, ggml_packed, spliced_bf16)
    types_bf16 = {t.name: t.tensor_type.name for t in gguf.GGUFReader(output_bf16).tensors}
    assert types_bf16["token_embd.weight"] == "F32"
    assert types_bf16["output.weight"] == "F32"
