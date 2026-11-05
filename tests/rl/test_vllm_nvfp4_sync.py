"""Publisher tests: layer mapping, fused-scale tying, and the no-op-resync property.

The vLLM engine is stood in for by a tree of plain modules whose ``scheme`` reproduces
``CompressedTensorsW4A4Fp4.process_weights_after_loading`` (rename, reciprocal global
scales, derived ``alpha``) without the CUDA-only kernel step, so the whole publish path —
scratch-layer construction, checkpoint-order reconstruction, in-place copy — runs on CPU.
"""

import pytest
import torch
from torch import nn

from qatfactory.quant.fake_quant import apply_fake_quant
from qatfactory.quant.formats.nvfp4 import NVFP4_SCALE_DENOMINATOR, quantize_nvfp4
from qatfactory.rl.vllm_nvfp4_sync import NVFP4WeightPublisher


HIDDEN = 32
KV = 16
INTERMEDIATE = 64
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _Config:
    num_experts = None
    tie_word_embeddings = False


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _Body(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(num_layers))


class _CausalLM(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.model = _Body(num_layers)
        self.config = _Config()


class _FakeScheme:
    """Mirrors vLLM's NVFP4 post-load transform, minus the kernel-specific relayout."""

    def process_weights_after_loading(self, layer):
        layer.weight = layer.weight_packed
        del layer.weight_packed
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = nn.Parameter(1.0 / weight_global_scale, requires_grad=False)
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = nn.Parameter(1.0 / input_global_scale_inv, requires_grad=False)
        layer.input_global_scale_inv = nn.Parameter(input_global_scale_inv, requires_grad=False)
        layer.alpha = nn.Parameter(layer.input_global_scale * layer.weight_global_scale, requires_grad=False)


class _FusedLinear(nn.Module):
    """A stand-in for a live vLLM quantized linear, populated as the loader would."""

    def __init__(self, weights, input_amax):
        super().__init__()
        self.scheme = _FakeScheme()
        self.logical_widths = [weight.shape[0] for weight in weights]
        self.input_size_per_partition = weights[0].shape[1]
        self.output_size_per_partition = sum(self.logical_widths)
        packed = quantize_nvfp4(
            torch.cat(weights, dim=0),
            tensor_amax=torch.stack([w.abs().amax() for w in weights]).amax().float(),
        )
        self.weight_packed = nn.Parameter(packed.packed_weight, requires_grad=False)
        self.weight_scale = nn.Parameter(packed.weight_scale, requires_grad=False)
        self.weight_global_scale = nn.Parameter(
            packed.weight_scale_2.reshape(1).reciprocal().repeat(len(weights)), requires_grad=False
        )
        self.input_global_scale = nn.Parameter(
            torch.full((len(weights),), NVFP4_SCALE_DENOMINATOR / input_amax, dtype=torch.float32),
            requires_grad=False,
        )
        self.scheme.process_weights_after_loading(self)


class _FakeVLLMAttention(nn.Module):
    def __init__(self, source: _Attention):
        super().__init__()
        self.qkv_proj = _FusedLinear(
            [source.q_proj.weight, source.k_proj.weight, source.v_proj.weight], input_amax=2.0
        )
        self.o_proj = _FusedLinear([source.o_proj.weight], input_amax=1.5)


class _FakeVLLMMLP(nn.Module):
    def __init__(self, source: _MLP):
        super().__init__()
        self.gate_up_proj = _FusedLinear([source.gate_proj.weight, source.up_proj.weight], input_amax=3.0)
        self.down_proj = _FusedLinear([source.down_proj.weight], input_amax=2.5)


class _FakeVLLMLayer(nn.Module):
    def __init__(self, source: _Layer):
        super().__init__()
        self.self_attn = _FakeVLLMAttention(source.self_attn)
        self.mlp = _FakeVLLMMLP(source.mlp)


class _FakeVLLMBody(nn.Module):
    def __init__(self, source: _Body):
        super().__init__()
        self.layers = nn.ModuleList(_FakeVLLMLayer(layer) for layer in source.layers)


class _FakeVLLMModel(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, source: _CausalLM):
        super().__init__()
        self.model = _FakeVLLMBody(source.model)


@pytest.fixture
def student():
    torch.manual_seed(0)
    model = _CausalLM()
    apply_fake_quant(model, TARGETS, quant_format="nvfp4", fused_runtime_scales=True)
    return model


def _snapshot(vllm_model):
    return {name: tensor.detach().clone() for name, tensor in vllm_model.state_dict().items()}


def test_targets_cover_every_quantized_linear(student):
    publisher = NVFP4WeightPublisher(student, _FakeVLLMModel(student))

    assert len(publisher.targets) == 8  # 2 layers x (qkv, o, gate_up, down)
    by_name = {target.vllm_name: target for target in publisher.targets}
    assert by_name["model.layers.0.self_attn.qkv_proj"].member_names == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    )
    assert by_name["model.layers.1.mlp.gate_up_proj"].member_names == (
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.up_proj",
    )
    assert by_name["model.layers.0.mlp.down_proj"].member_names == ("model.layers.0.mlp.down_proj",)
    covered = {name for target in publisher.targets for name in target.member_names}
    assert len(covered) == 2 * len(TARGETS)


def test_fused_layer_reuses_the_training_amax_group(student):
    publisher = NVFP4WeightPublisher(student, _FakeVLLMModel(student))

    for target in publisher.targets:
        # The scale the straight-through forward quantizes with is the same object the
        # publisher packs with, which is what keeps the two policies on one lattice.
        assert set(target.amax_group.member_names) == set(target.member_names)
        for member in target.members:
            assert member._amax_group is target.amax_group


def test_rejects_a_fused_layer_that_is_not_one_amax_family(student):
    vllm_model = _FakeVLLMModel(student)
    # Split q away from k/v so the layer no longer matches a QAD fused family.
    q_proj = student.model.layers[0].self_attn.q_proj
    q_proj._amax_group = type(q_proj._amax_group)(("model.layers.0.self_attn.q_proj",), (q_proj,))

    with pytest.raises(RuntimeError, match="amax group"):
        NVFP4WeightPublisher(student, vllm_model)


def test_resync_of_unchanged_latents_is_a_no_op(student):
    vllm_model = _FakeVLLMModel(student)
    publisher = NVFP4WeightPublisher(student, vllm_model)
    before = _snapshot(vllm_model)

    publisher.sync()

    after = vllm_model.state_dict()
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), f"{name} changed on a no-op resync"


def test_sync_publishes_updated_latents(student):
    vllm_model = _FakeVLLMModel(student)
    publisher = NVFP4WeightPublisher(student, vllm_model)
    before = _snapshot(vllm_model)

    with torch.no_grad():
        weight = student.model.layers[0].mlp.down_proj.weight
        weight.add_(torch.randn_like(weight))
    publisher.sync()

    after = vllm_model.state_dict()
    changed = {name for name, tensor in before.items() if not torch.equal(tensor, after[name])}
    assert changed == {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale",
        "model.layers.0.mlp.down_proj.weight_global_scale",
        "model.layers.0.mlp.down_proj.alpha",
    }


def test_activation_scales_are_never_rewritten(student):
    vllm_model = _FakeVLLMModel(student)
    publisher = NVFP4WeightPublisher(student, vllm_model)
    before = _snapshot(vllm_model)

    for _ in range(5):
        with torch.no_grad():
            weight = student.model.layers[0].mlp.up_proj.weight
            weight.add_(torch.randn_like(weight))
        publisher.sync()

    after = vllm_model.state_dict()
    for name in before:
        if name.endswith(("input_global_scale", "input_global_scale_inv")):
            assert torch.equal(before[name], after[name]), f"{name} drifted across syncs"


def _seed_input_amax(student, value=1.0, track=True):
    for module in student.modules():
        if hasattr(module, "_input_amax"):
            module._input_amax = torch.tensor(float(value))
            module._input_amax_static = torch.tensor(float(value))
            module._track_input_amax = track


def test_scale_sync_can_lower_a_scale_the_export_set_too_high(student):
    """The whole point: the engine must be able to move to a *finer* activation grid.

    Seeding the running maximum from the export would make the tracked value a max over
    max(export, observed), which can only grow — so a model whose activations are smaller
    than the donor's could never benefit.
    """

    vllm_model = _FakeVLLMModel(student)  # built with input_amax 2.0-3.0 per layer
    _seed_input_amax(student, value=0.25)
    publisher = NVFP4WeightPublisher(student, vllm_model, sync_input_scales=True, warmup_steps=0)
    before = _snapshot(vllm_model)

    publisher.sync()

    after = vllm_model.state_dict()
    layer = "model.layers.0.mlp.down_proj"
    assert after[f"{layer}.input_global_scale"] < before[f"{layer}.input_global_scale"]
    assert torch.allclose(
        after[f"{layer}.input_global_scale"], torch.tensor(0.25 / NVFP4_SCALE_DENOMINATOR)
    )


def test_scale_sync_defers_until_inputs_have_been_observed(student):
    vllm_model = _FakeVLLMModel(student)
    _seed_input_amax(student, value=2.0)
    for module in student.modules():  # nothing observed yet, as before the first forward
        if hasattr(module, "_input_amax"):
            module._input_amax = None
    publisher = NVFP4WeightPublisher(student, vllm_model, sync_input_scales=True, warmup_steps=0)
    before = _snapshot(vllm_model)

    publisher.sync()  # must not raise on the missing observations

    after = vllm_model.state_dict()
    for name in before:
        if name.endswith(("input_global_scale", "input_global_scale_inv")):
            assert torch.equal(before[name], after[name])


def test_scale_sync_requires_running_max_tracking(student):
    vllm_model = _FakeVLLMModel(student)
    _seed_input_amax(student, track=False)

    with pytest.raises(RuntimeError, match="running-max tracking"):
        NVFP4WeightPublisher(student, vllm_model, sync_input_scales=True)


def test_activation_scales_hold_during_warmup_then_track(student):
    vllm_model = _FakeVLLMModel(student)
    _seed_input_amax(student, value=2.0)
    step = {"value": 0}
    publisher = NVFP4WeightPublisher(
        student, vllm_model, sync_input_scales=True, warmup_steps=3, step_provider=lambda: step["value"]
    )
    before = _snapshot(vllm_model)
    scale_names = [n for n in before if n.endswith(("input_global_scale", "input_global_scale_inv", "alpha"))]
    assert scale_names

    # Observed activations grow, but nothing is published while inside the warmup.
    _seed_input_amax(student, value=8.0)
    publisher.sync()
    during = vllm_model.state_dict()
    for name in scale_names:
        if name.endswith("alpha"):
            continue  # alpha also tracks the weight scale, which is republished every sync
        assert torch.equal(before[name], during[name]), f"{name} moved during warmup"

    # Past the warmup the engine picks up the observed maximum.
    step["value"] = 3
    publisher.sync()
    after = vllm_model.state_dict()
    for name in scale_names:
        assert not torch.equal(before[name], after[name]), f"{name} did not track after warmup"

    # input_global_scale is amax / (6 * 448), and its companion is the reciprocal.
    layer = "model.layers.0.mlp.down_proj"
    expected = torch.tensor(8.0 / NVFP4_SCALE_DENOMINATOR)
    assert torch.allclose(after[f"{layer}.input_global_scale"], expected)
    assert torch.allclose(after[f"{layer}.input_global_scale_inv"], 1.0 / expected)


def test_scale_sync_off_leaves_activation_scales_untouched(student):
    vllm_model = _FakeVLLMModel(student)
    _seed_input_amax(student, value=2.0)
    publisher = NVFP4WeightPublisher(student, vllm_model)  # sync disabled (default)
    before = _snapshot(vllm_model)

    _seed_input_amax(student, value=64.0)
    publisher.sync()

    after = vllm_model.state_dict()
    for name in before:
        if name.endswith(("input_global_scale", "input_global_scale_inv")):
            assert torch.equal(before[name], after[name]), f"{name} changed with sync disabled"


def test_packing_a_fused_matrix_matches_packing_its_members(student):
    """Row-blocks of a jointly packed matrix equal the members packed alone.

    This is why the publisher can pack a fused layer in one shot: with the tensor scale
    fixed, NVFP4 is a per-row, per-16-group encoding.
    """

    attention = student.model.layers[0].self_attn
    weights = [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight]
    amax = torch.stack([weight.detach().abs().amax() for weight in weights]).amax().float()

    joint = quantize_nvfp4(torch.cat([weight.detach() for weight in weights], dim=0), tensor_amax=amax)
    offset = 0
    for weight in weights:
        rows = weight.shape[0]
        alone = quantize_nvfp4(weight.detach(), tensor_amax=amax)
        assert torch.equal(joint.packed_weight[offset : offset + rows], alone.packed_weight)
        assert torch.equal(joint.weight_scale[offset : offset + rows], alone.weight_scale)
        assert torch.equal(joint.weight_scale_2, alone.weight_scale_2)
        offset += rows


def test_skipping_the_reconstruction_does_not_change_the_packed_lattice():
    torch.manual_seed(0)
    weight = torch.randn(64, 32, dtype=torch.bfloat16)

    full = quantize_nvfp4(weight)
    lean = quantize_nvfp4(weight, compute_reconstruction=False)

    assert lean.reconstruction is None
    assert torch.equal(full.packed_weight, lean.packed_weight)
    assert torch.equal(full.weight_scale, lean.weight_scale)
    assert torch.equal(full.weight_scale_2, lean.weight_scale_2)
