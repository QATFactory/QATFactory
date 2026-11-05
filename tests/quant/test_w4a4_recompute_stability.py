"""W4A4 activation-quant scales must be immutable within an optimizer step.

The gradient-checkpoint recompute re-executes a layer's forward after the
original pass already updated the running input maxima. If quantization reads
the running max, a record-setting batch makes the recompute numerically drift,
which can flip MoE routing and change expert-dispatch tensor shapes — tripping
torch.utils.checkpoint's saved-tensor metadata check. These tests pin the invariant that repeating
a forward — the second call standing in for the recompute — is bit-identical
even on record batches, until commit_nvfp4_input_scales publishes the running
max between steps.
"""

import torch
from torch import nn

from qatfactory.quant.fake_quant import apply_fake_quant
from qatfactory.quant.formats.nvfp4 import NVFP4Linear, commit_nvfp4_input_scales
from qatfactory.quant.formats.nvfp4_experts import NVFP4Experts


def _w4a4_linear(in_features=32, out_features=16):
    torch.manual_seed(0)
    module = NVFP4Linear.from_linear(nn.Linear(in_features, out_features, bias=False))
    module._track_input_amax = True
    module._quantize_activations = True
    module.train()
    return module


def _w4a4_experts(num_experts=4, intermediate=16, hidden=32):
    torch.manual_seed(0)
    gate_up = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden) * 0.05)
    down = nn.Parameter(torch.randn(num_experts, hidden, intermediate) * 0.05)
    module = NVFP4Experts(gate_up, down, nn.SiLU())
    module._track_input_amax = True
    module._quantize_activations = True
    module.train()
    return module


def _routing(tokens, num_experts, top_k=2):
    top_k_index = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(tokens)])
    top_k_weights = torch.softmax(torch.randn(tokens, top_k), dim=-1)
    return top_k_index, top_k_weights


def test_nvfp4_linear_repeat_forward_bit_identical_after_record():
    module = _w4a4_linear()
    torch.manual_seed(1)
    module(torch.randn(4, 32) * 0.1)  # bootstrap the static scale from a small batch
    record = torch.randn(4, 32) * 10.0  # sets a new running-max record
    first = module(record)
    second = module(record)  # stands in for the gradient-checkpoint recompute
    assert torch.equal(first, second)
    assert module._input_amax > module._input_amax_static  # record observed, not yet published


def test_nvfp4_experts_repeat_forward_bit_identical_after_record():
    module = _w4a4_experts()
    torch.manual_seed(2)
    top_k_index, top_k_weights = _routing(tokens=8, num_experts=4)
    module(torch.randn(8, 32) * 0.1, top_k_index, top_k_weights)  # bootstrap both scales
    record = torch.randn(8, 32) * 10.0
    first = module(record, top_k_index, top_k_weights)
    second = module(record, top_k_index, top_k_weights)
    assert torch.equal(first, second)
    assert module._gate_up_input_amax > module._gate_up_input_amax_static
    assert module._down_input_amax > module._down_input_amax_static


def test_commit_publishes_running_max():
    model = nn.Sequential()
    model.linear = _w4a4_linear()
    model.experts = _w4a4_experts()
    torch.manual_seed(3)
    top_k_index, top_k_weights = _routing(tokens=8, num_experts=4)
    model.linear(torch.randn(4, 32) * 0.1)
    model.experts(torch.randn(8, 32) * 0.1, top_k_index, top_k_weights)
    model.linear(torch.randn(4, 32) * 10.0)
    model.experts(torch.randn(8, 32) * 10.0, top_k_index, top_k_weights)

    commit_nvfp4_input_scales(model)

    assert torch.equal(model.linear._input_amax_static, model.linear._input_amax)
    assert torch.equal(model.experts._gate_up_input_amax_static, model.experts._gate_up_input_amax)
    assert torch.equal(model.experts._down_input_amax_static, model.experts._down_input_amax)


def test_gradckpt_backward_after_record_survives(torch_delta_net):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(4)
    config = Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        attn_implementation="eager",
    )
    model = Qwen3_5MoeForCausalLM(config)
    apply_fake_quant(
        model,
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        fused_runtime_scales=True,
        quantize_shared_expert=True,
        quantize_activations=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    model(input_ids=input_ids, use_cache=False).logits.float().sum().backward()
    model.zero_grad(set_to_none=True)

    # Scale the embeddings up so every layer sees new running-max records on the
    # next step; the recompute in backward must stay bit-identical regardless.
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(8.0)
    model(input_ids=input_ids, use_cache=False).logits.float().sum().backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # Production cycle: publish the records between steps, then survive another
    # record-setting step — commit must not leak a mid-step mutable scale.
    model.zero_grad(set_to_none=True)
    commit_nvfp4_input_scales(model)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(8.0)
    model(input_ids=input_ids, use_cache=False).logits.float().sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
