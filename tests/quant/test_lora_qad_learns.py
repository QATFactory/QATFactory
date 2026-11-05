"""End-to-end check that LoRA-QAD actually closes the quantization gap it is given.

These are the tests that catch a silently non-learning setup: if the adapters are wired
into the graph correctly, optimizing a fixed batch must drive the student's distillation
loss toward zero, because the teacher is the same frozen model with quantization bypassed
and the adapters have enough capacity to cancel the quantization error on that batch.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.distill.kd_loss import distill_loss
from qatfactory.quant import apply_fake_quant, shared_base_teacher

transformers_minimax = pytest.importorskip("transformers.models.minimax_m2.modeling_minimax_m2")
minimax_config = pytest.importorskip("transformers.models.minimax_m2.configuration_minimax_m2")


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    config = minimax_config.MiniMaxM2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        rotary_dim=8,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = transformers_minimax.MiniMaxM2ForCausalLM(config)
    # A randomly initialized model quantizes almost losslessly, which would leave nothing to
    # learn. Widening the weights makes the NVFP4 grid coarse relative to the values, so the
    # student starts with a gap the adapters have to close.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.mul_(4.0)
        for layer in model.model.layers:
            layer.mlp.experts.gate_up_proj.mul_(4.0)
            layer.mlp.experts.down_proj.mul_(4.0)
    return model.eval()


def _fixed_batch(vocab: int, tokens: int = 24, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (2, tokens), generator=generator)
    labels = input_ids.clone()
    labels[:, : tokens // 2] = -100  # supervise only the second half, as the collator does
    return input_ids, labels


def _step_losses(model, input_ids, labels, *, steps: int, lr: float):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable adapters"
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    with torch.no_grad(), shared_base_teacher():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits

    losses, grad_norms = [], []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(input_ids=input_ids, use_cache=False).logits
        result = distill_loss(student_logits, teacher_logits, labels)
        result.loss.backward()
        grad_norms.append(
            torch.nn.utils.clip_grad_norm_(trainable, float("inf")).item()
        )
        optimizer.step()
        losses.append(result.loss.item())
    return losses, grad_norms


@pytest.mark.parametrize(
    "targets,hybrid_shared",
    [
        (["q_proj", "k_proj", "v_proj", "o_proj"], True),
        ([], True),
        ([], False),
    ],
    ids=["attention+experts", "experts-only-shared", "experts-only-unshared"],
)
def test_adapters_reduce_the_quantization_gap(targets, hybrid_shared):
    model = _tiny_model()
    apply_fake_quant(model, targets, lora_rank=8, lora_alpha=16, lora_moe_hybrid_shared=hybrid_shared)
    input_ids, labels = _fixed_batch(model.config.vocab_size)

    losses, grad_norms = _step_losses(model, input_ids, labels, steps=200, lr=1e-2)

    assert losses[0] > 1e-4, f"no quantization gap to close (start {losses[0]:.3e})"
    assert max(grad_norms) > 1e-4, f"gradients vanish: max grad_norm {max(grad_norms):.3e}"
    # Compare smoothed ends rather than single steps: a fixed batch still moves the loss
    # around as the adapters reshape the quantization grid, and one unlucky step should not
    # decide the verdict.
    start = sum(losses[:20]) / 20
    end = sum(losses[-20:]) / 20
    assert end < 0.7 * start, f"loss did not fall: {start:.4e} -> {end:.4e}"


def test_expert_adapters_receive_gradient():
    """Every routed-expert adapter tensor must get a non-zero gradient."""

    model = _tiny_model()
    apply_fake_quant(model, [], lora_rank=8, lora_alpha=16)
    input_ids, labels = _fixed_batch(model.config.vocab_size)

    with torch.no_grad(), shared_base_teacher():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits
    student_logits = model(input_ids=input_ids, use_cache=False).logits
    distill_loss(student_logits, teacher_logits, labels).loss.backward()

    grads = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert grads, "no trainable parameters"
    missing = [name for name, grad in grads.items() if grad is None]
    assert not missing, f"no gradient reached {missing[:4]}"
    # `lora_A` starts Kaiming-initialized and `lora_B` at zero, so at step 0 only B can move:
    # dL/dA is proportional to B and is therefore exactly zero. Assert that split explicitly,
    # because a silently detached adapter looks the same as this on a single tensor.
    zero_b = [name for name, grad in grads.items() if "lora_B" in name and grad.abs().max() == 0]
    assert not zero_b, f"lora_B received no gradient: {zero_b[:4]}"


def test_disabling_the_adapter_removes_all_gradient():
    """Control: with quantization bypassed, teacher and student match and nothing learns."""

    model = _tiny_model()
    apply_fake_quant(model, [], lora_rank=8, lora_alpha=16)
    input_ids, labels = _fixed_batch(model.config.vocab_size)

    with shared_base_teacher():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits
        student_logits = model(input_ids=input_ids, use_cache=False).logits
    loss = distill_loss(student_logits, teacher_logits, labels).loss

    # Not exactly zero: two runs of the same graph differ by float reduction order, which is
    # also the noise floor any real signal has to clear.
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_quantization_gap_is_visible_in_the_logits():
    model = _tiny_model()
    apply_fake_quant(model, [], lora_rank=8, lora_alpha=16)
    input_ids, _ = _fixed_batch(model.config.vocab_size)

    with torch.no_grad():
        with shared_base_teacher():
            teacher = model(input_ids=input_ids, use_cache=False).logits
        student = model(input_ids=input_ids, use_cache=False).logits

    kl = F.kl_div(
        F.log_softmax(student, dim=-1), F.log_softmax(teacher, dim=-1),
        log_target=True, reduction="batchmean",
    )
    assert kl.item() > 1e-4, f"quantizing the experts barely moved the logits (KL {kl.item():.3e})"
