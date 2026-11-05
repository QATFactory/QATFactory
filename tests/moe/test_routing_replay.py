from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qatfactory.moe import RoutingReplay
from qatfactory.trainer import QADConfig, QADTrainer


NUM_EXPERTS, HIDDEN, TOP_K = 8, 32, 2


class _LogitsRouter(nn.Module):
    """transformers>=5.9 contract: (raw logits, top-k weights, top-k indices)."""

    def __init__(self, norm_topk_prob=True):
        super().__init__()
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, HIDDEN), self.weight)
        weights, indices = torch.topk(torch.softmax(logits.float(), dim=-1), TOP_K, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(logits.dtype), indices


class _ProbsRouter(_LogitsRouter):
    """transformers<=5.5 contract: full fp32 softmax probabilities come first."""

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, HIDDEN), self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        weights, indices = torch.topk(probs, TOP_K, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return probs, weights.to(probs.dtype), indices


class _RecordingExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_indices = None
        self.last_weights = None

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.last_indices = top_k_index
        self.last_weights = top_k_weights
        return top_k_weights.sum(dim=-1, keepdim=True).to(hidden_states.dtype) * hidden_states


class _Block(nn.Module):
    def __init__(self, router_cls):
        super().__init__()
        self.gate = router_cls()
        self.experts = _RecordingExperts()

    def forward(self, hidden_states):
        _, weights, indices = self.gate(hidden_states)
        return self.experts(hidden_states, indices, weights)


class _TinyMoE(nn.Module):
    def __init__(self, router_cls=_LogitsRouter, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_Block(router_cls) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = hidden_states + layer(hidden_states)
        return hidden_states


def _pair(router_cls=_LogitsRouter, perturb=0.0, seed=0):
    torch.manual_seed(seed)
    teacher = _TinyMoE(router_cls)
    torch.manual_seed(seed)
    student = _TinyMoE(router_cls)
    if perturb:
        with torch.no_grad():
            for layer in student.layers:
                layer.gate.weight.add_(perturb * torch.randn_like(layer.gate.weight))
    return teacher, student


def test_attach_discovers_and_validates_routers():
    teacher, student = _pair()
    replay = RoutingReplay(teacher, student)
    assert replay.router_names == ("layers.0.gate", "layers.1.gate")
    assert not replay._first_is_probs

    probs_teacher, probs_student = _pair(_ProbsRouter)
    assert RoutingReplay(probs_teacher, probs_student)._first_is_probs

    with pytest.raises(ValueError, match="No MoE routers"):
        RoutingReplay(nn.Linear(4, 4), nn.Linear(4, 4))
    with pytest.raises(ValueError, match="mismatch"):
        RoutingReplay(_TinyMoE(num_layers=1), student)


def test_replay_overrides_student_expert_selection():
    teacher, student = _pair(perturb=0.5)
    replay = RoutingReplay(teacher, student)
    hidden = torch.randn(5, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)

    for teacher_layer, student_layer in zip(teacher.layers, student.layers):
        assert torch.equal(student_layer.experts.last_indices, teacher_layer.experts.last_indices)

    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)
    flips = sum(
        int(not torch.equal(s.experts.last_indices.sort(-1).values, t.experts.last_indices.sort(-1).values))
        for s, t in zip(student.layers, teacher.layers)
    )
    assert flips > 0  # the perturbed student routes differently on its own


def test_mask_mode_scores_are_renormalized_student_probabilities():
    teacher, student = _pair(perturb=0.5)
    replay = RoutingReplay(teacher, student, mode="mask")
    hidden = torch.randn(5, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)

    layer = student.layers[0]
    logits = F.linear(hidden.reshape(-1, HIDDEN), layer.gate.weight)
    probs = torch.softmax(logits.float(), dim=-1)
    expected = probs.gather(-1, teacher.layers[0].experts.last_indices)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(layer.experts.last_weights, expected.to(layer.experts.last_weights.dtype))


def test_mask_mode_contract_invariance_across_router_conventions():
    """Identical weights must produce identical replayed scores under both router styles."""

    results = []
    for router_cls in (_LogitsRouter, _ProbsRouter):
        teacher, student = _pair(router_cls, perturb=0.5, seed=3)
        replay = RoutingReplay(teacher, student, mode="mask")
        torch.manual_seed(7)
        hidden = torch.randn(5, HIDDEN)
        replay.begin_batch(replay_active=True)
        with torch.no_grad():
            teacher(hidden)
            student(hidden)
        results.append(student.layers[0].experts.last_weights.float())
    torch.testing.assert_close(results[0], results[1])


def test_mask_mode_respects_norm_topk_prob_flag():
    teacher, student = _pair(perturb=0.5, seed=5)
    for model in (teacher, student):
        for layer in model.layers:
            layer.gate.norm_topk_prob = False
    replay = RoutingReplay(teacher, student, mode="mask")
    hidden = torch.randn(4, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)

    layer = student.layers[0]
    logits = F.linear(hidden.reshape(-1, HIDDEN), layer.gate.weight)
    probs = torch.softmax(logits.float(), dim=-1)
    expected = probs.gather(-1, teacher.layers[0].experts.last_indices)  # no renormalization
    torch.testing.assert_close(layer.experts.last_weights, expected.to(layer.experts.last_weights.dtype))


def test_full_mode_replays_teacher_scores():
    teacher, student = _pair(perturb=0.5)
    replay = RoutingReplay(teacher, student, mode="full")
    hidden = torch.randn(5, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)

    for teacher_layer, student_layer in zip(teacher.layers, student.layers):
        torch.testing.assert_close(student_layer.experts.last_weights, teacher_layer.experts.last_weights)


def test_gradient_flows_through_mask_mode_gating():
    teacher, student = _pair(perturb=0.5)
    replay = RoutingReplay(teacher, student, mode="mask")
    hidden = torch.randn(5, HIDDEN, requires_grad=True)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden.detach())
    student(hidden).sum().backward()

    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_metrics_flip_rate_and_masking():
    teacher, student = _pair(perturb=0.0)  # identical models: zero flips
    replay = RoutingReplay(teacher, student)
    hidden = torch.randn(6, HIDDEN)

    replay.begin_batch(replay_active=False, supervision_mask=torch.tensor([True] * 4 + [False] * 2))
    with torch.no_grad():
        teacher(hidden)
        student(hidden)
    metrics = replay.metrics()
    torch.testing.assert_close(metrics["routing_flip_rate"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["routing_jaccard"], torch.tensor(1.0))

    with pytest.raises(RuntimeError, match="Supervision mask"):
        replay.begin_batch(replay_active=False, supervision_mask=torch.ones(3, dtype=torch.bool))
        with torch.no_grad():
            teacher(hidden)
            student(hidden)


def test_recompute_is_idempotent_and_missing_teacher_raises():
    teacher, student = _pair(perturb=0.5)
    replay = RoutingReplay(teacher, student)
    hidden = torch.randn(5, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(hidden)
        student(hidden)
        first = replay.metrics()
        student(hidden)  # gradient-checkpointing recomputation replays identical state
        second = replay.metrics()
    for key in first:
        torch.testing.assert_close(first[key], second[key])

    replay.begin_batch(replay_active=True)
    with pytest.raises(RuntimeError, match="teacher forward must precede"):
        with torch.no_grad():
            student(hidden)


def test_replay_probability_schedule():
    trainer = QADTrainer.__new__(QADTrainer)
    trainer.qad = QADConfig(replay_decay_start=0.5, replay_decay_end=0.8, replay_in_eval=False)
    trainer.args = SimpleNamespace(max_steps=100, seed=42)

    def probability_at(step):
        trainer.state = SimpleNamespace(max_steps=100, global_step=step)
        return trainer._replay_probability()

    assert probability_at(0) == 1.0
    assert probability_at(49) == 1.0
    assert abs(probability_at(65) - 0.5) < 1e-6
    assert probability_at(80) == 0.0
    assert probability_at(99) == 0.0
    trainer.state = SimpleNamespace(max_steps=100, global_step=0)
    assert trainer._replay_active_now(training=False) is False
    assert trainer._replay_active_now(training=True) is True


def test_real_transformers_qwen3_moe_block_full_replay():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    config = Qwen3MoeConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=16,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        norm_topk_prob=True,
    )
    torch.manual_seed(0)
    teacher_block = Qwen3MoeSparseMoeBlock(config)
    with torch.no_grad():
        for parameter in teacher_block.parameters():
            # Bare-module construction leaves torch.empty parameters uninitialized.
            nn.init.normal_(parameter, std=0.1)
    student_block = Qwen3MoeSparseMoeBlock(config)
    student_block.load_state_dict(teacher_block.state_dict())
    with torch.no_grad():
        student_block.gate.weight.add_(0.5 * torch.randn_like(student_block.gate.weight))

    replay = RoutingReplay(teacher_block, student_block, mode="full")
    hidden = torch.randn(2, 3, HIDDEN)

    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        free_out = student_block(hidden)
    assert replay.metrics()["routing_flip_rate"] > 0
    assert not torch.allclose(free_out, teacher_out)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        replay_out = student_block(hidden)
    # Full replay + identical experts => the student computes exactly the teacher's output.
    torch.testing.assert_close(replay_out, teacher_out)


def _qwen3_5_moe_config():
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    return Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def _qwen3_5_moe_block_pair(config, perturb=0.5, seed=0):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    torch.manual_seed(seed)
    teacher_block = Qwen3_5MoeSparseMoeBlock(config)
    with torch.no_grad():
        for parameter in teacher_block.parameters():
            # Bare-module construction leaves torch.empty parameters uninitialized.
            nn.init.normal_(parameter, std=0.1)
    student_block = Qwen3_5MoeSparseMoeBlock(config)
    student_block.load_state_dict(teacher_block.state_dict())
    with torch.no_grad():
        student_block.gate.weight.add_(perturb * torch.randn_like(student_block.gate.weight))
    return teacher_block, student_block


def test_real_qwen3_5_moe_block_full_replay():
    teacher_block, student_block = _qwen3_5_moe_block_pair(_qwen3_5_moe_config())
    replay = RoutingReplay(teacher_block, student_block, mode="full")
    hidden = torch.randn(2, 3, HIDDEN)

    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        free_out = student_block(hidden)
    assert replay.metrics()["routing_flip_rate"] > 0
    assert not torch.allclose(free_out, teacher_out)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        replay_out = student_block(hidden)
    # Full replay + identical experts and shared expert => exactly the teacher's output.
    torch.testing.assert_close(replay_out, teacher_out)


def test_mask_mode_renorm_matches_router_math():
    teacher_block, student_block = _qwen3_5_moe_block_pair(_qwen3_5_moe_config())
    # Our mask-mode hook renormalizes via getattr(module, "norm_topk_prob", True): the
    # Qwen3.5-MoE router renormalizes unconditionally and exposes no such flag. Fail
    # loudly if a future transformers version adds one (the default may then be wrong).
    assert not hasattr(student_block.gate, "norm_topk_prob")

    replay = RoutingReplay(teacher_block, student_block, mode="mask")
    received = {}

    def capture(module, args):
        received["indices"], received["weights"] = args[1], args[2]

    def capture_teacher(module, args):
        received["teacher_indices"] = args[1]

    student_block.experts.register_forward_pre_hook(capture)
    teacher_block.experts.register_forward_pre_hook(capture_teacher)
    hidden = torch.randn(2, 3, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher_block(hidden)
        student_block(hidden)

    assert torch.equal(received["indices"], received["teacher_indices"])
    logits = F.linear(hidden.reshape(-1, HIDDEN), student_block.gate.weight)
    probs = torch.softmax(logits.float(), dim=-1)
    gathered = probs.gather(-1, received["teacher_indices"])
    expected = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(replay.eps)
    # Bit-wise: the hook recomputes exactly this renormalization from the router logits.
    assert torch.equal(received["weights"], expected.to(received["weights"].dtype))


def test_metrics_at_256_experts():
    replay = RoutingReplay.__new__(RoutingReplay)
    replay._supervision_mask = None
    replay._stats = {}
    tokens, top_k = 9, 8
    teacher = torch.arange(tokens * top_k).reshape(tokens, top_k)  # experts 0..71 of 256
    student = teacher.clone()
    student[:3, 0] = 200 + torch.arange(3)  # flip exactly 1 of 8 experts on 1/3 of tokens
    student[3:] = student[3:].flip(-1)  # permutation only: must not count as a flip

    replay._update_stats("layers.0.mlp.gate", student, teacher)
    metrics = replay.metrics()
    # flip = per-token set difference; jaccard = |inter| / |union| = 7/9 flipped, 1 else.
    torch.testing.assert_close(metrics["routing_flip_rate"], torch.tensor(1 / 3))
    torch.testing.assert_close(metrics["routing_jaccard"], torch.tensor((3 * (7 / 9) + 6) / 9))
    torch.testing.assert_close(metrics["routing_flipped_layers_per_token"], torch.tensor(1 / 3))

    # Second layer without flips: rates average over layer-tokens, flipped-layers per token.
    replay._update_stats("layers.1.mlp.gate", teacher, teacher)
    metrics = replay.metrics()
    torch.testing.assert_close(metrics["routing_flip_rate"], torch.tensor(3 / 18))
    torch.testing.assert_close(metrics["routing_jaccard"], torch.tensor((3 * (7 / 9) + 15) / 18))
    torch.testing.assert_close(metrics["routing_flipped_layers_per_token"], torch.tensor(1 / 3))

    # Supervision mask keeps the 3 flipped tokens and 3 unflipped ones.
    replay._supervision_mask = torch.tensor([True] * 6 + [False] * 3)
    replay._stats = {}
    replay._update_stats("layers.0.mlp.gate", student, teacher)
    metrics = replay.metrics()
    torch.testing.assert_close(metrics["routing_flip_rate"], torch.tensor(3 / 6))
    torch.testing.assert_close(metrics["routing_jaccard"], torch.tensor((3 * (7 / 9) + 3) / 6))
    torch.testing.assert_close(metrics["routing_flipped_layers_per_token"], torch.tensor(3 / 6))


def test_replay_discovery_and_gradckpt_tiny_causallm(torch_delta_net):
    config = _qwen3_5_moe_config()
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    teacher = Qwen3_5MoeForCausalLM(config)
    torch.manual_seed(0)
    student = Qwen3_5MoeForCausalLM(config)
    with torch.no_grad():
        for layer in student.model.layers:
            layer.mlp.gate.weight.add_(0.5 * torch.randn_like(layer.mlp.gate.weight))

    replay = RoutingReplay(teacher, student, mode="mask")
    expected_routers = {f"model.layers.{index}.mlp.gate" for index in range(4)}
    assert set(replay.router_names) == expected_routers
    assert not any("shared_expert" in name or "linear_attn" in name for name in replay.router_names)

    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher(input_ids=input_ids, use_cache=False)
    loss = student(input_ids=input_ids, labels=input_ids.clone(), use_cache=False).loss
    before = replay.metrics()
    loss.backward()  # recomputes the checkpointed layers, re-running the replay hooks
    after = replay.metrics()

    assert torch.isfinite(loss)
    assert before["routing_flip_rate"] > 0
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_free_routing_pins_indices_across_recompute():
    """Under free routing with grad enabled, the step's first-pass routing must be
    pinned across a gradient-checkpoint recompute: kernel-level float jitter can flip
    a top-k boundary on recompute, changing expert-dispatch shapes and tripping
    torch's saved-tensor metadata check (killed the 35B W4 run at step 3507). A gate
    perturbation between the two forwards stands in for that jitter."""

    teacher_block, student_block = _qwen3_5_moe_block_pair(_qwen3_5_moe_config())
    replay = RoutingReplay(teacher_block, student_block, mode="mask")
    hidden = torch.randn(2, 3, HIDDEN)

    received = []

    def capture(module, args):
        received.append(args[1].detach().clone())

    student_block.experts.register_forward_pre_hook(capture)

    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_block(hidden)
    first_out = student_block(hidden)  # grad enabled: pass 1 pins its own routing

    # Perturb the gate so natural routing would flip, then "recompute".
    with torch.no_grad():
        student_block.gate.weight.add_(torch.randn_like(student_block.gate.weight))
    second_out = student_block(hidden)

    assert torch.equal(received[1], received[0])  # dispatch pinned to pass-1 routing
    assert first_out.shape == second_out.shape

    # A plain no-grad forward (eval) is never recomputed: routing must be native,
    # i.e. reflect the perturbed gate rather than the pinned indices.
    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_block(hidden)
        student_block(hidden)
    assert not torch.equal(received[2], received[0])

    replay.remove()


def test_free_routing_pass1_scores_match_native():
    """Pass-1 of the pinned free-routing path must be numerically identical to the
    unhooked block (same softmax+gather+renorm the router itself performs)."""

    config = _qwen3_5_moe_config()
    torch.manual_seed(7)
    teacher_block, student_block = _qwen3_5_moe_block_pair(config)
    hidden = torch.randn(2, 3, HIDDEN)

    native_out = student_block(hidden)  # before hooks attach

    replay = RoutingReplay(teacher_block, student_block, mode="mask")
    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_block(hidden)
    hooked_out = student_block(hidden)  # grad enabled -> pinned path, own indices

    torch.testing.assert_close(hooked_out, native_out.detach())
    replay.remove()


def _minimax_m2_config():
    pytest.importorskip("transformers.models.minimax_m2")
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

    return MiniMaxM2Config(
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def _minimax_m2_block_pair(config, perturb=0.5, seed=0):
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2SparseMoeBlock

    torch.manual_seed(seed)
    teacher_block = MiniMaxM2SparseMoeBlock(config)
    with torch.no_grad():
        for parameter in teacher_block.parameters():
            # Bare-module construction leaves torch.empty parameters uninitialized.
            nn.init.normal_(parameter, std=0.1)
        # Nonzero correction bias so top-k selection differs from bias-free sigmoid top-k.
        teacher_block.e_score_correction_bias.normal_(std=0.5)
    student_block = MiniMaxM2SparseMoeBlock(config)
    student_block.load_state_dict(teacher_block.state_dict())
    with torch.no_grad():
        student_block.gate.weight.add_(perturb * torch.randn_like(student_block.gate.weight))
    return teacher_block, student_block


def test_real_minimax_m2_block_attach_and_full_replay():
    teacher_block, student_block = _minimax_m2_block_pair(_minimax_m2_config())
    replay = RoutingReplay(teacher_block, student_block, mode="full")
    # The two-argument sigmoid router (probed with a zero e_score_correction_bias) must
    # be classified as logits-first and sigmoid-scored.
    assert not replay._first_is_probs
    assert replay._sigmoid_scoring
    hidden = torch.randn(2, 3, HIDDEN)

    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        free_out = student_block(hidden)
    assert replay.metrics()["routing_flip_rate"] > 0
    assert not torch.allclose(free_out, teacher_out)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher_out = teacher_block(hidden)
        replay_out = student_block(hidden)
    # Full replay + identical experts and bias => exactly the teacher's output.
    torch.testing.assert_close(replay_out, teacher_out)


def test_minimax_m2_mask_mode_sigmoid_renorm_matches_router_math():
    teacher_block, student_block = _minimax_m2_block_pair(_minimax_m2_config())
    # Our mask-mode hook renormalizes via getattr(module, "norm_topk_prob", True): the
    # MiniMaxM2 router renormalizes unconditionally and exposes no such flag. Fail
    # loudly if a future transformers version adds one (the default may then be wrong).
    assert not hasattr(student_block.gate, "norm_topk_prob")

    replay = RoutingReplay(teacher_block, student_block, mode="mask")
    received = {}

    def capture(module, args):
        received["indices"], received["weights"] = args[1], args[2]

    def capture_teacher(module, args):
        received["teacher_indices"] = args[1]

    student_block.experts.register_forward_pre_hook(capture)
    teacher_block.experts.register_forward_pre_hook(capture_teacher)
    hidden = torch.randn(2, 3, HIDDEN)

    replay.begin_batch(replay_active=True)
    with torch.no_grad():
        teacher_block(hidden)
        student_block(hidden)

    assert torch.equal(received["indices"], received["teacher_indices"])
    # The bias must have shifted at least one token's SELECTION away from bias-free
    # sigmoid top-k (it participates in selection only, never in the weights below).
    bias_free = torch.topk(
        torch.sigmoid(F.linear(hidden.reshape(-1, HIDDEN), teacher_block.gate.weight).float()), TOP_K, dim=-1
    ).indices
    assert (bias_free.sort(-1).values != received["teacher_indices"].sort(-1).values).any()
    logits = F.linear(hidden.reshape(-1, HIDDEN), student_block.gate.weight)
    scores = torch.sigmoid(logits.float())  # no e_score_correction_bias in the weights
    gathered = scores.gather(-1, received["teacher_indices"])
    expected = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(replay.eps)
    # Bit-wise: the hook recomputes exactly this renormalization from the router logits.
    assert torch.equal(received["weights"], expected.to(received["weights"].dtype))


def test_minimax_m2_free_routing_pinned_path_bit_identical():
    """Self-pinning at replay_p=0 must leave sigmoid gating BIT-IDENTICAL to the
    unhooked forward: the pinned path recomputes exactly the router's own bias-free
    sigmoid-gather-renormalize on the pass-1 indices."""

    config = _minimax_m2_config()
    torch.manual_seed(7)
    teacher_block, student_block = _minimax_m2_block_pair(config)
    hidden = torch.randn(2, 3, HIDDEN)

    native_out = student_block(hidden)  # before hooks attach

    replay = RoutingReplay(teacher_block, student_block, mode="mask")
    replay.begin_batch(replay_active=False)
    with torch.no_grad():
        teacher_block(hidden)
    hooked_out = student_block(hidden)  # grad enabled -> pinned path, own indices

    assert torch.equal(hooked_out, native_out)
    replay.remove()


def test_minimax_m2_causallm_discovery_and_config_detection():
    config = _minimax_m2_config()
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2ForCausalLM

    torch.manual_seed(0)
    teacher = MiniMaxM2ForCausalLM(config)
    torch.manual_seed(0)
    student = MiniMaxM2ForCausalLM(config)

    replay = RoutingReplay(teacher, student, mode="mask")
    assert set(replay.router_names) == {f"model.layers.{index}.mlp.gate" for index in range(2)}
    # Scoring is decided by the model config here (model_type == "minimax_m2").
    assert replay._sigmoid_scoring and not replay._first_is_probs
    replay.remove()
