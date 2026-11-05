"""Thin Hugging Face adapters: load a causal LM, inject fake-quant, build a frozen teacher.

No model reimplementation. We load whichever stock auto class the checkpoint is registered
with (causal-LM or multimodal) and only swap target linears for their fake-quant variant on
the student.
"""

import logging
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from ..quant.fake_quant import DEFAULT_EXCLUDE_PREFIXES, apply_fake_quant


logger = logging.getLogger(__name__)

# A multimodal checkpoint's language tower is still an ordinary causal LM, but transformers
# registers the wrapper under a different auto class, so asking for `AutoModelForCausalLM`
# raises instead of loading. Order matters: text-only checkpoints must keep taking the first.
_AUTO_MODEL_CLASSES = (AutoModelForCausalLM, AutoModelForImageTextToText)


def _auto_model_class(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    for candidate in _AUTO_MODEL_CLASSES:
        if type(config) in candidate._model_mapping:
            return candidate
    supported = ", ".join(cls.__name__ for cls in _AUTO_MODEL_CLASSES)
    raise ValueError(
        f"{type(config).__name__} is not registered with any of {supported}; QAD needs a "
        "checkpoint whose language tower loads through one of them."
    )


def load_causal_lm(model_path: str, *, dtype=torch.bfloat16, attn_implementation: str = "flash_attention_2"):
    """Load a stock causal LM with the library's own implementation.

    ``trust_remote_code=False`` is explicit rather than incidental. Several released MoE
    checkpoints still ship an ``auto_map`` to a legacy per-expert ``ModuleList``; the shipped
    ``transformers`` implementation stores the same experts as fused 3D tensors, which is the
    layout the fake-quant instrumentation and the exporter are written against.

    A pre-quantized checkpoint (e.g. block-FP8) keeps its quantized weights: LoRA-QAD only
    reads the frozen base, and dequantizing the whole backbone up front would roughly double
    parameter memory for no numerical gain.
    """

    model_class = _auto_model_class(model_path)
    if cpu_ram_efficient_loading_enabled() and torch.distributed.get_rank() != 0:
        # Only the main rank's weights are used: FSDP broadcasts them into every shard. But
        # transformers (5.x) still reads the full checkpoint on every rank and then swaps in
        # a full-size zero copy, i.e. one host copy of the backbone per rank, which does not
        # fit for a 457 GB MiniMax-M2.7 at 8 ranks per node. Build the other ranks on meta.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        with torch.device("meta"):
            return model_class.from_config(
                config, dtype=dtype, attn_implementation=attn_implementation, trust_remote_code=False
            )
    return model_class.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )


def _untie_frozen_lm_head(model) -> bool:
    """Break a tied ``lm_head``/``embed_tokens`` into two independent frozen parameters.

    Weight-only (and W4A4) QAD never trains or quantizes the input embedding or ``lm_head``
    (both are excluded and frozen), so the tied tensor never diverges — but FSDP2 rejects a
    parameter shared across two shard groups (``embed_tokens`` under the text model vs the
    top-level ``lm_head``). Cloning ``lm_head`` into its own frozen parameter and clearing
    ``tie_word_embeddings`` removes the shared reference while leaving the numerics identical.

    The untie is purely a training-time mechanic and never reaches disk: because the head is a
    frozen clone of the embedding, :class:`QADTrainer.save_model` drops the redundant
    ``lm_head.weight`` and writes the original tied config, so the checkpoint stays consistent
    with the (tied) deployment layout. Returns True if an untie happened.
    """

    output = model.get_output_embeddings()
    inputs = model.get_input_embeddings()
    if output is None or inputs is None or output.weight is not inputs.weight:
        return False
    output.weight = torch.nn.Parameter(inputs.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    if text_config is not model.config:
        text_config.tie_word_embeddings = False
    return True


def _promote_persistent_buffers_for_fsdp2(model) -> list[str]:
    """Turn persistent float buffers into frozen parameters so FSDP2 can shard them.

    FSDP2 shards parameters but replicates buffers, while the full-state-dict load path
    assumes every ``state_dict`` entry is a sharded ``DTensor``. A persistent float buffer —
    MiniMax-M2 keeps its router's ``e_score_correction_bias`` as one — therefore breaks
    ``cpu_ram_efficient_loading`` before training starts.

    Promoting such a buffer to a frozen parameter changes nothing numerically or on disk (the
    key and value are unchanged, and QAD never trains it), it only lets FSDP2 shard it like
    everything else. Non-persistent buffers (rotary caches, masks) are left alone: they are
    already absent from the state dict and are rebuilt per model.
    """

    promoted = []
    for module_name, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for buffer_name, value in list(module.named_buffers(recurse=False)):
            if buffer_name in non_persistent or value is None or not value.is_floating_point():
                continue
            del module._buffers[buffer_name]
            module.register_parameter(buffer_name, torch.nn.Parameter(value, requires_grad=False))
            promoted.append(f"{module_name}.{buffer_name}" if module_name else buffer_name)
    return promoted


def load_student_and_teacher(
    model_path: str,
    target_modules,
    *,
    quant_format: str = "nvfp4",
    teacher_model_path: str | None = None,
    exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    fused_runtime_scales: bool = False,
    quantize_activations: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 0.0,
    lora_moe_hybrid_shared: bool = True,
    expert_checkpointing: bool = True,
    quantize_shared_expert: bool = True,
    lazy_teacher: bool = False,
    master_fp32: bool = True,
    dtype=torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """Return ``(student, teacher, converted)``.

    The student's target linears are fake-quantized (weights always; input activations
    too when ``quantize_activations`` — W4A4). Without LoRA the quantized latent weights
    train and the teacher is a separate, fully-frozen full-precision copy. With LoRA
    (``lora_rank > 0``) only the injected adapters train and the base stays frozen, so the
    teacher shares that base: ``teacher is None`` and the trainer runs the student model in
    teacher-mode for the full-precision pass. ``converted`` records the swapped module FQNs.

    ``lazy_teacher=True`` returns a zero-argument loader in place of the teacher module;
    :class:`~qatfactory.trainer.QADTrainer` calls it only after the student is sharded, so
    two full host copies of a large backbone never coexist on a rank. ``master_fp32=False``
    keeps the trainable latents in the load dtype (BF16); pair it with a stochastic-rounding
    optimizer (``train.low_bit_optim: adamw8bit_sr``), since plain BF16 updates underflow at
    recovery learning rates (see :func:`upcast_trainable_to_fp32`).
    """

    lora_enabled = bool(lora_rank and lora_rank > 0)
    student = load_causal_lm(model_path, dtype=dtype, attn_implementation=attn_implementation)
    converted = apply_fake_quant(
        student,
        target_modules,
        quant_format=quant_format,
        exclude_prefixes=exclude_prefixes,
        fused_runtime_scales=fused_runtime_scales,
        quantize_activations=quantize_activations,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_moe_hybrid_shared=lora_moe_hybrid_shared,
        expert_checkpointing=expert_checkpointing,
        quantize_shared_expert=quantize_shared_expert,
    )
    if quant_format in ("q4_k", "q4_k_grid_search"):
        from ..quant.formats.q4_k import configure_q4k_reorders

        reordered = configure_q4k_reorders(student)
        if reordered:
            logger.info("Configured %d Q4_K linear-attn GGUF reorder permutations.", reordered)

    if _untie_frozen_lm_head(student):
        logger.info("Untied lm_head from input embeddings (kept frozen) for FSDP2 compatibility.")

    promoted = _promote_persistent_buffers_for_fsdp2(student)
    if promoted:
        logger.info(
            "Promoted %d persistent buffers to frozen parameters for FSDP2 (e.g. %s).",
            len(promoted),
            promoted[0],
        )

    if lora_enabled:
        # Shared-base LoRA: the frozen base is the teacher (student run in teacher-mode),
        # so no second copy of the weights is loaded.
        if teacher_model_path:
            raise ValueError("teacher_model_path is not supported with LoRA-QAD (the base is shared).")
        teacher = None
        logger.info("LoRA-QAD: teacher shares the frozen base weights (no separate teacher copy).")
    else:

        def load_teacher():
            teacher = load_causal_lm(
                teacher_model_path or model_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
            teacher.requires_grad_(False)
            teacher.eval()
            _untie_frozen_lm_head(teacher)
            return teacher

        teacher = load_teacher if lazy_teacher else load_teacher()

    if master_fp32:
        upcast_trainable_to_fp32(student)
    else:
        logger.warning(
            "master_fp32=False: trainable latents stay in %s. Use a stochastic-rounding optimizer "
            "(train.low_bit_optim: adamw8bit_sr); deterministic BF16 updates underflow at 1e-6.",
            dtype,
        )

    if cpu_ram_efficient_loading_enabled():
        # The non-main ranks built the student on meta, so its buffers (RoPE inv_freq, quantizer
        # state) are meta too, and accelerate's FSDP2 prepare copies every non-persistent buffer
        # to the device as-is and rejects persistent ones in the broadcast state dict. Take the
        # main rank's buffers, as the teacher does; QADTrainer re-persists them once the student
        # is sharded so checkpoints still save them.
        device = torch.device("cuda", torch.cuda.current_device())
        student._qatf_depersisted_buffers = cpu_ram_efficient_buffer_fixup(student, device)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    logger.info(
        "Loaded QAD student: %d/%d params trainable (%.2f%%), %d quantized projections "
        "(%d dense linears, %d fused expert modules).",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
        len(converted),
        len(converted.linears),
        len(converted.experts),
    )
    return student, teacher, converted


def upcast_trainable_to_fp32(model: torch.nn.Module) -> int:
    """Give every trainable parameter an FP32 master copy; return the number of elements upcast.

    The recipes train with ``learning_rate`` around 1e-6 under BF16 mixed precision. A BF16
    parameter cannot absorb such an update: the step (~1e-6) is far below half a BF16 ulp for
    any weight larger than ~2.5e-4, so ``param.add_(update)`` rounds back to the old value and
    the optimizer state drifts while the weights stay put. The trainer's held-out KL then sits
    at the RTN value for the whole run. The design relies on FSDP mixed precision holding an
    FP32 master (the checkpoint save casts it back to BF16), and ``accelerate`` normally
    upcasts trainable parameters when ``mixed_precision`` is set — but 1.13.0's FSDP2 path
    rebinds a local instead of the parameter (``param = param.to(torch.float32)``) and only
    logs that it did, so the students it wrapped trained in BF16. Do the upcast here so the
    training precision does not depend on the launcher. Frozen parameters (teacher, shared
    LoRA base, embeddings, norms) stay in the load dtype.
    """

    upcast = 0
    for param in model.parameters():
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            upcast += param.numel()
    if upcast:
        logger.info("Upcast %d trainable parameter elements to FP32 master copies.", upcast)
    return upcast


def cpu_ram_efficient_loading_enabled() -> bool:
    """True when FSDP cpu-ram-efficient loading is active for this process group.

    Rank-independent by construction (environment and process-group state only), so every
    rank gets the same answer and collectives may be gated on it. Never gate a collective on
    rank-local model state such as ``any(p.is_meta ...)``: the rank holding real weights has
    no meta tensors, would skip the branch, and leave every other rank blocked in a broadcast.
    """

    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "0") not in ("0", "false", "False", "")
    )


def assign_buffer(model: torch.nn.Module, fqn: str, tensor: torch.Tensor) -> None:
    """Bind ``tensor`` as the buffer at dotted ``fqn`` on ``model``."""

    module = model
    *parents, leaf = fqn.split(".")
    for parent in parents:
        module = getattr(module, parent)
    module._buffers[leaf] = tensor


def cpu_ram_efficient_buffer_fixup(model: torch.nn.Module, device) -> list:
    """Make a partly-meta model safe to shard under cpu-ram-efficient loading.

    Ranks that built the model on ``meta`` hold meta buffers (e.g. RoPE ``inv_freq``), which
    cannot be copied, and persistent buffers stay plain tensors after ``fully_shard``, which
    the DTensor-only state-dict broadcast rejects. Broadcast every (tiny) buffer from global
    rank 0 and temporarily mark the persistent ones non-persistent; returns the
    ``(module, name)`` pairs to restore with :func:`repersist_buffers` once sharding is done.
    """

    depersisted = []
    for module in model.modules():
        non_persistent = module._non_persistent_buffers_set
        for name, value in list(module._buffers.items()):
            if value is not None and name not in non_persistent:
                non_persistent.add(name)
                depersisted.append((module, name))
    with torch.no_grad():
        for name, buffer in list(model.named_buffers()):
            real = (
                torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                if buffer.is_meta
                else buffer.detach().to(device)
            )
            torch.distributed.broadcast(real, src=0)
            assign_buffer(model, name, real)
    return depersisted


def repersist_buffers(depersisted: list) -> None:
    """Undo :func:`cpu_ram_efficient_buffer_fixup`'s de-persist so the buffers are saved again."""

    for module, name in depersisted:
        module._non_persistent_buffers_set.discard(name)
