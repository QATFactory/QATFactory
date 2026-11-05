"""Build the QARL student: a fake-quantized policy pinned to a rollout export's scales.

Unlike distillation there is no teacher — the reward signal replaces it — so the only model
loaded here is the student. Its latent weights come from an ordinary Hugging Face causal LM
(``Qwen/Qwen3-8B``, or a QAD checkpoint to continue from), while the NVFP4 export supplies
the rollout engine's architecture, quantization config and, crucially, its **static
activation scales**.

Those activation scales are read out of the export and frozen for the whole run. vLLM
quantizes W4A4 activations against a static per-tensor ``input_global_scale`` baked into the
served checkpoint, and nothing in the rollout loop can update it; pinning the training side
to the same numbers is what keeps the two policies comparable, and it means the checkpoint
saved at the end still deploys with the activation scales it was trained under.
"""

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open

from ..models.llm import _untie_frozen_lm_head, load_causal_lm, upcast_trainable_to_fp32
from ..quant.fake_quant import DEFAULT_EXCLUDE_PREFIXES, apply_fake_quant
from ..quant.formats.nvfp4 import NVFP4_SCALE_DENOMINATOR, NVFP4Linear


logger = logging.getLogger(__name__)

# Config fields that must agree between the training latents and the rollout export, or the
# packed weights we publish would not describe the model vLLM is running.
_ARCHITECTURE_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
)


def _text_config(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def validate_rollout_export(model_dir: str, rollout_dir: str) -> dict:
    """Check that the rollout export is an NVFP4 checkpoint of the same architecture.

    Returns the export's ``config.json``.
    """

    rollout_config_path = Path(rollout_dir) / "config.json"
    if not rollout_config_path.is_file():
        raise FileNotFoundError(f"Rollout model {rollout_dir} has no config.json.")
    with rollout_config_path.open(encoding="utf-8") as handle:
        rollout_config = json.load(handle)

    quantization = rollout_config.get("quantization_config") or {}
    if quantization.get("format") != "nvfp4-pack-quantized":
        raise ValueError(
            f"Rollout model {rollout_dir} is not an NVFP4 checkpoint "
            f"(quantization_config.format={quantization.get('format')!r}). QARL rolls out on the "
            "packed NVFP4 model so the sampling policy matches deployment."
        )

    with (Path(model_dir) / "config.json").open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    latent, rollout = _text_config(model_config), _text_config(rollout_config)
    mismatched = {
        field: (latent.get(field), rollout.get(field))
        for field in _ARCHITECTURE_FIELDS
        if latent.get(field) != rollout.get(field)
    }
    if mismatched:
        raise ValueError(
            f"Training latents ({model_dir}) and rollout export ({rollout_dir}) describe different "
            f"architectures: {mismatched}."
        )
    return rollout_config


def verify_frozen_weights(model_dir: str, rollout_dir: str) -> int:
    """Require the export's non-quantized tensors to equal the training latents' on disk.

    QARL only ever pushes the quantized linears into the rollout engine — every other tensor
    (embeddings, norms, ``lm_head``) is frozen and comes from whatever export was handed to
    vLLM. If the export was built from different latents, the sampling policy would silently
    differ from the trained one in exactly the places the sync never touches. Comparing the
    two checkpoints directly is a cheaper and far more direct guarantee than trying to
    reconcile parameter names inside a running engine.

    Returns the number of tensors compared.
    """

    from ..quant.export.nvfp4_vllm import SafetensorDirectory

    packed_suffixes = (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")
    rollout_store = SafetensorDirectory(rollout_dir)
    latent_store = SafetensorDirectory(model_dir)
    latent_keys = latent_store.keys()

    compared = 0
    for key in sorted(rollout_store.keys()):
        if key.endswith(packed_suffixes):
            continue
        if key not in latent_keys:
            raise ValueError(
                f"Rollout export {rollout_dir} carries frozen tensor {key!r}, which the training latents "
                f"({model_dir}) do not have. The two must describe the same model."
            )
        if not torch.equal(rollout_store.get_tensor(key), latent_store.get_tensor(key)):
            raise ValueError(
                f"Frozen tensor {key!r} differs between the training latents ({model_dir}) and the rollout "
                f"export ({rollout_dir}). QARL never syncs frozen tensors, so the rollout policy would not "
                "match the trained policy. Export the rollout model from these latents."
            )
        compared += 1
    logger.info("Verified %d frozen tensors match between latents and rollout export.", compared)
    return compared


def read_export_input_amax(rollout_dir: str) -> dict[str, torch.Tensor]:
    """Recover per-module activation maxima from a packed export's ``input_global_scale``.

    The exporter stores the reciprocal of ``amax / (6 * 448)``, so this inverts that to the
    raw maximum the fake quantizer expects as its static tensor scale.
    """

    rollout_path = Path(rollout_dir)
    index_path = rollout_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        shards = {rollout_path / shard for key, shard in weight_map.items() if key.endswith("input_global_scale")}
    else:
        shards = set(rollout_path.glob("*.safetensors"))

    amax: dict[str, torch.Tensor] = {}
    for shard in sorted(shards):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                if not key.endswith(".input_global_scale"):
                    continue
                scale = handle.get_tensor(key).reshape(()).float()
                if not bool(torch.isfinite(scale)) or not bool(scale > 0):
                    raise ValueError(f"Rollout export has a non-positive input_global_scale for {key}.")
                amax[key.removesuffix(".input_global_scale")] = NVFP4_SCALE_DENOMINATOR / scale
    if not amax:
        raise ValueError(
            f"Rollout export {rollout_dir} carries no input_global_scale tensors; it was not produced by a "
            "fused-runtime-scale QAD run and has no activation scales to pin training to."
        )
    return amax


def pin_activation_scales(model, export_amax: dict[str, torch.Tensor], *, freeze: bool = True) -> int:
    """Seed every quantized linear's activation scale from the rollout export.

    With ``freeze`` (the default) running-max tracking is switched off, so the scale the
    student fake-quantizes with, the scale vLLM serves with, and the scale written out at
    save time are one number for the whole run.

    With ``freeze=False`` the export's value seeds only the step-static scale (what W4A4
    training quantizes with until its first commit) while the *running* maximum starts
    empty, so it is measured purely from this model's activations. Seeding it too would
    make the tracked value a running max over ``max(export, observed)``, which can only
    ever grow: the scale could never come down to a genuinely smaller observed maximum, and
    a larger amax is a coarser activation grid. Starting empty lets the scale move in
    whichever direction the data says.
    """

    modules = {name: module for name, module in model.named_modules() if isinstance(module, NVFP4Linear)}
    missing = sorted(set(modules) - set(export_amax))
    if missing:
        raise ValueError(
            f"Rollout export has no activation scale for {len(missing)} quantized linears: {missing[:5]}."
        )
    device = next(model.parameters()).device
    for name, module in modules.items():
        amax = export_amax[name].to(device=device, dtype=torch.float32).reshape(())
        module._input_amax = amax.clone() if freeze else None
        module._input_amax_static = amax.clone()
        # Frozen: stop the forward from growing the running maximum, so the recorded scale
        # cannot drift away from the one the policy is actually served with.
        module._track_input_amax = not freeze
    logger.info(
        "Seeded %d activation scales from the rollout export (%s).",
        len(modules),
        "frozen" if freeze else "tracking a running maximum",
    )
    return len(modules)


def load_qarl_student(
    model_dir: str,
    rollout_dir: str,
    target_modules,
    *,
    quantize_activations: bool = False,
    track_activation_scales: bool = False,
    exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    dtype=torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """Return ``(student, converted)`` for quantization-aware RL.

    The student's target linears are NVFP4 fake-quantized with fused (per-family) runtime
    scales — required, because the rollout engine fuses q/k/v and gate/up into single
    packed layers that carry one tensor scale each. ``quantize_activations`` additionally
    fake-quantizes the linear inputs (W4A4), matching how vLLM serves an export whose
    ``quantization_config`` declares input activations.
    """

    validate_rollout_export(model_dir, rollout_dir)
    verify_frozen_weights(model_dir, rollout_dir)
    student = load_causal_lm(model_dir, dtype=dtype, attn_implementation=attn_implementation)
    converted = apply_fake_quant(
        student,
        target_modules,
        quant_format="nvfp4",
        exclude_prefixes=exclude_prefixes,
        fused_runtime_scales=True,
        quantize_activations=quantize_activations,
    )
    if _untie_frozen_lm_head(student):
        logger.info("Untied lm_head from input embeddings (kept frozen) for FSDP2 compatibility.")

    pin_activation_scales(student, read_export_input_amax(rollout_dir), freeze=not track_activation_scales)
    upcast_trainable_to_fp32(student)  # same FP32-master requirement as QAD; see models.llm

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    logger.info(
        "Loaded QARL student: %d/%d params trainable (%.2f%%), %d quantized linears, %s.",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
        len(converted),
        "W4A4 (weight + input-activation)" if quantize_activations else "weight-only (W4A16)",
    )
    return student, converted
