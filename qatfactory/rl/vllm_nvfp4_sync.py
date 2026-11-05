"""Publish fake-quantized latent weights into a live, colocated vLLM NVFP4 engine.

The rollout engine serves a ``compressed-tensors nvfp4-pack-quantized`` checkpoint, so by
the time it is running there is nothing BF16 left to overwrite: vLLM's
``process_weights_after_loading`` has already renamed ``weight_packed`` to ``weight``,
swizzled the block scales into the kernel's blocked layout and folded the global scales
into a scalar ``alpha``. TRL's stock parameter-by-parameter sync therefore cannot be used;
this module re-derives those exact tensors after every optimizer step and copies them into
the live parameters **in place**, which keeps vLLM's captured CUDA graphs pointing at valid
storage.

Two properties make the result trustworthy rather than merely plausible:

- **One tensor scale, two consumers.** The NVFP4 tensor scale is read from the training
  model's own :class:`~qatfactory.quant.formats.nvfp4.NVFP4AmaxGroup` — the same cached,
  globally-reduced amax the straight-through forward quantizes with — instead of being
  recomputed here. The rollout policy and the training policy therefore sit on an identical
  lattice by construction. The publisher refuses to start unless every vLLM fused layer's
  members are exactly one QAD fused family (q/k/v, gate/up, or a singleton).
- **vLLM owns the layout.** Rather than reimplementing the swizzle/pad/alpha math, each
  update runs vLLM's own ``process_weights_after_loading`` against a scratch layer holding
  the freshly packed tensors, then copies the results across. Whatever kernel vLLM selected
  (cutlass, flashinfer, marlin, ...) defines the layout, so there is no second
  implementation to drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from ..quant.formats.lora import merge_lora_weight
from ..quant.formats.nvfp4 import NVFP4Linear, quantize_nvfp4


logger = logging.getLogger(__name__)

# nn.Module's own bookkeeping keys; everything else on a live vLLM layer is layer metadata
# (logical_widths, input_size_per_partition, scheme, ...) that the scratch layer must carry.
_MODULE_INTERNALS = frozenset(nn.Module().__dict__)

# Wrapper prefixes that activation checkpointing / FSDP can splice into module paths.
_WRAPPER_PREFIXES = ("_checkpoint_wrapped_module.", "_fsdp_wrapped_module.", "_orig_mod.")

# Live parameters recomputed from the latent weights on every sync. With static activation
# scales `input_global_scale`/`input_global_scale_inv` are deliberately absent, so they are
# never rewritten and cannot drift; scale sync adds them (see _published_names).
_PUBLISHED = ("weight", "weight_scale", "weight_global_scale", "alpha")
_PUBLISHED_WITH_INPUT_SCALES = (*_PUBLISHED, "input_global_scale", "input_global_scale_inv")


class _ScratchLayer:
    """A stand-in layer that vLLM's post-load transform can be run against.

    Deliberately not an ``nn.Module``: the transform reassigns and ``del``s attributes
    (``layer.weight = layer.weight_packed; del layer.weight_packed``), which is plain
    attribute manipulation on a bare object and would otherwise have to fight
    ``register_parameter`` bookkeeping.
    """

    def __init__(self, live: nn.Module):
        for key, value in live.__dict__.items():
            if key not in _MODULE_INTERNALS:
                setattr(self, key, value)


@dataclass
class _FusedTarget:
    """One vLLM quantized linear and the training-model linears whose rows it holds."""

    vllm_name: str
    vllm_layer: nn.Module
    member_names: tuple[str, ...]
    members: tuple[NVFP4Linear, ...]
    amax_group: object
    # The `input_global_scale` exactly as the checkpoint stored it (a reciprocal, per output
    # partition). Captured from the live layer once, so rebuilding the pre-transform inputs
    # never round-trips through a division and cannot drift over thousands of syncs.
    input_global_scale_loaded: torch.Tensor | None

    @property
    def row_counts(self) -> tuple[int, ...]:
        return tuple(member.out_features for member in self.members)


def _strip_wrappers(name: str) -> str:
    for prefix in _WRAPPER_PREFIXES:
        name = name.replace(prefix, "")
    return name


def _resolve_module(name: str, candidates: dict[str, nn.Module]) -> nn.Module | None:
    """Look up ``name`` in the training model, tolerating a differing root prefix.

    A vLLM layer is named ``model.layers.3.self_attn.q_proj`` while a multimodal-wrapped
    training model may hold it at ``model.language_model.layers.3.self_attn.q_proj``.
    Matching on the unique layer-qualified suffix keeps this convention-agnostic.
    """

    direct = candidates.get(name)
    if direct is not None:
        return direct
    core = name.split(".", 1)[1] if "." in name else name
    matches = [module for key, module in candidates.items() if key == core or key.endswith(f".{core}")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"vLLM layer {name!r} matches {len(matches)} training modules; cannot map unambiguously.")
    return None


def _packed_modules_mapping(vllm_model: nn.Module) -> dict[str, list[str]]:
    mapping = getattr(type(vllm_model), "packed_modules_mapping", None) or {}
    return {fused: list(sources) for fused, sources in mapping.items()}


def _full_weight(module: NVFP4Linear) -> torch.Tensor:
    """The module's full (unsharded) latent weight, with any LoRA adapter folded in.

    Mirrors what the straight-through forward quantizes: ``W`` normally, ``W + scaling *
    B @ A`` under LoRA-QAD. ``full_tensor`` is collective, so callers must visit modules in
    the same order on every rank.
    """

    weight = module.weight
    weight = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    lora_a = getattr(module, "lora_A", None)
    if lora_a is None:
        return weight.detach()
    lora_b = module.lora_B
    lora_a = lora_a.full_tensor() if hasattr(lora_a, "full_tensor") else lora_a
    lora_b = lora_b.full_tensor() if hasattr(lora_b, "full_tensor") else lora_b
    return merge_lora_weight(weight.detach(), lora_a.detach(), lora_b.detach(), module.lora_scaling)


class NVFP4WeightPublisher:
    """Keeps a colocated vLLM NVFP4 engine in sync with the fake-quantized training model.

    Args:
        hf_model: the training model (already instrumented by ``apply_fake_quant``).
        vllm_model: the live in-process vLLM model (colocate mode, ``external_launcher``).
        sync_input_scales: also publish activation scales, taken from the running maxima the
            training forward records, instead of leaving the export's static values in
            place. Requires the student to have been loaded with running-max tracking on.
        warmup_steps: optimizer steps to wait before the first activation-scale publish.
            The running maximum only grows, so an early publish is not unsafe, but it moves
            the rollout policy on every sync while the estimate is still climbing; waiting
            lets it settle first.
        step_provider: returns the current optimizer step; must agree across ranks, since
            the scale reduction is collective.
        process_group: process group for the cross-rank scale reduction.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        vllm_model: nn.Module,
        *,
        sync_input_scales: bool = False,
        warmup_steps: int = 0,
        step_provider=None,
        process_group=None,
    ):
        self.hf_model = hf_model
        self.vllm_model = vllm_model
        self.sync_input_scales = bool(sync_input_scales)
        self.warmup_steps = int(warmup_steps)
        self.step_provider = step_provider
        self.process_group = process_group
        self.targets = self._build_targets(hf_model, vllm_model)
        self._validate_amax_families()
        if self.sync_input_scales:
            self._validate_tracking()
        logger.info(
            "NVFP4 weight publisher ready: %d vLLM layers covering %d quantized linears "
            "(activation scales: %s).",
            len(self.targets),
            sum(len(target.members) for target in self.targets),
            f"synced after {self.warmup_steps} steps" if self.sync_input_scales else "static",
        )

    def _validate_tracking(self) -> None:
        """Refuse to 'sync' scales that nothing is updating.

        Without running-max tracking the publisher would faithfully republish the seeded
        export value forever, which looks like the feature working and is not.
        """

        untracked = [
            name
            for target in self.targets
            for name, module in zip(target.member_names, target.members)
            if not getattr(module, "_track_input_amax", False)
        ]
        if untracked:
            raise RuntimeError(
                f"Activation-scale sync is on but {len(untracked)} linears have running-max tracking "
                f"disabled, so their scales would never change (e.g. {untracked[:3]}). Load the student "
                "with track_activation_scales=True."
            )

    @staticmethod
    def _build_targets(hf_model: nn.Module, vllm_model: nn.Module) -> list[_FusedTarget]:
        hf_linears = {
            _strip_wrappers(name): module
            for name, module in hf_model.named_modules()
            if isinstance(module, NVFP4Linear)
        }
        if not hf_linears:
            raise ValueError("Training model has no NVFP4Linear modules; was apply_fake_quant applied?")
        packed_mapping = _packed_modules_mapping(vllm_model)

        targets: list[_FusedTarget] = []
        claimed: set[str] = set()
        for vllm_name, layer in vllm_model.named_modules():
            if getattr(layer, "scheme", None) is None or not hasattr(layer, "weight_scale"):
                continue
            parent, _, leaf = vllm_name.rpartition(".")
            source_leaves = packed_mapping.get(leaf, [leaf])
            member_names: list[str] = []
            members: list[NVFP4Linear] = []
            for source_leaf in source_leaves:
                source_name = f"{parent}.{source_leaf}" if parent else source_leaf
                module = _resolve_module(source_name, hf_linears)
                if module is None:
                    continue
                member_names.append(source_name)
                members.append(module)
            if not members:
                raise RuntimeError(
                    f"vLLM quantized layer {vllm_name!r} has no matching fake-quantized linear in the "
                    f"training model (looked for {source_leaves} under {parent!r}). The rollout export "
                    "and the training model must share an architecture."
                )
            claimed.update(member_names)
            groups = {id(member._amax_group) for member in members}
            if len(groups) != 1 or members[0]._amax_group is None:
                raise RuntimeError(
                    f"vLLM layer {vllm_name!r} spans {len(groups)} NVFP4 amax groups; the training model "
                    "must be built with fused_runtime_scales=True so each fused layer shares one scale."
                )
            targets.append(
                _FusedTarget(
                    vllm_name=vllm_name,
                    vllm_layer=layer,
                    member_names=tuple(member_names),
                    members=tuple(members),
                    amax_group=members[0]._amax_group,
                    input_global_scale_loaded=(
                        layer.input_global_scale_inv.detach().clone()
                        if hasattr(layer, "input_global_scale_inv")
                        else None
                    ),
                )
            )

        unclaimed = sorted(set(hf_linears) - claimed)
        if unclaimed:
            raise RuntimeError(
                f"{len(unclaimed)} fake-quantized linears have no vLLM counterpart and would never reach "
                f"the rollout engine: {unclaimed[:5]}."
            )
        if not targets:
            raise RuntimeError("The vLLM engine exposes no quantized linears; is it serving the NVFP4 export?")
        return targets

    def _validate_amax_families(self) -> None:
        """Require each vLLM fused layer to be exactly one QAD fused-scale family.

        This is what ties the rollout lattice to the training lattice: the scale the
        straight-through forward uses is a per-family maximum, so it is the correct scale for
        the packed layer only if the family and the fused layer contain the same members.
        """

        for target in self.targets:
            family = set(target.amax_group.member_names)
            members = set(target.member_names)
            if family != members:
                raise RuntimeError(
                    f"vLLM layer {target.vllm_name!r} fuses {sorted(members)} but the training model's "
                    f"NVFP4 amax family is {sorted(family)}. The shared tensor scale would not match the "
                    "rollout weights."
                )

    def _input_scales_due(self) -> bool:
        if not self.sync_input_scales:
            return False
        step = self.step_provider() if self.step_provider is not None else 0
        if int(step) < self.warmup_steps:
            return False
        # The running maxima start empty so they measure this model rather than inheriting
        # the export's. TRL syncs before the first training forward, so with a zero warmup
        # there may be nothing observed yet; publishing is simply deferred a step. The check
        # is on a model-wide invariant, so it stays consistent across ranks.
        unobserved = any(
            module._input_amax is None for target in self.targets for module in target.members
        )
        if unobserved:
            logger.info("Deferring activation-scale sync: no input observations recorded yet.")
            return False
        return True

    @torch.no_grad()
    def sync(self) -> int:
        """Re-pack every quantized layer from the current latents and publish it.

        Collective (``full_tensor``, and the activation-scale reduction) and therefore must
        be called on every rank, which the TRL generation hook does. Returns the number of
        layers updated.
        """

        input_scales = None
        if self._input_scales_due():
            # Same helper the exporter uses: per-fused-family maximum, all-reduced across
            # ranks and written back, so every engine quantizes activations identically and
            # the scale synced here is the one that lands in input_scales.safetensors.
            from ..quant.formats.nvfp4 import finalize_nvfp4_input_scales

            input_scales = finalize_nvfp4_input_scales(self.hf_model, self.process_group)

        for target in self.targets:
            weights = [_full_weight(member) for member in target.members]
            weight = weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)
            del weights
            # The same cached, globally-reduced amax the straight-through forward quantizes
            # with — not a fresh reduction — so rollout and training share one lattice.
            tensor_amax = target.amax_group.tensor_amax().to(weight.device)
            packed = quantize_nvfp4(weight, tensor_amax=tensor_amax, compute_reconstruction=False)
            del weight
            self._publish(target, packed, input_scales)
        return len(self.targets)

    def _publish(self, target: _FusedTarget, packed, input_scales: dict | None = None) -> None:
        live = target.vllm_layer
        scratch = _ScratchLayer(live)
        partitions = len(getattr(live, "logical_widths", None) or [1])
        # Rebuild the layer's pre-transform (checkpoint-order) tensors: compressed-tensors
        # stores both global scales as reciprocals, one per output partition.
        scratch.weight_packed = packed.packed_weight
        scratch.weight_scale = packed.weight_scale
        scratch.weight_global_scale = packed.weight_scale_2.reshape(1).float().reciprocal().repeat(partitions)
        loaded_input_scale = target.input_global_scale_loaded
        published = _PUBLISHED
        if input_scales is not None and loaded_input_scale is not None:
            # Members of a fused layer share one input tensor and one family scale, so any
            # member's entry is the layer's. Derived fresh from the amax each time rather
            # than from the previously published value, so repeated syncs cannot drift.
            family_scale = input_scales[f"{target.member_names[0]}.input_scale"]
            loaded_input_scale = family_scale.to(loaded_input_scale).reciprocal()
            published = _PUBLISHED_WITH_INPUT_SCALES
        if loaded_input_scale is not None:
            scratch.input_global_scale = loaded_input_scale.reshape(1).repeat(partitions)

        # Let vLLM's own scheme + kernel decide the deployed layout, so no swizzle, padding
        # or alpha arithmetic is duplicated (and re-derived) here.
        live.scheme.process_weights_after_loading(scratch)

        for name in published:
            current = getattr(live, name, None)
            if current is None:
                # Weight-only (a16) serving has no input scale and therefore no alpha.
                continue
            updated = getattr(scratch, name, None)
            if updated is None:
                raise RuntimeError(
                    f"vLLM's post-load transform did not produce {name!r} for {target.vllm_name}, so the "
                    "live parameter would silently keep its previous value."
                )
            updated = updated.data if isinstance(updated, nn.Parameter) else updated
            if updated.shape != current.shape or updated.dtype != current.dtype:
                raise RuntimeError(
                    f"Refusing to publish {target.vllm_name}.{name}: rebuilt {tuple(updated.shape)}/"
                    f"{updated.dtype} does not match the live {tuple(current.shape)}/{current.dtype}."
                )
            current.data.copy_(updated)


def vllm_model_from_engine(llm) -> nn.Module:
    """Reach the live model inside a colocated vLLM engine.

    Only valid for TRL's colocate mode, which runs the engine in-process with
    ``distributed_executor_backend="external_launcher"``; the default multiprocess engine
    keeps the model in a separate process where in-place tensor updates are impossible.
    """

    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError(
            "Could not reach the vLLM model in-process. QARL weight sync requires colocate mode "
            "(distributed_executor_backend='external_launcher')."
        ) from exc
