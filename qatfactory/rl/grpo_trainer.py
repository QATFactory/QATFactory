"""GRPO for a fake-quantized policy that rolls out on its own packed NVFP4 checkpoint.

The training loop is TRL's, unmodified: same advantages, same clipped objective, same
importance-sampling correction for the sampler/trainer gap. Only two things change.

The rollout engine serves the packed NVFP4 export instead of the BF16 training model, so
the sampled policy is the deployed policy rather than an approximation of it. And weight
sync is replaced, because a running NVFP4 engine has no BF16 parameters left to overwrite —
see :mod:`qatfactory.rl.vllm_nvfp4_sync`.

Everything that makes the two policies agree is inherited rather than re-derived: the
per-family tensor scales come from the QAD amax groups that the straight-through forward
already uses, refreshed by the same :class:`FusedScaleRefreshCallback` the distillation
trainer runs, and the activation scales are pinned to the export at load time.
"""

import functools
import importlib
import logging
from contextlib import contextmanager

from trl import GRPOTrainer
from trl.generation.vllm_generation import VLLMGeneration

from ..quant.fake_quant import ConvertedModules
from ..trainer.callbacks import FusedScaleRefreshCallback
from ..trainer.checkpoint import QuantCheckpointMixin
from .vllm_nvfp4_sync import NVFP4WeightPublisher, vllm_model_from_engine


logger = logging.getLogger(__name__)


@contextmanager
def _named_as(model, path: str):
    """Temporarily make a model report ``path`` as where it was loaded from.

    ``name_or_path`` is a plain attribute on the model in transformers 5 and a property on
    the config in every version, and callers read either, so both are swapped and restored.
    """

    targets = [(model, "name_or_path"), (model.config, "_name_or_path")]
    originals = [(holder, attribute, getattr(holder, attribute, None)) for holder, attribute in targets]
    for holder, attribute in targets:
        setattr(holder, attribute, path)
    try:
        yield
    finally:
        for holder, attribute, original in originals:
            setattr(holder, attribute, original)


class NVFP4VLLMGeneration(VLLMGeneration):
    """TRL colocate generation whose engine serves a packed NVFP4 export."""

    def __init__(
        self,
        *,
        rollout_model_path: str,
        sync_input_scales: bool = False,
        activation_scale_warmup_steps: int = 0,
        fused_process_group=None,
        **kwargs,
    ):
        self.rollout_model_path = rollout_model_path
        self.sync_input_scales = sync_input_scales
        self.activation_scale_warmup_steps = activation_scale_warmup_steps
        self.fused_process_group = fused_process_group
        self.publisher: NVFP4WeightPublisher | None = None
        # Set by the trainer once its state exists; the publisher needs the optimizer step
        # to know when the activation-scale warmup is over.
        self.step_provider = None
        super().__init__(**kwargs)

    def _init_vllm(self):
        if self.mode != "colocate":
            raise ValueError(
                f"QARL supports vllm_mode='colocate' only (got {self.mode!r}): the NVFP4 weight sync "
                "updates packed tensors in place inside the engine process."
            )
        # The base class builds the engine from `self.model.name_or_path`; point it at the
        # packed export for the duration of that call only, so the training model's own
        # identity is left untouched.
        with _named_as(self.model, self.rollout_model_path):
            super()._init_vllm()

    def sync_weights(self):
        """Re-pack the current latents into the live engine (replaces TRL's BF16 sync)."""

        vllm_model = vllm_model_from_engine(self.llm)
        if self.publisher is None:
            self.publisher = NVFP4WeightPublisher(
                self.model,
                vllm_model,
                sync_input_scales=self.sync_input_scales,
                warmup_steps=self.activation_scale_warmup_steps,
                step_provider=self.step_provider,
                process_group=self.fused_process_group,
            )
        self.publisher.sync()
        self.llm.reset_prefix_cache()


@contextmanager
def _generation_backend(factory):
    """Make ``GRPOTrainer.__init__`` build our generation backend instead of TRL's."""

    module = importlib.import_module("trl.trainer.grpo_trainer")
    original = module.VLLMGeneration
    module.VLLMGeneration = factory
    try:
        yield
    finally:
        module.VLLMGeneration = original


class QuantGRPOTrainer(QuantCheckpointMixin, GRPOTrainer):
    """GRPO over NVFP4 fake-quantized latents with a packed-NVFP4 rollout engine.

    Args:
        rollout_model_path: directory of the ``nvfp4-pack-quantized`` export vLLM serves.
        converted: the modules ``apply_fake_quant`` swapped, used for saving.
        original_config_path: ``config.json`` of the latent model, written into checkpoints
            so the exporters see a deployment-shaped config.
    """

    quant_format = "nvfp4"
    # Required: the rollout engine fuses q/k/v and gate/up into single packed layers that
    # each carry one tensor scale, which only matches training if the scales are shared.
    fused_runtime_scales = True

    def __init__(
        self,
        *args,
        rollout_model_path: str,
        converted: ConvertedModules,
        original_config_path: str,
        fused_process_group=None,
        sync_activation_scales: bool = False,
        activation_scale_warmup_steps: int = 0,
        **kwargs,
    ):
        self.converted = converted
        self.original_config_path = original_config_path
        self.fused_process_group = fused_process_group
        factory = functools.partial(
            NVFP4VLLMGeneration,
            rollout_model_path=rollout_model_path,
            sync_input_scales=sync_activation_scales,
            activation_scale_warmup_steps=activation_scale_warmup_steps,
            fused_process_group=fused_process_group,
        )
        with _generation_backend(factory):
            super().__init__(*args, **kwargs)
        if not self.use_vllm:
            raise ValueError("QARL requires use_vllm=True: the rollout policy is the packed NVFP4 model.")
        # The publisher is built lazily on the first sync, by which time this resolves.
        self.vllm_generation.step_provider = lambda: self.state.global_step
        self.add_callback(FusedScaleRefreshCallback(process_group=fused_process_group, quant_format=self.quant_format))

    @property
    def lora_metadata(self) -> dict:  # LoRA-QARL is not wired up; kept for the save mixin.
        raise NotImplementedError("LoRA is not supported for QARL.")
