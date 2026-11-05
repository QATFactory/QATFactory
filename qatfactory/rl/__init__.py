"""Quantization-aware reinforcement learning: GRPO with a packed-NVFP4 rollout engine."""

from .model import load_qarl_student, read_export_input_amax, validate_rollout_export, verify_frozen_weights
from .vllm_nvfp4_sync import NVFP4WeightPublisher, vllm_model_from_engine


def __getattr__(name):
    # Imported lazily so `qatfactory.rl.model` stays usable without TRL installed.
    if name in ("QuantGRPOTrainer", "NVFP4VLLMGeneration"):
        from . import grpo_trainer

        return getattr(grpo_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NVFP4VLLMGeneration",
    "NVFP4WeightPublisher",
    "QuantGRPOTrainer",
    "load_qarl_student",
    "read_export_input_amax",
    "validate_rollout_export",
    "verify_frozen_weights",
    "vllm_model_from_engine",
]
