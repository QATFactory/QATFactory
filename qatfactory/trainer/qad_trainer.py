"""HF ``Trainer`` subclass for NVFP4 quantization-aware self-distillation.

Reuses the Trainer loop (FSDP2 via accelerate, checkpointing, LR schedule, grad-accum,
logging, eval). The only overrides are the teacher/student dual forward + distillation
loss, a frozen FSDP2-sharded teacher, the fused-scale finalize-on-save hook, and — for
MoE students — the optional teacher routing-replay lifecycle with an annealed schedule.
"""

import contextlib
import logging
import os
import random
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer import SCHEDULER_NAME

from ..distill.kd_loss import IGNORE_INDEX, distill_hidden_states_loss, distill_loss
from ..quant.fake_quant import ConvertedModules
from ..quant.formats import get_quant_format
from ..quant.formats.lora import shared_base_teacher
from .checkpoint import RUNTIME_STATE_FILENAME, QuantCheckpointMixin


logger = logging.getLogger(__name__)

__all__ = ["QADConfig", "QADTrainer", "RUNTIME_STATE_FILENAME"]


@dataclass
class QADConfig:
    temperature: float = 1.0
    distill_weight: float = 1.0
    hard_label_weight: float = 0.0
    fused_runtime_scales: bool = True
    logits_chunk_size: int = 2048
    quant_format: str = "nvfp4"
    target_modules: list[str] = field(default_factory=list)
    # LoRA-QAD: rank > 0 trains only low-rank adapters over a frozen shared base.
    lora_rank: int = 0
    lora_alpha: float = 0.0
    lora_moe_hybrid_shared: bool = True
    # MoE teacher routing replay (R3-style): replay with probability 1 until
    # ``replay_decay_start`` of training, then anneal linearly to free routing at
    # ``replay_decay_end``. Only used when the trainer gets a ``routing_replay_mode``.
    replay_decay_start: float = 0.5
    replay_decay_end: float = 0.8
    replay_in_eval: bool = False
    # Long-context QAD: compute the distillation loss from the final hidden states, projecting
    # `logits_chunk_size` supervised tokens at a time through the frozen output head, instead
    # of materializing sequence x vocabulary logits for teacher and student (32K tokens x
    # Qwen3.5's 248K vocabulary would not fit). Numerically the same loss.
    streamed_logits: bool = False
    # When > 0, run one forward/backward on the first `kernel_warmup_seq_len` tokens of the
    # first batch before training, so long-context kernels autotune at small memory instead
    # of at the peak of a full-length backward. Its gradients are discarded.
    kernel_warmup_seq_len: int = 0
    # Teacher/student diagnostics (top-k agreement, fwd/rev KL, entropies).
    metrics_top_k: tuple[int, ...] = (1, 5, 10)
    # Only compute the (cheap but non-free) train-side metrics on logging steps.
    metrics_logging_only: bool = True
    # Keep the frozen teacher's decoder-layer FSDP2 shards in pinned host memory
    # (CPUOffloadPolicy); each layer forward copies a rank's shard in. Frees nearly all of the
    # teacher's GPU share, e.g. ~27 GiB per GPU for MiniMax-M2.7 on 16 ranks.
    teacher_cpu_offload: bool = False


def _move_unsharded_params(model: torch.nn.Module, device: torch.device) -> None:
    """Move the parameters FSDP has not sharded yet (the root leaves) to ``device``, leaving the
    already-sharded, host-offloaded layer parameters where they are. Tied parameters stay tied."""
    from torch.distributed.tensor import DTensor

    moved: dict[int, torch.nn.Parameter] = {}
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or isinstance(param, DTensor) or param.device == device:
                continue
            if id(param) not in moved:
                moved[id(param)] = torch.nn.Parameter(param.data.to(device), requires_grad=param.requires_grad)
            module._parameters[name] = moved[id(param)]


def _check_teacher_offload(teacher: torch.nn.Module) -> None:
    """Re-sync each offloaded FSDP2 group with its module's parameters, then fail fast, on every
    rank, if a shard is not on the host (FSDP2 would otherwise refuse the module at the first
    forward).

    A load after ``fully_shard`` (accelerate's broadcast ``fsdp2_load_full_state_dict``) assigns
    GPU shards to the modules. ``reset_sharded_param`` re-reads them and, under a pinned
    CPUOffloadPolicy, moves the local tensor into pinned host memory, but the DTensor wrapper
    keeps the device it was created with, and FSDP2's offload check reads that. So the host
    shard is re-wrapped the way FSDP2 builds offloaded parameters (``to_sharded_dtensor``),
    set on the modules, and re-synced.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule

    offloaded = on_gpu = 0
    for module in teacher.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = module._get_fsdp_state()._fsdp_param_group
        if group is None or not isinstance(group.offload_policy, CPUOffloadPolicy):
            continue
        for fsdp_param in group.fsdp_params:
            fsdp_param.reset_sharded_param()
            current = fsdp_param.sharded_param
            if current.device.type != "cpu" and current._local_tensor.device.type == "cpu":
                rewrapped = torch.nn.Parameter(
                    fsdp_param.to_sharded_dtensor(current._local_tensor), requires_grad=current.requires_grad
                )
                fsdp_param._setattr_on_modules(rewrapped)
                fsdp_param.reset_sharded_param()
            offloaded += 1
            on_gpu += fsdp_param.sharded_param.device.type != "cpu"
    if offloaded == 0 or on_gpu:
        raise RuntimeError(f"Teacher CPU offload: {on_gpu} of {offloaded} offloaded shards are not on the host.")
    logger.info("Teacher CPU offload: %d layer shards on the host; the root group stays on the GPU.", offloaded)


class QADTrainer(QuantCheckpointMixin, Trainer):
    def __init__(
        self,
        *args,
        teacher,
        qad_config: QADConfig,
        converted: ConvertedModules,
        original_config_path: str,
        routing_replay_mode: str | None = None,
        low_bit_optimizer: str | None = None,
        fused_process_group=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Ask the Trainer for the supervised-token count of the whole accumulation window and
        # normalize by it in `compute_loss`, rather than averaging per-microbatch token means
        # (which silently over-weights short microbatches). `_loss_shifts_labels` makes the
        # Trainer count over `labels[..., 1:]`, matching the shift the distillation loss does.
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True
        # ``teacher`` is a module, ``None`` (shared-base LoRA), or a zero-argument loader
        # (``load_student_and_teacher(lazy_teacher=True)``) called once the student is sharded.
        self.teacher = teacher
        self.replay = None
        self.routing_replay_mode = routing_replay_mode
        self.low_bit_optimizer = low_bit_optimizer
        if qad_config.streamed_logits and teacher is None:
            raise ValueError("streamed_logits needs a separate teacher model; it is not supported with LoRA-QAD.")
        self._kernel_warmup_done = False
        if routing_replay_mode and teacher is None:
            raise ValueError("Routing replay needs a separate teacher model (full-parameter MoE QAD), not LoRA.")
        self.qad = qad_config
        self.converted = converted
        self.original_config_path = original_config_path
        self.fused_process_group = fused_process_group
        self._teacher_prepared = False
        self._commit_input_scales = get_quant_format(qad_config.quant_format).commit_input_scales
        self._scale_commit_step = -1
        self._loss_components = {}
        self._metric_components = {}
        self._eval_metric_sums: dict[str, torch.Tensor] = {}
        self._eval_token_count = None

    # Quantization context consumed by QuantCheckpointMixin.
    @property
    def quant_format(self) -> str:
        return self.qad.quant_format

    @property
    def fused_runtime_scales(self) -> bool:
        return self.qad.fused_runtime_scales

    @property
    def lora_metadata(self) -> dict:
        return {
            "rank": int(self.qad.lora_rank),
            "alpha": float(self.qad.lora_alpha),
            "scaling": float(self.qad.lora_alpha) / float(self.qad.lora_rank),
            "moe_hybrid_shared": bool(self.qad.lora_moe_hybrid_shared),
        }

    def create_optimizer(self):
        """Optionally swap in torchao's 8-bit AdamW (``train.low_bit_optim``).

        ``adamw8bit_sr`` applies BF16 stochastic rounding to the parameter update — the regime
        for BF16 latents without an FP32 master (``qad.master_fp32: false``), where
        deterministic rounding would swallow recovery-scale updates. Used for the large MoE
        runs, where an FP32 master plus FP32 Adam moments does not fit.
        """

        if self.low_bit_optimizer is None:
            return super().create_optimizer()
        if self.low_bit_optimizer not in ("adamw8bit", "adamw8bit_sr"):
            raise ValueError(
                f"Unsupported low_bit_optim {self.low_bit_optimizer!r} (use 'adamw8bit' or 'adamw8bit_sr')."
            )
        if self.optimizer is None:
            # Built directly rather than via args.optim: transformers' torchao optimizer names
            # import a prototype path that torchao>=0.16 removed.
            from torchao.optim import AdamW8bit

            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            self.optimizer = AdamW8bit(
                parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                bf16_stochastic_round=self.low_bit_optimizer == "adamw8bit_sr",
            )
            logger.info("Using torchao AdamW8bit (%s) over %d trainable tensors.", self.low_bit_optimizer, len(parameters))
        return self.optimizer

    def _low_bit_sharded(self) -> bool:
        return self.low_bit_optimizer is not None and self.is_fsdp_enabled

    def _save_optimizer_and_scheduler(self, output_dir):
        if not self._low_bit_sharded():
            return super()._save_optimizer_and_scheduler(output_dir)
        # torchao's 8-bit optimizer state does not go through FSDP's full optimizer state dict:
        # its tensor subclass implements neither the broadcast that loading uses nor the dummy
        # zero-gradient step that initializes empty state first. Each rank saves its own shard
        # instead, so a resume needs the same world size.
        from accelerate.utils import save_fsdp_model
        from transformers.integrations.fsdp import get_fsdp_ckpt_kwargs

        # The base method also writes the FSDP model state that resuming loads (the FP32 latents);
        # keep that part, but sharded: every rank writes its own shard as a distributed
        # checkpoint. A full state dict would gather four bytes per parameter on the main rank's
        # host, about 0.9 TB for MiniMax-M2.7, more than a node has.
        with self._sharded_fsdp_state_dict() as plugin:
            save_fsdp_model(plugin, self.accelerator, self.model, output_dir, **get_fsdp_ckpt_kwargs())
        inner = getattr(self.optimizer, "optimizer", self.optimizer)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(inner.state_dict(), os.path.join(output_dir, _low_bit_optimizer_file()))
        if self.args.should_save:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    @contextlib.contextmanager
    def _sharded_fsdp_state_dict(self):
        """Use SHARDED_STATE_DICT for the FSDP plugin inside the block, then restore its setting."""

        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        plugin = self.accelerator.state.fsdp_plugin
        previous = (plugin.state_dict_type, plugin.state_dict_config, plugin.optim_state_dict_config)
        # Replace the configs too: set_state_dict_type keeps an existing FullStateDictConfig, whose
        # rank0_only becomes FSDP2's broadcast_from_rank0, which a sharded state dict rejects.
        plugin.state_dict_type = StateDictType.SHARDED_STATE_DICT
        plugin.state_dict_config = ShardedStateDictConfig(offload_to_cpu=False)
        plugin.optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=False)
        try:
            yield plugin
        finally:
            plugin.state_dict_type, plugin.state_dict_config, plugin.optim_state_dict_config = previous

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Read the FSDP model state in the layout it was written in: a sharded directory
        # (``pytorch_model_fsdp_0``, written by _save_optimizer_and_scheduler) or a full
        # ``pytorch_model_fsdp.bin`` from earlier checkpoints.
        from accelerate.utils.constants import FSDP_MODEL_NAME

        sharded = self.is_fsdp_enabled and os.path.isdir(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}_0"))
        with self._sharded_fsdp_state_dict() if sharded else contextlib.nullcontext():
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None or not self._low_bit_sharded():
            return super()._load_optimizer_and_scheduler(checkpoint)
        path = os.path.join(checkpoint, _low_bit_optimizer_file())
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{checkpoint} has no {os.path.basename(path)}; low_bit_optim runs resume from per-rank "
                "optimizer shards and need the world size they were saved with."
            )
        inner = getattr(self.optimizer, "optimizer", self.optimizer)
        inner.load_state_dict(torch.load(path, weights_only=False))
        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True))
        logger.info("Restored %s optimizer shard and scheduler from %s.", self.low_bit_optimizer, checkpoint)

    def _prepare_teacher(self) -> None:
        """Shard the frozen teacher with FSDP2 on the same (world) process group as the student."""

        # The student is sharded by now; restore the buffers cpu-ram-efficient loading de-persisted.
        depersisted = getattr(self.model, "_qatf_depersisted_buffers", None)
        if depersisted:
            from ..models.llm import repersist_buffers

            repersist_buffers(depersisted)
            self.model._qatf_depersisted_buffers = []

        if self.teacher is None:
            # Shared-base LoRA: the student model itself is the teacher (run in teacher-mode),
            # so there is nothing separate to shard.
            self._teacher_prepared = True
            return

        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if callable(self.teacher) and not isinstance(self.teacher, torch.nn.Module):
            logger.info("Loading the teacher now that the student is sharded (lazy teacher).")
            self.teacher = self.teacher()

        # Attach routing replay before sharding: its router-convention probe needs plain
        # weights, and the hooks survive fully_shard (module identity is kept).
        if self.replay is None and self.routing_replay_mode:
            from ..moe import RoutingReplay

            self.replay = RoutingReplay(self.teacher, self.model, mode=self.routing_replay_mode)

        device = self.accelerator.device
        if dist.is_initialized():
            from ..models.llm import cpu_ram_efficient_loading_enabled

            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            offload = self.qad.teacher_cpu_offload
            if offload:
                from torch.distributed.fsdp import CPUOffloadPolicy

                shard_kwargs = dict(offload_policy=CPUOffloadPolicy(pin_memory=True))
            else:
                shard_kwargs = {}
            no_split = set(getattr(self.teacher, "_no_split_modules", None) or [])
            # Gate on the rank-independent flag, never on rank-local meta state: the
            # cpu-ram-efficient path runs collectives every rank must join.
            if cpu_ram_efficient_loading_enabled():
                self._shard_teacher_cpu_ram_efficient(mp_policy, no_split, device, shard_kwargs)
            else:
                # Move each decoder layer to the device only right before sharding it, so the
                # full teacher (61 GB for a 30B model) never materializes on one GPU.
                for module in self.teacher.modules():
                    if type(module).__name__ in no_split:
                        if not offload:
                            module.to(device)
                        fully_shard(module, mp_policy=mp_policy, reshard_after_forward=True, **shard_kwargs)
                if offload:
                    # Layer parameters stay on the host; the root leaves (embeddings, final
                    # norm, lm_head) and every buffer (e.g. RoPE inv_freq) go to the device.
                    from ..models.llm import assign_buffer

                    _move_unsharded_params(self.teacher, device)
                    for name, buffer in list(self.teacher.named_buffers()):
                        assign_buffer(self.teacher, name, buffer.to(device))
                else:
                    self.teacher.to(device)  # remaining leaves: embeddings, final norm, lm_head
                # The root group (embeddings, final norm, lm_head; ~1/world of 2.4 GB for
                # MiniMax-M2.7) is never offloaded: FSDP2 would need its shards on the host too,
                # and the GPU-side broadcast load leaves them on the device.
                fully_shard(self.teacher, mp_policy=mp_policy, reshard_after_forward=True)
            if offload:
                _check_teacher_offload(self.teacher)
        else:
            self.teacher.to(device)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self._teacher_prepared = True

    def _shard_teacher_cpu_ram_efficient(self, mp_policy, no_split, device, shard_kwargs=None) -> None:
        """Shard a teacher whose real weights exist only on the main rank (meta elsewhere).

        Mirrors accelerate's FSDP2 cpu-ram-efficient prepare: make the buffers real, shard on
        meta so no rank holds a full copy, then broadcast the main rank's parameters into the
        shards one tensor at a time.
        """

        from accelerate.utils.fsdp_utils import fsdp2_load_full_state_dict
        from torch.distributed.fsdp import fully_shard

        from ..models.llm import assign_buffer, cpu_ram_efficient_buffer_fixup, repersist_buffers

        depersisted = cpu_ram_efficient_buffer_fixup(self.teacher, device)
        saved_buffers = {name: buffer.detach().clone() for name, buffer in self.teacher.named_buffers()}
        full_state_dict = self.teacher.state_dict()  # real on the main rank, meta elsewhere
        self.teacher = self.teacher.to("meta")
        if hasattr(self.teacher, "tie_weights"):
            self.teacher.tie_weights()
        shard_kwargs = shard_kwargs or {}
        for module in self.teacher.modules():
            if type(module).__name__ in no_split:
                fully_shard(module, mp_policy=mp_policy, reshard_after_forward=True, **shard_kwargs)
        fully_shard(self.teacher, mp_policy=mp_policy, reshard_after_forward=True)  # root: never offloaded
        fsdp2_load_full_state_dict(self.accelerator, self.teacher, full_state_dict)
        for name, buffer in saved_buffers.items():
            assign_buffer(self.teacher, name, buffer.to(device))
        repersist_buffers(depersisted)

    def _replay_probability(self) -> float:
        total_steps = self.state.max_steps or self.args.max_steps
        if not total_steps or total_steps <= 0:
            return 1.0
        fraction = self.state.global_step / total_steps
        start, end = self.qad.replay_decay_start, self.qad.replay_decay_end
        if fraction < start:
            return 1.0
        if fraction >= end:
            return 0.0
        return 1.0 - (fraction - start) / max(end - start, 1e-9)

    def _replay_active_now(self, training: bool) -> bool:
        if not training:
            return self.qad.replay_in_eval
        probability = self._replay_probability()
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        # Seeded per global step so every rank (and a checkpoint recompute) agrees.
        return random.Random(hash((self.args.seed, self.state.global_step))).random() < probability

    def _compute_streamed_loss(self, model, model_inputs, labels, *, compute_metrics: bool, loss_denominator=None):
        """Distillation loss from the final hidden states, projected through the output head in chunks.

        Both models run with ``logits_to_keep`` set to an empty index, so neither emits
        sequence x vocabulary logits. The loss is computed inside a forward hook on the
        student's output head, and that hook returns the loss tensor as the head's output, so
        FSDP's pre-backward unshard hook attaches to the loss graph. The head must be frozen
        and shared in value with the teacher's.
        """

        empty_indices = torch.empty(0, dtype=torch.long, device=labels.device)
        teacher_hidden = {}

        def capture_teacher_hidden(_module, _args, output):
            teacher_hidden["value"] = output.last_hidden_state

        handle = self.teacher.base_model.register_forward_hook(capture_teacher_hidden)
        try:
            with torch.no_grad():
                self.teacher(**model_inputs, use_cache=False, logits_to_keep=empty_indices)
        finally:
            handle.remove()
        if "value" not in teacher_hidden:
            raise RuntimeError("Streamed QAD failed to capture teacher hidden states.")

        unwrapped = self.accelerator.unwrap_model(model)
        output_head = unwrapped.get_output_embeddings()
        if output_head.weight.requires_grad:
            raise RuntimeError("Streamed QAD requires the output head to stay frozen.")
        student_hidden, result_box = {}, {}

        def capture_student_hidden(_module, _args, output):
            student_hidden["value"] = output.last_hidden_state

        def compute_at_output_head(module, _args, _output):
            if "value" not in student_hidden:
                raise RuntimeError("Streamed QAD output head ran before the student hidden states were captured.")
            result = distill_hidden_states_loss(
                student_hidden["value"],
                teacher_hidden["value"],
                labels,
                module.weight,
                temperature=self.qad.temperature,
                distill_weight=self.qad.distill_weight,
                hard_label_weight=self.qad.hard_label_weight,
                token_chunk_size=self.qad.logits_chunk_size,
                compute_metrics=compute_metrics,
                metrics_top_k=tuple(self.qad.metrics_top_k),
                loss_denominator=loss_denominator,
            )
            result_box["value"] = result
            return result.loss

        body_handle = unwrapped.base_model.register_forward_hook(capture_student_hidden)
        head_handle = output_head.register_forward_hook(compute_at_output_head)
        try:
            student_outputs = model(**model_inputs, use_cache=False, logits_to_keep=empty_indices)
        finally:
            head_handle.remove()
            body_handle.remove()
        if "value" not in result_box:
            raise RuntimeError("Streamed QAD failed to compute the loss at the output head.")
        return result_box["value"], student_outputs

    def _run_kernel_warmup(self, model, model_inputs, labels) -> None:
        """Autotune long-context kernels on a short prefix before the first full-length backward."""

        sequence_length = labels.shape[1]
        length = min(int(self.qad.kernel_warmup_seq_len), sequence_length)
        warmup_inputs = {
            key: (value[:, :length] if value.ndim >= 2 and value.shape[1] == sequence_length else value)
            for key, value in model_inputs.items()
        }
        warmup_labels = labels[:, :length]
        if self.qad.streamed_logits:
            result, _ = self._compute_streamed_loss(model, warmup_inputs, warmup_labels, compute_metrics=False)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher(**warmup_inputs, use_cache=False)
            student_outputs = model(**warmup_inputs, use_cache=False)
            result = distill_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                warmup_labels,
                temperature=self.qad.temperature,
                distill_weight=self.qad.distill_weight,
                hard_label_weight=self.qad.hard_label_weight,
                logits_chunk_size=self.qad.logits_chunk_size,
            )
        if not torch.isfinite(result.loss).item():
            raise FloatingPointError(f"Non-finite QAD kernel-warmup loss: {result.loss.detach().item()}.")
        self.accelerator.backward(result.loss)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._kernel_warmup_done = True
        logger.info("Kernel warmup done at sequence length %d (%d supervised tokens).", length, int(result.num_tokens))

    @staticmethod
    def _shared_base_teacher_forward(model, model_inputs):
        """Run one frozen-base reference forward and restore every module's original mode.

        Dropout and the fake-quant observers key off ``module.training``, so the teacher pass
        runs in eval mode; the per-module modes are restored even if the forward raises.
        """

        training_modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad(), shared_base_teacher():
                return model(**model_inputs, use_cache=False)
        finally:
            for module, training in training_modes:
                module.training = training

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: int) -> torch.Tensor:
        """Scale only the backward pass, leaving the reported loss value untouched."""

        if scale <= 1:
            return value
        return value + (value - value.detach()) * (scale - 1)

    def _is_train_logging_step(self) -> bool:
        """True on the (upcoming) global step that HF will emit a train log for.

        ``global_step`` only increments after the optimizer step, so the micro-batches that
        feed the step which becomes ``N`` run while ``state.global_step == N - 1``.
        """

        steps = int(getattr(self.args, "logging_steps", 0) or 0)
        if steps <= 0:
            return True
        return (self.state.global_step + 1) % steps == 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._teacher_prepared:
            self._prepare_teacher()
        if self._commit_input_scales is not None and model.training and self._scale_commit_step != self.state.global_step:
            # Publish running input maxima into the static W4A4 scales exactly once per
            # optimizer step, before any of its forwards: within the step (incl. the
            # gradient-checkpoint recompute in backward) the scales are then immutable,
            # and rank-consistent (the hook all-reduces the maxima across ranks).
            self._commit_input_scales(model, process_group=self.fused_process_group)
            self._scale_commit_step = self.state.global_step

        is_eval = not model.training
        want_metrics = is_eval or not self.qad.metrics_logging_only or self._is_train_logging_step()

        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        if self.teacher is None:
            # Shared-base LoRA: derive the teacher logits from the same model with the
            # frozen full-precision base path, then the quantized student path. Both read
            # one shared copy of the base weights.
            teacher_outputs = self._shared_base_teacher_forward(model, model_inputs)
            student_outputs = model(**model_inputs, use_cache=False)
        else:
            if self.replay is not None:
                self.replay.begin_batch(
                    replay_active=self._replay_active_now(model.training),
                    supervision_mask=labels != IGNORE_INDEX,
                )
            if model.training and not self._kernel_warmup_done and self.qad.kernel_warmup_seq_len:
                self._run_kernel_warmup(model, model_inputs, labels)
            if self.qad.streamed_logits:
                result, student_outputs = self._compute_streamed_loss(
                    model, model_inputs, labels, compute_metrics=want_metrics, loss_denominator=num_items_in_batch
                )
            else:
                with torch.no_grad():
                    teacher_outputs = self.teacher(**model_inputs, use_cache=False)
                student_outputs = model(**model_inputs, use_cache=False)

        if not (self.teacher is not None and self.qad.streamed_logits):
            result = distill_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                temperature=self.qad.temperature,
                distill_weight=self.qad.distill_weight,
                hard_label_weight=self.qad.hard_label_weight,
                logits_chunk_size=self.qad.logits_chunk_size,
                compute_metrics=want_metrics,
                metrics_top_k=tuple(self.qad.metrics_top_k),
                loss_denominator=num_items_in_batch,
            )
        if not torch.isfinite(result.loss).item():
            raise FloatingPointError(
                f"Non-finite QAD loss: combined={result.loss.detach().item()}, "
                f"distill={result.distill_loss.item()}, ce={result.ce_loss.item()}."
            )
        loss = result.loss
        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                # `num_items_in_batch` was gathered across ranks, so each rank's share must be
                # scaled back up before the data-parallel gradient average.
                loss_scale = self.accelerator.num_processes
                parallelism = getattr(self.accelerator, "parallelism_config", None)
                if parallelism is not None:
                    loss_scale //= parallelism.tp_size
                loss = loss * loss_scale
            # Each microbatch is already normalized by the whole window, so undo the divisor
            # Accelerate will apply in `backward()` — in the gradient only, so the logged loss
            # keeps its true value.
            loss = self._scale_gradient(loss, int(getattr(self.accelerator, "gradient_accumulation_steps", 1)))
        self._loss_components = {
            "distill_loss": result.distill_loss.detach(),
            "ce_loss": result.ce_loss.detach(),
        }
        if self.replay is not None:
            self._loss_components.update(self.replay.metrics())
            if model.training:
                self._loss_components["routing_replay_p"] = torch.tensor(self._replay_probability())
        if want_metrics and result.metrics:
            if is_eval:
                for key, value in result.metrics.items():
                    self._eval_metric_sums[key] = self._eval_metric_sums.get(key, 0) + value
                self._eval_token_count = (self._eval_token_count or 0) + result.num_tokens.detach()
            else:
                denominator = result.num_tokens.clamp_min(1)
                self._metric_components = {
                    key: (value / denominator).detach() for key, value in result.metrics.items()
                }
        return (loss, student_outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            for key, value in self._loss_components.items():
                logs.setdefault(key, round(value.item(), 6))
            for key, value in self._metric_components.items():
                logs.setdefault(key, round(value.item(), 6))
            for key, value in self._memory_logs().items():
                logs.setdefault(key, value)
        super().log(logs, start_time)

    @staticmethod
    def _memory_logs() -> dict:
        """Peak CUDA memory since the last log, split into the part that does not vary.

        ``persistent`` is what is still resident between steps — sharded parameters,
        gradients and optimizer state — so ``peak - persistent`` is the transient cost of a
        step (activations plus the parameter all-gathers). Reading them apart is what makes
        a micro-batch-size ceiling predictable instead of a bisection search.
        """

        if not torch.cuda.is_available():
            return {}
        peak = torch.cuda.max_memory_allocated() / 2**30
        persistent = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.reset_peak_memory_stats()
        return {
            "gpu_peak_gib": round(peak, 2),
            "gpu_persistent_gib": round(persistent, 2),
            "gpu_reserved_gib": round(reserved, 2),
        }

    def evaluation_loop(self, *args, **kwargs):
        """Wrap the eval loop to emit token-weighted, rank-reduced ``eval_*`` diagnostics.

        The per-batch metric token-sums are accumulated in :meth:`compute_loss`; here we
        all-reduce them (and the token count) across ranks and fold a single mean per metric
        into ``output.metrics`` so HF's W&B integration surfaces them under ``eval/*``.
        """

        self._eval_metric_sums = {}
        self._eval_token_count = None
        output = super().evaluation_loop(*args, **kwargs)

        if self._eval_metric_sums and self._eval_token_count is not None:
            prefix = kwargs.get("metric_key_prefix") or (args[4] if len(args) > 4 else "eval")
            device = self.accelerator.device
            tokens = self._eval_token_count.to(device).float()
            if dist.is_initialized():
                dist.all_reduce(tokens)
            tokens = tokens.clamp_min(1)
            for key, value in self._eval_metric_sums.items():
                reduced = value.to(device).float()
                if dist.is_initialized():
                    dist.all_reduce(reduced)
                output.metrics[f"{prefix}_{key}"] = round((reduced / tokens).item(), 6)

        self._eval_metric_sums = {}
        self._eval_token_count = None
        return output


def _low_bit_optimizer_file() -> str:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return f"low_bit_optimizer_rank{rank}.pt"
