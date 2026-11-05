"""FSDP2 launch options shared by the QAD and QARL entrypoints."""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def resolve_fsdp_options(train_cfg: dict, model: nn.Module | None) -> tuple[str, dict]:
    """Pop ``fsdp`` / ``fsdp_config`` from ``train_cfg`` and return them launch-ready.

    The wrap classes default to the model's declared non-split decoder layers, filtered to
    classes actually instantiated (a text-only causal LM load omits the vision-tower classes
    that appear in a multimodal ``_no_split_modules``). ``model=None`` means the arguments are
    built before the model is loaded (``cpu_ram_efficient_loading`` needs the FSDP env vars in
    place first), so the config must then name ``transformer_layer_cls_to_wrap`` explicitly.

    ``transformers`` only hands ``transformer_layer_cls_to_wrap`` to accelerate under the
    ``auto_wrap`` FSDP option. Without it FSDP2 shards the *root module only*: every rank
    all-gathers the entire model in each forward and backward, which costs a full BF16 model
    copy of transient memory per rank (fatal for a 30B MoE on 80 GB, and a ~2.5x step-time
    penalty on 9B once the allocator is pushed to the ceiling). Per-layer sharding is the
    intended layout whenever wrap classes are known, so ``auto_wrap`` is added when missing.
    """

    fsdp_config = dict(train_cfg.pop("fsdp_config", {}) or {})
    fsdp_config.setdefault("fsdp_version", 2)
    if model is not None:
        present_classes = {type(module).__name__ for module in model.modules()}
        wrap_classes = [name for name in (getattr(model, "_no_split_modules", None) or []) if name in present_classes]
        fsdp_config.setdefault("transformer_layer_cls_to_wrap", wrap_classes)
    elif not fsdp_config.get("transformer_layer_cls_to_wrap"):
        raise ValueError(
            "Building the FSDP options before the model is loaded requires "
            "train.fsdp_config.transformer_layer_cls_to_wrap to name the decoder layer class explicitly."
        )

    options = str(train_cfg.pop("fsdp", "full_shard auto_wrap") or "").split()
    if fsdp_config.get("transformer_layer_cls_to_wrap") and options and "auto_wrap" not in options:
        logger.warning(
            "fsdp=%r would shard the root module only; adding `auto_wrap` so each %s is its own FSDP2 unit.",
            " ".join(options),
            "/".join(fsdp_config["transformer_layer_cls_to_wrap"]),
        )
        options.append("auto_wrap")
    return " ".join(options), fsdp_config
