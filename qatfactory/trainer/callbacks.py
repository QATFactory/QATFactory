"""Fused-runtime-scale lifecycle as a Hugging Face ``TrainerCallback``.

For an exportable run the fused-family weight tensor scales are cached (not recomputed
from sharded shards inside every forward). This callback refreshes those globally-reduced
maxima once before training and after every optimizer step. Callback methods run on all
ranks, so the collective all-reduce inside ``refresh_nvfp4_weight_amax`` is well formed.
"""

import logging

from transformers import TrainerCallback

from ..quant.formats import get_quant_format


logger = logging.getLogger(__name__)


class FusedScaleRefreshCallback(TrainerCallback):
    def __init__(self, process_group=None, quant_format: str = "nvfp4"):
        self.process_group = process_group
        self._refresh = get_quant_format(quant_format).refresh_weight_amax
        if self._refresh is None:
            raise ValueError(f"quant_format={quant_format!r} has no fused-scale refresh hook.")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._refresh(model, self.process_group)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Runs after optimizer.step()/zero_grad() for each completed global step.
        self._refresh(model, self.process_group)
