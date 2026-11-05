"""Under FSDP cpu-ram-efficient loading only the main rank may read the checkpoint.

The other ranks receive their shards by broadcast, so they must build the model on meta. If
they read the checkpoint too, every rank holds a full host copy of the backbone, which for
MiniMax-M2.7 (457 GB, 8 ranks per node) exhausts host memory before training starts.
"""

import torch
from transformers import LlamaConfig

from qatfactory.models import llm


def _config_only_dir(tmp_path):
    # No weight files: a non-main rank that tried to read the checkpoint would fail.
    LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, vocab_size=32).save_pretrained(tmp_path)
    return str(tmp_path)


def test_a_non_main_rank_builds_on_meta_without_reading_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "cpu_ram_efficient_loading_enabled", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    model = llm.load_causal_lm(_config_only_dir(tmp_path), attn_implementation="eager")

    params = list(model.parameters())
    assert params and all(p.is_meta for p in params)
    assert all(p.dtype == torch.bfloat16 for p in params)
    assert model.config.num_hidden_layers == 2
