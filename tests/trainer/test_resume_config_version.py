"""Resuming must not parse the checkpoint config with the generic config class."""

import json

import pytest

transformers = pytest.importorskip("transformers")
import transformers.trainer as hf_trainer  # noqa: E402

from qatfactory.trainer.checkpoint import _version_only_checkpoint_config  # noqa: E402

# DeepSeek-R1-Distill-Llama-8B's RoPE block: llama3 scaling, which the generic parse cannot standardize.
LLAMA3_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "max_position_embeddings": 131072,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "transformers_version": "4.99.0",
}


def _config_class():
    return getattr(hf_trainer, "PreTrainedConfig", None) or hf_trainer.PretrainedConfig


def test_version_only_config_reads_version_and_restores(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(LLAMA3_CONFIG))
    original = _config_class()
    with _version_only_checkpoint_config():
        config = _config_class().from_json_file(str(path))
        assert config.transformers_version == "4.99.0"
    assert _config_class() is original


def test_version_only_config_restores_on_error(tmp_path):
    original = _config_class()
    with pytest.raises(RuntimeError):
        with _version_only_checkpoint_config():
            raise RuntimeError("load failed")
    assert _config_class() is original
