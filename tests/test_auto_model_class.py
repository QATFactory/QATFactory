"""Loading must follow the auto class a checkpoint is actually registered with.

A multimodal checkpoint's language tower is an ordinary causal LM, but transformers registers
the wrapper under `AutoModelForImageTextToText`, so asking for `AutoModelForCausalLM` raises
before a single weight is read. This is pinned against the real auto mappings, since the
point is exactly which class transformers considers responsible for a config.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import LlamaConfig, LlavaConfig, ViTConfig

from qatfactory.models.llm import _auto_model_class


@pytest.fixture
def config_from(monkeypatch):
    def install(config):
        monkeypatch.setattr(
            "qatfactory.models.llm.AutoConfig.from_pretrained",
            lambda path, **kwargs: config,
        )

    return install


def test_a_text_only_checkpoint_loads_as_a_causal_lm(config_from):
    config_from(LlamaConfig())
    assert _auto_model_class("any/path") is AutoModelForCausalLM


def test_a_multimodal_checkpoint_falls_back_to_the_image_text_class(config_from):
    config_from(LlavaConfig())
    assert _auto_model_class("any/path") is AutoModelForImageTextToText


def test_an_unsupported_architecture_is_rejected_by_name(config_from):
    config_from(ViTConfig())
    with pytest.raises(ValueError, match="ViTConfig is not registered"):
        _auto_model_class("any/path")
