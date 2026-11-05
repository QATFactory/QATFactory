import logging

import pytest
from torch import nn

from qatfactory.trainer.fsdp import resolve_fsdp_options


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class Vision(nn.Module):
    pass


class Toy(nn.Module):
    # Mirrors a multimodal `_no_split_modules`: the vision class is declared but a text-only
    # load never instantiates it.
    _no_split_modules = ["Block", "Vision"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])


def test_defaults_to_per_layer_sharding_with_instantiated_wrap_classes():
    fsdp, fsdp_config = resolve_fsdp_options({}, Toy())
    assert fsdp == "full_shard auto_wrap"
    assert fsdp_config == {"fsdp_version": 2, "transformer_layer_cls_to_wrap": ["Block"]}


def test_root_only_option_gets_auto_wrap_added(caplog):
    # `transformers` only forwards the wrap classes under `auto_wrap`; without it FSDP2 shards
    # the root module only and every rank all-gathers the whole model each forward.
    with caplog.at_level(logging.WARNING, logger="qatfactory.trainer.fsdp"):
        fsdp, _ = resolve_fsdp_options({"fsdp": "full_shard"}, Toy())
    assert fsdp == "full_shard auto_wrap"
    assert "root module only" in caplog.text


def test_explicit_options_and_config_are_preserved():
    train_cfg = {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {"fsdp_version": 2, "transformer_layer_cls_to_wrap": ["Block"], "cpu_ram_efficient_loading": True},
        "learning_rate": 1e-6,
    }
    fsdp, fsdp_config = resolve_fsdp_options(train_cfg, Toy())
    assert fsdp == "full_shard auto_wrap"
    assert fsdp_config["cpu_ram_efficient_loading"] is True
    assert fsdp_config["transformer_layer_cls_to_wrap"] == ["Block"]
    # The FSDP keys are consumed; the rest of the training config is left for TrainingArguments.
    assert train_cfg == {"learning_rate": 1e-6}


def test_disabling_fsdp_is_left_alone():
    fsdp, _ = resolve_fsdp_options({"fsdp": ""}, Toy())
    assert fsdp == ""


def test_building_before_the_model_loads_requires_explicit_wrap_classes():
    fsdp, fsdp_config = resolve_fsdp_options({"fsdp_config": {"transformer_layer_cls_to_wrap": ["Block"]}}, None)
    assert fsdp == "full_shard auto_wrap"
    assert fsdp_config["transformer_layer_cls_to_wrap"] == ["Block"]
    with pytest.raises(ValueError, match="transformer_layer_cls_to_wrap"):
        resolve_fsdp_options({}, None)
