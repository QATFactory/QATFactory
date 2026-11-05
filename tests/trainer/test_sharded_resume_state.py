from types import SimpleNamespace

import pytest
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

from qatfactory.trainer import checkpoint as checkpoint_module
from qatfactory.trainer.qad_trainer import QADTrainer


def _plugin():
    return SimpleNamespace(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )


def _host(plugin):
    host = QADTrainer.__new__(QADTrainer)  # only the attributes these methods read
    host.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    host.is_fsdp_enabled = True
    return host


def test_sharded_state_dict_swaps_the_configs_and_restores_them_on_error():
    plugin = _plugin()
    full = plugin.state_dict_config
    with pytest.raises(RuntimeError):
        with _host(plugin)._sharded_fsdp_state_dict() as inside:
            assert inside is plugin and plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT
            # FSDP2 maps rank0_only to broadcast_from_rank0, which a sharded state dict rejects.
            assert isinstance(plugin.state_dict_config, ShardedStateDictConfig)
            assert not getattr(plugin.state_dict_config, "rank0_only", False)
            raise RuntimeError("save failed")
    assert plugin.state_dict_type == StateDictType.FULL_STATE_DICT and plugin.state_dict_config is full


@pytest.mark.parametrize("layout", ["sharded", "full"])
def test_resume_reads_the_layout_on_disk(tmp_path, monkeypatch, layout):
    if layout == "sharded":
        (tmp_path / "pytorch_model_fsdp_0").mkdir()
    else:
        (tmp_path / "pytorch_model_fsdp.bin").write_bytes(b"")
    plugin = _plugin()
    seen = []
    monkeypatch.setattr(
        checkpoint_module.QuantCheckpointMixin,
        "_load_from_checkpoint",
        lambda self, path, model=None: seen.append(plugin.state_dict_type),
    )
    _host(plugin)._load_from_checkpoint(str(tmp_path))
    assert seen == [StateDictType.SHARDED_STATE_DICT if layout == "sharded" else StateDictType.FULL_STATE_DICT]
    assert plugin.state_dict_type == StateDictType.FULL_STATE_DICT
