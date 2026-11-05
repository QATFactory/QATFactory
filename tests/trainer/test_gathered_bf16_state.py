from types import SimpleNamespace

import pytest
import torch

from qatfactory.trainer.checkpoint import QuantCheckpointMixin


def _full_state():
    return {
        "layers.0.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7,
        "layers.0.experts.gate_up_proj": torch.randn(4, 6, 2, generator=torch.Generator().manual_seed(0)),
        "layers.0.scale": torch.tensor([3], dtype=torch.int32),
        "embed.weight": torch.randn(8, 4, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16),
    }


def _gather_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    import torch.distributed.checkpoint.state_dict as dcp_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        full = _full_state()
        sharded = {
            key: (distribute_tensor(value, mesh, [Shard(0)]) if value.dim() and value.dtype.is_floating_point else value)
            for key, value in full.items()
        }
        # Stand in for FSDP2: the trainer's model yields DTensor shards from get_model_state_dict.
        dcp_state_dict.get_model_state_dict = lambda model, options=None: sharded
        host = SimpleNamespace(
            accelerator=SimpleNamespace(is_fsdp2=True, device=torch.device("cpu")),
            args=SimpleNamespace(should_save=rank == 0),
            model=None,
        )
        everything = QuantCheckpointMixin._gathered_bf16_state(host)
        trainable_only = QuantCheckpointMixin._gathered_bf16_state(host, {"layers.0.weight"})
        torch.save({"all": everything, "kept": trainable_only}, f"{result_dir}/rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="requires torch.distributed")
def test_fsdp2_save_gathers_each_tensor_as_bf16_on_the_saving_rank_only(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _gather_worker, args=(world_size, str(tmp_path / "dist_init"), str(tmp_path)), nprocs=world_size, join=True
    )
    full = _full_state()
    main = torch.load(tmp_path / "rank0.pt")
    assert list(main["all"]) == list(full)
    for key, value in full.items():
        expected = value.to(torch.bfloat16) if value.dtype.is_floating_point else value
        assert main["all"][key].dtype == expected.dtype
        assert torch.equal(main["all"][key], expected)
        assert not isinstance(main["all"][key], torch.distributed.tensor.DTensor)
    assert list(main["kept"]) == ["layers.0.weight"]
    other = torch.load(tmp_path / "rank1.pt")
    assert other == {"all": {}, "kept": {}}


def test_non_fsdp2_save_narrows_the_accelerate_state_dict():
    full = _full_state()
    calls = []

    def get_state_dict(model):
        calls.append(model)
        return dict(full)

    for should_save in (True, False):
        host = SimpleNamespace(
            accelerator=SimpleNamespace(is_fsdp2=False, get_state_dict=get_state_dict),
            args=SimpleNamespace(should_save=should_save),
            model="model",
        )
        out = QuantCheckpointMixin._gathered_bf16_state(host, {"layers.0.weight", "layers.0.scale"})
        if should_save:
            assert list(out) == ["layers.0.weight", "layers.0.scale"]
            assert out["layers.0.weight"].dtype == torch.bfloat16
            assert torch.equal(out["layers.0.scale"], full["layers.0.scale"])
        else:
            assert out == {}
    assert calls == ["model", "model"]  # collective on every rank, saving or not


def test_streamed_gather_hands_each_tensor_to_the_sink_and_keeps_nothing():
    full = _full_state()
    host = SimpleNamespace(
        accelerator=SimpleNamespace(is_fsdp2=False, get_state_dict=lambda model: dict(full)),
        args=SimpleNamespace(should_save=True),
        model="model",
    )
    seen = {}
    out = QuantCheckpointMixin._gathered_bf16_state(host, sink=lambda key, value: seen.__setitem__(key, value))
    assert out == {}
    assert list(seen) == list(full)
    assert seen["layers.0.weight"].dtype == torch.bfloat16
    assert torch.equal(seen["layers.0.scale"], full["layers.0.scale"])
