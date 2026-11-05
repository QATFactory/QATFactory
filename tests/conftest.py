import importlib
import inspect

import pytest


def _torch_function(module, value):
    """The pure-torch function behind a kernel wrapper defined in `module`, or None."""

    if not inspect.isfunction(value):
        return None
    fn = inspect.unwrap(value)
    return fn if fn is not value and fn.__module__ == module.__name__ else None


@pytest.fixture
def torch_delta_net(monkeypatch):
    """Build Qwen3.5 linear-attention layers on their pure-torch path.

    When flash-linear-attention is installed (it is in any Qwen3.5 training environment),
    transformers binds its Triton kernels, and those kernels reject the CPU tensors the tiny test
    models run on. Older transformers import the kernels under their own names and take the torch
    path when a name is None. Newer releases wrap each torch function in a decorator that swaps in
    the kernel whenever its package imports, so the wrapper goes back to the function it wraps.
    """

    for name in ("qwen3_5", "qwen3_5_moe"):
        try:
            module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
        for attr, value in list(vars(module).items()):
            torch_fn = _torch_function(module, value)
            if torch_fn is not None:
                monkeypatch.setattr(module, attr, torch_fn)
        for attr in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "FusedRMSNormGated",
                     "causal_conv1d_fn", "causal_conv1d_update"):
            value = getattr(module, attr, None)
            if value is not None and getattr(value, "__module__", None) != module.__name__:
                monkeypatch.setattr(module, attr, None)
