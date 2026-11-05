"""A multimodal checkpoint's image encoder must stay out of the quantized set.

Vision towers reuse the text-side leaf names (`q_proj`, `k_proj`, `v_proj`), so target
selection by leaf name will happily reach into the encoder. Nothing downstream would
complain -- the run just silently quantizes and trains the image path -- so the exclusion
is pinned here against the naming conventions checkpoints actually use.
"""

import pytest
import torch
from torch import nn

from qatfactory.quant.fake_quant import DEFAULT_EXCLUDE_PREFIXES, apply_fake_quant

TEXT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _Attention(nn.Module):
    def __init__(self, dim, gated=False):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        if gated:
            self.gate_proj = nn.Linear(dim, dim, bias=False)


class _Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)


class _TextLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = _Attention(dim, gated=True)
        self.mlp = _Mlp(dim)


class _VisionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _Attention(dim)


class _Inner(nn.Module):
    def __init__(self, dim, vision_attr):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([_TextLayer(dim) for _ in range(2)])
        setattr(self, vision_attr, nn.Module())
        getattr(self, vision_attr).layers = nn.ModuleList([_VisionLayer(dim) for _ in range(2)])


class _Config:
    tie_word_embeddings = False


class _Multimodal(nn.Module):
    def __init__(self, dim=32, vision_attr="vision_tower"):
        super().__init__()
        self.model = _Inner(dim, vision_attr)
        self.lm_head = nn.Linear(dim, dim, bias=False)
        self.config = _Config()


@pytest.mark.parametrize("vision_attr", ["vision_tower", "visual"])
def test_vision_tower_linears_are_never_quantized(vision_attr):
    model = _Multimodal(vision_attr=vision_attr)

    converted = apply_fake_quant(
        model, TEXT_TARGETS, quant_format="nvfp4", lora_rank=8, lora_alpha=16,
        exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    )

    assert converted.linears, "the text tower should still be quantized"
    assert all(".language_model." in name for name in converted.linears)
    assert not any(vision_attr in name for name in converted.linears)


def test_vision_tower_parameters_stay_frozen():
    model = _Multimodal()

    apply_fake_quant(
        model, TEXT_TARGETS, quant_format="nvfp4", lora_rank=8, lora_alpha=16,
        exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    )

    vision = [(n, p) for n, p in model.named_parameters() if "vision_tower" in n]
    assert vision, "the fixture should have vision parameters"
    assert not any(p.requires_grad for _, p in vision)


def test_a_gated_attention_projection_is_quantized_with_the_rest_of_the_text_layer():
    """`self_attn.gate_proj` is a real text weight matrix, not an MLP name collision."""

    model = _Multimodal()

    converted = apply_fake_quant(
        model, TEXT_TARGETS, quant_format="nvfp4", lora_rank=8, lora_alpha=16,
        exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    )

    assert any(name.endswith("self_attn.gate_proj") for name in converted.linears)
    assert any(name.endswith("mlp.gate_proj") for name in converted.linears)
