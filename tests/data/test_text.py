import os
import shutil

import pytest

from qatfactory.data import load_tokenizer, resolve_chat_template
from qatfactory.data.text import _ROUNDTRIP_PROBE


# Stock Qwen3 tokenizer, resolved from the HF cache or Hub; the tests skip when offline.
QWEN3_TOKENIZER_ID = "Qwen/Qwen3-8B"

# Resolved from the local HF cache when present; the test skips otherwise.
DEEPSEEK_R1_DISTILL_TOKENIZER_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def _qwen3_tokenizer_dir():
    """Local snapshot directory holding the Qwen3 tokenizer assets, or skip."""

    hub = pytest.importorskip("huggingface_hub")
    try:
        return hub.snapshot_download(
            QWEN3_TOKENIZER_ID, allow_patterns=["tokenizer*", "vocab.json", "merges.txt"]
        )
    except Exception:  # offline / not cached
        pytest.skip("Qwen3 tokenizer assets not available")


def _deepseek_tokenizer_dir(tmp_path):
    """Copy the DeepSeek tokenizer assets into a local directory, or skip."""

    hub = pytest.importorskip("huggingface_hub")
    try:
        snapshot = hub.snapshot_download(DEEPSEEK_R1_DISTILL_TOKENIZER_ID, allow_patterns=["tokenizer*"])
    except Exception:  # offline / not cached
        pytest.skip("DeepSeek-R1-Distill tokenizer assets not available")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(snapshot, name), tmp_path / name)
    return str(tmp_path)


def test_load_tokenizer_recovers_from_a_misdeclared_tokenizer_class(tmp_path):
    """DeepSeek-R1-Distill declares `LlamaTokenizerFast` over a byte-level BPE vocab.

    Honoring that class for a local directory yields a tokenizer that drops every space,
    which would silently corrupt training data; the loader must detect and route around it.
    """

    pytest.importorskip("transformers")
    tokenizer = load_tokenizer(_deepseek_tokenizer_dir(tmp_path))

    ids = tokenizer(_ROUNDTRIP_PROBE, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids) == _ROUNDTRIP_PROBE
    # Spaces survive as their own byte-level tokens rather than being swallowed.
    assert tokenizer.decode(tokenizer("Hello world", add_special_tokens=False)["input_ids"]) == "Hello world"
    # The special tokens and chat template from tokenizer_config.json are still picked up.
    assert tokenizer.eos_token == "<｜end▁of▁sentence｜>"
    assert tokenizer.chat_template


def test_export_repairs_a_misdeclared_tokenizer_class(tmp_path):
    """The exported tokenizer_config.json must load correctly from a local directory.

    vLLM loads the tokenizer from the export directory. With transformers releases that build
    DeepSeek's declared `LlamaTokenizerFast` as a sentencepiece-style tokenizer from a local
    directory, an export that copies the declaration verbatim serves text with every space dropped
    even though the weights are bit-exact; the asset copy must then re-declare the generic fast
    backend. Releases that load the declared class correctly need no repair.
    """

    import json

    transformers = pytest.importorskip("transformers")
    from qatfactory.data.text import tokenizer_roundtrips
    from qatfactory.quant.export.nvfp4_vllm import _repair_declared_tokenizer_class

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _deepseek_tokenizer_dir(export_dir)
    assert json.loads((export_dir / "tokenizer_config.json").read_text())["tokenizer_class"] == "LlamaTokenizerFast"
    declared_loads = tokenizer_roundtrips(transformers.AutoTokenizer.from_pretrained(str(export_dir)))

    _repair_declared_tokenizer_class(export_dir)

    config = json.loads((export_dir / "tokenizer_config.json").read_text())
    assert config["tokenizer_class"] == ("LlamaTokenizerFast" if declared_loads else "PreTrainedTokenizerFast")
    assert config["eos_token"]  # the rest of the config is untouched
    served = transformers.AutoTokenizer.from_pretrained(str(export_dir))
    assert served.decode(served("Hello world, 你好", add_special_tokens=False)["input_ids"]) == "Hello world, 你好"
    # Idempotent: a repaired config is left alone.
    _repair_declared_tokenizer_class(export_dir)
    assert json.loads((export_dir / "tokenizer_config.json").read_text()) == config


def test_export_keeps_a_healthy_tokenizer_class(tmp_path):
    import json

    pytest.importorskip("transformers")
    from qatfactory.quant.export.nvfp4_vllm import _repair_declared_tokenizer_class

    qwen3_dir = _qwen3_tokenizer_dir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(qwen3_dir, name), tmp_path / name)
    before = json.loads((tmp_path / "tokenizer_config.json").read_text())

    _repair_declared_tokenizer_class(tmp_path)

    assert json.loads((tmp_path / "tokenizer_config.json").read_text()) == before


def test_load_tokenizer_keeps_a_healthy_tokenizer_class():
    transformers = pytest.importorskip("transformers")
    qwen3_dir = _qwen3_tokenizer_dir()

    declared = transformers.AutoTokenizer.from_pretrained(qwen3_dir)
    loaded = load_tokenizer(qwen3_dir)

    # A round-tripping tokenizer is returned as-is, not swapped for the generic backend.
    assert type(loaded) is type(declared)
    assert loaded(_ROUNDTRIP_PROBE)["input_ids"] == declared(_ROUNDTRIP_PROBE)["input_ids"]


def test_resolve_qwen3_template_is_generation_marked_and_balanced():
    template = resolve_chat_template("qwen3")
    # Exactly one balanced generation span wraps the assistant turn.
    assert template.count("{%- generation %}") == 1
    assert template.count("{%- endgeneration %}") == 1


def test_resolve_qwen3_5_template_still_resolves():
    assert "{%- endgeneration %}" in resolve_chat_template("qwen3_5")


def test_resolve_deepseek_r1_distill_template_is_generation_marked_and_balanced():
    template = resolve_chat_template("deepseek_r1_distill")
    assert template.count("{%- generation %}") == 1
    assert template.count("{%- endgeneration %}") == 1


def test_missing_template_raises():
    with pytest.raises(FileNotFoundError):
        resolve_chat_template("nonexistent_model_type")


def test_qwen3_template_masks_the_assistant_turn_only():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_qwen3_tokenizer_dir())
    template = resolve_chat_template("qwen3")
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "<think>\nAdd them.\n</think>\n\n4"},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    mask = encoded["assistant_masks"]

    # A well-formed generation wrap masks in some (assistant) tokens but not all (the user
    # turn and system/prefix tokens stay unmasked).
    assert any(mask)
    assert not all(mask)
    # The final assistant `<|im_end|>` token is inside the generation span.
    assert mask[-1] == 1
    # The very first token (prefix / user turn) is outside the generation span.
    assert mask[0] == 0


def test_deepseek_r1_distill_template_keeps_the_reasoning_trace():
    """The QAD template must supervise the chain of thought the stock template discards."""

    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(DEEPSEEK_R1_DISTILL_TOKENIZER_ID)
    except Exception:  # offline / not cached
        pytest.skip("DeepSeek-R1-Distill tokenizer assets not available")

    template = resolve_chat_template("deepseek_r1_distill")
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "reasoning_content": "Add them.", "content": "4"},
    ]

    text = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=False
    )
    assert text.endswith("<｜Assistant｜><think>\nAdd them.\n</think>\n\n4<｜end▁of▁sentence｜>")

    # Training rendering and serving must agree: the generation prompt is exactly the
    # prefix of the assistant turn the student is trained to continue.
    prompt = tokenizer.apply_chat_template(
        messages[:1], chat_template=template, tokenize=False, add_generation_prompt=True
    )
    assert text.startswith(prompt)

    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    mask = encoded["assistant_masks"]

    # The assistant turn (through its trailing `<｜end▁of▁sentence｜>`) is supervised; the
    # bos/user prefix is not.
    assert any(mask)
    assert not all(mask)
    assert mask[0] == 0
    assert mask[-1] == 1
