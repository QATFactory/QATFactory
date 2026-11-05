"""Conversation tokenization for QAD: apply_chat_template + cached parallel build.

Records are ``{"messages": [{"role", "content", "reasoning_content"?, "tool_calls"?}, ...]}``.
We tokenize with the model's (generation-marked) chat template so ``reasoning_content`` is
rendered natively and the assistant turn is recovered as a token-level supervision mask.

Tokenization is parallel (``datasets.map(num_proc=...)``) and cached to an Arrow dataset on
disk, keyed by a fingerprint of the source file, tokenizer, template, sequence length, and
pipeline version. Subsequent runs (and non-zero ranks) load from that cache instead of
re-tokenizing. Only the main process builds the cache; other ranks wait via
``PartialState.main_process_first``.
"""

import hashlib
import json
import logging
import os
from importlib import resources
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..distill.kd_loss import IGNORE_INDEX


logger = logging.getLogger(__name__)

# Bump when the tokenization logic changes in a way that invalidates existing caches.
# v2: content-addressed tokenizer fingerprint (full backend definition + special-token roles),
# replacing the path-based fingerprint.
PIPELINE_VERSION = 3


# Exercises the properties a corrupt tokenizer class silently loses: interior spaces,
# newlines, and digit/punctuation grouping.
_ROUNDTRIP_PROBE = "Hello world,\n2 + 2 = 4"


def tokenizer_roundtrips(tokenizer) -> bool:
    """True if ``tokenizer`` encodes and decodes the probe text losslessly."""
    try:
        ids = tokenizer(_ROUNDTRIP_PROBE, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids) == _ROUNDTRIP_PROBE
    except Exception:  # a tokenizer too broken to encode is not usable either
        return False


def load_tokenizer(model_dir: str):
    """Load a tokenizer from a local directory, guarding against a mis-declared class.

    ``tokenizer_config.json`` may name a ``tokenizer_class`` that does not match the
    serialized ``tokenizer.json`` backend. DeepSeek-R1-Distill declares
    ``LlamaTokenizerFast`` over a byte-level BPE vocab; transformers honors that class for
    a local directory (but not for the same files fetched by Hub id) and builds a
    sentencepiece-style tokenizer that silently drops every space in both directions —
    ``"Hello world"`` encodes to two tokens and decodes back as ``"Helloworld"``. Training
    on that is silent corruption, so probe an encode/decode round-trip and, when it fails,
    fall back to the generic fast backend, which loads ``tokenizer.json`` verbatim while
    still picking up the special tokens and chat template from the same config.

    The fallback is only taken when it actually round-trips, so a tokenizer that is
    legitimately lossy (normalizers that lowercase or strip accents) keeps its declared
    class.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer_roundtrips(tokenizer):
        return tokenizer
    if not os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
        logger.warning(
            "Tokenizer %s does not round-trip the probe text and %s has no tokenizer.json to fall "
            "back to; tokenization may be corrupt.",
            type(tokenizer).__name__,
            model_dir,
        )
        return tokenizer

    fallback = PreTrainedTokenizerFast.from_pretrained(model_dir)
    if not tokenizer_roundtrips(fallback):
        logger.warning(
            "Neither the declared tokenizer class %s nor the generic fast backend round-trips the "
            "probe text; keeping %s.",
            type(tokenizer).__name__,
            type(tokenizer).__name__,
        )
        return tokenizer
    logger.warning(
        "Declared tokenizer class %s corrupts the probe text (%d tokens, decoded %r); using the "
        "generic fast backend from %s/tokenizer.json instead.",
        type(tokenizer).__name__,
        len(tokenizer(_ROUNDTRIP_PROBE, add_special_tokens=False)["input_ids"]),
        tokenizer.decode(tokenizer(_ROUNDTRIP_PROBE, add_special_tokens=False)["input_ids"]),
        model_dir,
    )
    return fallback


# Model types whose stock chat template is identical to another's (Qwen3-MoE ships the
# dense Qwen3 template verbatim).
_TEMPLATE_ALIASES = {"qwen3_moe": "qwen3"}


def resolve_chat_template(model_type: str, chat_template_file: str | None = None) -> str:
    """Return the generation-marked chat template string for a model type.

    An explicit ``chat_template_file`` wins; otherwise the shipped
    ``templates/{model_type}_qad.jinja`` is used.
    """

    if chat_template_file:
        return Path(chat_template_file).read_text(encoding="utf-8")
    model_type = _TEMPLATE_ALIASES.get(model_type, model_type)
    candidates = [model_type, model_type.removesuffix("_text")]
    for name in candidates:
        resource = resources.files("qatfactory.data.templates") / f"{name}_qad.jinja"
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No generation-marked chat template for model_type {model_type!r}. "
        f"Add qatfactory/data/templates/{model_type}_qad.jinja or set data.chat_template_file."
    )


def _tokenizer_fingerprint(tokenizer) -> str:
    """Content-addressed fingerprint of everything that changes the tokenized output.

    What the tokenization actually depends on (and therefore what belongs in the hash):

    - The tokenizer's full definition — vocab, **BPE merges**, normalizer, pre-tokenizer,
      post-processor, and added tokens. For fast tokenizers this is exactly
      ``backend_tokenizer.to_str()`` (the serialized ``tokenizer.json``), read from memory so it
      works whether the tokenizer came from a local dir or a Hub id. Slow tokenizers fall back
      to the sorted vocab (no merges available, but they are rare here).
    - **Special-token roles** (bos/eos/pad/unk + additional). The chat template renders these by
      role and the assistant-mask boundaries depend on them, yet role assignments live in
      ``tokenizer_config.json`` rather than the backend definition, so include them explicitly.

    Deliberately **path-independent** (no ``name_or_path``): two identical tokenizers at
    different paths share a cache; two different tokenizers never collide. The chat template,
    ``max_seq_len``, ``truncate`` and ``PIPELINE_VERSION`` are hashed separately in
    :func:`_cache_key`.
    """

    digest = hashlib.sha256()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        digest.update(backend.to_str().encode("utf-8"))
    else:
        digest.update(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode("utf-8"))
    special = {role: getattr(tokenizer, role, None) for role in ("bos_token", "eos_token", "pad_token", "unk_token")}
    special["additional_special_tokens"] = list(getattr(tokenizer, "additional_special_tokens", None) or [])
    digest.update(json.dumps(special, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    digest.update(str(len(tokenizer)).encode())
    return digest.hexdigest()


def _cache_key(source_path: str, tokenizer, template: str, max_seq_len: int, truncate: bool) -> str:
    stat = os.stat(source_path)
    digest = hashlib.sha256()
    for part in (
        os.path.abspath(source_path),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        _tokenizer_fingerprint(tokenizer),
        hashlib.sha256(template.encode()).hexdigest(),
        str(max_seq_len),
        str(bool(truncate)),
        str(PIPELINE_VERSION),
    ):
        digest.update(part.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def _tokenize_batch(batch, tokenizer, template, max_seq_len, truncate):
    """Tokenize conversations into input_ids plus an assistant-only label mask.

    Examples whose truncated window contains no assistant token are dropped. Keeping them
    would feed the trainer microbatches with nothing to supervise: the distillation loss is a
    per-token mean, so such a step contributes a zero to the running average and consumes an
    optimizer step without producing any gradient.
    """

    input_ids_batch, labels_batch = [], []
    for raw in batch["raw"]:
        messages = json.loads(raw)["messages"]
        encoded = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = encoded["input_ids"]
        assistant_mask = encoded["assistant_masks"]
        if truncate and len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            assistant_mask = assistant_mask[:max_seq_len]
        # The loss shifts by one, so the first position can never be supervised.
        if not any(assistant_mask[1:]):
            continue
        labels = [token if flag else IGNORE_INDEX for token, flag in zip(input_ids, assistant_mask)]
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
    return {"input_ids": input_ids_batch, "labels": labels_batch}


def build_tokenized_dataset(
    path: str,
    tokenizer,
    max_seq_len: int,
    *,
    model_type: str,
    cache_dir: str = "cache/tokenized",
    num_proc: int | None = None,
    chat_template_file: str | None = None,
    truncate: bool = True,
) -> datasets.Dataset:
    """Tokenize a JSONL conversation file to a cached Arrow ``Dataset`` of input_ids/labels."""

    from accelerate import PartialState

    template = resolve_chat_template(model_type, chat_template_file)
    key = _cache_key(path, tokenizer, template, max_seq_len, truncate)
    cache_path = os.path.join(cache_dir, f"{Path(path).stem}-{key}")
    num_proc = num_proc or min(os.cpu_count() or 1, 16)

    state = PartialState()
    with state.main_process_first():
        if not os.path.isfile(os.path.join(cache_path, "dataset_info.json")):
            logger.info("Tokenizing %s -> %s (num_proc=%d)", path, cache_path, num_proc)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
            raw_dataset = datasets.Dataset.from_dict({"raw": lines})
            tokenized = raw_dataset.map(
                _tokenize_batch,
                batched=True,
                num_proc=num_proc,
                remove_columns=["raw"],
                load_from_cache_file=False,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "template": template,
                    "max_seq_len": max_seq_len,
                    "truncate": truncate,
                },
                desc="tokenize",
            )
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tokenized.save_to_disk(cache_path)

    dataset = datasets.load_from_disk(cache_path)
    logger.info("Loaded %d tokenized examples from %s", len(dataset), cache_path)
    return dataset


class QADCollator:
    """Right-pads a batch of tokenized conversations and synthesizes the attention mask."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            label = list(feature["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(label + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
