"""Tokenization drops examples that truncation leaves with nothing to supervise."""

import json

from qatfactory.data.text import IGNORE_INDEX, _tokenize_batch


class _StubTokenizer:
    """Emits one token per word and marks the assistant turn."""

    def apply_chat_template(self, messages, **kwargs):
        input_ids, assistant_masks = [], []
        for message in messages:
            words = message["content"].split()
            input_ids.extend(range(len(input_ids), len(input_ids) + len(words)))
            assistant_masks.extend([1 if message["role"] == "assistant" else 0] * len(words))
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


def _batch(*conversations):
    return {"raw": [json.dumps({"messages": list(messages)}) for messages in conversations]}


def _conversation(prompt_words: int, answer_words: int):
    return [
        {"role": "user", "content": " ".join(["prompt"] * prompt_words)},
        {"role": "assistant", "content": " ".join(["answer"] * answer_words)},
    ]


def test_examples_truncated_before_the_assistant_turn_are_dropped():
    batch = _batch(_conversation(4, 4), _conversation(40, 4))

    result = _tokenize_batch(batch, _StubTokenizer(), template="", max_seq_len=8, truncate=True)

    # The second conversation's prompt alone overflows the window, so it carries no
    # supervised token and must not reach the trainer as an empty microbatch.
    assert len(result["input_ids"]) == 1
    assert len(result["labels"]) == 1
    assert any(label != IGNORE_INDEX for label in result["labels"][0])


def test_prompt_tokens_stay_masked_and_answer_tokens_are_kept():
    result = _tokenize_batch(
        _batch(_conversation(3, 2)), _StubTokenizer(), template="", max_seq_len=64, truncate=True
    )

    labels = result["labels"][0]
    assert labels[:3] == [IGNORE_INDEX] * 3
    assert labels[3:] == result["input_ids"][0][3:]


def test_a_single_supervised_token_at_the_window_edge_is_not_enough():
    # Only position 0 could be supervised, and the loss shift discards it.
    result = _tokenize_batch(
        _batch([{"role": "assistant", "content": "answer"}]),
        _StubTokenizer(),
        template="",
        max_seq_len=8,
        truncate=True,
    )
    assert result["input_ids"] == []
