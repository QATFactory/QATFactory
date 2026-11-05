"""The MiniMax-M2 QAD chat template must render like the released one and expose a mask."""

from importlib import resources

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from qatfactory.data import resolve_chat_template


CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
    {
        "role": "assistant",
        "reasoning_content": "Stale thinking from an earlier turn.",
        "content": "It is 4.",
    },
    {"role": "user", "content": "And 3 + 3?"},
    {
        "role": "assistant",
        "reasoning_content": "Three plus three is six.",
        "content": "It is 6.",
    },
]


def _released_template() -> str:
    """The QAD template with its generation markers stripped back out."""

    qad = resolve_chat_template("minimax_m2")
    return qad.replace("        {%- generation -%}\n", "").replace("        {%- endgeneration -%}\n", "")


def _render(template: str, **kwargs) -> str:
    rendered = render_jinja_template(
        conversations=[CONVERSATION], chat_template=template, **kwargs
    )
    return rendered[0][0] if isinstance(rendered, tuple) else rendered[0]


def test_generation_markers_do_not_change_the_rendered_text():
    assert _render(resolve_chat_template("minimax_m2")) == _render(_released_template())


def test_assistant_mask_covers_exactly_the_assistant_turns():
    template = resolve_chat_template("minimax_m2")
    rendered, indices = render_jinja_template(
        conversations=[CONVERSATION],
        chat_template=template,
        return_assistant_tokens_mask=True,
    )
    text = rendered[0]
    spans = [text[start:end] for start, end in indices[0]]

    assert len(spans) == 2, spans
    # Supervision starts after the `]~b]ai` role cue and ends with the turn terminator, so
    # visible content is trained on but the role cue itself is not.
    assert spans[0] == "It is 4.[e~[\n"
    # MiniMax deliberately renders reasoning only for the turn after the last user message;
    # earlier thinking is stripped as history, so the mask must not claim tokens it dropped.
    assert "Stale thinking from an earlier turn." not in text
    assert spans[1].startswith("<think>\nThree plus three is six.\n</think>")
    assert spans[1].endswith("It is 6.[e~[\n")
    for span in spans:
        assert "]~b]ai" not in span
        assert "]~b]user" not in span
    # The user turns stay outside every supervised span.
    assert "What is 2 + 2?" not in "".join(spans)


def test_template_ships_with_the_package():
    resource = resources.files("qatfactory.data.templates") / "minimax_m2_qad.jinja"
    assert resource.is_file()


def test_unknown_model_type_is_rejected():
    with pytest.raises(FileNotFoundError, match="No generation-marked chat template"):
        resolve_chat_template("definitely_not_a_model")
