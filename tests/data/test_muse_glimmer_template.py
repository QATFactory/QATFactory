"""The Muse-Glimmer QAD chat template must render like the released one and expose a mask.

Two things can silently go wrong when adding generation markers to a released template: the
rendered text can drift from what the model was trained to see, or the mask can cover the
wrong span. Either produces a run that trains happily on the wrong objective.
"""

from importlib import resources

from transformers.utils.chat_template_utils import render_jinja_template

from qatfactory.data import resolve_chat_template


CONVERSATION = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "reasoning_content": "Two plus two.", "content": "It is 4."},
    {"role": "user", "content": "And 3 + 3?"},
    {"role": "assistant", "content": "It is 6."},
]


def _released_template() -> str:
    """The QAD template with its generation markers stripped back out."""

    qad = resolve_chat_template("muse_glimmer")
    return qad.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")


def _render(template: str, **kwargs):
    rendered = render_jinja_template(
        conversations=[CONVERSATION], chat_template=template, bos_token="<|begin_of_text|>", **kwargs
    )
    return rendered[0][0] if isinstance(rendered, tuple) else rendered[0]


def test_generation_markers_do_not_change_the_rendered_text():
    assert _render(resolve_chat_template("muse_glimmer")) == _render(_released_template())


def test_assistant_mask_covers_the_payloads_and_not_the_role_headers():
    rendered, indices = render_jinja_template(
        conversations=[CONVERSATION],
        chat_template=resolve_chat_template("muse_glimmer"),
        bos_token="<|begin_of_text|>",
        return_assistant_tokens_mask=True,
    )
    text = rendered[0]
    spans = [text[start:end] for start, end in indices[0]]

    # Reasoning and reply are separate assistant messages in this template's rendering, so
    # the first turn contributes two supervised spans and the second one.
    assert spans == ["Two plus two.<|eom|>", "It is 4.<|eot|>", "It is 6.<|eot|>"], spans
    for span in spans:
        assert "<|start|>" not in span
    joined = "".join(spans)
    assert "What is 2 + 2?" not in joined
    assert "You are a helpful AI assistant." not in joined


def test_template_ships_with_the_package():
    resource = resources.files("qatfactory.data.templates") / "muse_glimmer_qad.jinja"
    assert resource.is_file()
