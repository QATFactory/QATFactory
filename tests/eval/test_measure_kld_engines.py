import argparse
import math

import pytest

from scripts.eval.quality import measure_kld as mk


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _capture(monkeypatch, payload):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"], sent["json"] = url, json
        return _Resp(payload)

    monkeypatch.setattr(mk.requests, "post", fake_post)
    return sent


def test_parse_endpoint_defaults_to_vllm_and_accepts_sglang():
    assert mk.parse_endpoint("bf16=http://h:1/v1=M") == ("bf16", "http://h:1/v1", "M", "vllm")
    assert mk.parse_endpoint("q=http://h:2/v1=Q=sglang") == ("q", "http://h:2/v1", "Q", "sglang")
    with pytest.raises(argparse.ArgumentTypeError):
        mk.parse_endpoint("q=http://h:2/v1=Q=trtllm")


def test_sglang_request_uses_native_generate_and_parses_input_top_logprobs(monkeypatch):
    # SGLang returns one entry per input token: None for position 0, then [logprob, token_id, text] triples.
    payload = {"meta_info": {"input_top_logprobs": [
        None,
        [[math.log(0.7), 11, None], [math.log(0.2), 12, None]],
        [[math.log(0.9), 13, None]],
    ]}}
    sent = _capture(monkeypatch, payload)
    out = mk.fetch_prompt_logprobs("http://h:2/v1", "Q", [5, 11, 13], top_k=2, timeout=1, api="sglang")
    assert sent["url"] == "http://h:2/generate"  # the /v1 suffix is stripped for the native API
    assert sent["json"]["input_ids"] == [5, 11, 13]
    assert sent["json"]["return_logprob"] and sent["json"]["logprob_start_len"] == 0
    assert sent["json"]["top_logprobs_num"] == 2
    assert sent["json"]["sampling_params"] == {"max_new_tokens": 1, "temperature": 0.0}
    assert out[0] is None
    assert out[1] == {11: pytest.approx(math.log(0.7)), 12: pytest.approx(math.log(0.2))}
    assert out[2] == {13: pytest.approx(math.log(0.9))}


def test_vllm_request_is_unchanged(monkeypatch):
    payload = {"choices": [{"prompt_logprobs": [None, {"11": {"logprob": -0.1}, "12": {"logprob": -2.0}}]}]}
    sent = _capture(monkeypatch, payload)
    out = mk.fetch_prompt_logprobs("http://h:1/v1", "M", [5, 11], top_k=2, timeout=1)
    assert sent["url"] == "http://h:1/v1/completions"
    assert sent["json"]["prompt"] == [5, 11] and sent["json"]["prompt_logprobs"] == 2
    assert out == [None, {11: -0.1, 12: -2.0}]


def test_identical_distributions_from_the_two_engines_give_zero_kl(monkeypatch):
    dist = [(math.log(0.6), 7), (math.log(0.3), 8)]
    _capture(monkeypatch, {"meta_info": {"input_top_logprobs": [None, [[lp, t, None] for lp, t in dist]]}})
    sglang_seq = mk.fetch_prompt_logprobs("http://h:2", "Q", [1, 7], top_k=2, timeout=1, api="sglang")
    _capture(monkeypatch, {"choices": [{"prompt_logprobs": [None, {str(t): {"logprob": lp} for lp, t in dist}]}]})
    vllm_seq = mk.fetch_prompt_logprobs("http://h:1/v1", "M", [1, 7], top_k=2, timeout=1)
    kl_sum, tokens = mk.kl_accumulate(vllm_seq, sglang_seq)
    assert tokens == 1 and abs(kl_sum) < 1e-12
