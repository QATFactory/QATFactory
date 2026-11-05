import math

import numpy as np

from scripts.eval.deploy_kl_vllm import compare
from scripts.eval.gsm8k_vllm import final_number


def test_final_number_prefers_boxed_and_reads_after_the_thinking_block():
    assert final_number("<think>3+4=7 ... maybe 8</think> The answer is \\boxed{7}.") == 7.0
    assert final_number("<think>1,000 or 2,000?</think> So the total is 1,000 dollars, i.e. 1000.") == 1000.0
    assert final_number("<think>only reasoning, no answer") is None
    assert final_number("\\boxed{-2.5}") == -2.5


def test_compare_is_zero_for_identical_distributions_and_positive_otherwise():
    dist = {1: math.log(0.6), 2: math.log(0.3), 3: math.log(0.1)}
    ref = [[(5, 1, dist)]]
    same = compare(ref, [[(5, 1, dict(dist))]])
    assert same["n_compared"] == 1 and same["top1_agree_vs_bf16"] == 1.0
    assert abs(same["kl_topk_mean"]) < 1e-9

    shifted = {1: math.log(0.2), 2: math.log(0.7), 3: math.log(0.1)}
    diff = compare(ref, [[(5, 1, shifted)]])
    expected = sum(math.exp(lp) * (lp - shifted[t]) for t, lp in dist.items())
    assert np.isclose(diff["kl_topk_mean"], expected)
    # Same support and both already normalised: the reference-support convention agrees.
    assert np.isclose(diff["kl_ref_topk_mean"], expected)
    assert diff["top1_agree_vs_bf16"] == 0.0


def test_reference_support_kl_floors_tokens_missing_from_the_model_top_k():
    ref_dist = {1: math.log(0.5), 2: math.log(0.3), 3: math.log(0.1)}  # unnormalised: top-3 of a longer tail
    model_dist = {1: math.log(0.4), 4: math.log(0.3), 2: math.log(0.2)}  # token 3 absent -> floored at log(0.2)
    result = compare([[(0, 1, ref_dist)]], [[(0, 1, model_dist)]])
    floor = min(model_dist.values())
    expected = sum(math.exp(lp) * (lp - model_dist.get(t, floor)) for t, lp in ref_dist.items())
    assert np.isclose(result["kl_ref_topk_mean"], expected)
    assert result["top1_agree_vs_bf16"] == 1.0


def test_compare_skips_positions_missing_from_the_model_pass():
    dist = {1: 0.0}
    result = compare([[(5, 1, dist), (6, 1, dist)]], [[(5, 1, dist)]])
    assert result["n_compared"] == 1
