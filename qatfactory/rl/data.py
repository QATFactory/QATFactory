"""Prompt dataset and reward-function plumbing for the RL entry points.

Kept out of the entry point so every RL variant reads its data and scores its rollouts
through one implementation. A second copy would eventually drift, and a difference in how
prompts are built or answers are graded is indistinguishable, from the outside, from a
difference in the policy itself.
"""

import functools
import importlib
import logging
import sys

from datasets import load_dataset


logger = logging.getLogger(__name__)

# Python 3.11+ caps int -> str at 4300 digits. math_verify prints parsed answers through
# sympy, and a model will eventually emit an astronomically large integer, so the default
# cap turns one absurd completion into a dead run.
sys.set_int_max_str_digits(1_000_000)


def fault_tolerant_reward(reward_func):
    """Stop one pathological completion from killing a multi-hour run.

    Reward functions push untrusted model text through parsers (math_verify/sympy here),
    which can raise on inputs nobody anticipated; TRL propagates that and every rank dies.
    On failure this re-scores the batch one completion at a time so only the offending one
    is lost, reported as ``None`` — which TRL already treats as "no reward" (NaN).
    """

    @functools.wraps(reward_func)
    def wrapper(**kwargs):
        try:
            return reward_func(**kwargs)
        except Exception as error:
            logger.warning("Reward %s raised on a batch (%s); re-scoring per completion.", name, error)

        size = len(kwargs["completions"])
        rewards = []
        for index in range(size):
            single = {
                key: (value[index : index + 1] if isinstance(value, list) and len(value) == size else value)
                for key, value in kwargs.items()
            }
            try:
                rewards.extend(reward_func(**single))
            except Exception as error:
                logger.warning("Reward %s failed on one completion (%s); scoring it None.", name, error)
                rewards.append(None)
        return rewards

    name = getattr(reward_func, "__name__", repr(reward_func))
    return wrapper


def resolve_reward_funcs(names: list[str]) -> list:
    """Map config names to reward callables.

    Bare names come from ``trl.rewards``; a dotted ``package.module:function`` reference
    loads a project-specific reward without touching the entry point.
    """

    from trl import rewards

    resolved = []
    for name in names:
        if ":" in name:
            module_name, _, attribute = name.partition(":")
            resolved.append(getattr(importlib.import_module(module_name), attribute))
            continue
        if not hasattr(rewards, name):
            available = sorted(n for n in dir(rewards) if n.endswith("reward"))
            raise ValueError(f"Unknown reward function {name!r}; trl.rewards offers {available}.")
        resolved.append(getattr(rewards, name))
    return [fault_tolerant_reward(func) for func in resolved]


def append_prompt_suffix(dataset, suffix: str):
    """Append an instruction to every prompt's final user turn.

    Verifiable-reward datasets state the problem but not the answer format. A post-trained
    model already anchors its answer; one that does not scores zero everywhere, and a group
    with no reward variance has no advantage and contributes no gradient at all. Spelling
    the format out trades that risk for a policy that depends on the instruction.
    """

    def add(example):
        prompt = example["prompt"]
        if isinstance(prompt, str):
            example["prompt"] = f"{prompt}\n{suffix}"
            return example
        messages = [dict(message) for message in prompt]
        messages[-1]["content"] = f"{messages[-1]['content']}\n{suffix}"
        example["prompt"] = messages
        return example

    return dataset.map(add)


def build_datasets(data_cfg: dict):
    """Load the prompt dataset; RL needs only prompts (+ whatever the rewards read)."""

    path = data_cfg["dataset"]
    train_dataset = load_dataset(path, split=data_cfg.get("train_split", "train"))
    max_train = data_cfg.get("max_train_samples")
    if max_train:
        train_dataset = train_dataset.select(range(min(int(max_train), len(train_dataset))))

    eval_split = data_cfg.get("eval_split")
    eval_dataset = load_dataset(path, split=eval_split) if eval_split else None
    max_eval = data_cfg.get("max_eval_samples")
    if eval_dataset is not None and max_eval:
        eval_dataset = eval_dataset.select(range(min(int(max_eval), len(eval_dataset))))

    suffix = data_cfg.get("prompt_suffix")
    if suffix:
        train_dataset = append_prompt_suffix(train_dataset, suffix)
        if eval_dataset is not None:
            eval_dataset = append_prompt_suffix(eval_dataset, suffix)
        logger.info("Appended prompt suffix to every prompt: %r", suffix)

    logger.info(
        "Loaded %d train prompts%s from %s.",
        len(train_dataset),
        f" / {len(eval_dataset)} eval prompts" if eval_dataset is not None else "",
        path,
    )
    return train_dataset, eval_dataset
