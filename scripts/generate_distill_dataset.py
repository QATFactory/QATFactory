#!/usr/bin/env python3
"""Generate an on-policy conversation JSONL for QAD by sampling a model with vLLM.

Takes a prompt corpus (a Hugging Face dataset or a local JSONL), keeps the first user
turn of each conversation, samples one response per prompt from the teacher model, and
writes the ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`` JSONL that
``scripts/train_llm_qat.py`` consumes via ``data.train_eval_path``.

Everything is driven by a YAML config plus dotted CLI overrides, the same convention as the
training entry point::

    python scripts/generate_distill_dataset.py configs/data/<recipe>.yaml \\
        --prompts.num_samples 64 --output.path data/smoke.jsonl

Sampling runs against a **running vLLM OpenAI-compatible server** rather than an in-process
engine, because the reasoning split has to come from the server's ``--reasoning-parser``:
that parser is what knows the model's thinking convention (for Qwen3.8 the chat template
prefills ``<think>\\n`` into the generation prompt, so the sampled text has a closing tag and
no opening one). Taking ``reasoning_content`` from the parser keeps this script free of
per-model tag surgery, and lets vLLM own data parallelism and load balancing behind one
endpoint.

Requests are keyed by prompt index and appended to a raw shard as they land, so an
interrupted run resumes where it stopped. Filtering and assembly happen at merge time over
the raw shard, so the filter can be retuned without regenerating anything.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qatfactory.utils import apply_dotted_overrides, load_yaml_config, resolve_model_dir  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("qatfactory.gendata")


DEFAULTS = {
    "model": {
        "model_path": None,
        "revision": None,          # Hub commit to pin the tokenizer (and the served weights) to
        # Optional Jinja file overriding the checkpoint's own chat template. Leave unset to
        # use the serving template shipped with the model, which is what makes the samples
        # on-policy. Only used here to measure rendered prompt length; the server renders
        # the prompt it actually samples from.
        "chat_template_file": None,
        "system_prompt": None,
        # Forwarded to the server as `chat_template_kwargs`, e.g. {"reasoning_effort": "xhigh"}
        # for Qwen3.8. Recorded in the provenance file because the training-side template must
        # be rendered with the same settings or the corpus is off-distribution.
        "chat_template_kwargs": {},
    },
    "prompts": {
        "dataset": None,          # Hub dataset id, or a path to a local .jsonl/.json
        "split": "train",
        "name": None,             # Hub dataset config name
        "revision": None,         # Hub dataset commit; pin it so the prompt pool is reproducible
        # Optional per-source quotas, e.g. {"meta-math/MetaMathQA": 35000, "openbmb/UltraInteract_sft": null}:
        # only these sources are kept, each up to its count (null keeps every usable prompt of that
        # source), and num_samples is ignored. source_field names the column to match on.
        "source_quota": None,
        "source_field": "source",
        "conversation_field": "conversations",
        "role_key": "from",
        "content_key": "value",
        "user_roles": ["human", "user"],
        "metadata_fields": [],    # extra columns carried into the prompt pool for provenance
        "num_samples": 1000,
        "seed": 42,
        "dedup": True,
        "max_prompt_tokens": 8192,
        "min_prompt_chars": 1,
    },
    "sampling": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,              # -1 leaves the server default (no top-k truncation)
        "max_tokens": 8192,
        "seed": None,             # per-request sampling seed; null leaves it nondeterministic
    },
    "server": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": None,            # served model name; defaults to model.model_path
        "api_key": "EMPTY",
        "concurrency": 256,       # requests in flight across all data-parallel replicas
        "timeout": 3600,          # per-request seconds; a full 8k-token trace is not fast
        "max_retries": 5,
        "write_batch_size": 512,  # completions per durable flush
    },
    "filter": {
        "require_stop": True,     # drop truncated (length-capped) generations
        "require_reasoning": True,
        "min_answer_chars": 1,
        "max_answer_chars": None,
    },
    "output": {
        "path": None,
        "shard_dir": None,        # defaults to "<output.path>.shards"
        "shuffle_seed": 42,       # shuffle merged records; null keeps prompt-pool order
        "overwrite": False,
        # Split generation across servers (e.g. one per node): worker k answers the pool
        # prompts with index % num_workers == k into raw-<k>.jsonl and does not merge. Build
        # the pool once with --pool-only first, then run --merge-only when every worker is done.
        "num_workers": 1,
        "worker_index": 0,
    },
}


def merge_defaults(config: dict) -> dict:
    """Fill unset keys from DEFAULTS, one level deep into each section."""

    merged = {}
    for section, defaults in DEFAULTS.items():
        values = dict(defaults)
        values.update(config.get(section) or {})
        merged[section] = values
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}. Expected {sorted(DEFAULTS)}.")
    return merged


# ---------------------------------------------------------------------------------------
# Prompt pool
# ---------------------------------------------------------------------------------------


def _load_prompt_source(cfg: dict):
    from datasets import load_dataset

    dataset = cfg["dataset"]
    if dataset is None:
        raise ValueError("prompts.dataset is required (a Hub dataset id or a local .jsonl path).")
    if os.path.exists(dataset):
        return load_dataset("json", data_files=dataset, split="train")
    return load_dataset(dataset, cfg["name"], split=cfg["split"], revision=cfg["revision"])


def _first_user_turn(conversation, role_key: str, content_key: str, user_roles) -> str | None:
    """Return the first user message of a conversation, or None if it does not open with one.

    Conversations that open with a non-user turn (a system preamble, or an assistant-first
    record) are skipped rather than reordered: the prompt has to be something the model would
    plausibly be asked cold.
    """

    if not conversation:
        return None
    turn = conversation[0]
    if not isinstance(turn, dict):
        return None
    if turn.get(role_key) not in user_roles:
        return None
    content = turn.get(content_key)
    return content if isinstance(content, str) else None


def build_prompt_pool(cfg: dict, tokenizer, render_prompt) -> list[dict]:
    """Extract, dedupe, shuffle, length-filter and truncate the prompt corpus to num_samples.

    Length filtering runs on the fully rendered chat prompt (post chat template), because that
    is what has to fit under the server's max model length alongside the sampled continuation.
    It runs after the shuffle and stops as soon as num_samples survive, so tokenizing the whole
    corpus is never necessary. With ``source_quota`` the pool instead keeps each listed source
    up to its quota, in the same shuffled order.
    """

    source = _load_prompt_source(cfg)
    field = cfg["conversation_field"]
    if field not in source.column_names:
        raise ValueError(
            f"prompts.conversation_field={field!r} is not a column of the dataset "
            f"(columns: {source.column_names})."
        )
    metadata_fields = [f for f in cfg["metadata_fields"] if f in source.column_names]
    quota = cfg.get("source_quota") or None
    source_field = cfg.get("source_field") or "source"
    if quota is not None:
        if source_field not in source.column_names:
            raise ValueError(f"prompts.source_field={source_field!r} is not a column of the dataset.")
        present = set(source[source_field])
        missing = sorted(set(quota) - present)
        if missing:
            raise ValueError(f"prompts.source_quota names sources the dataset does not have: {missing}.")
        if source_field not in metadata_fields:
            metadata_fields.append(source_field)
    user_roles = set(cfg["user_roles"])
    min_chars = int(cfg["min_prompt_chars"])

    columns = source.select_columns([field] + metadata_fields)
    candidates: list[dict] = []
    seen: set[str] = set()
    dedup = bool(cfg["dedup"])
    for orig_row, row in enumerate(columns):
        text = _first_user_turn(row[field], cfg["role_key"], cfg["content_key"], user_roles)
        if text is None or len(text) < min_chars:
            continue
        if dedup:
            if text in seen:
                continue
            seen.add(text)
        # orig_row (the prompt's row in the source dataset) plus the metadata fields give every
        # conversation a stable identity, e.g. for the generalization splits.
        candidates.append({"prompt": text, "orig_row": orig_row, **{f: row[f] for f in metadata_fields}})
    logger.info("Prompt corpus: %d rows -> %d usable first-user turns.", len(source), len(candidates))

    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(candidates)

    wanted = int(cfg["num_samples"])
    max_prompt_tokens = int(cfg["max_prompt_tokens"])
    pool: list[dict] = []
    too_long = 0
    taken: Counter = Counter()
    for item in candidates:
        if quota is not None:
            src = item[source_field]
            if src not in quota or (quota[src] is not None and taken[src] >= quota[src]):
                continue
        token_ids = tokenizer(render_prompt(item["prompt"]), add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_prompt_tokens:
            too_long += 1
            continue
        item["index"] = len(pool)
        item["num_prompt_tokens"] = len(token_ids)
        pool.append(item)
        if quota is None:
            if len(pool) >= wanted:
                break
        else:
            taken[item[source_field]] += 1
            if all(cap is not None and taken[src] >= cap for src, cap in quota.items()):
                break
    if quota is not None:
        for src, cap in quota.items():
            logger.info("  source %-45s %7d prompts%s", src, taken[src], "" if cap is None else f" (quota {cap})")
            if cap is not None and taken[src] < cap:
                logger.warning("Source %s exhausted at %d of its quota %d.", src, taken[src], cap)
    elif len(pool) < wanted:
        logger.warning("Prompt pool exhausted at %d of the requested %d samples.", len(pool), wanted)
    logger.info(
        "Prompt pool: %d prompts (dropped %d over %d tokens), median %d tokens.",
        len(pool), too_long, max_prompt_tokens,
        sorted(p["num_prompt_tokens"] for p in pool)[len(pool) // 2] if pool else 0,
    )
    return pool


def write_jsonl(path: str | Path, records) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------------------
# Chat prompt rendering (local, for length filtering only)
# ---------------------------------------------------------------------------------------


def make_prompt_renderer(model_cfg: dict, tokenizer):
    """Return ``prompt_text -> rendered_chat_string`` using the model's own chat template.

    This mirrors what the server does to the same messages, so the token counts used for
    length filtering describe the prompt that is actually sampled.
    """

    chat_template = None
    template_file = model_cfg.get("chat_template_file")
    if template_file:
        chat_template = Path(template_file).read_text(encoding="utf-8")
    system_prompt = model_cfg.get("system_prompt")
    template_kwargs = dict(model_cfg.get("chat_template_kwargs") or {})

    def render(text: str) -> str:
        messages = build_messages(system_prompt, text)
        kwargs = {"tokenize": False, "add_generation_prompt": True, **template_kwargs}
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        return tokenizer.apply_chat_template(messages, **kwargs)

    logger.info("Rendered generation prompt: %r", render("PROBE"))
    return render


def build_messages(system_prompt: str | None, text: str) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return messages


# ---------------------------------------------------------------------------------------
# Generation against the vLLM server
# ---------------------------------------------------------------------------------------


async def _one_request(client, semaphore, config: dict, item: dict) -> dict | None:
    server_cfg = config["server"]
    sampling_cfg = config["sampling"]
    model_cfg = config["model"]

    body = {
        "model": server_cfg["model"],
        "messages": build_messages(model_cfg.get("system_prompt"), item["prompt"]),
        "temperature": float(sampling_cfg["temperature"]),
        "top_p": float(sampling_cfg["top_p"]),
        "max_tokens": int(sampling_cfg["max_tokens"]),
    }
    if int(sampling_cfg["top_k"]) > 0:
        body["top_k"] = int(sampling_cfg["top_k"])
    if sampling_cfg["seed"] is not None:
        body["seed"] = int(sampling_cfg["seed"])
    if model_cfg.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = dict(model_cfg["chat_template_kwargs"])

    last_error = None
    for attempt in range(int(server_cfg["max_retries"])):
        try:
            async with semaphore:
                response = await client.post("/chat/completions", json=body)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                await asyncio.sleep(min(2**attempt, 30))
                continue
            payload = response.json()
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            message = choice["message"]
            return {
                "index": item["index"],
                "prompt": item["prompt"],
                # The server's reasoning parser splits these; never re-derive them by
                # string surgery here, or the two would drift apart per model. vLLM renamed
                # the field to `reasoning` (0.27); older builds emit `reasoning_content`.
                "reasoning_content": message.get("reasoning") or message.get("reasoning_content") or "",
                "content": message.get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "num_prompt_tokens": usage.get("prompt_tokens"),
                "num_output_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last_error = repr(error)
            await asyncio.sleep(min(2**attempt, 30))
    logger.warning("Prompt %d failed after %s retries: %s", item["index"], server_cfg["max_retries"], last_error)
    return None


async def _generate_all(config: dict, todo: list[dict], out_path: Path) -> None:
    import httpx

    server_cfg = config["server"]
    concurrency = int(server_cfg["concurrency"])
    semaphore = asyncio.Semaphore(concurrency)
    started = time.time()
    done = 0
    failed = 0
    output_tokens = 0

    limits = httpx.Limits(max_connections=concurrency + 16, max_keepalive_connections=concurrency + 16)
    headers = {"Authorization": f"Bearer {server_cfg['api_key']}"}
    async with httpx.AsyncClient(
        base_url=server_cfg["base_url"], timeout=float(server_cfg["timeout"]), limits=limits, headers=headers
    ) as client:
        with open(out_path, "a", encoding="utf-8") as handle:
            pending = [asyncio.create_task(_one_request(client, semaphore, config, item)) for item in todo]
            since_flush = 0
            for future in asyncio.as_completed(pending):
                record = await future
                done += 1
                if record is None:
                    failed += 1
                else:
                    output_tokens += record["num_output_tokens"]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    since_flush += 1
                if since_flush >= int(server_cfg["write_batch_size"]):
                    handle.flush()
                    os.fsync(handle.fileno())
                    since_flush = 0
                if done % 200 == 0 or done == len(todo):
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-6)
                    logger.info(
                        "%d/%d done (%d failed, %.2f prompts/s, %.0f out-tok/s, %.0fs elapsed, "
                        "~%.0fs remaining).",
                        done, len(todo), failed, rate, output_tokens / max(elapsed, 1e-6), elapsed,
                        (len(todo) - done) / max(rate, 1e-6),
                    )
            handle.flush()
            os.fsync(handle.fileno())


def _worker(config: dict) -> tuple[int, int]:
    num_workers = int(config["output"]["num_workers"])
    worker_index = int(config["output"]["worker_index"])
    if num_workers < 1 or not 0 <= worker_index < num_workers:
        raise ValueError(f"Need 0 <= output.worker_index < output.num_workers, got {worker_index} of {num_workers}.")
    return worker_index, num_workers


def run_generation(config: dict, shard_dir: Path) -> None:
    worker_index, num_workers = _worker(config)
    pool = [item for item in read_jsonl(shard_dir / "prompts.jsonl") if item["index"] % num_workers == worker_index]
    out_path = shard_dir / f"raw-{worker_index:03d}.jsonl"
    done = set()
    if out_path.exists():
        done = {record["index"] for record in read_jsonl(out_path)}
        logger.info("Resuming: %d completions already on disk.", len(done))
    todo = [item for item in pool if item["index"] not in done]
    logger.info(
        "Worker %d of %d: %d prompts in its part of the pool, %d to generate.",
        worker_index, num_workers, len(pool), len(todo),
    )
    if not todo:
        return
    asyncio.run(_generate_all(config, todo, out_path))


# ---------------------------------------------------------------------------------------
# Merge: raw completions -> conversation JSONL
# ---------------------------------------------------------------------------------------


def merge_shards(config: dict, shard_dir: Path) -> dict:
    filter_cfg = config["filter"]
    out_path = Path(config["output"]["path"])

    pool = {item["index"]: item for item in read_jsonl(shard_dir / "prompts.jsonl")}
    raw_files = sorted(shard_dir.glob("raw-*.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"No raw shard files under {shard_dir}.")

    rejects = Counter()
    records: list[tuple[int, dict]] = []
    seen: set[int] = set()
    total_raw = 0
    output_tokens = 0
    min_answer = int(filter_cfg["min_answer_chars"])
    max_answer = filter_cfg["max_answer_chars"]

    for raw_file in raw_files:
        for row in read_jsonl(raw_file):
            total_raw += 1
            if row["index"] in seen:
                rejects["duplicate_index"] += 1
                continue
            seen.add(row["index"])
            if filter_cfg["require_stop"] and row["finish_reason"] != "stop":
                rejects[f"finish_reason:{row['finish_reason']}"] += 1
                continue
            reasoning = (row.get("reasoning_content") or "").strip()
            answer = (row.get("content") or "").strip()
            if filter_cfg["require_reasoning"] and not reasoning:
                rejects["missing_reasoning"] += 1
                continue
            if len(answer) < min_answer:
                rejects["empty_answer"] += 1
                continue
            if max_answer is not None and len(answer) > int(max_answer):
                rejects["answer_too_long"] += 1
                continue
            output_tokens += row["num_output_tokens"]
            prompt_item = pool.get(row["index"], {})
            provenance = {key: prompt_item[key] for key in ("source", "orig_row") if key in prompt_item}
            # Server-side token counts; shards from older runs fall back to the pool's own count.
            prompt_tokens = row.get("num_prompt_tokens")
            if prompt_tokens is None:
                prompt_tokens = prompt_item.get("num_prompt_tokens", 0)
            completion_tokens = row["num_output_tokens"]
            records.append((row["index"], {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    # `reasoning_content` is a first-class field, not an inline <think> block:
                    # the QAD chat template renders it natively into the supervised turn.
                    {"role": "assistant", "content": answer, "reasoning_content": reasoning},
                ],
                # Prompt provenance and generation stats; the trainer reads only `messages`.
                # The generalization split builder selects on `finish_reason` and
                # `usage.total_tokens` (scripts/generalization/prepare_splits.py).
                **provenance,
                "finish_reason": row["finish_reason"],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }))

    records.sort(key=lambda pair: pair[0])
    ordered = [record for _, record in records]
    shuffle_seed = config["output"]["shuffle_seed"]
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed)).shuffle(ordered)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(out_path, ordered)

    sources = Counter()
    for index in seen:
        item = pool.get(index) or {}
        if "source" in item:
            sources[item["source"]] += 1

    stats = {
        "prompt_pool": len(pool),
        "raw_completions": total_raw,
        "written": written,
        "yield": round(written / max(total_raw, 1), 4),
        "rejected": dict(rejects.most_common()),
        "mean_output_tokens": round(output_tokens / max(written, 1), 1),
        "prompt_sources": dict(sources.most_common()),
    }
    logger.info("Merged %d records -> %s (yield %.1f%%).", written, out_path, 100 * stats["yield"])
    for reason, count in rejects.most_common():
        logger.info("  rejected %-24s %d", reason, count)
    return stats


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------


def wait_for_server(config: dict, attempts: int = 120, delay: float = 10.0) -> None:
    """Block until the server answers /models, so a slow model load is not a crash."""

    import httpx

    base_url = config["server"]["base_url"]
    for attempt in range(attempts):
        try:
            response = httpx.get(f"{base_url}/models", timeout=10.0)
            if response.status_code == 200:
                served = [entry["id"] for entry in response.json().get("data", [])]
                logger.info("Server ready at %s, serving %s.", base_url, served)
                if config["server"]["model"] not in served:
                    raise ValueError(
                        f"server.model={config['server']['model']!r} is not served at {base_url} "
                        f"(available: {served})."
                    )
                return
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - the server is simply not up yet
            pass
        if attempt % 6 == 0:
            logger.info("Waiting for %s to come up ... (%ds)", base_url, int(attempt * delay))
        time.sleep(delay)
    raise TimeoutError(f"vLLM server at {base_url} did not become ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="YAML recipe; any value is overridable with --section.key value")
    parser.add_argument("--merge-only", action="store_true", help="Re-merge existing shards without generating.")
    parser.add_argument("--pool-only", action="store_true", help="Build the prompt pool and stop.")
    args, overrides = parser.parse_known_args()

    config = merge_defaults(apply_dotted_overrides(load_yaml_config(args.config), overrides))
    if not config["output"]["path"]:
        raise ValueError("output.path is required.")
    if not config["server"]["model"]:
        config["server"]["model"] = config["model"]["model_path"]

    shard_dir = Path(config["output"]["shard_dir"] or f"{config['output']['path']}.shards")
    _, num_workers = _worker(config)

    out_path = Path(config["output"]["path"])
    if out_path.exists() and not config["output"]["overwrite"] and not args.merge_only:
        raise FileExistsError(f"{out_path} already exists; set output.overwrite true to replace it.")

    if not args.merge_only:
        shard_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = shard_dir / "prompts.jsonl"
        if prompts_path.exists():
            logger.info("Reusing the prompt pool at %s.", prompts_path)
        elif num_workers > 1:
            # Workers building the pool at once would race on it.
            raise FileNotFoundError(f"No prompt pool at {prompts_path}; build it once with --pool-only first.")
        else:
            from qatfactory.data.text import load_tokenizer

            model_dir = resolve_model_dir(config["model"]["model_path"], config["model"]["revision"])
            # Not AutoTokenizer: DeepSeek-R1-Distill's local snapshot declares a tokenizer class that
            # drops every space, which would miscount prompt tokens and change the pool.
            tokenizer = load_tokenizer(model_dir)
            render = make_prompt_renderer(config["model"], tokenizer)
            pool = build_prompt_pool(config["prompts"], tokenizer, render)
            # Write via a temp file so an interrupted build never leaves a short pool behind,
            # which later runs would silently reuse.
            tmp_path = prompts_path.with_suffix(".jsonl.tmp")
            write_jsonl(tmp_path, pool)
            tmp_path.replace(prompts_path)
        if args.pool_only:
            return

        wait_for_server(config)
        run_generation(config, shard_dir)
        if num_workers > 1:
            logger.info("Worker done; merge with --merge-only once every worker has finished.")
            return

    stats = merge_shards(config, shard_dir)
    meta = {
        "config": config,
        "config_path": os.path.abspath(args.config),
        "stats": stats,
    }
    meta_path = Path(f"{config['output']['path']}.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote provenance to %s.", meta_path)


if __name__ == "__main__":
    main()
