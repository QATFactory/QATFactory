import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_distill_dataset.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_distill_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_keeps_provenance_and_token_usage(tmp_path):
    """Merged rows carry what the generalization split builder selects on."""

    gen = _load_generator()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write(shard_dir / "prompts.jsonl", [
        {"index": 0, "prompt": "p0", "orig_row": 7, "source": "meta-math/MetaMathQA", "num_prompt_tokens": 11},
        {"index": 1, "prompt": "p1", "orig_row": 9, "source": "meta-math/MetaMathQA", "num_prompt_tokens": 12},
        {"index": 2, "prompt": "p2", "orig_row": 3, "source": "meta-math/MetaMathQA", "num_prompt_tokens": 13},
    ])
    _write(shard_dir / "raw-000.jsonl", [
        # Server-reported prompt tokens win over the pool's count.
        {"index": 0, "prompt": "p0", "reasoning_content": "think", "content": "a0",
         "finish_reason": "stop", "num_prompt_tokens": 20, "num_output_tokens": 100},
        # A shard written before prompt tokens were recorded falls back to the pool.
        {"index": 1, "prompt": "p1", "reasoning_content": "think", "content": "a1",
         "finish_reason": "stop", "num_output_tokens": 50},
        {"index": 2, "prompt": "p2", "reasoning_content": "think", "content": "a2",
         "finish_reason": "length", "num_prompt_tokens": 13, "num_output_tokens": 8192},
    ])
    config = gen.merge_defaults({"output": {"path": str(tmp_path / "out.jsonl"), "shuffle_seed": None}})

    stats = gen.merge_shards(config, shard_dir)

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert stats["written"] == 2  # the length-capped generation is filtered out
    assert [(r["source"], r["orig_row"]) for r in rows] == [("meta-math/MetaMathQA", 7), ("meta-math/MetaMathQA", 9)]
    assert all(r["finish_reason"] == "stop" for r in rows)
    assert rows[0]["usage"] == {"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120}
    assert rows[1]["usage"] == {"prompt_tokens": 12, "completion_tokens": 50, "total_tokens": 62}
    assert rows[0]["messages"][1] == {"role": "assistant", "content": "a0", "reasoning_content": "think"}


def test_workers_split_the_pool_and_merge_dedups(tmp_path, monkeypatch):
    """Each worker answers its own index residue into its own raw shard; the merge reads them all."""

    gen = _load_generator()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _write(shard_dir / "prompts.jsonl", [{"index": i, "prompt": f"p{i}", "num_prompt_tokens": 5} for i in range(7)])

    async def fake_generate_all(config, todo, out_path):
        with open(out_path, "a") as handle:
            for item in todo:
                handle.write(json.dumps({
                    "index": item["index"], "prompt": item["prompt"], "reasoning_content": "t",
                    "content": f"a{item['index']}", "finish_reason": "stop", "num_output_tokens": 3,
                }) + "\n")

    monkeypatch.setattr(gen, "_generate_all", fake_generate_all)
    seen = {}
    for worker in range(3):
        config = gen.merge_defaults({"output": {
            "path": str(tmp_path / "out.jsonl"), "shuffle_seed": None, "num_workers": 3, "worker_index": worker,
        }})
        gen.run_generation(config, shard_dir)
        seen[worker] = [r["index"] for r in gen.read_jsonl(shard_dir / f"raw-{worker:03d}.jsonl")]
        gen.run_generation(config, shard_dir)  # resuming a finished worker adds nothing
        assert [r["index"] for r in gen.read_jsonl(shard_dir / f"raw-{worker:03d}.jsonl")] == seen[worker]

    assert seen == {0: [0, 3, 6], 1: [1, 4], 2: [2, 5]}
    stats = gen.merge_shards(gen.merge_defaults({"output": {"path": str(tmp_path / "out.jsonl"), "shuffle_seed": None}}), shard_dir)
    assert stats["written"] == 7 and stats["raw_completions"] == 7


def test_worker_index_must_fit_num_workers():
    gen = _load_generator()
    try:
        gen._worker(gen.merge_defaults({"output": {"path": "x", "num_workers": 2, "worker_index": 2}}))
    except ValueError:
        return
    raise AssertionError("worker_index 2 of 2 was accepted")


class _FakeDataset:
    """Just enough of datasets.Dataset for build_prompt_pool."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        return [row[column] for row in self.rows]

    def select_columns(self, columns):
        return [{c: row[c] for c in columns} for row in self.rows]


def test_source_quota_keeps_listed_sources_up_to_their_quota(monkeypatch):
    gen = _load_generator()
    sources = ["a"] * 6 + ["b"] * 4 + ["c"] * 5
    rows = [{"conversations": [{"from": "human", "value": f"prompt {i} " + "x" * 20}], "source": s}
            for i, s in enumerate(sources)]
    monkeypatch.setattr(gen, "_load_prompt_source", lambda cfg: _FakeDataset(rows))
    cfg = gen.merge_defaults({"prompts": {
        "dataset": "fake", "metadata_fields": [], "num_samples": 1,
        "source_quota": {"a": 2, "b": None},
    }, "output": {"path": "x"}})["prompts"]
    tokenizer = lambda text, add_special_tokens=False: {"input_ids": text.split()}  # noqa: E731

    pool = gen.build_prompt_pool(cfg, tokenizer, lambda text: text)

    counts = {}
    for item in pool:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
        assert sources[item["orig_row"]] == item["source"]  # orig_row is the dataset row
    assert counts == {"a": 2, "b": 4}  # c is not listed; b has no cap; num_samples is ignored
    assert [item["index"] for item in pool] == list(range(len(pool)))
