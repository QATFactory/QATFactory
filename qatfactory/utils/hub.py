"""Resolve a model reference (local directory or Hugging Face Hub repo id) to a local dir.

The training/export scripts historically assumed ``model_path`` was a local directory
(they read ``config.json`` and copy tokenizer assets from it). Accepting a bare Hub repo
id (e.g. ``Qwen/Qwen3-8B``) means we first materialize a local snapshot, then everything
downstream keeps using an on-disk directory.
"""

import os


def resolve_model_dir(path_or_repo: str, revision: str | None = None) -> str:
    """Return a local directory for a path or a Hugging Face Hub repo id.

    A local directory is returned unchanged. Otherwise the reference is treated as a Hub
    repo id and downloaded to the default cache via ``snapshot_download``, at ``revision``
    (a commit hash, branch or tag) when one is given.
    """

    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    return snapshot_download(path_or_repo, revision=revision)
