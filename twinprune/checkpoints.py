"""Resolve a pinned checkpoint and verify every local weight shard."""

from __future__ import annotations

import hashlib
from pathlib import Path


def resolve_pinned_checkpoint(cell: dict, override: str | None) -> tuple[str, dict | None]:
    """Check bytes before loading; a local directory name is not an identity."""
    if not cell.get('model_revision'):
        return override or cell['model'], None
    candidate = Path(override).expanduser() if override else None
    if candidate is not None and candidate.is_dir():
        directory = candidate
    else:
        if override and override != cell['model']:
            raise ValueError(f"Expected a local checkpoint or {cell['model']}; got {override}")
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=cell['model'], revision=cell['model_revision']))
    expected = cell.get('checkpoint_weight_sha256', {})
    if not expected:
        raise ValueError('A pinned checkpoint requires full weight-file digests')
    observed = {}
    for name, reference in expected.items():
        path = directory / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f'Missing checkpoint weight shard: {name}')
        with path.open('rb') as stream:
            observed[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if observed[name] != reference:
            raise ValueError(f"Checkpoint weight mismatch: {name}; require {cell['model']} "
                             f"revision {cell['model_revision']}")
    return str(directory), {
        'model': cell['model'], 'revision': cell['model_revision'],
        'weight_sha256': observed, 'verification': 'all weight-file SHA256 digests',
    }
