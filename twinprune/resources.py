"""Bundled metric evidence, also available from a built wheel."""

from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parent / 'resources'
DEFAULT_MANIFEST = RESOURCE_ROOT / 'unified_internvl35.json'


def archive_path(cell: dict) -> Path:
    return RESOURCE_ROOT / 'reference_scores' / Path(cell['archive']).name
