"""Check the exact data-file fingerprints used by the release validation."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, default=ROOT / 'configs/data.sha256.json')
    args = parser.parse_args()
    paths = yaml.safe_load(args.data_config.read_text())
    inventory = json.loads(args.inventory.read_text())
    checked = 0
    for benchmark, files in inventory['benchmarks'].items():
        if benchmark not in paths:
            continue
        base = Path(paths[benchmark])
        if base.is_file():
            base = base.parent
        for name, expected in files.items():
            path = base / name
            if not path.exists():
                raise FileNotFoundError(f'Missing pinned data file: {benchmark}/{name}')
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected['sha256']:
                raise AssertionError(f'Changed data file: {benchmark}/{name}')
            checked += 1
    if not checked:
        raise ValueError('No benchmark files selected')
    print(f'Dataset fingerprints: PASS ({checked} files)')


if __name__ == '__main__':
    main()
