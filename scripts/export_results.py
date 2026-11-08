"""Export only TwinPrune's rows in main Tables 2 and 3."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinprune.resources import DEFAULT_MANIFEST, RESOURCE_ROOT, archive_path
from twinprune.verify import verify_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--results-dir', type=Path, help='Full fresh inference outputs')
    source.add_argument('--reference', action='store_true', help='Audit stored scores; no model inference')
    parser.add_argument('--timing-dir', type=Path, help='Fresh measure_speedup.py outputs, named ROW.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.reference and args.timing_dir:
        parser.error('--timing-dir requires --results-dir')
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    timing_rows = json.loads((RESOURCE_ROOT / 'timing.json').read_text())['rows']
    measured = {}
    for cell in manifest['cells']:
        if args.reference:
            raw = archive_path(cell).read_bytes()
            if hashlib.sha256(raw).hexdigest() != cell['archive_sha256']:
                raise AssertionError(f'Corrupt archive: {cell["id"]}')
            payload = json.loads(gzip.decompress(raw))
        else:
            payload = json.loads((args.results_dir / f'{cell["id"]}.json').read_text())
        measured[cell['id']] = verify_scores(cell, payload, live=not args.reference)['value']

    prefill = []
    for timing in timing_rows:
        cells = [c for c in manifest['cells'] if c.get('paper_row') == timing['id']]
        row = {'Block': timing['id'].split('_')[1], 'Layer': timing['layer'],
               'Prune ratio': timing['prune_ratio'], 'TPL %': timing['tpl_percent']}
        row.update({c['benchmark']: measured[c['id']] for c in cells})
        row['Avg-4'] = sum(measured[c['id']] for c in cells) / 4
        row['Speedup'] = timing['speedup'] if args.reference else ''
        row['Speedup source'] = 'reference timing' if args.reference else 'not measured'
        if args.timing_dir:
            report = json.loads((args.timing_dir / f'{timing["id"]}.json').read_text())
            if (report['row'] != timing['id'] or report['status'] != 'MEASURED'
                    or not report['reference_a6000'] or report['protocol'] != timing['protocol']):
                raise AssertionError(f'Timing protocol differs: {timing["id"]}')
            identity = report['runtime'].get('checkpoint_identity', {})
            if (identity.get('model') != cells[0]['model']
                    or identity.get('revision') != cells[0]['model_revision']
                    or identity.get('weight_sha256') != cells[0]['checkpoint_weight_sha256']):
                raise AssertionError('Timing checkpoint differs from the accuracy checkpoint')
            if bool(report['runtime'].get('adapter_sha256', {})) != timing['adapter_required']:
                raise AssertionError('Timing adapter does not match this row')
            raw = report['raw_per_sample_ms']
            if any(len(raw[k]) != timing['protocol']['repeats'] for k in ['full_before', 'trial', 'full_after']):
                raise AssertionError('Incomplete timing repetitions')
            control_ms = (statistics.median(raw['full_before']) + statistics.median(raw['full_after'])) / 2
            speedup = control_ms / statistics.median(raw['trial'])
            if abs(speedup - report['speedup']) > 1e-10:
                raise AssertionError('Speedup disagrees with raw timings')
            row['Speedup'] = speedup
            row['Speedup source'] = 'fresh timing'
        prefill.append(row)

    decode = []
    for keep in [0.5, 0.25, 0.125]:
        cells = [c for c in manifest['cells'] if c['phase'] == 'decode' and c['decode_keep_ratio'] == keep]
        decode.append({'Keep ratio': keep, **{c['benchmark']: measured[c['id']] for c in cells}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('table2_twinprune.csv', prefill), ('table3_twinprune.csv', decode)]:
        with (args.output_dir / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    mode = 'reference audit' if args.reference else 'fresh inference'
    print(f'Exported 25 verified {mode} cells to {args.output_dir}')


if __name__ == '__main__':
    main()
