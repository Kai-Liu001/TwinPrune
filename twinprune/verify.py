"""Verify archived score evidence or newly generated model outputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from twinprune.resources import DEFAULT_MANIFEST, archive_path

ROOT = Path(__file__).resolve().parents[1]


def verify_scores(cell: dict, payload: dict, *, live: bool = False) -> dict:
    results = payload['results']
    if len(results) != cell['samples']:
        raise AssertionError(f'{cell["id"]}: {len(results)} != {cell["samples"]} samples')
    if live:
        if not cell.get('eligible_for_metric_reproduction', True):
            raise AssertionError(f'{cell["id"]}: historical target has invalid judge evidence; archive audit only')
        validation = payload['validation']
        if not validation['complete'] or validation['errors']:
            raise AssertionError(f'{cell["id"]}: incomplete or errored inference')
        for key in ['model', 'architecture', 'benchmark', 'phase', 'method', 'seed', 'layer', 'prune_ratio',
                    'decode_keep_ratio', 'decode_strategy', 'max_new_tokens', 'beta', 'dtype', 'scoring',
                    'judge_question_max_chars', 'model_revision', 'checkpoint_weight_sha256',
                    'position_mode', 'input_size', 'tiles']:
            if payload['config'].get(key) != cell.get(key):
                raise AssertionError(f'{cell["id"]}: changed protocol field {key}')
        for key, expected in cell.get('reference_stack', {}).items():
            if validation.get(key) != expected:
                raise AssertionError(f'{cell["id"]}: expected {key}={expected}, found {validation.get(key)}')
        if cell.get('model_revision'):
            identity = validation.get('checkpoint_identity', {})
            for key, expected in [('model', cell['model']), ('revision', cell['model_revision']),
                                  ('weight_sha256', cell['checkpoint_weight_sha256'])]:
                if identity.get(key) != expected:
                    raise AssertionError(f'{cell["id"]}: checkpoint identity differs: {key}')
            if validation.get('device') != cell['reference_gpu']:
                raise AssertionError(f'{cell["id"]}: reference GPU differs')
            if (validation.get('backend') != cell['required_backend']
                    or validation.get('strict_backend') is not True):
                raise AssertionError(f'{cell["id"]}: selection backend differs')
            for key in ['input_size', 'tiles', 'position_mode']:
                if validation.get(key) != cell.get(key):
                    raise AssertionError(f'{cell["id"]}: runtime {key} differs')
            if cell.get('adapter_required') and not validation.get('adapter_sha256', {}).get('train_log.json'):
                raise AssertionError(f'{cell["id"]}: missing full adapter training evidence')
            if not cell.get('adapter_required') and validation.get('adapter_sha256'):
                raise AssertionError(f'{cell["id"]}: unexpected adapter')
            if cell['benchmark'] == 'mmvet':
                if (payload['metrics'].get('judge_model') != cell['judge_model']
                        or payload['metrics'].get('judge_failures') != 0):
                    raise AssertionError(f'{cell["id"]}: MM-Vet judge protocol differs')
    scores = [float(result['score']) for result in results]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise AssertionError(f'{cell["id"]}: invalid per-sample score')
    measured = sum(scores) / len(scores)
    reported = float(payload['metrics'][cell['metric']])
    if abs(measured - reported) > 1e-10:
        raise AssertionError(f'{cell["id"]}: summary disagrees with per-sample scores')
    tolerance = cell.get('live_tolerance', 0.00005) if live else 1e-10
    if abs(measured - cell['expected']) > tolerance:
        raise AssertionError(f'{cell["id"]}: {measured:.8f} vs target {cell["expected"]:.8f}, tolerance {tolerance}')
    ids_hash = hashlib.sha256(json.dumps([str(result['id']) for result in results]).encode()).hexdigest()
    if ids_hash != cell['sample_ids_sha256']:
        raise AssertionError(f'{cell["id"]}: dataset order or sample IDs differ')
    return {'cell': cell['id'], 'metric': cell['metric'], 'value': measured,
            'expected': cell['expected'], 'difference': measured - cell['expected'],
            'samples': len(scores), 'mode': 'fresh inference' if live else 'archived-score audit', 'status': 'PASS',
            'eligible_for_metric_reproduction': cell.get('eligible_for_metric_reproduction', True),
            'reference_issues': cell.get('reference_issues', [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--group', nargs='+')
    parser.add_argument('--cell', nargs='+')
    parser.add_argument('--architecture', choices=['internvl'])
    parser.add_argument('--results-dir', type=Path, help='Fresh inference; otherwise audit archived per-sample scores')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = [cell for cell in manifest['cells']
                if (not args.group or cell['group'] in args.group)
                and (not args.cell or cell['id'] in args.cell)
                and (not args.architecture or cell['architecture'] == args.architecture)]
    if args.results_dir and args.group:
        selected = [cell for cell in selected if cell.get('eligible_for_metric_reproduction', True)]
    if not selected:
        parser.error('No matching cells')
    if args.cell and set(args.cell) - {cell['id'] for cell in selected}:
        parser.error('Unknown requested cell')
    if args.list:
        for cell in selected:
            note = ' [ARCHIVE ONLY: failed historical judging]' if not cell.get('eligible_for_metric_reproduction', True) else ''
            print(f'{cell["id"]:52} {cell["metric"]}={cell["expected"]:.8f} n={cell["samples"]}{note}')
        return
    rows = []
    for cell in selected:
        if args.results_dir:
            payload = json.loads((args.results_dir / f'{cell["id"]}.json').read_text())
        else:
            path = archive_path(cell)
            if hashlib.sha256(path.read_bytes()).hexdigest() != cell['archive_sha256']:
                raise AssertionError(f'Corrupt archive: {path}')
            payload = json.loads(gzip.decompress(path.read_bytes()))
        rows.append(verify_scores(cell, payload, live=bool(args.results_dir)))
    report = {'status': 'PASS', 'mode': rows[0]['mode'], 'cells': rows,
              'ineligible_reference_targets': [row['cell'] for row in rows
                                               if not row['eligible_for_metric_reproduction']],
              'final_manuscript_frozen': manifest['final_manuscript_frozen']}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(rows)} cells; {report["mode"]}')
    if report['ineligible_reference_targets']:
        print('Archive-only targets with invalid judge evidence: ' + ', '.join(report['ineligible_reference_targets']))
    if not manifest['final_manuscript_frozen']:
        print('These references do not certify all values in the published arXiv version.')


if __name__ == '__main__':
    main()
