"""Assemble independently scored GPU shards and verify the full frozen cell."""

import argparse
import hashlib
import json
from pathlib import Path

from twinprune.resources import DEFAULT_MANIFEST
from twinprune.verify import verify_scores


def merge_shards(cell: dict, payloads: list[dict]) -> dict:
    if not payloads:
        raise AssertionError('No shards found')
    payloads = sorted(payloads, key=lambda payload: payload['validation']['shard']['index'])
    first = payloads[0]
    count = first['validation']['shard']['count']
    if count < 2 or len(payloads) != count:
        raise AssertionError('Missing shards or invalid shard count')
    results, evidence = [], []
    for index, payload in enumerate(payloads):
        validation = payload['validation']
        shard = validation['shard']
        expected = {'index': index, 'count': count,
                    'start': cell['samples'] * index // count,
                    'stop': cell['samples'] * (index + 1) // count,
                    'full_sample_count': cell['samples']}
        if shard != expected or validation['errors'] or validation['complete']:
            raise AssertionError('Invalid, duplicate, or incomplete shard coverage')
        if payload['cell'] != cell['id'] or payload['config'] != first['config']:
            raise AssertionError('Shard protocols differ')
        for key in ['device', 'torch', 'transformers', 'cuda', 'flash_attention',
                    'backend', 'strict_backend', 'source_sha256', 'checkpoint_config_sha256',
                    'checkpoint_identity', 'adapter_sha256', 'input_size', 'tiles', 'position_mode']:
            if validation.get(key) != first['validation'].get(key):
                raise AssertionError(f'Shard runtime differs: {key}')
        if len(payload['results']) != shard['stop'] - shard['start']:
            raise AssertionError('Incomplete shard predictions')
        scores = [float(result['score']) for result in payload['results']]
        if abs(sum(scores) / len(scores) - payload['metrics'][cell['metric']]) > 1e-10:
            raise AssertionError('Shard metric differs from its per-sample scores')
        if payload['metrics'].get('judge_failures', 0):
            raise AssertionError('Shard has invalid judge evidence')
        results.extend(payload['results'])
        evidence.append({'index': index, 'start': shard['start'], 'stop': shard['stop']})
    scores = [float(result['score']) for result in results]
    mean = sum(scores) / len(scores)
    metrics = {cell['metric']: mean, 'correct': sum(scores), 'total': len(scores)}
    if cell['benchmark'] == 'mmvet':
        metrics.update(mmvet_score=mean, judge_failures=0,
                       judge_model=first['metrics']['judge_model'])
    validation = {**first['validation'], 'complete': True, 'shards': evidence,
                  'elapsed_seconds': sum(payload['validation'].get('elapsed_seconds', 0) for payload in payloads),
                  'elapsed_kind': 'sum of shard wall times',
                  'sample_ids_sha256': hashlib.sha256(json.dumps([str(r['id']) for r in results]).encode()).hexdigest()}
    validation.pop('shard')
    return {'cell': cell['id'], 'config': first['config'], 'metrics': metrics,
            'results': results, 'validation': validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--input-dirs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cells = {cell['id']: cell for cell in json.loads(args.manifest.read_text())['cells']}
    cell = cells[args.cell]
    paths = sorted(path for directory in args.input_dirs
                   for path in directory.glob(f'{args.cell}.shard-*.json')
                   if not path.name.endswith('.predictions.json'))
    payload = merge_shards(cell, [json.loads(path.read_text()) for path in paths])
    payload['validation']['shard_result_sha256'] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.cell}.json'
    if output.exists():
        raise FileExistsError(f'Refusing stale assembled output: {output}')
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    row = verify_scores(cell, payload, live=True)
    print(f'PASS: {cell["id"]}, {row["samples"]} samples from {len(paths)} real-inference shards')


if __name__ == '__main__':
    main()
