"""Reject checkpoint and protocol mixing, without model weights or CUDA."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twinprune.checkpoints import resolve_pinned_checkpoint
from twinprune.evaluate import adapter_fingerprints
from twinprune.resources import DEFAULT_MANIFEST
from twinprune.verify import verify_scores


def main() -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    cells = manifest['cells']
    assert len(cells) == len({cell['id'] for cell in cells}) == 25
    assert sum(cell['paper_table'] == 2 for cell in cells) == 16
    assert sum(cell.get('adapter_required', False) for cell in cells) == 4
    assert all(c['method'] == 'cac_v4' for c in cells if c['phase'] == 'prefill')
    assert all(c['decode_strategy'] == 'density' and c['decode_keep_ratio'] < 1
               for c in cells if c['phase'] == 'decode')
    assert {cell['model'] for cell in cells} == {'OpenGVLab/InternVL3_5-8B-Instruct'}
    assert {cell['model_revision'] for cell in cells} == {'27506812aa9914804018996329e895977ee2d0c8'}
    assert all(len(cell['checkpoint_weight_sha256']) == 4 for cell in cells)
    assert DEFAULT_MANIFEST.read_bytes() == (ROOT / 'configs/unified_internvl35.json').read_bytes()

    with tempfile.TemporaryDirectory() as name:
        directory = Path(name) / 'misleading-model-name'
        directory.mkdir()
        path = directory / 'weights.safetensors'
        path.write_bytes(b'first checkpoint')
        cell = copy.deepcopy(cells[0])
        cell['checkpoint_weight_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
        _, identity = resolve_pinned_checkpoint(cell, str(directory))
        assert identity['weight_sha256'] == cell['checkpoint_weight_sha256']
        path.write_bytes(b'other checkpoint')
        try:
            resolve_pinned_checkpoint(cell, str(directory))
        except ValueError:
            pass
        else:
            raise AssertionError('A different same-size checkpoint was accepted')

        adapter = directory / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'synthetic adapter evidence')
        lora_cell = next(c for c in cells if c.get('adapter_required'))
        log = {
            'method': lora_cell['method'], 'layer': lora_cell['layer'],
            'prune_ratio': lora_cell['prune_ratio'], 'seed': 42,
            'n_samples': 2000, 'processed_examples': 2000, 'epochs': 1,
            'lora_r': 16, 'lora_alpha': 32, 'lr': 1e-4, 'grad_accum': 4,
            'model_hidden_size': 4096, 'dataset': 'aokvqa',
            'target_modules': ['q_proj', 'v_proj'],
        }
        identity = {'model': lora_cell['model'], 'revision': lora_cell['model_revision'],
                    'weight_sha256': lora_cell['checkpoint_weight_sha256']}
        for bad_identity in [{}, {**identity, 'model': 'OpenGVLab/InternVL3_5-8B'}]:
            log['base_checkpoint_identity'] = bad_identity
            (adapter / 'train_log.json').write_text(json.dumps(log))
            try:
                adapter_fingerprints(adapter, [lora_cell])
            except ValueError:
                pass
            else:
                raise AssertionError('Same-width adapter from an unidentified/different base was accepted')
        log['base_checkpoint_identity'] = identity
        (adapter / 'train_log.json').write_text(json.dumps(log))
        assert len(adapter_fingerprints(adapter, [lora_cell])) == 3

    cell = copy.deepcopy(cells[0])
    cell.update(samples=2, expected=.5,
                sample_ids_sha256=hashlib.sha256(json.dumps(['a', 'b']).encode()).hexdigest())
    payload = {
        'config': copy.deepcopy(cell), 'metrics': {cell['metric']: .5},
        'results': [{'id': 'a', 'score': 1}, {'id': 'b', 'score': 0}],
        'validation': {
            'complete': True, 'errors': 0, 'device': cell['reference_gpu'],
            **cell['reference_stack'], 'backend': 'legacy', 'strict_backend': True,
            'input_size': 448, 'tiles': 1, 'position_mode': 'legacy',
            'checkpoint_identity': {'model': cell['model'], 'revision': cell['model_revision'],
                                    'weight_sha256': cell['checkpoint_weight_sha256']},
        },
    }
    verify_scores(cell, payload, live=True)
    for key, value in [('device', 'NVIDIA GeForce RTX 4090'), ('backend', 'triton'),
                       ('position_mode', 'original'), ('checkpoint_identity', {}),
                       ('adapter_sha256', {'train_log.json': 'unexpected'})]:
        bad = copy.deepcopy(payload)
        bad['validation'][key] = value
        try:
            verify_scores(cell, bad, live=True)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Changed live {key} was accepted')
    print('25-cell coverage, checkpoint identity, adapters and runtime protocol: PASS')


if __name__ == '__main__':
    main()
