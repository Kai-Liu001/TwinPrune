"""Run real model inference for a frozen cell in the metric manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

from twinprune.resources import DEFAULT_MANIFEST


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_ids_hash(items: list[dict]) -> str:
    ids = [str(item.get('id', item.get('index', index))) for index, item in enumerate(items)]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def source_hashes() -> dict:
    hashes = {}
    for package in ['src', 'phase2/benchmarks', 'twinprune']:
        for path in sorted((ROOT / package).rglob('*.py')):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def adapter_fingerprints(path: Path, cells: list[dict], require_full: bool = True) -> dict:
    """Reject partial training or an adapter trained for another policy."""
    files = ['adapter_config.json', 'adapter_model.safetensors']
    log = {}
    log_path = path / 'train_log.json'
    if log_path.exists():
        log = json.loads(log_path.read_text())
        for cell in cells:
            for key, expected in [('method', cell['method']), ('layer', cell['layer']),
                                  ('prune_ratio', cell['prune_ratio']), ('seed', cell['seed'])]:
                if log.get(key) != expected:
                    raise ValueError(f'Adapter training {key} does not match {cell["id"]}')
        if require_full:
            expected = {'n_samples': 2000, 'processed_examples': 2000, 'epochs': 1,
                        'lora_r': 16, 'lora_alpha': 32, 'lr': 1e-4, 'grad_accum': 4,
                        'model_hidden_size': 4096, 'dataset': 'aokvqa',
                        'target_modules': ['q_proj', 'v_proj']}
            if any(log.get(key) != value for key, value in expected.items()):
                raise ValueError('Adapter lacks the complete published-checkpoint training recipe')
        files.append('train_log.json')
    elif require_full:
        raise ValueError('Full LoRA replay requires its training log')
    fingerprints = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in files}
    for cell in cells:
        if not cell.get('model_revision'):
            continue
        if fingerprints == cell.get('reference_adapter_sha256'):
            continue
        identity = log.get('base_checkpoint_identity', {})
        if (identity.get('model') != cell['model']
                or identity.get('revision') != cell['model_revision']
                or identity.get('weight_sha256') != cell['checkpoint_weight_sha256']):
            raise ValueError('Adapter was not trained on the pinned Instruct checkpoint')
    return fingerprints


def benchmark_for(cell: dict, data_root: Path, data_config: Path | None):
    from phase2.benchmarks import BENCHMARKS
    paths = {
        'mmbench': 'MMBench', 'pope': 'POPE', 'scienceqa': 'scienceqa/data',
        'textvqa': 'TextVQA_raw/data', 'docvqa': 'DocVQA_raw/DocVQA',
        'mmvet': 'mmvet',
    }
    if data_config:
        import yaml
        overrides = yaml.safe_load(data_config.read_text())
    else:
        overrides = {}
    name = cell['benchmark']
    path = Path(overrides.get(name, data_root / paths[name]))
    if not path.exists():
        raise FileNotFoundError(f'Missing {name} dataset: {path}')
    return BENCHMARKS[name](data_path=str(path))


def make_engine(cell: dict, model, processor, backend: str):
    from src.latc import LATCConfig, LATCInferenceEngine

    if cell['architecture'] != 'internvl':
        raise ValueError('The main comparison tables use InternVL only')
    if cell['phase'] == 'decode':
        if cell['decode_strategy'] != 'density':
            raise ValueError('Only TwinPrune density eviction is included')
        from src.internvl_decode_eviction_engine import InternVLDecodeEvictionEngine
        return InternVLDecodeEvictionEngine(
            model, processor, keep_ratio=cell['decode_keep_ratio'],
            strategy='density', seed=cell['seed'],
        )
    if cell['phase'] != 'prefill' or cell['method'] != 'cac_v4' or cell['beta'] != 0.3:
        raise ValueError('Only the main-table TwinPrune prefill configuration is included')
    config = LATCConfig(
        stages=[(cell['layer'], cell['prune_ratio'])], strategy='cac_v4',
        seed=cell['seed'], position_mode=cell.get('position_mode', 'legacy'),
    )
    return LATCInferenceEngine(model, processor, config)


def load_model(cell: dict, checkpoint: str | None, device: str):
    if cell['architecture'] != 'internvl' or cell['model'] != 'OpenGVLab/InternVL3_5-8B-Instruct':
        raise ValueError('The main tables require the pinned InternVL3.5-8B-Instruct checkpoint')
    try:
        import flash_attn
    except ImportError as exc:
        raise RuntimeError('Metric reproduction requires flash-attn==2.8.3; see REPRODUCE.md') from exc
    if flash_attn.__version__ != '2.8.3':
        raise RuntimeError(f'Expected FlashAttention 2.8.3, found {flash_attn.__version__}')
    from src.model_utils import load_internvl
    from twinprune.checkpoints import resolve_pinned_checkpoint
    resolved, identity = resolve_pinned_checkpoint(cell, checkpoint)
    model, processor = load_internvl(
        resolved, device=device, dtype=cell.get('dtype', 'bfloat16'),
    )
    language = model.config.llm_config
    actual = (language.num_hidden_layers, language.hidden_size, language.num_key_value_heads)
    if actual != (36, 4096, 8):
        raise RuntimeError(f'Checkpoint language shape {actual} differs from the pinned model')
    model._twinprune_checkpoint_identity = identity
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--cell', nargs='+', help='Frozen cell IDs; use twinprune-verify --list')
    selection.add_argument('--group', nargs='+', help='Evaluate every cell in the selected manifest groups')
    parser.add_argument('--architecture', choices=['internvl'],
                        help='Select one model architecture, including its pinned environment')
    parser.add_argument('--model', help='Local checkpoint directory or Hugging Face ID')
    parser.add_argument('--adapter', type=Path,
                        help='Fresh method-specific LoRA adapter for adapter-required cells')
    parser.add_argument('--data-root', type=Path, default=os.environ.get('DATASETS_ROOT', 'data'))
    parser.add_argument('--data-config', type=Path, help='YAML mapping of benchmarks to local paths')
    parser.add_argument('--backend', choices=['legacy', 'graph', 'triton'], default='legacy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, help='Smoke-test prefix; never reported as full reproduction')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error('--max-samples must be positive')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Require num-shards >= 1 and 0 <= shard-index < num-shards')
    if args.num_shards > 1 and args.max_samples:
        parser.error('Use full frozen samples when sharding; smoke prefixes cannot be assembled')
    os.environ['TWINPRUNE_FPS_BACKEND'] = args.backend
    os.environ['CAC_FPS_BACKEND'] = args.backend
    os.environ['TWINPRUNE_STRICT_BACKEND'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import numpy as np
    import torch
    from tqdm import tqdm
    from src.model_utils import prepare_pixel_values
    manifest = read_manifest(args.manifest)
    implementation_hashes = source_hashes()
    cells = {cell['id']: cell for cell in manifest['cells']}
    if args.group:
        unknown_groups = set(args.group) - {cell['group'] for cell in cells.values()}
        if unknown_groups:
            parser.error(f'Unknown groups: {sorted(unknown_groups)}')
        args.cell = [cell['id'] for cell in cells.values() if cell['group'] in args.group]
        excluded = [identifier for identifier in args.cell
                    if not cells[identifier].get('eligible_for_metric_reproduction', True)]
        if excluded:
            print('Skipping archive-only targets: ' + ', '.join(excluded), flush=True)
        args.cell = [identifier for identifier in args.cell if identifier not in excluded]
    if args.architecture:
        args.cell = [identifier for identifier in args.cell
                     if identifier in cells and cells[identifier]['architecture'] == args.architecture]
    if not args.cell:
        parser.error('No eligible cells match the selection')
    if args.model and len({cells[identifier]['model'] for identifier in args.cell if identifier in cells}) > 1:
        parser.error('--model cannot override multiple distinct checkpoints; select one model per run')
    unknown = set(args.cell) - cells.keys()
    if unknown:
        parser.error(f'Unknown cells: {sorted(unknown)}')
    adapter_cells = [cells[name] for name in args.cell if cells[name].get('adapter_required')]
    if adapter_cells and not args.adapter:
        parser.error('Selected LoRA cells require --adapter; a base checkpoint is not a recovery result')
    if args.adapter and (len(adapter_cells) != len(args.cell) or
                         len({c['method'] for c in adapter_cells}) != 1):
        parser.error('One adapter may only evaluate its own method-specific LoRA cells')
    adapter_hashes = {}
    if args.adapter:
        adapter_hashes = adapter_fingerprints(args.adapter, adapter_cells,
                                              require_full=args.max_samples is None)
    if not torch.cuda.is_available():
        raise RuntimeError('Real model evaluation requires CUDA')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded_key = None
    model = processor = None
    for identifier in args.cell:
        cell = cells[identifier]
        if cell.get('required_backend') and args.backend != cell['required_backend']:
            raise ValueError(f'{identifier}: requires the {cell["required_backend"]} backend')
        if not cell.get('eligible_for_metric_reproduction', True):
            raise RuntimeError(f'{identifier}: historical judge failures make this target ineligible; see release audit')
        stack = cell.get('reference_stack', {})
        if stack.get('torch') and torch.__version__ != stack['torch']:
            raise RuntimeError(f'{identifier}: requires Torch {stack["torch"]}; use the CUDA 12.6 setup in REPRODUCE.md')
        suffix = f'.shard-{args.shard_index}-of-{args.num_shards}' if args.num_shards > 1 else ''
        output = args.output_dir / f'{identifier}{suffix}.json'
        if output.exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to reuse stale output: {output}')
        random.seed(cell['seed']); np.random.seed(cell['seed']); torch.manual_seed(cell['seed'])
        torch.cuda.manual_seed_all(cell['seed'])
        key = (cell['model'], cell['architecture'], cell.get('dtype', 'bfloat16'),
               cell.get('device_map'))
        if key != loaded_key:
            del model, processor
            torch.cuda.empty_cache()
            model, processor = load_model(cell, args.model, args.device)
            if args.adapter:
                if cell['architecture'] != 'internvl':
                    raise ValueError('This adapter loader supports InternVL language transformers only')
                from peft import PeftModel
                transformer = PeftModel.from_pretrained(model.language_model.model, str(args.adapter))
                model.language_model.model = transformer.merge_and_unload()
                model.eval()
            loaded_key = key
        benchmark = benchmark_for(cell, args.data_root, args.data_config)
        if cell['benchmark'] == 'mmvet':
            benchmark.pin_reference_judgments(cell)
        items = benchmark.load_data()
        if len(items) < cell['samples']:
            raise ValueError(f'{identifier}: found {len(items)} samples, require {cell["samples"]}')
        limit = min(args.max_samples or cell['samples'], cell['samples'])
        items = items[:limit]
        if limit == cell['samples'] and sample_ids_hash(items) != cell['sample_ids_sha256']:
            raise ValueError(f'{identifier}: dataset order or sample IDs differ before inference')
        start = limit * args.shard_index // args.num_shards
        stop = limit * (args.shard_index + 1) // args.num_shards
        items = items[start:stop]
        if not items:
            raise ValueError('Empty shard; reduce num-shards')
        engine = make_engine(cell, model, processor, args.backend)
        results = []
        started = time.monotonic()
        for item in tqdm(items, desc=identifier):
            question = benchmark.format_prompt(item)
            image = benchmark.get_image(item)
            if image is None:
                raise ValueError(f'Missing image at sample {len(results)}')
            model_input = prepare_pixel_values(
                image, model, args.device, input_size=cell.get('input_size', 448),
            )
            prediction = engine.generate(model_input, question, max_new_tokens=cell['max_new_tokens'])
            if not prediction.strip():
                raise RuntimeError(f'Empty model output at sample {len(results)}')
            answer = item.get('answer', item.get('answers', []))
            if hasattr(answer, 'tolist'):
                answer = answer.tolist()
            recorded_question = question
            if cell.get('judge_question_max_chars'):
                recorded_question = question[:cell['judge_question_max_chars']]
            results.append({'id': item.get('id', item.get('index', start + len(results))),
                            'question': recorded_question, 'prediction': prediction,
                            'answer': answer,
                            'score': float(benchmark.score(prediction, item))})
        if cell['benchmark'] == 'mmvet':
            # Retain real generations if an uncached judge request fails later.
            generation_path = output.with_suffix('.predictions.json')
            generation_path.write_text(json.dumps(
                {'cell': identifier, 'config': cell, 'results': results,
                 'validation': {'complete': False, 'errors': 0, 'stage': 'generation_only',
                                'generation_complete': len(results) == stop - start}},
                ensure_ascii=False, indent=2) + '\n')
        metrics = benchmark.evaluate(results)
        if metrics.get('judge_failures', 0):
            raise RuntimeError('MM-Vet judge failures: metric reproduction is invalid')
        payload = {'cell': identifier, 'config': cell, 'metrics': metrics, 'results': results,
                   'validation': {'complete': limit == cell['samples'] and args.num_shards == 1, 'errors': 0,
                                  'shard': {'index': args.shard_index, 'count': args.num_shards,
                                            'start': start, 'stop': stop, 'full_sample_count': cell['samples']},
                                  'backend': args.backend, 'device': torch.cuda.get_device_name(),
                                  'strict_backend': True, 'source_sha256': implementation_hashes,
                                  'checkpoint_config_sha256': hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
                                  'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                                  'flash_attention': importlib.metadata.version('flash-attn') if cell['architecture'] == 'internvl' else None,
                                  'cuda': torch.version.cuda, 'elapsed_seconds': time.monotonic() - started,
                                  'sample_ids_sha256': hashlib.sha256(json.dumps([str(r['id']) for r in results]).encode()).hexdigest()}}
        if cell.get('model_revision'):
            payload['validation']['checkpoint_identity'] = model._twinprune_checkpoint_identity
            payload['validation']['input_size'] = cell['input_size']
            payload['validation']['tiles'] = 1
            payload['validation']['position_mode'] = cell.get('position_mode')
        if adapter_hashes:
            payload['validation']['adapter_sha256'] = adapter_hashes
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(f'{identifier}: {metrics}, complete={payload["validation"]["complete"]}', flush=True)


if __name__ == '__main__':
    main()
