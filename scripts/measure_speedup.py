"""Measure the four TwinPrune speedup rows from main Table 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--row', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tiles', type=int, default=12)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=4)
    parser.add_argument('--backend', choices=['legacy', 'graph', 'triton'], default='triton')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing an existing timing output')
    if min(args.batch_size, args.tiles, args.repeats, args.max_new_tokens) < 1 or args.warmups < 0:
        parser.error('Invalid timing counts')
    os.environ['TWINPRUNE_FPS_BACKEND'] = args.backend
    os.environ['CAC_FPS_BACKEND'] = args.backend
    os.environ['TWINPRUNE_STRICT_BACKEND'] = '1'
    import numpy as np
    import torch
    from PIL import Image
    from src.latc import LATCConfig
    from src.latc_batched import LATCBatchedInferenceEngine
    from src.model_utils import prepare_pixel_values
    from twinprune.evaluate import adapter_fingerprints, load_model, source_hashes
    from twinprune.resources import DEFAULT_MANIFEST, RESOURCE_ROOT
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    timing = json.loads((RESOURCE_ROOT / 'timing.json').read_text())
    rows = {row['id']: row for row in timing['rows']}
    if args.row not in rows:
        parser.error(f'Unknown TwinPrune timing row: {args.row}')
    row = rows[args.row]
    reference = next(c for c in manifest['cells'] if c.get('paper_row') == args.row)
    if bool(row['adapter_required']) != bool(args.adapter):
        parser.error('LoRA rows require their own --adapter; other rows must use the base model')
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model, tokenizer = load_model(reference, args.model, 'cuda:0')
    adapter_hashes = {}
    if args.adapter:
        adapter_hashes = adapter_fingerprints(args.adapter, [reference])
        from peft import PeftModel
        model.language_model.model = PeftModel.from_pretrained(
            model.language_model.model, str(args.adapter)).merge_and_unload()
    model.eval()
    image = Image.fromarray(np.random.RandomState(42).randint(0, 255, (448, 448, 3), dtype=np.uint8))
    pixels = prepare_pixel_values(image, model, 'cuda:0').repeat(args.tiles, 1, 1, 1)
    question = '<image>\nDescribe this image briefly.'
    stages = [] if row['method'] == 'baseline' else [(row['layer'], row['prune_ratio'])]
    control = LATCBatchedInferenceEngine(model, tokenizer, LATCConfig(stages=[], seed=42))
    trial = LATCBatchedInferenceEngine(model, tokenizer, LATCConfig(
        stages=stages, strategy=row['method'] if stages else 'merge', seed=42))

    def measure(engine, trace: dict | None = None) -> list[float]:
        repetitions = []
        for index in range(args.warmups + args.repeats):
            handles = []
            if trace is not None and index == 0:
                def capture(layer_index):
                    def hook(module, inputs):
                        trace.setdefault(layer_index, int(inputs[0].shape[1]))
                    return hook
                handles = [layer.register_forward_pre_hook(capture(i))
                           for i, layer in enumerate(model.language_model.model.layers)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                outputs = engine.generate_batched(pixels, question, batch=args.batch_size,
                                                 max_new_tokens=args.max_new_tokens)
            finally:
                for handle in handles:
                    handle.remove()
            torch.cuda.synchronize()
            if len(outputs) != args.batch_size:
                raise RuntimeError('Batched output count differs')
            if index >= args.warmups:
                repetitions.append((time.perf_counter()-started)*1000/args.batch_size)
        return repetitions

    control_trace, trial_trace = {}, {}
    before = measure(control, control_trace)
    trial_times = measure(trial, trial_trace)
    after = measure(control)
    control_ms = (statistics.median(before)+statistics.median(after))/2
    trial_ms = statistics.median(trial_times)
    n_visual = int(model.num_image_token*pixels.shape[0])
    n_text = control_trace[0]-n_visual
    visual_tpl = 100*sum(n-n_text for n in trial_trace.values())/(len(control_trace)*n_visual)
    complete = (args.batch_size == 8 and args.tiles == 12 and
                args.max_new_tokens == 4 and args.warmups >= 5 and args.repeats >= 20)
    report = {'status': 'MEASURED' if complete else 'INCOMPLETE_INTEGRATION_CHECK',
              'row': row['id'], 'published_speedup': row['speedup'],
              'speedup': control_ms/trial_ms, 'control_per_sample_ms': control_ms,
              'trial_per_sample_ms': trial_ms, 'raw_per_sample_ms': {
                  'full_before': before, 'trial': trial_times, 'full_after': after},
              'token_layer_budget': {'published_visual_tpl_percent': row['tpl_percent'],
                                     'measured_visual_tpl_percent': visual_tpl,
                                     'control_prefill_layer_lengths': control_trace,
                                     'trial_prefill_layer_lengths': trial_trace,
                                     'text_tokens': n_text,
                                     'rule': 'Sum visual tokens entering each decoder layer / full visual token-layer count'},
              'protocol': {'batch_size': args.batch_size, 'tiles': args.tiles,
                           'visual_tokens': n_visual,
                           'warmups': args.warmups, 'repeats': args.repeats,
                           'max_new_tokens': args.max_new_tokens, 'backend': args.backend,
                           'shared_image_and_prompt': True,
                           'boundary': 'generate_batched: GPU vision/projector + language prefill + up to 4 generated tokens; preprocessing excluded'},
              'runtime': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                          'transformers': __import__('transformers').__version__,
                          'timing_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'source_sha256': source_hashes(), 'adapter_sha256': adapter_hashes,
                          'checkpoint_identity': model._twinprune_checkpoint_identity},
              'pure_prefill_speedup_verified': False,
              'reference_a6000': torch.cuda.get_device_name() == 'NVIDIA RTX A6000'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'], 'speedup', report['speedup'], flush=True)


if __name__ == '__main__':
    main()
