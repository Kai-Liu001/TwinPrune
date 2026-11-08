# Reproduce the TwinPrune rows in the main comparison tables

This checkout contains only **TwinPrune's Table 2 and Table 3 experiments**:
16 prefill scores (including Block 3b LoRA recovery), nine decode scores,
and the four Table 2 speedups. The uncompressed path is retained as the
control needed to measure speedup. Other methods and appendix experiments
are outside this checkout's scope.

## Setup

The accuracy references use Python 3.12, RTX A6000, Torch 2.7.0+cu126,
Transformers 5.3.0, FlashAttention 2.8.3 and BF16:

```bash
conda create -n twinprune python=3.12 -y
conda activate twinprune
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install ninja psutil packaging
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
export PYTHONPATH="$PWD"
```

Every experiment uses **OpenGVLab/InternVL3_5-8B-Instruct**, revision
`27506812aa9914804018996329e895977ee2d0c8` (36 decoder layers, hidden size
4096, eight KV heads). Without `--model`, the evaluator downloads that
revision. A local `--model` directory must match all four weight SHA-256
digests in [the manifest](configs/unified_internvl35.json).

Download the six datasets following [DATASETS.md](docs/DATASETS.md).
Copy `configs/data.example.yaml` to `configs/data.local.yaml` and set your
paths. Validate the files with:

```bash
python scripts/check_data.py --data-config configs/data.local.yaml
```

The frozen counts are MMBench 1,292; POPE 6,000; ScienceQA 2,017;
TextVQA 5,000; DocVQA 5,349; MM-Vet 218. Full evaluation checks sample
order before inference. MM-Vet uses the disclosed Doubao judge
`doubao-seed-2-0-lite-260215`, rather than the official GPT-4 evaluator.
Exact ID/prediction matches reuse the successful judgments bundled with
each reference score archive; new predictions require `ARK_API_KEY`.
Judge failures invalidate a run.

## Quick check

```bash
python -m twinprune.verify
python scripts/test_unified_contracts.py
TWINPRUNE_MODEL=models/InternVL3_5-8B-Instruct \
TWINPRUNE_DATA_CONFIG=configs/data.local.yaml bash scripts/quickstart.sh
```

The verifier audits stored per-sample scores without running a model.
The quickstart runs 20 actual TextVQA examples for TwinPrune B2; it is a
smoke test and cannot pass full-metric verification.

## Table 2: prefill

```bash
python -m twinprune.evaluate \
  --group internvl35_instruct_prefill_b1 internvl35_instruct_prefill_b2 \
          internvl35_instruct_prefill_b3 \
  --model models/InternVL3_5-8B-Instruct \
  --data-config configs/data.local.yaml --backend legacy \
  --output-dir outputs/main
```

B1 uses layer 15 / 50% pruning; B2 layer 3 / 85%; B3 layer 0 / 80%.
Accuracy uses one 448-pixel tile (256 visual tokens), seed 42, beta 0.3,
legacy FPS and legacy positions. The retained visual counts are 128, 38
and 51, respectively. TextVQA uses normalized soft scoring
`min(answer matches / 3, 1)` with a 32-token generation cap.

For Block 3b, first train TwinPrune's adapter on 2,000 A-OKVQA examples.
The recipe uses one epoch, rank 16, alpha 32, learning rate 1e-4,
gradient accumulation four, q/v projections, seed 42 and first-answer-token
loss. `--parquet-path` accepts the directory containing
`a-okvqa-train-*.arrow` files:

```bash
python scripts/train_lora.py \
  --model-path models/InternVL3_5-8B-Instruct \
  --parquet-path data/aokvqa --save-dir outputs/adapter \
  --dataset aokvqa --method cac_v4 --layer 0 --ratio 0.70 \
  --n-samples 2000 --epochs 1 --lora-r 16 --lora-alpha 32 \
  --lr 1e-4 --grad-accum 4 --target-modules q_proj,v_proj --seed 42

python -m twinprune.evaluate --group internvl35_instruct_prefill_b3b \
  --model models/InternVL3_5-8B-Instruct --adapter outputs/adapter \
  --data-config configs/data.local.yaml --backend legacy \
  --output-dir outputs/main
```

The evaluator rejects incomplete training or an adapter trained on another
base checkpoint. Adapter weights are trained separately; the reviewed
adapter's fingerprints are preserved in the manifest. A fresh training run
is checked against the same recipe and base identity, and its evaluation
must still meet the frozen score tolerances.

## Table 3: decode

```bash
python -m twinprune.evaluate --group internvl35_instruct_decode_policies \
  --model models/InternVL3_5-8B-Instruct \
  --data-config configs/data.local.yaml --backend legacy \
  --output-dir outputs/main
```

This runs density eviction at visual KV keep ratios 50%, 25% and 12.5%
on TextVQA, MM-Vet and DocVQA. Generation caps are 20, 512 and 20 tokens,
respectively. DocVQA uses ANLS with a 0.5 distance cutoff. All text KV
entries and the original decode position offsets are retained.

## Timing and table export

```bash
python scripts/measure_speedup.py \
  --model models/InternVL3_5-8B-Instruct --row table2_b1_cac_v4 \
  --output outputs/timing/table2_b1_cac_v4.json
```

Repeat for `table2_b2_cac_v4`, `table2_b3_cac_v4` and
`table2_b3b_cac_v4`; the latter also requires `--adapter outputs/adapter`.
The timing protocol uses batch eight, 12 tiles, five warmups, 20 repeats
and up to four generated tokens. It measures GPU vision/projector, language
prefill and generation, excluding preprocessing. Its cached Triton FPS
backend is separate from the legacy backend used for accuracy. Speedup is
the mean of the before/after control medians divided by the trial median;
wall-clock values can vary between runs.

```bash
python -m twinprune.verify --results-dir outputs/main
python scripts/export_results.py --results-dir outputs/main \
  --timing-dir outputs/timing --output-dir outputs/tables
```

These commands verify all 25 full inference results before exporting the
TwinPrune rows to CSV, including Avg-4, actual integer-token TPL and fresh
speedups. Omit `--timing-dir` to export accuracy alone. To inspect stored
reference scores and timings without model inference, run:

```bash
python scripts/export_results.py --reference --output-dir outputs/reference
```

`python -m twinprune.verify --list` lists the 25 cell IDs. Evaluation also
supports `--cell`, `--num-shards` and `--shard-index`; use
`python -m twinprune.assemble --help` to merge full scored shards.
