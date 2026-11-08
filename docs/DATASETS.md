# Data preparation

Download licensed benchmark copies separately. Point `configs/data.example.yaml`
to their local paths, then run `python scripts/check_data.py --data-config PATH`.
`configs/data.sha256.json` records the exact evaluation-file fingerprints.
The inference CLI checks the frozen sample-ID order before a full run.

| Benchmark | Files and layout expected by the adapter |
|---|---|
| MMBench | English V1.1 dev TSV with base64 images; direct-image rows only |
| POPE | Test parquet files below the configured root |
| ScienceQA | `data/test-*.parquet` or `test-*.parquet`; image questions only |
| TextVQA | `validation-*.parquet`; validation answers and image bytes |
| DocVQA | Validation parquet files from the DocVQA configuration |
| MM-Vet | Test parquet with question, answer, ID, and image bytes; legacy JSON also supported |

The frozen MMBench export contains 1,292 directly decodable image rows.
It excludes image aliases and invalid-image rows; it is not the full official
MMBench evaluation. Resolving all aliases changes the evaluated subset and
does not reproduce these targets. The POPE export contains 6,000 rows.
Do not substitute another split and keep the same table label.

Public dataset formats can be obtained from the benchmark authors and the
LMMS dataset exports, including
[DocVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA),
[ScienceQA](https://huggingface.co/datasets/lmms-lab-encoder/ScienceQA), and
[POPE](https://github.com/RUCAIBox/POPE). File fingerprints, evaluated counts,
and sample-ID order determine whether a particular export matches this release.

The repository includes score hashes and a hashed MM-Vet judge cache. It
does not contain raw images, questions, answers, weights, or credentials.
