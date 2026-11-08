"""MM-Vet benchmark evaluator.

MM-Vet evaluates integrated vision-language capabilities across
recognition, OCR, knowledge, math, spatial awareness, and language
generation. Original evaluation uses GPT-4 scoring; here we use
Doubao (Volcengine Ark) as a drop-in LLM judge.

Data format supported:
  - HuggingFace parquet (lmms-lab/MMVet): columns = question_id, image(bytes),
    question, answer, image_source, capability
  - Legacy mmvet.json format with images/ subdir

Ground-truth answer syntax (from MM-Vet paper):
  - "A<OR>B<OR>C": any of A/B/C accepted (partial credit for containment)
  - "A<AND>B": both A and B required (product of sub-scores)

Scoring: Doubao LLM judge returns a soft score in [0, 1].
Judge responses are cached (question_id, prediction) -> score. The bundled
reference cache contains hashes and scores; new calls require ARK_API_KEY.
"""

import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

from .base import BaseBenchmark


# Default cache path (repo-relative)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = Path(os.environ.get("MMVET_JUDGE_CACHE", "outputs/mmvet_judge_cache.json"))


# LLM judge prompt (adapted from MM-Vet official GPT-4 template)
_JUDGE_PROMPT = """You are an expert evaluator assessing a vision-language model's answer.

Question: {question}
Ground truth: {gt}
Model answer: {pred}

Rules for ground truth:
- "A<OR>B" means any of A, B (or an equivalent phrasing) is acceptable.
- "A<AND>B" means the answer must cover ALL of A and B.
- Numbers and short facts should match exactly or via obvious equivalence (e.g., "5" == "five", "0.75" == "3/4").
- For open-ended answers, partial correctness is allowed.

Output a single soft score in [0.0, 1.0]:
- 1.0 = fully correct
- 0.5 = partially correct
- 0.0 = wrong or irrelevant

Respond with ONLY the number (e.g., "0.8"), nothing else."""


class _DoubaoJudge:
    """Thin wrapper around Doubao (Volcengine Ark) chat completion."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError("ARK_API_KEY is required for uncached MM-Vet judge calls")
        from openai import OpenAI
        import httpx
        self.model = model or os.environ.get("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215")
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )

    def score(self, question: str, gt: str, pred: str, max_retries: int = 3) -> float | None:
        prompt = _JUDGE_PROMPT.format(question=question, gt=gt, pred=pred)
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=16,
                )
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"[01](?:\.\d+)?", text)
                if m:
                    return max(0.0, min(1.0, float(m.group(0))))
                return None
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [judge] failed after {max_retries} retries: {type(e).__name__}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None


class MMVetBenchmark(BaseBenchmark):
    """MM-Vet open-ended VLM benchmark with Doubao LLM judge."""

    name = "mmvet"

    def __init__(self, data_path: str, max_samples: int | None = None,
                 cache_path: str | None = None, use_llm_judge: bool = True):
        super().__init__(data_path, max_samples)
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self.use_llm_judge = use_llm_judge
        self._judge: _DoubaoJudge | None = None
        self._cache: dict | None = None
        self._reference_judgments: dict = {}

    def pin_reference_judgments(self, cell: dict) -> None:
        """Reuse frozen judgments only for an identical ID and prediction hash."""
        from twinprune.resources import archive_path
        path = archive_path(cell)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != cell['archive_sha256']:
            raise RuntimeError('MM-Vet reference-score archive is corrupt')
        archive = json.loads(gzip.decompress(data))
        expected_model = archive.get('judge_model', 'doubao-seed-2-0-lite-260215')
        if os.environ.get('DOUBAO_MODEL', expected_model) != expected_model:
            raise RuntimeError('Changing the MM-Vet judge model changes the frozen evaluation protocol')
        self._reference_judgments = archive.get('judge_cache', {})

    # ── cache ───────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    def _save_cache(self):
        if self._cache is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge under a process lock: atomic replacement alone loses judgments
        # when two workers start from the same old cache snapshot.
        lock = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            existing = {}
            if self.cache_path.exists():
                existing = json.loads(self.cache_path.read_text())
            merged = {**self._cache, **existing}  # First persisted score wins.
            tmp = self.cache_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(merged, ensure_ascii=False))
            tmp.replace(self.cache_path)
            self._cache.clear()
            self._cache.update(merged)

    @staticmethod
    def _cache_key(qid: str, prediction: str) -> str:
        h = hashlib.sha1(f"{qid}||{prediction}".encode("utf-8")).hexdigest()
        return h

    # ── data loading ────────────────────────────────────────────────────────

    def load_data(self) -> list[dict]:
        data_dir = self.data_path
        if data_dir.is_file():
            data_dir = data_dir.parent

        # Prefer parquet (HuggingFace lmms-lab/MMVet format)
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if parquet_files:
            return self._load_from_parquet(parquet_files)

        # Fallback: legacy JSON format
        json_candidates = [
            data_dir / "mmvet.json",
            data_dir / "mm-vet.json",
            data_dir / "MMVet.json",
        ]
        json_path = None
        for candidate in json_candidates:
            if candidate.exists():
                json_path = candidate
                break
        if json_path is None:
            json_files = sorted(data_dir.glob("*.json"))
            if json_files:
                json_path = json_files[0]
        if json_path is not None:
            return self._load_from_json(json_path, data_dir)

        raise FileNotFoundError(f"No MM-Vet parquet or JSON data found in {data_dir}")

    def _load_from_parquet(self, parquet_files: list[Path]) -> list[dict]:
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        data = []
        for idx, row in df.iterrows():
            img_field = row.get("image")
            img_bytes = None
            if isinstance(img_field, dict) and "bytes" in img_field:
                img_bytes = img_field["bytes"]
            qid = row.get("question_id", row.get("id", idx))
            data.append({
                "id": str(qid),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "image_data": img_bytes,
                "image_path": None,
                "capability": str(row.get("capability", "")),
                "source": "parquet",
            })
        print(f"  MM-Vet: {len(data)} questions loaded from parquet")
        return data

    def _load_from_json(self, json_path: Path, data_dir: Path) -> list[dict]:
        with open(json_path) as f:
            raw = json.load(f)
        if len(raw) <= 3 and all(isinstance(v, dict) and len(v) > 10 for v in raw.values()):
            raw = next(iter(raw.values()))

        images_dir = None
        for candidate in [data_dir / "images", data_dir / "image", data_dir]:
            if candidate.is_dir():
                has_images = any(candidate.glob("*.jpg")) or any(candidate.glob("*.png"))
                if has_images or candidate != data_dir:
                    images_dir = candidate
                    break

        data = []
        for qid, info in raw.items():
            if not isinstance(info, dict):
                continue
            imagename = info.get("imagename", info.get("image", ""))
            image_path = None
            if images_dir and imagename:
                image_path = images_dir / imagename
                if not image_path.exists():
                    image_path = None
                    for ext in ["", ".jpg", ".png", ".jpeg"]:
                        c = images_dir / f"{Path(imagename).stem}{ext}"
                        if c.exists():
                            image_path = c
                            break
            data.append({
                "id": str(qid),
                "question": info.get("question", ""),
                "answer": info.get("answer", ""),
                "image_path": str(image_path) if image_path else None,
                "image_data": None,
                "capability": info.get("capability", []),
                "source": "json",
            })
        print(f"  MM-Vet: {len(data)} questions loaded from {json_path.name}")
        return data

    # ── prompt / image ──────────────────────────────────────────────────────

    def format_prompt(self, item: dict) -> str:
        return f"<image>\n{item['question']}"

    def get_image(self, item: dict) -> Image.Image | None:
        image_path = item.get("image_path")
        if image_path and Path(image_path).exists():
            return Image.open(image_path).convert("RGB")
        img_data = item.get("image_data")
        if img_data is None:
            return None
        if isinstance(img_data, dict) and "bytes" in img_data:
            return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        if isinstance(img_data, Image.Image):
            return img_data.convert("RGB")
        return None

    # ── scoring ─────────────────────────────────────────────────────────────

    def _simple_score(self, prediction: str, item: dict) -> float:
        """Fallback exact-match (used if use_llm_judge=False)."""
        answer = item.get("answer", "").strip().lower()
        pred = prediction.strip().lower()
        if not answer:
            return 0.0
        # Handle <OR> / <AND>
        if "<OR>" in answer or "<or>" in answer:
            options = [a.strip() for a in re.split(r"<or>", answer, flags=re.IGNORECASE)]
            for opt in options:
                if opt and (opt == pred or opt in pred):
                    return 1.0
            return 0.0
        if "<AND>" in answer or "<and>" in answer:
            parts = [a.strip() for a in re.split(r"<and>", answer, flags=re.IGNORECASE)]
            hits = sum(1 for p in parts if p and p in pred)
            return hits / len(parts) if parts else 0.0
        if pred == answer or answer in pred:
            return 1.0
        return 0.0

    def score(self, prediction: str, item: dict) -> float:
        """Per-sample score (inline, not used when run() overrides).

        Falls back to exact-match for cheap incremental scoring during generation;
        the LLM judge pass runs after generation in evaluate().
        """
        # Always return simple score during the generation loop; final metric
        # is computed in evaluate() with the LLM judge.
        return self._simple_score(prediction, item)

    def _llm_judge_score(self, qid: str, question: str, gt: str, pred: str) -> float | None:
        """Call Doubao judge with cache; returns None on failure."""
        cache = self._load_cache()
        key = self._cache_key(qid, pred)
        if key in self._reference_judgments:
            return self._reference_judgments[key]
        if key in cache:
            return cache[key]

        if self._judge is None:
            self._judge = _DoubaoJudge()

        score = self._judge.score(question, gt, pred)
        if score is not None:
            cache[key] = score
            # Periodically flush (cheap, small JSON)
            self._save_cache()
        return score

    def evaluate(self, results: list[dict]) -> dict:
        """Aggregate with LLM judge over all generated predictions."""
        if not self.use_llm_judge:
            return super().evaluate(results)

        print(f"  MM-Vet: running Doubao LLM judge on {len(results)} predictions...")
        judge_scores = []
        n_fail = 0
        for r in results:
            qid = str(r.get("id", ""))
            pred = r.get("prediction", "") or ""
            gt = r.get("answer", "") or ""
            question = r.get("question", "") or ""
            # Strip the leading "<image>\n" prefix from question for judge
            question_clean = question.replace("<image>\n", "").strip()

            score = self._llm_judge_score(qid, question_clean, gt, pred)
            if score is None:
                n_fail += 1
                raise RuntimeError(f"MM-Vet judge failed for sample {qid}; refusing substitute scoring")
            r["score"] = score  # overwrite simple-score with judge score
            r["judge_score"] = score
            judge_scores.append(score)

        # Final flush
        self._save_cache()

        total = len(judge_scores)
        mean = sum(judge_scores) / total if total else 0.0
        metrics = {
            "accuracy": mean,          # MM-Vet convention: mean soft score
            "mmvet_score": mean,
            "correct": sum(judge_scores),
            "total": total,
            "judge_failures": n_fail,
            "judge_model": (self._judge.model if self._judge else "doubao-seed-2-0-lite-260215"),
        }
        return metrics
