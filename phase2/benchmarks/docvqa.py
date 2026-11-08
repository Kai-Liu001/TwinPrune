"""DocVQA benchmark evaluator.

DocVQA is a document understanding VQA benchmark.
Questions are about document images (scans, forms, tables, etc.).
Metric: ANLS (Average Normalized Levenshtein Similarity).
Data format: HuggingFace parquet files.
"""

import io
import re
from pathlib import Path

from PIL import Image

from .base import BaseBenchmark


def _anls_score(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Compute ANLS score between prediction and a list of reference answers.

    ANLS = 1 - NLS if NLS < threshold else 0, then max over answers.
    NLS = edit_distance(pred, ref) / max(len(pred), len(ref)).
    """
    if not answers:
        return 0.0

    pred = _normalize_text(prediction)

    best = 0.0
    for ans in answers:
        ref = _normalize_text(ans)
        if not pred and not ref:
            score = 1.0
        elif not pred or not ref:
            score = 0.0
        else:
            nls = _normalized_levenshtein(pred, ref)
            score = 0.0 if nls >= threshold else (1.0 - nls)
        best = max(best, score)

    return best


def _normalize_text(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower()


def _normalized_levenshtein(s1: str, s2: str) -> float:
    """Levenshtein distance normalized by max length."""
    if s1 == s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return _levenshtein(s1, s2) / max_len


def _levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    # Use single row DP to save memory
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


class DocVQABenchmark(BaseBenchmark):
    """DocVQA benchmark using ANLS metric on validation split."""

    name = "docvqa"

    def load_data(self) -> list[dict]:
        import pandas as pd

        data_dir = self.data_path
        if data_dir.is_file():
            data_dir = data_dir.parent

        # Prefer validation split (has ground-truth answers); fall back to test
        val_files = sorted(data_dir.glob("validation-*.parquet"))
        if not val_files:
            val_files = sorted(data_dir.glob("*val*.parquet"))
        if not val_files:
            val_files = sorted(data_dir.glob("test-*.parquet"))
        if not val_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        dfs = [pd.read_parquet(f) for f in val_files]
        df = pd.concat(dfs, ignore_index=True)

        data = []
        for _, row in df.iterrows():
            answers = row.get("answers")
            if answers is None:
                answers = []
            elif hasattr(answers, "tolist"):
                answers = answers.tolist()
            else:
                answers = list(answers)

            data.append({
                "id": str(row.get("questionId", "")),
                "question": row["question"],
                "answers": answers,
                "image_data": row.get("image"),
            })
        return data

    def format_prompt(self, item: dict) -> str:
        question = item["question"]
        return f"<image>\n{question}\nAnswer the question with a short phrase based on the document."

    def get_image(self, item: dict) -> Image.Image | None:
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

    def score(self, prediction: str, item: dict) -> float:
        """ANLS score for a single prediction."""
        answers = item.get("answers", [])
        if not answers:
            return 0.0
        return _anls_score(prediction, answers)

    def evaluate(self, results: list[dict]) -> dict:
        """Aggregate ANLS scores."""
        scores = [r["score"] for r in results]
        anls = sum(scores) / len(scores) if scores else 0.0
        return {
            "anls": anls,
            "total": len(scores),
        }
