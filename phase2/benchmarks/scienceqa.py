"""ScienceQA benchmark evaluator.

ScienceQA is a multimodal multiple-choice benchmark spanning science topics.
Only IMAGE questions (has_image == True) are evaluated.

Data format: HuggingFace dataset (parquet files) with columns:
  question, choices (list), answer (int index), image (bytes/dict),
  hint, task, grade, subject, topic, category, skill, has_image, lecture,
  solution.

Metric: accuracy (predicted choice letter matches ground truth).
"""

import io
import re
from pathlib import Path

from PIL import Image

from .base import BaseBenchmark


CHOICE_LETTERS = "ABCDE"


class ScienceQABenchmark(BaseBenchmark):
    """ScienceQA multiple-choice benchmark (image questions only)."""

    name = "scienceqa"

    def load_data(self) -> list[dict]:
        import pandas as pd

        data_dir = self.data_path
        if data_dir.is_file():
            data_dir = data_dir.parent

        # Find test/validation parquet files
        parquet_files = []
        for pattern in [
            "test-*.parquet",
            "data/test-*.parquet",
            "validation-*.parquet",
            "data/validation-*.parquet",
            "**/*.parquet",
        ]:
            parquet_files = sorted(data_dir.glob(pattern))
            if parquet_files:
                break

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        # Filter to image-only questions
        if "has_image" in df.columns:
            df = df[df["has_image"] == True].reset_index(drop=True)
        elif "image" in df.columns:
            df = df[df["image"].notna()].reset_index(drop=True)

        print(f"  ScienceQA: {len(df)} image questions loaded")

        data = []
        for idx, row in df.iterrows():
            choices = row.get("choices", [])
            if choices is None:
                choices = []
            # Handle both list and string representations
            if isinstance(choices, str):
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except (ValueError, SyntaxError):
                    choices = []

            answer_idx = int(row.get("answer", 0))
            answer_letter = CHOICE_LETTERS[answer_idx] if answer_idx < len(CHOICE_LETTERS) else ""

            data.append({
                "id": str(row.get("index", idx)),
                "question": row.get("question", ""),
                "choices": list(choices),
                "answer": answer_letter,
                "answer_idx": answer_idx,
                "hint": row.get("hint", ""),
                "image_data": row.get("image"),
                "subject": row.get("subject", ""),
                "grade": row.get("grade", ""),
                "topic": row.get("topic", ""),
            })
        return data

    def format_prompt(self, item: dict) -> str:
        question = item["question"]
        hint = item.get("hint", "")

        options = []
        for i, choice in enumerate(item["choices"]):
            if i < len(CHOICE_LETTERS):
                options.append(f"({CHOICE_LETTERS[i]}) {choice}")
        options_str = "\n".join(options)

        prompt_parts = []
        if hint and str(hint).strip() and str(hint).strip().lower() not in ("nan", "none", ""):
            prompt_parts.append(f"Context: {hint}")
        prompt_parts.append(question)
        prompt_parts.append(options_str)
        prompt_parts.append("Answer with the option letter only.")

        return "<image>\n" + "\n".join(prompt_parts)

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
        answer = item.get("answer", "").strip().upper()
        pred = self._extract_choice(prediction, len(item.get("choices", [])))
        return 1.0 if pred == answer else 0.0

    def _extract_choice(self, text: str, num_choices: int = 5) -> str:
        """Extract choice letter from model output."""
        valid = CHOICE_LETTERS[:num_choices]
        text = text.strip()

        # Direct single letter
        if text.upper() in valid:
            return text.upper()

        # First character check
        if text and text[0].upper() in valid:
            if len(text) == 1 or text[1] in (".", ")", ":", " ", ",", "\n"):
                return text[0].upper()

        # Pattern matching
        patterns = [
            r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-E])\)?",
            r"\(([A-E])\)",
            r"^([A-E])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                if letter in valid:
                    return letter

        # Fallback: first valid letter found
        for ch in text.upper():
            if ch in valid:
                return ch
        return ""

    def evaluate(self, results: list[dict]) -> dict:
        """Accuracy overall and per subject."""
        from collections import defaultdict

        metrics = super().evaluate(results)

        # Per-subject breakdown
        subj_correct: dict[str, int] = defaultdict(int)
        subj_total: dict[str, int] = defaultdict(int)
        for r in results:
            subj = r.get("subject", "unknown")
            subj_correct[subj] += int(r["score"])
            subj_total[subj] += 1

        per_subject = {}
        for subj, total in subj_total.items():
            per_subject[subj] = {
                "accuracy": subj_correct[subj] / total if total > 0 else 0.0,
                "total": total,
            }
        metrics["per_subject"] = per_subject
        return metrics

    def run(self, engine, max_new_tokens: int = 16, save_path: str | None = None) -> dict:
        """Override to attach subject metadata to results."""
        import json
        import time
        from tqdm import tqdm

        data = self.load_data()
        if self.max_samples:
            data = data[:self.max_samples]

        results = []
        t0 = time.time()

        for item in tqdm(data, desc=f"Eval {self.name}"):
            question, response = self._generate_response(engine, item, max_new_tokens)

            sc = self.score(response, item)
            results.append({
                "id": item.get("id", len(results)),
                "question": question[:200],
                "prediction": response,
                "answer": item.get("answer", ""),
                "subject": item.get("subject", ""),
                "score": sc,
            })

        elapsed = time.time() - t0
        metrics = self.evaluate(results)
        metrics["time_seconds"] = elapsed
        metrics["samples_per_second"] = len(results) / elapsed if elapsed > 0 else 0
        metrics["errors"] = 0
        metrics["empty_predictions"] = sum(not row["prediction"].strip() for row in results)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)

        return metrics
