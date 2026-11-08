"""POPE benchmark evaluator.

POPE (Polling-based Object Probing Evaluation) tests object hallucination.
Questions are "Is there a [object] in the image?" → Yes/No answers.
Metrics: accuracy, precision, recall, F1.
Data format: HuggingFace parquet with embedded images.
"""

import io
from pathlib import Path

from PIL import Image

from .base import BaseBenchmark


class POPEBenchmark(BaseBenchmark):
    name = "pope"

    def load_data(self) -> list[dict]:
        import pandas as pd

        data_dir = self.data_path
        if data_dir.is_file():
            data_dir = data_dir.parent

        # Find parquet files
        parquet_files = sorted(data_dir.glob("**/test-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(data_dir.glob("**/*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        data = []
        for _, row in df.iterrows():
            data.append({
                "id": row.get("id", ""),
                "question": row["question"],
                "answer": row["answer"],
                "image_data": row.get("image"),
                "category": row.get("category", ""),
            })
        return data

    def format_prompt(self, item: dict) -> str:
        question = item["question"]
        return f"<image>\n{question}\nAnswer with Yes or No only."

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
        answer = item.get("answer", "").strip().lower()
        pred = self._extract_yesno(prediction)
        return 1.0 if pred == answer else 0.0

    def _extract_yesno(self, text: str) -> str:
        text = text.strip().lower()
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        if "yes" in text and "no" not in text:
            return "yes"
        if "no" in text and "yes" not in text:
            return "no"
        return text.split()[0] if text else ""

    def evaluate(self, results: list[dict]) -> dict:
        metrics = super().evaluate(results)

        tp = fp = tn = fn = 0
        for r in results:
            pred = self._extract_yesno(r["prediction"])
            true = r["answer"].strip().lower()
            if pred == "yes" and true == "yes":
                tp += 1
            elif pred == "yes" and true == "no":
                fp += 1
            elif pred == "no" and true == "no":
                tn += 1
            elif pred == "no" and true == "yes":
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": (tp + fp) / len(results) if results else 0,
        })
        return metrics
