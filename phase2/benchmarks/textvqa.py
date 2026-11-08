"""TextVQA benchmark evaluator.

TextVQA is an open-ended VQA benchmark focused on text recognition in images.
Uses VQA accuracy metric: prediction matches if it appears in ≥3 of 10 annotator answers.
Data format: HuggingFace parquet files.
"""

import io
import re
from pathlib import Path

from PIL import Image

from .base import BaseBenchmark


class TextVQABenchmark(BaseBenchmark):
    name = "textvqa"

    def load_data(self) -> list[dict]:
        import pandas as pd

        data_dir = self.data_path
        if data_dir.is_file():
            data_dir = data_dir.parent

        # Find validation parquet files
        val_files = sorted(data_dir.glob("validation-*.parquet"))
        if not val_files:
            val_files = sorted(data_dir.glob("*val*.parquet"))
        if not val_files:
            # Try data/ subdirectory
            val_files = sorted((data_dir / "data").glob("validation-*.parquet"))

        if not val_files:
            raise FileNotFoundError(f"No validation parquet files found in {data_dir}")

        dfs = [pd.read_parquet(f) for f in val_files]
        df = pd.concat(dfs, ignore_index=True)

        data = []
        for _, row in df.iterrows():
            item = {
                "id": row.get("question_id", ""),
                "question": row["question"],
                "answers": row.get("answers", []),
                "image_data": row.get("image"),
            }
            data.append(item)
        return data

    def format_prompt(self, item: dict) -> str:
        question = item["question"]
        return f"<image>\n{question}\nAnswer the question with a short phrase."

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
        """VQA accuracy: pred matches if ≥3 annotators agree."""
        answers = item.get("answers", [])
        if answers is None or (hasattr(answers, '__len__') and len(answers) == 0):
            return 0.0

        pred = self._normalize(prediction)
        match_count = sum(1 for a in answers if self._normalize(a) == pred)

        # Standard VQA accuracy: min(match_count / 3, 1)
        return min(match_count / 3.0, 1.0)

    def _normalize(self, text: str) -> str:
        """Normalize text for VQA comparison."""
        text = text.strip().lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text
