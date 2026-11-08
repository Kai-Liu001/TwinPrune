"""MMBench benchmark evaluator.

MMBench is a multiple-choice VLM benchmark with base64-encoded images in TSV format.
Evaluation: model outputs A/B/C/D, compared against ground truth.
"""

import csv
import io
import base64
import sys
from PIL import Image

from .base import BaseBenchmark

# MMBench TSV has large base64 image fields
csv.field_size_limit(sys.maxsize)


class MMBenchBenchmark(BaseBenchmark):
    name = "mmbench"

    def load_data(self) -> list[dict]:
        tsv_path = self.data_path
        if tsv_path.is_dir():
            # Find the main TSV file
            candidates = sorted(tsv_path.glob("*DEV_EN*real.tsv")) + sorted(tsv_path.glob("*DEV_EN*.tsv"))
            if not candidates:
                candidates = sorted(tsv_path.glob("*.tsv"))
            if not candidates:
                raise FileNotFoundError(f'No MMBench TSV in {tsv_path}')
            tsv_path = candidates[0]

        data = []
        skipped = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                answer = row.get("answer", "")
                if answer not in ("A", "B", "C", "D"):
                    continue
                # Skip rows with corrupted/missing image data
                img = row.get("image", "")
                if len(img) < 100:
                    skipped += 1
                    continue
                data.append(row)
        if skipped:
            print(f"  Note: skipped {skipped} samples with invalid images, {len(data)} valid samples")
        return data

    def format_prompt(self, item: dict) -> str:
        question = item.get("question", "")
        hint = item.get("hint", "")

        options = []
        for ch in "ABCD":
            opt = item.get(ch, "")
            if opt:
                options.append(f"({ch}) {opt}")
        options_str = "\n".join(options)

        prompt_parts = []
        if hint and hint.strip() and hint.strip().lower() != "nan":
            prompt_parts.append(f"Context: {hint}")
        prompt_parts.append(question)
        prompt_parts.append(options_str)
        prompt_parts.append("Answer with the option letter only.")

        return "<image>\n" + "\n".join(prompt_parts)

    def get_image(self, item: dict) -> Image.Image | None:
        img_data = item.get("image", "")
        if not img_data:
            return None
        # Handle JSON list format: ["base64_data"]
        if img_data.startswith("["):
            import json
            img_data = json.loads(img_data)[0]
        # Fix base64 padding
        missing_padding = len(img_data) % 4
        if missing_padding:
            img_data += "=" * (4 - missing_padding)
        img_bytes = base64.b64decode(img_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def score(self, prediction: str, item: dict) -> float:
        answer = item.get("answer", "")
        pred = self._extract_choice(prediction)
        return 1.0 if pred == answer else 0.0

    def _extract_choice(self, text: str) -> str:
        """Extract A/B/C/D from model output."""
        text = text.strip()
        # Direct single letter
        if text in ("A", "B", "C", "D"):
            return text
        # First character
        if text and text[0] in ("A", "B", "C", "D"):
            # Check if followed by punctuation or space
            if len(text) == 1 or text[1] in (".", ")", ":", " ", ",", "\n"):
                return text[0]
        # Look for patterns like "Answer: B" or "The answer is C"
        import re
        patterns = [
            r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-D])\)?",
            r"\(([A-D])\)",
            r"^([A-D])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Fallback: first A-D letter found
        for ch in text:
            if ch in "ABCD":
                return ch
        return ""
