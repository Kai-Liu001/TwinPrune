"""Base benchmark interface for VLM evaluation."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm


class BaseBenchmark(ABC):
    """Base class for VLM benchmark evaluation.

    Subclasses implement data loading, prompt formatting, and scoring.
    The run() method handles the evaluation loop.
    """

    name: str = "base"

    def __init__(self, data_path: str, max_samples: int | None = None):
        self.data_path = Path(data_path)
        self.max_samples = max_samples

    @abstractmethod
    def load_data(self) -> list[dict]:
        """Load benchmark data. Returns list of sample dicts."""
        ...

    @abstractmethod
    def format_prompt(self, item: dict) -> str:
        """Format benchmark item into a question string for the model."""
        ...

    @abstractmethod
    def get_image(self, item: dict):
        """Get PIL Image for the sample. Returns None if no image."""
        ...

    @abstractmethod
    def score(self, prediction: str, item: dict) -> float:
        """Score a single prediction against ground truth. Returns 0 or 1."""
        ...

    def evaluate(self, results: list[dict]) -> dict:
        """Aggregate individual scores into benchmark metrics."""
        scores = [r["score"] for r in results]
        return {
            "accuracy": sum(scores) / len(scores) if scores else 0,
            "correct": sum(scores),
            "total": len(scores),
        }

    def _generate_response(self, engine, item: dict, max_new_tokens: int) -> tuple[str, str]:
        """Generate one answer; infrastructure errors invalidate the run."""
        from src.model_utils import prepare_pixel_values

        try:
            question = self.format_prompt(item)
            image = self.get_image(item)
            if image is None:
                raise FileNotFoundError("Benchmark image is missing")
            pixels = prepare_pixel_values(image, engine.model, str(engine.device))
            response = engine.generate(pixels, question, max_new_tokens=max_new_tokens)
            return question, response
        except Exception as exc:
            sample_id = item.get("id", item.get("index", "unknown"))
            raise RuntimeError(
                f"{self.name} inference failed at sample {sample_id}; "
                "no valid benchmark metric was produced"
            ) from exc

    def run(self, engine, max_new_tokens: int = 128, save_path: str | None = None) -> dict:
        """Run full benchmark evaluation.

        Args:
            engine: LATCInferenceEngine instance
            max_new_tokens: Max tokens for generation
            save_path: Optional path to save per-sample results

        Returns:
            dict with aggregate metrics
        """
        data = self.load_data()
        if self.max_samples:
            data = data[:self.max_samples]

        results = []
        t0 = time.time()

        for item in tqdm(data, desc=f"Eval {self.name}"):
            question, response = self._generate_response(engine, item, max_new_tokens)

            sc = self.score(response, item)
            import os as _os_dbg
            if _os_dbg.environ.get("DUMP_PREDS") == "1":
                print(f"[PRED-DBG] sample={len(results)} ans=\"{item.get('answer', '')}\" score={sc} pred=\"{response[:100]}\"", flush=True)
            results.append({
                "id": item.get("id", item.get("index", len(results))),
                "question": question[:200] if 'question' in dir() else "",
                "prediction": response,
                "answer": item.get("answer", ""),
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
