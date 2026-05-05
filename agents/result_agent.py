"""
Agent — result: test metrikleri ve tahmin listesini JSON dosyalarına yazar.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.data_agent import (
    PREDICTIONS_JSON,
    RESULTS_JSON,
    configure_logging,
)

logger = logging.getLogger(__name__)


def _json_float_or_null(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    if math.isnan(x):
        return None
    return x


class ResultAgent:
    """``test_results.json`` ve ``test_predictions.json`` dosyalarını yazar."""

    def save(
        self,
        metrics: dict[str, Any],
        predictions: list[dict[str, Any]],
    ) -> None:
        results_payload = {
            "top1_accuracy": float(metrics.get("top1_accuracy", 0.0)),
            "same_similarity": _json_float_or_null(metrics.get("same_similarity")),
            "different_similarity": _json_float_or_null(
                metrics.get("different_similarity")
            ),
            "num_samples": int(metrics.get("num_samples", 0)),
        }

        with RESULTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s", RESULTS_JSON)

        with PREDICTIONS_JSON.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s (%d rows)", PREDICTIONS_JSON, len(predictions))


def run_cli() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from agents.testing_agent import TestingAgent

    configure_logging()
    metrics, predictions = TestingAgent().run()
    ResultAgent().save(metrics, predictions)
    print(
        f"Saved results — top1={metrics['top1_accuracy']:.4f} "
        f"samples={int(metrics['num_samples'])}",
        flush=True,
    )


if __name__ == "__main__":
    run_cli()
