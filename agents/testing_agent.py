"""
Agent — testing: embedding + cosine Top-1 retrieval ).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.data_agent import (
    DATA_SPLIT_DIR,
    TEST_ANNOTATIONS_NAME,
    TRAIN_ANNOTATIONS_NAME,
    configure_logging,
)
from agents.preprocessing_agent import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    custom_objects_for_model,
    load_image_label_pairs,
    make_paths_dataset,
)

logger = logging.getLogger(__name__)


def _l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def _require_model() -> None:
    if not MODEL_SAVE_PATH.is_file():
        raise RuntimeError("Önce training_agent.py çalıştırılmalıdır.")


def _require_split() -> None:
    train_ok = (DATA_SPLIT_DIR / "train").is_dir() and (
        DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME
    ).is_file()
    test_ok = (DATA_SPLIT_DIR / "test").is_dir() and (
        DATA_SPLIT_DIR / TEST_ANNOTATIONS_NAME
    ).is_file()
    if not train_ok or not test_ok:
        raise RuntimeError("Önce data_agent.py çalıştırılmalıdır.")


def _predict_embeddings(model: keras.Model, paths: list[str]) -> np.ndarray:
    if not paths:
        dim = int(model.output_shape[-1])
        return np.zeros((0, dim), dtype=np.float32)
    ds = make_paths_dataset(
        paths,
        IMAGE_SIZE,
        shuffle=False,
        augment=False,
        batch_size=BATCH_SIZE,
    )
    return np.asarray(model.predict(ds, verbose=1), dtype=np.float32)


class TestingAgent:
    """
    Döndürür:
    - metrics: top1_accuracy, same_similarity, different_similarity, num_samples
    - predictions: her test görseli için retrieval satırı (result_agent uyumlu)
    """

    def run(self) -> tuple[dict[str, float], list[dict]]:
        _require_model()
        _require_split()

        train_paths, train_ids = load_image_label_pairs(
            DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME,
            DATA_SPLIT_DIR / "train",
        )
        test_paths, test_ids = load_image_label_pairs(
            DATA_SPLIT_DIR / TEST_ANNOTATIONS_NAME,
            DATA_SPLIT_DIR / "test",
        )
        if not test_paths:
            raise RuntimeError("Test görüntüsü yok.")

        model = keras.models.load_model(
            MODEL_SAVE_PATH,
            safe_mode=False,
            custom_objects=custom_objects_for_model(),
        )

        emb_train = _predict_embeddings(model, train_paths)
        emb_test = _predict_embeddings(model, test_paths)
        emb_train = _l2_normalize_rows(emb_train)
        emb_test = _l2_normalize_rows(emb_test)

        all_paths = list(train_paths) + list(test_paths)
        all_ids = list(train_ids) + list(test_ids)
        all_emb = np.vstack([emb_train, emb_test])
        n_train = len(train_paths)

        id_arr = np.array(all_ids)
        predictions: list[dict] = []
        correct_flags: list[bool] = []
        same_sims: list[float] = []
        diff_sims: list[float] = []

        for t, q_path in enumerate(test_paths):
            q_idx = n_train + t
            true_id = test_ids[t]
            q_emb = all_emb[q_idx]

            sims = all_emb @ q_emb
            sims = sims.copy()
            sims[q_idx] = -np.inf

            j_best = int(np.argmax(sims))
            pred_id = all_ids[j_best]
            best_sim = float(sims[j_best])
            is_correct = pred_id == true_id
            correct_flags.append(is_correct)

            same_mask = (id_arr == true_id) & (np.arange(len(all_paths)) != q_idx)
            diff_mask = (id_arr != true_id) & (np.arange(len(all_paths)) != q_idx)
            if same_mask.any():
                same_sims.append(float(np.max(sims[same_mask])))
            if diff_mask.any():
                diff_sims.append(float(np.max(sims[diff_mask])))

            predictions.append(
                {
                    "image_path": q_path,
                    "true_identity": true_id,
                    "predicted_identity": pred_id,
                    "similarity": best_sim,
                    "correct": is_correct,
                }
            )

        top1 = float(np.mean(correct_flags)) if correct_flags else 0.0
        same_mean = float(np.mean(same_sims)) if same_sims else float("nan")
        diff_mean = float(np.mean(diff_sims)) if diff_sims else float("nan")

        metrics = {
            "top1_accuracy": top1,
            "same_similarity": same_mean,
            "different_similarity": diff_mean,
            "num_samples": float(len(test_paths)),
        }
        logger.info(
            "Top-1 retrieval: %.4f (n=%d) same_sim=%.4f diff_sim=%.4f",
            top1,
            len(test_paths),
            same_mean,
            diff_mean,
        )
        return metrics, predictions


def run_cli() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_logging()
    metrics, _preds = TestingAgent().run()
    print(
        f"top1_accuracy={metrics['top1_accuracy']:.4f} "
        f"samples={int(metrics['num_samples'])}",
        flush=True,
    )


if __name__ == "__main__":
    run_cli()
