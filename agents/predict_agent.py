"""
Agent — predict: tek sorgu görseli için train galerisinde top-5 kimlik (cosine).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.data_agent import (
    DATA_SPLIT_DIR,
    TRAIN_ANNOTATIONS_NAME,
    configure_logging,
)
from agents.preprocessing_agent import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    custom_objects_for_model,
    load_image_file,
    load_image_label_pairs,
    make_paths_dataset,
)

logger = logging.getLogger(__name__)


def _l2_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def _l2_normalize_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def _require_model() -> None:
    if not MODEL_SAVE_PATH.is_file():
        raise RuntimeError("Önce training_agent.py çalıştırılmalıdır.")


def _require_train_split() -> None:
    train_dir = DATA_SPLIT_DIR / "train"
    ann = DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME
    if not train_dir.is_dir() or not ann.is_file():
        raise RuntimeError("Önce data_agent.py çalıştırılmalıdır.")


def _embed_one(model: keras.Model, image_path: Path) -> np.ndarray:
    pt = tf.constant(str(image_path.resolve()))
    img = load_image_file(pt, IMAGE_SIZE)
    batch = tf.expand_dims(img, 0)
    e = model.predict(batch, verbose=0)
    return np.asarray(e[0], dtype=np.float32)


def top5_identities(
    query_emb: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_ids: list[str],
    k: int = 5,
) -> list[dict[str, float]]:
    sims = gallery_embs @ query_emb
    order = np.argsort(-sims)
    out: list[dict[str, float]] = []
    seen: set[str] = set()
    for j in order:
        ident = gallery_ids[int(j)]
        if ident in seen:
            continue
        seen.add(ident)
        out.append({"identity": ident, "similarity": float(sims[j])})
        if len(out) >= k:
            break
    return out


class PredictAgent:
    def predict_image(self, image_path: Path, *, top_k: int = 5) -> list[dict[str, float]]:
        _require_model()
        _require_train_split()
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Görsel bulunamadı: {path}")

        train_paths, train_ids = load_image_label_pairs(
            DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME,
            DATA_SPLIT_DIR / "train",
        )
        if not train_paths:
            raise RuntimeError("Train galerisi boş.")

        model = keras.models.load_model(
            MODEL_SAVE_PATH,
            safe_mode=False,
            custom_objects=custom_objects_for_model(),
        )

        gal_ds = make_paths_dataset(
            train_paths,
            IMAGE_SIZE,
            shuffle=False,
            augment=False,
            batch_size=BATCH_SIZE,
        )
        gallery_embs = np.asarray(model.predict(gal_ds, verbose=0), dtype=np.float32)
        gallery_embs = _l2_normalize_rows(gallery_embs)

        q_emb = _embed_one(model, path)
        q_emb = _l2_normalize_vec(q_emb)
        return top5_identities(q_emb, gallery_embs, train_ids, k=top_k)


def run_cli() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    parser = argparse.ArgumentParser(description="Tek görsel için top-5 Re-ID")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Sorgu görselinin yolu",
    )
    args = parser.parse_args()
    configure_logging()
    results = PredictAgent().predict_image(Path(args.image))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Top-%d written to stdout", len(results))


if __name__ == "__main__":
    run_cli()
