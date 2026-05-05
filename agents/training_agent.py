"""
Agent — training: Triplet Loss ile L2-normalize 128-D embedding (ResNet50).
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.data_agent import (
    DATA_SPLIT_DIR,
    RANDOM_STATE,
    TRAIN_ANNOTATIONS_NAME,
    configure_logging,
)
from agents.preprocessing_agent import (
    BATCH_SIZE,
    FREEZE_BACKBONE_EPOCHS,
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    TRIPLET_EPOCHS,
    TRIPLET_MARGIN,
    LEARNING_RATE_BACKBONE_FROZEN,
    LEARNING_RATE_FINETUNE,
    build_embedding_model,
    custom_objects_for_model,
    load_image_file,
    load_image_label_pairs,
    make_optimizer,
    set_resnet_backbone_trainable,
)

logger = logging.getLogger(__name__)

_STEPS_PER_EPOCH: int = 120
_EARLY_STOPPING_PATIENCE: int = 4
_EARLY_STOPPING_MIN_DELTA: float = 1e-4


def _build_run_model_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MODELS_DIR / f"turtle_reid_model_{stamp}.keras"


class SaveBestEmbedding(keras.callbacks.Callback):
    """Eğitim ``loss`` düştükçe çıkarım modelini (embedding ağı) diske yazar."""

    def __init__(self, embedding_net: keras.Model, path: Path) -> None:
        super().__init__()
        self._net = embedding_net
        self._path = path
        self._best = float("inf")
        self.saved = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw = logs.get("loss")
        if raw is None:
            return
        loss = float(raw)
        if np.isnan(loss):
            return
        if loss < self._best:
            self._best = loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self._net.save(self._path)
            self.saved = True
            logger.info(
                "En iyi triplet loss=%.6f — model kaydedildi: %s",
                loss,
                self._path,
            )


class TripletTrainingModel(keras.Model):
    """Üçlü girişte paylaşımlı embedding ağı; triplet hinge loss."""

    def __init__(self, embedding_net: keras.Model, margin: float) -> None:
        super().__init__()
        self.embedding_net = embedding_net
        self.margin = margin

    def call(self, inputs, training: bool = False):
        a, p, n = inputs
        return (
            self.embedding_net(a, training=training),
            self.embedding_net(p, training=training),
            self.embedding_net(n, training=training),
        )

    def train_step(self, data):
        (a, p, n), _ = data
        with tf.GradientTape() as tape:
            ea, ep, en = self((a, p, n), training=True)
            d_ap = tf.reduce_sum(tf.square(ea - ep), axis=-1)
            # Batch-hard negative: bu batch'teki tüm negatiflerden en yakın olanı seç.
            # Böylece model zor negatiflerle daha iyi ayrım öğrenir.
            d_an_matrix = tf.reduce_sum(
                tf.square(tf.expand_dims(ea, axis=1) - tf.expand_dims(en, axis=0)),
                axis=-1,
            )
            d_an_hard = tf.reduce_min(d_an_matrix, axis=1)
            loss = tf.reduce_mean(tf.nn.relu(d_ap - d_an_hard + self.margin))
        trainable = self.embedding_net.trainable_variables
        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))
        return {"loss": loss}


def _require_prepared_split() -> None:
    train_dir = DATA_SPLIT_DIR / "train"
    ann = DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME
    if not train_dir.is_dir() or not ann.is_file():
        raise RuntimeError("Önce data_agent.py çalıştırılmalıdır.")


def _group_by_identity(paths: list[str], identities: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ident in enumerate(identities):
        groups[ident].append(i)
    return dict(groups)


def _make_triplet_generator(
    paths: list[str],
    groups: dict[str, list[int]],
    batch_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    path_arr = np.array(paths)
    all_ids = list(groups.keys())
    anchor_pool = [k for k, idxs in groups.items() if len(idxs) >= 2]
    if not anchor_pool:
        raise RuntimeError(
            "Triplet eğitimi için en az bir kimlikte 2+ görüntü gerekir."
        )

    def gen():
        while True:
            a_paths, p_paths, n_paths = [], [], []
            for _ in range(batch_size):
                aid = rng.choice(anchor_pool)
                idxs = groups[aid]
                i, j = rng.choice(len(idxs), 2, replace=False)
                neg_choices = [k for k in all_ids if k != aid]
                nid = rng.choice(neg_choices)
                n_j = rng.choice(groups[nid])
                a_paths.append(path_arr[idxs[i]])
                p_paths.append(path_arr[idxs[j]])
                n_paths.append(path_arr[n_j])
            yield np.array(a_paths), np.array(p_paths), np.array(n_paths)

    return gen


def _load_triplet_batch(ap, pp, np_, image_size: tuple[int, int], augment: bool):
    def one_path(pt):
        img = load_image_file(pt, image_size)
        return img

    a = tf.map_fn(one_path, ap, fn_output_signature=tf.float32)
    p = tf.map_fn(one_path, pp, fn_output_signature=tf.float32)
    n = tf.map_fn(one_path, np_, fn_output_signature=tf.float32)
    if augment:
        from agents.preprocessing_agent import augment_train_image

        a = tf.map_fn(augment_train_image, a, fn_output_signature=tf.float32)
        p = tf.map_fn(augment_train_image, p, fn_output_signature=tf.float32)
        n = tf.map_fn(augment_train_image, n, fn_output_signature=tf.float32)
    return a, p, n


def build_triplet_dataset(
    paths: list[str],
    identities: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    *,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    groups = _group_by_identity(paths, identities)
    generator = _make_triplet_generator(paths, groups, batch_size, seed)
    sig = (
        tf.TensorSpec(shape=(batch_size,), dtype=tf.string),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.string),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.string),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=sig)

    def _map(ap, pp, np_):
        return _load_triplet_batch(ap, pp, np_, image_size, augment)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class TrainingAgent:
    def run(self) -> None:
        _require_prepared_split()
        paths, idents = load_image_label_pairs(
            DATA_SPLIT_DIR / TRAIN_ANNOTATIONS_NAME,
            DATA_SPLIT_DIR / "train",
        )
        if len(paths) < 6:
            raise RuntimeError("Eğitim için yeterli görüntü yok (en az birkaç örnek gerekir).")

        embedding_net = build_embedding_model(IMAGE_SIZE)
        triplet = TripletTrainingModel(embedding_net, TRIPLET_MARGIN)

        train_ds = build_triplet_dataset(
            paths,
            idents,
            IMAGE_SIZE,
            BATCH_SIZE,
            augment=True,
            seed=RANDOM_STATE,
        )
        train_ds_y = train_ds.map(
            lambda a, p, n: ((a, p, n), tf.zeros(BATCH_SIZE)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        run_model_path = _build_run_model_path()
        save_best = SaveBestEmbedding(embedding_net, run_model_path)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=_EARLY_STOPPING_PATIENCE,
            min_delta=_EARLY_STOPPING_MIN_DELTA,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )
        set_resnet_backbone_trainable(embedding_net, False)
        triplet.compile(optimizer=make_optimizer(LEARNING_RATE_BACKBONE_FROZEN))
        logger.info("Phase 1: frozen backbone, %d epochs", FREEZE_BACKBONE_EPOCHS)
        triplet.fit(
            train_ds_y,
            epochs=FREEZE_BACKBONE_EPOCHS,
            steps_per_epoch=_STEPS_PER_EPOCH,
            verbose=1,
            callbacks=[save_best, early_stopping],
        )

        if save_best.saved:
            embedding_net = keras.models.load_model(
                run_model_path,
                safe_mode=False,
                custom_objects=custom_objects_for_model(),
            )
            triplet = TripletTrainingModel(embedding_net, TRIPLET_MARGIN)

        set_resnet_backbone_trainable(embedding_net, True)
        triplet.compile(optimizer=make_optimizer(LEARNING_RATE_FINETUNE))
        logger.info("Phase 2: fine-tune full model, up to %d epochs", TRIPLET_EPOCHS)
        triplet.fit(
            train_ds_y,
            epochs=TRIPLET_EPOCHS,
            steps_per_epoch=_STEPS_PER_EPOCH,
            verbose=1,
            callbacks=[save_best, early_stopping],
        )

        if not save_best.saved:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            embedding_net.save(run_model_path)
            logger.warning(
                "Hiç kayıt tetiklenmedi; son ağırlıklar yazıldı: %s",
                run_model_path,
            )
        else:
            logger.info(
                "Kayıtlı model en düşük eğitim triplet loss anındaki ağırlıklar: %s",
                run_model_path,
            )

        shutil.copy2(run_model_path, MODEL_SAVE_PATH)
        logger.info(
            "En güncel model pointer güncellendi: %s -> %s",
            run_model_path.name,
            MODEL_SAVE_PATH,
        )


def run_cli() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_logging()
    TrainingAgent().run()


if __name__ == "__main__":
    run_cli()
