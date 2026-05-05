"""
Agent — preprocessing: ResNet ön-işleme, embedding omurgası, görüntü yükleme.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.data_agent import PROJECT_ROOT

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_SAVE_PATH: Path = MODELS_DIR / "turtle_reid_model.keras"

IMAGE_SIZE: tuple[int, int] = (224, 224)
EMBEDDING_DIM: int = 128
BATCH_SIZE: int = 32

FREEZE_BACKBONE_EPOCHS: int = 4
TRIPLET_EPOCHS: int = 24
LEARNING_RATE_BACKBONE_FROZEN: float = 1e-3
LEARNING_RATE_FINETUNE: float = 5e-5
WEIGHT_DECAY: float = 1e-4
TRIPLET_MARGIN: float = 0.35

_PREPROCESS_NAME = "imagenet_preprocess"
_GAP_NAME = "gap"


@keras.saving.register_keras_serializable(package="sea_turtle")
class ResNetPreprocessLayer(layers.Layer):
    def call(self, inputs):
        return preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="sea_turtle")
class L2NormalizeLayer(layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


def set_resnet_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    after_preprocess = False
    for layer in model.layers:
        if layer.name == _PREPROCESS_NAME:
            after_preprocess = True
            continue
        if layer.name == _GAP_NAME:
            break
        if after_preprocess:
            layer.trainable = trainable


def build_embedding_model(
    image_size: tuple[int, int] | None = None,
    embedding_dim: int = EMBEDDING_DIM,
) -> keras.Model:
    size = image_size or IMAGE_SIZE
    inputs = keras.Input(shape=(*size, 3), name="image", dtype="float32")
    x = ResNetPreprocessLayer(name=_PREPROCESS_NAME)(inputs)
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None,
    )
    y = layers.GlobalAveragePooling2D(name=_GAP_NAME)(backbone.output)
    emb = layers.Dense(embedding_dim, name="embedding")(y)
    out = L2NormalizeLayer(name="l2_normalize")(emb)
    return keras.Model(inputs, out, name="turtle_reid_embedding")


def load_image_label_pairs(
    annotations_path: Path,
    images_root: Path,
) -> tuple[list[str], list[str]]:
    with annotations_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    image_by_id = {int(img["id"]): img for img in data["images"]}
    paths: list[str] = []
    labels: list[str] = []
    seen_images: set[int] = set()
    for ann in data["annotations"]:
        iid = int(ann["image_id"])
        if iid in seen_images or iid not in image_by_id:
            continue
        img = image_by_id[iid]
        rel = str(img.get("path") or img.get("file_name") or "")
        if not rel:
            continue
        identity = str(ann.get("identity") or "")
        if not identity:
            continue
        abs_path = str((images_root / Path(rel)).resolve())
        paths.append(abs_path)
        labels.append(identity)
        seen_images.add(iid)
    return paths, labels


def augment_train_image(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    # Hafif döndürme: farklı pozlar için embedding dayanıklılığını artırır.
    img = tf.image.rot90(img, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=0.18)
    img = tf.image.random_contrast(img, lower=0.82, upper=1.18)
    img = tf.image.random_saturation(img, lower=0.78, upper=1.22)
    img = tf.image.random_hue(img, max_delta=0.04)
    return tf.clip_by_value(img, 0.0, 255.0)


def load_image_file(path_tensor: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size)
    return tf.cast(img, tf.float32)


def make_paths_dataset(
    paths: list[str],
    image_size: tuple[int, int],
    *,
    shuffle: bool,
    augment: bool,
    batch_size: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(min(len(paths), 8192), reshuffle_each_iteration=True)

    def _map(p):
        img = load_image_file(p, image_size)
        if augment:
            img = augment_train_image(img)
        return img

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_optimizer(lr: float) -> keras.optimizers.Optimizer:
    if hasattr(keras.optimizers, "AdamW"):
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
    return keras.optimizers.Adam(learning_rate=lr)


def custom_objects_for_model() -> dict[str, type]:
    return {
        "ResNetPreprocessLayer": ResNetPreprocessLayer,
        "L2NormalizeLayer": L2NormalizeLayer,
    }
