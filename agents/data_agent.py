"""
Agent — data: COCO anotasyonları, kimlik bazlı split, ``data_split/`` kopyalama.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_ROOT: Path = Path(
    os.environ.get("SEA_TURTLE_DATASET", str(PROJECT_ROOT.parent / "dataset"))
)
ANNOTATION_FILE: Path = DATASET_ROOT / "annotations.json"

DATA_SPLIT_DIR: Path = PROJECT_ROOT / "data_split"

PREDICTIONS_JSON: Path = PROJECT_ROOT / "test_predictions.json"
ACCURACY_JSON: Path = PROJECT_ROOT / "test_predictions_accuracy.json"
RESULTS_JSON: Path = PROJECT_ROOT / "test_results.json"

TRAIN_RATIO: float = 0.8
RANDOM_STATE: int = 42

TRAIN_ANNOTATIONS_NAME: str = "train_annotations.json"
TEST_ANNOTATIONS_NAME: str = "test_annotations.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    identity: str
    relative_path: str
    file_name: str


@dataclass(frozen=True)
class CocoPayload:
    info: Mapping[str, Any]
    categories: list[Mapping[str, Any]]
    images: list[Mapping[str, Any]]
    annotations: list[Mapping[str, Any]]


class AnnotationLoader:
    def __init__(self, annotation_path: Path) -> None:
        self._annotation_path = annotation_path

    def load_raw(self) -> CocoPayload:
        if not self._annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {self._annotation_path}"
            )
        with self._annotation_path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        required = ("info", "categories", "images", "annotations")
        for key in required:
            if key not in data:
                raise KeyError(f"Missing '{key}' in COCO annotations")
        return CocoPayload(
            info=data["info"],
            categories=list(data["categories"]),
            images=list(data["images"]),
            annotations=list(data["annotations"]),
        )

    def build_image_records(self) -> list[ImageRecord]:
        payload = self.load_raw()
        image_by_id: dict[int, Mapping[str, Any]] = {
            int(img["id"]): img for img in payload.images
        }
        records: list[ImageRecord] = []
        for ann in payload.annotations:
            image_id = int(ann["image_id"])
            if image_id not in image_by_id:
                continue
            img = image_by_id[image_id]
            rel = str(img.get("path") or img.get("file_name") or "")
            if not rel:
                continue
            identity = str(ann.get("identity") or ann.get("category_id") or "")
            if not identity:
                continue
            path_posix = rel.replace("\\", "/")
            file_name = Path(path_posix).name
            records.append(
                ImageRecord(
                    image_id=image_id,
                    identity=identity,
                    relative_path=path_posix,
                    file_name=file_name,
                )
            )
        return _dedupe_by_image_id(records)


def _dedupe_by_image_id(records: Sequence[ImageRecord]) -> list[ImageRecord]:
    by_id: dict[int, ImageRecord] = {}
    for r in records:
        if r.image_id not in by_id:
            by_id[r.image_id] = r
    return list(by_id.values())


def filter_existing_files(
    records: Sequence[ImageRecord],
    dataset_root: Path,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    existing: list[ImageRecord] = []
    missing: list[ImageRecord] = []
    for r in records:
        src = dataset_root / Path(r.relative_path)
        if src.is_file():
            existing.append(r)
        else:
            missing.append(r)
    return existing, missing


def build_coco_subset(
    payload: CocoPayload,
    image_ids: set[int],
) -> dict[str, Any]:
    images = [img for img in payload.images if int(img["id"]) in image_ids]
    anns = [
        a for a in payload.annotations if int(a["image_id"]) in image_ids
    ]
    return {
        "info": dict(payload.info),
        "categories": list(payload.categories),
        "images": images,
        "annotations": anns,
    }


def warn_missing_files(missing_records: list[ImageRecord], dataset_root: Path) -> None:
    for r in missing_records:
        logger.warning(
            "Missing image skipped: %s (identity=%s, image_id=%s)",
            dataset_root / Path(r.relative_path),
            r.identity,
            r.image_id,
        )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_image_preserving_relative_path(
    source_root: Path,
    dest_root: Path,
    relative_path: str,
) -> Path:
    rel = Path(relative_path)
    src = source_root / rel
    dst = dest_root / rel
    if not src.is_file():
        raise FileNotFoundError(str(src))
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)
    return dst


class DatasetSplitter:
    def __init__(
        self,
        train_ratio: float,
        random_state: int,
    ) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        self._train_ratio = train_ratio
        self._random_state = random_state

    def split(
        self,
        records: Sequence[ImageRecord],
    ) -> tuple[list[ImageRecord], list[ImageRecord]]:
        by_identity: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in records:
            by_identity[r.identity].append(r)

        rng = np.random.default_rng(self._random_state)
        test_fraction = 1.0 - self._train_ratio

        train: list[ImageRecord] = []
        test: list[ImageRecord] = []

        for identity, group in sorted(by_identity.items(), key=lambda x: x[0]):
            if len(group) < 2:
                train.extend(group)
                warnings.warn(
                    f"Identity '{identity}' has only {len(group)} image(s); "
                    "assigned entirely to train (cannot split across train/test).",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ids = [g.image_id for g in group]
            n = len(ids)
            n_test = int(round(n * test_fraction))
            n_test = max(1, n_test)
            n_test = min(n_test, n - 1)

            order = rng.permutation(n)
            test_positions = set(order[:n_test].tolist())
            train_positions = set(order[n_test:].tolist())

            id_to_rec = {g.image_id: g for g in group}
            for pos in sorted(train_positions):
                train.append(id_to_rec[ids[pos]])
            for pos in sorted(test_positions):
                test.append(id_to_rec[ids[pos]])

        return train, test


class DatasetCopyAgent:
    def __init__(
        self,
        dataset_root: Path | None = None,
        annotation_file: Path | None = None,
        output_dir: Path | None = None,
        train_ratio: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self._dataset_root = dataset_root or DATASET_ROOT
        self._annotation_file = annotation_file or ANNOTATION_FILE
        self._output_dir = output_dir or DATA_SPLIT_DIR
        self._train_ratio = train_ratio if train_ratio is not None else TRAIN_RATIO
        self._random_state = random_state if random_state is not None else RANDOM_STATE

    def run(self) -> None:
        loader = AnnotationLoader(self._annotation_file)
        payload = loader.load_raw()
        records = loader.build_image_records()

        existing, missing = filter_existing_files(records, self._dataset_root)
        warn_missing_files(missing, self._dataset_root)

        splitter = DatasetSplitter(
            train_ratio=self._train_ratio,
            random_state=self._random_state,
        )
        train_records, test_records = splitter.split(existing)

        train_dir = self._output_dir / "train"
        test_dir = self._output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        self._copy_split(train_records, train_dir, "Copy train")
        self._copy_split(test_records, test_dir, "Copy test")

        train_ids = {r.image_id for r in train_records}
        test_ids = {r.image_id for r in test_records}

        train_doc = build_coco_subset(payload, train_ids)
        test_doc = build_coco_subset(payload, test_ids)

        train_json_path = self._output_dir / TRAIN_ANNOTATIONS_NAME
        test_json_path = self._output_dir / TEST_ANNOTATIONS_NAME

        self._write_json(train_doc, train_json_path)
        self._write_json(test_doc, test_json_path)

        logger.info(
            "Split complete: train=%d images, test=%d images (skipped missing=%d)",
            len(train_records),
            len(test_records),
            len(missing),
        )

    def _copy_split(
        self,
        records: list[ImageRecord],
        dest_root: Path,
        desc: str,
    ) -> None:
        for r in tqdm(records, desc=desc, unit="img"):
            try:
                copy_image_preserving_relative_path(
                    self._dataset_root,
                    dest_root,
                    r.relative_path,
                )
            except FileNotFoundError as exc:
                logger.warning("Copy failed (file vanished?): %s", exc)

    @staticmethod
    def _write_json(doc: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def run_cli() -> None:
    configure_logging()
    DatasetCopyAgent().run()


if __name__ == "__main__":
    run_cli()
