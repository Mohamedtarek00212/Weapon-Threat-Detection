from __future__ import annotations

import csv
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .artifacts import write_json
from .engineering import MERGED_CLASS_NAMES, _write_data_yaml


def load_augmentation_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["augmentation"]


def _build_pipeline(config: dict[str, Any]) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=config["horizontal_flip"]["probability"]),
            A.RandomBrightnessContrast(
                brightness_limit=config["brightness_contrast"]["brightness_limit"],
                contrast_limit=config["brightness_contrast"]["contrast_limit"],
                p=config["brightness_contrast"]["probability"],
            ),
            A.RandomGamma(gamma_limit=tuple(config["gamma"]["gamma_limit"]), p=config["gamma"]["probability"]),
            A.MotionBlur(blur_limit=tuple(config["motion_blur"]["kernel_size"]), p=config["motion_blur"]["probability"]),
            A.GaussNoise(var_limit=tuple(config["gaussian_noise"]["variance_limit"]), p=config["gaussian_noise"]["probability"]),
            A.RandomShadow(shadow_roi=tuple(config["random_shadow"]["shadow_roi"]), p=config["random_shadow"]["probability"]),
            A.RandomFog(
                fog_coef_lower=config["random_fog"]["fog_coef_lower"],
                fog_coef_upper=config["random_fog"]["fog_coef_upper"],
                alpha_coef=config["random_fog"]["alpha_coef"],
                p=config["random_fog"]["probability"],
            ),
            A.Affine(
                rotate=(-config["affine"]["rotation_degrees"], config["affine"]["rotation_degrees"]),
                scale=(1 - config["affine"]["scale_limit"], 1 + config["affine"]["scale_limit"]),
                translate_percent=(-config["affine"]["translation_limit"], config["affine"]["translation_limit"]),
                p=config["affine"]["probability"],
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=config["min_bbox_visibility"]),
    )


def _copy_source_dataset(source: Path, target: Path, splits: Iterable[str]) -> None:
    for split in splits:
        for kind in ("images", "labels"):
            source_directory = source / split / kind
            target_directory = target / split / kind
            target_directory.mkdir(parents=True, exist_ok=True)
            for item in sorted(source_directory.iterdir()):
                destination = target_directory / item.name
                if kind == "images":
                    destination.symlink_to(item.resolve())
                else:
                    shutil.copy2(item, destination)


def _person_only_candidates(source: Path, split: str, person_id: int) -> list[tuple[Path, Path, list[list[float]]]]:
    candidates = []
    for label in sorted((source / split / "labels").glob("*.txt")):
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and {int(row[0]) for row in rows} == {person_id}:
            image = next((path for path in (source / split / "images").glob(f"{label.stem}.*") if path.is_file()), None)
            if image is None:
                raise FileNotFoundError(f"No image for candidate label: {label}")
            boxes = [[float(value) for value in row[1:]] for row in rows]
            candidates.append((image, label, boxes))
    return candidates


def _select_candidates(candidates: list[tuple[Path, Path, list[list[float]]]], required_annotations: int, seed: int) -> list[tuple[Path, Path, list[list[float]]]]:
    randomized = list(candidates)
    random.Random(seed).shuffle(randomized)
    parent: dict[int, tuple[int, int] | None] = {0: None}
    for index, candidate in enumerate(randomized):
        count = len(candidate[2])
        for total in sorted(parent, reverse=True):
            next_total = total + count
            if next_total <= required_annotations and next_total not in parent:
                parent[next_total] = (total, index)
        if required_annotations in parent:
            break
    if required_annotations not in parent:
        raise ValueError("Unable to select enough Person-only annotations for the requested balance target")
    selection = []
    total = required_annotations
    while total:
        previous, index = parent[total]
        selection.append(randomized[index])
        total = previous
    return list(reversed(selection))


def _bounded_box(box: list[float]) -> list[float]:
    x, y, width, height = box
    margin = 1e-6
    left, right = max(margin, x - width / 2), min(1 - margin, x + width / 2)
    top, bottom = max(margin, y - height / 2), min(1 - margin, y + height / 2)
    if right <= left or bottom <= top:
        raise ValueError("Transformed box has no visible area after bounds clipping")
    return [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]


def _write_label(path: Path, boxes: list[list[float]], person_id: int) -> None:
    bounded_boxes = [_bounded_box(box) for box in boxes]
    path.write_text("\n".join(f"{person_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for x, y, width, height in bounded_boxes) + "\n", encoding="utf-8")


def repair_generated_label_bounds(dataset_root: str | Path) -> dict[str, int]:
    labels = Path(dataset_root) / "train" / "labels"
    changed_files = changed_annotations = 0
    for label in labels.glob("*__person_aug_*.txt"):
        repaired = []
        changed = False
        for line in label.read_text(encoding="utf-8").splitlines():
            class_id, *values = line.split()
            original = [float(value) for value in values]
            bounded = _bounded_box(original)
            changed |= any(abs(before - after) > 1e-12 for before, after in zip(original, bounded))
            repaired.append((class_id, bounded))
        if changed:
            label.write_text("\n".join(f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}" for class_id, (x, y, width, height) in repaired) + "\n", encoding="utf-8")
            changed_files += 1
            changed_annotations += len(repaired)
    return {"changed_files": changed_files, "changed_annotations": changed_annotations}


def reconstruct_augmentation_records(source_root: str | Path, target_root: str | Path) -> list[dict[str, str]]:
    source, target = Path(source_root), Path(target_root)
    records = []
    for label in sorted((target / "train" / "labels").glob("*__person_aug_*.txt")):
        augmented_stem = label.stem
        source_stem = augmented_stem.split("__person_aug_", maxsplit=1)[0]
        image = next((path for path in (target / "train" / "images").glob(f"{augmented_stem}.*") if path.is_file()), None)
        source_image = next((path for path in (source / "train" / "images").glob(f"{source_stem}.*") if path.is_file()), None)
        source_label = source / "train" / "labels" / f"{source_stem}.txt"
        if image is None or source_image is None or not source_label.is_file():
            raise FileNotFoundError(f"Could not reconstruct augmentation record for: {label}")
        records.append({
            "source_image": str(source_image),
            "source_label": str(source_label),
            "augmented_image": str(image),
            "augmented_label": str(label),
            "person_annotations": str(len([line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()])),
            "attempt": "recovered",
        })
    return records


def build_final_dataset(source_root: str | Path, target_root: str | Path, config_path: str | Path, splits: Iterable[str]) -> dict[str, Any]:
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Processed source dataset is missing: {source}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Final target already exists and is not empty: {target}")
    config = load_augmentation_config(config_path)
    person_id = config["person_class_id"]
    source_counts = Counter()
    for split in splits:
        for label in (source / split / "labels").glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    source_counts[int(line.split()[0])] += 1
    non_person_counts = [count for class_id, count in source_counts.items() if class_id != person_id]
    target_person_count = math.ceil(min(non_person_counts) * config["target_person_to_smallest_non_person_ratio"])
    required_annotations = max(0, target_person_count - source_counts[person_id])
    candidates = _person_only_candidates(source, config["source_split"], person_id)
    selected = _select_candidates(candidates, required_annotations, config["seed"])
    _copy_source_dataset(source, target, splits)
    pipeline = _build_pipeline(config)
    generated_records = []
    for index, (image_path, label_path, boxes) in enumerate(selected, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable augmentation source image: {image_path}")
        augmented = None
        for attempt in range(10):
            random.seed(config["seed"] + index * 100 + attempt)
            np.random.seed(config["seed"] + index * 100 + attempt)
            result = pipeline(image=image, bboxes=boxes, class_labels=[person_id] * len(boxes))
            if len(result["bboxes"]) == len(boxes):
                augmented = result
                break
        if augmented is None:
            raise RuntimeError(f"Could not preserve all Person annotations for: {image_path}")
        name = f"{image_path.stem}__person_aug_{index:04d}{image_path.suffix.lower()}"
        output_image = target / config["source_split"] / "images" / name
        output_label = target / config["source_split"] / "labels" / f"{Path(name).stem}.txt"
        if not cv2.imwrite(str(output_image), augmented["image"]):
            raise OSError(f"Could not save augmented image: {output_image}")
        _write_label(output_label, [list(box) for box in augmented["bboxes"]], person_id)
        generated_records.append({
            "source_image": str(image_path),
            "source_label": str(label_path),
            "augmented_image": str(output_image),
            "augmented_label": str(output_label),
            "person_annotations": str(len(boxes)),
            "attempt": str(attempt + 1),
        })
    data_yaml = _write_data_yaml(target)
    metadata = {
        "source_root": str(source),
        "target_root": str(target),
        "source_modified": False,
        "class_names": MERGED_CLASS_NAMES,
        "person_only_source_images": True,
        "generated_images": len(generated_records),
        "generated_person_annotations": sum(int(record["person_annotations"]) for record in generated_records),
        "balance_target_person_annotations": target_person_count,
        "data_yaml": str(data_yaml),
        "augmentation_config": str(Path(config_path).resolve()),
    }
    write_json(target / "metadata.json", metadata)
    return {"metadata": metadata, "generated_records": generated_records}


def save_augmentation_charts(before: dict[str, Any], after: dict[str, Any], output_directory: str | Path) -> list[str]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(after["annotations_per_class"])
    before_values = [before["annotations_per_class"][name] for name in names]
    after_values = [after["annotations_per_class"][name] for name in names]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - 0.2, before_values, 0.4, label="Before", color="#8da0cb")
    axis.bar(positions + 0.2, after_values, 0.4, label="After", color="#1f4e79")
    axis.set_xticks(positions, names, rotation=20)
    axis.set_ylabel("Annotations")
    axis.set_title("Class Distribution Before and After Person Augmentation")
    axis.legend()
    figure.tight_layout()
    comparison = directory / "final_dataset_before_after_distribution.png"
    figure.savefig(comparison, dpi=300, bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, after_values, color="#1f4e79")
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Annotations")
    axis.set_title("Final Dataset Class Distribution")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    distribution = directory / "final_dataset_class_distribution.png"
    figure.savefig(distribution, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [str(comparison), str(distribution)]


def write_augmentation_report(records: list[dict[str, str]], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_image", "source_label", "augmented_image", "augmented_label", "person_annotations", "attempt"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return target
