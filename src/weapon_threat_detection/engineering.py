from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib.pyplot as plt
import yaml

from .artifacts import write_json

MERGED_CLASS_NAMES = ["Handheld_Weapon", "Explosive", "Fire_Smoke", "Firearm", "Person"]
CLASS_ID_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 4, 6: 0}


def _link_image(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve())


def _write_data_yaml(dataset_root: Path) -> Path:
    target = dataset_root / "data.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(MERGED_CLASS_NAMES),
                "names": MERGED_CLASS_NAMES,
            },
            handle,
            sort_keys=False,
        )
    return target


def create_merged_dataset(source_root: str | Path, target_root: str | Path, splits: Iterable[str]) -> dict[str, object]:
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Cleaned source dataset is missing: {source}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Target dataset already exists and is not empty: {target}")
    merged_annotations = Counter()
    files_by_split: dict[str, dict[str, int]] = {}
    for split in splits:
        source_images = source / split / "images"
        source_labels = source / split / "labels"
        target_images = target / split / "images"
        target_labels = target / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        images = sorted(path for path in source_images.iterdir() if path.is_file())
        labels = sorted(path for path in source_labels.glob("*.txt"))
        for image in images:
            _link_image(image, target_images / image.name)
        for label in labels:
            output_lines: list[str] = []
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, *coordinates = line.split()
                merged_id = CLASS_ID_MAP[int(class_id)]
                output_lines.append(" ".join([str(merged_id), *coordinates]))
                merged_annotations[MERGED_CLASS_NAMES[merged_id]] += 1
            (target_labels / label.name).write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
        files_by_split[split] = {"images": len(images), "labels": len(labels)}
    data_yaml = _write_data_yaml(target)
    metadata = {
        "source_root": str(source),
        "target_root": str(target),
        "source_modified": False,
        "class_mapping": {str(key): value for key, value in CLASS_ID_MAP.items()},
        "class_names": MERGED_CLASS_NAMES,
        "files_by_split": files_by_split,
        "merged_annotation_counts": dict(merged_annotations),
        "data_yaml": str(data_yaml),
    }
    write_json(target / "metadata.json", metadata)
    return metadata


def collect_dataset_statistics(dataset_root: str | Path, class_names: list[str], splits: Iterable[str]) -> dict[str, object]:
    root = Path(dataset_root)
    annotations = Counter()
    images_per_class = Counter()
    split_statistics: dict[str, dict[str, int]] = {}
    total_images = total_annotations = background_images = 0
    for split in splits:
        images = [path for path in (root / split / "images").iterdir() if path.is_file()]
        labels = root / split / "labels"
        split_annotations = split_background = 0
        for image in images:
            label = labels / f"{image.stem}.txt"
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                split_background += 1
            image_classes = {int(line.split()[0]) for line in lines}
            for class_id in image_classes:
                images_per_class[class_names[class_id]] += 1
            for line in lines:
                class_id = int(line.split()[0])
                annotations[class_names[class_id]] += 1
                split_annotations += 1
        total_images += len(images)
        total_annotations += split_annotations
        background_images += split_background
        split_statistics[split] = {
            "images": len(images),
            "annotations": split_annotations,
            "background_images": split_background,
        }
    percentages = {
        class_name: (annotations[class_name] / total_annotations * 100 if total_annotations else 0.0)
        for class_name in class_names
    }
    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "background_images": background_images,
        "annotations_per_class": {class_name: annotations[class_name] for class_name in class_names},
        "images_per_class": {class_name: images_per_class[class_name] for class_name in class_names},
        "percentage_per_class": percentages,
        "splits": split_statistics,
    }


def calculate_class_weights(statistics: dict[str, object], class_names: list[str]) -> dict[str, object]:
    counts = statistics["annotations_per_class"]
    total = statistics["total_annotations"]
    class_count = len(class_names)
    weights = {name: total / (class_count * counts[name]) for name in class_names}
    return {
        "method": "balanced_inverse_frequency",
        "formula": "weight(class) = total_annotations / (number_of_classes * class_annotations)",
        "class_weights": weights,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def analyze_person_class(dataset_root: str | Path, person_id: int, splits: Iterable[str]) -> dict[str, object]:
    root = Path(dataset_root)
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    image_stems: set[tuple[str, str]] = set()
    for split in splits:
        for label in (root / split / "labels").glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, _, _, width, height = line.split()
                if int(class_id) == person_id:
                    width_value, height_value = float(width), float(height)
                    widths.append(width_value)
                    heights.append(height_value)
                    areas.append(width_value * height_value)
                    aspect_ratios.append(width_value / height_value)
                    image_stems.add((split, label.stem))
    annotation_count = len(areas)
    average = annotation_count / len(image_stems) if image_stems else 0.0
    small_fraction = sum(area < 0.02 for area in areas) / annotation_count if annotation_count else 0.0
    recommendation = (
        "Augmentation is justified: Person annotations are comparatively scarce and include a substantial small-object component. Use the configured geometric and illumination transforms after baseline evaluation."
        if annotation_count and small_fraction >= 0.2
        else "Augmentation is not justified solely by Person object size; establish a baseline before enabling transforms."
    )
    return {
        "class_name": "Person",
        "annotation_count": annotation_count,
        "image_count": len(image_stems),
        "average_objects_per_image": average,
        "width": _summary(widths),
        "height": _summary(heights),
        "area": _summary(areas),
        "aspect_ratio": _summary(aspect_ratios),
        "small_object_fraction_area_lt_0_02": small_fraction,
        "augmentation_recommendation": recommendation,
    }


def save_publication_charts(statistics: dict[str, object], person_analysis: dict[str, object], output_directory: str | Path) -> list[str]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 11})
    counts = statistics["annotations_per_class"]
    names, values = list(counts.keys()), list(counts.values())
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, values, color="#1f4e79")
    axis.set_ylabel("Annotations")
    axis.set_title("Merged Class Distribution")
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    distribution = directory / "merged_class_distribution.png"
    figure.savefig(distribution, bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.pie(values, labels=names, autopct="%1.1f%%", startangle=90, colors=plt.cm.Blues([0.45, 0.6, 0.75, 0.9, 0.35]))
    axis.set_title("Merged Class Annotation Share")
    figure.tight_layout()
    share = directory / "merged_class_share.png"
    figure.savefig(share, bbox_inches="tight")
    plt.close(figure)
    return [str(distribution), str(share)]


def build_quality_report(before_cleaning: dict[str, Any], after_cleaning: dict[str, Any], after_merging: dict[str, Any], weights: dict[str, object], person_analysis: dict[str, object]) -> dict[str, object]:
    counts = after_merging["annotations_per_class"]
    maximum, minimum = max(counts.values()), min(counts.values())
    before_annotations = before_cleaning.get("total_annotations", before_cleaning.get("annotations", 0))
    return {
        "before_cleaning": before_cleaning,
        "after_cleaning": after_cleaning,
        "after_merging": after_merging,
        "annotation_difference_after_cleaning": after_cleaning["total_annotations"] - before_annotations,
        "merged_class_distribution": counts,
        "balance_ratio_max_to_min": maximum / minimum,
        "class_weights": weights,
        "person_analysis": person_analysis,
        "recommendations": [
            "Use processed_v1/data.yaml for all future training and evaluation.",
            "Retain the configured class weights for minority-aware experiments.",
            person_analysis["augmentation_recommendation"],
            "Do not enable augmentation until a baseline model has been evaluated.",
        ],
    }


def write_yaml(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return target


def build_augmentation_pipeline(config_path: str | Path):
    import albumentations as A

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["pipeline"]
    transforms = []
    if config["horizontal_flip"]["enabled"]:
        transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]["probability"]))
    if config["brightness"]["enabled"] or config["contrast"]["enabled"]:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=config["brightness"]["limit"],
            contrast_limit=config["contrast"]["limit"],
            p=max(config["brightness"]["probability"], config["contrast"]["probability"]),
        ))
    if config["gamma"]["enabled"]:
        transforms.append(A.RandomGamma(gamma_limit=tuple(config["gamma"]["gamma_limit"]), p=config["gamma"]["probability"]))
    if config["motion_blur"]["enabled"]:
        transforms.append(A.MotionBlur(blur_limit=tuple(config["motion_blur"]["kernel_size"]), p=config["motion_blur"]["probability"]))
    if config["gaussian_noise"]["enabled"]:
        transforms.append(A.GaussNoise(var_limit=tuple(config["gaussian_noise"]["variance_limit"]), p=config["gaussian_noise"]["probability"]))
    if config["random_shadow"]["enabled"]:
        transforms.append(A.RandomShadow(shadow_roi=tuple(config["random_shadow"]["shadow_roi"]), p=config["random_shadow"]["probability"]))
    affine_probability = max(config["rotation"]["probability"], config["scale"]["probability"], config["translation"]["probability"])
    transforms.append(A.Affine(
        rotate=(-config["rotation"]["limit_degrees"], config["rotation"]["limit_degrees"]),
        scale=(1 - config["scale"]["limit"], 1 + config["scale"]["limit"]),
        translate_percent=(-config["translation"]["limit"], config["translation"]["limit"]),
        p=affine_probability,
    ))
    return A.Compose(transforms, bbox_params=A.BboxParams(format=config["bbox_format"], label_fields=["class_labels"]))
