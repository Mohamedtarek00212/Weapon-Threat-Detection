from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ValidationRecord:
    split: str
    image: str
    label: str
    status: str
    detail: str
    annotations: int


def _label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _validate_label(label_path: Path, class_count: int) -> tuple[str, str, int]:
    if not label_path.exists():
        return "missing_label", "No matching label file", 0
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "valid_background", "Empty label file is a valid negative sample", 0
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            return "invalid_label", f"Line {line_number}: expected 5 YOLO fields", len(lines)
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return "invalid_label", f"Line {line_number}: non-numeric YOLO values", len(lines)
        if not 0 <= class_id < class_count:
            return "invalid_label", f"Line {line_number}: class id {class_id} outside range", len(lines)
        x, y, width, height = values
        if not all(0 <= value <= 1 for value in values) or width == 0 or height == 0:
            return "invalid_label", f"Line {line_number}: invalid normalized box", len(lines)
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            return "invalid_label", f"Line {line_number}: box exceeds image bounds", len(lines)
    return "valid", "", len(lines)


def validate_dataset(dataset_root: str | Path, splits: Iterable[str], class_count: int, image_extensions: Iterable[str]) -> list[ValidationRecord]:
    root = Path(dataset_root)
    allowed = {extension.lower() for extension in image_extensions}
    records: list[ValidationRecord] = []
    for split in splits:
        images_dir, labels_dir = root / split / "images", root / split / "labels"
        if not images_dir.is_dir():
            records.append(ValidationRecord(split, "", "", "missing_images_directory", f"Missing {images_dir}", 0))
        if not labels_dir.is_dir():
            records.append(ValidationRecord(split, "", "", "missing_labels_directory", f"Missing {labels_dir}", 0))
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed)
        image_stems = {image_path.stem for image_path in images}
        for label_path in sorted(labels_dir.glob("*.txt")):
            if label_path.stem not in image_stems:
                records.append(ValidationRecord(split, "", str(label_path), "orphan_label", "No matching image file", 0))
        for image_path in images:
            label_path = _label_path(image_path, labels_dir)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError) as error:
                records.append(ValidationRecord(split, str(image_path), str(label_path), "corrupt_image", str(error), 0))
                continue
            try:
                status, detail, annotations = _validate_label(label_path, class_count)
            except (OSError, UnicodeDecodeError) as error:
                status, detail, annotations = "unreadable_label", str(error), 0
            records.append(ValidationRecord(split, str(image_path), str(label_path), status, detail, annotations))
    return records


def build_integrity_report(records: Iterable[ValidationRecord], expected_splits: Iterable[str]) -> dict[str, object]:
    records = list(records)
    status_counts = summarize_records(records)
    issues = [
        {"split": record.split, "image": record.image, "label": record.label, "status": record.status, "detail": record.detail}
        for record in records
        if record.status not in {"valid", "valid_background"}
    ]
    return {
        "expected_splits": list(expected_splits),
        "status_counts": status_counts,
        "issue_count": len(issues),
        "issues": issues,
        "deletion_policy": "No files were deleted or modified during validation.",
        "background_policy": "Empty label files are valid background samples and remain untouched.",
    }


def build_dataset_statistics(records: Iterable[ValidationRecord], class_names: list[str]) -> dict[str, object]:
    records = list(records)
    split_counts: dict[str, Counter[str]] = {}
    class_counts = Counter()
    for record in records:
        split_counts.setdefault(record.split, Counter())[record.status] += 1
        if record.status == "valid":
            for line in Path(record.label).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    class_counts[class_names[int(line.split()[0])]] += 1
    return {
        "image_records": sum(1 for record in records if record.image),
        "annotations": sum(record.annotations for record in records if record.status == "valid"),
        "class_annotation_counts": dict(sorted(class_counts.items())),
        "splits": {split: dict(sorted(counts.items())) for split, counts in sorted(split_counts.items())},
        "background_images": sum(record.status == "valid_background" for record in records),
    }


def summarize_records(records: Iterable[ValidationRecord]) -> dict[str, int]:
    return dict(Counter(record.status for record in records))


def write_validation_csv(records: Iterable[ValidationRecord], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ValidationRecord.__dataclass_fields__.keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return target
