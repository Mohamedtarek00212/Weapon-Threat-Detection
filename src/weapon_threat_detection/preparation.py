from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable

import yaml

from .artifacts import write_json


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if mode == "symlink":
        try:
            destination.symlink_to(source.resolve())
            return
        except OSError:
            mode = "copy"
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    raise ValueError("mode must be 'symlink' or 'copy'")


def _write_dataset_yaml(target: Path, class_names: list[str]) -> Path:
    dataset_yaml = {
        "path": str(target),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    target_path = target / "data.yaml"
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)
    return target_path


def prepare_dataset(source_root: str | Path, target_root: str | Path, splits: Iterable[str], class_names: list[str], mode: str = "symlink") -> dict[str, object]:
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if source == target or source in target.parents:
        raise ValueError("Processed dataset must be outside the read-only original dataset")
    files_by_split: dict[str, int] = {}
    for split in splits:
        count = 0
        for kind in ("images", "labels"):
            source_directory = source / split / kind
            target_directory = target / split / kind
            target_directory.mkdir(parents=True, exist_ok=True)
            for item in sorted(source_directory.iterdir()):
                if item.is_file():
                    _materialize(item, target_directory / item.name, mode)
                    count += 1
        files_by_split[split] = count
    dataset_yaml = _write_dataset_yaml(target, class_names)
    metadata = {"source_root": str(source), "target_root": str(target), "mode": mode, "files_by_split": files_by_split, "dataset_yaml": str(dataset_yaml)}
    write_json(target / "metadata.json", metadata)
    return metadata


def materialize_clean_dataset(source_root: str | Path, target_root: str | Path, validation_csv: str | Path, splits: Iterable[str], class_names: list[str], mode: str = "symlink") -> list[dict[str, str]]:
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if source == target or source in target.parents:
        raise ValueError("Processed dataset must be outside the read-only original dataset")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Target dataset already exists and is not empty: {target}")
    with Path(validation_csv).open("r", encoding="utf-8", newline="") as handle:
        approved = [row for row in csv.DictReader(handle) if row["status"] == "invalid_label"]
    excluded_images = {Path(row["image"]).resolve() for row in approved}
    excluded_labels = {Path(row["label"]).resolve() for row in approved}
    if len(excluded_images) != len(approved) or len(excluded_labels) != len(approved):
        raise ValueError("Validation report contains duplicate invalid-pair records")
    for row in approved:
        if not Path(row["image"]).is_file() or not Path(row["label"]).is_file():
            raise FileNotFoundError(f"Approved invalid pair no longer exists: {row}")
    for split in splits:
        for kind in ("images", "labels"):
            source_directory = source / split / kind
            destination_directory = target / split / kind
            destination_directory.mkdir(parents=True, exist_ok=True)
            excluded = excluded_images if kind == "images" else excluded_labels
            for item in sorted(source_directory.iterdir()):
                if item.is_file() and item.resolve() not in excluded:
                    _materialize(item, destination_directory / item.name, mode)
    _write_dataset_yaml(target, class_names)
    write_json(target / "metadata.json", {
        "source_root": str(source),
        "target_root": str(target),
        "mode": mode,
        "excluded_pairs": len(approved),
        "source_modified": False,
        "exclusion_policy": "Only approved invalid image-label pairs were excluded from this derived dataset.",
    })
    return [
        {
            "split": row["split"],
            "source_image": row["image"],
            "source_label": row["label"],
            "reason": row["detail"],
            "action": "excluded_from_derived_dataset",
        }
        for row in approved
    ]


def write_removal_report(records: list[dict[str, str]], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "source_image", "source_label", "reason", "action"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return target
