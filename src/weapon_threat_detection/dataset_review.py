from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def audit_duplicate_annotations(dataset_root: str | Path, splits: Iterable[str]) -> dict[str, object]:
    root = Path(dataset_root)
    files_scanned = duplicate_files = duplicate_annotations = 0
    details = []
    for split in splits:
        for label in sorted((root / split / "labels").glob("*.txt")):
            files_scanned += 1
            seen: set[tuple[str, ...]] = set()
            duplicates = []
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                fields = tuple(line.split())
                if fields and fields in seen:
                    duplicates.append(line_number)
                elif fields:
                    seen.add(fields)
            if duplicates:
                duplicate_files += 1
                duplicate_annotations += len(duplicates)
                details.append({"split": split, "label": str(label), "duplicate_line_numbers": duplicates, "duplicate_annotations": len(duplicates)})
    return {
        "files_scanned": files_scanned,
        "files_with_duplicates": duplicate_files,
        "duplicate_annotations": duplicate_annotations,
        "details": details,
    }


def remove_duplicate_annotations(dataset_root: str | Path, splits: Iterable[str]) -> dict[str, object]:
    root = Path(dataset_root)
    files_scanned = files_repaired = duplicates_removed = 0
    repairs = []
    for split in splits:
        for label in sorted((root / split / "labels").glob("*.txt")):
            files_scanned += 1
            lines = label.read_text(encoding="utf-8").splitlines()
            seen: set[tuple[str, ...]] = set()
            retained = []
            removed_lines = []
            for line_number, line in enumerate(lines, start=1):
                fields = tuple(line.split())
                if fields and fields in seen:
                    removed_lines.append(line_number)
                    continue
                if fields:
                    seen.add(fields)
                retained.append(line)
            if removed_lines:
                label.write_text("\n".join(retained) + "\n", encoding="utf-8")
                files_repaired += 1
                duplicates_removed += len(removed_lines)
                repairs.append({"split": split, "label": str(label), "removed_line_numbers": removed_lines, "duplicates_removed": len(removed_lines)})
    return {
        "files_scanned": files_scanned,
        "files_repaired": files_repaired,
        "duplicates_removed": duplicates_removed,
        "repairs": repairs,
    }


def write_duplicate_repair_csv(repairs: list[dict[str, object]], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "removed_line_numbers", "duplicates_removed"])
        writer.writeheader()
        writer.writerows({**record, "removed_line_numbers": ";".join(map(str, record["removed_line_numbers"]))} for record in repairs)
    return target


def class_balance_summary(statistics: dict[str, object]) -> dict[str, object]:
    counts = statistics["annotations_per_class"]
    maximum = max(counts.values())
    minimum = min(counts.values())
    return {"max_class": max(counts, key=counts.get), "min_class": min(counts, key=counts.get), "max_to_min_ratio": maximum / minimum, "counts": counts}
