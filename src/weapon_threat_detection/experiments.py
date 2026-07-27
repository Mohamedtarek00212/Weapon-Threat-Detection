from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import torch

from .artifacts import configure_logger, ensure_directory, utc_timestamp, write_json
from .config import load_config, project_root
from .device import DeviceInfo, select_device


@dataclass(frozen=True)
class Experiment:
    name: str
    directory: Path
    metadata_path: Path
    log_path: Path
    device: DeviceInfo


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_experiment(config_path: str | Path, name: str) -> Experiment:
    config_file = Path(config_path).resolve()
    root = project_root(config_file)
    config = load_config(config_file)
    runs_directory = ensure_directory(root / config["project"]["runs_dir"])
    identifier = f"{utc_timestamp()}_{name}"
    directory = ensure_directory(runs_directory / identifier)
    device = select_device()
    logger = configure_logger(directory, "experiment")
    config_snapshot = directory / "config.yaml"
    shutil.copy2(config_file, config_snapshot)
    metadata_path = write_json(
        directory / "metadata.json",
        {
            "experiment_name": name,
            "experiment_id": identifier,
            "created_at_utc": utc_timestamp(),
            "config_path": str(config_file),
            "config_snapshot": str(config_snapshot),
            "git_revision": _git_revision(root) if config["experiments"]["capture_git_revision"] else None,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": {"name": device.name, "backend": device.backend},
        },
    )
    log_path = next(directory.glob("experiment_*.log"))
    logger.info("Initialized experiment %s on %s", identifier, device.name)
    return Experiment(name=identifier, directory=directory, metadata_path=metadata_path, log_path=log_path, device=device)
