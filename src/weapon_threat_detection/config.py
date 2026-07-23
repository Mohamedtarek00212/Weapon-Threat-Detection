from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent.parent
