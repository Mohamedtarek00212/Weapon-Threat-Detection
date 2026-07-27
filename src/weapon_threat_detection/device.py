from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    backend: str
    platform: str


def select_device() -> DeviceInfo:
    if torch.cuda.is_available():
        return DeviceInfo(name="cuda:0", backend="cuda", platform=platform.platform())
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(name="mps", backend="mps", platform=platform.platform())
    return DeviceInfo(name="cpu", backend="cpu", platform=platform.platform())
