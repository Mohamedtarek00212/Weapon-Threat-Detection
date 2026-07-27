from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
import yaml
from ultralytics.nn import tasks
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction_ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(self.pool(inputs)))


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((inputs.mean(dim=1, keepdim=True), inputs.amax(dim=1, keepdim=True)), dim=1)
        return torch.sigmoid(self.convolution(pooled))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return inputs
        attended = inputs * self.channel_attention(inputs)
        return attended * self.spatial_attention(attended)


class ElementwiseFocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float, class_weights: torch.Tensor | None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        probabilities = predictions.sigmoid()
        target_probability = targets * probabilities + (1 - targets) * (1 - probabilities)
        modulation = (1 - target_probability).pow(self.gamma)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        class_weight = self.class_weights.to(predictions.device) if self.class_weights.numel() else 1.0
        return bce * modulation * alpha_weight * class_weight


class ConfigurableDetectionLoss(v8DetectionLoss):
    def __init__(self, model: nn.Module, gamma: float, alpha: float) -> None:
        super().__init__(model)
        self.bce = ElementwiseFocalLoss(gamma, alpha, getattr(model, "class_weights", None))


class ProjectYOLO11s(DetectionModel):
    def __init__(self, model_config: str | Path, training_config: str | Path, nc: int = 5, verbose: bool = False, cbam_enabled: bool | None = None) -> None:
        self.training_config = load_yaml(training_config)
        model_definition = load_yaml(model_config)
        cbam = model_definition["cbam"]
        enabled = cbam["enabled"] if cbam_enabled is None else cbam_enabled
        for layer in model_definition["backbone"]:
            if layer[2] == "CBAM":
                if enabled:
                    layer[3][1] = cbam["reduction_ratio"]
                    layer[3][2] = True
                else:
                    layer[2] = "nn.Identity"
                    layer[3] = []
        tasks.CBAM = CBAM
        super().__init__(cfg=model_definition, ch=3, nc=nc, verbose=verbose)
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        class_weights = self.training_config["loss"]["class_weights"]
        self.class_weights = torch.tensor(class_weights["values"], dtype=torch.float32) if class_weights["enabled"] else None

    def init_criterion(self) -> ConfigurableDetectionLoss:
        focal = self.training_config["loss"]["focal"]
        if focal["enabled"]:
            return ConfigurableDetectionLoss(self, focal["gamma"], focal["alpha"])
        return v8DetectionLoss(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class HardwareReport:
    device: str
    backend: str
    chip: str
    total_memory_gb: float | None
    cpu_cores: int
    recommended_workers: int
    torch_version: str


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def detect_hardware() -> HardwareReport:
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        return HardwareReport("cuda:0", "cuda", properties.name, properties.total_memory / 1024**3, os.cpu_count() or 1, min(8, os.cpu_count() or 1), torch.__version__)
    if torch.backends.mps.is_available():
        memory = _sysctl("hw.memsize")
        chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
        return HardwareReport("mps", "mps", chip, int(memory) / 1024**3 if memory else None, os.cpu_count() or 1, min(8, max(1, (os.cpu_count() or 1) // 2)), torch.__version__)
    return HardwareReport("cpu", "cpu", platform.processor() or "CPU", None, os.cpu_count() or 1, min(4, os.cpu_count() or 1), torch.__version__)


def model_summary(model: nn.Module, image_size: int, device: str, cbam_locations: list[str], focal_config: dict[str, Any]) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    flops = get_flops_with_torch_profiler(model, image_size)
    return {
        "total_parameters": parameters,
        "trainable_parameters": trainable,
        "gflops": flops,
        "image_size": image_size,
        "device": device,
        "cbam_locations": cbam_locations,
        "loss_configuration": focal_config,
    }


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()


def _peak_memory_mb(device: str) -> float | None:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024**2
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024**2
    return None


def _output_scalar(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float().mean()
    if isinstance(output, dict):
        return sum((_output_scalar(value) for value in output.values()), start=torch.zeros((), device=next(iter(output.values())).device))
    if isinstance(output, (list, tuple)):
        return sum((_output_scalar(value) for value in output), start=torch.zeros((), device=next(iter(output)).device))
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def benchmark_memory(model: nn.Module, device: str, image_size: int, batch_sizes: list[int], iterations: int, warmup_iterations: int) -> list[dict[str, Any]]:
    model = model.to(device).train()
    results = []
    for batch_size in batch_sizes:
        record: dict[str, Any] = {"batch_size": batch_size, "status": "failed", "peak_memory_mb": None, "images_per_second": None, "error": None}
        try:
            _reset_peak_memory(device)
            for _ in range(warmup_iterations):
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                output = model(images)
                loss = _output_scalar(output)
                loss.backward()
                model.zero_grad(set_to_none=True)
            _synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                output = model(images)
                loss = _output_scalar(output)
                loss.backward()
                model.zero_grad(set_to_none=True)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            record.update({"status": "stable", "peak_memory_mb": _peak_memory_mb(device), "images_per_second": batch_size * iterations / elapsed})
        except RuntimeError as error:
            record["error"] = str(error)
            if device == "mps":
                torch.mps.empty_cache()
        results.append(record)
    return results


def recommend_hyperparameters(hardware: HardwareReport, benchmark: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [record for record in benchmark if record["status"] == "stable"]
    best = max(stable, key=lambda record: record["batch_size"]) if stable else None
    return {
        "model": "YOLO11s with CBAM at P3, P4, and P5",
        "epochs": 120,
        "freeze_epochs": 5,
        "unfreeze_epochs": 115,
        "batch_size": best["batch_size"] if best else None,
        "image_size": 640,
        "learning_rate": 0.003,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "scheduler": "cosine",
        "warmup_epochs": 3,
        "workers": hardware.recommended_workers,
        "cache_mode": "disk",
        "amp": True,
        "early_stopping_patience": 25,
        "checkpoint_frequency": 1,
        "validation_frequency": 1,
        "benchmark_basis": best,
        "approval_required": True,
    }


def serialize_hardware(report: HardwareReport) -> dict[str, Any]:
    return asdict(report)
