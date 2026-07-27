from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .model_engineering import ProjectYOLO11s


@dataclass(frozen=True)
class TransferReport:
    source_checkpoint: str
    total_model_parameters: int
    loaded_parameters: int
    newly_initialized_parameters: int
    transferred_percentage: float
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    incompatible_keys: list[str]
    layer_mapping: dict[str, int]


def _layer_mapping(source: torch.nn.Module, target: torch.nn.Module) -> dict[int, int]:
    source_layers = list(source.model)
    target_layers = list(target.model)
    mapping: dict[int, int] = {}
    source_index = 0
    for target_index, target_layer in enumerate(target_layers):
        if type(target_layer).__name__ == "CBAM":
            continue
        while source_index < len(source_layers) and type(source_layers[source_index]).__name__ != type(target_layer).__name__:
            source_index += 1
        if source_index >= len(source_layers):
            raise ValueError(f"No compatible source layer for custom layer {target_index}: {type(target_layer).__name__}")
        mapping[target_index] = source_index
        source_index += 1
    return mapping


def _source_key(target_key: str, mapping: dict[int, int]) -> str | None:
    parts = target_key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return target_key
    target_index = int(parts[1])
    if target_index not in mapping:
        return None
    parts[1] = str(mapping[target_index])
    return ".".join(parts)


def load_pretrained_weights(model_config: str | Path, training_config: str | Path, checkpoint: str | Path, nc: int = 5) -> tuple[ProjectYOLO11s, TransferReport]:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Official YOLO11s checkpoint is missing: {checkpoint_path}")
    source = YOLO(str(checkpoint_path)).model
    target = ProjectYOLO11s(model_config, training_config, nc=nc, verbose=False)
    source_state = source.state_dict()
    target_state = target.state_dict()
    mapping = _layer_mapping(source, target)
    loaded_keys: list[str] = []
    missing_keys: list[str] = []
    incompatible_keys: list[str] = []
    used_source_keys: set[str] = set()
    with torch.no_grad():
        for target_key, target_tensor in target_state.items():
            candidate = _source_key(target_key, mapping)
            if candidate is None:
                missing_keys.append(target_key)
                continue
            source_tensor = source_state.get(candidate)
            if source_tensor is None:
                missing_keys.append(target_key)
                continue
            if source_tensor.shape != target_tensor.shape:
                incompatible_keys.append(f"{target_key} <- {candidate}: {tuple(target_tensor.shape)} != {tuple(source_tensor.shape)}")
                continue
            target_tensor.copy_(source_tensor)
            loaded_keys.append(target_key)
            used_source_keys.add(candidate)
    target.load_state_dict(target_state, strict=False)
    parameter_names = set(dict(target.named_parameters()))
    total = sum(parameter.numel() for parameter in target.parameters())
    loaded = sum(target_state[key].numel() for key in loaded_keys if key in parameter_names)
    unexpected = sorted(set(source_state) - used_source_keys)
    report = TransferReport(
        source_checkpoint=str(checkpoint_path),
        total_model_parameters=total,
        loaded_parameters=loaded,
        newly_initialized_parameters=total - loaded,
        transferred_percentage=loaded / total * 100,
        loaded_keys=loaded_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected,
        incompatible_keys=incompatible_keys,
        layer_mapping={str(target): source for target, source in mapping.items()},
    )
    return target, report


def freeze_transferred_backbone_layers(model: ProjectYOLO11s, freeze_through_layer: int) -> dict[str, int]:
    frozen = trainable = 0
    for index, layer in enumerate(model.model):
        requires_grad = index > freeze_through_layer
        for parameter in layer.parameters():
            parameter.requires_grad = requires_grad
            if requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
    return {"freeze_through_layer": freeze_through_layer, "frozen_parameters": frozen, "trainable_parameters": trainable}


def serialize_transfer_report(report: TransferReport) -> dict[str, Any]:
    return {
        "source_checkpoint": report.source_checkpoint,
        "total_model_parameters": report.total_model_parameters,
        "loaded_parameters": report.loaded_parameters,
        "newly_initialized_parameters": report.newly_initialized_parameters,
        "transferred_percentage": report.transferred_percentage,
        "loaded_key_count": len(report.loaded_keys),
        "missing_keys": report.missing_keys,
        "unexpected_keys": report.unexpected_keys,
        "incompatible_keys": report.incompatible_keys,
        "layer_mapping": report.layer_mapping,
    }
