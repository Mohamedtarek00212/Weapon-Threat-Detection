from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import __version__ as ultralytics_version

from .artifacts import configure_logger, ensure_directory, utc_timestamp, write_json
from .engineering import MERGED_CLASS_NAMES
from .model_engineering import detect_hardware, load_yaml, serialize_hardware
from .transfer_learning import (
    freeze_transferred_backbone_layers,
    load_pretrained_weights,
    serialize_transfer_report,
)


@dataclass(frozen=True)
class LauncherContext:
    root: Path
    training_path: Path
    model_path: Path
    experiment_path: Path
    run_directory: Path
    dataset_path: Path
    configuration: dict[str, Any]
    resume_checkpoint: Path | None


def _root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_configuration(context: LauncherContext) -> None:
    training = context.configuration["training"]
    loss = context.configuration["loss"]
    required = {
        "epochs": int,
        "freeze_epochs": int,
        "unfreeze_epochs": int,
        "freeze_through_layer": int,
        "batch_size": int,
        "image_size": int,
        "learning_rate": (int, float),
        "workers": int,
        "seed": int,
    }
    missing = [name for name in required if name not in training]
    invalid = [name for name, expected in required.items() if name in training and not isinstance(training[name], expected)]
    if missing or invalid:
        raise ValueError(f"Invalid training configuration: missing={missing}, invalid={invalid}")
    if training["epochs"] != training["freeze_epochs"] + training["unfreeze_epochs"]:
        raise ValueError("epochs must equal freeze_epochs plus unfreeze_epochs")
    if training["freeze_epochs"] != 10 or training["freeze_through_layer"] != 10:
        raise ValueError("The approved schedule requires layers 0-10 frozen for the first 10 epochs")
    if training["image_size"] != 800 or training["batch_size"] != 28:
        raise ValueError("The approved configuration requires image_size=800 and batch_size=28")
    if training["optimizer"] != "AdamW" or training["scheduler"] != "cosine":
        raise ValueError("The approved configuration requires AdamW with cosine scheduling")
    if not training["amp"] or not training["deterministic"] or training["seed"] != 42:
        raise ValueError("AMP, deterministic mode, and seed=42 are required")
    augmentation = training["augmentation"]
    for name in ("mosaic", "mixup", "copy_paste", "perspective", "shear", "vertical_flip"):
        if augmentation.get(name) != 0.0:
            raise ValueError(f"CCTV augmentation '{name}' must be disabled")
    focal = loss["focal"]
    if not focal["enabled"] or focal["gamma"] != 2.0 or focal["alpha"] != 0.25:
        raise ValueError("The approved focal-loss configuration is enabled with gamma=2.0 and alpha=0.25")
    class_weights = loss["class_weights"]
    if not class_weights["enabled"] or len(class_weights["values"]) != len(MERGED_CLASS_NAMES):
        raise ValueError("Five enabled audited class weights are required")
    if not context.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset YAML is missing: {context.dataset_path}")
    checkpoint = context.root / training["pretrained_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official pretrained checkpoint is missing: {checkpoint}")


def create_context(root: str | Path | None = None) -> LauncherContext:
    project_root = Path(root).resolve() if root else _root_from_module()
    training_path = project_root / "configs" / "training.yaml"
    model_path = project_root / "configs" / "model.yaml"
    experiment_path = project_root / "configs" / "experiment.yaml"
    configuration = load_yaml(training_path)
    experiment = load_yaml(experiment_path)["experiment"]
    run_directory = project_root / "runs" / "final_training"
    last = run_directory / "weights" / "last.pt"
    resume_checkpoint = last if configuration["training"]["resume"] and last.is_file() else None
    context = LauncherContext(
        root=project_root,
        training_path=training_path,
        model_path=model_path,
        experiment_path=experiment_path,
        run_directory=run_directory,
        dataset_path=project_root / configuration["training"]["dataset"],
        configuration=configuration,
        resume_checkpoint=resume_checkpoint,
    )
    _validate_configuration(context)
    return context


def _configure_reproducibility(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def _trainer_overrides(context: LauncherContext) -> dict[str, Any]:
    training = context.configuration["training"]
    augmentation = training["augmentation"]
    device = detect_hardware().device if training["device"] == "auto" else training["device"]
    return {
        "task": "detect",
        "mode": "train",
        "model": str(context.model_path),
        "data": str(context.dataset_path),
        "epochs": training["epochs"],
        "patience": training["early_stopping_patience"],
        "batch": training["batch_size"],
        "imgsz": training["image_size"],
        "optimizer": training["optimizer"],
        "lr0": training["learning_rate"],
        "lrf": training["final_learning_rate_factor"],
        "momentum": training["momentum"],
        "weight_decay": training["weight_decay"],
        "warmup_epochs": training["warmup_epochs"],
        "cos_lr": training["scheduler"] == "cosine",
        "workers": training["workers"],
        "cache": training["cache_mode"],
        "device": device,
        "amp": training["amp"],
        "seed": training["seed"],
        "deterministic": training["deterministic"],
        "val": True,
        "save": True,
        "save_period": training["checkpoint_frequency"],
        "project": str(context.run_directory.parent),
        "name": context.run_directory.name,
        "exist_ok": True,
        "pretrained": False,
        "freeze": list(range(training["freeze_through_layer"] + 1)),
        "resume": str(context.resume_checkpoint) if context.resume_checkpoint else False,
        "mosaic": augmentation["mosaic"],
        "mixup": augmentation["mixup"],
        "copy_paste": augmentation["copy_paste"],
        "perspective": augmentation["perspective"],
        "shear": augmentation["shear"],
        "translate": augmentation["translate"],
        "scale": augmentation["scale"],
        "fliplr": augmentation["horizontal_flip"],
        "flipud": augmentation["vertical_flip"],
        "hsv_h": augmentation["hsv_hue"],
        "hsv_s": augmentation["hsv_saturation"],
        "hsv_v": augmentation["hsv_value"],
        "degrees": 0.0,
        "plots": True,
    }


class ProjectDetectionTrainer(DetectionTrainer):
    def __init__(self, context: LauncherContext) -> None:
        self.context = context
        self.project_model = None
        self.transfer_report: dict[str, Any] | None = None
        super().__init__(overrides=_trainer_overrides(context))
        self.add_callback("on_train_epoch_start", self._unfreeze_after_warm_start)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        if self.project_model is None:
            model, report = load_pretrained_weights(
                self.context.model_path,
                self.context.training_path,
                self.context.root / self.context.configuration["training"]["pretrained_checkpoint"],
                nc=self.data["nc"],
            )
            model.names = self.data["names"]
            self.project_model = model
            self.transfer_report = serialize_transfer_report(report)
        return self.project_model

    def _unfreeze_after_warm_start(self, trainer) -> None:
        freeze_epochs = self.context.configuration["training"]["freeze_epochs"]
        if trainer.epoch == freeze_epochs:
            for parameter in trainer.model.parameters():
                parameter.requires_grad = True
            trainer.args.freeze = []

    def _load_checkpoint_state(self, checkpoint) -> None:
        super()._load_checkpoint_state(checkpoint)
        if checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("epoch", -1) + 1 >= self.context.configuration["training"]["freeze_epochs"]:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.args.freeze = []

    def save_model(self):
        result = super().save_model()
        for checkpoint_path in (self.last, self.best):
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint["scheduler"] = self.scheduler.state_dict()
                checkpoint["freeze_epochs"] = self.context.configuration["training"]["freeze_epochs"]
                torch.save(checkpoint, checkpoint_path)
        return result


def _write_initial_reports(context: LauncherContext, transfer_report: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Path]:
    run_directory = ensure_directory(context.run_directory)
    training_log = run_directory / "training.log"
    training_log.write_text("Training launcher initialized; no epoch has started.\n", encoding="utf-8")
    metrics = run_directory / "metrics.csv"
    if not metrics.exists():
        metrics.write_text("epoch,fitness,precision,recall,map50,map50_95\n", encoding="utf-8")
    report = run_directory / "training_report.md"
    if not report.exists():
        report.write_text("# Training Report\n\nTraining has not started. This file is completed after final evaluation.\n", encoding="utf-8")
    summary = run_directory / "experiment_summary.json"
    if not summary.exists():
        write_json(summary, {"completed": False, "training_started": False})
    paths = {
        "hyperparameters": write_json(run_directory / "hyperparameters.json", context.configuration),
        "system_information": write_json(run_directory / "system_information.json", hardware),
        "launcher_report": write_json(run_directory / "launcher_report.json", {"configuration_valid": True, "transfer": transfer_report, "resume_checkpoint": str(context.resume_checkpoint) if context.resume_checkpoint else None}),
        "training_log": training_log,
        "metrics": metrics,
        "training_report": report,
        "experiment_summary": summary,
    }
    return paths


def _write_final_reports(context: LauncherContext, trainer: ProjectDetectionTrainer) -> None:
    results = context.run_directory / "results.csv"
    metrics = context.run_directory / "metrics.csv"
    if results.is_file():
        shutil.copy2(results, metrics)
    rows = list(csv.DictReader(results.open(encoding="utf-8"))) if results.is_file() else []
    final_metrics = rows[-1] if rows else {}
    summary = {
        "completed": True,
        "best_checkpoint": str(trainer.best),
        "last_checkpoint": str(trainer.last),
        "best_fitness": trainer.best_fitness,
        "final_epoch": trainer.epoch,
        "metrics_rows": len(rows),
        "final_metrics": final_metrics,
        "confusion_matrix": str(context.run_directory / "confusion_matrix.png"),
        "training_curves": str(context.run_directory / "results.png"),
    }
    write_json(context.run_directory / "experiment_summary.json", summary)
    (context.run_directory / "training_report.md").write_text(
        "# Training Report\n\n"
        f"- Final epoch: {trainer.epoch}\n"
        f"- Best fitness: {trainer.best_fitness}\n"
        f"- Precision, recall, mAP50, and mAP50-95: {final_metrics}\n"
        f"- Per-class metrics: `{context.run_directory / 'results.csv'}`\n"
        f"- Confusion matrix: `{context.run_directory / 'confusion_matrix.png'}`\n"
        f"- Training and validation curves: `{context.run_directory / 'results.png'}`\n"
        f"- Best model: `{trainer.best}`\n"
        f"- Last checkpoint: `{trainer.last}`\n"
        f"- Metrics: `{metrics}`\n"
        "- Comparison against previous experiments: no prior training experiment exists; this is the approved sole run.\n",
        encoding="utf-8",
    )


def smoke_test(root: str | Path | None = None) -> dict[str, Any]:
    context = create_context(root)
    _configure_reproducibility(context.configuration["training"]["seed"], context.configuration["training"]["deterministic"])
    hardware = serialize_hardware(detect_hardware())
    dataset_data = yaml.safe_load(context.dataset_path.read_text(encoding="utf-8"))
    dataset = YOLODataset(
        img_path=str(context.dataset_path.parent / dataset_data["train"]),
        imgsz=context.configuration["training"]["image_size"],
        augment=False,
        cache=False,
        data=dataset_data,
    )
    trainer = ProjectDetectionTrainer(context)
    model, transfer = load_pretrained_weights(
        context.model_path,
        context.training_path,
        context.root / context.configuration["training"]["pretrained_checkpoint"],
        nc=len(MERGED_CLASS_NAMES),
    )
    freeze = freeze_transferred_backbone_layers(model, context.configuration["training"]["freeze_through_layer"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=context.configuration["training"]["learning_rate"], weight_decay=context.configuration["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=context.configuration["training"]["epochs"])
    criterion = model.init_criterion()
    reports = _write_initial_reports(context, serialize_transfer_report(transfer), hardware)
    result = {
        "configuration_loading": True,
        "dataset_loading": len(dataset) > 0,
        "trainer_construction": isinstance(trainer, ProjectDetectionTrainer),
        "model_construction": True,
        "pretrained_weight_loading": transfer.transferred_percentage > 99,
        "cbam_initialization": sum(1 for layer in model.model if type(layer).__name__ == "CBAM") == 3,
        "focal_loss_initialization": type(criterion).__name__ == "ConfigurableDetectionLoss",
        "class_weight_loading": model.class_weights.tolist(),
        "optimizer_creation": type(optimizer).__name__,
        "scheduler_creation": type(scheduler).__name__,
        "checkpoint_initialization": {"best": str(context.run_directory / "weights" / "best.pt"), "last": str(context.run_directory / "weights" / "last.pt")},
        "resume_logic": str(context.resume_checkpoint) if context.resume_checkpoint else "no existing checkpoint; fresh approved run will initialize checkpoints on epoch 1",
        "freeze_strategy": freeze,
        "reports": {name: str(path) for name, path in reports.items()},
        "training_started": False,
    }
    write_json(context.run_directory / "smoke_test_report.json", result)
    return result


def train(root: str | Path | None = None) -> None:
    context = create_context(root)
    _configure_reproducibility(context.configuration["training"]["seed"], context.configuration["training"]["deterministic"])
    trainer = ProjectDetectionTrainer(context)
    hardware = serialize_hardware(detect_hardware())
    preview_model, transfer = load_pretrained_weights(context.model_path, context.training_path, context.root / context.configuration["training"]["pretrained_checkpoint"], nc=len(MERGED_CLASS_NAMES))
    del preview_model
    _write_initial_reports(context, serialize_transfer_report(transfer), hardware)
    trainer.train()
    _write_final_reports(context, trainer)


def main(argv: list[str] | None = None) -> None:
    arguments = argv if argv is not None else sys.argv[1:]
    if arguments == ["--smoke-test"]:
        print(json.dumps(smoke_test(), indent=2, default=str))
        return
    if arguments:
        raise SystemExit("Usage: python train.py [--smoke-test]")
    train()
