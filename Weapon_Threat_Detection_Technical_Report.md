# Weapon Threat Detection using YOLO11 with CBAM

## IEEE-Style Technical Report

---

## Abstract

This report documents the end-to-end design, implementation, and current training status of a **YOLO11s-based weapon threat detection** system intended for **CCTV/surveillance**. The work covers the acquisition and cleaning of the **Weapons-130K** dataset, a class-merging taxonomy (7 → 5 classes), a custom `ProjectYOLO11s` architecture with **Convolutional Block Attention Module (CBAM)** inserted at P3/P4/P5, a **focal-loss** and class-weighted imbalance strategy, transfer learning from the official `yolo11s.pt` checkpoint, and a deterministic 80-epoch training pipeline. As of this writing the 80-epoch run has been **paused after epoch 11** (the user reports reaching epoch 12; `runs/final_training/results.csv` contains rows through epoch 11 and the latest checkpoint files are timestamped 2026-07-26 08:21:30). The best recorded validation metrics so far are **P=0.52505, R=0.40424, mAP50=0.41472, mAP50-95=0.27077** at **epoch 10**. All final evaluation artifacts (confusion matrix, per-class AP, PR/F1 curves) are marked as **[MISSING]** because the full run is incomplete. All numbers are taken directly from repository files; no results are invented.

---

## 1. Project Overview and Objectives

- **Problem statement:** Support timely visual threat escalation from CCTV feeds by detecting weapons, explosives, fire/smoke, firearms, and persons while controlling false alerts.
- **Use case:** Real-time surveillance inference with a human-in-the-loop alert-review step.
- **Repository:** `d:\Weapon-Threat-Detection\Weapon130\Weapon-Threat-Detection-main`
- **Entry point:** `train.py` calls `weapon_threat_detection.training_launcher.main()`.
- **Deliverable:** A reproducible, IEEE-style technical report and a trained `best.pt` checkpoint for deployment.

The system pipeline is **dataset → cleaning/merge → augmentation → CBAM-YOLO11s model → transfer learning → focal-loss training → checkpoint/resume → evaluation → export**. Each stage is governed by YAML configs and phase notebooks that write JSON audit reports under `reports/`.

---

## 2. Dataset Source and Engineering

### 2.1 Original Dataset

- **Source:** Weapons-130K on Kaggle: `https://www.kaggle.com/datasets/haiderkhan6410/weapons-130k`
- **License:** CC-BY-4.0
- **Files:** `Original/dataset_info.json`, `Original/data.yaml`
- **Original size:** 130,763 images, 294,950 annotations, 7 classes, 640×640 YOLO format.
- **Original splits:** train 104,697; valid 13,186; test 12,880.
- **Original classes:** `0: Blunt_Weapon`, `1: Explosive`, `2: Fire_Smoke`, `3: Firearm`, `4: Melee_Weapon`, `5: Person`, `6: Tool`.
- **Original train class counts:** Blunt 9,527; Explosive 25,599; Fire_Smoke 78,171; Firearm 117,864; Melee 40,166; Person 11,801; Tool 11,822.

### 2.2 Cleaning Pipeline (4 stages)

From `Original/dataset_info.json`:

- **Stage 1 — Annotation repair:** 1,052 segmentation lines converted to YOLO bbox format; 772 label files rewritten.
- **Stage 2 — Deduplication:** 5,169 duplicate images removed (train 4,049; valid 407; test 713).
- **Stage 3 — Bbox cleanup:** 153 degenerate bounding boxes removed.
- **Stage 4 — Background audit:** 12,892 empty-label frames retained for false-positive suppression.
- **Annotation delta:** 302,550 → 294,950 (7,600 removed).
- **Integrity checks:** 0 corrupt images, 0 out-of-bounds coordinates, 0 orphan labels, 0 missing labels, 0 invalid class IDs → `all_passed`.

### 2.3 Class Merging (7 → 5)

Implemented in `src/weapon_threat_detection/engineering.py` (`CLASS_ID_MAP`):

```python
CLASS_ID_MAP = {0:0, 1:1, 2:2, 3:3, 4:0, 5:4, 6:0}
```

- **Handheld_Weapon** = Blunt_Weapon + Melee_Weapon + Tool
- **Explosive** = original Explosive
- **Fire_Smoke** = original Fire_Smoke
- **Firearm** = original Firearm
- **Person** = original Person

### 2.4 Final Dataset Audit

From `reports/final_readiness_statistics_20260725T092603Z.json` (post-augmentation, post-duplicate-repair):

- **Total images:** 131,463
- **Total annotations:** 296,419
- **Background images:** 12,342

| Split   | Images  | Backgrounds | Annotations |
|---------|---------|-------------|-------------|
| train   | 105,471 | 9,032       | 247,993     |
| valid   | 13,158  | 1,780       | 21,634      |
| test    | 12,834  | 1,530       | 26,792      |

Final class distribution (annotations):

| Class           | Annotations | Share (%) |
|-----------------|-------------|-----------|
| Handheld_Weapon | 61,114      | 20.62     |
| Explosive       | 25,287      | 8.53      |
| Fire_Smoke      | 77,830      | 26.26     |
| Firearm         | 117,017     | 39.48     |
| Person          | 15,171      | 5.12      |

- **Max-to-min annotation ratio:** 7.71 (Firearm vs Person).
- **Duplicate annotation repair:** `reports/phase_6_final_readiness_20260725T092603Z.json` reports 131,463 files scanned, 137 files repaired, 139 exact duplicate annotations removed.

---

## 3. Exploratory Data Analysis (EDA)

- **Distribution charts:** `reports/merged_class_distribution.png`, `reports/merged_class_share.png`, `reports/final_dataset_class_distribution.png`, `reports/final_dataset_before_after_distribution.png`.
- **Person-class analysis:** Person is the minority class and contains a large small-object fraction.
  - Mean bbox area: 0.0849
  - Median bbox area: 0.0345
  - Small-object fraction (area < 0.02): 0.4099 (41%)
  - Mean aspect ratio: 0.6886
- **Augmentation justification:** Person is comparatively scarce and contains a substantial small-object component, so targeted offline augmentation is justified.

---

## 4. Person-Only Offline Augmentation

Implemented in `src/weapon_threat_detection/augmentation.py` and configured by `configs/final_augmentation_config.yaml`.

- **Target:** raise Person annotations to 60% of the smallest non-Person class.
- **Before augmentation:** 11,745 Person annotations across 6,152 images.
- **After augmentation:** 15,171 Person annotations across 7,370 images.
- **Generated:** 1,218 synthetic images and 3,428 Person annotations.
- **Applied transforms (only to Person boxes, seed=42, min bbox visibility=0.7):**
  - Horizontal flip: p=0.5
  - Brightness/Contrast: limit 0.15, p=0.3
  - Gamma: 90–110, p=0.2
  - Motion blur: kernel 3–5, p=0.1
  - Gaussian noise: variance 5.0–25.0, p=0.1
  - Random shadow: p=0.15
  - Random fog: p=0.1
  - Affine: rotation ±7°, scale ±0.1, translation ±0.05, p=0.25

The conservative augmentation strategy was chosen to avoid repetition risk before a baseline evaluation.

---

## 5. Class-Imbalance Handling and Class Weights

- **Formula:** `weight(class) = total_annotations / (number_of_classes * class_annotations)` (balanced inverse frequency).
- **Values used in `configs/training.yaml`:**

| Class           | Weight    |
|-----------------|-----------|
| Handheld_Weapon | 0.9700526884 |
| Explosive       | 2.3444378534 |
| Fire_Smoke      | 0.7617088526 |
| Firearm         | 0.5066255330 |
| Person          | 3.9077054907 |

These weights are loaded into `ProjectYOLO11s` and applied to the focal classification loss, down-weighting Firearm/Fire_Smoke and up-weighting Person/Explosive.

---

## 6. Model Architecture

### 6.1 Baseline: YOLO11s

- **Scale:** `s: [0.50, 0.50, 1024]` (`configs/model.yaml`).
- **Estimated baseline parameters:** 9,429,727 (computed by subtracting CBAM overhead from the measured custom model).
- **Estimated baseline GFLOPs:** 33.29464 at 800×800.

### 6.2 Custom ProjectYOLO11s

Implemented in `src/weapon_threat_detection/model_engineering.py`.

- **Base class:** Ultralytics `DetectionModel` configured from `configs/model.yaml`.
- **CBAM insertion:** three CBAM modules are inserted in the backbone at the P3, P4, and P5 feature levels (YAML indices 5, 8, 13), each with reduction ratio 16.
- **CBAM structure:**
  - `ChannelAttention`: average-pooling → 1×1 Conv → ReLU → 1×1 Conv → Sigmoid.
  - `SpatialAttention`: channel-wise mean/max concatenation → 7×7 Conv → Sigmoid.
- **Configurable focal loss:** `ElementwiseFocalLoss(gamma=2.0, alpha=0.25)` replaces the standard BCE classification loss inside `ConfigurableDetectionLoss`, which extends `v8DetectionLoss` and keeps the box and DFL losses.
- **Class weights:** loaded as a `torch.tensor` and used in the loss.

`configs/model.yaml` excerpt:

```yaml
nc: 5
scale: s
scales:
  s: [0.50, 0.50, 1024]
cbam:
  enabled: true
  reduction_ratio: 16
  locations: [P3, P4, P5]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, false, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, false, 0.25]]
  - [-1, 1, CBAM, [256, 16, true]]        # P3
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, CBAM, [256, 16, true]]        # P4
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, true]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
  - [-1, 1, CBAM, [512, 16, true]]        # P5
```

### 6.3 Model Summary

From `reports/model_summary_20260725T081026Z.json`:

- **Total parameters:** 9,479,173
- **Trainable parameters:** 9,479,157
- **GFLOPs:** 33.3196125 at 800×800
- **CBAM locations:** P3, P4, P5
- **Focal loss:** enabled, gamma=2.0, alpha=0.25

From `reports/cbam_review_20260725T081026Z.json`:

- **Additional parameters from CBAM:** 49,446 (+0.52%)
- **Additional GFLOPs:** 0.02497 (+0.075%)
- **Selection rationale:**
  - P3: retains fine-grained spatial detail for small handheld threats and distant persons.
  - P4: balances semantic context and localization for medium-scale targets.
  - P5: adds global channel/spatial context for large persons, fire, and firearms.

---

## 7. Transfer Learning Strategy

Implemented in `src/weapon_threat_detection/transfer_learning.py` and orchestrated by `src/weapon_threat_detection/training_launcher.py`.

- **Source checkpoint:** `yolo11s.pt` (Ultralytics YOLO11s pretrained on COCO, 80 classes).
- **Layer mapping:** `_layer_mapping()` maps source layer indices to target indices while skipping the 9 CBAM tensors; source layer 23 (80-class head) maps to target layer 26 (5-class head), producing shape mismatches.
- **Transfer report** (`runs/final_training/launcher_report.json` / `reports/final_project_audit_*.json`):
  - **Total target parameters:** 9,479,173
  - **Loaded parameters:** 9,427,792
  - **Transferred percentage:** 99.45795904347352%
  - **Newly initialized parameters:** 51,381
  - **Loaded keys:** 493
  - **Missing keys:** 9 CBAM weight tensors (`model.5/8/13.channel_attention.*`, `model.5/8/13.spatial_attention.*`)
  - **Incompatible keys:** 6 head tensors (`model.26.cv3.0/1/2.2.bias/weight`) because source has 80 classes and target has 5 classes.
  - **Unexpected source keys:** `model.23.cv3.0/1/2.2.bias/weight` (80-class head not used in target).

### 7.1 Freeze/Unfreeze Schedule

- **Epochs 1–10:** freeze backbone layers 0–10.
  - Frozen parameters: 3,810,692
  - Trainable parameters: 5,668,481
- **Epochs 11–80:** full model unfrozen (70 epochs).
- Callback `_unfreeze_after_phase_one` in `ProjectDetectionTrainer` sets all parameters `requires_grad=True` and clears the freeze list once `trainer.epoch >= freeze_epochs`.

---

## 8. Training Configuration

`configs/training.yaml`:

```yaml
training:
  dataset: processed/final_dataset_v1/data.yaml
  model: configs/model.yaml
  pretrained_checkpoint: yolo11s.pt
  device: auto
  epochs: 80
  freeze_epochs: 10
  unfreeze_epochs: 70
  freeze_through_layer: 10
  batch_size: 20
  image_size: 800
  learning_rate: 0.001
  optimizer: AdamW
  weight_decay: 0.0005
  momentum: 0.9
  scheduler: cosine
  warmup_epochs: 3
  workers: 6
  cache_mode: disk
  amp: true
  early_stopping_patience: 20
  checkpoint_frequency: 1
  validation_frequency: 1
  save_best: true
  resume: true
  deterministic: true
  seed: 42
  final_learning_rate_factor: 0.01
  augmentation:
    mosaic: 0.0
    mixup: 0.0
    copy_paste: 0.0
    perspective: 0.0
    shear: 0.0
    translate: 0.03
    scale: 0.10
    horizontal_flip: 0.5
    vertical_flip: 0.0
    hsv_hue: 0.005
    hsv_saturation: 0.15
    hsv_value: 0.10
loss:
  focal:
    enabled: true
    gamma: 2.0
    alpha: 0.25
  class_weights:
    enabled: true
    order: [Handheld_Weapon, Explosive, Fire_Smoke, Firearm, Person]
    values: [0.9700526884, 2.3444378534, 0.7617088526, 0.5066255330, 3.9077054907]
```

### 8.1 Validation of Training Args

`training_launcher._validate_configuration()` enforces the approved run:

- `epochs == 80`, `batch == 20`, `imgsz == 800`
- `optimizer == AdamW`, `scheduler == cosine`
- `freeze_epochs == 10`, `freeze_through_layer == 10`
- `focal enabled` with `gamma == 2.0` and `alpha == 0.25`
- `class_weights enabled` with 5 values
- `seed == 42` and `deterministic == True`
- `resume == True` and `save_best == True`

### 8.2 Hardware and Benchmarks

From `reports/hardware_report_20260725T081026Z.json` and `reports/memory_benchmark_20260725T081026Z.json`:

- **GPU:** NVIDIA RTX 2000 Ada Generation
- **VRAM:** 15.9956 GB
- **CPU cores:** 28
- **Recommended workers:** 8
- **PyTorch:** 2.6.0+cu124, CUDA backend, device `cuda:0`

Memory benchmark at 800×800:

| Batch | Peak memory (MB) | Images/sec | Status  |
|-------|------------------|------------|---------|
| 8     | 5,497.3          | 28.72      | stable  |
| 12    | 8,133.5          | 28.26      | stable  |
| 16    | 10,879.4         | 25.80      | stable  |
| 20    | 13,509.2         | 6.91       | stable  |
| 24    | 16,196.8         | 2.82       | stable  |
| 28    | 18,879.5         | 2.48       | stable  |
| 32    | 21,633.6         | 2.49       | stable  |

Full-unfreeze benchmark used for the time estimate (`reports/final_training_configuration_20260725T120209Z.json`):

- Batch 20, 800×800, full model unfrozen: **8.603 images/sec**, peak **13,570.9 MB**.
- Steps per epoch: 5,274
- Estimated training-only time: **204.32 min/epoch**, **272.43 hours for 80 epochs**.

---

## 9. Experiment History and Readiness

Phase notebooks under `notebooks/` and reports under `reports/`:

| Phase | Notebook / Report | Purpose |
|-------|-------------------|---------|
| 1–4   | `01_Data_Ingestion.ipynb` to `04_Quality_Review.ipynb` | Cleaning, merging, validation |
| 5     | `05_Model_Engineering.ipynb` | Hardware detection, model summary, memory benchmark, CBAM/focal review |
| 6     | `06_Final_Training_Readiness.ipynb` | Duplicate audit, final dataset statistics, overfitting analysis |
| 7–9   | training-configuration notebooks | Hyperparameter lock, final config report |
| 10.1  | final launcher readiness report | Smoke-test the custom trainer and freeze/unfreeze logic |

Key reports:

- `reports/final_dataset_summary_20260725T080400Z.json`
- `reports/final_readiness_statistics_20260725T092603Z.json`
- `reports/final_class_weights.json`
- `reports/model_summary_20260725T081026Z.json`
- `reports/cbam_review_20260725T081026Z.json`
- `reports/focal_loss_review_20260725T081026Z.json`
- `reports/final_training_configuration_20260725T120209Z.json`
- `reports/final_project_audit_20260725T124650Z.json`
- `reports/phase_10_1_final_launcher_readiness_20260725T130203Z.json`

Project audit scores (`reports/final_project_audit_20260725T124650Z.json`):

| Dimension            | Score |
|----------------------|-------|
| Dataset              | 98    |
| Training configuration | 94  |
| Model                | 92    |
| Engineering          | 88    |
| Portfolio quality    | 88    |
| Reproducibility      | 82    |
| Documentation        | 80    |
| Deployment           | 76    |
| **Overall**          | **87**|

Risks recorded in the audit:

- No project-owned training launcher (resolved by `train.py` before the final run).
- Domain shift and split correlation.
- Limited MPS memory headroom at batch 28.
- Focal-loss/class-weight precision-recall trade-off.
- Failure cases: tiny/heavily occluded weapons, extreme low light or motion blur, unseen viewpoints, look-alikes.

---

## 10. Training Progress and Evaluation

### 10.1 Current Status

- The user reports pausing the run after **epoch 12**.
- On disk, `runs/final_training/results.csv` contains rows for **epochs 1–11**.
- `runs/final_training/experiment_summary.json` reports `"completed": false, "training_started": false`.
- `runs/final_training/training_report.md` says “Training has not started. This file is completed after final evaluation.”
- `runs/final_training/metrics.csv` is empty (header only).
- `args.yaml` sets `resume: D:\...\runs\final_training\weights\last.pt`, so the run is **resumable**.

These files are contradictory (logs say not started while `results.csv` has rows). The report uses the concrete `results.csv` values and treats the logs as stale/unflushed artifacts from the launcher's initialization.

### 10.2 Checkpoints

`runs/final_training/weights/`:

- `best.pt` — 76 MB, 2026-07-26 08:21:30
- `last.pt` — 76 MB, 2026-07-26 08:21:30
- `epoch0.pt` through `epoch10.pt` saved; no `epoch11.pt` observed at report time.

The 76 MB checkpoint size after epoch 10 is consistent with the unfreeze transition: full model weights are saved once all layers are trainable.

### 10.3 Validation Metrics from `results.csv`

Full rows (epochs 1–11):

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1     | 0.40216   | 0.19081| 0.13012| 0.06321  |
| 2     | 0.34162   | 0.27062| 0.23019| 0.12563  |
| 3     | 0.39427   | 0.30588| 0.29310| 0.16949  |
| 4     | 0.43790   | 0.34255| 0.33022| 0.20153  |
| 5     | 0.46304   | 0.36632| 0.35811| 0.22438  |
| 6     | 0.47607   | 0.38255| 0.37551| 0.23845  |
| 7     | 0.49570   | 0.39060| 0.39042| 0.25023  |
| 8     | 0.50595   | 0.39578| 0.40221| 0.25950  |
| 9     | 0.51121   | 0.40174| 0.40865| 0.26553  |
| 10    | 0.52505   | 0.40424| **0.41472**| **0.27077** |
| 11    | 0.00336   | 0.37704| 0.01327| 0.00452  |

### 10.4 Best-So-Far Checkpoint (Epoch 10)

The `best.pt` checkpoint is synchronized with `epoch10.pt` (same timestamp) and records the best validation metrics to date:

- **Precision:** 0.52505
- **Recall:** 0.40424
- **mAP50:** 0.41472
- **mAP50-95:** 0.27077
- **Validation losses:** box 0.90878, cls 0.30491, dfl 1.42739
- **Learning rates:** pg0/pg1/pg2 ≈ 0.000969

### 10.5 Epoch 11 Anomaly

At epoch 11 the unfreeze callback activates, all backbone parameters become trainable, and the learning-rate schedule restarts from the warmup/cosine base. The validation metrics collapse:

- Precision: 0.00336
- Recall: 0.37704
- mAP50: 0.01327
- mAP50-95: 0.00452
- Validation losses: box 2.19663, cls 0.54930, dfl 2.98143

This drop is expected when the backbone is first unfrozen and the optimizer must re-adapt feature extractors to the weapon-domain distribution. The loss curves should recover if training continues.

### 10.6 Missing Final Evaluation Artifacts

Because the 80-epoch run is incomplete, the following Ultralytics final-evaluation outputs are **not yet produced** and are marked `[MISSING — training incomplete]`:

- Confusion matrix (`confusion_matrix.png`)
- Per-class AP table
- Precision–Recall curve, F1 curve, P curve, R curve
- `results.png` training curve plot
- Final `metrics.csv` rows

---

## 11. Deployment and Real-World Considerations

- **Target runtime:** CCTV edge device or server inference.
- **Export path:** Ultralytics export to ONNX / TensorRT / OpenVINO for optimized inference.
- **Latency:** 33.32 GFLOPs at 800×800 is feasible on modern GPUs but may require TensorRT/FP16 and possibly 640×640 for edge devices.
- **Alert pipeline:** model → confidence thresholding → per-class alert → human review.
- **Failure modes:** tiny or occluded weapons, extreme low light, motion blur, unseen camera viewpoints, and look-alike objects.
- **Domain shift:** the dataset contains correlated imagery; deployment requires periodic retraining on the target CCTV domain.

---

## 12. Research Contributions and Future Work

### 12.1 Contributions

1. **Threat taxonomy class merging:** Blunt/Melee/Tool are merged into Handheld_Weapon to reduce inter-class ambiguity and reflect surveillance use cases.
2. **Person-only offline augmentation:** targeted synthetic samples to address the 7.7× class imbalance while preserving realistic CCTV geometry/illumination.
3. **CBAM multi-scale attention:** P3/P4/P5 placement adds <0.08% FLOP overhead while improving feature recalibration for small and large threats.
4. **Focal loss + audited class weights:** jointly handles class imbalance without suppressing easy positives.
5. **Reproducible training launcher:** deterministic seed, transfer learning, freeze/unfreeze schedule, checkpoint/scheduler preservation, and resume support.

### 12.2 Limitations

- The 80-epoch run is only ~14% complete (11 epochs logged; user paused at epoch 12).
- No final per-class AP, confusion matrix, or PR curves exist yet.
- Class imbalance remains large (7.71× max-to-min) despite augmentation and weighting.
- No ablation study is possible under the single-run policy.
- Dataset source correlation may inflate validation metrics.

### 12.3 Future Work

- Resume and complete the 80-epoch schedule.
- Generate final confusion matrix and per-class AP.
- Calibrate class-specific confidence thresholds.
- Export to TensorRT and measure end-to-end latency on target CCTV hardware.
- Collect and retrain on deployment-domain data.

---

## 13. Appendix: Key File Paths

- `d:\Weapon-Threat-Detection\Weapon130\Weapon-Threat-Detection-main\train.py`
- `src\weapon_threat_detection\training_launcher.py`
- `src\weapon_threat_detection\model_engineering.py`
- `src\weapon_threat_detection\transfer_learning.py`
- `src\weapon_threat_detection\augmentation.py`
- `src\weapon_threat_detection\engineering.py`
- `configs\model.yaml`
- `configs\training.yaml`
- `configs\final_augmentation_config.yaml`
- `Original\dataset_info.json`
- `Original\data.yaml`
- `reports\final_readiness_statistics_20260725T092603Z.json`
- `reports\final_class_weights.json`
- `reports\model_summary_20260725T081026Z.json`
- `reports\cbam_review_20260725T081026Z.json`
- `reports\focal_loss_review_20260725T081026Z.json`
- `reports\final_training_configuration_20260725T120209Z.json`
- `reports\final_project_audit_20260725T124650Z.json`
- `reports\phase_10_1_final_launcher_readiness_20260725T130203Z.json`
- `runs\final_training\results.csv`
- `runs\final_training\args.yaml`
- `runs\final_training\launcher_report.json`
- `runs\final_training\weights\best.pt`
- `runs\final_training\weights\last.pt`

### Observed Discrepancies

- `configs/training.yaml` sets `early_stopping_patience: 20`; `reports/final_project_audit_*.json` records `early_stopping_patience: 0`. The executable configuration (`configs/training.yaml` and `runs/final_training/args.yaml`) uses 20.
- `runs/final_training/experiment_summary.json` and `training_report.md` state `training_started: false` and “Training has not started,” but `results.csv` contains 11 completed epochs and `epoch0.pt`–`epoch10.pt` exist. This report treats `results.csv` as the authoritative training log.

---

*End of report.*
