"""Full detector fine-tuning, resumable checkpoints, preflight, and persistent evaluation."""

from dataclasses import dataclass, field

import gc
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from tqdm.auto import tqdm

from fog_prepare import atomic_json
from fog_research import (
    BETA_SCALE,
    CLASSES,
    DensityCNN,
    Detector,
    FogAdapter,
    beta_for,
    coco_metrics,
    digest,
    epoch_indices,
    epoch_size,
    image_tensor,
    seed_all,
    summarize,
)

ENGINE_VERSION = 5
PROFILE_NAMES = ("setup", "partial-2k", "partial-5k", "full")
VALID_EVALUATION_SPLITS = {"val", "test"}
FULL_RUN_IMMUTABLE_KEYS = (
    "epochs",
    "density_epochs",
    "seeds",
    "arms",
    "size",
    "accumulate",
    "batch_size",
    "effective_batch_size",
    "loader",
    "save_every",
    "repeats",
    "learning_rates",
    "adapter_lr",
    "amp",
    "device",
    "density_checkpoint",
    "density_sampling",
    "checkpoints",
)
FULL_RUN_DEFAULTS = {
    "batch_size": 1,
    "effective_batch_size": 4,
    "loader": {
        "num_workers": 0,
        "prefetch_factor": None,
        "pin_memory": False,
    },
    "density_sampling": "natural",
}


def default_config(root):
    """Return the legacy full-study configuration for terminal helpers."""
    root = Path(root).resolve()
    return {
        "output": str(root / "reports/fog/full-training-complete"),
        "manifest": str(root / "data/fog/nuscenes_all.json"),
        "city_manifest": str(root / "data/fog/cityscapes.json"),
        "density_checkpoint": None,
        "density_sampling": "natural",
        "checkpoints": {
            "YOLO11n": str(root / "yolo11n.pt"),
            "YOLO11m": str(root / "yolo11m.pt"),
            "DETR": "facebook/detr-resnet-50",
        },
        "arms": ["rgb_control", "predicted_density"],
        "seeds": [42, 43, 44],
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "size": 640,
        "epochs": 30,
        "density_epochs": 15,
        "batch_size": 1,
        "effective_batch_size": 4,
        "accumulate": 4,
        "loader": FULL_RUN_DEFAULTS["loader"],
        "save_every": 200,
        "repeats": 1,
        "learning_rates": {"YOLO11n": 1e-4, "YOLO11m": 1e-4, "DETR": 1e-5},
        "adapter_lr": 1e-3,
        "amp": True,
        "profile_name": "full",
        "initialization": "existing_full_run",
        "evaluation_split": "test",
        "early_stopping_patience": 0,
    }


def summarize_cumulative_training_progress(progress, epochs, images_per_epoch):
    total_image_steps = epochs * images_per_epoch
    completed_image_steps = min(
        total_image_steps, progress.epoch * images_per_epoch + progress.step
    )
    return {
        "completed_epochs": progress.epoch,
        "planned_epochs": epochs,
        "saved_image_step_in_epoch": progress.step,
        "images_per_epoch": images_per_epoch,
        "completed_image_steps": completed_image_steps,
        "planned_image_steps": total_image_steps,
        "overall_training_percent": 100 * completed_image_steps / total_image_steps
        if total_image_steps
        else 0.0,
    }


def display_checkpoint_continuation(session, epochs, annotations, balanced=False):
    summary = summarize_cumulative_training_progress(
        session.progress, epochs, epoch_size(annotations, balanced)
    )
    print(f"Checkpoint continuation: {session.checkpoint_directory}")
    print(
        f"Overall training: {summary['overall_training_percent']:.2f}% | "
        f"completed epochs {summary['completed_epochs']}/{epochs} | "
        f"saved image steps {summary['saved_image_step_in_epoch']}/{len(annotations)}"
    )
    if session.progress.history:
        print("Cumulative loss history, including previous sessions:")
        print(pd.DataFrame(session.progress.history).to_string(index=False))


def read_run_continuation_summary(configuration):
    output_directory = Path(configuration["output"])
    expected_runs = []
    for seed in configuration["seeds"]:
        if "predicted_density" in configuration["arms"]:
            expected_runs.append((f"density-{seed}", configuration["density_epochs"]))
        for model in configuration["checkpoints"]:
            for arm in configuration["arms"]:
                expected_runs.append((f"{model}-{arm}-{seed}", configuration["epochs"]))
    records = []
    for run_name, epochs in expected_runs:
        directory = output_directory / run_name
        checkpoint_path = directory / "last.pt"
        record = {
            "run": run_name,
            "completed_epochs": 0,
            "planned_epochs": epochs,
            "saved_image_step_in_epoch": 0,
            "overall_training_percent": 0.0,
            "status": "not started",
        }
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            progress = DetectorTrainingProgress(
                **{
                    key: saved[key]
                    for key in (
                        "epoch",
                        "step",
                        "best_loss",
                        "running_sum",
                        "running_count",
                        "history",
                    )
                }
            )
            manifest_path = output_directory / "input-manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                split = (
                    "density_train"
                    if run_name.startswith("density-")
                    else "detector_train"
                )
                record.update(
                    summarize_cumulative_training_progress(
                        progress,
                        epochs,
                        epoch_size(
                            partitions(manifest)[split],
                            split == "density_train"
                            and configuration.get("density_sampling", "natural")
                            == "balanced_bins",
                        ),
                    )
                )
                record["status"] = (
                    "training complete"
                    if progress.epoch >= epochs
                    else "checkpoint saved"
                )
            else:
                # A prior process can leave last.pt before run identity is recorded.
                # Keep the progress table usable and let run_training issue its
                # safer "choose a new output folder" error for that orphan.
                record["status"] = "orphan checkpoint (manifest missing)"
        if (directory / "status.json").exists():
            record["status"] = json.loads((directory / "status.json").read_text())[
                "status"
            ]
        records.append(record)
    return pd.DataFrame(records)


def fingerprint(rows):
    # Annotation/identity fingerprint plus image stat records; does not re-read every JPEG.
    value = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        stat = Path(row["image"]).stat()
        value.update(
            json.dumps([row, stat.st_size, stat.st_mtime_ns], sort_keys=True).encode()
        )
    return value.hexdigest()


def partitions(rows, config=None):
    parts = {
        name: [
            r
            for r in rows
            if r["split"] == name
            and (name == "density_train" or r.get("detection_labeled", True))
            and (name not in ("detector_train", "val") or not any(r.get("iscrowd", [])))
        ]
        for name in ("detector_train", "density_train", "val", "test")
    }
    parts["density_val"] = [
        r for r in rows if r["split"] == "val" and r["beta"] is not None
    ]
    parts["density_test"] = [
        r for r in rows if r["split"] == "test" and r["beta"] is not None
    ]
    if config and config.get("evaluation_split") == "val":
        parts["evaluation"] = parts["val"]
        parts["density_evaluation"] = parts["density_val"]
    else:
        parts["evaluation"] = parts["test"]
        parts["density_evaluation"] = parts["density_test"]
    return parts


def preflight(config, rows):
    """Return actionable checks without starting training or downloading anything."""
    parts = partitions(rows, config)
    checks = []

    def check(name, good, detail):
        checks.append({"check": name, "ready": bool(good), "detail": detail})

    check(
        "training images",
        bool(parts["detector_train"]),
        f"{len(parts['detector_train'])} labeled images",
    )
    check(
        "tuning images",
        bool(parts["val"]),
        f"{len(parts['val'])} scene-held-out images for checkpoint selection",
    )
    evaluation_split = config.get("evaluation_split", "test")
    check(
        "evaluation split",
        evaluation_split in VALID_EVALUATION_SPLITS,
        evaluation_split,
    )
    check(
        "evaluation images",
        bool(parts["evaluation"]),
        f"{len(parts['evaluation'])} {evaluation_split} images for metrics",
    )
    valid_arms = {"rgb_control", "predicted_density", "oracle_density"}
    check(
        "arms",
        bool(config["arms"])
        and len(set(config["arms"])) == len(config["arms"])
        and set(config["arms"]) <= valid_arms,
        ", ".join(config["arms"]),
    )
    groups = {}
    clean = True
    ids = set()
    for row in rows:
        key = (row["dataset"], row["group"])
        clean &= key not in groups or groups[key] == row["split"]
        clean &= row["id"] not in ids
        groups[key] = row["split"]
        ids.add(row["id"])
    check(
        "scene/ID isolation",
        clean,
        "No scene crosses train, tuning, density-training or final-test splits",
    )
    for key in (
        "size",
        "epochs",
        "density_epochs",
        "accumulate",
        "save_every",
        "repeats",
    ):
        check(key, isinstance(config[key], int) and config[key] > 0, str(config[key]))
    check(
        "size alignment", config["size"] % 32 == 0, "Image size must be divisible by 32"
    )
    check(
        "seeds",
        bool(config["seeds"]) and len(set(config["seeds"])) == len(config["seeds"]),
        str(config["seeds"]),
    )
    profile = config.get("profile_name", "full")
    check("profile", profile in PROFILE_NAMES, profile)
    check(
        "initialization",
        config.get("initialization", "pretrained")
        in {"pretrained", "existing_full_run"},
        config.get("initialization", "pretrained"),
    )
    check(
        "batch contract",
        config.get("batch_size", 1) == 1
        and config.get("effective_batch_size", config.get("accumulate", 1))
        == config.get("accumulate", 1),
        "real detector training uses batch_size=1 with gradient accumulation",
    )
    density_sampling = config.get("density_sampling", "natural")
    check(
        "density sampling",
        density_sampling in {"natural", "balanced_bins"},
        density_sampling,
    )
    coverage_path = Path(config["manifest"]).with_suffix(".coverage.json")
    if config.get("test_scope", {}).get("enabled"):
        check(
            "subset test scope",
            bool(rows),
            "Small-data execution test; not full-study coverage or scientific results",
        )
    elif any(r["dataset"] == "nuscenes" for r in rows):
        coverage = (
            json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        )
        complete = coverage.get("complete") is True
        check(
            "nuScenes coverage",
            complete,
            f"complete={complete}; coverage report: {coverage_path}. All expected camera keyframes are required.",
        )
    if "predicted_density" in config["arms"]:
        source = config.get("density_checkpoint")
        if source:
            try:
                saved = torch.load(source, map_location="cpu", weights_only=True)
                correct = (
                    saved.get("kind") == "fog-density-v1"
                    and saved.get("beta_scale") == BETA_SCALE
                )
                trained_groups = {tuple(x) for x in saved.get("train_groups", [])}
                held_out = {
                    (r["dataset"], r["group"])
                    for r in rows
                    if r["split"] in ("val", "test", "detector_train")
                }
                correct &= bool(trained_groups) and not (trained_groups & held_out)
                check(
                    "density checkpoint",
                    correct,
                    "Requires this pipeline format, beta units, and independent training groups",
                )
            except Exception as error:
                check("density checkpoint", False, f"{type(error).__name__}: {error}")
        else:
            train = parts["density_train"]
            known_val = parts["density_val"]
            check(
                "density supervision",
                bool(train)
                and all(r["beta"] is not None for r in train)
                and len({r["beta"] for r in train}) > 1
                and bool(known_val),
                "Add Foggy Cityscapes images (beta comes from filenames; gtFine is not required), prepare its manifest, "
                "or set density_checkpoint to an independently trained fog-density checkpoint. "
                "Native nuScenes has no beta labels; its visibility field is NOT fog density.",
            )
    if "oracle_density" in config["arms"]:
        check(
            "oracle labels",
            all(
                r["beta"] is not None
                for r in parts["detector_train"] + parts["val"] + parts["test"]
            ),
            "Oracle is optional and requires known beta for every matched train/validation/test image. "
            "Use the default rgb_control/predicted_density comparison for native nuScenes.",
        )
    check(
        "models",
        bool(config["checkpoints"])
        and set(config["checkpoints"]) <= {"YOLO11n", "YOLO11m", "DETR"},
        "Three supported pretrained architectures; missing public weights download on first load",
    )
    check(
        "device",
        not config["device"].startswith("cuda") or torch.cuda.is_available(),
        config["device"],
    )
    return pd.DataFrame(checks)


def save_torch(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass
class DensityTrainingSession:
    model: object
    optimizer: object
    dataset_partitions: dict
    configuration: dict
    checkpoint_directory: Path
    seed: int
    progress: object


def restore_density_training_checkpoint(session):
    checkpoint_path = session.checkpoint_directory / "last.pt"
    if not checkpoint_path.exists():
        return
    saved_checkpoint = torch.load(
        checkpoint_path, map_location=session.configuration["device"], weights_only=True
    )
    session.model.load_state_dict(saved_checkpoint["model"])
    session.optimizer.load_state_dict(saved_checkpoint["optimizer"])
    session.progress = DetectorTrainingProgress(
        **{
            name: saved_checkpoint[name]
            for name in (
                "epoch",
                "step",
                "best_loss",
                "running_sum",
                "running_count",
                "history",
            )
        }
    )

    display_checkpoint_continuation(
        session,
        session.configuration["density_epochs"],
        session.dataset_partitions["density_train"],
        balanced=session.configuration.get("density_sampling") == "balanced_bins",
    )


def save_density_training_checkpoint(session, improved=False):
    training_groups = sorted(
        {
            (annotation["dataset"], annotation["group"])
            for annotation in session.dataset_partitions["density_train"]
        }
    )
    payload = {
        "kind": "fog-density-v1",
        "beta_scale": BETA_SCALE,
        "model": session.model.state_dict(),
        "optimizer": session.optimizer.state_dict(),
        **vars(session.progress),
        "train_groups": training_groups,
    }
    if improved:
        save_torch(session.checkpoint_directory / "best.pt", payload)
    save_torch(session.checkpoint_directory / "last.pt", payload)


def train_one_density_epoch(session):
    configuration = session.configuration
    progress = session.progress
    training_annotations = session.dataset_partitions["density_train"]
    image_order = epoch_indices(
        training_annotations,
        session.seed,
        progress.epoch,
        balanced=configuration.get("density_sampling", "natural") == "balanced_bins",
    )
    session.model.train()
    description = f"Density seed={session.seed} epoch={progress.epoch + 1}"
    for image_step in tqdm(
        range(progress.step, len(image_order)),
        total=len(image_order),
        initial=progress.step,
        desc=description,
        mininterval=2,
    ):
        annotation = training_annotations[image_order[image_step]]
        image = image_tensor(annotation, configuration["size"], configuration["device"])
        target_density = image.new_tensor([[annotation["beta"]]])
        session.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            session.model(image) / BETA_SCALE, target_density / BETA_SCALE
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite density loss")
        loss.backward()
        session.optimizer.step()
        progress.running_sum += float(loss.detach())
        progress.running_count += 1
        progress.step = image_step + 1
        if progress.step % configuration["save_every"] == 0:
            save_density_training_checkpoint(session)


def measure_density_validation_loss(session):
    session.model.eval()
    squared_errors = []
    with torch.inference_mode():
        for annotation in tqdm(
            session.dataset_partitions["density_val"],
            desc="Density tuning",
            mininterval=2,
        ):
            image = image_tensor(
                annotation,
                session.configuration["size"],
                session.configuration["device"],
            )
            predicted_density = session.model(image).item()
            squared_errors.append((predicted_density - annotation["beta"]) ** 2)
    return float(np.mean(squared_errors))


def record_completed_density_epoch(session, validation_loss):
    progress = session.progress
    progress.history.append(
        {
            "epoch": progress.epoch + 1,
            "train_normalized_mse": progress.running_sum / progress.running_count,
            "val_beta_rmse": validation_loss**0.5,
        }
    )
    improved = validation_loss < progress.best_loss
    progress.best_loss = min(progress.best_loss, validation_loss)
    progress.epoch += 1
    progress.step = 0
    progress.running_sum, progress.running_count = 0.0, 0
    save_density_training_checkpoint(session, improved)
    pd.DataFrame(progress.history).to_csv(
        session.checkpoint_directory / "history.csv", index=False
    )


def should_stop_early(history, metric, patience):
    if not isinstance(patience, int) or patience <= 0 or len(history) <= patience:
        return False
    prior_best = min(row[metric] for row in history[:-patience])
    return all(row[metric] >= prior_best for row in history[-patience:])


def fit_density(parts, config, directory, seed):
    """Restore the density estimator, train each epoch and select its best weights."""
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = DensityCNN().to(config["device"])
    source_checkpoint = config.get("density_checkpoint")
    if source_checkpoint:
        saved_checkpoint = torch.load(
            source_checkpoint, map_location=config["device"], weights_only=True
        )
        model.load_state_dict(saved_checkpoint["model"])
        return model.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    session = DensityTrainingSession(
        model, optimizer, parts, config, directory, seed, DetectorTrainingProgress()
    )
    restore_density_training_checkpoint(session)
    while session.progress.epoch < config["density_epochs"]:
        train_one_density_epoch(session)
        validation_loss = measure_density_validation_loss(session)
        record_completed_density_epoch(session, validation_loss)
        if should_stop_early(
            session.progress.history,
            "val_beta_rmse",
            config.get("early_stopping_patience", 0),
        ):
            progress_path = directory / "progress.json"
            atomic_json(
                progress_path,
                {"epoch": session.progress.epoch, "reason": "early_stopping"},
            )
            break
    best_checkpoint = torch.load(
        directory / "best.pt", map_location=config["device"], weights_only=True
    )
    model.load_state_dict(best_checkpoint["model"])
    return model.eval().requires_grad_(False)


def preflight_density_model(config, rows):
    """Check only beta supervision, checkpoint and prediction inputs."""
    parts = partitions(rows, config)
    checks = []
    mode = config["density_mode"]
    if mode == "train":
        training_rows = parts["density_train"]
        tuning_rows = parts["density_val"]
        enough_beta_values = len({row["beta"] for row in training_rows}) > 1
        append_readiness_check(
            checks,
            "labeled density training images",
            bool(training_rows)
            and all(row["beta"] is not None for row in training_rows)
            and enough_beta_values,
            f"{len(training_rows)} images with multiple known beta values",
        )
        append_readiness_check(
            checks,
            "held-out density check",
            bool(tuning_rows),
            f"{len(tuning_rows)} labeled images in the profile evaluation split",
        )
    checkpoint = config.get("density_checkpoint")
    if mode == "load":
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            compatible = (
                saved.get("kind") == "fog-density-v1"
                and saved.get("beta_scale") == BETA_SCALE
            )
            append_readiness_check(
                checks,
                "saved density checkpoint",
                compatible,
                "Checkpoint architecture and beta units match this notebook."
                if compatible
                else "Checkpoint type or beta units do not match this notebook.",
            )
            training_groups = {tuple(group) for group in saved.get("train_groups", [])}
            held_out_groups = {
                (row["dataset"], row["group"]) for row in parts["density_evaluation"]
            }
            no_leakage = bool(training_groups) and not (
                training_groups & held_out_groups
            )
            append_readiness_check(
                checks,
                "held-out scene separation",
                no_leakage,
                "The saved model did not train on the images used for its beta check."
                if no_leakage
                else "Checkpoint training scenes overlap the selected beta-check images, or are unrecorded.",
            )
        except Exception as error:
            append_readiness_check(
                checks,
                "saved density checkpoint",
                False,
                f"{type(error).__name__}: {error}",
            )
    prediction_split = config["density_prediction_split"]
    prediction_rows = [
        row
        for row in rows
        if prediction_split == "all" or row["split"] == prediction_split
    ]
    append_readiness_check(
        checks,
        "prediction images",
        bool(prediction_rows),
        f"{len(prediction_rows)} images selected from {prediction_split}",
    )
    append_readiness_check(
        checks,
        "device",
        not config["device"].startswith("cuda") or torch.cuda.is_available(),
        config["device"],
    )
    return pd.DataFrame(checks)


def append_readiness_check(checks, name, ready, detail):
    checks.append({"check": name, "ready": bool(ready), "detail": detail})


def run_density_model_only(rows, config):
    """Train or load one density model, predict beta and save held-out errors."""
    checks = preflight_density_model(config, rows)
    if not checks.ready.all():
        failures = checks.loc[~checks.ready, ["check", "detail"]].to_string(index=False)
        raise RuntimeError("Fog density setup failed:\n" + failures)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        validate_and_record_run_identity(rows, config, output)
        parts = partitions(rows, config)
        checkpoint_directory = output / f"density-{config['density_seed']}"
        model = fit_density(parts, config, checkpoint_directory, config["density_seed"])
        selected = [
            row
            for row in rows
            if config["density_prediction_split"] == "all"
            or row["split"] == config["density_prediction_split"]
        ]
        predictions = predict_density_rows(model, selected, config)
        quality = predict_density_quality(model, parts, config)
        predictions.to_csv(output / "fog-density-predictions.csv", index=False)
        quality.to_csv(output / "density-quality.csv", index=False)
        atomic_json(
            output / "density-model-summary.json",
            {
                "mode": config["density_mode"],
                "checkpoint": str(checkpoint_directory / "best.pt")
                if config["density_mode"] == "train"
                else config["density_checkpoint"],
                "beta_unit": "m^-1",
                "prediction_split": config["density_prediction_split"],
                "prediction_images": len(predictions),
                "quality_images": int(quality["images"].sum())
                if not quality.empty
                else 0,
            },
        )
    return predictions, quality


def predict_density_rows(model, rows, config):
    records = []
    model.eval()
    with torch.inference_mode():
        for row in tqdm(rows, desc="Estimate fog density", mininterval=2):
            image = image_tensor(row, config["size"], config["device"])
            records.append(
                {
                    "id": row["id"],
                    "image": row["image"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "target_beta": row["beta"],
                    "predicted_beta": float(model(image).item()),
                }
            )
    return pd.DataFrame(records)


def predict_density_quality(model, parts, config):
    checked_rows = parts["density_evaluation"]
    predictions = predict_density_rows(model, checked_rows, config)
    known = predictions[predictions.target_beta.notna()].copy()
    if known.empty:
        return pd.DataFrame(columns=["dataset", "images", "MAE_beta", "RMSE_beta"])
    known["absolute_error"] = (known.predicted_beta - known.target_beta).abs()
    known["squared_error"] = (known.predicted_beta - known.target_beta) ** 2
    summary = known.groupby("dataset").agg(
        images=("id", "size"),
        MAE_beta=("absolute_error", "mean"),
        RMSE_squared_error=("squared_error", "mean"),
    )
    summary["RMSE_beta"] = np.sqrt(summary.pop("RMSE_squared_error"))
    return summary.reset_index()


@dataclass
class DetectorTrainingProgress:
    """Fields persisted in existing detector checkpoints; their names are stable."""

    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    running_sum: float = 0.0
    running_count: int = 0
    history: list = field(default_factory=list)


@dataclass
class DetectorTrainingSession:
    detector: object
    adapter: object
    density_estimator: object
    dataset_partitions: dict
    comparison_arm: str
    configuration: dict
    checkpoint_directory: Path
    seed: int
    optimizer: object
    gradient_scaler: object
    detector_parameters: list
    mixed_precision_enabled: bool
    progress: DetectorTrainingProgress = field(default_factory=DetectorTrainingProgress)


def create_detector_training_session(
    detector, adapter, density, parts, arm, config, directory, seed
):
    detector_parameters = [
        parameter
        for parameter in detector.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": detector_parameters,
                "lr": config["learning_rates"][detector.name],
            },
            {"params": adapter.parameters(), "lr": config["adapter_lr"]},
        ],
        weight_decay=1e-4,
    )
    mixed_precision_enabled = config["amp"] and config["device"].startswith("cuda")
    gradient_scaler = torch.amp.GradScaler(
        "cuda", enabled=mixed_precision_enabled, init_scale=128.0
    )
    return DetectorTrainingSession(
        detector,
        adapter,
        density,
        parts,
        arm,
        config,
        directory,
        seed,
        optimizer,
        gradient_scaler,
        detector_parameters,
        mixed_precision_enabled,
    )


def restore_detector_training_checkpoint(session):
    checkpoint_path = session.checkpoint_directory / "last.pt"
    if not checkpoint_path.exists():
        return
    saved_checkpoint = torch.load(
        checkpoint_path, map_location=session.configuration["device"], weights_only=True
    )
    session.detector.model.load_state_dict(saved_checkpoint["model"])
    session.adapter.load_state_dict(saved_checkpoint["adapter"])
    session.optimizer.load_state_dict(saved_checkpoint["optimizer"])
    session.gradient_scaler.load_state_dict(saved_checkpoint["scaler"])
    session.progress = DetectorTrainingProgress(
        **{
            name: saved_checkpoint[name]
            for name in (
                "epoch",
                "step",
                "best_loss",
                "running_sum",
                "running_count",
                "history",
            )
        }
    )
    torch.set_rng_state(saved_checkpoint["rng"].cpu())
    if session.configuration["device"].startswith("cuda"):
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in saved_checkpoint["cuda_rng"]]
        )

    display_checkpoint_continuation(
        session,
        session.configuration["epochs"],
        session.dataset_partitions["detector_train"],
    )


def save_detector_training_checkpoint(session, improved=False, reason="periodic"):
    progress = session.progress
    payload = {
        "kind": "fog-detector-v1",
        "model": session.detector.model.state_dict(),
        "adapter": session.adapter.state_dict(),
        "optimizer": session.optimizer.state_dict(),
        "scaler": session.gradient_scaler.state_dict(),
        **vars(progress),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if session.configuration["device"].startswith("cuda")
        else [],
        "checkpoint_reason": reason,
    }
    if improved:
        save_torch(session.checkpoint_directory / "best.pt", payload)
    save_torch(session.checkpoint_directory / "last.pt", payload)
    atomic_json(
        session.checkpoint_directory / "progress.json",
        {
            "model": session.detector.name,
            "arm": session.comparison_arm,
            "seed": session.seed,
            "epoch": progress.epoch,
            "step": progress.step,
            "best_loss": float(progress.best_loss)
            if np.isfinite(progress.best_loss)
            else None,
            "reason": reason,
            **summarize_cumulative_training_progress(
                progress,
                session.configuration["epochs"],
                len(session.dataset_partitions["detector_train"]),
            ),
        },
    )


def calculate_finite_detector_loss(session, image, annotation, fog_density):
    with torch.autocast(
        device_type=image.device.type, enabled=session.mixed_precision_enabled
    ):
        loss = session.detector.loss(session.adapter(image, fog_density), annotation)
    if torch.isfinite(loss):
        return loss
    if session.mixed_precision_enabled:
        # Some examples overflow in float16; retry the same loss in float32.
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss = session.detector.loss(
                session.adapter(image.float(), fog_density.float()), annotation
            )
        if torch.isfinite(loss):
            return loss
    raise ValueError(
        f"Nonfinite loss for {annotation['id']}; AMP fallback was also nonfinite"
    )


def apply_accumulated_detector_gradients(session):
    session.gradient_scaler.unscale_(session.optimizer)
    torch.nn.utils.clip_grad_norm_(
        session.detector_parameters + list(session.adapter.parameters()), 10
    )
    session.gradient_scaler.step(session.optimizer)
    session.gradient_scaler.update()
    session.optimizer.zero_grad(set_to_none=True)


def capture_detector_random_state(configuration):
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if configuration["device"].startswith("cuda")
        else []
    )
    return cpu_state, cuda_states


def restore_detector_random_state(configuration, random_state):
    cpu_state, cuda_states = random_state
    torch.set_rng_state(cpu_state)
    if configuration["device"].startswith("cuda"):
        torch.cuda.set_rng_state_all(cuda_states)


def accumulate_one_detector_image(session, annotation, accumulation_window_size):
    configuration = session.configuration
    image = image_tensor(annotation, configuration["size"], configuration["device"])
    fog_density = beta_for(
        session.comparison_arm, image, annotation, session.density_estimator
    )
    loss = calculate_finite_detector_loss(session, image, annotation, fog_density)
    session.gradient_scaler.scale(loss / accumulation_window_size).backward()
    session.progress.running_sum += float(loss.detach())
    session.progress.running_count += 1


def checkpoint_completed_optimizer_update(session, next_image_step, image_count):
    apply_accumulated_detector_gradients(session)
    session.progress.step = next_image_step
    completed_window_count = next_image_step // session.configuration["accumulate"]
    checkpoint_due = completed_window_count % session.configuration["save_every"] == 0
    if checkpoint_due or next_image_step == image_count:
        save_detector_training_checkpoint(session)


def train_detector_microsteps(session, image_order):
    configuration = session.configuration
    progress = session.progress
    completed_loss_statistics = progress.running_sum, progress.running_count
    completed_random_state = capture_detector_random_state(configuration)
    description = f"{session.detector.name}/{session.comparison_arm} seed={session.seed} epoch={progress.epoch + 1}"
    image_steps = tqdm(
        range(progress.step, len(image_order)),
        total=len(image_order),
        initial=progress.step,
        desc=description,
        mininterval=2,
    )
    try:
        for image_step in image_steps:
            annotation = session.dataset_partitions["detector_train"][
                image_order[image_step]
            ]
            window_start = (image_step // configuration["accumulate"]) * configuration[
                "accumulate"
            ]
            window_size = min(
                configuration["accumulate"], len(image_order) - window_start
            )
            accumulate_one_detector_image(session, annotation, window_size)
            next_image_step = image_step + 1
            optimizer_boundary = next_image_step % configuration[
                "accumulate"
            ] == 0 or next_image_step == len(image_order)
            if optimizer_boundary:
                checkpoint_completed_optimizer_update(
                    session, next_image_step, len(image_order)
                )
                completed_loss_statistics = progress.running_sum, progress.running_count
                completed_random_state = capture_detector_random_state(configuration)
    except KeyboardInterrupt:
        # Replay the partial window with its original RNG state and loss statistics.
        session.optimizer.zero_grad(set_to_none=True)
        progress.running_sum, progress.running_count = completed_loss_statistics
        restore_detector_random_state(configuration, completed_random_state)
        save_detector_training_checkpoint(session, reason="interrupt")
        raise


def train_one_detector_epoch(session):
    session.detector.training_mode()
    session.adapter.train()
    training_image_count = len(session.dataset_partitions["detector_train"])
    image_order = random.Random(session.seed + session.progress.epoch).sample(
        range(training_image_count), training_image_count
    )
    session.optimizer.zero_grad(set_to_none=True)
    train_detector_microsteps(session, image_order)


def measure_detector_validation_loss(session):
    session.detector.model.eval()
    session.adapter.eval()
    validation_losses = []
    description = f"{session.detector.name}/{session.comparison_arm} tuning"
    with torch.no_grad():
        for annotation in tqdm(
            session.dataset_partitions["val"], desc=description, mininterval=2
        ):
            image = image_tensor(
                annotation,
                session.configuration["size"],
                session.configuration["device"],
            )
            fog_density = beta_for(
                session.comparison_arm, image, annotation, session.density_estimator
            )
            loss = calculate_finite_detector_loss(
                session, image, annotation, fog_density
            )
            validation_losses.append(float(loss))
    validation_loss = float(np.mean(validation_losses))
    if not np.isfinite(validation_loss):
        raise ValueError("Nonfinite validation loss")
    return validation_loss


def record_completed_detector_epoch(session, validation_loss):
    progress = session.progress
    progress.history.append(
        {
            "epoch": progress.epoch + 1,
            "train_loss": progress.running_sum / progress.running_count,
            "val_loss": validation_loss,
        }
    )
    improved = validation_loss < progress.best_loss
    progress.best_loss = min(progress.best_loss, validation_loss)
    progress.epoch += 1
    progress.step = 0
    progress.running_sum, progress.running_count = 0.0, 0
    save_detector_training_checkpoint(session, improved, reason="epoch")
    pd.DataFrame(progress.history).to_csv(
        session.checkpoint_directory / "history.csv", index=False
    )


def load_best_detector_weights(session):
    saved_checkpoint = torch.load(
        session.checkpoint_directory / "best.pt",
        map_location=session.configuration["device"],
        weights_only=True,
    )
    session.detector.model.load_state_dict(saved_checkpoint["model"])
    session.adapter.load_state_dict(saved_checkpoint["adapter"])
    session.detector.model.eval()
    session.adapter.eval()


def fit_detector(detector, adapter, density, parts, arm, config, directory, seed):
    """Resume, train, validate and checkpoint one detector comparison."""
    session = create_detector_training_session(
        detector, adapter, density, parts, arm, config, directory, seed
    )
    restore_detector_training_checkpoint(session)
    while session.progress.epoch < config["epochs"]:
        train_one_detector_epoch(session)
        validation_loss = measure_detector_validation_loss(session)
        record_completed_detector_epoch(session, validation_loss)
        if should_stop_early(
            session.progress.history,
            "val_loss",
            config.get("early_stopping_patience", 0),
        ):
            progress_path = directory / "progress.json"
            saved_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            saved_progress["reason"] = "early_stopping"
            atomic_json(progress_path, saved_progress)
            break
    load_best_detector_weights(session)


def read_completed_prediction_records(predictions_path, ordered_annotations):
    completed_predictions = []
    if not predictions_path.exists():
        return completed_predictions
    with predictions_path.open("rb+") as stream:
        complete_record_end = 0
        while record_line := stream.readline():
            if not record_line.endswith(b"\n"):
                stream.truncate(complete_record_end)
                break
            completed_predictions.append(json.loads(record_line))
            complete_record_end = stream.tell()
    completed_ids = [prediction["id"] for prediction in completed_predictions]
    expected_ids = [
        annotation["id"]
        for annotation in ordered_annotations[: len(completed_predictions)]
    ]
    if completed_ids != expected_ids:
        raise ValueError(
            "Saved predictions do not match input order; use a new run directory"
        )
    return completed_predictions


def synchronize_inference_device(configuration):
    if configuration["device"].startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def predict_one_image_with_timing(
    detector, adapter, density_estimator, comparison_arm, annotation, configuration
):
    import time

    synchronize_inference_device(configuration)
    start_time = time.perf_counter()
    image = image_tensor(annotation, configuration["size"], configuration["device"])
    synchronize_inference_device(configuration)
    before_density_time = time.perf_counter()
    fog_density = beta_for(comparison_arm, image, annotation, density_estimator)
    synchronize_inference_device(configuration)
    after_density_time = time.perf_counter()
    predictions = detector.predict(adapter(image, fog_density), annotation)
    synchronize_inference_device(configuration)
    end_time = time.perf_counter()
    timing = {
        "total_ms": (end_time - start_time) * 1000,
        "density_ms": (after_density_time - before_density_time) * 1000,
        "detector_adapter_ms": (end_time - after_density_time) * 1000,
    }
    return predictions, float(fog_density.item()), timing


def evaluate_resumable(detector, adapter, density, arm, rows, config, path):
    """Keep valid prediction records, then evaluate only the remaining images."""
    completed_predictions = read_completed_prediction_records(path, rows)
    remaining_annotations = rows[len(completed_predictions) :]
    if remaining_annotations:
        for warmup_index in range(3):
            predict_one_image_with_timing(
                detector, adapter, density, arm, remaining_annotations[0], config
            )
    with path.open("a", encoding="utf-8") as stream:
        for annotation in tqdm(
            remaining_annotations, desc=f"Evaluate {detector.name}/{arm}", mininterval=2
        ):
            timings = []
            for repeat_index in range(config["repeats"]):
                predictions, fog_density, timing = predict_one_image_with_timing(
                    detector, adapter, density, arm, annotation, config
                )
                timings.append(timing)
            prediction_record = {
                "id": annotation["id"],
                "prediction": predictions,
                "beta_input": fog_density,
                "timings": timings,
            }
            stream.write(json.dumps(prediction_record) + "\n")
            stream.flush()
            completed_predictions.append(prediction_record)
    return completed_predictions


def export_trained_model(detector, adapter, directory, identity, config):
    """Write a compact inference artifact without optimizer, scaler or RNG state."""
    payload = {
        "kind": "fog-detector-export-v1",
        **identity,
        "checkpoint_source": config["checkpoints"][detector.name],
        "input_size": config["size"],
        "classes": list(CLASSES),
        "model": {
            key: value.detach().cpu()
            for key, value in detector.model.state_dict().items()
        },
        "adapter": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        },
    }
    save_torch(directory / "trained-model.pt", payload)


def export_saved_checkpoint(name, directory, identity, config):
    """Create the model-only artifact for a run completed by an older engine."""
    checkpoint = directory / "best.pt"
    if not checkpoint.exists() or (directory / "trained-model.pt").exists():
        return
    detector = Detector(
        name, config["checkpoints"][name], config["device"], trainable=False
    )
    adapter = FogAdapter().to(config["device"])
    saved = torch.load(checkpoint, map_location=config["device"], weights_only=True)
    detector.model.load_state_dict(saved["model"])
    adapter.load_state_dict(saved["adapter"])
    export_trained_model(detector, adapter, directory, identity, config)
    del detector, adapter


def read_detector_stop_metadata(directory, configuration):
    progress_path = directory / "progress.json"
    if not progress_path.exists():
        return {"epochs_trained": 0, "stop_reason": "unknown"}
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    reason = progress.get("reason", "unknown")
    if reason == "epoch" and progress.get("epoch", 0) >= configuration["epochs"]:
        reason = "epoch_budget"
    return {"epochs_trained": progress.get("epoch", 0), "stop_reason": reason}


def build_run_signature(rows, configuration):
    selected_ids = sorted(row["id"] for row in rows)
    selection_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    return {
        "engine": ENGINE_VERSION,
        "config": configuration,
        "inputs": fingerprint(rows),
        "profile": configuration.get("profile_name", "full"),
        "selection": {
            "schema": 1,
            "ids": selected_ids,
            "ids_sha256": selection_hash,
            "evaluation_split": configuration.get("evaluation_split", "test"),
        },
        "density_sha256": digest(configuration["density_checkpoint"])
        if configuration.get("density_checkpoint")
        else None,
        "source_sha256": {
            path.name: digest(path) for path in Path(__file__).parent.glob("fog_*.py")
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "ultralytics", "transformers", "pycocotools")
        },
    }


def describe_run_identity_changes(previous, current):
    """Explain rejected continuation without dumping the selected image list."""
    changes = []
    for section in ("config", "versions"):
        before, after = previous.get(section, {}), current.get(section, {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                if isinstance(before.get(name), dict) and isinstance(
                    after.get(name), dict
                ):
                    fields = sorted(
                        key
                        for key in set(before[name]) | set(after[name])
                        if before[name].get(key) != after[name].get(key)
                    )
                    changes.append(
                        f"{section}.{name}: changed fields {', '.join(fields)}"
                    )
                else:
                    changes.append(
                        f"{section}.{name}: saved={str(before.get(name, 'not recorded'))[:120]}; requested={str(after.get(name, 'not recorded'))[:120]}"
                    )
    for name in ("engine", "density_sha256"):
        if previous.get(name) != current.get(name):
            changes.append(
                f"{name}: saved={previous.get(name)}; requested={current.get(name)}"
            )
    return "\n".join(changes[:15]) or "Checkpoint engine compatibility differs."


def validate_and_record_run_identity(rows, configuration, output_directory):
    signature = build_run_signature(rows, configuration)
    settings_path = output_directory / "settings.json"
    has_checkpoints = any(output_directory.glob("*/last.pt"))
    if has_checkpoints and not settings_path.exists():
        if configuration.get("profile_name") != "full":
            raise ValueError(
                "Existing checkpoints have no profile identity; choose a new output folder."
            )
    if settings_path.exists():
        previous_signature = json.loads(settings_path.read_text())
        if previous_signature != signature:
            immutable_fields = ("config", "inputs", "density_sha256", "versions")
            compatible = all(
                previous_signature.get(name) == signature.get(name)
                for name in immutable_fields
            )
            is_existing_full = (
                configuration.get("profile_name") == "full"
                and configuration.get("initialization") == "existing_full_run"
            )
            previous_config = previous_signature.get("config", {})
            full_contract_matches = all(
                previous_config.get(key, FULL_RUN_DEFAULTS.get(key))
                == configuration.get(key, FULL_RUN_DEFAULTS.get(key))
                for key in FULL_RUN_IMMUTABLE_KEYS
            )
            if (
                is_existing_full
                and full_contract_matches
                and previous_signature.get("inputs") == signature.get("inputs")
            ):
                print(
                    "Continuing existing full-run checkpoints with recorded settings.",
                    flush=True,
                )
            elif compatible and previous_signature.get("engine") in (
                ENGINE_VERSION - 1,
                ENGINE_VERSION,
            ):
                print(
                    "Migrating compatible checkpoint metadata to the current resumable engine.",
                    flush=True,
                )
            elif previous_signature.get("inputs") != signature.get("inputs"):
                raise ValueError(
                    "The input manifest changed since this run was started. The saved checkpoint belongs to a different dataset version; use a new output folder for the complete-data run. The old checkpoint is preserved."
                )
            else:
                raise ValueError(
                    "Run configuration or package versions changed:\n"
                    + describe_run_identity_changes(previous_signature, signature)
                    + "\nResume with the saved training settings and compatible packages. "
                    "Use analyze for saved results, or retrain=True for a separate fresh run. "
                    "Existing checkpoints were preserved."
                )
    atomic_json(settings_path, signature)
    atomic_json(output_directory / "input-manifest.json", rows)
    atomic_json(
        output_directory / "input-selection.json",
        signature["selection"],
    )


def prepare_density_estimator_for_seed(
    dataset_partitions, configuration, output_directory, seed
):
    if "predicted_density" not in configuration["arms"]:
        return None
    density_directory = output_directory / f"density-{seed}"
    density_estimator = fit_density(
        dataset_partitions, configuration, density_directory, seed
    )
    predictions_path = density_directory / "test-predictions.csv"
    if not predictions_path.exists():
        density_predictions = []
        with torch.inference_mode():
            for annotation in tqdm(
                dataset_partitions["density_evaluation"],
                desc="Held-out density quality",
                mininterval=2,
            ):
                image = image_tensor(
                    annotation, configuration["size"], configuration["device"]
                )
                estimate = float(density_estimator(image).item())
                density_predictions.append(
                    {
                        "id": annotation["id"],
                        "dataset": annotation["dataset"],
                        "target_beta": annotation["beta"],
                        "predicted_beta": estimate,
                    }
                )
        pd.DataFrame(
            density_predictions,
            columns=["id", "dataset", "target_beta", "predicted_beta"],
        ).to_csv(predictions_path, index=False)
    return density_estimator


def evaluate_detector_comparison(
    detector,
    adapter,
    density_estimator,
    dataset_partitions,
    configuration,
    directory,
    identity,
    trainable_parameter_count,
):
    summaries = []
    evaluation_annotations = dataset_partitions["evaluation"]
    for dataset_name in sorted(
        {annotation["dataset"] for annotation in evaluation_annotations}
    ):
        test_annotations = sorted(
            [
                annotation
                for annotation in evaluation_annotations
                if annotation["dataset"] == dataset_name
            ],
            key=lambda annotation: annotation["id"],
        )
        uses_cuda = configuration["device"].startswith("cuda")
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats()
        predictions = evaluate_resumable(
            detector,
            adapter,
            density_estimator,
            identity["arm"],
            test_annotations,
            configuration,
            directory / f"{dataset_name}-predictions.jsonl",
        )
        peak_gpu_megabytes = (
            torch.cuda.max_memory_allocated() / 2**20 if uses_cuda else None
        )
        for metrics in summarize(test_annotations, predictions):
            summaries.append(
                {
                    **identity,
                    "dataset": dataset_name,
                    **metrics,
                    "peak_gpu_mb": peak_gpu_megabytes,
                    "trainable_detector_parameters": trainable_parameter_count,
                    "trained_checkpoint_mb": Path(
                        identity.get("source_checkpoint", directory / "best.pt")
                    )
                    .stat()
                    .st_size
                    / 2**20,
                }
            )
        curves = []
        for condition in ("clear", "foggy"):
            indices = [
                index
                for index, row in enumerate(test_annotations)
                if row["beta"] is not None
                and (row["beta"] == 0 if condition == "clear" else row["beta"] > 0)
            ]
            if indices:
                curve = coco_metrics(
                    [test_annotations[index] for index in indices],
                    [predictions[index] for index in indices],
                    include_curve=True,
                )
                curves.append(
                    {
                        **identity,
                        "dataset": dataset_name,
                        "condition": condition,
                        "precision": curve["precision_curve"],
                        "recall": curve["recall_curve"],
                    }
                )
        atomic_json(directory / f"{dataset_name}-precision-recall.json", curves)
    pd.DataFrame(summaries).to_csv(directory / "results.csv", index=False)


def validate_matched_evaluation(rows, source_rows, split, max_pairs=None):
    """Require the same labeled scenes at each fog level, held out from fitting."""
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No matched evaluation images in split {split}.")
    pairs = {}
    for row in selected:
        if not row.get("detection_labeled") or not row.get("annotation_source"):
            raise ValueError("Matched evaluation requires gtFine detection labels.")
        if not Path(row["annotation_source"]).is_file():
            raise FileNotFoundError(row["annotation_source"])
        if row.get("beta") is None or not row.get("pair_id"):
            raise ValueError("Matched evaluation requires known beta and pair_id.")
        pairs.setdefault(row["pair_id"], []).append(row)
    expected = {0.0, 0.005, 0.01, 0.02}
    for pair, variants in pairs.items():
        if len(variants) != len(expected) or {r["beta"] for r in variants} != expected:
            raise ValueError(f"Incomplete or duplicate clear/fog variants: {pair}")
        reference = variants[0]
        for row in variants:
            if any(
                row[key] != reference[key]
                for key in ("boxes", "labels", "iscrowd", "width", "height", "group")
            ):
                raise ValueError(f"Mismatched ground truth: {pair}")
    fitted_groups = {
        (r["dataset"], r["group"])
        for r in source_rows
        if r["split"] in ("detector_train", "density_train")
        or (split == "test" and r["split"] == "val")
    }
    if any((r["dataset"], r["group"]) in fitted_groups for r in selected):
        raise ValueError("Evaluation scenes overlap checkpoint fitting/tuning scenes.")
    if max_pairs is not None:
        if (
            not isinstance(max_pairs, int)
            or isinstance(max_pairs, bool)
            or max_pairs < 1
        ):
            raise ValueError("max_pairs must be a positive integer or None.")
        chosen = set(
            sorted(pairs, key=lambda pair: hashlib.sha256(pair.encode()).hexdigest())[
                :max_pairs
            ]
        )
        selected = [row for row in selected if row["pair_id"] in chosen]
    return sorted(selected, key=lambda row: row["id"])


def run_checkpoint_evaluation(rows, configuration):
    """Evaluate saved best weights; source runs are read-only and never trained."""
    from fog_analytics import export_analytics

    source = Path(configuration["evaluation_source"]).resolve()
    output = Path(configuration["output"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Evaluation output must be separate from the training run.")
    output.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(source / "run.lock"), timeout=0),
        FileLock(str(output / "run.lock"), timeout=0),
    ):
        source_settings = json.loads((source / "settings.json").read_text())
        source_config = source_settings["config"]
        if configuration["size"] != source_config["size"]:
            raise ValueError("Use the source checkpoint's image size for evaluation.")
        source_rows = json.loads((source / "input-manifest.json").read_text())
        evaluation_rows = validate_matched_evaluation(
            rows,
            source_rows,
            configuration["evaluation_split"],
            configuration.get("evaluation_max_pairs"),
        )
        sources = []
        for seed in configuration["seeds"]:
            for model in configuration["checkpoints"]:
                for arm in configuration["arms"]:
                    checkpoint = source / f"{model}-{arm}-{seed}" / "best.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"Missing trained arm: {checkpoint}")
                    if (
                        arm not in source_config["arms"]
                        or seed not in source_config["seeds"]
                    ):
                        raise ValueError("Arm/seed was not trained in the source run.")
                    sources.append((model, arm, seed, checkpoint))
            if "predicted_density" in configuration["arms"]:
                density_path = source / f"density-{seed}" / "best.pt"
                if not density_path.is_file():
                    raise FileNotFoundError(density_path)
        identity_config = dict(
            configuration,
            initialization="pretrained",
            source_training_config=source_config,
            checkpoint_sources={
                str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                for _, _, _, path in sources
            },
            evaluation_scope="matched_cityscapes_domain_shift",
        )
        identity_config["density_sources"] = {
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in (
                source / f"density-{seed}" / "best.pt"
                for seed in configuration["seeds"]
            )
            if path.exists()
        }
        identity_config["annotation_sources"] = {
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in {Path(row["annotation_source"]) for row in evaluation_rows}
        }
        validate_and_record_run_identity(evaluation_rows, identity_config, output)
        for model, arm, seed, checkpoint in sources:
            evaluate_saved_comparison(
                model, arm, seed, checkpoint, evaluation_rows, configuration, output
            )
        return export_analytics(output, evaluation_rows), pd.read_csv(
            output / "status.csv"
        )


def evaluate_saved_comparison(model, arm, seed, checkpoint, rows, config, output):
    directory = output / f"{model}-{arm}-{seed}"
    directory.mkdir(exist_ok=True)
    status_path = directory / "status.json"
    if (
        status_path.exists()
        and json.loads(status_path.read_text())["status"] == "complete"
    ):
        return
    identity = {
        "model": model,
        "arm": arm,
        "seed": seed,
        "profile": config.get("profile_name", "full"),
        "evaluation_split": config["evaluation_split"],
        "source_checkpoint": str(checkpoint),
        "evaluation_scope": "matched_cityscapes_domain_shift",
    }
    detector = adapter = density = None
    try:
        detector = Detector(
            model, config["checkpoints"][model], config["device"], trainable=True
        )
        adapter = FogAdapter().to(config["device"])
        saved = torch.load(checkpoint, map_location=config["device"], weights_only=True)
        source_config = json.loads(
            (checkpoint.parent.parent / "settings.json").read_text()
        )["config"]
        identity["planned_epochs"] = source_config["epochs"]
        identity["best_epoch"] = saved.get("epoch")
        detector.model.load_state_dict(saved["model"])
        adapter.load_state_dict(saved["adapter"])
        detector.model.eval()
        adapter.eval()
        if arm == "predicted_density":
            density = DensityCNN().to(config["device"])
            payload = torch.load(
                Path(config["evaluation_source"]) / f"density-{seed}" / "best.pt",
                map_location=config["device"],
                weights_only=True,
            )
            density.load_state_dict(payload["model"])
            density.eval().requires_grad_(False)
        identity.update(read_detector_stop_metadata(checkpoint.parent, source_config))
        atomic_json(status_path, {**identity, "status": "evaluating"})
        evaluate_detector_comparison(
            detector,
            adapter,
            density,
            {"evaluation": rows},
            config,
            directory,
            identity,
            sum(p.numel() for p in detector.model.parameters()),
        )
        normal_path = checkpoint.parent / "results.csv"
        if normal_path.exists():
            normal = pd.read_csv(normal_path, dtype={"density": str})
            normal = normal[normal.dataset == "nuscenes"].copy()
            normal["evaluation_scope"] = "source_normal_use_fog_unknown"
            measured = pd.read_csv(directory / "results.csv", dtype={"density": str})
            pd.concat([measured, normal], ignore_index=True).to_csv(
                directory / "results.csv", index=False
            )
        atomic_json(status_path, {**identity, "status": "complete"})
    except BaseException as error:
        atomic_json(
            status_path,
            {
                **identity,
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "error",
                "reason": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        del detector, adapter, density
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_detector_comparison(
    model_name,
    comparison_arm,
    seed,
    density_estimator,
    dataset_partitions,
    configuration,
    output_directory,
):
    directory = output_directory / f"{model_name}-{comparison_arm}-{seed}"
    directory.mkdir(exist_ok=True)
    identity = detector_comparison_identity(
        model_name, comparison_arm, seed, configuration
    )
    status_path = directory / "status.json"
    if detector_comparison_is_complete(status_path):
        export_saved_checkpoint(model_name, directory, identity, configuration)
        return json.loads(status_path.read_text())
    detector, adapter = create_detector_comparison_models(
        model_name, seed, configuration
    )
    try:
        run_detector_training_and_evaluation(
            detector,
            adapter,
            density_estimator,
            dataset_partitions,
            configuration,
            directory,
            identity,
            status_path,
        )
    except BaseException as error:
        record_detector_comparison_error(status_path, identity, error)
        raise
    finally:
        release_detector_comparison_models(detector, adapter)
    return json.loads(status_path.read_text())


def detector_comparison_identity(model_name, comparison_arm, seed, configuration):
    return {
        "model": model_name,
        "arm": comparison_arm,
        "seed": seed,
        "profile": configuration.get("profile_name", "full"),
        "evaluation_split": configuration.get("evaluation_split", "test"),
        "planned_epochs": configuration["epochs"],
    }


def detector_comparison_is_complete(status_path):
    return (
        status_path.exists()
        and json.loads(status_path.read_text()).get("status") == "complete"
    )


def create_detector_comparison_models(model_name, seed, configuration):
    seed_all(seed)
    detector = Detector(
        model_name,
        configuration["checkpoints"][model_name],
        configuration["device"],
        trainable=True,
    )
    seed_all(seed)
    adapter = FogAdapter().to(configuration["device"])
    return detector, adapter


def run_detector_training_and_evaluation(
    detector,
    adapter,
    density_estimator,
    dataset_partitions,
    configuration,
    directory,
    identity,
    status_path,
):
    trainable_parameters = record_detector_training_started(
        status_path, identity, detector
    )
    fit_detector(
        detector,
        adapter,
        density_estimator,
        dataset_partitions,
        identity["arm"],
        configuration,
        directory,
        identity["seed"],
    )
    identity.update(read_detector_stop_metadata(directory, configuration))
    export_trained_model(detector, adapter, directory, identity, configuration)
    atomic_json(status_path, {**identity, "status": "evaluating"})
    evaluate_detector_comparison(
        detector,
        adapter,
        density_estimator,
        dataset_partitions,
        configuration,
        directory,
        identity,
        trainable_parameters,
    )
    atomic_json(status_path, {**identity, "status": "complete"})


def record_detector_training_started(status_path, identity, detector):
    trainable_parameters = sum(
        parameter.numel()
        for parameter in detector.model.parameters()
        if parameter.requires_grad
    )
    atomic_json(
        status_path,
        {
            **identity,
            "status": "training",
            "trainable_detector_parameters": trainable_parameters,
        },
    )
    return trainable_parameters


def record_detector_comparison_error(status_path, identity, error):
    status = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
    atomic_json(
        status_path,
        {**identity, "status": status, "reason": f"{type(error).__name__}: {error}"},
    )


def release_detector_comparison_models(detector, adapter):
    del detector, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_training(rows, config, reset_setup=True):
    """Validate identity, resume each comparison, and produce the study reports."""
    require_ready_preflight(config, rows)
    output_directory = Path(config["output"])
    validate_setup_overwrite(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_directory / "run.lock"), timeout=0):
        clear_setup_output_if_requested(config, output_directory, reset_setup)
        validate_and_record_run_identity(rows, config, output_directory)
        dataset_partitions = partitions(rows, config)
        statuses = run_comparison_matrix(dataset_partitions, config, output_directory)
        return export_training_results(output_directory, rows, statuses)


def require_ready_preflight(config, rows):
    checks = preflight(config, rows)
    if checks.ready.all():
        return
    failures = checks.loc[~checks.ready, ["check", "detail"]].to_string(index=False)
    raise RuntimeError("Preflight failed:\n" + failures)


def validate_setup_overwrite(config):
    overwrite = config.get("overwrite_existing", False)
    if overwrite and config.get("profile_name") != "setup":
        raise ValueError("overwrite_existing is only allowed for the setup profile.")


def clear_setup_output_if_requested(config, output_directory, reset_setup):
    if not config.get("overwrite_existing", False) or not reset_setup:
        return
    for child in output_directory.iterdir():
        if child.name in ("run.lock", "subset-selection.json"):
            continue
        if not child.resolve().is_relative_to(output_directory.resolve()):
            raise ValueError("Setup cleanup target escapes its output folder.")
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_comparison_matrix(dataset_partitions, config, output_directory):
    statuses = []
    model_names = list(config["checkpoints"])
    for seed_index, seed in enumerate(config["seeds"]):
        density = prepare_density_estimator_for_seed(
            dataset_partitions, config, output_directory, seed
        )
        model_order = (
            model_names[seed_index % len(model_names) :]
            + model_names[: seed_index % len(model_names)]
        )
        for model_name in model_order:
            for arm in config["arms"]:
                statuses.append(
                    run_detector_comparison(
                        model_name,
                        arm,
                        seed,
                        density,
                        dataset_partitions,
                        config,
                        output_directory,
                    )
                )
    return statuses


def export_training_results(output_directory, rows, statuses):
    from fog_analytics import export_analytics

    return export_analytics(output_directory, rows), pd.DataFrame(statuses)
