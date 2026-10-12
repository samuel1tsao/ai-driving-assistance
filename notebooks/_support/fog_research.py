"""Detector, density-conditioning and scoring primitives for the fog study.

The supported full training entry point is fog_training.run_training.
"""

import contextlib
import hashlib
import io
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

CLASSES = ["person", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
COCO_IDS = [1, 3, 8, 6, 7, 4, 2]
YOLO_IDS = [0, 2, 7, 5, 6, 3, 1]
MODEL_SPECS = {
    "YOLO11n": "yolo11n.pt",
    "YOLO11m": "yolo11m.pt",
    "DETR": "facebook/detr-resnet-50",
}
ARMS = ["rgb_control", "predicted_density", "oracle_density"]
BETA_SCALE = 0.02
DENSITY_BIN_NAMES = ("clear", "light", "moderate", "dense")
DENSITY_BIN_UPPER_BOUNDS = (0.0, 0.005, 0.01, BETA_SCALE)
PROFILE_NAMES = ("setup", "partial-2k", "partial-5k", "full")
PROFILE_SELECTION_SCHEMA = 1


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def density_bin(beta):
    """Map a known beta to a stable, interpretable fog-density stratum."""
    if beta is None:
        return "unknown"
    value = float(beta)
    for name, upper_bound in zip(DENSITY_BIN_NAMES, DENSITY_BIN_UPPER_BOUNDS):
        if value <= upper_bound:
            return name
    raise ValueError(f"Beta is outside the supported range: {value}")


def epoch_indices(rows, seed, epoch, balanced=False):
    """Return a deterministic epoch order, optionally equalizing known beta strata."""
    if not rows:
        return []
    groups = {}
    for index, row in enumerate(rows):
        label = density_bin(row.get("beta")) if balanced else "all"
        groups.setdefault(label, []).append(index)
    generator = random.Random(seed + epoch)
    if not balanced:
        return generator.sample(range(len(rows)), len(rows))
    target_size = max(len(indices) for indices in groups.values())
    order = []
    for label in sorted(groups):
        indices = groups[label]
        order.extend(generator.choices(indices, k=target_size))
    generator.shuffle(order)
    return order


def epoch_size(rows, balanced=False):
    """Return the number of image steps represented by one configured epoch."""
    if not rows or not balanced:
        return len(rows)
    counts = {}
    for row in rows:
        label = density_bin(row.get("beta"))
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) * len(counts)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_module_digest(module_name):
    """Use embedded source fingerprints in Colab and module files in tests."""
    notebook_fingerprints = globals().get("NOTEBOOK_SOURCE_DIGESTS")
    if notebook_fingerprints is not None:
        return notebook_fingerprints[module_name]
    return digest(Path(__file__).with_name(module_name))


def source_module_digests():
    """Return stable source fingerprints for the notebook's helper modules."""
    notebook_fingerprints = globals().get("NOTEBOOK_SOURCE_DIGESTS")
    if notebook_fingerprints is not None:
        return dict(notebook_fingerprints)
    return {path.name: digest(path) for path in Path(__file__).parent.glob("fog_*.py")}


def read_manifest(path, verify_images=False):
    """Fail closed on incomplete datasets; never turn missing annotations into empty GT."""
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty list of labeled images.")
    ids, images, groups = set(), set(), {}
    for row in records:
        for key in (
            "id",
            "image",
            "dataset",
            "split",
            "group",
            "boxes",
            "labels",
            "beta",
        ):
            if key not in row:
                raise ValueError(f"Missing {key} in manifest row")
        if row["dataset"] not in ("foggy_cityscapes", "nuscenes"):
            raise ValueError("Unknown dataset")
        if row["split"] not in ("density_train", "detector_train", "val", "test"):
            raise ValueError("Invalid split")
        if not row["id"] or not row["group"]:
            raise ValueError("Empty image/group identity")
        image = (path.parent / row["image"]).resolve()
        if row["id"] in ids or str(image) in images:
            raise ValueError("Duplicate image identity/path")
        ids.add(row["id"])
        images.add(str(image))
        if not image.is_file():
            raise FileNotFoundError(image)
        row["image"] = str(image)
        if verify_images or "width" not in row or "height" not in row:
            with Image.open(image) as img:
                width, height = img.size
                if verify_images:
                    img.verify()
        else:
            width, height = row["width"], row["height"]
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")
        boxes = np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
        labels = np.asarray(row["labels"])
        if len(boxes) != len(labels) or not np.isfinite(boxes).all():
            raise ValueError("Invalid boxes/labels")
        crowds = row.get("iscrowd", [0] * len(boxes))
        if len(crowds) != len(boxes) or any(c not in (0, 1) for c in crowds):
            raise ValueError("iscrowd must have one binary flag per annotation")
        if len(labels) and (not np.isin(labels, range(len(CLASSES))).all()):
            raise ValueError("Labels must use shared class IDs 0..6")
        if len(boxes) and (
            np.any(boxes[:, 2:] <= boxes[:, :2])
            or np.any(boxes < 0)
            or np.any(boxes > [width, height, width, height])
        ):
            raise ValueError("Boxes must be positive-area XYXY pixels inside the image")
        beta = row["beta"]
        if beta is not None and (not np.isfinite(beta) or not 0 <= beta <= BETA_SCALE):
            raise ValueError(
                "Beta must be null or an attenuation coefficient in [0, .02] m^-1"
            )
        if row["dataset"] == "foggy_cityscapes" and beta is None:
            raise ValueError("Foggy Cityscapes requires known beta")
        key = (row["dataset"], row["group"])
        if key in groups and groups[key] != row["split"]:
            raise ValueError("Scene/sequence leakage across splits")
        groups[key] = row["split"]
        row["width"], row["height"] = width, height
    return records


def select_groups(rows, limit=None):
    """Select whole sequences, preserving all fog variants; None selects everything."""
    keys = sorted({(r["dataset"], r["group"]) for r in rows})
    if limit is not None:
        if limit < 1:
            raise ValueError("Group limit must be positive or None")
        keys = keys[:limit]
    keys = set(keys)
    return sorted(
        [r for r in rows if (r["dataset"], r["group"]) in keys], key=lambda r: r["id"]
    )


def _stable_order(values, seed, label):
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{PROFILE_SELECTION_SCHEMA}:{seed}:{label}:{value}".encode()
        ).hexdigest(),
    )


def select_scene_safe_rows(rows, image_cap, seed, split, label):
    """Select complete scene groups without crossing split boundaries."""
    candidates = [row for row in rows if row["split"] == split]
    if image_cap is None:
        return sorted(candidates, key=lambda row: row["id"])
    if not isinstance(image_cap, int) or image_cap < 1:
        raise ValueError("Profile image caps must be positive integers or None")
    groups = {}
    for row in candidates:
        groups.setdefault((row["dataset"], row["group"]), []).append(row)
    selected, count = [], 0
    for group in _stable_order(groups, seed, label):
        group_rows = sorted(groups[group], key=lambda row: row["id"])
        if count and count + len(group_rows) > image_cap:
            continue
        selected.extend(group_rows)
        count += len(group_rows)
        if count >= image_cap:
            break
    if not selected and candidates:
        smallest = min(groups.values(), key=len)
        selected = sorted(smallest, key=lambda row: row["id"])
    return sorted(selected, key=lambda row: row["id"])


def select_profile_rows(rows, profile, seed, limits=None):
    """Build deterministic, scene-safe rows for one independent profile."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown training profile: {profile}")
    if profile == "full":
        return sorted(rows, key=lambda row: row["id"])
    profile_limits = {
        "setup": {"detector_train": 48, "density_train": 48, "val": 24},
        "partial-2k": {"detector_train": 2000, "density_train": 1000, "val": 200},
        "partial-5k": {"detector_train": 5000, "density_train": 2000, "val": 500},
    }[profile]
    limits = limits or profile_limits
    selected = []
    for split in ("detector_train", "density_train", "val"):
        if limits is not None and split not in limits:
            continue
        label = "tuning" if split == "val" else f"{profile}:{split}"
        selected.extend(select_scene_safe_rows(rows, limits[split], seed, split, label))
    return sorted(selected, key=lambda row: row["id"])


def image_tensor(row, size, device):
    with Image.open(row["image"]) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image).copy()
    return (
        torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
    )


class DensityCNN(nn.Module):
    """Small image-to-beta regressor used as the frozen fog predictor."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, images):
        return (
            self.features(
                F.interpolate(images, (128, 128), mode="bilinear", align_corners=False)
            ).sigmoid()
            * BETA_SCALE
        )


def get_fog_density(images, predictor):
    """Estimate beta for a batch with the frozen image-to-density predictor."""
    if predictor is None:
        raise ValueError("A loaded fog-density predictor is required")
    with torch.inference_mode():
        return predictor(images)


class FogAdapter(nn.Module):
    """Identity-initialized RGB residual with scalar FiLM at hidden features.

    The control has exactly the same parameters, but always receives zero.
    This tests input adaptation, not internal YOLO/DETR feature conditioning.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 8, 3, padding=1)
        self.film = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16))
        self.output = nn.Conv2d(8, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, beta):
        scale, shift = self.film(beta.reshape(-1, 1) / BETA_SCALE).chunk(2, dim=1)
        features = self.features(image)
        features = features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return (image + 0.1 * self.output(F.relu(features)).tanh()).clamp(0, 1)


def train_density(rows, device, epochs, seed):
    if not rows or any(
        r["split"] != "density_train" or r["beta"] is None for r in rows
    ):
        raise ValueError(
            "Density training requires a separate labeled density_train split"
        )
    if len({r["beta"] for r in rows}) < 2:
        raise ValueError("Density estimator needs multiple density values")
    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    seed_all(seed)
    model = DensityCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        order = random.Random(seed + epoch).sample(rows, len(rows))
        losses = []
        for row in order:
            x = image_tensor(row, 128, device)
            target = torch.tensor([[row["beta"]]], device=device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x) / BETA_SCALE, target / BETA_SCALE)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "normalized_mse": float(np.mean(losses))})
    return model.eval().requires_grad_(False), history


class Detector:
    """Differentiable detector loss plus consistent, original-pixel predictions."""

    def __init__(self, name, checkpoint, device, trainable=True):
        self.name, self.device = name, device
        self.yolo = name.startswith("YOLO")
        if self.yolo:
            # Ultralytics selects compiled NMS only when torchvision is loaded.
            # Load it before warm-up/timing to avoid the slow Python fallback.
            import torchvision.ops  # noqa: F401 - import enables compiled NMS
            from ultralytics import YOLO
            from ultralytics.utils import DEFAULT_CFG_DICT

            self.model = YOLO(str(checkpoint)).model.to(device).float()
            self.model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        else:
            from transformers import DetrConfig, DetrForObjectDetection

            kwargs = {} if Path(checkpoint).is_dir() else {"revision": "no_timm"}
            raw, _ = DetrConfig.get_config_dict(str(checkpoint), **kwargs)
            # The official no_timm export predates strict bool validation in v5.
            if raw.get("dilation") is None:
                raw["dilation"] = False
            if raw.get("use_pretrained_backbone") is None:
                raw["use_pretrained_backbone"] = False
            configuration = DetrConfig.from_dict(raw)
            self.model = DetrForObjectDetection.from_pretrained(
                str(checkpoint), config=configuration, **kwargs
            ).to(device)
        self.model.eval().requires_grad_(trainable)
        if self.yolo:
            # YOLO's fixed distribution-to-distance integral is not a learned convolution.
            for name, parameter in self.model.named_parameters():
                if ".dfl." in name:
                    parameter.requires_grad_(False)

    def training_mode(self):
        self.model.train()
        # Batch-one microsteps: preserve pretrained BN statistics, but train affine weights.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, x):
        if self.yolo:
            return self.model(x)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return self.model(pixel_values=(x - mean) / std)

    def loss(self, x, row):
        boxes = x.new_tensor(row["boxes"]).reshape(-1, 4)
        boxes = boxes / x.new_tensor([row["width"], row["height"]] * 2)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        wh = boxes[:, 2:] - boxes[:, :2]
        boxes = torch.cat([center, wh], dim=1)
        mapping = YOLO_IDS if self.yolo else COCO_IDS
        labels = torch.tensor([mapping[i] for i in row["labels"]], device=self.device)
        if self.yolo:
            batch = {
                "img": x,
                "batch_idx": x.new_zeros(len(labels)),
                "cls": labels.reshape(-1, 1),
                "bboxes": boxes,
            }
            return self.model.loss(batch, self.forward(x))[0].sum()
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return self.model(
            pixel_values=(x - mean) / std,
            labels=[{"class_labels": labels.long(), "boxes": boxes}],
        ).loss

    def predict(self, x, row):
        output = self.forward(x)
        if self.yolo:
            from ultralytics.utils.nms import non_max_suppression

            result = non_max_suppression(
                output, conf_thres=0.001, iou_thres=0.7, classes=YOLO_IDS, max_det=100
            )[0]
            boxes, scores, labels = result[:, :4], result[:, 4], result[:, 5].long()
            boxes = boxes / x.shape[-1]
            mapping = YOLO_IDS
        else:
            probabilities = output.logits.softmax(-1)[0, :, :-1]
            scores, labels = probabilities.max(-1)
            cxcy, wh = output.pred_boxes[0].split(2, dim=-1)
            boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=-1)
            mapping = COCO_IDS
        boxes = boxes.clamp(0, 1) * x.new_tensor([row["width"], row["height"]] * 2)
        result = []
        for box, score, label in zip(
            boxes.detach().cpu().tolist(),
            scores.detach().cpu().tolist(),
            labels.detach().cpu().tolist(),
        ):
            if (
                label in mapping
                and score >= 0.001
                and box[2] > box[0]
                and box[3] > box[1]
            ):
                result.append(
                    {"box": box, "score": score, "label": mapping.index(label)}
                )
        return sorted(result, key=lambda r: -r["score"])[:100]


def beta_for(arm, x, row, density):
    if arm == "rgb_control":
        return x.new_zeros((1, 1))
    if arm == "predicted_density":
        return get_fog_density(x, density)
    if arm == "oracle_density" and row["beta"] is not None:
        return x.new_tensor([[row["beta"]]])
    raise ValueError(
        "Oracle density is unavailable; do not substitute zero or a prediction"
    )


def train_adapter(detector, rows, density, arm, device, size, epochs, seed):
    if not rows or any(r["split"] != "detector_train" for r in rows) or epochs < 1:
        raise ValueError(
            "Adapter training requires detector_train rows and positive epochs"
        )
    seed_all(seed)
    adapter = FogAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        losses = []
        for row in random.Random(seed + epoch).sample(rows, len(rows)):
            x = image_tensor(row, size, device)
            beta = beta_for(arm, x, row, density)
            optimizer.zero_grad()
            loss = detector.loss(adapter(x, beta), row)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite detector loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return adapter.eval(), history


def evaluate(detector, adapter, density, arm, rows, device, size, repeats=3):
    if not rows or repeats < 1:
        raise ValueError("Evaluation requires images and positive repeats")

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def single(row):
        # Start before decode, resize, transfer and scalar estimation.
        sync()
        start = time.perf_counter()
        x = image_tensor(row, size, device)
        sync()
        prepared = time.perf_counter()
        beta = beta_for(arm, x, row, density)
        sync()
        estimated = time.perf_counter()
        prediction = detector.predict(adapter(x, beta), row)
        sync()
        finished = time.perf_counter()
        return (
            prediction,
            float(beta.item()),
            {
                "total_ms": (finished - start) * 1000,
                "density_ms": (estimated - prepared) * 1000,
                "detector_adapter_ms": (finished - estimated) * 1000,
            },
        )

    for _ in range(3):
        single(rows[0])
    results = []
    for row in rows:
        timings = []
        for _ in range(repeats):
            prediction, beta, timing = single(row)
            timings.append(timing)
        results.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "beta_input": beta,
                "timings": timings,
            }
        )
    return results


def coco_metrics(rows, results, include_curve=False):
    """COCO bbox AP, including crowd annotations; seven shared categories."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if [r["id"] for r in rows] != [r["id"] for r in results]:
        raise ValueError("Predictions are not paired with the exact evaluation images")
    images, annotations, predictions = [], [], []
    for index, (row, result) in enumerate(zip(rows, results), 1):
        images.append({"id": index, "width": row["width"], "height": row["height"]})
        crowds = row.get("iscrowd", [0] * len(row["boxes"]))
        for box, label, crowd in zip(row["boxes"], row["labels"], crowds):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": int(crowd),
                }
            )
        for pred in result["prediction"]:
            x1, y1, x2, y2 = pred["box"]
            predictions.append(
                {
                    "image_id": index,
                    "category_id": pred["label"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": pred["score"],
                }
            )
    truth = COCO()
    truth.dataset = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        truth.createIndex()
        if predictions:
            predicted = truth.loadRes(predictions)
        else:
            predicted = COCO()
            predicted.dataset = {**truth.dataset, "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(truth, predicted, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    tp = fp = positives = 0
    for item in evaluator.evalImgs:
        if item is None or item["aRng"] != evaluator.params.areaRng[0]:
            continue
        keep = np.asarray(item["dtScores"]) >= 0.25
        ignored = np.asarray(item["dtIgnore"])[0].astype(bool)
        matched = np.asarray(item["dtMatches"])[0] > 0
        tp += int(np.sum(keep & ~ignored & matched))
        fp += int(np.sum(keep & ~ignored & ~matched))
        positives += int(np.sum(~np.asarray(item["gtIgnore"]).astype(bool)))
    metrics = {
        "mAP50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AR100": float(evaluator.stats[8]),
        "precision_at_025": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_025": tp / positives if positives else np.nan,
    }
    for index, name in enumerate(CLASSES):
        values = evaluator.eval["precision"][:, :, index, 0, -1]
        supported = values[values >= 0]
        metrics[f"AP_{name}"] = float(supported.mean()) if len(supported) else np.nan
    metrics = {key: np.nan if value == -1 else value for key, value in metrics.items()}
    if include_curve:
        precision = evaluator.eval["precision"][0, :, :, 0, -1]
        valid = precision >= 0
        counts = valid.sum(axis=1)
        means = np.divide(
            np.where(valid, precision, 0).sum(axis=1),
            counts,
            out=np.full(len(counts), np.nan),
            where=counts > 0,
        )
        metrics["precision_curve"] = means.tolist()
        metrics["recall_curve"] = evaluator.params.recThrs.tolist()
    return metrics


def summarize(rows, results):
    groups = [("all", "all", list(range(len(rows))))]
    fog_indices = [
        i for i, row in enumerate(rows) if row["beta"] is not None and row["beta"] > 0
    ]
    if fog_indices:
        groups.append(("foggy", "foggy", fog_indices))
    for beta in sorted({r["beta"] for r in rows if r["beta"] is not None}):
        groups.append(
            (
                str(beta),
                density_bin(beta),
                [i for i, r in enumerate(rows) if r["beta"] == beta],
            )
        )
    if any(r["beta"] is None for r in rows):
        groups.append(
            (
                "unknown",
                "unknown",
                [i for i, r in enumerate(rows) if r["beta"] is None],
            )
        )
    summaries = []
    for bucket, bin_name, indices in groups:
        subset = [rows[i] for i in indices]
        outputs = [results[i] for i in indices]
        values = [t["total_ms"] for r in outputs for t in r["timings"]]
        metric = coco_metrics(subset, outputs)
        summaries.append(
            {
                "density": bucket,
                "density_bin": bin_name,
                "condition": "clear"
                if bin_name == "clear"
                else "foggy"
                if bin_name in ("light", "moderate", "dense", "foggy")
                else "unknown"
                if bin_name == "unknown"
                else "mixed",
                "distribution": "natural",
                "images": len(subset),
                **metric,
                "mean_ms": float(np.mean(values)),
                "p50_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "serial_fps": 1000 / float(np.mean(values)),
                "density_mean_ms": float(
                    np.mean([t["density_ms"] for r in outputs for t in r["timings"]])
                ),
            }
        )
    return summaries


def plot_comparison(results):
    import matplotlib.pyplot as plt

    if results.empty:
        print(
            "No measured results. Prepare data and run experiments; no placeholder scores are plotted."
        )
        return
    for dataset, frame in results.groupby("dataset"):
        overall = frame[frame.density == "all"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        markers = {"rgb_control": "o", "predicted_density": "^", "oracle_density": "s"}
        for (model, arm), part in overall.groupby(["model", "arm"]):
            axes[0].scatter(
                part.mean_ms,
                part.mAP50_95,
                marker=markers[arm],
                label=f"{model}: {arm}",
            )
        axes[0].set(
            xlabel="End-to-end mean latency (ms)",
            ylabel="COCO bbox mAP@[.50:.95]",
            title=f"{dataset}: accuracy vs latency",
            ylim=(0, 1),
        )
        numeric = frame[~frame.density.isin(["all", "unknown", "balanced"])].copy()
        numeric["beta"] = numeric.density.astype(float)
        for (model, arm), part in numeric.groupby(["model", "arm"]):
            means = part.groupby("beta").mAP50_95.mean()
            axes[1].plot(
                means.index, means.values, marker=markers[arm], label=f"{model}: {arm}"
            )
        axes[1].set(
            xlabel="Known attenuation coefficient beta (m^-1)",
            ylabel="Mean mAP across seeds",
            title="Accuracy by known fog density",
            ylim=(0, 1),
        )
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        plt.show()


def _legacy_adapter_experiment(
    rows,
    output,
    checkpoints,
    device="cpu",
    size=640,
    seeds=(42, 43, 44),
    density_epochs=10,
    adapter_epochs=5,
    repeats=3,
):
    """Sequential runs; immutable output directory prevents stale/concurrent result mixing."""
    import importlib.metadata

    if size < 32 or size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use unique nonempty seeds")
    density_rows = [r for r in rows if r["split"] == "density_train"]
    # Excluding crowd-bearing training images avoids treating ignored objects as negatives.
    train_rows = [
        r
        for r in rows
        if r["split"] == "detector_train" and not any(r.get("iscrowd", []))
    ]
    test_sets = {
        d: [r for r in rows if r["split"] == "test" and r["dataset"] == d]
        for d in sorted({r["dataset"] for r in rows})
    }
    test_sets = {d: r for d, r in test_sets.items() if r}
    if not density_rows or not train_rows or not test_sets:
        raise ValueError(
            "Need independent density_train, detector_train and test data; prepare Foggy Cityscapes first"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "size": size,
        "seeds": list(seeds),
        "density_epochs": density_epochs,
        "adapter_epochs": adapter_epochs,
        "repeats": repeats,
        "device": str(device),
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        "method": "frozen detector, trained input FiLM adapter; identical zero-density control",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "ultralytics", "pycocotools")
        },
        "hardware": torch.cuda.get_device_name()
        if str(device).startswith("cuda")
        else "CPU",
        "excluded_crowd_train_images": sum(
            r["split"] == "detector_train" and any(r.get("iscrowd", [])) for r in rows
        ),
        "input_sha256": {r["id"]: digest(r["image"]) for r in rows},
        "source_sha256": source_module_digest("fog_research.py"),
    }
    (output / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summaries, status, density_scores = [], [], []
    for seed in seeds:
        density, history = train_density(density_rows, device, density_epochs, seed)
        torch.save(density.state_dict(), output / f"density-{seed}.pt")
        pd.DataFrame(history).to_csv(
            output / f"density-{seed}-training.csv", index=False
        )
        # Held-out density quality, split by dataset; null beta is never scored.
        for dataset, subset in test_sets.items():
            known = [r for r in subset if r["beta"] is not None]
            with torch.inference_mode():
                estimates = [
                    float(density(image_tensor(r, size, device)).item()) for r in known
                ]
            if known:
                error = np.array(estimates) - np.array([r["beta"] for r in known])
                density_scores.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "images": len(known),
                        "MAE_beta": float(np.abs(error).mean()),
                        "RMSE_beta": float(np.sqrt((error**2).mean())),
                    }
                )
        pd.DataFrame(density_scores).to_csv(output / "density-quality.csv", index=False)
        for name, checkpoint in checkpoints.items():
            detector = None
            try:
                detector = Detector(name, checkpoint, device)
                # Hash the actual loaded parameters, including downloaded HF weights.
                weights_hash = hashlib.sha256()
                for key, tensor in detector.model.state_dict().items():
                    weights_hash.update(key.encode())
                    weights_hash.update(
                        tensor.detach().cpu().contiguous().numpy().tobytes()
                    )
                (output / f"{name}-{seed}-weights.sha256").write_text(
                    weights_hash.hexdigest()
                )
                for arm in ARMS:
                    adapter, history = train_adapter(
                        detector,
                        train_rows,
                        density,
                        arm,
                        device,
                        size,
                        adapter_epochs,
                        seed,
                    )
                    torch.save(adapter.state_dict(), output / f"{name}-{seed}-{arm}.pt")
                    pd.DataFrame(history).to_csv(
                        output / f"{name}-{seed}-{arm}-training.csv", index=False
                    )
                    for dataset, subset in test_sets.items():
                        identity = {
                            "model": name,
                            "arm": arm,
                            "seed": seed,
                            "dataset": dataset,
                        }
                        if arm == "oracle_density" and any(
                            r["beta"] is None for r in subset
                        ):
                            status.append(
                                {
                                    **identity,
                                    "status": "unavailable",
                                    "reason": "No known beta for this dataset",
                                }
                            )
                            continue
                        if str(device).startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats()
                        predictions = evaluate(
                            detector,
                            adapter,
                            density,
                            arm,
                            subset,
                            device,
                            size,
                            repeats,
                        )
                        (output / f"{name}-{seed}-{arm}-{dataset}.json").write_text(
                            json.dumps(predictions), encoding="utf-8"
                        )
                        peak = (
                            torch.cuda.max_memory_allocated() / 2**20
                            if str(device).startswith("cuda")
                            else None
                        )
                        for summary in summarize(subset, predictions):
                            summaries.append(
                                {**identity, **summary, "peak_allocated_gpu_mb": peak}
                            )
                        status.append({**identity, "status": "complete"})
                        pd.DataFrame(summaries).to_csv(
                            output / "results.csv", index=False
                        )
                        pd.DataFrame(status).to_csv(output / "status.csv", index=False)
            except Exception as error:
                status.append(
                    {
                        "model": name,
                        "seed": seed,
                        "status": "error",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"{name}, seed {seed}: {error}")
            finally:
                del detector
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pd.DataFrame(status).to_csv(output / "status.csv", index=False)
    return pd.DataFrame(summaries), pd.DataFrame(status)
