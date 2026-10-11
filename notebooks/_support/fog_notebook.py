"""Storage, data loading, execution and presentation for this notebook."""

import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from IPython.display import Image, display

from fog_analytics import export_analytics
from fog_research import PROFILE_NAMES, read_manifest, select_profile_rows
from fog_training import preflight, run_training


PROFILE_DEFAULTS = {
    "setup": {
        "training_image_cap": 48,
        "density_image_cap": 48,
        "tuning_image_cap": 24,
        "epochs": 1,
        "density_epochs": 1,
        "size": 320,
        "save_every": 2,
        "seed": 42,
        "models": ["YOLO11n", "YOLO11m", "DETR"],
        "evaluation_split": "val",
        "early_stopping_patience": 0,
        "density_sampling": "balanced_bins",
        "overwrite_existing": True,
        "run_name": "setup-check-1",
        "run_suffix": "-setup-1",
    },
    "partial-2k": {
        "training_image_cap": 2000,
        "density_image_cap": 1000,
        "tuning_image_cap": 500,
        "epochs": 5,
        "density_epochs": 5,
        "size": 640,
        "save_every": 200,
        "seed": 42,
        "models": ["YOLO11n", "YOLO11m", "DETR"],
        "evaluation_split": "val",
        "early_stopping_patience": 2,
        "density_sampling": "balanced_bins",
        "overwrite_existing": False,
        "run_name": "partial-2k",
        "run_suffix": "-partial-2k",
    },
    "partial-5k": {
        "training_image_cap": 5000,
        "density_image_cap": 2000,
        "tuning_image_cap": 500,
        "epochs": 5,
        "density_epochs": 5,
        "size": 640,
        "save_every": 200,
        "seed": 42,
        "models": ["YOLO11n", "YOLO11m", "DETR"],
        "evaluation_split": "val",
        "early_stopping_patience": 2,
        "density_sampling": "balanced_bins",
        "overwrite_existing": False,
        "run_name": "partial-5k",
        "run_suffix": "-partial-5k",
    },
    "full": {
        "training_image_cap": None,
        "density_image_cap": None,
        "tuning_image_cap": None,
        "epochs": 30,
        "density_epochs": 15,
        "size": 640,
        "save_every": 200,
        "seed": 42,
        "models": ["YOLO11n", "YOLO11m", "DETR"],
        "evaluation_split": "test",
        "early_stopping_patience": 0,
        "density_sampling": "natural",
        "overwrite_existing": False,
        "run_name": "colab-full-study",
        "run_suffix": "-full",
    },
}


def profile_settings(name):
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown training profile: {name}")
    return dict(PROFILE_DEFAULTS[name])


def relative_path(value):
    value = str(value)
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"Expected a nonempty relative path: {value!r}")
    return Path(value)


def location_path(location):
    if location["kind"] == "drive":
        return (
            Path(SETTINGS["drive_mount"])
            / relative_path(SETTINGS["drive_subdir"])
            / relative_path(location["path"])
        )
    if location["kind"] == "path":
        path = Path(location["path"])
        if not path.is_absolute():
            raise ValueError("Filesystem locations must be absolute paths.")
        return path
    raise ValueError("Location kind must be 'drive' or 'path'.")


def single_name(value):
    path = relative_path(value)
    if len(path.parts) != 1 or value in (".", ".."):
        raise ValueError(f"Expected a single filename or directory name: {value!r}")
    return value


def validate_study_settings():
    global ROOT
    if SETTINGS["action"] not in {"train", "preflight", "analyze"}:
        raise ValueError("Unknown action.")
    profile = SETTINGS.get("profile", {})
    profile_name = profile.get("name")
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"profile.name must be one of {PROFILE_NAMES}")
    if profile.get("initialization") not in {"pretrained", "existing_full_run"}:
        raise ValueError("profile.initialization is invalid")
    if profile_name == "full" and profile.get("initialization") != "existing_full_run":
        raise ValueError("The full profile must identify the existing full run.")
    if profile_name != "full" and profile.get("initialization") != "pretrained":
        raise ValueError("Partial profiles must start from pretrained weights.")
    data_settings = SETTINGS["data"]
    if data_settings["mode"] not in {"zip", "directory"}:
        raise ValueError("Data mode must be zip or directory.")
    if data_settings["minimum_free_gib"] < 0:
        raise ValueError("minimum_free_gib cannot be negative.")
    single_name(data_settings["archive"])
    single_name(SETTINGS["output"]["run_name"])
    single_name(SETTINGS["output"]["export_name"])
    single_name(SETTINGS["output"]["packages_file"])
    ROOT = Path(SETTINGS["runtime_root"])
    if not ROOT.is_absolute() or not Path(SETTINGS["drive_mount"]).is_absolute():
        raise ValueError("runtime_root and drive_mount must be absolute.")
    if data_settings["mode"] == "directory" and data_settings["url"]:
        raise ValueError("A data URL is only supported in zip mode.")
    if data_settings["low_disk"] and (
        data_settings["location"]["kind"] != "drive" or data_settings["url"]
    ):
        raise ValueError("Low-disk mode requires data in Drive with no URL.")


def connect_and_resolve_study_storage():
    global DATA_ROOT, RESOLVED_PATHS, TRAINING_CONFIGURATION
    data_settings = SETTINGS["data"]
    # Resolve paths; authentication already ran before setup.
    data_location = location_path(data_settings["location"])
    output_location = location_path(SETTINGS["output"]["location"])
    if any(
        loc["kind"] == "drive"
        for loc in (data_settings["location"], SETTINGS["output"]["location"])
    ):
        drive_root = Path(SETTINGS["drive_mount"]) / SETTINGS["drive_subdir"]
        if not drive_root.is_dir():
            raise RuntimeError(
                "Google Drive is disconnected. Rerun the connection cell near the top."
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    (ROOT / "notebooks/_support").mkdir(parents=True, exist_ok=True)
    DATA_ROOT = data_location if data_settings["mode"] == "directory" else ROOT
    RESOLVED_PATHS = {
        "data_location": data_location,
        "data_root": DATA_ROOT,
        "archive": data_location / data_settings["archive"],
        "output_root": output_location,
        "run": output_location / SETTINGS["output"]["run_name"],
        "export": output_location / SETTINGS["output"]["export_name"],
        "marker": ROOT / relative_path(data_settings["extraction_marker"]),
        "packages": ROOT / SETTINGS["output"]["packages_file"],
        "city_images": DATA_ROOT / relative_path(data_settings["city_images"]),
        "nuscenes": DATA_ROOT / relative_path(data_settings["nuscenes"]),
    }
    RESOLVED_PATHS["run"].mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIGURATION = dict(SETTINGS["training"])


def apply_subset_execution_settings():
    global DATA_ROOT, SETTINGS
    output_location = RESOLVED_PATHS["output_root"]
    test_settings = SETTINGS.get("test", {})
    if test_settings.get("enabled"):
        smoke = dict(PROFILE_DEFAULTS["setup"])
        smoke["name"] = "setup"
        smoke["initialization"] = "pretrained"
        smoke["models"] = test_settings.get("models", smoke["models"])
        smoke["seed"] = test_settings.get("seed", smoke["seed"])
        smoke["epochs"] = test_settings.get("epochs", smoke["epochs"])
        smoke["density_epochs"] = test_settings.get(
            "density_epochs", smoke["density_epochs"]
        )
        smoke["size"] = test_settings.get("image_size", smoke["size"])
        images_per_stratum = test_settings.get("images_per_stratum", 4)
        smoke["training_image_cap"] = max(6, images_per_stratum * 6)
        smoke["density_image_cap"] = max(3, images_per_stratum * 3)
        smoke["tuning_image_cap"] = max(3, images_per_stratum * 3)
        SETTINGS["profile"] = smoke
    profile = SETTINGS["profile"]
    profile_name = profile["name"]
    if profile_name != "full":
        single_name(profile["run_suffix"])
        RESOLVED_PATHS["run"] = output_location / SETTINGS["output"]["run_name"]
        RESOLVED_PATHS["export"] = output_location / SETTINGS["output"]["export_name"]
        RESOLVED_PATHS["run"].mkdir(parents=True, exist_ok=True)
        TRAINING_CONFIGURATION.update(
            epochs=profile["epochs"],
            density_epochs=profile["density_epochs"],
            size=profile["size"],
            save_every=profile["save_every"],
            seeds=[profile["seed"]],
            early_stopping_patience=profile["early_stopping_patience"],
            profile_name=profile_name,
            initialization="pretrained",
            evaluation_split=profile["evaluation_split"],
            density_sampling=profile["density_sampling"],
            overwrite_existing=profile.get("overwrite_existing", False),
        )
        TRAINING_CONFIGURATION["profile_scope"] = dict(profile)
        TRAINING_CONFIGURATION["test_scope"] = {
            "enabled": True,
            "profile": profile_name,
            "scientific_results": profile_name != "setup",
        }
        RESOLVED_PATHS["data_root"] = ROOT / ("profile-" + profile_name)
        RESOLVED_PATHS["data_root"].mkdir(parents=True, exist_ok=True)
        DATA_ROOT = RESOLVED_PATHS["data_root"]
    else:
        TRAINING_CONFIGURATION.update(
            profile_name="full",
            initialization="existing_full_run",
            evaluation_split="test",
            early_stopping_patience=0,
            density_sampling=profile.get("density_sampling", "natural"),
        )


def resolve_manifest_and_model_paths():
    data_settings = SETTINGS["data"]
    selected_models = SETTINGS["profile"]["models"]
    TRAINING_CONFIGURATION.update(
        {
            key: str(DATA_ROOT / relative_path(value))
            for key, value in data_settings["manifests"].items()
        }
    )
    TRAINING_CONFIGURATION["output"] = str(RESOLVED_PATHS["run"])
    TRAINING_CONFIGURATION["checkpoints"] = {
        name: value if name == "DETR" else str(ROOT / Path(value))
        for name, value in SETTINGS["models"].items()
        if name in selected_models
    }
    if TRAINING_CONFIGURATION["density_checkpoint"]:
        TRAINING_CONFIGURATION["density_checkpoint"] = str(
            ROOT / Path(TRAINING_CONFIGURATION["density_checkpoint"])
        )


def resolve_study_configuration(settings):
    """Validate settings, connect storage and resolve one experiment configuration."""
    global SETTINGS
    SETTINGS = settings
    validate_study_settings()
    connect_and_resolve_study_storage()
    apply_subset_execution_settings()
    resolve_manifest_and_model_paths()
    return TRAINING_CONFIGURATION, RESOLVED_PATHS


def check_free_space(required=0):
    free = shutil.disk_usage(ROOT).free
    reserve = int(SETTINGS["data"]["minimum_free_gib"] * 2**30)
    if required + reserve > free:
        raise OSError(
            f"Need {(required + reserve) / 2**30:.2f} GiB free; available {free / 2**30:.2f} GiB."
        )


def select_subset(annotations):
    profile = SETTINGS["profile"]
    eligible = [
        annotation
        for annotation in annotations
        if not (
            annotation["split"] in ("detector_train", "val")
            and (
                not annotation.get("detection_labeled", True)
                or any(annotation.get("iscrowd", []))
            )
            and annotation["beta"] is None
        )
    ]
    selected = select_profile_rows(
        eligible,
        profile["name"],
        profile["seed"],
        {
            "detector_train": profile["training_image_cap"],
            "density_train": profile["density_image_cap"],
            "val": profile["tuning_image_cap"],
        },
    )
    if not selected:
        raise ValueError("No eligible subset images in this manifest.")
    return selected


def prepare_subset():
    from pathlib import PurePosixPath

    data_settings = SETTINGS["data"]
    destination_root = RESOLVED_PATHS["data_root"].resolve()
    archive = None
    if data_settings["mode"] == "zip":
        if data_settings["url"]:
            raise ValueError(
                "Subset ZIP mode requires an existing archive, not a URL download."
            )
        if not RESOLVED_PATHS["archive"].is_file():
            raise FileNotFoundError(RESOLVED_PATHS["archive"])
        archive = zipfile.ZipFile(RESOLVED_PATHS["archive"])
    manifest_annotations, images = {}, set()
    try:
        for key, relative in data_settings["manifests"].items():
            relative_path(relative)
            manifest_text = (
                archive.read(relative).decode("utf-8")
                if archive
                else (RESOLVED_PATHS["data_location"] / relative).read_text(
                    encoding="utf-8"
                )
            )
            annotations = select_subset(json.loads(manifest_text))
            for annotation in annotations:
                image = str(annotation["image"])
                if archive is None:
                    source_image = (
                        RESOLVED_PATHS["data_location"] / relative
                    ).parent / image
                    source_image = source_image.resolve()
                    archive_member_name = source_image.relative_to(
                        RESOLVED_PATHS["data_location"].resolve()
                    ).as_posix()
                    annotation["image"] = os.path.relpath(
                        archive_member_name, str(PurePosixPath(relative).parent)
                    ).replace("\\", "/")
                    images.add(archive_member_name)
                    continue
                # Portable manifests must refer to files within the data bundle.
                if PurePosixPath(image).is_absolute() or "\\" in image or ":" in image:
                    raise ValueError(f"Nonportable manifest image: {image}")
                manifest_parent = PurePosixPath(relative).parent
                path_components = []
                for part in (manifest_parent / image).parts:
                    if part == "..":
                        if not path_components:
                            raise ValueError(f"Image escapes dataset root: {image}")
                        path_components.pop()
                    elif part != ".":
                        path_components.append(part)
                archive_member_name = "/".join(path_components)
                if not archive_member_name.startswith(
                    tuple(data_settings["allowed_archive_roots"])
                ):
                    raise ValueError(f"Unexpected dataset image: {archive_member_name}")
                images.add(archive_member_name)
            manifest_annotations[relative] = annotations
        check_free_space(
            sum(archive.getinfo(name).file_size for name in images)
            if archive
            else sum(
                (RESOLVED_PATHS["data_location"] / name).stat().st_size
                for name in images
            )
        )
        for archive_member_name in sorted(images):
            target = destination_root / archive_member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if archive:
                info = archive.getinfo(archive_member_name)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(
                        f"Archive image is a symlink: {archive_member_name}"
                    )
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            else:
                shutil.copy2(
                    RESOLVED_PATHS["data_location"] / archive_member_name, target
                )
        for relative, annotations in manifest_annotations.items():
            manifest = destination_root / relative
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(annotations), encoding="utf-8")
        selected_ids = sorted(
            annotation["id"]
            for annotations in manifest_annotations.values()
            for annotation in annotations
        )
        report = {
            "profile": SETTINGS["profile"],
            "selected_ids": selected_ids,
            "scope": {
                "training_image_cap": SETTINGS["profile"]["training_image_cap"],
                "tuning_image_cap": SETTINGS["profile"]["tuning_image_cap"],
                "density_image_cap": SETTINGS["profile"]["density_image_cap"],
            },
            "images": len(images),
            "records_per_manifest": {
                name: len(annotations)
                for name, annotations in manifest_annotations.items()
            },
            "scientific_results": SETTINGS["profile"]["name"] != "setup",
        }
        (RESOLVED_PATHS["run"] / "subset-selection.json").write_text(
            json.dumps(report, indent=2)
        )
        print("Subset prepared:", json.dumps(report, indent=2))
    finally:
        if archive:
            archive.close()


def prepare_configured_dataset():
    data_settings = SETTINGS["data"]
    if SETTINGS["action"] == "analyze":
        print("Analysis uses saved run outputs; dataset extraction is skipped.")
        return
    if SETTINGS["profile"]["name"] != "full":
        prepare_subset()
        return
    if data_settings["mode"] == "directory":
        for name in data_settings["manifests"].values():
            path = RESOLVED_PATHS["data_root"] / relative_path(name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared manifest: {path}")
        check_free_space()
        return
    dataset_archive = find_or_download_dataset_archive()
    validate_and_extract_full_dataset_archive(dataset_archive)
    verify_prepared_manifest_files()


def find_or_download_dataset_archive():
    data_settings = SETTINGS["data"]
    bundle = RESOLVED_PATHS["archive"]
    if data_settings["url"]:
        if not data_settings["url"].startswith("https://"):
            raise ValueError("Data URL must use HTTPS.")
        # Use a URL-specific cache name; changing URL does not reuse a different bundle.
        bundle = ROOT / (
            hashlib.sha256(data_settings["url"].encode()).hexdigest()[:16]
            + "-"
            + data_settings["archive"]
        )
        if not bundle.exists():
            partial = bundle.with_suffix(bundle.suffix + ".part")
            urllib.request.urlretrieve(data_settings["url"], partial)
            partial.replace(bundle)
    if not bundle.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {bundle}")
    return bundle


def validate_and_extract_full_dataset_archive(bundle):
    data_settings = SETTINGS["data"]
    info = bundle.stat()
    identity = {
        "path": str(bundle.resolve()),
        "size": info.st_size,
        "modified_ns": info.st_mtime_ns,
        "manifests": data_settings["manifests"],
    }
    marker = RESOLVED_PATHS["marker"]
    try:
        previous = json.loads(marker.read_text()) if marker.exists() else None
    except (ValueError, OSError):
        previous = None
    if previous != identity:
        with zipfile.ZipFile(bundle) as archive:
            members = archive.infolist()
            required_manifests = set(data_settings["manifests"].values())
            if not required_manifests.issubset(archive.namelist()):
                raise ValueError(
                    f"Archive must contain prepared manifests: {sorted(required_manifests)}"
                )
            for archive_member_name in members:
                destination = (ROOT / archive_member_name.filename).resolve()
                if (
                    not destination.is_relative_to(ROOT.resolve())
                    or not archive_member_name.filename.startswith(
                        tuple(data_settings["allowed_archive_roots"])
                    )
                    or stat.S_ISLNK(archive_member_name.external_attr >> 16)
                ):
                    raise ValueError(
                        f"Unsafe or unexpected archive member: {archive_member_name.filename}"
                    )
            check_free_space(
                sum(archive_member_name.file_size for archive_member_name in members)
            )
            archive.extractall(ROOT)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity), encoding="utf-8")


def verify_prepared_manifest_files():
    for path in (
        TRAINING_CONFIGURATION["manifest"],
        TRAINING_CONFIGURATION["city_manifest"],
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing extracted manifest: {path}")
    check_free_space()
    print("Prepared data:", RESOLVED_PATHS["data_root"])


def load_and_validate_dataset_annotations(configuration):
    annotations = []
    for key in ("manifest", "city_manifest"):
        path = Path(configuration[key])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared manifest not found: {path}")
        if path.exists():
            print(f"Validating {path.name}")
            portable_rows = json.loads(path.read_text(encoding="utf-8"))
            for annotation in portable_rows:
                image_path = (path.parent / annotation["image"]).resolve()
                os.utime(image_path, ns=(946684800000000000, 946684800000000000))
            annotations.extend(read_manifest(path))
    return annotations


def display_dataset_split_summary(annotations, configuration):
    if not annotations:
        return
    frame = pd.DataFrame(annotations)
    display(
        frame.groupby(["dataset", "split"]).agg(
            images=("id", "size"),
            scenes=("group", "nunique"),
            known_beta=("beta", "count"),
        )
    )
    coverage_path = Path(configuration["manifest"]).with_suffix(".coverage.json")
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        display(
            pd.DataFrame(
                [
                    {
                        "expected": coverage["expected_images"],
                        "available": coverage["available_images"],
                        "missing": len(coverage["missing_images"]),
                        "complete": coverage["complete"],
                    }
                ]
            )
        )


def display_training_readiness_checks(configuration, annotations):
    checks = preflight(configuration, annotations)
    display(checks)
    if not checks.ready.all():
        display(checks.loc[~checks.ready, ["check", "detail"]])
    return checks


def execute_analysis(configuration):
    return export_analytics(configuration["output"])


def execute_training(action, configuration, annotations, checks):
    if action == "preflight":
        return pd.DataFrame(), None
    if not checks.ready.all():
        print("Training is blocked by the checks above.")
        return pd.DataFrame(), None
    return run_training(annotations, configuration)


def train_or_analyze_configured_study(action, configuration, annotations, checks):
    if action == "analyze":
        return execute_analysis(configuration), None
    return execute_training(action, configuration, annotations, checks)


def display_accuracy_latency_and_density_tables(results, output):
    if results.empty:
        print("No completed runs yet.")
        return
    display(results[results.density.isin(["all", "balanced"])].round(4))
    paired = pd.read_csv(Path(output) / "paired-deltas.csv", dtype={"density": str})
    display(
        paired.loc[
            paired.density.isin(["all", "balanced"]),
            ["model", "dataset", "seed", "arm", "delta_mAP", "delta_latency_ms"],
        ].round(4)
    )
    density = pd.read_csv(Path(output) / "density-quality.csv")
    if not density.empty:
        display(density.round(6))


def display_saved_comparison_figures(output, results):
    if results.empty:
        return
    for pattern in (
        "*-comparison.png",
        "*-balanced.png",
        "*-deltas.png",
        "*-classes.png",
        "*-density.png",
        "*-loss.png",
    ):
        for path in sorted((Path(output) / "figures").glob(pattern)):
            display(Image(filename=str(path), width=1000))


def verify_subset_completion_and_checkpoint_reuse(results, annotations):
    """Check subset artifacts and reuse completed checkpoints without training again."""
    # Fail the test if any stage or required checkpoint/report is missing.
    if SETTINGS["profile"]["name"] != "full" and SETTINGS["action"] == "train":
        required = [
            RESOLVED_PATHS["run"] / name
            for name in (
                "settings.json",
                "input-manifest.json",
                "input-selection.json",
                "results.csv",
                "paired-deltas.csv",
                "density-quality.csv",
                "subset-selection.json",
            )
        ]
        for seed in TRAINING_CONFIGURATION["seeds"]:
            if "predicted_density" in TRAINING_CONFIGURATION["arms"]:
                required.extend(
                    RESOLVED_PATHS["run"] / f"density-{seed}" / name
                    for name in ("last.pt", "best.pt")
                )
            for model in TRAINING_CONFIGURATION["checkpoints"]:
                for arm in TRAINING_CONFIGURATION["arms"]:
                    directory = RESOLVED_PATHS["run"] / f"{model}-{arm}-{seed}"
                    required.extend(
                        directory / name
                        for name in ("last.pt", "best.pt", "status.json", "results.csv")
                    )
                    state = json.loads((directory / "status.json").read_text())
                    if state["status"] != "complete":
                        raise RuntimeError(f"Incomplete test run: {directory}")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing test artifacts: {missing}")
        if results.empty:
            raise RuntimeError("Subset test produced no metrics.")
        import math

        overall = results.loc[results.density == "all"]
        for metric in ("mAP50_95", "AP50", "AR100", "mean_ms", "p95_ms"):
            if overall.empty or not overall[metric].map(math.isfinite).all():
                raise RuntimeError(
                    f"Undefined subset metric: {metric}. Check selected ground truth."
                )
        saved_times = {
            str(path): path.stat().st_mtime_ns
            for path in RESOLVED_PATHS["run"].glob("*/last.pt")
        }
        # Exercise completed-run resume; the engine should reuse checkpoints and saved results.
        resumed_results, resumed_status = run_training(
            annotations, TRAINING_CONFIGURATION
        )
        if any(
            Path(path).stat().st_mtime_ns != timestamp
            for path, timestamp in saved_times.items()
        ):
            raise RuntimeError(
                "Completed-run resume unexpectedly retrained checkpoints."
            )
        if resumed_results.empty or not resumed_status.status.eq("complete").all():
            raise RuntimeError("Completed-run resume failed.")
        print(
            "SUBSET E2E PASSED: training, evaluation, reports, checkpoints and completed-run resume."
        )


def export_checkpoints_and_reports():
    """Package the configured run and its recorded dependency versions."""
    if not RESOLVED_PATHS["run"].exists():
        raise FileNotFoundError(f"No run directory: {RESOLVED_PATHS['run']}")
    if RESOLVED_PATHS["packages"].exists():
        shutil.copy2(
            RESOLVED_PATHS["packages"],
            RESOLVED_PATHS["run"] / SETTINGS["output"]["packages_file"],
        )
    archive_path = shutil.make_archive(
        str(RESOLVED_PATHS["export"]), "zip", str(RESOLVED_PATHS["run"])
    )

    return archive_path
