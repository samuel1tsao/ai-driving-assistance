"""Storage, data loading, execution and presentation for this notebook."""

import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from uuid import uuid4
from pathlib import Path, PurePosixPath

import pandas as pd
from IPython.display import Image, display

from fog_analytics import export_analytics
from fog_research import PROFILE_NAMES, read_manifest, select_profile_rows
from fog_training import preflight, run_training, run_checkpoint_evaluation


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
    SETTINGS["action"] = str(SETTINGS["action"]).strip().lower()
    if SETTINGS["action"] not in {"train", "preflight", "analyze", "evaluate"}:
        raise ValueError("Unknown action.")
    if not isinstance(SETTINGS.get("retrain", False), bool):
        raise ValueError("retrain must be True or False.")
    if SETTINGS.get("retrain", False) and SETTINGS["action"] != "train":
        raise ValueError("retrain=True requires action='train'.")
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
    saved_configuration = restore_existing_profile_settings()
    apply_training_profile(saved_configuration)
    resolve_default_export_path()
    allocate_fresh_retrain_output()
    configure_checkpoint_evaluation()
    return TRAINING_CONFIGURATION, RESOLVED_PATHS


def apply_training_profile(saved_configuration):
    """Resolve new settings unless this output folder owns a saved experiment."""
    apply_subset_execution_settings()
    resolve_manifest_and_model_paths()
    if saved_configuration is None:
        return
    TRAINING_CONFIGURATION.clear()
    TRAINING_CONFIGURATION.update(saved_configuration)


def resolve_default_export_path():
    """Keep a default report folder beside the selected run folder."""
    expected = SETTINGS["profile"]["run_name"] + "-results"
    if SETTINGS["output"]["export_name"] != expected:
        return
    run = RESOLVED_PATHS["run"]
    RESOLVED_PATHS["export"] = run.with_name(run.name + "-results")


def allocate_fresh_retrain_output():
    """Give retraining its own folder so prior checkpoints remain recoverable."""
    if not SETTINGS.get("retrain", False) or SETTINGS["profile"]["name"] == "setup":
        return
    run = RESOLVED_PATHS["run"]
    output = run.with_name(run.name + "-retrain-" + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    TRAINING_CONFIGURATION.update(
        output=str(output), initialization="pretrained", density_checkpoint=None
    )
    RESOLVED_PATHS["run"] = output
    RESOLVED_PATHS["export"] = output.with_name(output.name + "-results")
    print("Fresh training requested. Previous run preserved. New output:", output)


def configure_checkpoint_evaluation():
    """Switch to a separate output only for matched checkpoint evaluation."""
    if SETTINGS["action"] != "evaluate":
        return
    options = SETTINGS["evaluation"]
    source = Path(options["source_run"] or TRAINING_CONFIGURATION["output"])
    output = RESOLVED_PATHS["output_root"] / single_name(options["output_name"])
    if source.resolve() == output.resolve():
        raise ValueError("Evaluation must use a separate output_name.")
    TRAINING_CONFIGURATION.update(
        evaluation_source=str(source),
        output=str(output),
        evaluation_split=options["split"],
        overwrite_existing=False,
        evaluation_max_pairs=options.get("max_pairs"),
        evaluation_cityscapes_root=str(Path(options["cityscapes_root"]).resolve()),
    )
    RESOLVED_PATHS["run"] = output
    RESOLVED_PATHS["export"] = output.with_name(output.name + "-results")


def restore_existing_profile_settings():
    """An existing output folder defines the experiment; new defaults do not."""
    path = RESOLVED_PATHS["run"] / "settings.json"
    if (
        SETTINGS["action"] != "train"
        or SETTINGS.get("retrain", False)
        or SETTINGS.get("test", {}).get("enabled", False)
        or SETTINGS["profile"]["name"] == "setup"
        or not path.exists()
    ):
        return None
    recorded = json.loads(path.read_text(encoding="utf-8"))["config"]
    if recorded.get("profile_name", "full") != SETTINGS["profile"]["name"]:
        raise ValueError("Saved output belongs to another profile.")
    SETTINGS["profile"].update(recorded.get("profile_scope", {}))
    SETTINGS["profile"]["density_sampling"] = recorded.get(
        "density_sampling", "natural"
    )
    SETTINGS["profile"]["overwrite_existing"] = False
    recorded["output"] = str(RESOLVED_PATHS["run"])
    if SETTINGS["profile"]["name"] == "full":
        recorded.setdefault("profile_name", "full")
        recorded.setdefault("initialization", "existing_full_run")
    print("Using this run's recorded training settings (including density sampling).")
    return recorded


def should_reuse_completed_training(configuration):
    """Check saved experiment statuses without opening images or model weights."""
    if (
        SETTINGS["action"] != "train"
        or SETTINGS["profile"]["name"] == "setup"
        or SETTINGS.get("retrain", False)
    ):
        return False
    output = Path(configuration["output"])
    if not (output / "settings.json").is_file():
        return False
    directories = [
        output / f"{model}-{arm}-{seed}"
        for model in configuration["checkpoints"]
        for arm in configuration["arms"]
        for seed in configuration["seeds"]
    ]
    return bool(directories) and all(
        (directory / "results.csv").is_file()
        and (directory / "status.json").is_file()
        and json.loads((directory / "status.json").read_text()).get("status")
        == "complete"
        for directory in directories
    )


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
    if should_reuse_completed_training(TRAINING_CONFIGURATION):
        print(
            "Training already complete. Reusing saved results; data preparation skipped."
        )
        return
    if SETTINGS["action"] in {"analyze", "evaluate"}:
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
    if should_reuse_completed_training(configuration):
        return []
    if SETTINGS["action"] == "evaluate":
        from fog_data import cityscapes_manifest

        options = SETTINGS["evaluation"]
        manifest = Path(configuration["output"]) / "matched-cityscapes.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        from filelock import FileLock

        with FileLock(str(manifest) + ".lock", timeout=0):
            if not manifest.exists():
                cityscapes_manifest(
                    options["cityscapes_root"],
                    manifest,
                    with_detection_labels=True,
                    splits=("val",),
                )
        annotations = read_manifest(manifest, verify_images=True)
        root = Path(options["cityscapes_root"]).resolve()
        if any(not Path(row["image"]).is_relative_to(root) for row in annotations):
            raise ValueError(
                "cityscapes_root changed; choose a new evaluation output_name."
            )
        return annotations
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
    if should_reuse_completed_training(configuration):
        checks = pd.DataFrame(
            [
                {
                    "check": "saved training complete",
                    "ready": True,
                    "detail": "Reuse completed/early-stopped comparisons; no training or inference",
                }
            ]
        )
        display(checks)
        return checks
    if SETTINGS["action"] == "evaluate":
        from fog_training import validate_matched_evaluation

        source = Path(configuration["evaluation_source"])
        source_rows = json.loads((source / "input-manifest.json").read_text())
        selected = validate_matched_evaluation(
            annotations,
            source_rows,
            configuration["evaluation_split"],
            configuration.get("evaluation_max_pairs"),
        )
        checks = pd.DataFrame(
            [
                {
                    "check": "matched held-out images",
                    "ready": True,
                    "detail": f"{len(selected)} labeled images",
                }
            ]
        )
        display(checks)
        return checks
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
    if action != "train":
        raise ValueError("Only action='train' can enter training.")
    if should_reuse_completed_training(configuration):
        print(
            "All comparisons complete. Reusing saved reports with their original settings."
        )
        results = execute_analysis(configuration)
        return results, pd.read_csv(Path(configuration["output"]) / "status.csv")
    if not checks.ready.all():
        print("Training is blocked by the checks above.")
        return pd.DataFrame(), None
    return run_training(annotations, configuration)


def train_or_analyze_configured_study(action, configuration, annotations, checks):
    action = str(action).strip().lower()
    if action not in {"train", "preflight", "analyze", "evaluate"}:
        raise ValueError(
            "Choose train, preflight, analyze or evaluate; unknown actions never train."
        )
    print(f"Action: {action} | Output: {configuration['output']}")
    if action == "analyze":
        return execute_analysis(configuration), None
    if action == "evaluate":
        return run_checkpoint_evaluation(annotations, configuration)
    return execute_training(action, configuration, annotations, checks)


def run_optional_synthetic_benchmark(enabled):
    """Measure a synthetic batch-64 ceiling without changing study results."""
    import time

    import torch
    import torchvision
    from torch.utils.data import DataLoader, TensorDataset

    can_run = enabled and torch.cuda.is_available()
    can_run = can_run and SETTINGS["profile"]["name"] == "full"
    if not can_run:
        print("Synthetic batch-64 benchmark skipped for this profile or CUDA state.")
        return
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    image_count, batch_size = 2000, 64
    images = torch.randn(image_count, 3, 640, 640, device="cuda", dtype=torch.float16)
    labels = torch.randn(image_count, 3, 640, 640, device="cuda", dtype=torch.float16)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    model = torchvision.models.resnet50(weights=None).to("cuda").half()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        model(images[:batch_size])
    torch.cuda.synchronize()
    started = time.perf_counter()
    for batch_images, _ in loader:
        optimizer.zero_grad()
        model(batch_images).sum().backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"Processed {image_count} images in {elapsed:.2f} seconds.")
    print(f"Synthetic benchmark throughput: {image_count / elapsed:.2f} images/second.")


def display_accuracy_latency_and_density_tables(results, output):
    if results.empty:
        print("No completed runs yet.")
        return
    columns = [
        "dataset",
        "model",
        "arm",
        "seed",
        "density",
        "images",
        "mAP50_95",
        "AP50",
        "precision_at_025",
        "recall_at_025",
        "AR100",
    ]
    print(
        "Detection quality: higher is better. Clear = beta 0; foggy = pooled positive beta."
    )
    display(
        results.loc[
            results.density.isin(["all", "balanced", "0.0", "foggy"]), columns
        ].round(4)
    )
    print("Inference cost: lower milliseconds is better; higher FPS is better.")
    display(
        results.loc[
            results.density == "all",
            [
                "dataset",
                "model",
                "arm",
                "seed",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "serial_fps",
                "density_mean_ms",
            ],
        ].round(2)
    )
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
    hypothesis = pd.read_csv(Path(output) / "fog-vs-clear.csv")
    if not hypothesis.empty:
        print("Positive delta means density helped detection; negative means it hurt.")
        display(hypothesis.round(4))
        summary = pd.read_csv(Path(output) / "hypothesis-summary.csv")
        display(summary.round(4))
    else:
        print(
            "Clear/fog hypothesis untested: this run has no labeled matched benchmark."
        )
    print(
        "Checkpoint health: best_epoch is the evaluated checkpoint; epochs_trained is total fitted epochs."
    )
    display(pd.read_csv(Path(output) / "training-health.csv"))
    print("Evaluation sample counts and labeled object counts:")
    display(pd.read_csv(Path(output) / "dataset-audit.csv"))


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
        "*-conditions.png",
        "*-effects.png",
        "*-condition-classes.png",
        "*-precision-recall.png",
        "density-diagnostics.png",
        "dataset-audit.png",
    ):
        for path in sorted((Path(output) / "figures").glob(pattern)):
            display(Image(filename=str(path), width=1000))


def verify_subset_completion_and_checkpoint_reuse(results, annotations):
    """Check subset artifacts and reuse completed checkpoints without training again."""
    if should_reuse_completed_training(TRAINING_CONFIGURATION):
        print("Completed comparisons reused; no checkpoint loading or optimizer steps.")
        return
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
            annotations, TRAINING_CONFIGURATION, reset_setup=False
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
