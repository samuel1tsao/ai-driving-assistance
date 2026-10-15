"""Execute the consolidated notebook on a bounded real-data subset locally."""

import argparse
import ast
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]


def create_fixture_density_checkpoint(shell, runtime):
    import torch

    checkpoint_path = runtime / "synthetic-density-predictor.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = shell.user_ns["TRAINING_CONFIGURATION"]
    torch.manual_seed(42)
    torch.save(
        {
            "kind": "fog-density-v1",
            "beta_scale": shell.user_ns["BETA_SCALE"],
            "train_groups": [["nuscenes", "fixture-held-out"]],
            "planned_epochs": configuration["density_epochs"],
            "density_sampling": "natural",
            "seed": 42,
            "epoch": configuration["density_epochs"],
            "step": 0,
            "best_loss": 0.0,
            "running_sum": 0.0,
            "running_count": 0,
            "history": [],
            "model": shell.user_ns["FogDensityRegressor"]().state_dict(),
        },
        checkpoint_path,
    )
    shell.user_ns["SETTINGS"]["density_predictor"]["checkpoint"] = str(checkpoint_path)
    configuration["density_checkpoint"] = str(checkpoint_path)


def create_fixture_cityscapes_data(shell, runtime):
    import numpy as np
    from PIL import Image

    original_rows = shell.user_ns["fog_density_rows"]
    first_base_by_split = {}
    for row in sorted(original_rows, key=lambda value: value["id"]):
        first_base_by_split.setdefault(row["split"], row["id"].split("/")[1])
    selected_bases = set(first_base_by_split.values())
    rows = [row for row in original_rows if row["id"].split("/")[1] in selected_bases]
    root = runtime / "cityscapes-labeled-fixture"
    for row in rows:
        image = Path(row["image"])
        split, city = image.parents[1].name, image.parent.name
        base = row["id"].split("/")[1]
        with Image.open(image) as source:
            width, height = source.size
        mask_path = root / "gtFine" / split / city / f"{base}_gtFine_instanceIds.png"
        clear_path = root / "leftImg8bit" / split / city / f"{base}_leftImg8bit.png"
        if not mask_path.exists():
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((height, width), dtype=np.uint16)
            mask[2:8, 2:8] = 26001
            Image.fromarray(mask).save(mask_path)
            clear_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (width, height), "gray").save(clear_path)
    shell.user_ns["fog_density_rows"] = rows
    shell.user_ns["SETTINGS"]["evaluation"]["cityscapes_root"] = str(root)


def main():
    print("Loading and validating notebook", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Test selective extraction from the existing data ZIP",
    )
    parser.add_argument("--images", type=int, default=2)
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / "notebooks/fog_density_study.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        compile(cell.source, f"notebook cell {index}", "exec")
    runtime = ROOT / ".runtime/consolidated-notebook-check"
    output = ROOT / "reports/fog/consolidated-notebook-check"
    settings = next(
        cell for cell in notebook.cells if cell.metadata.get("fogStudySettings")
    )
    overrides = {
        "runtime_root": str(runtime),
        "drive_mount": str(ROOT / ".runtime/unused-drive-mount"),
    }
    settings.source += (
        "\n# Local verification overrides (the source notebook is unchanged).\n"
    )
    settings.source += "SETTINGS.update(" + repr(overrides) + ")\n"
    settings.source += (
        'SETTINGS["density_predictor"]["checkpoint"] = '
        + repr(str(runtime / "synthetic-density-predictor.pt"))
        + "\n"
    )
    settings.source += (
        'SETTINGS["data"]["location"] = '
        + repr({"kind": "path", "path": str(ROOT / "dist" if args.zip else ROOT)})
        + "\n"
    )
    settings.source += (
        'SETTINGS["data"].update('
        + repr({"mode": "zip" if args.zip else "directory", "low_disk": False})
        + ")\n"
    )
    settings.source += (
        'SETTINGS["output"]["location"] = '
        + repr({"kind": "path", "path": str(output)})
        + "\n"
    )
    settings.source += 'SETTINGS["output"]["run_name"] = "checkpoint-refactor-zip-validation" if SETTINGS["data"]["mode"] == "zip" else "checkpoint-refactor-directory-validation"\n'
    settings.source += 'SETTINGS["output"]["export_name"] = SETTINGS["output"]["run_name"] + "-results"\n'
    settings.source += (
        'SETTINGS["evaluation"]["cityscapes_root"] = '
        + repr(str(runtime / "cityscapes-labeled-fixture"))
        + "\n"
    )
    settings.source += (
        'SETTINGS["test"]["images_per_stratum"] = ' + repr(args.images) + "\n"
    )
    settings.source += 'SETTINGS["test"]["enabled"] = True\n'
    settings.source += (
        'SETTINGS["models"]["YOLO11n"] = ' + repr(str(ROOT / "yolo11n.pt")) + "\n"
    )
    dependency = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "subprocess.check_call" in cell.source
    )
    dependency_tree = ast.parse(dependency.source)
    install_call = next(
        node
        for node in dependency_tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "subprocess.check_call"
    )
    dependency_lines = dependency.source.splitlines()
    dependency.source = "\n".join(
        dependency_lines[: install_call.lineno - 1]
        + dependency_lines[install_call.end_lineno :]
    )
    output.mkdir(parents=True, exist_ok=True)
    executed = output / (
        "zip-executed.ipynb" if args.zip else "directory-executed.ipynb"
    )
    shell = InteractiveShell.instance()
    print("Executing self-contained cells with project Python", flush=True)
    try:
        execution_count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell.source.strip() == "prepare_fog_data()":
                create_fixture_density_checkpoint(shell, runtime)
            training_call = next(
                (
                    node
                    for node in ast.walk(ast.parse(cell.source))
                    if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "train_and_validate_detector"
                ),
                None,
            )
            if training_call and shell.user_ns["SETTINGS"]["test"]["enabled"]:
                model_name = shell.user_ns[ast.unparse(training_call.args[4])]
                if (
                    model_name
                    not in shell.user_ns["TRAINING_CONFIGURATION"]["checkpoints"]
                ):
                    message = (
                        f"Local smoke test skips unconfigured model: {model_name}.\n"
                    )
                    cell.execution_count = execution_count + 1
                    cell.outputs = [
                        nbformat.v4.new_output("stream", name="stdout", text=message)
                    ]
                    print(message, flush=True)
                    execution_count += 1
                    continue
            if (
                "labeled_clear_fog_data = load_labeled_clear_fog_data()" in cell.source
                and shell.user_ns["SETTINGS"]["test"]["enabled"]
            ):
                message = "Local smoke test skips the separate matched clear/fog report cell.\n"
                cell.execution_count = execution_count + 1
                cell.outputs = [
                    nbformat.v4.new_output("stream", name="stdout", text=message)
                ]
                print(message, flush=True)
                execution_count += 1
                continue
            if (
                "run_saved_fog_comparison(labeled_clear_fog_data)" in cell.source
                and shell.user_ns["SETTINGS"]["test"]["enabled"]
            ):
                message = "Local smoke test skips the separate matched clear/fog report cell.\n"
                cell.execution_count = execution_count + 1
                cell.outputs = [
                    nbformat.v4.new_output("stream", name="stdout", text=message)
                ]
                print(message, flush=True)
                execution_count += 1
                continue
            print(f"Executing cell {index}", flush=True)
            execution_count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
                if cell.source.strip().startswith("fog_density_rows = load_study_manifest("):
                    create_fixture_cityscapes_data(shell, runtime)
            cell.execution_count = execution_count
            cell.outputs = []
            for name, value in (
                ("stdout", captured.stdout),
                ("stderr", captured.stderr),
            ):
                if value:
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name=name, text=value)
                    )
            for rich in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=rich.data, metadata=rich.metadata
                    )
                )
            if captured.stdout:
                print(captured.stdout[-2500:], flush=True)
            error = result.error_before_exec or result.error_in_exec
            if error:
                if captured.stderr:
                    print(captured.stderr[-3000:], flush=True)
                raise error
    finally:
        nbformat.write(notebook, executed)
        print(f"Executed notebook: {executed}", flush=True)
    print(
        "PASS: bounded nuScenes and Cityscapes detector training, validation, reports and checkpoint export completed; the separate matched analysis is covered by its standalone unit test.",
        flush=True,
    )


if __name__ == "__main__":
    main()
