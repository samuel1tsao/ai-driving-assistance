"""Execute the consolidated notebook on a bounded real-data subset locally."""

import argparse
import ast
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]


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
    notebook = nbformat.read(
        ROOT / "notebooks/fog_density_consolidated.ipynb", as_version=4
    )
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
            if (
                "resolve_density_predictor_checkpoint(TRAINING_CONFIGURATION)"
                in cell.source
            ):
                import torch

                fixture_checkpoint = runtime / "synthetic-density-predictor.pt"
                fixture_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(42)
                torch.save(
                    {
                        "kind": "fog-density-v1",
                        "beta_scale": shell.user_ns["BETA_SCALE"],
                        "train_groups": [["nuscenes", "fixture-held-out"]],
                        "model": shell.user_ns["FogDensityRegressor"]().state_dict(),
                    },
                    fixture_checkpoint,
                )
                shell.user_ns["SETTINGS"]["density_predictor"]["checkpoint"] = str(
                    fixture_checkpoint
                )
                shell.user_ns["TRAINING_CONFIGURATION"]["density_checkpoint"] = str(
                    fixture_checkpoint
                )
            print(f"Executing cell {index}", flush=True)
            execution_count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
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
        "PASS: consolidated notebook completed including checkpoint reuse and ZIP export.",
        flush=True,
    )


if __name__ == "__main__":
    main()
