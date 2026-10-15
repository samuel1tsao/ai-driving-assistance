"""Enforce one notebook source, centralized settings, and a short visible workflow."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/fog_density_study.ipynb"
MAX_WORKFLOW_STATEMENTS = 5
MAX_DEFINITION_SECTIONS = 13
ORCHESTRATION_FUNCTIONS = {
    "resolve_study_configuration",
    "prepare_detector_study",
    "select_profile_training_rows",
    "select_scene_safe_rows",
    "train_and_validate_detector",
    "load_study_manifest",
    "load_labeled_cityscapes_data",
    "load_labeled_clear_fog_data",
    "run_saved_fog_comparison",
    "run_configured_detector_study",
    "run_comparison_matrix",
    "run_detector_comparison",
    "run_detector_training_and_evaluation",
    "resolve_density_predictor_checkpoint",
    "fit_fog_density_model",
    "run_checkpoint_evaluation",
    "evaluate_saved_comparison",
    "validate_matched_evaluation",
    "validate_study_settings",
    "apply_subset_execution_settings",
}


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert not (ROOT / "notebooks/_support").exists(), (
        "Remove the duplicate support implementation"
    )
    definitions = {}
    helper_cells = []
    constants_cells = [
        cell for cell in notebook.cells if cell.metadata.get("fogStudyConstants")
    ]
    assert len(constants_cells) == 1, "Use one shared constants/profile cell"
    workflow_started = False
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            workflow_started |= cell.source.startswith("## 3.")
            continue
        assert "%%writefile" not in cell.source, (
            f"cell {index} writes implementation files"
        )
        tree = ast.parse(cell.source, filename=f"notebook cell {index}")
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("fog_")
            for node in ast.walk(tree)
        ), f"cell {index} imports external study logic"
        if cell.metadata.get("fogStudyDefinitions"):
            if not cell.metadata.get("fogStudyConstants"):
                helper_cells.append(cell)
                assert cell.metadata.get("cellView") == "form", (
                    f"cell {index} must start collapsed"
                )
                assert cell.metadata.get("jupyter", {}).get("source_hidden"), (
                    f"cell {index} must also hide code in Jupyter"
                )
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    assert node.name not in definitions, (
                        f"duplicate definition: {node.name}"
                    )
                    definitions[node.name] = node
                    if node.name in ORCHESTRATION_FUNCTIONS:
                        assert node.end_lineno - node.lineno < 40, (
                            f"{node.name} must stay short"
                        )
            continue
        if workflow_started:
            assert len(tree.body) <= MAX_WORKFLOW_STATEMENTS, (
                f"workflow cell {index} is too dense"
            )
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body
            ), f"workflow cell {index} should call named steps"
    assert len(helper_cells) <= MAX_DEFINITION_SECTIONS
    step_validators = {name for name in definitions if name.startswith("validate_")}
    assert step_validators == {
        "validate_training_inputs",
        "validate_study_settings",
        "validate_and_record_run_identity",
        "validate_matched_evaluation",
        "validate_and_extract_full_dataset_archive",
    }
    assert not any(
        name.startswith("require_") or name == "readiness_check" for name in definitions
    )
    assert "_legacy_adapter_experiment" not in definitions
    assert "resolve_density_predictor_checkpoint" in definitions
    assert "load_fog_density_model" in definitions
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    setup_heading = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and cell.source.startswith("### Fit or resume the fog-density model")
    )
    predictor_workflow = notebook.cells[setup_heading + 1].source
    assert "get_fog_model_training_data(fog_density_rows)" in code
    assert "fit_fog_density_model(" in predictor_workflow
    assert "ARM_DENSITY" not in predictor_workflow
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "train_detector_profile" not in definitions
    assert "prepare_detector_study(" in code
    assert "labeled_cityscapes_data = load_labeled_cityscapes_data(" in code
    assert "run_identity_rows = nuscenes_data + labeled_cityscapes_data" in code
    assert "detector_partitions = prepare_detector_study(" in code
    detector_heading = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## 5.")
    )
    assert all(name in detector_heading for name in ("YOLO11n", "YOLO11m", "DETR"))
    assert (
        "RGB only" in detector_heading
        and "predicted fog-density value" in detector_heading
    )
    training_calls = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "train_and_validate_detector"
            ):
                training_calls.append(
                    (
                        ast.unparse(node.args[0]),
                        ast.unparse(node.args[4]),
                        ast.unparse(node.args[5]),
                    )
                )
    assert training_calls == [
        ("detector_partitions", model, arm)
        for model in ("YOLO_SMALL", "YOLO_MEDIUM", "DETR_MODEL")
        for arm in ("ARM_RGB", "ARM_DENSITY")
    ]
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert (
        'nuscenes_data = load_study_manifest(TRAINING_CONFIGURATION["manifest"])'
        in code
    )
    assert 'TRAINING_CONFIGURATION["city_manifest"]' in code
    assert '"gtFine"' in code
    assert '"detector_split"' in code
    assert "labeled_clear_fog_data = load_labeled_clear_fog_data()" in code
    assert "rows = load_labeled_cityscapes_data()" in code
    assert "run_saved_fog_comparison(labeled_clear_fog_data)" in code
    assert 'SETTINGS["action"]' not in "\n".join(cell.source for cell in notebook.cells)
    assert "setup_fog_density_model" not in definitions
    assert "load_or_train_fog_density_model" not in definitions
    print(
        f"PASS: notebook-only study, centralized constants, {len(helper_cells)} collapsed implementation sections, and high-level workflow."
    )


if __name__ == "__main__":
    main()
