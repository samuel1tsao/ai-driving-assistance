"""Enforce one notebook source, centralized settings, and a short visible workflow."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/fog_density_consolidated.ipynb"
MAX_WORKFLOW_STATEMENTS = 5
MAX_DEFINITION_SECTIONS = 8
ORCHESTRATION_FUNCTIONS = {
    "resolve_study_configuration",
    "train_or_analyze_configured_study",
    "execute_training",
    "run_training",
    "run_comparison_matrix",
    "run_detector_comparison",
    "run_detector_training_and_evaluation",
    "resolve_density_predictor_checkpoint",
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
    assert "_legacy_adapter_experiment" not in definitions
    assert "resolve_density_predictor_checkpoint" in definitions
    assert "load_fog_density_model" in definitions
    setup_heading = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and cell.source.startswith("### Load the fog-density model")
    )
    predictor_workflow = notebook.cells[setup_heading + 1].source
    assert (
        "resolve_density_predictor_checkpoint(TRAINING_CONFIGURATION)"
        in predictor_workflow
    )
    assert "load_fog_density_model(" in predictor_workflow
    assert "ARM_DENSITY not in" in predictor_workflow
    assert 'TRAINING_CONFIGURATION["arms"]' in predictor_workflow
    assert "setup_fog_density_model" not in definitions
    print(
        f"PASS: notebook-only study, centralized constants, {len(helper_cells)} collapsed implementation sections, and high-level workflow."
    )


if __name__ == "__main__":
    main()
