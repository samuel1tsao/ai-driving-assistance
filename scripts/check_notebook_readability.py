"""Enforce a self-contained notebook with a small, readable execution flow."""

import ast
from pathlib import Path

import nbformat

from sync_notebook_support import MODULE_ORDER, remove_local_imports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/fog_density_consolidated.ipynb"
MAX_WORKFLOW_STATEMENTS = 5
MAX_ORCHESTRATION_LINES = 40
MAX_IMPLEMENTATION_CELLS = len(MODULE_ORDER)
ORCHESTRATION_FUNCTIONS = {
    "resolve_study_configuration",
    "train_or_analyze_configured_study",
    "execute_training",
    "run_training",
    "run_comparison_matrix",
    "run_detector_comparison",
    "run_detector_training_and_evaluation",
    "configure_density_predictor_checkpoint",
}


def implementation_sources(notebook):
    sources = {filename: [] for filename, _ in MODULE_ORDER}
    cells = 0
    for index, cell in enumerate(notebook.cells):
        filename = cell.metadata.get("fogStudyImplementation")
        if not filename:
            continue
        if filename not in sources:
            raise AssertionError(f"cell {index} embeds unexpected module {filename}")
        if cell.cell_type != "code":
            raise AssertionError(f"implementation cell {index} must contain Python")
        if cell.metadata.get("cellView") != "form":
            raise AssertionError(f"implementation cell {index} must start collapsed")
        if cell.metadata.get("tags", [])[:1] != ["fog-study-implementation"]:
            raise AssertionError(f"implementation cell {index} needs its section tag")
        if not cell.source.startswith("#@title Supporting code:"):
            raise AssertionError(
                f"implementation cell {index} needs a visible collapsed-cell title"
            )
        tree = ast.parse(cell.source, filename=f"notebook implementation cell {index}")
        sources[filename].append(cell.source)
        cells += 1
    if cells > MAX_IMPLEMENTATION_CELLS:
        raise AssertionError(
            f"implementation has {cells} cells; limit is {MAX_IMPLEMENTATION_CELLS}"
        )
    return {name: "\n\n".join(parts) for name, parts in sources.items()}, cells


def require_self_contained(notebook):
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for index, source in enumerate(code):
        if "%%writefile" in source or "fog-study-implementation" in source:
            raise AssertionError(f"notebook code cell {index} writes helper files")
        tree = ast.parse(source, filename=f"notebook cell {index}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("fog_")
            ):
                raise AssertionError(
                    f"notebook cell {index} imports an external fog module"
                )
    for expected in (
        "### Fog-density predictor setup",
        "## 3. Prepare the study data",
        "## 4. Validate",
        "## 5. Train and evaluate the detectors",
        "## 6. Review the outputs",
        "## 7. Export the run",
    ):
        if not any(
            cell.cell_type == "markdown" and cell.source.startswith(expected)
            for cell in notebook.cells
        ):
            raise AssertionError(f"missing notebook workflow section: {expected}")


def require_source_parity(sources):
    support = PROJECT_ROOT / "notebooks/_support"
    for filename, source in sources.items():
        expected = remove_local_imports(
            (support / filename).read_text(encoding="utf-8")
        )
        if ast.dump(ast.parse(source), include_attributes=False) != ast.dump(
            ast.parse(expected), include_attributes=False
        ):
            raise AssertionError(f"embedded notebook code differs from {filename}")


def require_short_flat_orchestration(sources):
    for filename, source in sources.items():
        for node in ast.parse(source, filename=filename).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ORCHESTRATION_FUNCTIONS:
                continue
            if node.end_lineno - node.lineno + 1 > MAX_ORCHESTRATION_LINES:
                raise AssertionError(f"{filename}:{node.name} exceeds 40 lines")
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                for child in ast.walk(node)
                if child is not node
            ):
                raise AssertionError(f"{filename}:{node.name} contains nested helpers")


def require_short_visible_workflow(notebook):
    workflow_started = False
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith("## 3."):
            workflow_started = True
            continue
        if not workflow_started or cell.cell_type != "code":
            continue
        if cell.metadata.get("fogStudyImplementation"):
            continue
        tree = ast.parse(cell.source, filename=f"notebook workflow cell {index}")
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        ):
            raise AssertionError(f"workflow cell {index} must call high-level steps")
        if len(tree.body) > MAX_WORKFLOW_STATEMENTS:
            raise AssertionError(f"workflow cell {index} has too many statements")


def main():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    require_self_contained(notebook)
    sources, cell_count = implementation_sources(notebook)
    require_source_parity(sources)
    require_short_flat_orchestration(sources)
    require_short_visible_workflow(notebook)
    print(
        "PASS: single-notebook implementation, grouped collapsed helpers, "
        f"and high-level workflow ({cell_count} implementation cells)."
    )


if __name__ == "__main__":
    main()
