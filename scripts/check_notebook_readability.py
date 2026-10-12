"""Enforce the readable, single-notebook study structure."""

import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/fog_density_consolidated.ipynb"
SUPPORT_DIRECTORY = PROJECT_ROOT / "notebooks/_support"
MAX_WORKFLOW_STATEMENTS = 5
MAX_ORCHESTRATION_LINES = 40
MAX_IMPLEMENTATION_CELL_LINES = 140
MAX_IMPLEMENTATION_CELLS = 70
ORCHESTRATION_FUNCTIONS = {
    "resolve_study_configuration",
    "train_or_analyze_configured_study",
    "execute_training",
    "run_training",
    "run_comparison_matrix",
    "run_detector_comparison",
    "run_detector_training_and_evaluation",
    "run_density_model_only",
}


def writefile_cell_parts(cell):
    directive, source = cell.source.split("\n", 1)
    target = Path(directive.split()[-1])
    append = " -a " in directive
    return target, append, source


def assembled_support_sources(notebook):
    sources = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.startswith("%%writefile"):
            continue
        target, append, source = writefile_cell_parts(cell)
        if target.parent != Path("notebooks/_support") or target.suffix != ".py":
            continue
        if not append:
            sources[target.name] = []
        if target.name not in sources:
            raise AssertionError(f"cell {index} appends before creating {target}")
        sources[target.name].append(source + "\n")
        if cell.metadata.get("cellView") != "form":
            raise AssertionError(f"implementation cell {index} must start collapsed")
    return {name: "".join(parts) for name, parts in sources.items()}


def require_one_definition_per_implementation_cell(notebook):
    started = set()
    implementation_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.startswith("%%writefile"):
            continue
        target, append, source = writefile_cell_parts(cell)
        if target.parent != Path("notebooks/_support") or target.suffix != ".py":
            continue
        tree = ast.parse(source, filename=f"notebook cell {index}")
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if append and not definitions:
            raise AssertionError(
                f"implementation cell {index} must add a named helper or class"
            )
        cell_lines = len(cell.source.splitlines())
        if cell_lines > MAX_IMPLEMENTATION_CELL_LINES and len(definitions) != 1:
            raise AssertionError(
                f"implementation cell {index} has {cell_lines} lines; "
                "grouped cells must be smaller than 140 lines"
            )
        if not append and target.name in started:
            raise AssertionError(
                f"cell {index} overwrites {target} after its first cell"
            )
        started.add(target.name)
        implementation_cells += 1
    if implementation_cells > MAX_IMPLEMENTATION_CELLS:
        raise AssertionError(
            f"runtime implementation has {implementation_cells} cells; "
            f"limit is {MAX_IMPLEMENTATION_CELLS}"
        )


def require_short_flat_orchestration(sources):
    for filename, source in sources.items():
        tree = ast.parse(source, filename=filename)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ORCHESTRATION_FUNCTIONS:
                continue
            line_count = node.end_lineno - node.lineno + 1
            if line_count > MAX_ORCHESTRATION_LINES:
                raise AssertionError(
                    f"{filename}:{node.name} is {line_count} lines; "
                    f"limit is {MAX_ORCHESTRATION_LINES}"
                )
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                for child in ast.walk(node)
                if child is not node
            ):
                raise AssertionError(
                    f"{filename}:{node.name} must call named helpers, not nested functions"
                )


def require_short_visible_workflow_cells(notebook):
    workflow_started = False
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.startswith("## 3."):
            workflow_started = True
            continue
        if not workflow_started or cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"notebook cell {index}")
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        ):
            raise AssertionError(
                f"workflow cell {index} must call helpers, not define them"
            )
        if len(tree.body) > MAX_WORKFLOW_STATEMENTS:
            raise AssertionError(
                f"workflow cell {index} has {len(tree.body)} statements; "
                f"limit is {MAX_WORKFLOW_STATEMENTS}"
            )


def main():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    sources = assembled_support_sources(notebook)
    require_one_definition_per_implementation_cell(notebook)
    require_short_flat_orchestration(sources)
    require_short_visible_workflow_cells(notebook)
    for name, source in sources.items():
        mirror = SUPPORT_DIRECTORY / name
        if not mirror.is_file() or mirror.read_text(encoding="utf-8") != source:
            raise AssertionError(f"{mirror} must be regenerated from the notebook")
    print("PASS: readable notebook workflow, flat orchestration and generated mirrors.")


if __name__ == "__main__":
    main()
