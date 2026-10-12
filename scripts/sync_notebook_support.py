"""Regenerate collapsed notebook helper cells from the authored support modules."""

import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/fog_density_consolidated.ipynb"
SUPPORT_DIRECTORY = PROJECT_ROOT / "notebooks/_support"
MAX_CELL_SOURCE_LINES = 120
BENCHMARK_CELL = """RUN_SYNTHETIC_BENCHMARK = False
workflow.run_optional_synthetic_benchmark(RUN_SYNTHETIC_BENCHMARK)
"""


def definition_starts(lines):
    tree = ast.parse("".join(lines))
    starts = {0, len(lines)}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorator_lines = [decorator.lineno for decorator in node.decorator_list]
            starts.add(min([node.lineno, *decorator_lines]) - 1)
    return sorted(starts)


def module_cells(module_path):
    lines = module_path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = definition_starts(lines)
    ranges = []
    chunk_start = starts[0]
    chunk_definitions = []
    for definition_start in starts[1:-1]:
        while (
            chunk_definitions
            and definition_start - chunk_start >= MAX_CELL_SOURCE_LINES
        ):
            boundary = (
                definition_start
                if len(chunk_definitions) == 1
                else chunk_definitions[-1]
            )
            ranges.append((chunk_start, boundary))
            chunk_definitions = [boundary] if boundary in chunk_definitions else []
            chunk_start = boundary
            if boundary == definition_start:
                break
        chunk_definitions.append(definition_start)
    while (
        len(lines) - chunk_start > MAX_CELL_SOURCE_LINES and len(chunk_definitions) > 1
    ):
        boundary = chunk_definitions[-1]
        ranges.append((chunk_start, boundary))
        chunk_start = boundary
        chunk_definitions = [boundary]
    ranges.append((chunk_start, starts[-1]))
    cells = []
    for index, (start, end) in enumerate(ranges):
        source = "".join(lines[start:end]).rstrip("\n")
        if end != len(lines):
            source += "\n\n"
        directive = "%%writefile" if index == 0 else "%%writefile -a"
        cell = nbformat.v4.new_code_cell(
            f"{directive} {module_path.relative_to(PROJECT_ROOT).as_posix()}\n{source}"
        )
        cell.metadata["cellView"] = "form"
        cells.append(cell)
    return cells


def is_support_write_cell(cell):
    if cell.cell_type != "code" or not cell.source.startswith("%%writefile"):
        return False
    target = Path(cell.source.splitlines()[0].split()[-1])
    return target.parent == Path("notebooks/_support") and target.suffix == ".py"


def main():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells = []
    emitted = set()
    for cell in notebook.cells:
        if not is_support_write_cell(cell):
            if cell.cell_type == "code" and "RUN_SYNTHETIC_BENCHMARK =" in cell.source:
                cell.source = BENCHMARK_CELL
            cells.append(cell)
            continue
        target = Path(cell.source.splitlines()[0].split()[-1])
        if target in emitted:
            continue
        cells.extend(module_cells(PROJECT_ROOT / target))
        emitted.add(target)
    notebook.cells = cells
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
