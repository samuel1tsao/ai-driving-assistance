"""Embed editable study implementation directly in the single Colab notebook."""

import ast
import hashlib
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/fog_density_consolidated.ipynb"
SUPPORT_DIRECTORY = PROJECT_ROOT / "notebooks/_support"
MODULE_ORDER = (
    ("fog_research.py", "Detector models and fog-density prediction"),
    ("fog_prepare.py", "nuScenes data preparation"),
    ("fog_data.py", "Image manifests and data validation"),
    ("fog_analytics.py", "Tables, comparisons, and charts"),
    ("fog_training.py", "Detector training and checkpoint recovery"),
    ("fog_notebook.py", "Notebook setup and high-level workflow steps"),
)
LOCAL_MODULES = {Path(name).stem for name, _ in MODULE_ORDER}
BENCHMARK_CELL = """RUN_SYNTHETIC_BENCHMARK = False
run_optional_synthetic_benchmark(RUN_SYNTHETIC_BENCHMARK)
"""


def remove_local_imports(source):
    """Remove imports between helper files; notebook cells share one namespace."""
    lines = source.splitlines(keepends=True)
    nodes = ast.walk(ast.parse(source))
    spans = [
        (node.lineno - 1, node.end_lineno)
        for node in nodes
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.split(".")[0] in LOCAL_MODULES
    ]
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    return "".join(lines)


def source_fingerprint_cell():
    fingerprints = {
        name: hashlib.sha256((SUPPORT_DIRECTORY / name).read_bytes()).hexdigest()
        for name, _ in MODULE_ORDER
    }
    code = "NOTEBOOK_SOURCE_DIGESTS = " + repr(fingerprints)
    cell = nbformat.v4.new_code_cell(code)
    cell.metadata["fogStudyFingerprints"] = True
    cell.metadata["cellView"] = "form"
    return cell


def implementation_cells():
    section = nbformat.v4.new_markdown_cell(
        "### Supporting functions and classes (collapsed)\n\n"
        "These implementation cells stay inside this notebook. They are grouped by "
        "purpose below and are collapsed so the main workflow reads as a short series "
        "of high-level steps. No helper files are written or imported."
    )
    section.metadata["heading_collapsed"] = True
    cells = [section, source_fingerprint_cell()]
    for filename, title in MODULE_ORDER:
        heading = nbformat.v4.new_markdown_cell(f"#### {title}")
        cells.append(heading)
        source = remove_local_imports(
            (SUPPORT_DIRECTORY / filename).read_text(encoding="utf-8")
        )
        code = nbformat.v4.new_code_cell(f"#@title Supporting code: {title}\n{source}")
        code.metadata["tags"] = ["fog-study-implementation", Path(filename).stem]
        code.metadata["fogStudyImplementation"] = filename
        code.metadata["cellView"] = "form"
        cells.append(code)
    return cells


def main():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    retained = []
    skip_until = None
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith(
            "## 2. Runtime implementation (collapsed)"
        ):
            skip_until = "### Install packages and check the GPU"
            continue
        if cell.cell_type == "markdown" and cell.source.startswith(
            "### Supporting functions and classes (collapsed)"
        ):
            skip_until = "### Resolve storage and show the selected run"
            continue
        if skip_until:
            if cell.cell_type != "markdown" or not cell.source.startswith(skip_until):
                continue
            skip_until = None
        if cell.metadata.get("fogStudyImplementation") or cell.metadata.get(
            "fogStudyFingerprints"
        ):
            continue
        if cell.cell_type == "code" and cell.source.startswith(
            "%%writefile notebooks/_support/"
        ):
            continue
        if cell.cell_type == "code" and cell.source.startswith(
            "%%writefile -a notebooks/_support/"
        ):
            continue
        if cell.cell_type == "code" and "RUN_SYNTHETIC_BENCHMARK =" in cell.source:
            cell.source = BENCHMARK_CELL
        retained.append(cell)

    install_code_index = next(
        index
        for index, cell in enumerate(retained)
        if cell.cell_type == "code"
        and "subprocess.check_call" in cell.source
        and "pip" in cell.source
    )
    retained[install_code_index + 1 : install_code_index + 1] = implementation_cells()
    notebook.cells = retained
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
