"""Validate and lint Python cells, including modules written across append cells."""

import ast
import subprocess
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = PROJECT_ROOT / "notebooks/fog_density_consolidated.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    source_directory = PROJECT_ROOT / ".runtime/notebook-code-check"
    source_directory.mkdir(parents=True, exist_ok=True)
    module_paths = set()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if not cell.source.startswith("%%writefile"):
            ast.parse(cell.source, filename=f"notebook cell {index}")
            continue
        directive, source = cell.source.split("\n", 1)
        filename = Path(directive.split()[-1]).name
        if not filename.endswith(".py"):
            continue
        module_path = source_directory / filename
        with module_path.open(
            "a" if " -a " in directive else "w", encoding="utf-8"
        ) as stream:
            stream.write(source + "\n")
        module_paths.add(module_path)
    for module_path in module_paths:
        ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    module_arguments = [str(path) for path in sorted(module_paths)]
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *module_arguments], check=True
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *module_arguments],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(notebook_path)], check=True
    )
    print("PASS: notebook format, Python syntax, lint and module formatting.")


if __name__ == "__main__":
    main()
