"""Validate and lint the notebook, the sole source of study implementation."""

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
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/check_notebook_readability.py")],
        check=True,
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook cell {index}")
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", str(notebook_path)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--ignore",
            "F811",
            str(notebook_path),
        ],
        check=True,
    )
    print("PASS: notebook schema, direct-cell syntax, readability, lint and format.")


if __name__ == "__main__":
    main()
