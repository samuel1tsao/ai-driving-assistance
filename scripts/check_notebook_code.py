"""Validate and lint the notebook, the sole source of study implementation."""

import ast
import subprocess
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = PROJECT_ROOT / "notebooks/fog_density_study.ipynb"
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
    shared_source = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--ignore",
            "F811,E402",
            "--stdin-filename",
            "fog_density_notebook.py",
            "-",
        ],
        input=shared_source,
        text=True,
        check=True,
    )
    print(
        "PASS: notebook schema, cell syntax, shared-namespace lint, readability and format."
    )


if __name__ == "__main__":
    main()
