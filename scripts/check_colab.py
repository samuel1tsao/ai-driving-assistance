"""Check the single self-contained notebook used locally and in Colab."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = ROOT / "notebooks/fog_density_study.ipynb"
    notebook_files = sorted(path.name for path in (ROOT / "notebooks").glob("fog_density_*.ipynb"))
    assert notebook_files == [notebook_path.name], (
        f"Expected one active fog study notebook, found {notebook_files}"
    )
    notebook = nbformat.read(notebook_path, 4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert "%%writefile" not in cell.source
            ast.parse(cell.source, filename=f"Colab cell {index}")
    print("PASS: one self-contained fog study notebook is valid for Colab and local use.")


if __name__ == "__main__":
    main()
