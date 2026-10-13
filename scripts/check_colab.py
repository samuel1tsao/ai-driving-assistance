"""Check that the published Colab artifact exactly mirrors the notebook source."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = nbformat.read(ROOT / "notebooks/fog_density_consolidated.ipynb", 4)
    colab = nbformat.read(ROOT / "notebooks/fog_density_colab.ipynb", 4)
    nbformat.validate(colab)
    assert source == colab, "Regenerate the Colab mirror from the notebook source"
    for index, cell in enumerate(colab.cells):
        if cell.cell_type == "code":
            assert "%%writefile" not in cell.source
            ast.parse(cell.source, filename=f"Colab cell {index}")
    print("PASS: Colab mirrors the notebook exactly and needs no support source files.")


if __name__ == "__main__":
    main()
