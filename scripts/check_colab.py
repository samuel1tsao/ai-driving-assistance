"""Check the standalone Colab notebook mirrors the current profile notebook."""

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/fog_density_colab.ipynb"
MODULE_NAMES = {
    "fog_analytics.py",
    "fog_data.py",
    "fog_notebook.py",
    "fog_prepare.py",
    "fog_research.py",
    "fog_training.py",
}


def source_from_cell(cell):
    if not cell.source.startswith("%%writefile"):
        return cell.source
    return cell.source.split("\n", 1)[1]


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    embedded = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("%%writefile"):
            target = Path(cell.source.splitlines()[0].split()[-1]).name
            if target in MODULE_NAMES:
                embedded[target] = source_from_cell(cell)
        else:
            ast.parse(cell.source, filename=f"notebook cell {cell.id}")
    for name in MODULE_NAMES:
        expected = (
            (ROOT / "notebooks/_support" / name).read_text(encoding="utf-8").rstrip()
        )
        assert embedded.get(name, "").rstrip() == expected, name
    all_source = "\n".join(embedded.values())
    config_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for profile in ("setup", "partial-2k", "partial-5k", "full"):
        assert profile in config_source, profile
    assert '"rgb_control", "predicted_density"' in config_source
    assert "batch_size = 64" in config_source
    assert "fog-colab-data.zip" in config_source
    assert "PROFILE_NAMES" in all_source
    print(
        "PASS: Colab notebook syntax, embedded modules, profiles, arms and batch-64 benchmark"
    )


if __name__ == "__main__":
    main()
