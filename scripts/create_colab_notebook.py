"""Publish the current profile notebook as the standalone Colab artifact."""

import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/fog_density_consolidated.ipynb"
TARGET = ROOT / "notebooks/fog_density_colab.ipynb"


def build():
    notebook = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(notebook)
    return notebook


def main():
    notebook = build()
    nbformat.write(notebook, TARGET)
    # Support modules are generated mirrors of the notebook's editable Python.
    for cell in notebook.cells:
        if cell.cell_type != "code" or not cell.source.startswith("%%writefile"):
            continue
        directive, source = cell.source.split("\n", 1)
        target = Path(directive.split()[-1])
        if target.parent != Path("notebooks/_support") or target.suffix != ".py":
            continue
        with (ROOT / target).open(
            "a" if " -a " in directive else "w", encoding="utf-8"
        ) as handle:
            handle.write(source + "\n")
    destination = ROOT / "dist/fog-colab-code.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(TARGET, TARGET.name)
        archive.write(
            ROOT / "scripts/package_colab_data.py", "scripts/package_colab_data.py"
        )
        archive.write(ROOT / "docs/colab-guide.md", "colab-guide.md")
    print(TARGET)
    print(destination)


if __name__ == "__main__":
    main()
