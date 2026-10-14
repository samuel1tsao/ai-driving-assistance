"""Package the one self-contained fog study notebook for Colab."""

import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/fog_density_study.ipynb"


def build():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return notebook


def main():
    build()
    destination = ROOT / "dist/fog-colab-code.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(NOTEBOOK, NOTEBOOK.name)
        archive.write(
            ROOT / "scripts/package_colab_data.py", "scripts/package_colab_data.py"
        )
        archive.write(ROOT / "docs/colab-guide.md", "colab-guide.md")
    print(NOTEBOOK)
    print(destination)


if __name__ == "__main__":
    main()
