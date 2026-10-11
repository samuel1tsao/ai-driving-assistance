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
    nbformat.write(build(), TARGET)
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
