"""Check the standalone Colab notebook mirrors the self-contained study."""

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


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    sources = {name: [] for name in MODULE_NAMES}
    code_sources = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if "%%writefile" in cell.source:
            raise AssertionError(f"Colab cell {index} still writes a helper file")
        ast.parse(cell.source, filename=f"Colab cell {index}")
        filename = cell.metadata.get("fogStudyImplementation")
        if filename:
            if filename not in sources:
                raise AssertionError(f"unexpected embedded helper: {filename}")
            sources[filename].append(cell.source)
            continue
        code_sources.append(cell.source)
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("fog_")
            ):
                raise AssertionError(
                    f"Colab cell {index} imports a separate fog helper"
                )
    for filename in MODULE_NAMES:
        if not sources[filename]:
            raise AssertionError(f"Colab notebook does not contain {filename}")
    all_source = "\n".join(source for parts in sources.values() for source in parts)
    config_source = "\n".join(code_sources)
    for profile in ("setup", "partial-2k", "partial-5k", "full"):
        assert profile in config_source, profile
    assert '"rgb_control", "predicted_density"' in config_source
    assert "image_count, batch_size = 2000, 64" in all_source
    assert "fog-colab-data.zip" in config_source
    assert "PROFILE_NAMES" in all_source
    assert 'SETTINGS["density_predictor"]' in config_source
    print("PASS: self-contained Colab code, profiles, detection arms, and benchmark.")


if __name__ == "__main__":
    main()
