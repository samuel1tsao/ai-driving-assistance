# Run the fog study in Google Colab

Open the [current Drive notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC), or upload `notebooks/fog_density_study.ipynb` and choose a GPU runtime. All study code is in the notebook; no repository checkout, support modules, or file-writing cells are needed. See the [notebook guide](fog-density-consolidated-notebook.md) for the workflow and analytics.

## Configuration

Shared constants and `PROFILE_SETTINGS` are near the top. Change a profile's version there for a new experiment. Setup always uses its disposable fixed folder. **Run settings** chooses the environment, profile, data/output paths, evaluation source, and density checkpoint.

For Drive storage, use locations relative to MyDrive:

```python
SETTINGS["data"]["location"] = {"kind": "drive", "path": "fog-study"}
SETTINGS["output"]["location"] = {"kind": "drive", "path": "Research/Fog results"}
```

This reads `fog-colab-data.zip` from `MyDrive/fog-study/` and writes runs under `MyDrive/Research/Fog results/`. The archive download URL is defined once as `DATA_ARCHIVE_URL`; the notebook uses the Drive copy when it is already present. A local archive can use `kind="path"` with its containing folder.

All profiles run the same data preparation: load the dataset ZIP, extract its prepared images and manifests, then load the annotations. Training applies the selected profile's image caps when it creates training partitions. No profile gets a separate copied dataset. Connect Drive, install dependencies, run the collapsed definitions, and resolve the configuration. The density step loads the saved predictor or trains and saves it if it is missing. `get_fog_density(images)` uses the loaded predictor.

Rerunning the training cells resumes unfinished comparisons or reuses completed results. `retrain=True` starts fresh detectors in a sibling folder and preserves the old run. Setup overwrites its fixed output. The independent matched analysis cell reads saved detector checkpoints and displays the fog/clear comparison. After changing settings, rerun configuration and the following workflow cells, or restart and Run all.

Each partial profile starts independently; partial-5k does not continue partial-2k weights. All three models run both comparison arms. The full profile keeps its recorded 30-epoch, multi-seed experiment. Training retains its existing batch/gradient-accumulation contract; the optional synthetic batch-64 benchmark is a separate throughput check.

Checkpoints preserve detector/adapter weights, optimizer, AMP scaler, random state, and cumulative loss history. One writer per run and atomic saves protect interrupted work. Changed inputs or incompatible settings/packages cannot silently mix experiments. A presentation-only notebook refactor does not require retraining compatible weights.

## Labeled Cityscapes data and analysis

Detector training and the separate analysis cell both use one labeled Cityscapes manifest. Set `SETTINGS["evaluation"]["cityscapes_root"]` to a folder containing the official Cityscapes clear images, Foggy Cityscapes images, and `gtFine` instance masks:

```text
cityscapes-labeled/
  leftImg8bit/val/<city>/*_leftImg8bit.png
  leftImg8bit_foggy/val/<city>/*_leftImg8bit_foggy_beta_<beta>.png
  gtFine/val/<city>/*_gtFine_instanceIds.png
```

Each scene needs clear plus beta .005, .01, and .02 fog views with bounding boxes extracted from its instance mask. The data ZIP contains the fog images and beta labels, but not the Cityscapes `gtFine` masks or clear images. Keep this labeled root available in Drive before running training. The detector training step uses labeled training scenes; the independent analysis cell selects only `evaluation.split` from the same manifest.

Set `evaluation.source_run=""` to use the selected profile's saved run, or provide another run path. `max_pairs=100` selects 100 complete held-out scene groups (400 images); `None` uses all available scenes. Predictions, tables, figures, and export ZIP go to `evaluation.output_name`; source weights stay intact. Compare fog-minus-control and clear-minus-control deltas for each model and training seed. One seed is exploratory; correlated fog variants are not independent experimental repeats. Unknown-beta nuScenes results are a separate normal-use benchmark, not verified clear performance.

For a visual record, save a notebook copy with outputs after running. CSVs and PNGs persist in the run even if cell outputs are cleared. `analyze` redraws them without another training run.

## Data bundle and publication

The existing `fog-colab-data.zip` contains prepared nuScenes images/labels and density-only Foggy Cityscapes rows. It measures about 49.25 GiB extracted. Full extraction with the 15 GiB working reserve requires about 64.25 GiB free. Every profile uses the same extracted data; 2k and 5k limits affect only the rows used for training. Code-only changes do not require rebuilding the data archive.

Build data only when it changes:

```powershell
.venv/Scripts/python.exe scripts/package_colab_data.py --include-fog
```

The final notebook step exports its run to a ZIP beside the output directory. Colab runtime-local files can disappear after a runtime replacement, so use Drive outputs for checkpoint continuation. Save a Drive notebook copy before replacing the main notebook if it has outputs you want to review.

Edit the notebook directly, then create its exact publication copy and optional code bundle:

```powershell
.venv/Scripts/python.exe scripts/package_colab_notebook.py
.venv/Scripts/python.exe scripts/check_notebook_code.py
.venv/Scripts/python.exe scripts/check_colab.py
```

The optional `fog-colab-code.zip` contains the standalone notebook, this guide, and the data-packaging script; it contains no support implementation. Unit tests and bounded notebook checks execute the notebook source itself. Local tests do not establish full-study scientific results or hosted Colab runtime behavior.
