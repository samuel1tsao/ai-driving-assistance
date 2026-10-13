# Run the fog study in Google Colab

Open the [current Drive notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC), or upload `notebooks/fog_density_colab.ipynb` and choose a GPU runtime. All study code is in the notebook; no repository checkout, support modules, or file-writing cells are needed. See the [notebook guide](fog-density-consolidated-notebook.md) for the workflow and analytics.

## Configuration

Shared constants and `PROFILE_SETTINGS` are near the top. Change a profile's version there for a new experiment. Setup always uses its disposable fixed folder. **Run settings** chooses the environment, profile, action, data/output paths, evaluation source, and pretrained density checkpoint.

For Drive storage, use locations relative to MyDrive:

```python
SETTINGS["data"]["location"] = {"kind": "drive", "path": "Research/Fog study"}
SETTINGS["output"]["location"] = {"kind": "drive", "path": "Research/Fog results"}
SETTINGS["data"]["mode"] = "zip"
```

This reads the configured data archive from `MyDrive/Research/Fog study/` and writes runs under `MyDrive/Research/Fog results/`. For a prepared directory, use `mode="directory"` and point at the folder containing `data/` and the referenced images. An absolute local path uses `kind="path"`. A partial profile stages its scene-safe subset locally from either source.

Connect Drive, install dependencies, run the collapsed definitions, and resolve the configuration. The visible **Load the fog-density model** cell actually restores the checkpoint, switches it to evaluation mode, freezes its parameters, and prints the loaded path and state. `get_fog_density(images)` uses that predictor. New partial profiles default to the completed full-run estimator; set `density_predictor.checkpoint` if it moved. Full keeps its existing per-seed continuation.

## Continue, reuse, analyze, or retrain

- `action="train", retrain=False`: resume unfinished comparisons with their recorded training settings, or reuse completed/early-stopped results. Setup overwrites its fixed output.
- `action="preflight"`: display readiness checks.
- `action="analyze"`: read saved CSVs and redraw graphs; no training, inference, or predictor loading.
- `action="evaluate"`: evaluate saved detector weights on matched labeled clear/fog images in a separate output.

`retrain=True` with `train` starts fresh detectors in a sibling folder and preserves the old run. The pretrained density predictor remains the input predictor. To resume that new run later, set `output.run_name` to the printed folder and turn retrain off. After changing settings, rerun configuration and the following workflow cells, or restart and Run all.

Each partial profile starts independently; partial-5k does not continue partial-2k weights. All three models run both comparison arms. The full profile keeps its recorded 30-epoch, multi-seed experiment. Training retains its existing batch/gradient-accumulation contract; the optional synthetic batch-64 benchmark is a separate throughput check.

Checkpoints preserve detector/adapter weights, optimizer, AMP scaler, random state, and cumulative loss history. One writer per run and atomic saves protect interrupted work. Changed inputs or incompatible settings/packages cannot silently mix experiments. A presentation-only notebook refactor does not require retraining compatible weights.

## Matched clear/fog evaluation

Choose the source training profile and set `action="evaluate"`. `evaluation.source_run=""` uses its saved run; an absolute path selects another. Match that experiment's models, seeds, and image size.

The evaluation root needs:

```text
cityscapes-labeled/
  leftImg8bit/val/<city>/*_leftImg8bit.png
  leftImg8bit_foggy/val/<city>/*_leftImg8bit_foggy_beta_<beta>.png
  gtFine/val/<city>/*_gtFine_instanceIds.png
```

Every scene needs clear, beta .005, .01, and .02 views with shared bounding-box labels from the instance mask. The fog-image archive alone lacks those labels. `max_pairs=100` selects 100 complete scene groups (400 images); `None` uses all available scenes. Missing variants/labels or fitting-scene overlap stops evaluation.

New predictions, tables, figures, and export ZIP go to `evaluation.output_name`; source weights stay intact. Evaluation loads that source experiment's density estimator. Compare fog-minus-control and clear-minus-control deltas for each model and training seed. One seed is exploratory; correlated fog variants are not independent experimental repeats. Unknown-beta nuScenes results are a separate normal-use benchmark, not verified clear performance.

For a visual record, save a notebook copy with outputs after running. CSVs and PNGs persist in the run even if cell outputs are cleared. `analyze` redraws them without another training run.

## Data bundle and publication

The existing `fog-colab-data.zip` contains prepared nuScenes images/labels and density-only Foggy Cityscapes rows. The prepared bundle measures about 49.25 GiB extracted. Full ZIP extraction with the 15 GiB working reserve requires about 64.25 GiB free; partial profiles stage fewer images. Low-disk mode requires a mounted Drive archive and blocks extra dataset downloads. Code-only changes do not require rebuilding the data archive.

Build data only when it changes:

```powershell
.venv/Scripts/python.exe scripts/package_colab_data.py --include-fog
```

The final notebook step exports its run to a ZIP beside the output directory. Colab runtime-local files can disappear after a runtime replacement, so use Drive outputs for checkpoint continuation. Save a Drive notebook copy before replacing the main notebook if it has outputs you want to review.

Edit the notebook directly, then create its exact publication copy and optional code bundle:

```powershell
.venv/Scripts/python.exe scripts/create_colab_notebook.py
.venv/Scripts/python.exe scripts/check_notebook_code.py
.venv/Scripts/python.exe scripts/check_colab.py
```

The optional `fog-colab-code.zip` contains the standalone notebook, this guide, and the data-packaging script; it contains no support implementation. Unit tests and bounded notebook checks execute the notebook source itself. Local tests do not establish full-study scientific results or hosted Colab runtime behavior.
