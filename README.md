# Fog-density object detection

Does an explicit fog-density estimate help camera-only detectors in fog? This
project fine-tunes **YOLO11n, YOLO11m and DETR** with and without a predicted fog
density and compares accuracy and latency. See [docs/research.md](docs/research.md)
for the research question, design and how to interpret results.

Everything lives in one notebook,
[`notebooks/fog_density_study.ipynb`](notebooks/fog_density_study.ipynb), used both
locally and in [Google Colab](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC).
Contributor and agent conventions are in [AGENTS.md](AGENTS.md).

## Data

- **nuScenes** trainval keyframes (detection boxes).
- **Foggy Cityscapes** images (beta labels parsed from filenames) for the density
  model. Included in the dataset ZIP.
- **Cityscapes `gtFine/` + matched `leftImg8bit/`** for the clear/fog benchmark.
  Not in the ZIP; place them in the configured `cityscapes-labeled` folder.
  Missing annotations stop data loading rather than becoming negatives.

## Run on Google Colab

1. Upload the notebook and select a GPU runtime.
2. Keep the dataset ZIP in Drive. About 64.25 GiB free is required (49.25 GiB data
   plus 15 GiB working space).
3. In **Run settings**, choose the profile and set `SETTINGS["data"]["location"]`
   to the Drive folder containing the ZIP.
4. Run all. Dependencies and model weights are installed/downloaded as needed.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/fog_density_study.ipynb
```

Set `RUN_ENVIRONMENT="local"` and point the data and output locations in
**Run settings** at your data and run root.

## Using the notebook

In the first section, shared constants and `PROFILE_SETTINGS` define study defaults.
In code cell 4 (**Run settings**), configure:
- `RUN_ENVIRONMENT`: `"colab"` or `"local"`
- `TRAINING_PROFILE`: `PROFILE_SETUP` (`"setup"`), `PROFILE_2K` (`"partial-2k"`), `PROFILE_5K` (`"partial-5k"`), or `PROFILE_FULL` (`"full"`)
- `SETTINGS["retrain"]`: `False` (default, resumes or reuses saved checkpoints) or `True` (starts a fresh run)

The notebook executes in numbered sections:

1. **Profile and run settings**: choose environment, profile, and data/output locations.
2. **Runtime setup**: mount Drive (if on Colab), set directories, and install packages.
3. **Prepare study data**: unpack and organize datasets via `prepare_fog_data()`.
4. **Load datasets and fit fog model**: load manifests, prepare labeled Cityscapes pairs (`prepare_cityscapes_data()`), and fit or restore the fog model (`fit_fog_density_model()`).
5. **Train and validate detectors**: for each comparison, load or initialize the model via `load_or_create_model_file()`, train or resume it via `train_detector()`, and evaluate it via `validate_detector()`. This section also includes the optional matched clear/fog evaluation (`run_saved_fog_comparison()`).
6. **Review outputs**: compute analytics tables and display comparison charts (`generate_study_analytics()`, `display_saved_comparison_figures()`).
7. **Export the run**: package checkpoints and reports via `export_checkpoints_and_reports()`.

- **Fog model:** `fit_fog_density_model` trains on the full density split (or restores its checkpoint) and is frozen for all detector runs.
- **Detectors:** `load_or_create_model_file` restores from `checkpoint.pt` or initializes fresh models. `train_detector` trains or reuses completed checkpoints, and `validate_detector` evaluates on the shared validation split. Profile caps apply only to detector-training rows.
- **Training defaults:** 640 px inputs, one-image micro-batches with four accumulated gradients, mixed precision on CUDA. The optional A100 throughput benchmark uses batch 64 and does not affect detector training.

## Checkpoints and resume

Rerun with the same settings and output folder to resume; completed runs are
reused. Resume restores detector/adapter weights, optimizer, AMP scaler, RNG state
and loss history. Checkpoints save atomically, one writer per run is enforced, and
changed inputs/settings/package versions are rejected. `retrain=True` starts fresh
in a new folder; `setup` overwrites its disposable folder. Only work in a
completed checkpoint survives a runtime loss.

## Outputs

Each run folder contains `settings.json`, the input manifest, per-run `last.pt`
(resumable) and `best.pt`, `trained-model.pt` (inference only), loss histories,
predictions/timings, `results.csv`, `paired-deltas.csv`, `seed-summary.csv`,
`density-quality.csv`, `fog-vs-clear.csv` and a `figures/` directory.
