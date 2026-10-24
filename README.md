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
3. In **Run settings**, choose the profile, data directory, output directory,
   models, and seed. The default data directory is
   `/content/drive/MyDrive/fog-study`.
4. Run all. Dependencies and model weights are installed/downloaded as needed.

## Using the notebook

The first section contains fixed study contracts and profile defaults. In code
cell 4 (**Run settings**), configure only:

- `RUN_PROFILE`: `PROFILE_SETUP`, `PROFILE_2K`, `PROFILE_5K`, or `PROFILE_FULL`
- `RUN_STORAGE`: `"drive"` or `"local"`
- `RUN_DIRECTORY`: the single folder containing the dataset, checkpoints, and outputs
- `RUN_MODELS`: the model names to compare
- `RUN_SEED`: the random seed

The notebook executes in numbered sections:

1. **Profile and run settings**: choose environment, profile, and data/output locations.
2. **Runtime setup**: mount Drive (if on Colab), set directories, and install packages.
3. **Prepare study data**: unpack and organize datasets via `prepare_fog_data()`.
4. **Load datasets and fit fog model**: load manifests, prepare labeled Cityscapes pairs (`prepare_cityscapes_data()`), and fit or restore the fog model (`fit_fog_density_model()`).
5. **Train and evaluate detectors**: for each comparison, load or initialize the model via `load_or_create_model_file()`, train or resume it via `train_detector()`, and evaluate it via `evaluate_detector()`. This section also includes the optional matched clear/fog evaluation (`run_saved_fog_comparison()`).
6. **Review outputs**: compute analytics tables and display comparison charts (`generate_study_analytics()`, `display_saved_comparison_figures()`).
7. **Export the run**: package checkpoints and reports via `export_checkpoints_and_reports()`.

- **Fog model:** `fit_fog_density_model` trains on the full density split (or restores its checkpoint) and is frozen for all detector runs.
- **Detectors:** `load_or_create_model_file` restores the current resumable state from `last.pt` or initializes fresh models. `best.pt` holds the lowest held-out-loss weights observed so far; `train_detector` trains or reuses completed checkpoints, and `evaluate_detector` evaluates those best weights on the held-out split. Profile caps apply only to detector-training rows.
- **Training defaults:** 640 px inputs, one-image micro-batches with four accumulated gradients, mixed precision on CUDA. The optional A100 throughput benchmark uses batch 64 and does not affect detector training.

## Checkpoints and resume

Rerun with the same settings and output folder to resume; completed runs are
reused. Resume restores detector/adapter weights, optimizer, AMP scaler, RNG state
and loss history. Checkpoints save atomically, one writer per run is enforced, and
changed inputs/settings/package versions are rejected. `retrain=True` starts fresh
in a new folder; `setup` overwrites its disposable folder. Only work in a
completed checkpoint survives a runtime loss.

## Outputs

Each run folder contains the resumable training state: `settings.json`, the
input manifest, per-run `last.pt` and `best.pt`, `trained-model.pt` (inference
only), loss histories, and prediction/timing intermediates. The review section
recomputes result, paired, density-quality, and hypothesis tables in memory and
displays them in the notebook; those findings are not persisted as report CSVs.
