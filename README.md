# Fog-density object detection

Does an explicit fog-density estimate help camera-only detectors in fog? This
project fine-tunes **YOLO11n, YOLO11m and DETR** with and without a predicted fog
density and compares accuracy and latency. See [docs/research.md](docs/research.md)
for the research question, design and how to interpret results.

The baseline study lives in [`notebooks/fog_density_study.ipynb`](notebooks/fog_density_study.ipynb).
[`notebooks/film_experiment.ipynb`](notebooks/film_experiment.ipynb) compares all
three detectors with and without internal feature FiLM on clear and foggy images.
It exports overall and per-class AP, FiLM deltas, latency, resource metrics, and
charts under `models/film/<profile>/`. See [the FiLM experiment](docs/film-experiment.md)
for its architecture, matched comparisons, analytics, and checkpoint behavior.
Both notebooks run locally and in
[Google Colab](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC).
Contributor and agent conventions are in [AGENTS.md](AGENTS.md).

## Data

- **nuScenes** camera images and annotations train and evaluate the detectors.
  The prepared project bundle contains these files and the notebook's manifest.
- **Foggy Cityscapes** foggy images and beta labels train the frozen fog-density
  model. The same images are used for detector training with boxes derived from
  Cityscapes instance masks. The prepared project bundle contains the images and
  manifest.
- **Cityscapes** clear images and `gtFine` instance masks support those boxes and
  the clear/fog evaluation. Download `leftImg8bit_trainvaltest.zip` and
  `gtFine_trainvaltest.zip` from the [official Cityscapes site](https://www.cityscapes-dataset.com/)
  after registering and agreeing to its terms.

## Fresh local setup

Set `RUN_STORAGE = "local"` in the notebook. Prepared inputs are shared by all
profiles under `data/fog-study/`. Checkpoints and reports are kept outside the data
folder at `models/<profile>/<model>/`; profile names are stable and do not get
version or timestamp suffixes:

1. `prepare_fog_data()` needs `fog-colab-data.zip`. The notebook currently
   fetches this prepared bundle from [its Google Drive link](https://drive.google.com/uc?id=14fNOg5ysWcZyll6oW3Nr28ROmFnWSexU).
   If Drive reports its download quota, save a copy manually as
   `data/fog-study/fog-colab-data.zip`, then rerun the preparation cell. The
   notebook extracts it under `data/fog-study/` and checks the required
   manifests. The bundle includes nuScenes, Foggy Cityscapes, and prepared
   manifests; it does not include the separate Cityscapes inputs below.
2. Register at the [Cityscapes site](https://www.cityscapes-dataset.com/),
   download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`, and
   extract both into `data/fog-study/cityscapes-labeled/`. The resulting folders
   should be `leftImg8bit/` and `gtFine/` directly under `cityscapes-labeled/`.
3. In the notebook, select `RUN_PROFILE = PROFILE_SETUP` for the first run and
   run cells in order. `prepare_cityscapes_data()` recognizes the extracted
   layout; then the notebook loads both datasets, fits or restores the
   fog-density model, and trains or resumes the detectors. Select a larger
   profile after setup succeeds.

The archive's Google Drive quota error means the shared bundle is temporarily
unavailable from that link. It is not a model-training error. The [official
nuScenes download page](https://www.nuscenes.org/nuscenes) requires an account
and acceptance of its terms; Foggy Cityscapes is distributed through
[Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) because of its
source-data licensing. Replacing the prepared bundle with direct source
downloads requires preparing the notebook's manifests as well.

## Run on Google Colab

1. Upload the notebook and select a GPU runtime.
2. Keep the dataset ZIP in Drive. About 64.25 GiB free is required (49.25 GiB data
   plus 15 GiB working space).
3. In **Run settings**, choose the profile, data directory, output directory,
   models, and seed. The default data directory is
   `/content/drive/MyDrive/fog-study`.
   For local runs, the notebook stores shared prepared data in `data/fog-study`
   and model artifacts in `models/<profile>/<model>/`. Kaggle caches use
   `data/datasets`.
4. Run all. Dependencies and model weights are installed/downloaded as needed.

## Using the notebook

The first section contains fixed study contracts and profile defaults. In code
cell 4 (**Run settings**), configure only:

- `RUN_PROFILE`: `PROFILE_SETUP`, `PROFILE_2K`, `PROFILE_5K`, or `PROFILE_FULL`
- `RUN_STORAGE`: `"drive"` or `"local"`
- `RUN_DIRECTORY`: the shared prepared-data root; checkpoints and reports are stored under `models/<profile>/<model>/`
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

Rerun with the same profile to resume; completed model checkpoints are reused.
Resume restores detector/adapter weights, optimizer, AMP scaler, RNG state and loss
history. Checkpoints save atomically, one writer per model comparison is enforced,
and changed inputs/settings/package versions are rejected. Each profile uses a
stable directory; remove that profile's model directory only when intentionally
starting its training from scratch. Only work in a completed checkpoint survives a
runtime loss.

## Outputs

Each stable profile folder contains `settings.json` and the input manifest. Its
`<model>/<comparison-arm>/seed-<seed>/` folders contain resumable checkpoints,
model metrics, predictions, timings, and loss histories. The fog-density model is
stored under `fog-density/seed-<seed>/`. Matched evaluation uses the profile's
`matched-evaluation/` folder. The notebook recomputes study tables in memory.
