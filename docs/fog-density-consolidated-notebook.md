# Fog-density study notebook

Open the [current Google Colab notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC). The one local source is `notebooks/fog_density_study.ipynb`; upload that same file to Colab. All study implementation lives in the notebook. There are no duplicate notebook copies, support Python modules, or runtime file-writing cells.

## Reading and configuring the notebook

The first section contains shared constants and `PROFILE_SETTINGS`. Model names, seeds, density scale, run names, checkpoint/report filenames, and training defaults are defined once there. Each profile owns its version. Setup keeps its fixed disposable folder; partial profiles start independently from pretrained detector weights; full continues the saved full experiment.

The **Run settings** cell chooses environment, profile, data/output locations, evaluation source, and the density-predictor checkpoint. The archive URL is one constant, `DATA_ARCHIVE_URL`; the notebook reads the existing Drive ZIP when it is already in the configured folder.

A new developer should be able to follow the notebook from its visible cells in a few minutes:

1. **Load data:** use the configured archive and manifests to prepare the same full dataset for every profile.
2. **Check it:** confirm images, detection boxes, split boundaries, and fog-density labels.
3. **Fit the fog model:** beta is the dataset's numeric fog-density label; larger beta means denser synthetic fog. Restore its checkpoint and epoch count when available, continue unfinished epochs, or start with seeded default weights when absent. Save the best checkpoint and freeze this one model for every detector run.
4. **Train six detector runs:** YOLO11n, YOLO11m, and DETR each get a separate RGB-only run and a separate RGB-plus-estimated-density run. Each run has its own checkpoint and follows the selected profile's epoch limit.
5. **Analyze:** inspect the configured test split and reports. For the direct clear-versus-fog hypothesis test, run the matched evaluation cell; it scores each saved detector on the same labeled Cityscapes scenes in clear and fog conditions.
6. **Export:** package the selected profile's checkpoints and reports.

Supporting functions are organized by purpose in collapsed cells. Expand one only when you need its implementation; the visible cells show the order and purpose of the work. The A100 synthetic throughput check remains optional and uses batch size 64; real detector training keeps its image microbatch and gradient accumulation settings.

## Where the fog model and detector models are fitted

The visible fog-model cell calls `get_fog_model_training_data`, `resolve_density_predictor_checkpoint`, and `fit_fog_density_model`. It passes the full density-training split to one fit function. That function restores model weights, optimizer, and saved epoch progress when available; it continues unfinished training, stops at the configured epoch count (or saved early stop), and returns a frozen model. With no checkpoint, it starts from seeded default weights and saves a new one. `get_fog_density(images)` returns the model's estimated beta for detector use.

The next visible training cell calls `train_detector_profile`. Its comparison loop creates separate runs for every configured model and arm: YOLO11n, YOLO11m, and DETR with RGB only, then each with the fog estimate. Detector checkpoints and epoch counts are independent. Profile limits apply only when selecting detector training rows.

The configured test split measures the detector on its labeled images. Native nuScenes has no fog beta, so it cannot by itself provide a clear-versus-fog result. The separate matched analysis cell loads saved detector checkpoints and evaluates the same labeled Foggy Cityscapes scenes in clear and fog variants. It can run again without retraining detectors; the `gtFine` labels provide shared boxes.

## Training and checkpoint continuation

Rerunning the training cell resumes unfinished comparisons or reuses completed/early-stopped results. The saved run owns its training settings. `retrain=True` starts fresh in a separate folder; setup overwrites its fixed folder. The matched clear/fog cell is a separate high-level analysis step.

Resume preserves detector and adapter weights, optimizer, AMP scaler, random-number state, cumulative loss history, and completed accumulation boundaries. Atomic checkpoint saves and one writer per run protect interrupted work. Changed inputs/settings/package versions cannot silently mix experiments. A presentation-only refactor can migrate compatible engine metadata while keeping checkpoints.

## What the comparison measures

Compare YOLO11n, YOLO11m, and DETR with RGB control and predicted density. Partial-2k and partial-5k use independent pretrained starts and the same full validation split; the full profile uses its held-out final set. The A100-specific Colab benchmark settings remain unchanged; the optional synthetic batch-64 benchmark is a separate throughput check.

The hypothesis benchmark needs matching `leftImg8bit/`, `leftImg8bit_foggy/`, and `gtFine/` assets. The same scenes and bounding-box labels are evaluated in clear, light, moderate, and dense fog conditions. Density-only fog images cannot become detection negatives. nuScenes has unknown physical beta, so its results are not a verified clear benchmark.

Outputs include clear/fog metrics, arm differences, seed variation, density-level curves, per-class AP, precision/recall, loss/history and checkpoint health, latency including density cost, sample counts, and the dataset audit. A single training seed cannot establish statistical significance. Loss histories describe the training/tuning split; clear/fog evaluation reports describe checkpoint detection performance on those conditions.

## Verification and publication

Run `python -m unittest discover -s tests -v`, `scripts/check_notebook_code.py`, and `scripts/check_colab.py` using the project Python. Tests load the notebook itself and cover interruption recovery, optimizer/RNG/history parity, checkpoint reuse, invalid inputs/leakage, concurrent writers, prediction recovery, frozen-predictor loading/reuse, and shared artifact names.

Run the directory and ZIP variants of `scripts/check_consolidated_notebook.py --images 2` after changing data or orchestration. Its tiny synthetic predictor is only an integration fixture, not a research result. CPU checks do not prove Colab CUDA, Drive mount, or runtime-termination behavior.

Run `scripts/package_colab_notebook.py` to refresh the optional code bundle. It packages the same single notebook used locally and in Drive; it does not create a second notebook copy. Preserve the current Drive notebook, including any outputs, before replacing its contents.
