# Fog-density study notebook

Open the [current Google Colab notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC). The one local source is `notebooks/fog_density_study.ipynb`; upload that same file to Colab. All study implementation lives in the notebook. There are no duplicate notebook copies, support Python modules, or runtime file-writing cells.

## Reading and configuring the notebook

The first section contains shared constants and `PROFILE_SETTINGS`. Model names, seeds, density scale, run names, checkpoint/report filenames, and training defaults are defined once there. Each profile owns its version. Setup keeps its fixed disposable folder; partial profiles start independently from pretrained detector weights; full continues the saved full experiment.

The **Run settings** cell chooses environment, profile, data/output locations, evaluation source, and the density-predictor checkpoint. The archive URL is one constant, `DATA_ARCHIVE_URL`; the notebook reads the existing Drive ZIP when it is already in the configured folder.

The visible workflow is:

1. Load and prepare the same full dataset for every profile.
2. Validate the annotations.
3. Load the saved, frozen fog-density model; if no checkpoint exists, train it once on the full density-training split, save its best checkpoint, then keep it frozen. Partial-profile limits apply only to detector rows.
4. Apply profile row limits inside the training partitions, then train or resume the detector comparisons.
5. Run the independent matched clear/fog analysis and review its tables and graphs.
6. Export checkpoints and reports.

Supporting functions are organized by purpose in collapsed cells. Expand them to inspect or edit implementation; their source is ordinary Python. The unused legacy adapter experiment and duplicate profile/default implementations were removed. Checkpoint recovery, matched validation, and analytics remain because they support the study's requested behavior.

## The fog-density setup actually loads a model

The visible predictor cell calls `load_or_train_fog_density_model(configuration, rows)`. It loads a saved checkpoint when present; otherwise it trains the image-to-beta model from the loaded density rows and saves it. `get_fog_density(images)` uses that model during detector training and analysis.

Profile limits do not change dataset preparation. Each profile reads the same manifests and image paths; the training partition step applies the configured training, density, and tuning caps. The full profile keeps its previous row order and checkpoint settings so an existing full run can continue.

The independent analysis cell loads the source run's saved density and detector checkpoints if they are not already in memory. It does not require another detector training run. Matched analysis needs labeled Foggy Cityscapes clear/fog pairs; the configured `gtFine` data supplies the shared boxes.

## Training and checkpoint continuation

Rerunning the training cell resumes unfinished comparisons or reuses completed/early-stopped results. The saved run owns its training settings. `retrain=True` starts fresh in a separate folder; setup overwrites its fixed folder. The matched clear/fog cell is a separate high-level analysis step.

Resume preserves detector and adapter weights, optimizer, AMP scaler, random-number state, cumulative loss history, and completed accumulation boundaries. Atomic checkpoint saves and one writer per run protect interrupted work. Changed inputs/settings/package versions cannot silently mix experiments. A presentation-only refactor can migrate compatible engine metadata while keeping checkpoints.

## What the comparison measures

Compare YOLO11n, YOLO11m, and DETR with RGB control and predicted density. Partial-2k and partial-5k use independent pretrained starts and fixed tuning rows; the full profile uses its held-out final set. The A100-specific Colab benchmark settings remain unchanged; the optional synthetic batch-64 benchmark is a separate throughput check.

The hypothesis benchmark needs matching `leftImg8bit/`, `leftImg8bit_foggy/`, and `gtFine/` assets. The same scenes and bounding-box labels are evaluated in clear, light, moderate, and dense fog conditions. Density-only fog images cannot become detection negatives. nuScenes has unknown physical beta, so its results are not a verified clear benchmark.

Outputs include clear/fog metrics, arm differences, seed variation, density-level curves, per-class AP, precision/recall, loss/history and checkpoint health, latency including density cost, sample counts, and the dataset audit. A single training seed cannot establish statistical significance. Loss histories describe the training/tuning split; clear/fog evaluation reports describe checkpoint detection performance on those conditions.

## Verification and publication

Run `python -m unittest discover -s tests -v`, `scripts/check_notebook_code.py`, and `scripts/check_colab.py` using the project Python. Tests load the notebook itself and cover interruption recovery, optimizer/RNG/history parity, checkpoint reuse, invalid inputs/leakage, concurrent writers, prediction recovery, frozen-predictor loading/reuse, and shared artifact names.

Run the directory and ZIP variants of `scripts/check_consolidated_notebook.py --images 2` after changing data or orchestration. Its tiny synthetic predictor is only an integration fixture, not a research result. CPU checks do not prove Colab CUDA, Drive mount, or runtime-termination behavior.

Run `scripts/package_colab_notebook.py` to refresh the optional code bundle. It packages the same single notebook used locally and in Drive; it does not create a second notebook copy. Preserve the current Drive notebook, including any outputs, before replacing its contents. See the [coding standard](coding-standards.md) and [progress record](../progress.md).
