# Fog-density study notebook

Open the [current Google Colab notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC). The source is `notebooks/fog_density_consolidated.ipynb`; `fog_density_colab.ipynb` is an exact publication copy. All study implementation lives in the notebook. There are no support Python modules or runtime file-writing cells.

## Reading and configuring the notebook

The first section contains shared constants and `PROFILE_SETTINGS`. Model names, seeds, density scale, run names, checkpoint/report filenames, and training defaults are defined once there. Each profile owns its version. Setup keeps its fixed disposable folder; partial profiles start independently from pretrained detector weights; full continues the saved full experiment.

The **Run settings** cell chooses environment, profile, action, data/output locations, evaluation source, and the saved density-predictor checkpoint. Data and outputs can use either a Drive location relative to MyDrive or an absolute local path. Prepared ZIP input stages only the selected profile's images; directory input uses the same manifests and annotations.

The visible workflow is:

1. Connect storage and install dependencies.
2. Resolve the selected run and its recorded settings.
3. Load the fog-density model.
4. Prepare and validate data.
5. Train/resume detectors or evaluate existing weights.
6. Read tables and graphs; export checkpoints and reports.

Supporting functions are organized by purpose in collapsed cells. Expand them to inspect or edit implementation; their source is ordinary Python. The unused legacy adapter experiment and duplicate profile/default implementations were removed. Checkpoint recovery, matched validation, and analytics remain because they support the study's requested behavior.

## The fog-density setup actually loads a model

The visible predictor cell decides whether the density model is needed and selects the checkpoint. `resolve_density_predictor_checkpoint(configuration)` only chooses the saved path; `load_fog_density_model(checkpoint_path, device)` only restores weights, switches to evaluation mode, disables gradients, and returns the cached model. `get_fog_density(images)` uses that model in inference mode. A full run without a completed predictor checkpoint continues or trains the predictor inside the training workflow.

For a new partial profile, the default is the completed estimator from `colab-full-study/density-42/best.pt` under the configured run root. Change `SETTINGS["density_predictor"]["checkpoint"]` if it lives elsewhere. Evaluation loads the source experiment's estimator, so an unrelated current preset cannot silently change the predictor being tested. Missing partial-profile weights fail visibly. Preflight also checks the estimator format, beta units, and independent fitting scenes.

`analyze` and completed-run reuse read saved reports and skip model loading. The existing full run retains its original per-seed predictor/checkpoint continuation; any unfinished historical density training can still resume before its detectors run.

## Actions and checkpoint continuation

- `train`: continue the selected unfinished experiment or reuse completed/early-stopped comparisons. The saved run owns its training settings. `retrain=True` starts fresh in a separate folder; setup overwrites its fixed folder.
- `preflight`: display readiness checks before training.
- `analyze`: redraw the saved result tables and graphs without training or inference.
- `evaluate`: evaluate saved detector weights on matched labeled clear/fog images in a separate output directory.

Resume preserves detector and adapter weights, optimizer, AMP scaler, random-number state, cumulative loss history, and completed accumulation boundaries. Atomic checkpoint saves and one writer per run protect interrupted work. Changed inputs/settings/package versions cannot silently mix experiments. A presentation-only refactor can migrate compatible engine metadata while keeping checkpoints.

## What the comparison measures

Compare YOLO11n, YOLO11m, and DETR with RGB control and predicted density. Partial-2k and partial-5k use independent pretrained starts and fixed tuning rows; the full profile uses its held-out final set. The detector batch/accumulation contract is preserved. The optional synthetic batch-64 benchmark is separate from detector training.

The hypothesis benchmark needs matching `leftImg8bit/`, `leftImg8bit_foggy/`, and `gtFine/` assets. The same scenes and bounding-box labels are evaluated in clear, light, moderate, and dense fog conditions. Density-only fog images cannot become detection negatives. nuScenes has unknown physical beta, so its results are not a verified clear benchmark.

Outputs include clear/fog metrics, arm differences, seed variation, density-level curves, per-class AP, precision/recall, loss/history and checkpoint health, latency including density cost, sample counts, and the dataset audit. A single training seed cannot establish statistical significance. Loss histories describe the training/tuning split; clear/fog evaluation reports describe checkpoint detection performance on those conditions.

## Verification and publication

Run `python -m unittest discover -s tests -v`, `scripts/check_notebook_code.py`, and `scripts/check_colab.py` using the project Python. Tests load the notebook itself and cover interruption recovery, optimizer/RNG/history parity, checkpoint reuse, invalid inputs/leakage, concurrent writers, prediction recovery, frozen-predictor loading/reuse, and shared artifact names.

Run the directory and ZIP variants of `scripts/check_consolidated_notebook.py --images 2` after changing data or orchestration. Its tiny synthetic predictor is only an integration fixture, not a research result. CPU checks do not prove Colab CUDA, Drive mount, or runtime-termination behavior.

Run `scripts/create_colab_notebook.py` to publish the exact notebook mirror. Preserve the current Drive notebook, including any outputs, before replacing its contents. See the [coding standard](coding-standards.md) and [progress record](../progress.md).
