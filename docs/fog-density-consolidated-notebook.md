# Consolidated fog-density notebook

Open the current refactored copy in [Google Colab](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC).

`notebooks/fog_density_consolidated.ipynb` is a self-contained, readable version of the uploaded Colab notebook. It contains visible module cells rather than an escaped `EMBEDDED_FILES` dictionary. Running those cells writes the visible Python modules into the runtime for normal imports.

## Configuration

Edit only the `SETTINGS` cell. Data and output locations are independent:

- `{"kind": "drive", "path": "fog-study"}` resolves relative to the configured Drive mount and MyDrive directory.
- `{"kind": "path", "path": "/mnt/datasets/fog"}` uses an absolute filesystem directory.
- `data.mode` selects a prepared ZIP or a prepared directory.
- `output.location` and `output.run_name` determine where reports and training checkpoints are written.
- Choose `setup`, `partial-2k`, `partial-5k`, or `full` in `TRAINING_PROFILE`. Partial profiles start from pretrained weights and write to independent directories. The full profile points at the existing `colab-full-study` run by default; keep that name and its recorded settings when continuing it.
- Reuse the same profile, run name and training settings to resume. Change the run name for a different experiment.

The resolver builds `RESOLVED_PATHS` and `TRAINING_CONFIGURATION` once. Training settings do not come from a second hidden default configuration. A directory data source must contain the configured prepared manifests and their referenced images. ZIP input uses the same prepared layout; package both datasets with the repository's `scripts/package_colab_data.py --include-fog`.

Data may live in Drive while outputs use a mounted filesystem volume, or vice versa. Colab's `/content` storage is temporary. Drive outputs survive runtime replacement. Low-disk mode requires a Drive data source and prevents additional dataset downloads.

## Behavior and validation

### Cumulative continuation progress

Before training, the notebook lists every configured run with its saved overall training percentage, completed epochs, image steps within the current epoch, and evaluation status. Percentages count completed image steps across all planned epochs and include previous sessions. Each restored trainer prints cumulative loss history retained in the checkpoint. Overall training completion does not imply evaluation completion. Eleven CPU regression tests passed, including progress across earlier epochs; a real-data ZIP subset rerun passed with completed checkpoints and continuation output. See the [laptop handoff guide](laptop-handoff.md).

### Independent profiles and subset execution

The profile selector is the experiment switch. `setup` runs an all-model smoke profile; `partial-2k` and `partial-5k` run all three models and both comparison arms from pretrained weights on deterministic scene-safe training caps. They use fixed tuning rows for preliminary metrics and do not evaluate final-test rows. ZIP mode reads manifests first and extracts only selected referenced images rather than the entire roughly 53 GB Drive archive. Directory mode copies selected images into local profile storage.

Partial and setup profiles balance the labeled beta strata during Foggy Cityscapes density training, while reports retain natural per-bin metrics and add equal-bin macro rows. The full profile keeps natural sampling for continuity with the existing full-run checkpoint. nuScenes and Foggy Cityscapes remain separate analysis populations; density-only Foggy rows are never treated as detection negatives.

The legacy `SETTINGS["test"]` switch remains available for the bounded notebook checker. It intentionally narrows that checker to YOLO11n so notebook validation stays inexpensive; it is not a research profile.

Test checkpoints and reports use a separate output directory. Test preflight explicitly labels subset scope instead of claiming complete nuScenes coverage. The completion cell checks finite overall metrics, saved checkpoints and reports, and reruns completed training to exercise checkpoint reuse without rewriting checkpoints. This checks completed-run resume. The CPU regression suite additionally checks interrupted detector and density checkpoint recovery; a forced Colab runtime termination remains untested. A one-model checker does not validate YOLO11m or DETR execution; use the `setup` profile for that all-model smoke check.

The source dataset archive was found in `MyDrive/fog-study/fog-colab-data.zip`. The remote Colab run passed on a T4 GPU with 48 real images: 12 nuScenes detector images and 36 Foggy Cityscapes density images. Both YOLO11n arms completed training and evaluation, required checkpoints/reports were present, overall metrics were finite, and completed-run resume reused checkpoints without rewriting them. The notebook printed `SUBSET E2E PASSED` and exported the results successfully. The first Drive mount failed; the second attempt mounted successfully.

The local ZIP and prepared-directory tests both passed on the NVIDIA RTX 2070 using 24 real images, one YOLO11n model, one seed, one detector epoch and one density epoch. Both RGB-control and predicted-density arms completed training and evaluation. The executed notebooks contain finite metrics and rendered charts; the ZIP comparison chart was visually inspected. Checkpoint existence, completed-run resume without checkpoint rewrites, and ZIP export passed. Directory mode accepts prepared manifests with absolute paths inside the configured data root and rewrites selected image references relative to copied subset manifests. These small-data metrics are execution-test outputs, not scientific results.

Reproduce locally with `.venv/Scripts/python.exe scripts/check_consolidated_notebook.py --zip --images 2`. The runner executes the notebook's cells with IPython, including its visible module-writing cells. It reuses installed project dependencies and records filesystem overrides in the executed copy, leaving the source settings unchanged. Output is `reports/fog/consolidated-notebook-check/zip-executed.ipynb`; the run and ZIP export are saved beside it. Omit `--zip` to test a prepared directory source.

The notebook preserves the uploaded research modules and study defaults. The copied training module's unused `default_config` function has been removed so the notebook settings are the sole experiment defaults. Readable module filenames and report filenames remain internal code conventions.

Dataset extraction validates archive paths and free space. Its completion marker records archive identity and manifest settings instead of blindly trusting a fixed marker. Analysis skips dataset extraction and loading and reads existing run outputs.

Notebook format and Python cell syntax were checked with nbformat and compilation. Local GPU execution passed for both data modes as described above. Colab dependency installation and GPU/NMS validation now pass on Python 3.13 with Torch 2.11.0+cu130 and torchvision 0.26.0+cu130. Remote subset extraction, training, evaluation, completed-run resume and export all passed. A broader local `pip check` reports existing environment dependency conflicts (including OpenCV/NumPy and missing optional packages); the notebook's tested local paths still executed successfully. The original uploaded notebook is preserved. The legacy `create_colab_notebook.py` generator still produces the older standalone notebook; it does not generate this consolidated notebook.

### Colab dependency repair

The initial Colab setup tried to build NumPy 1.26.4 under Python 3.13 because the raw-data devkit restricts NumPy below 2. The [devkit's dependency list](https://github.com/nutonomy/nuscenes-devkit/blob/master/setup/requirements/requirements_base.txt) documents that cap; [NumPy's 1.26 release notes](https://numpy.org/doc/1.26/release/1.26.0-notes.html) describe its Python 3.12 support. This notebook requires already-prepared manifests and does not invoke raw nuScenes preparation, so the devkit dependency and unused preparation imports were removed. The remaining runtime requirements retain Colab's installed torch/torchvision pair, request wheels only, and show pip progress. Fresh dependency setup completed in 29 seconds on the observed T4 runtime.


### Verified Colab artifacts

- [Executed remote notebook](https://colab.research.google.com/drive/1ZO0-PYFjQe0xz021KelWqqiat3KLTHJN), with saved execution outputs.
- [Drive run directory](https://drive.google.com/drive/folders/1UyfxKRln9F4-JfOG3GBhpNBJtO8gy-mE): `MyDrive/fog-study/runs/colab-full-study-smoke-v2`.
- [Results ZIP](https://drive.google.com/file/d/1PMCcqrG1VChf4Qg96iONu6s30-EIjTbw/view): `colab-full-study-results-smoke-v2.zip` (138,293,716 bytes).
- Local execution record: `reports/fog/consolidated-notebook-check/colab-executed.ipynb`. Its cell sources preserve the version tested before the readability refactor, and saved outputs contain the passing completion check and export path with no error outputs.

This validates one model, one seed and one epoch on a real-data subset. Full-study completion, the other architectures in Colab, and forced remote runtime termination remain untested. Set `SETTINGS["test"]["enabled"] = False` to restore the configured full-study settings.


### Full-study launch

At the user's request, subset mode was disabled and Run all started in the remote Colab notebook. The resolver confirmed YOLO11n, YOLO11m and DETR; seeds 42/43/44; 30 detector epochs; 15 density epochs; both arms; image size 640; and checkpoints every 200 steps. Persistent output is `MyDrive/fog-study/runs/colab-full-study`. The T4 runtime is connected and full dataset preparation is running; full training/completion is not yet verified. The clean local notebook also has subset mode disabled.

### Readability refactor and checkpoint regression coverage

The notebook follows seven visible steps: configure, set up, load data, check readiness, train/evaluate, review, and export. Ordinary Python implementation cells sit under a collapsed section near the top, with one function or class per cell. They replace the escaped embedded-project text while keeping this notebook self-contained: the runtime writes the displayed source files before importing them. Expand that section to inspect or edit the implementation.

Training orchestration now uses named steps to create a session, restore state, train an epoch, measure validation loss, persist progress, and select the best weights. Density training and resumable evaluation use the same structure. Settings and resolved paths use descriptive names. The [coding standard](coding-standards.md) documents naming, function size, notebook structure, and checkpoint compatibility requirements.

## Density estimator by itself

Set `SETTINGS["action"] = "density"` to work only with the beta estimator. Choose `density_model.mode = "train"` to fit it from Foggy Cityscapes beta labels, or `"load"` to use a saved density checkpoint. A full profile trains from its complete beta-labeled training split; a partial profile applies its own configured caps and epochs from fresh initialization. This action never starts a detector.

The density action prepares only the required Foggy Cityscapes images; it does not extract nuScenes or bounding-box annotations. It saves the estimator checkpoint (when trained), `fog-density-predictions.csv`, `density-quality.csv`, and a small notebook view with a beta scatter plot. The quality table reports held-out MAE and RMSE in inverse meters (`m^-1`). A prediction split controls which rows are written; `"all"` writes predictions for every Cityscapes row in the dataset manifest.

Use the trained checkpoint as an input to a separate detector comparison by selecting `action = "train"`, `density_model.mode = "load"`, and setting `density_model.checkpoint`. Beta-estimation quality describes how well the estimator predicts the synthetic dataset's known beta values. It does not by itself demonstrate improved foggy-image detection; that question is answered by the detector comparison on labeled clear and foggy images.

Ten inexpensive CPU regression tests import the actual notebook modules. They cover original pre-refactor reference results, exact detector weights/optimizer/RNG/history parity after interruption and process loss, density checkpoint recovery, completed-run reuse, incompatible configuration rejection, partial prediction repair, prediction-order rejection, failed atomic saves, and concurrent-writer rejection. The interruption tests exposed a partial-accumulation issue: loss statistics and RNG state now roll back to the last completed optimizer boundary before unfinished work is replayed. Existing checkpoint payloads and filenames remain compatible.

Run `.venv/Scripts/python.exe -m unittest discover -s tests -v` and `.venv/Scripts/python.exe scripts/check_notebook_code.py` before expensive training. The code check includes the notebook readability guard: visible workflow cells stay short, orchestration uses named flat steps, every runtime helper cell begins collapsed, and generated support files must match the notebook. The original reference values are stored in `tests/fixtures/pre-refactor-detector-checkpoint.json`; they were generated from the preserved pre-refactor notebook, not the refactored implementation.

The refactored ZIP and prepared-directory subsets passed on the local RTX 2070 with 24 real images, both YOLO11n arms, density training, finite metrics, reports, checkpoints, completed-run reuse, and export. Syntax, lint, and formatting checks passed for the ordinary cells and assembled Python modules. These checks do not establish CUDA interruption parity, Drive timeout behavior, or full-study completion. The earlier successful T4 subset predates this refactor; this new Colab copy has not yet been executed remotely.

