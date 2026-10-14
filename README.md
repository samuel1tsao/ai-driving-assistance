# Fog-density object detection research

Open the [fog density study notebook](notebooks/fog_density_consolidated.ipynb). It runs independent `setup`, `partial-2k`, `partial-5k`, and `full` profiles across **YOLO11n, YOLO11m and DETR**, then exports accuracy, latency, per-class AP, paired differences and charts. See the [research plan](docs/fog-density-research-plan.md) and the [two-dataset validation implementation plan](docs/two-dataset-fog-validation-implementation-plan.md).

The detector study is the notebook's purpose. Its predicted-density arm uses a frozen beta estimator as an input predictor; the paired RGB control receives zero beta. Partial profiles load the completed full-run estimator from `colab-full-study/density-42/best.pt` by default, so they train object detectors and adapters without retraining the density model. Use `"evaluate"` to test saved detector weights on matched labeled clear/fog images, and `"analyze"` to redraw saved reports. The consolidated notebook is the sole implementation source; its Colab copy is an exact mirror. Test and CLI tools execute notebook definitions directly; no support modules are generated or imported. Shared constants and profile settings are centralized near the top, and the visible predictor step actually loads and freezes the saved model. See the [consolidated notebook guide](docs/fog-density-consolidated-notebook.md) and [Colab guide](docs/colab-guide.md).

## Run on Google Colab

See the [proposed setup, partial and full training profiles](docs/training-run-profiles.md) for bounded preliminary results and early stopping. See the [progressive-stage implementation plan](docs/training-run-profiles-implementation-plan.md) and [latest Colab source refresh](docs/colab-drive-sync.md).

For profile configuration, checkpoint continuation and validation notes, see [the consolidated notebook guide](docs/fog-density-consolidated-notebook.md). Partial profiles use fixed tuning rows for preliminary analysis; the full profile continues the existing full run and evaluates the final held-out set.

Shared names/defaults and `PROFILE_SETTINGS` live in the first code cell. Choose the run in **Run settings** and set `SETTINGS["data"]["location"]` to the Drive folder or local folder containing the configured dataset ZIP. Every profile prepares the same full dataset; profile caps are applied when training partitions are made. See [configuration](docs/colab-guide.md#configuration).

Upload [the standalone Colab notebook](notebooks/fog_density_colab.ipynb) and select a GPU runtime. Keep the complete dataset ZIP in Drive; the notebook extracts the same data for every profile and requires about 64.25 GiB free (49.25 GiB data plus 15 GiB working space). It embeds the code, installs dependencies without a pip download cache, and downloads model weights when needed. Follow the [Colab guide](docs/colab-guide.md). The optional code bundle is `dist/fog-colab-code.zip`.

## What was fixed

For the comparison of the Colab-edited `Untitled0.ipynb` with the standalone baseline, see the [notebook comparison and storage review](docs/colab-notebook-comparison.md).

- Detector weights now train. Previously only an image adapter trained around frozen detectors.
- Each model/arm starts from the same pretrained weights; state cannot leak from one comparison to another.
- nuScenes preparation covers all six cameras and all labeled trainval keyframes, records missing files, preserves scene splits and resumes by scene. Projected boxes use convex-hull clipping; clipping a bounding rectangle was not equivalent.
- The default comparison is no density versus **estimated** density. Unknown nuScenes beta no longer crashes a mandatory oracle arm.
- Training saves detector, adapter, optimizer, AMP scaler and RNG state. Interrupted work resumes; completed runs are skipped. Modified settings, inputs or code require a new run folder.
- Errors stop visibly with saved status. Analytics can be regenerated without training, and plotting works without a desktop GUI.
- DETR loads the official `no_timm` checkpoint with compatibility normalization for its older null configuration fields.
- YOLO uses compiled torchvision NMS, and mixed precision starts with a conservative gradient scale.

## Current data on this computer

The completed scan now finds **204,894 of 204,894 expected nuScenes camera keyframes**. There are **0 missing images** across all 850 trainval scenes. Prepared splits contain 150,948 training, 17,832 tuning and 36,114 final-test images. The [coverage report](data/fog/nuscenes_all.coverage.json) records `complete: true`. The ten source archives are preserved under `data/depth/datasets/nuscenes/archives/`.

Your supplied `Untitled2.ipynb` downloaded 15,000 Foggy Cityscapes images into `/root/.cache/kagglehub/datasets/minhducnguyen9705/foggy-cityscapes/versions/1` in a Linux environment. The project now has the corresponding 15,000-image cache under `.cache/kagglehub/`. The loader checks the project cache, common Downloads folders and Kaggle cache locations, accepts an explicit dataset root, or downloads the same source when absent.

**Foggy Cityscapes images alone are sufficient to train the density estimator.** Beta is parsed from filenames; nuScenes supplies detection annotations. `gtFine` is only needed for the optional separate Cityscapes detection benchmark. Density-only images are explicitly excluded from detector training/evaluation, so missing annotations never become negative examples.

## Run in Jupyter

From this project directory in PowerShell:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-fog-research.txt
.venv/Scripts/python.exe -m jupyter lab notebooks/fog_density_consolidated.ipynb
```

Use the project Python kernel, then review the first settings cell:

1. Set `RUN_ENVIRONMENT="local"` and point the **Run settings** data and output locations at your prepared data and chosen run root.
2. Select the training profile and verify the saved density-predictor checkpoint path.
3. Use `SETTINGS["action"]="train"` and **Restart Kernel and Run All** to resume/train, or `"preflight"` to check readiness.
4. Use `"analyze"` for saved reports or `"evaluate"` for the labeled matched clear/fog benchmark.

The full profile uses 30 detector epochs, 15 density epochs, seeds 42/43/44, 640-pixel square inputs, single-image microsteps with four accumulated gradients, and mixed-precision training on CUDA. Partial profiles use seed 42, five epochs, deterministic scene-safe caps and a frozen, already-trained density predictor. The full profile keeps natural density sampling so an existing full checkpoint can continue without changing its training distribution. The separate synthetic GPU benchmark uses batch 64; real detector training remains batch-one with accumulation four.

## Run from a terminal

Prepare all available local nuScenes images and download/prepare Foggy Cityscapes if needed:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py prepare --download-fog
```

To reuse a dataset directory instead of downloading it:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py prepare --fog-root "C:/path/to/foggy-cityscapes"
```

Run preparation and confirm complete coverage before training:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py preflight
.venv/Scripts/python.exe scripts/run_fog_research.py train
```

These commands resume automatically when repeated. JSON overrides are supported with `--config path.json`; valid fields are documented by `default_training_configuration` in the notebook. Public detector weights can be fetched independently:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py download-models
```

If you want to start the no-density baseline before obtaining Foggy Cityscapes, use a separate folder:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py train --baseline-only --output reports/fog/baseline
```

That baseline-only run does not establish the with/without-density comparison. Run both matched arms in one experiment for the paired study.

Regenerate reports from completed experiments, including completed models in an interrupted run:

```powershell
.venv/Scripts/python.exe scripts/run_fog_research.py analyze --output reports/fog/full-training-complete
```

## Outputs and interpretation

Each run contains `settings.json`, the exact input manifest, per-model/arm/seed `last.pt` and `best.pt`, `progress.json`, `trained-model.pt`, loss histories, raw predictions/timings, status, `results.csv`, `paired-deltas.csv`, `seed-summary.csv`, `density-quality.csv`, and a `figures/` directory. `last.pt` resumes optimizer/scaler/RNG state; `trained-model.pt` is the compact inference artifact without optimizer state. Press Interrupt Kernel or stop the process, then rerun with the same output directory and settings. The current accumulation window is retried, while completed optimizer updates are preserved. A changed input manifest requires a new output folder; the earlier partial-data run is preserved at `archive/legacy-before-fog/runs/full-training-partial-before-complete-dataset/`. One writer per run is enforced by a file lock.

The active full-study output is `reports/fog/full-training-complete/`. `.cache/` contains the Foggy Cityscapes and DETR caches needed to rerun; `.runtime/`, `.venv/`, and `.ultralytics/` contain the project runtimes/configuration. The `scripts/` directory contains the preparation, notebook-driven CLI, publication and verification tools used by the project, not temporary agent files. `reports/fog/verification/` contains small TEST-ONLY regression charts and checks.

Accuracy uses one COCO bbox evaluator: mAP@[.50:.95], AP50, AR100, precision/recall at confidence .25 and IoU .5, and per-class AP. The same seven categories, preprocessing and image sets are used across arms. End-to-end latency includes decoding, resizing, transfer, density estimation, detector/adapter and postprocessing; downloads and training are excluded. Seed standard deviation is not a confidence interval or a significance test.

The no-density control and estimated-density arm jointly train identical adapters and detectors; the only input difference is the fog scalar. Native nuScenes has no known beta, so oracle-density training/evaluation is rejected unless every matched image has a supplied ground-truth beta. A prediction learned from synthetic Foggy Cityscapes is not a calibrated physical measurement of real fog in nuScenes.

nuScenes boxes are projected 3D boxes and are an imperfect 2D target, not official nuScenes 3D mAP. All visibility levels are retained. Official train scenes are split by scene into training and tuning; official val is held out for final evaluation. Test images never contribute gradients. This preserves all available labeled frames across the experiment without training on the final evaluation set.

## Verification

Run the fast regression checks:

```powershell
.venv/Scripts/python.exe scripts/check_fog_research.py
```

Run the bounded integration check with downloaded pretrained models:

```powershell
.venv/Scripts/python.exe scripts/verify_fog_training.py --size 640 --amp
```

Integration fixtures use a few real images/boxes and arbitrary beta targets solely to exercise the density path. They verify detector weight updates, both arms, metrics, chart exports and resume. They are not fog-performance evidence. Verification figures are explicitly prefixed `TEST-ONLY` under `reports/fog/verification/`.

The 640-pixel CUDA/AMP integration check passed for all six model/arm combinations, including detector weight changes, completed-run resume and configuration mismatch rejection. Fast checks also cover the nested Kaggle layout without gtFine and exclusion of density-only images from detection scoring. These checks do not constitute a full training run or a fog-accuracy result.

## Sources and preserved work

Dataset reuse follows the user-supplied `Untitled2.ipynb` download source. Consult the [Foggy Cityscapes authors](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/), and [official DETR model card](https://huggingface.co/facebook/detr-resnet-50) for source details.

Earlier app, occupancy and depth work remains under [the archive](archive/legacy-before-fog/README.md). Prior fog-pilot notebook versions are also preserved there. The active notebook, runtime dependencies and nuScenes data remain in place.

## Notebook readability and checkpoint safety

See the [laptop handoff guide](docs/laptop-handoff.md) for environment setup, Google Drive checkpoint reuse, and cumulative continuation progress.

Open the current [fog density study Colab notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC) or [local consolidated notebook](notebooks/fog_density_consolidated.ipynb). See the [coding standard](docs/coding-standards.md) and [configuration and validation notes](docs/fog-density-consolidated-notebook.md).

Before expensive training, run `.venv/Scripts/python.exe -m unittest discover -s tests -v` and `.venv/Scripts/python.exe scripts/check_notebook_code.py`. The CPU tests exercise checkpoint recovery and original training reference results without downloading model weights. Rerun with the same settings and persistent output directory to resume; only work saved in completed checkpoints survives runtime loss.
