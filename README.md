# Fog-density object detection research

Open the [fog density study notebook](notebooks/fog_density_study.ipynb). It runs independent `setup`, `partial-2k`, `partial-5k`, and `full` profiles across **YOLO11n, YOLO11m and DETR**, then exports accuracy, latency, per-class AP, paired differences and charts. See the [research plan](docs/fog-density-research-plan.md)..

The main notebook loads nuScenes images and Foggy Cityscapes beta labels, then joins Foggy Cityscapes images to `gtFine` boxes and matched clear images. All six detector comparisons train on both nuScenes and labeled Cityscapes examples, so the predicted-density arm sees foggy detection cases while the RGB control sees the same images without the density input. Results stay separated by dataset and clear/fog condition. The configured `cityscapes-labeled` folder must contain `gtFine/` and matched `leftImg8bit/` images; the data ZIP includes Foggy Cityscapes images and beta labels but not those Cityscapes annotations. Beta is the numeric fog-density label; larger beta means denser synthetic fog. [The single notebook](notebooks/fog_density_study.ipynb) is used locally and in Colab. Test and CLI tools execute its definitions directly; no support modules or notebook copies are generated. Shared constants and profile settings are centralized near the top. See the [notebook guide](docs/fog-density-consolidated-notebook.md).

## Run on Google Colab

See the [proposed setup, partial and full training profiles](docs/training-run-profiles.md) for bounded preliminary results and early stopping.

For profile configuration, checkpoint continuation and validation notes, see [the consolidated notebook guide](docs/fog-density-consolidated-notebook.md). Partial profiles use their full validation split for preliminary analysis; the full profile continues the existing full run and evaluates the final held-out set.

Shared names/defaults and `PROFILE_SETTINGS` live in the first code cell. Choose the run in **Run settings** and set `SETTINGS["data"]["location"]` to the Drive folder or local folder containing the configured dataset ZIP. Every profile prepares the same full dataset; profile caps apply only to detector training rows; density-training, validation, and test rows remain unchanged. The frozen fog-density predictor uses the full density-training split whenever its saved checkpoint is missing.

Upload [the study notebook](notebooks/fog_density_study.ipynb) and select a GPU runtime. Keep the complete dataset ZIP in Drive; the notebook extracts the same data for every profile and requires about 64.25 GiB free (49.25 GiB data plus 15 GiB working space). It embeds the code, installs dependencies without a pip download cache, and downloads model weights when needed. The optional code bundle is `dist/fog-colab-code.zip`.

## What was fixed


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

**Foggy Cityscapes images alone are sufficient to train the density estimator.** Beta is parsed from filenames; nuScenes supplies detection annotations. `gtFine` and matched clear images are required for the labeled Cityscapes detector examples. The notebook loads these once into a shared manifest: all six detector comparisons train on the labeled training rows, and the separate analysis cell evaluates saved models on the held-out clear/fog rows. Missing annotations stop that data-loading step rather than becoming negative examples.

## Run in Jupyter

From this project directory in PowerShell:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-fog-research.txt
.venv/Scripts/python.exe -m jupyter lab notebooks/fog_density_study.ipynb
```

Use the project Python kernel, then review the first settings cell:

1. Set `RUN_ENVIRONMENT="local"` and point the **Run settings** data and output locations at your prepared data and chosen run root.
2. Select the training profile and verify the saved density-predictor checkpoint path.
3. Use `SETTINGS["action"]="train"` and **Restart Kernel and Run All** to resume/train, or `"preflight"` to check readiness.
4. Use `"analyze"` for saved reports or `"evaluate"` for the labeled matched clear/fog benchmark.

The full profile uses 30 detector epochs, 15 density epochs, seeds 42/43/44, 640-pixel square inputs, single-image microsteps with four accumulated gradients, and mixed-precision training on CUDA. Partial profiles use seed 42, five epochs, scene-safe caps on detector training rows only; they restore the shared fog model or fit and save it when its checkpoint is missing. The full profile keeps natural density sampling so an existing full checkpoint can continue without changing its training distribution. The optional A100 throughput benchmark uses batch 64; real detector training remains batch-one with accumulation four.

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


Open the current [fog density study Colab notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC) or [local consolidated notebook](notebooks/fog_density_study.ipynb). See the [agent/coding standards](AGENTS.md) and [configuration and validation notes](docs/fog-density-consolidated-notebook.md).

Before expensive training, run `.venv/Scripts/python.exe -m unittest discover -s tests -v` and `.venv/Scripts/python.exe scripts/check_notebook_code.py`. The CPU tests exercise checkpoint recovery and original training reference results without downloading model weights. Rerun with the same settings and persistent output directory to resume; only work saved in completed checkpoints survives runtime loss.
