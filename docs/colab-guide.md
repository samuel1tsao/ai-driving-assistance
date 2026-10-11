# Run the fog study in Google Colab

Upload `notebooks/fog_density_colab.ipynb` using **File → Upload notebook** and select a GPU runtime. The notebook embeds the current profile-aware support modules and installs dependencies; no repository checkout is needed. Choose `TRAINING_PROFILE` in the first settings cell. `dist/fog-colab-code.zip` is an optional small bundle of the notebook, this guide and the data packager.

## Evaluate clear versus fog without training again

Choose the profile that produced your saved checkpoints and set `SETTINGS["action"] = "evaluate"`. `evaluation.source_run` can stay empty to use that profile's saved run, or contain an absolute path to another saved run. Match its model names, seeds and image size. All configured model/arm checkpoints must exist; adding an oracle arm requires a separately trained oracle checkpoint.

Set `evaluation.cityscapes_root` to a folder with these official validation subfolders:

```text
cityscapes-labeled/
  leftImg8bit/val/<city>/*_leftImg8bit.png
  leftImg8bit_foggy/val/<city>/*_leftImg8bit_foggy_beta_<beta>.png
  gtFine/val/<city>/*_gtFine_instanceIds.png
```

Each scene needs clear plus beta .005, .01 and .02 images. Bounding boxes come from the same instance mask for all four views. The existing fog archive is density-only and does not supply these detection labels. Missing files stop evaluation; the notebook does not download or invent annotations. Only validation files are scanned, keeping preparation bounded.

`max_pairs=100` selects a deterministic set of 100 complete scene pairs (400 images); `None` uses every available scene. New predictions, metrics, figures and export ZIP go into `evaluation.output_name`, separate from source training checkpoints. Source and output locks prevent concurrent training/evaluation from mixing results. Changing inputs, source weights or evaluation settings requires a new output name.

Start with the clear/fog bars and density-minus-control deltas. The proposed pattern is a positive fog delta and negative clear delta within a model. Unknown-beta nuScenes results remain a separate normal-use benchmark; they are not a verified clear dataset. This evaluates transfer to Cityscapes for nuScenes-trained checkpoints. One seed is exploratory, and multiple fog versions of a scene are correlated.

Use `analyze` with `SETTINGS["output"]["run_name"]` set to the saved evaluation folder to redraw graphs without images or model weights. Save a notebook copy with outputs after running if you want a visual snapshot. CSVs and PNGs persist independently of cleared cell outputs. The optional synthetic batch-64 benchmark is disabled by default; existing real training settings are preserved.

Model/data API versions are pinned to the local research environment in `requirements-colab.txt`; the runtime's installed CUDA-compatible PyTorch/torchvision pair is retained and checked with GPU NMS before training.

## Configure everything in the first code cell

The first code cell contains storage paths, `ACTION`, `RUN_NAME`, model checkpoints and all study controls. Later cells resolve and display these settings; they do not contain a second set of experiment settings. Defaults retain the three detectors, two arms, three seeds and 30 epochs.

Paths are relative to **MyDrive**, using forward slashes. For example:

```python
DRIVE_DATA_DIR = "Research/Fog study"
DRIVE_OUTPUT_DIR = "Research/Fog results"
DATA_MODE = "zip"
DATA_ZIP_NAME = "fog-colab-data.zip"
```

This reads `MyDrive/Research/Fog study/fog-colab-data.zip` and saves runs under `MyDrive/Research/Fog results/`. Do not include `/content/drive/MyDrive` in either relative setting. Spaces and nested folders are supported. Use `""` to select MyDrive itself.

Choose the data mode explicitly:

- `DATA_MODE="zip"` preserves the original workflow: read the ZIP from Drive and extract one dataset copy locally. The free-space requirements below apply.
- `DATA_MODE="directory"` reads an already-extracted portable dataset directly from Drive without making a local dataset copy. Set `DRIVE_DATA_DIR` to the folder **containing `data/`**, not to `data/` itself. For example, it must contain `data/fog/nuscenes_all.json`, `data/fog/cityscapes.json`, and every image referenced by those manifests. Preserve the portable bundle's complete directory structure, including `.cache/` if present. This mode requires mounted Drive and no URL. It does not extract a ZIP for you.

Code and model caches remain under `RUNTIME_ROOT` in both modes; checkpoints and results use `DRIVE_OUTPUT_DIR`. Directory mode retains the runtime working-space check but does not require the local dataset extraction allowance. It still needs Drive capacity for the extracted data and outputs. The loader retains timestamp normalization for resume and therefore needs write access to referenced images. Moving data roots changes the input paths used by resume validation; use a new `RUN_NAME` when switching modes or data roots for a different experiment.

Run from the top after editing configuration. Start with `ACTION="preflight"` to inspect resolved settings and data checks, then use `"train"` when ready. Hosted Drive directory access and full GPU training have not been validated locally.

## Transfer data once

From the project directory, package the existing prepared nuScenes labels and referenced camera images:

```powershell
.venv/Scripts/python.exe scripts/package_colab_data.py --include-fog
```

Upload `dist/fog-colab-data.zip` to `MyDrive/fog-study/fog-colab-data.zip` in Google Drive. Keep `LOW_DISK_MODE=True`, `USE_DRIVE=True`, and `DATA_ZIP_URL=""` for a runtime with about 100 GB total disk. Low-disk mode requires the complete ZIP under mounted Drive and actual free space for extraction plus a 15 GiB working reserve. It blocks local ZIP downloads and additional Foggy Cityscapes downloads. Pip uses `--no-cache-dir`.

The complete bundle extracts to **49.25 GiB (52.88 GB)**. With the default reserve, at least **64.25 GiB (about 69 GB) must be free after dependency installation**, before extraction. Total capacity includes space already used by Colab. The reserve is an allowance, not a guarantee for extra caches introduced by modified code. Checkpoints and result ZIPs stay in Drive, which needs its own capacity. Code-only changes do not require rebuilding the data ZIP.

Direct HTTPS download is available only after explicitly disabling low-disk mode and requires room for both ZIP and extracted data. The notebook does not publish or share your data.

The supplied data bundle was built with `--include-fog`: 215,319 images referenced by the current prepared manifests (204,894 nuScenes and 10,425 Foggy Cityscapes), totaling about 49.08 GiB of image data. This preserves the prepared selection, rather than copying every image in the raw cache. No separate Foggy Cityscapes download is needed when using this bundle. Upload the ZIP once; Colab extracts it to its own disk. A direct-URL download requires room for both the ZIP and extracted data on that disk.

Use `--dry-run` to estimate size without writing a ZIP. Add `--include-fog` when rebuilding for low-disk mode. Pretrained weights still download when needed. Only with low-disk mode disabled can missing Foggy Cityscapes download from the project's version-1 Kaggle source; that download is estimated at 29.1 GB compressed, plus extraction space, in the existing setup notes.

The ZIP excludes lidar, sweeps, archive copies, environments, old projects and checkpoints. It rewrites image references to relative paths and preserves the full prepared study. All referenced images must exist inside the workspace and nuScenes coverage must be complete. Existing output ZIPs are never overwritten; use `--output` for a new path. Remove an incomplete ZIP after an interrupted packaging attempt before retrying.

nuScenes official downloads require registration/login, so the notebook transfers the data already on your PC. See [nuScenes](https://www.nuscenes.org/nuscenes).

## Train and resume

1. Keep `USE_DRIVE=True` to save checkpoints under `MyDrive/fog-study/runs/colab-full-study`.
2. Run setup, embedded code restoration, dependency installation and data extraction.
3. Review `ACTION` and `CONFIG`. Defaults preserve the full three-model, two-arm, three-seed study with 30 epochs. `ACTION='preflight'` checks readiness without training.
4. Run the remaining training and review cells. For a new experiment, change `RUN_NAME`.
5. To resume, keep the same notebook, runtime paths, dataset, package versions, configuration and run name. Rerun from the top. Colab image timestamps are normalized so re-extracting identical inputs does not break checkpoint validation. Different inputs/settings/dependencies may require a new run name. Windows checkpoints are not portable resume targets because paths and environment differ.
6. The final cell exports a ZIP of results and checkpoints into Drive, or starts a browser download when Drive is disabled. Installed versions are recorded in `colab-packages.txt`.

Colab can delete idle/expired virtual machines; runtime-local data must then be loaded again. The full study may exceed a session or your tier's available GPU memory/disk. Check the displayed free disk and allow space for extracted data, Foggy Cityscapes downloads and checkpoints. Drive also needs room for its ZIP and checkpoints. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Rebuild and validation

After editing the active support code or `scripts/create_fog_notebook.py`, regenerate the notebook and code bundle. The Colab builder uses that intact generator; the existing local notebook has an unfinished `config.update` call and is preserved unchanged.

```powershell
.venv/Scripts/python.exe scripts/create_colab_notebook.py
.venv/Scripts/python.exe scripts/check_colab.py
```

To also validate every image reference in the completed data ZIP and compare sampled image bytes with the originals:

```powershell
.venv/Scripts/python.exe scripts/check_colab.py --data-zip dist/fog-colab-data.zip
```

Local validation covers notebook format, code-cell syntax, embedded source restoration and portable ZIP behavior. Hosted Colab GPU execution, Drive authorization, network downloads, full training and resulting chart presentation have not been verified here. To validate them, upload the notebook and follow steps 1–4 above, first with `ACTION='preflight'`, then with `ACTION='train'`.
