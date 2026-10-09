# Colab notebook comparison

Reviewed October 8, 2026. Baseline: `notebooks/fog_density_colab.ipynb`; edited copy: `notebooks/Untitled0.ipynb`. Cell numbers below are one-based and include Markdown cells. This is a review; neither notebook was modified.

**Subsequent update:** the canonical `fog_density_colab.ipynb` now has centralized configuration and an optional existing-Drive-directory mode. This report records the versions inspected before that update; its cell numbers and identical-storage findings describe those earlier versions. `Untitled0.ipynb` remains preserved. See the [current configuration guide](colab-guide.md#configure-everything-in-the-first-code-cell).

## Verdict

The edited notebook preserves the visible experiment configuration and training/review calls, but it is **not a storage-only change and is not a reliable replacement on a fresh Colab runtime**. Its code-restoration step is missing, and its replacement dependency setup can silently skip installation. The Drive storage logic is identical to the baseline.

Use the baseline notebook for the existing low-disk workflow. It already reads the dataset ZIP from Drive and writes checkpoints/results to Drive. If the intent is to keep the extracted dataset in Drive as well, neither notebook implements that requirement.

## Original goal and what is preserved

The [research plan](fog-density-research-plan.md) defines a matched study of whether estimated fog density changes camera-only detection accuracy and latency. It fine-tunes YOLO11n, YOLO11m and DETR with identical adapters, comparing zero density against estimated density. It preserves scene-held-out splits, complete six-camera nuScenes coverage, separate Foggy Cityscapes density training, resumable state and paired accuracy/latency reporting.

The edited configuration, preparation, preflight, training, review and export cells match the baseline source after trimming trailing whitespace. The visible settings retain 30 epochs, seeds 42/43/44, 640-pixel inputs, accumulation of four and both `rgb_control` and `predicted_density` arms. All three detector defaults are in the baseline's embedded `fog_training.py`, which matches the local support module. However, the edited notebook no longer supplies that module, so its actual research behavior depends on whatever external files happen to exist in the runtime.

## Findings

### High: embedded research code removed; replacement archive cannot restore it

Baseline cells 4–5 restore five support modules and three requirements files. Those cells are absent from the edited notebook. Edited cell 5 instead unpacks `/content/drive/MyDrive/fog-study/fog-colab-code.zip` if the requirements file is missing.

The inspected local `dist/fog-colab-code.zip` contains only:

- `fog_density_colab.ipynb`
- `scripts/package_colab_data.py`
- `colab-guide.md`

Extracting that archive does not execute the embedded restoration cell inside its notebook. A temporary-directory reproduction confirmed that neither `requirements-colab.txt` nor `notebooks/_support/fog_training.py` appears after extraction. On a fresh runtime using this archive, the installer skips its conditional requirements block, and the later project imports cannot succeed without another source of those modules. If the archive has not been uploaded, extraction fails even earlier.

The Drive copy was not inspected, so a manually rebuilt Drive archive could differ. That would still introduce an external code dependency absent from the baseline. The edited introduction incorrectly continues to promise embedded modules and no required code ZIP.

**Correction:** restore the baseline embedded-code cells. The small code payload does not account for the dataset storage problem.

### Medium: dependency installation no longer fails reliably or preserves the same environment

Edited cell 5 replaces a single requirements-file install with per-line installs, catches failures and continues, and skips installation entirely if the requirements file remains absent. Requirements files contain nested `-r` directives, so treating each line as a single package argument is also not equivalent to letting pip process the requirements file.

The new unconditional `pip install --upgrade numpy` changes the environment beyond the requested storage scope and omits `--no-cache-dir`. It is not evidence of a resolved compatibility problem. A CUDA NMS smoke check establishes only that that operation works; it does not validate project imports or the full dependency set. The edited saved output shows extraction and a successful GPU check, but no completed study.

**Correction:** retain the baseline `pip install --no-cache-dir -r ...` call and fail visibly on errors. Diagnose any actual dependency error separately rather than changing packages unconditionally. Do not mix changed code or package versions into an existing run without the pipeline's resume validation.

### Low: dataset extraction duplicated

Edited cells 7 and 8 are identical copies of baseline cell 9. The completion marker normally prevents a second extraction, so this does not itself create two dataset copies. It does add a redundant execution step.

**Correction:** keep one extraction cell.

### Cosmetic changes

The storage heading moved below its setup cell. Notes and export headings lost Markdown heading markers. These changes do not alter the research computation.

## Storage behavior: unchanged in both notebooks

| Item | Location with defaults | Implication |
|---|---|---|
| Dataset ZIP | `/content/drive/MyDrive/fog-study/fog-colab-data.zip` | Read from Drive; not downloaded as a second runtime-local ZIP |
| Extracted dataset and manifests | `/content/fog-study/data/` and any allowed `.cache/` members | Still consumes Colab local disk |
| Project code and installed packages | Runtime-local | Needed again in a fresh runtime |
| Hugging Face cache and YOLO weight paths | Under `/content/fog-study` by default | Model downloads can still consume local disk |
| Checkpoints, metrics and charts | `/content/drive/MyDrive/fog-study/runs/colab-full-study` | Persist in Drive |
| Results archive | `/content/drive/MyDrive/fog-study/runs/colab-full-study-results.zip` | Adds another copy of run artifacts in Drive |
| Package record | Local initially; copied to the run during export | Run the export cell to persist it there |

Both versions retain `USE_DRIVE=True`, `LOW_DISK_MODE=True`, an empty `DATA_ZIP_URL`, and `MIN_FREE_GIB=15`. They block runtime-local dataset downloads in low-disk mode, require both prepared datasets and check free disk before extraction. The local data ZIP's directory entries total **49.248 GiB uncompressed**, requiring **64.248 GiB free before extraction** with the configured reserve. This was calculated from the archive metadata without extracting the full dataset.

The reserve is not a guarantee that every future cache fits. Neither notebook checks the user's actual Drive quota, and the results ZIP requires additional Drive space. The exact Colab free space and Drive capacity were not available in this local review.

For the existing hybrid approach, the baseline already implements the requested Drive input/output paths. If available runtime disk is below the extraction requirement, a separate storage change is necessary: for example, persist extracted data in Drive and update manifest roots/path handling while retaining all study inputs and settings. Simply moving the ZIP to Drive does not move the extracted images there. Such a change has not been made or validated in this review.

## Validation and limits

- Both notebooks passed notebook-format validation and Python AST parsing for all code cells.
- Compared every edited cell against baseline cells; configuration through review and the results-export code are unchanged after trailing-whitespace normalization.
- All five baseline embedded modules and three requirements files match their local source files.
- Ran `.venv/Scripts/python.exe scripts/check_colab.py`: passed embedded restoration, portable fixture extraction, repeat extraction, overwrite protection, path-traversal rejection, low-disk download rejection and insufficient-space rejection.
- Reproduced the edited archive-restoration gap using the actual local code ZIP in a temporary directory.
- Inspected the data ZIP directory metadata to calculate extraction space; did not revalidate every image or train models.

Hosted Colab execution, Drive authorization/quota, dependency resolution on a fresh Colab image, GPU training, resume across sessions and rendered research outputs remain unverified. For end-to-end confirmation, upload the baseline notebook, select a GPU, mount Drive with the complete data ZIP, restart and run all with `ACTION='preflight'`. Resolve any blockers, then run with `ACTION='train'` using a compatible existing run or a new run name. Completed matched runs are needed to establish any accuracy or latency outcome.
