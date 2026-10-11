# Implementation progress

Plan: `docs/training-run-profiles-implementation-plan.md`

Goal: add independent setup, partial-2k, partial-5k, and full training profiles to the current fog-density study notebook while preserving the existing full-run continuation contract.

Assumptions:

- The October 10 `fog_density_study.ipynb` downloaded under `reports/fog/drive-sync/` is the current notebook source.
- Partial profiles train all three detector models with RGB control and predicted density from pretrained weights in separate output directories.
- Partial profiles report on tuning data; only the full profile evaluates the final held-out test set.
- The synthetic CUDA benchmark remains separate from real detector training, which keeps micro-batch size one and gradient accumulation four.

## Work items

| Status | Work item | Files/modules | Acceptance and verification |
|---|---|---|---|
| DONE | Port current notebook support modules and add profile resolution/validation | `notebooks/_support/fog_notebook.py`, `fog_research.py`, `fog_training.py`, `fog_analytics.py`, notebook | Profiles resolve to independent settings, deterministic scene-safe subsets, and separate output paths; invalid profiles fail before training. Verified by profile unit tests. |
| DONE | Add immutable run identity, full-run compatibility, resume/reuse and checkpoint metadata | `fog_training.py` | Same-profile reruns reuse checkpoints; changed settings and cross-profile checkpoints are rejected; existing full run can resume with recorded settings. Existing checkpoint regression suite passed. |
| DONE | Add profile-aware six-way analysis and review output | `fog_analytics.py`, notebook review cells | Results record profile, evaluation split, planned/trained epochs, stop reason, paired deltas, and all configured model/arm combinations. |
| DONE | Regenerate checked-in notebooks from the current Drive notebook and updated modules | `notebooks/fog_density_consolidated.ipynb`, `notebooks/fog_density_colab.ipynb`, `scripts/create_colab_notebook.py` | Notebook code is self-contained and preserves the guarded synthetic batch-64 benchmark. Colab and notebook checks passed. |
| DONE | Positive, negative, performance, and race/idempotency checks | `tests/`, `scripts/` | 15 unit tests passed; notebook, Colab, focused research, ZIP and directory checks passed. The bounded checker intentionally uses one YOLO11n model; the `setup` profile is the all-model smoke path. |

## Open decisions and risks

- Existing full checkpoints may have been created before profile metadata existed. The full profile will compare all recorded training/data settings and allow only additive metadata migration.
- Real detector training is intentionally not changed to batch 64; the notebook's GPU benchmark measures a separate synthetic, GPU-resident DataLoader path.

## Validation results

- `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v` — 15 passed.
- `.venv/Scripts/python.exe scripts/check_notebook_code.py` — passed.
- `.venv/Scripts/python.exe scripts/check_colab.py` — passed.
- `.venv/Scripts/python.exe scripts/check_fog_research.py` — passed.
- `.venv/Scripts/python.exe scripts/check_consolidated_notebook.py --images 2` — directory mode passed.
- `.venv/Scripts/python.exe scripts/check_consolidated_notebook.py --zip --images 1` — ZIP mode passed.

The bounded notebook checker uses its legacy `SETTINGS["test"]` override and narrows to YOLO11n to keep validation time reasonable. It is separate from the research `setup` profile, which is configured for all three models.

## Current implementation: two-dataset imbalance handling

Plan/source: `docs/two-dataset-fog-validation-implementation-plan.md`.

Goal: apply the dataset-imbalance policy to the active consolidated notebook without modifying saved partial-run notebooks. Keep nuScenes and Foggy Cityscapes as separate analysis populations, stratify density supervision by beta bin, and make any Foggy detection benchmark explicitly opt-in because it requires `gtFine` labels.

| Status | Work item | Files/modules | Acceptance and verification |
|---|---|---|---|
| DONE | Add shared beta-bin labels and balanced sampling utilities | `notebooks/_support/fog_research.py`, `fog_training.py` | Density training samples known beta strata deterministically when `density_sampling="balanced_bins"`; invalid beta remains rejected. |
| DONE | Add separate natural and balanced density-bin analysis outputs | `fog_research.py`, `fog_analytics.py`, notebook | Results retain natural rows and add equal-bin macro rows without pooling datasets. |
| DONE | Expose balancing settings in the consolidated notebook and mirror Colab artifact | `notebooks/fog_density_consolidated.ipynb`, `fog_density_colab.ipynb` | Partial/setup profiles use balanced density sampling; full keeps natural sampling for existing-checkpoint continuity. Saved 2k/5k notebooks were not edited. |
| DONE | Positive, negative, performance, and idempotency checks | `tests/`, notebook checks | 18 unit tests, notebook syntax/lint/format, Colab artifact checks, and focused research checks pass. |

Open decision: a Foggy Cityscapes detector track needs a rebuilt manifest with `with_detection_labels=True` and `gtFine` included. The current implementation must not silently treat density-only Foggy rows as detection negatives.

Validation note: the bounded end-to-end notebook checker reached the training cell but hung after execution with no Python process remaining visible; it was interrupted. This does not replace the passing unit, checkpoint, syntax, lint, format, Colab, and focused research checks.
