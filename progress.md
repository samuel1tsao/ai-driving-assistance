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

## Current implementation: matched clear and fog hypothesis

Source: `docs/two-dataset-fog-validation-implementation-plan.md`. Preserve the existing training profiles and full continuation; implement evaluation from saved checkpoints in the single authored notebook. Repository readability conventions take precedence over the skill's 250-character function limit because the user explicitly requests fewer abstractions.

| Status | Work item | Acceptance / verification |
|---|---|---|
| DONE | Labeled matched clear/fog contract and evaluation-only action | Complete pairs, shared gtFine boxes, split isolation, compatible size/arms/seeds and immutable source fingerprints; source checkpoints remain unchanged. |
| DONE | Beginner analytics and notebook reading order | Condition metrics, arm effects, seed uncertainty, PR curves, density/class/latency graphs; ordinary Python split into collapsed cohesive cells. Charts inspected using explicitly synthetic fixtures. |
| DONE | Positive and negative tests | 27 unit tests pass, including full notebook evaluation reading order, missing pairs/labels, leakage, changed settings, missing arms and legacy checkpoint parity. |
| DONE | Performance and idempotency tests | All six comparisons on bounded CPU fixtures; interrupted predictions recover, completed evaluation skips inference, locks block a second writer; directory and ZIP real-data smoke runs pass. |
| DONE | Publish Git and existing Colab file | Drive updated and metadata verified; implementation commit `a2ddf34` pushed to `origin/main`. |

Dependency: labeled Cityscapes `gtFine`, original clear images and fog variants are required for the new benchmark. No scientific result will be fabricated when these are unavailable.

Verification: `unittest discover -s tests -v` (27 passed), `check_notebook_code.py`, `check_colab.py`, `check_fog_research.py`, and both `check_consolidated_notebook.py --images 2` / `--zip --images 2` passed. The focused checker needed TEMP/TMP pointed inside `.runtime` because the Windows sandbox's default temporary directory was not writable.

The bounded checker exposed and verified fixes for setup deleting its newly written subset audit and resetting training during its internal reuse check. Cleanup now holds the output lock, verifies resolved child paths, and preserves the current audit; normal setup reruns still overwrite.

Drive readback: `fog_density_study.ipynb`, ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`, 349,483 bytes, updated 2026-10-11 17:35:33 UTC. Preserved the latest 5k v2 setting. Existing notebook outputs were backed up in the same folder as `fog_density_study_before_matched_validation_20261011.ipynb`, ID `12pdIfAEi3bHV07s1VGluzG2itcmU6-s2`.

Limits: the local manifest has 10,425 density-only Foggy images and no detection labels. CPU fixtures are not scientific results; CUDA/Colab execution and the actual labeled fog benchmark require the user's runtime and gtFine/clear assets. No full training or dataset upload was launched. Unrelated working-tree changes remain untouched.

Publication: `git push origin main` succeeded against `https://github.com/samuel1tsao/ai-driving-assistance.git`. The final progress record is committed separately after successful publication. All planned implementation work is complete; the next research step is supplying labeled matched assets and running `evaluate` in Colab.

## Continuation and explicit retraining follow-up

User contract: an existing profile resumes unfinished training or reuses completed comparisons; only an explicit retrain flag starts fresh, except disposable setup. Analyze only reads saved results.

| Status | Work item | Verification |
|---|---|---|
| DONE | Diagnose actual Drive error and make dispatch/reuse explicit | Saved notebook and traceback used train. All six 2k-v1 comparisons are complete. Old config had natural density sampling; new defaults added balanced sampling and profile metadata. |
| DONE | Explicit retrain flag with preserved prior runs; actionable setting differences | Existing runs use recorded config/caps before preparing data. Finished runs redraw saved reports without training or weights. Retrain allocates a fresh sibling; setup preserves its disposable contract. Unknown actions never train; mismatch messages list changed fields. |
| DONE | Positive/negative/performance/race tests | 34 tests, notebook lint/format, Colab checks and directory/ZIP smoke runs passed. Legacy full continuation identity is retained. Existing checkpoint/input/package protections and locks remain; fresh folders are allocated exclusively. |
| DONE | Publish continuation fix | Main Drive notebook updated and verified (359,669 bytes, 2026-10-12 01:12:49 UTC); commit `40125c5` pushed to origin/main. |

The operational retrain flag is not added to checkpoint fingerprints. On retraining, the previous output and archive remain intact; default exports follow the selected fresh folder. Continuing that fresh run later requires selecting its printed output name and setting retrain=False. A Windows temporary-file lock interrupted one existing density checkpoint test; its isolated 11-test suite and the full 33-test suite both passed on retry without changing checkpoint-save semantics.

Saved Drive evidence: `partial-2k-v1/settings.json` (ID `1fhubuYtYGNzREynxIO_5Ver6j00cAR98`) omitted density_sampling, which means natural sampling; all six rows of `status.csv` were complete, including early stops. The latest main notebook had action=train, not analyze, and its traceback entered execute_training. Existing 2k/5k/full run files were read only. The main notebook's latest partial-2k selection and 5k v2 setting were preserved. Its outputs were backed up as `fog_density_study_before_continuation_fix_20261011.ipynb` (ID `1oms11pbIVz8hB95AwEEd3hlS0uPbbGhb`) before replacing the main notebook.

## Notebook readability enforcement

Goal: keep the single authored Colab notebook understandable from its visible workflow while preserving its self-contained runtime implementation and checkpoint behavior.

| Status | Work item | Verification |
|---|---|---|
| DONE | Refactor the configuration and training orchestration into named, flat steps | Configuration now applies saved/selected profiles, export routing, retrain allocation and evaluation routing as separate steps. Training now performs preflight, setup cleanup, comparison scheduling and export through separate helpers. |
| DONE | Reduce visible execution cells to high-level calls | The optional synthetic batch-64 benchmark moved behind one descriptive workflow call; it remains separate from detector training. |
| DONE | Add mechanical notebook readability enforcement | `scripts/check_notebook_readability.py` requires collapsed runtime cells, one cohesive definition per helper cell, exact notebook/support mirroring, five-or-fewer visible workflow statements, and 40-line flat orchestration functions. `scripts/sync_notebook_support.py` regenerates the notebook helper cells. |
| DONE | Preserve checkpoint/reuse behavior | The full 34-test suite passed after the refactor, including interruption/recovery, completed-run reuse, profile continuation, output locking and matched checkpoint evaluation. Notebook syntax/lint/format and the readability guard passed. |

The current Colab notebook was updated in place and verified by Drive metadata: ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`, 366,432 bytes, updated 2026-10-12 01:29:15 UTC. The implementation is published as `a9f3de4` on `origin/main`. The local bounded end-to-end check reached its setup training stage but was stopped by the 30-second command limit, so it is not counted as a pass. The `fog_density_detection_research.ipynb` and existing report/Drive-sync working-tree changes remain unrelated and untouched.

### Standalone density-estimator action

The notebook now has `action="density"` to train or load only the beta estimator, write beta predictions and held-out MAE/RMSE, and show a readable known-versus-estimated beta plot. It prepares only the needed Foggy Cityscapes images and never starts detector training. A saved checkpoint can be passed into the separate detector study with `density_model.mode="load"`. Partial profiles use their selected evaluation rows for predictions; choosing `prediction_split="all"` requests the complete manifest. The notebook guide and README describe this split between estimating fog density and evaluating detection.

Implementation files: `notebooks/_support/fog_notebook.py`, `notebooks/_support/fog_training.py`, `notebooks/fog_density_consolidated.ipynb`, and generated `notebooks/fog_density_colab.ipynb`. Related readability limits now group related helper definitions in collapsed cells, with an upper bound on cell size and total helper-cell count. No tests were run for this change.

Published commit `bd543bd` to `origin/main`. Updated Drive notebook `fog_density_study.ipynb` in place (ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`; 367,266 bytes; 2026-10-12 03:05:18 UTC). Before the update, preserved the prior Drive notebook as `fog_density_study_before_density_only_20261011.ipynb` (ID `1gTbYuXIDWVI4PD8qDis1qhYjhrRdo5Um`). Drive readback confirms the density action, prediction output name, and runner are present.
