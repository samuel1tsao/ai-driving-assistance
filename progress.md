# Implementation progress

### One active fog study notebook

User request: keep a single file named `fog_density_study.ipynb` for local work and Colab, remove the older notebook copies, and explain the fog-predictor partition call.

| Status | Work | Verification |
|---|---|---|
| DONE | Consolidate the local notebook source and documentation around `notebooks/fog_density_study.ipynb`; remove the old active notebook names and generators | Notebook structure, syntax, Ruff, and readability checks pass. |
| DONE | Remove older notebook copies from the Drive study folder | Folder inspection confirms the canonical notebook remains alongside the guide, data, code bundle, runtime, and run folders; prior version copies are deleted. |
| DONE | Keep the missing-checkpoint predictor fallback, but train it from the full density split in every profile | Profile tests pass with an intentionally small density cap; only detector train/tuning rows are capped. The predictor is frozen after loading or fitting. |
| DONE | Package and sync the final notebook/docs to Drive, remove old Drive copies, and prepare intended local changes for Git publication | Drive folder was rechecked. All 44 unit tests and static checks pass; the local bounded end-to-end run stalled inside its training cell and is reported as inconclusive. |

The partition call exists only for the missing-checkpoint fallback: it provides beta-labeled training and validation rows to fit the predictor. Existing checkpoints load frozen without fitting. Detector profiles now share the full density train/validation data even when their detector rows are capped. The detector run no longer writes held-out beta-prediction evaluation CSVs.

### Single-notebook runtime and readable workflow correction

Goal: make the Colab notebook genuinely self-contained and understandable. Keep detector and checkpoint behavior; define implementation directly in organized, collapsed notebook sections; expose clear high-level predictor, data, validation, training, and analytics calls in the visible workflow.

The user's single-notebook requirement explicitly overrides `docs/coding-standards.md`'s generated `%%writefile` helper-module pattern for the Colab artifact. Local `notebooks/_support/*.py` modules remain regression-test sources; generated notebook implementation cells will contain their Python directly, with intra-module imports resolved through notebook execution order. No helper file will be written or imported at Colab runtime.

| Status | Work item | Acceptance / verification |
|---|---|---|
| DONE | Embed the implementation directly in the notebook | No `%%writefile` or runtime support-module imports. Six purpose-labeled helper cells execute in notebook order and remain beneath a collapsed section. |
| DONE | Expose the high-level notebook workflow | Visible steps are predictor setup, runtime setup, data preparation, validation, training, analytics, and export. |
| DONE | Update notebook checks and Colab packaging | Self-contained notebook, Colab mirror, direct-cell AST/Ruff checks pass. |
| DONE | Run regression and bounded end-to-end checks | 34 unit tests pass; directory and ZIP subset workflows both train, report, reuse checkpoints, and export successfully. No full GPU study launched. |
| DONE | Preserve Drive outputs and publish the notebook | Backed up the prior Drive notebook, replaced the canonical notebook in place, verified the downloaded notebook has six helper cells and no helper-file writes/imports, then pushed the source to `origin/main`. |

No detector methodology, profile settings, trained weights, or checkpoint formats are in scope. Existing user edits in `notebooks/fog_density_detection_research.ipynb` and `reports/fog/` remain untouched.

Plan: `docs/training-run-profiles-implementation-plan.md`

Implementation checks: `scripts/check_notebook_readability.py` reports exactly six embedded helper cells. The notebook runtime executes those functions directly from the notebook; checked-in support modules are only the local source and test fixtures. Helper code is not exported as extra Colab files.

Drive backup: `fog_density_study_before_self_contained_helpers_20261012.ipynb` (ID `15G0ZpezrGhB3eXMxjXbj3PmKv6ODimZF`). Canonical notebook `fog_density_study.ipynb` (ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`) was updated 2026-10-12 05:35 UTC and verified by raw readback: 46 cells, six helper cells, no `%%writefile`, and no runtime support imports.

Published source commit: `c024eda` (`Make fog study notebook self-contained and readable`), pushed to `origin/main`. Validation: 34 unit tests, notebook/Colab AST+Ruff checks, and directory and ZIP bounded end-to-end notebook checks all passed.

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

## Simplify dataset preparation across profiles

User direction: prepare the same full dataset for every profile, apply profile caps only when making training partitions, assume the archive URL is a configured HTTPS constant, remove unnecessary validation/fallback branches, and preserve Colab A100 input-throughput settings.

| Status | Work item | Acceptance / verification |
|---|---|---|
| DONE | Make data preparation profile-independent and remove reduced-copy staging helpers | All profiles prepare the same configured archive/manifests. Profile caps are applied only inside training partition creation. |
| DONE | Simplify the archive URL and storage configuration | One HTTPS URL at the settings level; no runtime scheme check or URL-specific cache branches. |
| DONE | Preserve detector throughput settings and checkpoint compatibility | Existing full-run identity and A100 benchmark/loader settings stay unchanged; partial profile versions advance for new selected-row behavior. |
| DONE | Verify selection, preparation, resume, and Colab notebook | 44 unit tests pass; notebook syntax/lint/format/readability and exact Colab mirror pass; focused fog checks pass; bounded directory workflow completes training, validation, reports, and checkpoint export. ZIP extraction is covered by selection/preparation tests; a full 49 GiB ZIP smoke was not repeated. |

## Notebook readability enforcement

Goal: keep the single authored Colab notebook understandable from its visible workflow while preserving its self-contained runtime implementation and checkpoint behavior.

| Status | Work item | Verification |
|---|---|---|
| DONE | Refactor the configuration and training orchestration into named, flat steps | Configuration now applies saved/selected profiles, export routing, retrain allocation and evaluation routing as separate steps. Training now performs preflight, setup cleanup, comparison scheduling and export through separate helpers. |
| DONE | Reduce visible execution cells to high-level calls | The optional synthetic batch-64 benchmark moved behind one descriptive workflow call; it remains separate from detector training. |
| DONE | Add mechanical notebook readability enforcement | `scripts/check_notebook_readability.py` requires collapsed runtime cells, one cohesive definition per helper cell, exact notebook/support mirroring, five-or-fewer visible workflow statements, and 40-line flat orchestration functions. `scripts/sync_notebook_support.py` regenerates the notebook helper cells. |
| DONE | Preserve checkpoint/reuse behavior | The full 34-test suite passed after the refactor, including interruption/recovery, completed-run reuse, profile continuation, output locking and matched checkpoint evaluation. Notebook syntax/lint/format and the readability guard passed. |

The current Colab notebook was updated in place and verified by Drive metadata: ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`, 366,432 bytes, updated 2026-10-12 01:29:15 UTC. The implementation is published as `a9f3de4` on `origin/main`. The local bounded end-to-end check reached its setup training stage but was stopped by the 30-second command limit, so it is not counted as a pass. The `fog_density_detection_research.ipynb` and existing report/Drive-sync working-tree changes remain unrelated and untouched.

### Frozen beta predictor for detector comparisons

The standalone density action was the wrong workflow and has been removed. The notebook's main action remains `train`: partial profiles load the already completed full-run estimator at `colab-full-study/density-42/best.pt`, freeze it, and call it as the `predicted_density` arm's image-to-beta predictor. The paired RGB control receives zero beta. New partial profiles skip Foggy Cityscapes staging and density-model training; missing or incompatible predictor weights block preflight. Profile versions were advanced so these detector results do not mix with earlier runs. Existing full-profile settings are preserved for continuation.

Implementation files: `notebooks/_support/fog_notebook.py`, `notebooks/_support/fog_training.py`, `notebooks/_support/fog_research.py`, `notebooks/fog_density_consolidated.ipynb`, and generated `notebooks/fog_density_colab.ipynb`. Related helper definitions remain grouped in collapsed cells; the visible notebook section now describes only the frozen predictor and detector comparison. The earlier `bd543bd` commit introduced the incorrect density action; this correction removes the action and its separate runner. Drive notebook `fog_density_study.ipynb` (ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`) now has the corrected 92-cell notebook, updated 2026-10-12 03:24:21 UTC; readback confirms the detector section and frozen predictor with no density-only action. Notebook readability, schema, AST, and diff checks passed; no training tests were run.

### Make the predictor setup easy to find

The notebook now has a small, separately labeled **Fog-density predictor setup (pretrained, frozen)** cell near the top. It selects the existing `colab-full-study`/seed-42 checkpoint; detector training loads it automatically. The collapsed runtime section explains that `%%writefile -a` appends helper-code chunks to the named support file, and each chunk starts with its target file and function/class. Before replacing the Drive notebook, preserved its current copy as `fog_density_study_before_predictor_setup_20261012.ipynb` (ID `1mYWE8-wQkmIXdWyxSXysiUcoOT7R3eph`). Updated Drive notebook `fog_density_study.ipynb` in place (ID `1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC`, 94 cells, 349,585 bytes, updated 2026-10-12 03:31:16 UTC); readback confirms the setup cell and `%%writefile -a` explanation.

### Simplify training and standalone fog comparison

Goal: simplify the single notebook's run lifecycle. Always stage/validate data; automatically restore or train/save the fog-density predictor; resume the selected detector profile or reuse completed checkpoints; and expose a standalone matched clear/fog analysis cell that can load its saved models when run independently.

| Status | Work item | Files/modules | Acceptance / verification |
|---|---|---|---|
| DONE | Replace action/arm skip routing in the visible workflow with simple data, predictor, and detector steps | `notebooks/fog_density_consolidated.ipynb` | Visible order is load/validate data, load or train/save the beta predictor, then resume or train all configured detector/model arms. |
| DONE | Make matched fog/clear analysis independently runnable and reuse in-memory models or restore saved ones | Notebook evaluation and analysis cells | Dedicated analysis call resolves saved checkpoints when needed and displays tables and figures without requiring the training cell in the current kernel. |
| DONE | Positive and negative lifecycle checks | `tests/`, notebook checker | Unit coverage includes missing/saved density predictors, full and incomplete detector profiles, completed checkpoint reuse, and saved-model comparison loading. |
| DONE | Performance/load and idempotency/concurrency checks | `tests/`, notebook smoke checks | The 41-test suite and notebook/Colab/research checks pass. Directory-mode bounded smoke completed YOLO11n training, validation, reports, and export; ZIP mode was not run because it would extract the full 52.9 GB archive. Its local fixture skips matched Cityscapes because it contains no labeled Foggy Cityscapes data; standalone matched analysis is covered by its unit test. |

Repository notebook guidance takes precedence over the implementation skill's 250-character function limit. The user's explicit readability request favors a short visible flow and focused helpers; the established notebook standard targets 40 logical lines and rejects extra abstraction layers. Verification passed: `.venv/Scripts/python.exe -m unittest discover -s tests -v` (41 tests), `scripts/check_notebook_code.py`, `scripts/check_colab.py`, `scripts/check_fog_research.py`, and bounded ZIP run `scripts/check_consolidated_notebook.py --zip --images 2`. The local checker skips only the matched Cityscapes analysis cell because this bounded fixture contains nuScenes but no labeled Foggy Cityscapes; fixture-backed standalone analysis passes `test_standalone_notebook_cell_loads_saved_models_and_runs_analysis`. The temp directory must be pointed inside `.runtime` in this sandbox.
## Explicit experiment flow

Plan/source: `docs/fog-notebook-visible-workflow-implementation-plan.md`; research contract: `docs/fog-density-research-plan.md`.

Goal: make the visible notebook name nuScenes, Foggy Cityscapes density-training data, each detector model/arm, and each validation target. Preserve one notebook, existing run identities/checkpoints, profile selection, matched clear/fog analysis, and A100 optimizations. User direction: no extra support files, no six trivial wrappers, and no helper validation sprawl; prefer a clear visible sequence with shared training mechanics underneath.

| Status | Work item | Files | Acceptance / verification |
|---|---|---|---|
| DONE | Inspect current dataset/run identity contracts; expose separately named nuScenes and Foggy Cityscapes inputs | `notebooks/fog_density_study.ipynb` | Predictor rows load separately; existing fingerprint row selection and stable timestamps are preserved. |
| DONE | Expose all six detector/arm training calls and named per-model validation results | notebook | Each call selects one model and arm through shared locked checkpoint mechanics; no duplicated wrappers. |
| DONE | Make matched clear/fog analysis explicitly load its labeled dataset independently | notebook | A dedicated loader returns matched `gtFine` rows; the analysis cell passes them to saved-model evaluation. |
| DONE | Positive and negative behavior tests | `tests/`, `scripts/check_notebook_readability.py` | Exact visible ordering/models/arms; invalid or missing labeled evaluation data errors at the corresponding step. |
| DONE | Performance and output consistency | `scripts/check_consolidated_notebook.py`, tests | Bounded run finishes; profile budgets, fingerprints, output columns and checkpoint reuse remain unchanged. |
| DONE | Concurrency and recovery | tests | Repeated per-comparison calls preserve run locking, interrupted resume, completed reuse and separate analysis idempotency. |
| DONE | Standard review and publish | README, plan, notebook, Drive | Run full suite/checks, inspect final diff, push and sync the existing Drive notebook/bundle. |

Coding-standard resolution: retain the repository's focused-helper/40-line guidance. The implementation skill reference's 250-character total-function rule would force the wrapper proliferation the user has explicitly rejected; keep cohesive short functions and document any justified existing-function exceptions here rather than splitting for character count alone.


### Explicit experiment flow implementation (2026-10-14)

| Status | Work item | Verification |
|---|---|---|
| DONE | Load nuScenes detector rows and Foggy Cityscapes beta-training rows separately; retain run fingerprints and shared preparation | Notebook readability guard and full suite |
| DONE | Show six explicit YOLO11n, YOLO11m, and DETR calls for RGB-only and predicted-density arms, each with named results | Exact call-order check; bounded directory-mode smoke trained and validated both configured YOLO11n arms |
| DONE | Load labeled matched clear/fog Cityscapes data in its own analysis cell | Matched fog evaluation tests |
| DONE | Verify positive/negative behavior, profile/output consistency, lock contention, and recovery | 47 unit tests; notebook, Colab, and research checks |
| PENDING | Push to GitHub | The implementation commit is local; automatic review requires user approval for the configured origin |

The ZIP-mode smoke was not run because its fixture requires extracting the full 52.9 GB dataset. Directory-mode bounded smoke exercised training, validation, reporting, and export. The notebook and code bundle were synced in place to Drive and verified in the target folder. The implementation commit is local; the external Git push is pending approval after the automatic review rejection.

### Add labeled fog examples to detector training (2026-10-15)

The prior visible workflow trained the density predictor on Foggy Cityscapes but trained all six detectors on nuScenes only. Detector input now combines nuScenes with matched clear/fog Cityscapes images and `gtFine` boxes. Each detector arm sees the same images and labels; the density arm additionally receives the frozen beta estimate. Cityscapes groups reuse the beta predictor's split so predictor training groups remain in detector training, with separate held-out validation and test groups. Partial image caps are divided evenly across nuScenes and Cityscapes.

| Status | Work item | Verification |
|---|---|---|
| DONE | Reuse the existing labeled clear/fog data loader for detector training and held-out analysis | Shared-manifest fixture and standalone saved-model analysis pass |
| DONE | Preserve held-out scene splits and require labels for detector training | Fixture test checks boxes, matched pairs, and train/validation/test group mapping |
| DONE | Ensure partial caps include both input datasets | Profile test confirms equal dataset quotas and fog rows in detector training |
| PENDING | Confirm the configured Cityscapes root contains `gtFine/` and `leftImg8bit/` | The notebook already uses this root for matched evaluation; the Foggy Cityscapes beta bundle itself omits these files |
| DONE | Rerun checks and publish the revised notebook/bundle/guide to Drive | Full suite and bounded smoke pass; all three artifacts updated and verified in the target folder |
| DONE | Commit and push the follow-up | Commit `48f5b26` pushed to `origin/main` after user review and authorization |


The labeled Cityscapes loader now has one shared entry point. Training passes in the already loaded beta manifest and receives all labeled rows; the independently runnable analysis cell loads the same manifest and filters to the configured held-out split. This preserves scene-level separation while eliminating the second, separately generated matched manifest.

Verification note: the full 49-test suite, notebook format/schema/readability/Colab checks, fog research checks, and bounded directory-mode notebook smoke pass. The smoke trains and validates configured YOLO11n arms on a synthetic labeled Cityscapes fixture. Drive search confirms the configured `fog-study` folder does not yet contain the `cityscapes-labeled` root; Colab training therefore requires that external Cityscapes data to be placed there.


Drive publication for the labeled-fog training update completed 2026-10-15: `fog_density_study.ipynb`, `fog-colab-code.zip`, and `colab-guide.md` were updated in place and verified by matching Drive sizes/timestamps. GitHub push completed after user review and authorization. The configured Drive folder still lacks `cityscapes-labeled`; training cannot use labeled fog examples until the official `gtFine/` and clear Cityscapes files are added there.


### Fix partial-profile manifest root (2026-10-15)

Colab reports that `nuscenes_all.json` is missing under `/content/fog-study/profile-partial-2k/`. Profile setup incorrectly changes the prepared data root to a profile-specific directory after the archive extracts under `/content/fog-study/`. Keep the profile output/checkpoint directories independent, but make all profiles reuse the one extracted data root.

| Status | Work item | Verification |
|---|---|---|
| DONE | Add a regression test proving partial profiles resolve and verify manifests from the shared data root | `test_partial_profile_uses_shared_extracted_data_root` passes |
| DONE | Remove the partial-profile data-root override; rerun unit/notebook/Colab checks; update Drive and GitHub | 50 tests and notebook, format, readability, and Colab checks pass; Drive notebook and bundle updated |
