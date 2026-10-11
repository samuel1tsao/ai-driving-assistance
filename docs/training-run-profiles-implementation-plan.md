# Independent training profile implementation plan

## Intended workflow

Run separate experiments at increasing data scopes. Each profile starts from the original pretrained detector weights, trains independently, and produces its own six-way analysis.

| Profile | Training data | Starting point | Output |
|---|---:|---|---|
| setup | Tiny smoke subset | Pretrained weights | Disposable smoke run |
| partial-2k | 2,000 images | Pretrained weights | `partial-2k` checkpoints/reports |
| partial-5k | 5,000 images | Pretrained weights | `partial-5k` checkpoints/reports |
| full | All training images | Existing saved full run | Continue its checkpoints/reports |

Every research profile runs all three models with and without density. Use seed 42 for quick comparisons; the existing full run keeps its recorded seeds. Shared tuning and nested training subsets aid comparison, but new profiles never inherit weights or history.

## Configuration contract

Extend `SETTINGS` with profile name/output, training and tuning limits, subset seed, detector/density epochs, batch/effective-batch settings, loader settings, budget and stopping policy. Settings are immutable after checkpoints exist. Re-running resumes that profile's `last.pt`; changed inputs/settings require a new directory. The existing full run is a compatibility exception: locate its output directory and resume only with its recorded settings. Reject invalid selections before training.

Persist config, selected IDs and hashes in `settings.json` and `input-manifest.json`, including batch/loader settings. Keep checkpoints and state inside each profile directory. Prohibit cross-profile warm starts except explicit continuation of the existing full run. Use a lock and atomic saves; reuse completed runs without rewriting checkpoints.

## Training and analysis flow

Run one density estimator per profile when needed. Schedule all six combinations and show counts, caps, batch/loader settings, ETA and paths. Fit a partial budget by reducing data or epochs, never by dropping a model/arm. Preserve the existing full run's input pipeline while resuming it; report actual epochs and stop reason.

After each run, export AP, latency, density quality, curves and paired deltas. Profile reports compare all six combinations. Cross-profile charts must label profiles as independently trained.

## Notebook changes

- `fog_notebook.py`: resolve profiles, validate limits and write profile paths.
- `fog_research.py`: select deterministic scene-safe subsets and fixed tuning IDs.
- `fog_training.py`: retain within-profile resume/reuse and recovery; add no parent initialization. Add explicit compatibility checks for the existing full run's batch/loader contract.
- `fog_analytics.py`/review cells: include profile, epochs, stop reason and paired results.

No new remote API or database is needed. Keep Drive mount, quota, timeout and dependency failures visible. Do not evaluate final-test data as a partial-profile result unless explicitly requested; use the same held-out final set for the full evaluation. The latest notebook includes a separate synthetic `DataLoader(batch_size=64)` benchmark, but real detector training remains batch-one with `accumulate: 4`; changing real training to batch 64 is a separate compatibility-sensitive implementation.

## Acceptance checks

Extend tests for profile validation, nested selection, fixed tuning IDs, six-way coverage, independent directories, setting rejection, resume/reuse, stop/budget boundaries, invalid inputs, NaNs, interruption, atomic saves, locks and idempotency. Confirm profiles cannot load one another's checkpoints.

Run `scripts/check_notebook_code.py`, ZIP and directory `scripts/check_consolidated_notebook.py` checks, and a tiny all-model profile test. Measure Colab time, memory and I/O before claiming that a profile fits 20 hours. Publish the notebook only after the checks pass.
