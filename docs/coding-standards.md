# Readable notebook and Python code

Readability is a primary requirement alongside correct behavior. A reader should understand the study's execution from its top-level cells without opening the implementation.

## Notebook reading order

- Keep one settings object near the top. Explain the fields that users change, with filesystem and Drive examples.
- Use a single numbered workflow: configure, set up, load, validate, train, review, export.
- Give each execution cell one purpose. Prefer one to five statements calling descriptive step functions.
- Keep implementation sections collapsed initially, but retain ordinary editable Python source. Put one function or cohesive class in each implementation cell; do not encode source inside strings.
- Explain what each step needs and produces. Separate result tables from charts and retain saved outputs in verification copies.
- Keep setup details out of result explanations. Never label a subset execution check as a scientific result.

## Python structure and names

- Use descriptive verbs for functions and concrete nouns for variables: `restore_detector_training_checkpoint`, `validation_loss`, `dataset_partitions`, `checkpoint_directory`.
- Avoid single-letter names in new workflow and training code. Existing public API names and persisted dictionary keys may remain for compatibility.
- Keep orchestration flat: its body should read as a sequence of named steps. Extract validation, state restoration, gradient accumulation, checkpoint writing and reporting into focused functions.
- Aim for functions with at most 40 logical lines. Split a longer function when it mixes responsibilities; a cohesive loop or class may be longer when splitting would hide its invariants. Do not add abstraction layers merely to satisfy a count.
- Avoid nested function closures in new training orchestration, semicolon-separated statements, compressed conditionals, and multiple unrelated actions on one line.
- Format Python with Ruff using an 88-character target. Imports after dependency installation and module-file generation are intentional notebook exceptions.
- Comments explain constraints and recovery decisions. Docstrings describe contracts and side effects when they are not obvious from the function name.

## Checkpoint compatibility and recovery

- Report resumed progress cumulatively across saved epochs and image steps. Preserve previous loss history. Label training percentage separately from evaluation status and elapsed time.

- Preserve checkpoint filenames, payload keys, optimizer/scaler state, RNG state, training order, and run-identity checks during a refactor.
- Save checkpoints atomically. Only commit a detector training step after a completed optimizer update; discard unfinished gradients when interrupting.
- Restore loss statistics and RNG state to the same completed accumulation boundary when replaying an unfinished window.
- Reject changed experiment settings, inputs or package versions before replacing run metadata or checkpoints. Retain the existing compatible engine-migration policy.
- Reuse completed runs without retraining or rewriting training checkpoints. Repair a trailing incomplete prediction record while preserving valid earlier records.
- A runtime termination can recover only from the latest checkpoint that finished saving. Unsaved work is replayed; a Drive output directory must remain mounted and accessible.

## Required regression checks

Run the inexpensive checkpoint tests before launching an expensive training workload:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_notebook_code.py
```

The checkpoint tests extract and import the actual notebook modules. Small CPU models test accumulated-gradient interruption, persisted-checkpoint recovery, optimizer/RNG/history parity, unchanged checkpoint payloads, completed-run reuse, incompatible-run rejection, density recovery, atomic-save failures, concurrent writers, prediction-file repair, and original pre-refactor reference results. They do not download weights or train the study models.

When orchestration, data loading, or module boundaries change, also execute the real-data subset:

```powershell
.venv/Scripts/python.exe scripts/check_consolidated_notebook.py --zip --images 2
.venv/Scripts/python.exe scripts/check_consolidated_notebook.py --images 2
```

These commands reuse the installed local dependencies and isolate test outputs. They do not start the full study. CPU tests do not prove CUDA, mixed-precision, Drive mount, or Colab timeout behavior; keep those limits explicit.
