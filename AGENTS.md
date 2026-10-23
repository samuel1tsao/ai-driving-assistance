# Agent context and coding standards

This file is the context for AI agents and contributors. Keep `docs/` limited to
high-level research docs about the notebook's purpose and investigation; do not
add implementation plans, progress logs, or handoff notes there.

## Readable notebook and Python code

Readability is a primary requirement alongside correct behavior. A reader should understand the study's execution from its top-level cells without opening the implementation.

## Notebook reading order

- Keep one settings object near the top. Explain the fields that users change, with filesystem and Drive examples.
- Use a single numbered workflow: configure, set up, load, validate, train, review, export.
- Give each execution cell one purpose. Prefer one to five statements calling descriptive step functions.
- `notebooks/fog_density_study.ipynb` is the single source of study implementation for local Jupyter and Colab. Keep ordinary editable Python definitions in a few purpose-labeled sections collapsed in both; visible cells call the high-level steps. Do not generate helper modules, duplicate implementation in Python files, or encode source inside strings.
- Use one central constants/profile section for shared names, model specifications, seeds, checkpoint/report filenames, and training defaults. Keep configurable experiment settings separate from fixed contracts. Do not create constants for every dictionary key or every display/style value. Persisted schemas and dataset labels retain their original meanings.
- Edit `notebooks/fog_density_study.ipynb` directly and upload the same notebook file to Drive. Never generate helper modules or a second notebook copy.
- Explain what each step needs and produces. Separate result tables from charts and retain saved outputs in verification copies.
- Keep setup details out of result explanations. Never label a subset execution check as a scientific result.

## Python structure and names

- Use descriptive verbs for functions and concrete nouns for variables: `restore_detector_training_checkpoint`, `validation_loss`, `dataset_partitions`, `checkpoint_directory`.
- Avoid single-letter names in new workflow and training code. Existing public API names and persisted dictionary keys may remain for compatibility.
- Keep orchestration flat: its body should read as a sequence of named steps. Extract validation, state restoration, gradient accumulation, checkpoint writing and reporting into focused functions.
- Aim for functions with at most 40 logical lines. Split a longer function when it mixes responsibilities; a cohesive loop or class may be longer when splitting would hide its invariants. Do not add abstraction layers merely to satisfy a count.
- Avoid nested function closures in new training orchestration, semicolon-separated statements, compressed conditionals, and multiple unrelated actions on one line.
- Format Python with Ruff using an 88-character target. Imports after dependency installation are an intentional notebook exception.
- Comments explain constraints and recovery decisions. Docstrings describe contracts and side effects when they are not obvious from the function name.

## Checkpoint compatibility and recovery

- Report resumed progress cumulatively across saved epochs and image steps. Preserve previous loss history. Label training percentage separately from evaluation status and elapsed time.

- Preserve checkpoint filenames, payload keys, optimizer/scaler state, RNG state, training order, and run-identity checks during a refactor.
- Save checkpoints atomically. Only commit a detector training step after a completed optimizer update; discard unfinished gradients when interrupting.
- Restore loss statistics and RNG state to the same completed accumulation boundary when replaying an unfinished window.
- Reject changed experiment settings, inputs or package versions before replacing run metadata or checkpoints. Retain the existing compatible engine-migration policy.
- Reuse completed runs without retraining or rewriting training checkpoints. Repair a trailing incomplete prediction record while preserving valid earlier records.
- A runtime termination can recover only from the latest checkpoint that finished saving. Unsaved work is replayed; a Drive output directory must remain mounted and accessible.

## Verification
 
Before expensive training, run the notebook with the `setup` profile
(`TRAINING_PROFILE = PROFILE_SETUP`) to exercise data loading, all six arms,
checkpoints and exports on a smoke subset. CPU or smoke runs do not prove CUDA,
mixed-precision, Drive mount, or Colab timeout behavior; keep those limits
explicit, and never present a smoke run as a scientific result.
