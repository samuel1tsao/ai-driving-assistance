# Fog-versus-clear validation implementation plan

## Goal and current gap

Test whether density conditioning lowers average accuracy but improves fog accuracy. The recent run did not test this: detection rows were nuScenes `val` images with `beta=None`; Foggy Cityscapes supplied density supervision

## Data contract

`beta` is the attenuation coefficient parsed from Foggy Cityscapes filenames, in m^-1: `0.0` is clear; positive values such as `0.005`, `0.01`, and `0.02` are denser fog. It is the adapter scalar: `rgb_control` supplies zero, `predicted_density` the estimator output, and `oracle_density` known beta. nuScenes has no measured beta and stays `unknown`.

The city manifest must build `gtFine` boxes, preserve clear/fog pairs, assign splits, and record pair, condition, beta/bin, label status, and provenance. Missing masks, duplicate pairs, invalid beta, leakage, and unsupported oracle fail preflight. Beta-only images are never detection negatives.

## Configuration and report contract

Add `evaluation_datasets`, `fog_detector_scope`, `fog_density_bins`, and evaluation-only checkpoint source to the profile and fingerprint. Keep nuScenes (normal use) and Foggy Cityscapes (in-domain fog) separate; require balanced clear/fog and beta-bin counts while retaining natural rows.

The notebook’s manifest/settings are authoritative. Preserve profiles, checkpoints, resume, and version behavior. Export results, paired deltas, seed summaries, and `fog-vs-clear.csv` with dataset, condition, beta/bin, model, arm, seed, and split. Reuse matching checkpoints for evaluation-only runs; version changed inputs/scope.

## Beginner-readable analytics and figures

Every table/figure needs a plain-language title, labels, a “how to read this” note, counts, and a seed warning:

- **Dataset audit:** counts by dataset/split, clear/fog, beta bin, class, pair coverage, label completeness, and imbalance.
- **Training health:** train/validation loss by epoch, best epoch, early stopping, runtime, throughput, and checkpoint status.
- **Detection quality:** mAP50-95, AP50, precision, recall, AR100, and per-class AP. Include mean, standard deviation, seed count, and paired deltas.
- **Hypothesis result:** clear-versus-fog bars, fog-minus-clear tables, beta-bin curves, and a model-by-metric heatmap; state whether clear accuracy is preserved and fog accuracy improves.
- **Density quality:** true-versus-predicted beta scatter, MAE/RMSE, residuals by bin, and prediction histograms; keep this separate from detection quality.
- **Deployment tradeoff:** accuracy-versus-latency scatter, mean/p50/p95 latency, FPS, and density-estimation overhead.
- **Class/error view:** per-class AP heatmaps and, when supported, precision-recall or confusion plots. Loss alone is not detection quality.

## Single-notebook flow and readability

Keep one authored notebook at `notebooks/fog_density_study.ipynb` for local Jupyter and Colab. Preserve its config, profiles, checkpoints, resume, and preflight. Organize visible cells into short sections: choose profile, audit data, reuse checkpoints, evaluate clear/fog, render analytics, and interpret results. Keep helpers labeled; do not add user-facing modules or duplicate config. Upload this same file to Drive; Git tracks the one notebook and this plan.

## Tests and rollout

Test setup, preflight, training/resume, evaluation-only reuse, paired clear/fog metrics, all models/arms, oracle known-beta rows, multiple seeds, balanced bins, legacy reports, deterministic bins, COCO parity, finite metrics/latency, and bounded 2k resources. Cover bad/missing beta or masks, duplicate/empty groups, no labeled fog rows, leakage, stale settings, unknown beta, unsupported oracle, interruption, partial writes, and missing dependencies.

Roll out with setup/preflight, a small labeled fog evaluation using existing checkpoints, then retraining if needed. Do not claim support until clear/fog deltas exist across multiple seeds.

## Implemented behavior

The notebook now has `evaluate` for a separate, resumable matched-Cityscapes evaluation of existing best checkpoints. This is explicitly a domain-transfer benchmark for nuScenes-trained weights. Its fingerprint records source training settings, checkpoint and mask file versions, selected images, pair cap and evaluation split. Original normal-use metrics retain their dataset and split. Training profiles, full continuation and per-profile versions remain intact; setup resets under its run lock while preserving the current subset audit.

The report includes clear/pooled-fog mAP, per-beta curves, paired clear/fog arm effects and their difference, seed summaries and exploratory seed-bootstrap intervals only with at least three seeds. Graphs cover condition accuracy, effects, AP by class/condition, COCO precision-recall at IoU .50, accuracy/latency, beta scatter/errors, data balance and available source loss histories. Tables retain AP50, precision, recall, AR100, latency percentiles, FPS, beta MAE/RMSE, class coverage and checkpoint health. A confusion matrix is not substituted for COCO matching. Older runs without matched labeled data explicitly leave the hypothesis untested; unrecorded historical training runtime/throughput is not inferred from inference timing.

The consolidated notebook is the sole authored engine. The existing support files and Colab artifact are generated from its ordinary Python cells. CPU fixtures validate execution and graph layout; actual fog findings still require the original clear images and `gtFine` labels, which are absent from the current density-only manifest.
