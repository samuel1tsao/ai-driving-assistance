# Independent training profiles

Status: implemented in the consolidated study notebook.

Each profile is an independent experiment from the original pretrained weights. Results are not cumulative model training.

| Profile | Training images | Suggested starting cap | Scope |
|---|---:|---:|---|
| setup | Scene-safe smoke subset | 1 detector / 1 density epoch | Verify GPU, data, all six arms, checkpoints and exports |
| partial-2k | Up to 2,000 detector-training images | 5 detector / 5 density epochs | Preliminary six-way comparison |
| partial-5k | Up to 5,000 detector-training images | 5 detector / 5 density epochs | More-data six-way comparison |
| full | All training data | Existing full run, then its recorded settings | Continue the saved study |

Each profile runs all three models with RGB control and predicted density. Setup uses a scene-safe smoke subset; partial profiles use deterministic scene-safe caps and fixed tuning IDs. Use seed 42 for preliminary profiles. The existing full run keeps its recorded seeds/configuration; new repetitions can use seeds 43/44.

Use nested subsets and one fixed tuning subset. New profiles have separate outputs and resume only their own checkpoints. The existing full run is the exception: preserve its output directory and continue it with the exact recorded settings. No new profile loads another profile's weights or history.

Before training, display counts, combinations, epoch caps, batch/effective-batch and loader settings, ETA, paths and profile. Reduce data or epochs when needed; never drop a model/arm. For the existing full run, changes to batch, accumulation, workers/prefetch or other recorded settings must be rejected.

Early stopping is independent per run. Reports include actual epochs, stop reason, AP, latency, density quality, paired deltas and curves. Cross-profile charts must label profiles as independent.

Partial profiles use their fixed tuning rows for preliminary metrics and do not copy or evaluate final-test rows. The full profile uses the held-out final-test split.

The notebook records the profile, selected IDs and hashes, epoch caps, evaluation split, stopping reason, and batch/loader contract in the run metadata. Real detector training remains a one-image micro-batch with four accumulated gradients; the separate synthetic GPU benchmark uses `batch_size=64` to measure a throughput ceiling.
