# FiLM conditioning experiment

The experiment in [`film_experiment.ipynb`](../notebooks/film_experiment.ipynb)
asks whether fog-density conditioning inside a detector changes object detection
on foggy and clear images. It extends the baseline study in
[`fog_density_study.ipynb`](../notebooks/fog_density_study.ipynb).

## Six matched comparisons

YOLO11n, YOLO11m, and DETR each train in two arms:

| Arm | Detector input and architecture | Fog-density inference |
|---|---|---|
| `without_film` | Original pretrained detector, fine-tuned on RGB | Skipped |
| `with_film` | Same detector plus a feature FiLM layer | Frozen predictor supplies beta |

Both arms use identical detector initialization, training images, scene splits,
seed, image order, and training budget. Both see the same mixture of nuScenes,
labeled Foggy Cityscapes, and matched clear Cityscapes. The comparison measures
adding FiLM and its predictor at inference; the additional FiLM parameters are
reported. It does not isolate the effect of additional capacity from the effect
of the density information.

Known beta labels never condition detector features. They supervise the separate
fog-density predictor and group evaluation images. The predictor is fitted or
restored once per profile and frozen for all three FiLM detector runs.
Its supervision comes from the Foggy Cityscapes beta manifest. Synthetic beta
estimates are not assumed to be calibrated measurements of real nuScenes weather;
density-estimation error is reported separately from detection quality.

## Feature conditioning

For a feature map with C channels, a small network maps normalized predicted beta
to C scale adjustments and C shifts:

```text
normalized_beta = predicted_beta / 0.02
scale, shift = Linear(32, 2*C)(SiLU(Linear(1, 32)(normalized_beta)))
conditioned_features = features * (1 + scale) + shift
```

Scale and shift broadcast across spatial positions. The final linear layer starts
at zero, so FiLM initially preserves the pretrained features exactly. Each
detector learns its own FiLM weights while the density predictor stays frozen.

| Detector | FiLM location | Role |
|---|---|---|
| YOLO11n | Final backbone C2PSA output, `model.10` | Conditions features before the neck |
| YOLO11m | Final backbone C2PSA output, `model.10` | Same location; actual channel count is read from the model |
| DETR | Output of `model.input_projection` | Conditions projected backbone features before the transformer |

These locations follow the [YOLO11 architecture](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)
and [DETR implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py).
The notebook checks each target's type and attaches a
[PyTorch forward hook](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
to transform its output. A no-FiLM detector has no conditioning hook or trainable
adapter. This extension replaces the baseline's shallow RGB residual adapter.

## Evaluation and interpretation

Every arm uses the same COCO bounding-box evaluator over person, car, truck, bus,
train, motorcycle, and bicycle. The matched comparison requires complete clear,
light, moderate, and dense versions of each held-out Cityscapes scene with the
same labels. Partial profiles use validation scenes; the full profile uses the
final test split. Scene overlap with fitting data is rejected.

Results are grouped by dataset and weather condition: clear beta 0, pooled foggy
beta greater than 0, each individual beta, and unknown beta. nuScenes images
without measured beta stay in the unknown-weather group. Pooled fog results count
each fog variant as an image; their scenes are repeated across fog levels.

The analytics include:

- Overall mAP@[.50:.95], AP50, AP75, AR100, and precision/recall/F1 at confidence
  .25 and IoU .50.
- Per-class AP@[.50:.95], AP50, AP75, AR100, precision/recall/F1, and noncrowd
  ground-truth object and image counts. A class without ground truth has missing
  AP, rather than a fabricated zero. Per-class average precision is AP; mAP is
  the mean over supported classes.
- FiLM minus no-FiLM differences matched by model, seed, class, and weather
  group. Positive AP differences mean an improvement on those evaluated images.
- End-to-end mean/p50/p95 latency, serial FPS, density inference time, detector
  and FiLM parameter counts, peak allocated GPU memory, and checkpoint size.
  FiLM timings include its density predictor and feature modulation.
- Training and validation loss, data coverage, density-estimation error, and
  clear/fog effect tables. GPU memory describes the measured evaluation process;
  it is not an isolated parameter-storage measurement.

One seed describes an individual run and does not establish significance.
Exploratory bootstrap intervals appear only when at least three independent
training seeds have actually contributed results. The visible six-run workflow
uses the selected seed; separate seed experiments require separate output roots
and must be combined deliberately for uncertainty analysis.

## Running and saved outputs

Use `PROFILE_SETUP` first. It retains all six detector comparisons with reduced
scene-safe detector and density subsets and one detector/density epoch. This is an execution check, not a
scientific study result. Larger profiles retain the matched design.

Prepared data is shared with the baseline. FiLM artifacts use
`models/film/<profile>/<model>/<arm>/seed-<seed>/` locally and
`/content/drive/MyDrive/models/film/<profile>/...` on Drive. Ordinary evaluation
reports and figures live under the profile; the saved-checkpoint matched
comparison has its own `matched-evaluation/` subdirectory.

The `reports/` directory includes `detection-metrics.csv`,
`per-class-metrics.csv`, `film-deltas.csv`, `per-class-deltas.csv`,
`seed-summary.csv`, `density-quality.csv`, `data-coverage.csv`,
`fog-vs-clear.csv`, and `hypothesis-summary.csv`. Generated charts are saved in
`figures/`. The export ZIP packages these outputs with the checkpoints.

FiLM weights remain in the checkpoint's `adapter` field; detector weights remain
in `model`. The factory reconstructs the correct hook before restoring the
weights. Architecture, arm, settings, input labels/files, predictor weights, and
package versions participate in compatibility checks. Changed experiments use a
new output root. Completed detector runs reuse best weights without rewriting
training checkpoints, and prediction files resume from completed records.

Mixed-precision overflow replays the unfinished accumulation window at a lower
gradient scale, restoring its original random state and loss statistics. Image
progress and checkpoints advance only after a completed optimizer update.
Interrupting an accumulation window discards its partial gradients; those images
are replayed after resuming from the last saved completed boundary.
