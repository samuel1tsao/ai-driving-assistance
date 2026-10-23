# Fog-density object detection research

## Question

CMPE 249 notebook study: does giving a camera-only detector an explicit fog-density
estimate change its accuracy and latency? The working hypothesis is that density
conditioning may cost a little average accuracy but improve accuracy in fog.
Positive, null and negative outcomes are all valid results that students should be
able to rerun and explain.

## Models and arms

YOLO11n, YOLO11m and DETR are each fine-tuned twice from the same pretrained
weights:

- **RGB control:** detector plus an input adapter that receives a zero fog scalar.
- **Predicted density:** identical detector and adapter that receive the estimated
  beta from a frozen fog-density model.

The only input difference between arms is the fog scalar. Initialization, seeds,
data and training budget are matched, and density estimation is included in
latency. Oracle (known) beta is optional and only allowed when every matched image
has a ground-truth beta.

## Data

- **nuScenes:** all labeled trainval keyframes across six cameras (204,894 images,
  850 scenes). Official train scenes are split by scene into training and tuning;
  official val is the held-out final test. Projected 3D boxes are an imperfect 2D
  target, not official nuScenes 3D mAP. nuScenes has no measured beta, so its
  results are not a verified clear-weather benchmark.
- **Foggy Cityscapes:** beta (attenuation coefficient, m^-1) is parsed from
  filenames; `0.0` is clear and larger values (`0.005`, `0.01`, `0.02`) are denser
  synthetic fog. These images train the separate density estimator. A prediction
  learned from synthetic fog is not a calibrated measurement of real nuScenes fog.
- **Cityscapes `gtFine` + matched clear images:** provide shared boxes so the same
  scenes can be scored in clear, light, moderate and dense fog. This matched
  benchmark is the direct test of the hypothesis.

Density-only images never become detection negatives. Whole scenes stay in one
split, and test images never contribute gradients.

## Experiment profiles

Each profile is an independent experiment from pretrained weights, not cumulative
training. Every profile runs all three models in both arms; data or epochs may be
reduced, but a model/arm is never dropped.

| Profile | Detector-training images | Epoch cap | Purpose |
|---|---:|---:|---|
| setup | Scene-safe smoke subset | 1 detector / 1 density | Verify GPU, data, arms, checkpoints, exports |
| partial-2k | Up to 2,000 | 5 / 5 | Preliminary six-way comparison |
| partial-5k | Up to 5,000 | 5 / 5 | More-data six-way comparison |
| full | All | 30 / 15, seeds 42/43/44 | The study result |

Partial profiles use nested, deterministic scene-safe subsets and a fixed tuning
set for preliminary metrics; only the full profile evaluates the held-out final
test split. Early stopping is independent per run. Cross-profile charts must label
profiles as independent.

## Measurements

- **Accuracy:** one COCO bbox evaluator: mAP@[.50:.95], AP50, AR100,
  precision/recall at confidence .25 and IoU .5, and per-class AP over the same
  seven categories, preprocessing and image sets.
- **Hypothesis:** clear vs. fog mAP, per-beta curves, and paired arm deltas on
  the matched Cityscapes scenes.
- **Density quality:** beta MAE/RMSE, residuals by bin, true-vs-predicted scatter
  (kept separate from detection quality).
- **Latency:** end-to-end mean/p50/p95 and FPS, including decoding, resizing,
  transfer, density estimation, detector/adapter and postprocessing.

## Interpreting results

- Loss histories describe training/tuning; detection quality comes from COCO
  metrics, not loss.
- Seed standard deviation is not a confidence interval. One seed cannot establish
  significance; bootstrap intervals are only reported with at least three seeds.
- Runs without matched labeled clear/fog data leave the hypothesis untested.
  Support is only claimed once clear/fog deltas exist across multiple seeds.
- Failures are shown, never replaced with fabricated scores.

## Sources

[Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/),
[YOLO11](https://docs.ultralytics.com/models/yolo11/),
[DETR model card](https://huggingface.co/facebook/detr-resnet-50).
