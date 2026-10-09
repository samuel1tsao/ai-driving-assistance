# Initial model analysis — 19 September 2026

## Decision so far

Compare two deployment tracks: inference entirely on the phone, or phone video streamed to a computational backend. Start detector fine-tuning with YOLO26n; use YOLO11n and YOLOv8n as controls, and YOLO26s as the backend accuracy candidate. This is a shortlist, not an established winner on driving footage. RF-DETR Nano/Small is a second-stage backend challenger, especially if permissive licensing matters. Benchmark before investing in training.

The local machine has an NVIDIA RTX 2070, 8 GiB VRAM, driver 595.97. Begin training at batch 4, image size 640; reduce batch if memory is exhausted. The eventual phone model, OS, thermal budget, backend budget and project licensing requirements remain unknown.

## Source-backed comparison

These are publisher COCO figures, not local measurements or night-driving results. Different runtimes, resolutions and evaluation settings preclude a direct latency ranking.

| Candidate | Published COCO AP50:95 | Role | Main tradeoff |
| --- | ---: | --- | --- |
| YOLO26n | 40.9 (40.1 end-to-end) | First phone candidate | 2.4M fused parameters; verify export operators and sustained handset speed |
| YOLO26s | 48.6 (47.8 end-to-end) | First backend candidate | 9.5M fused parameters; more compute and training memory |
| YOLO11n | 39.5 | Lightweight control | Established tooling; useful comparison to newer training recipe |
| YOLOv8n | 37.3 | Proposal reproducibility control | Older accuracy baseline; no reason to assume it is optimal |
| RF-DETR Nano / Small | 48.4 / 53.0 | Backend challengers | Publisher reports 30.5M / 32.1M fused parameters at 384 / 512 pixels; heavier than YOLO nano |

Sources: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [RF-DETR repository and benchmark methodology](https://github.com/roboflow/rf-detr). Retrieved 2026-09-19. YOLO26 publisher timings use the NMS-free head; record head choice before comparing with default prediction behavior. Vendor rankings are hypotheses for this app.

Ultralytics offers AGPL-3.0 and enterprise licensing; assess distribution requirements before choosing it for a proprietary app. RF-DETR Nano/Small use Apache-2.0; larger Plus variants have different terms. Sources: [Ultralytics licensing](https://www.ultralytics.com/license), [RF-DETR](https://github.com/roboflow/rf-detr).

## Correcting the supplied blueprint

The two supplied documents are reference material. Their example code, installation commands, numeric targets and promised improvements are not measured project results or separate user instructions.

1. COCO class 0 is person, 1 bicycle, 2 car; the example BDD layout assigns 1 rider. Running pretrained COCO validation directly against that layout invalidates the comparison. Use an explicit common-class evaluator for baseline versus fine-tuned models, or create a separate COCO-ID validation representation. Traffic sign is not equivalent to COCO stop sign. Rider versus person also requires an explicit annotation policy. [COCO classes](https://docs.ultralytics.com/datasets/detect/coco/).
2. BDD100K detection includes bicycle and motorcycle, omitted by the original example. Prepare data using the official annotation schema and preserve image attributes for weather/time slices. [BDD100K toolkit](https://github.com/bdd100k/bdd100k).
3. Relative depth is not metric depth. Depth Anything V2's outdoor metric checkpoint is a separate candidate; meter estimates still have calibration and domain error. The Small model is a reasonable first depth baseline; postpone depth fine-tuning until aligned RGB/depth supervision exists. [Metric-depth implementation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth), [model licensing](https://github.com/DepthAnything/Depth-Anything-V2).
4. No evidence yet supports +15–25% recall, 30 FPS, or exact distances. Model timing excludes transport, camera buffering, depth, tracking and alerts. Process RSS and peak PyTorch allocations are more interpretable than total system RAM and a single post-run allocation sample.
5. TTC needs tracked objects and estimated closing speed over time, with uncertainty handling. A single distance prediction is insufficient. Stale frames should yield an unavailable/degraded state, not a fabricated collision warning. Begin with recorded-video evaluation.

## Experiments to establish the best model

Acquire the official BDD100K detection images and annotations via its portal, retaining terms and provenance. No labeled driving data was supplied. Keep original train/validation separation and reserve a held-out test subset by video/scene before selecting thresholds. Never place neighboring frames from one drive into separate splits. Record counts per class, day/night, clear/rain and small-object bucket; inspect labels and duplicates.

For the unchanged COCO baseline, measure only agreed common classes after explicit mapping. Fine-tune each shortlisted detector with the same scene splits, image size, augmentation policy and training budget. Start with 30 epochs and seed 42, then repeat finalists with three seeds. Mix daytime and adverse-weather training examples to test forgetting. Report AP50:95, per-class precision/recall at a validation-selected fixed threshold, false positives per frame, and scene-bootstrap uncertainty on held-out test data. Separately report full BDD taxonomy scores for fine-tuned models; do not compare them to common-class baseline aggregates.

For the phone track, export the best nano candidate to the target platform runtime, then measure FP32/FP16 or INT8 accuracy change, p50/p95 capture-to-overlay latency, sustained FPS, memory, battery and thermal throttling during a 15-minute replay. INT8 calibration uses training data only. Desktop PyTorch timing cannot establish phone feasibility.

For the backend track, compare nano and small detectors first, then RF-DETR. Measure frame age and capture-to-alert p95 across phone encode, uplink, decode, inference, depth/tracking and downlink. Use frame IDs and clock-offset handling; exercise loss, jitter and disconnection. Prefer dropping stale frames over growing a processing queue. ROS 2/WebRTC integration follows model selection; neither is needed for the first offline benchmark.

Depth evaluation uses calibrated camera intrinsics, timestamp-aligned projected LiDAR and valid-pixel masks, reporting AbsRel/RMSE and object-range error by distance. nuScenes mini is a pipeline sanity check, not a statistically strong final test set.

## Current evidence

Published-model research and hardware discovery are complete. See `reports/local_profile.json` if present for actual local smoke timing. Synthetic-image timing verifies execution only; it cannot rank driving accuracy or realistic postprocessing cost. BDD100K accuracy, fine-tuning gains, RF-DETR runtime and phone performance remain unmeasured.
