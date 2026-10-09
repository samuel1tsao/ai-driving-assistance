# Fog-density detection research

**Scope/value.** CMPE 249 notebook study: measure whether explicit fog density changes camera-only detection accuracy and latency. Fine-tune YOLO11n, YOLO11m and DETR on all available labeled nuScenes keyframes across six cameras. Students can rerun and explain positive, null or negative outcomes.

**Data.** Keep whole scenes in training, tuning or final test; all labeled frames participate without evaluating on training data. Require complete six-camera keyframe coverage before training. Use the supplied Untitled2 notebook's Foggy Cityscapes images to train a separate beta estimator from filenames; density-only images never become detection negatives. gtFine is optional for a separate Cityscapes detection benchmark.

**Comparison.** Jointly fine-tune each detector and an identical input adapter with zero or estimated beta. Match pretrained initialization, seeds, data and budget. Density estimation is included in latency. Oracle beta is optional and requires known labels for every matched image. Native visibility is not beta. Use tuning loss for selection; report final mAP, AP50, recall, precision, per-class AP, beta error and mean/p95 latency with paired deltas and seed spread.

**Experience.** Open notebook, set dataset roots/budget, inspect coverage and blockers, Run All, follow progress, resume after interruption, then inspect tables/plots or export analytics without retraining.

**Acceptance/edges.** Reject malformed, missing, duplicate/nonfinite inputs, invalid classes and scene leakage. Preserve unknown beta, crowds, empty labels and unsupported classes. Show failures; never fabricate scores. Save optimizer/RNG state; lock concurrent writers; reject stale settings; record source/input fingerprints. Bound GPU memory through microsteps and accumulation. Use labeled readable plots and scrollable tables. Projected boxes are a 2D proxy; improvements require completed matched runs and uncertainty analysis. Archive unrelated work.
