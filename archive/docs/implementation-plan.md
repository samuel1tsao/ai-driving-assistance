# Driving assistant implementation plan

Both deployment modes requested; local compute spike implemented (see deployment-modes.md).

## Scope and assumptions
First deliver offline model comparison, then a phone prototype. Existing: CUDA setup, profiler, trainer and BDD template. Missing: converter, evaluator, tests, app/server. No approved feature plan exists; app behavior is proposed. Open: phone/OS, local or remote backend, budget, deadline and licensing intent. Defer ROS 2, depth and TTC alerts.

## Phases
1. Prepare authorized BDD data: audit annotations/duplicates; freeze scene-disjoint train/validation/test manifests. Version COCO/BDD mapping with explicit rider/person and unsupported-class policies.
2. Baseline YOLOv8n, YOLO11n, YOLO26n/s on identical data. Report AP, class recall, false positives/frame and night/rain/small-object slices; profile road replay. Gate: checked mappings and reproducible results.
3. Fine-tune with matched splits, augmentation and budget; start 30 epochs, batch 4, 640 pixels. Tune thresholds on validation; repeat finalists across three seeds. Compare common-class metrics, report full BDD separately. Select recall at matched false-positive rate, daytime regression and latency. Set accuracy gates after baseline, before final testing.
4. Support both phone nano and backend small modes. Measure 15-minute capture-to-overlay latency, FPS, thermals, memory and network loss. Proposed goals: >=15 processed FPS, p95 <=150 ms, stale cutoff 250 ms; not safety guarantees. Tune each mode from evidence.
5. App: choose phone/backend -> permission -> start -> detections -> stop. Show degraded/unavailable states and frame age. Switch modes only while stopped; never silently upload. Depth/TTC needs a separate validated plan.

## Proposed files and contracts
scripts/prepare_data.py -> src/driving/data.py validates inputs and emits manifests/YOLO labels; configs/taxonomy.json stores mappings. scripts/evaluate.py -> src/driving/evaluation.py maps predictions and emits metrics. Extend scripts/train.py and scripts/profile_models.py with frozen configs and run metadata. Add tests/ fixtures. Mobile/server paths and frameworks await device selection.

Schema v1 image: {id, scene_id, sha256, split, width, height, attributes, boxes}; box: {class_id, normalized_xyxy}; prediction adds score. Reject missing images, invalid classes, nonfinite/out-of-range or degenerate boxes. Run: {id, dataset_hash, model_hash, config, seed, versions, device, metrics, status}. Error: {code, message, item_id, retryable}; nonzero CLI exit.

## Storage and integration
Raw data/manifests are source of truth. JSON/JSONL per run, unique IDs, atomic publish and exclusive run lock. Conversion is hash-idempotent; incompatible schemas fail. No database/index migration or backfill yet.
Acquire data manually with provenance. Cache/checksum weights; bounded download timeout/retries and explicit failure. Offline: no service auth, pagination or webhooks. Proposed backend: encrypted authenticated session, versioned frame IDs/timestamps and detection responses, one pending latest frame, payload/rate limits, stale/out-of-order rejection. Reconnect never replays stale frames. Exact transport limits follow the spike.

## Tests and rollout
Cover valid/empty scenes, hand-checked matches, missing/corrupt images, malformed boxes, duplicate/leaked scenes, unsupported classes, boundary thresholds, slice filtering/order and schema rejection. Test partial writes, disk failure/recovery, concurrent run IDs, repeat conversion, GPU OOM and unavailable weights. Long replay/oversized-input tests log stage latency, frame age, drops and memory against phase-4 targets. App tests: permission denial, auth expiry, timeout/retry, loss/reconnect, stale responses, start/stop races, responsive layout and accessible state text. Start with recorded replay, then stationary trials; retain baseline checkpoints for rollback.
