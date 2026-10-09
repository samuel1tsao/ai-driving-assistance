# Basic notebook evaluation

Open [notebooks/model_evaluation_basics.ipynb](../notebooks/model_evaluation_basics.ipynb) in Jupyter and run it from top to bottom.

The notebook is intentionally simple: load eight detectors, point them at the same labeled validation set, collect mAP/precision/recall and collect per-image preprocessing/inference/postprocessing time from the same validation pass. It writes `reports/notebook/model_eval_results.csv` and `.json`. The models are YOLOv8n, YOLO11n, YOLO26n, YOLO26s, RT-DETR-L, Faster R-CNN R50-FPN, RetinaNet R50-FPN, and SSD300 VGG16. YOLO and RT-DETR use the Ultralytics API; the other three use torchvision's official COCO weights and a shared adapter that converts their predictions into the notebook's metric format.

The default `coco8.yaml` run is only a smoke test. It checks that the notebook, checkpoints and Ultralytics validation API work. It is too small for a model decision. For a general-purpose labeled benchmark, change `USE_FULL_DATASET = True` and use COCO validation. COCO val2017 has 5,000 labeled images, but Ultralytics' standard `coco.yaml` download can include the 19 GB COCO training archive; inspect the prompt and available disk before continuing.

For driving accuracy, prepare BDD100K detection data and set `DATA_YAML` to its local YOLO-format YAML. BDD100K is the better source for day/night, weather and driving-camera conditions. It is not an apples-to-apples validation set for untouched COCO checkpoints: class IDs and label meanings differ. Evaluate raw checkpoints on COCO first, then fine-tune on BDD100K and evaluate on held-out BDD100K scenes with an explicit taxonomy.

## Run

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv pip install --python .venv/Scripts/python.exe -r requirements-notebook.txt
.venv/Scripts/python.exe -m jupyter lab
```

Open the notebook, run the smoke version, then change `USE_FULL_DATASET` or `DATA_YAML` in the configuration cell. Keep `EVAL_FRACTION = 1.0` for reportable results. The first full-data run can be slow and may require substantial disk space. Missing model files are downloaded by the matching official loader; missing dataset files are resolved by `check_det_dataset` from the configured YAML.

The notebook also renders up to three random validation images for visual inspection. Cyan boxes are labels; lime boxes are predictions and include confidence scores. Use the plots to catch missed objects, duplicate boxes, incorrect classes, and visually poor localization that a single aggregate score can hide. The smoke run remains a pipeline check on four COCO8 images; repeat the benchmark on a full held-out validation set before choosing a model.

## Why these tests are the right first tests

Object detection accuracy must be measured against boxes that a human or dataset annotator labeled. Confidence scores and a single street photo cannot measure missed objects. mAP summarizes box overlap; recall exposes missed objects; precision exposes false alarms. Speed is measured on the same images so the comparison is controlled.

This is an adequate first model-screening experiment, not a complete driving-assistance evaluation. A driving-specific follow-up must add BDD100K slices, held-out scene splits, small-object and adverse-weather recall, false alarms per frame, sustained phone thermals, capture-to-warning latency, tracking, metric depth and time-to-collision validation.
