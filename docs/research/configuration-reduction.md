# Minimum configuration for the fog study

## Recommendation

The notebook should expose one small editable block:

```python
RUN = {
    "profile": "partial-2k",
    "data_root": Path("..."),
    "output_root": Path("..."),
    "models": ["YOLO11n", "YOLO11m", "DETR"],
    "seed": 42,
}
```

Profiles should be a short base configuration plus overrides for training and
held-out sample counts. Device, AMP, checkpoint paths, manifest paths, and
derived output paths should be computed after this block rather than edited.

## Remove from the editable surface

- Environment profiles and `RUN_ENVIRONMENT`; choose paths directly in `RUN`.
- Local archive variables such as `LOCAL_CITYSCAPES_ARCHIVE`; ingestion should
  use one explicit data root and the documented archive layout.
- Run names, version strings, output filenames, and report filenames.
- Result/report artifact constants (`class-coverage.csv`, result summaries,
  hypothesis tables, audit tables, and figure filenames).
- Settings migration, retrain switches, test scopes, and source-run selection.

## Keep internally

- Detector model names/checkpoints and class-ID mappings (fixed study
  contracts, not routine run settings).
- Dataset manifest locations and the density model constants.
- `best.pt` and `last.pt`, optimizer/scaler/RNG state, and progress metadata.
- Profile overrides for epochs, training image cap, held-out image cap, image
  size, and evaluation split.

## Output model

Training checkpoints remain on disk so interrupted runs can resume. Evaluation
and analytics should return DataFrames and display them in notebook cells.
Findings do not need CSV/JSON/PNG report files or a second output directory;
rerunning evaluation regenerates the tables and charts from the checkpoints.
