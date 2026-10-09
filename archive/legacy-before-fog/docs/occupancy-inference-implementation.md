# Real occupancy inference setup

## Requested workflow

The notebook selects one held-out dataset sample list, loads pretrained models, generates occupancy grids, measures synchronized inference latency, scores the grids against visibility-masked labels, and shows matching examples. Dataset folders and executable paths are explicit; do not modify user environment variables.

## Verified model compatibility

- FlashOCC M1 R50: published Occ3D-nuScenes checkpoint and a 200×200×16, 18-class grid. Official setup uses Python 3.8, PyTorch 1.10 and MMCV 1.5.3 with compiled CUDA operations.
- GaussianOcc: published nuScenes semantic occupancy checkpoint; official setup uses Python 3.8, PyTorch 1.9.1 and compiled Gaussian rasterization extensions. Its self-supervised training protocol must be disclosed in comparisons.
- OccFormer: the original released nuScenes checkpoint is for LiDAR segmentation, not a validated Occ3D 18-class occupancy checkpoint. Do not silently reshape or relabel its output and report comparable free-space scores. Its adapter needs a verified compatible checkpoint or separate training/evaluation scope.
- Waymo: downloaded labels are usable, but the above nuScenes checkpoints are not automatically valid Waymo runners. Begin actual inference on nuScenes.

## Implementation order

1. Verify a working Linux CUDA runtime through Docker. Keep the working detector notebook environment intact.
2. Obtain official sources and checkpoints, recording provenance.
3. Prepare a small fixed nuScenes validation subset with the six camera images, calibration and poses; match every input to its actual occupancy frame token.
4. Run FlashOCC on the subset, export predictions and timing, and validate axes, class IDs and visible-voxel metrics.
5. Add GaussianOcc on those same frames and report training-protocol differences.
6. Connect the tested runner to a short notebook cell. Keep unsupported checkpoints explicit.

## Validation

- Positive: a real checkpoint produces a correctly shaped prediction on a real held-out frame.
- Negative: missing runtime, calibration, checkpoint, incompatible dataset or output taxonomy fails explicitly.
- Performance: batch-one FP32 after warm-up, synchronized CUDA timing; record GPU and input preparation scope.
- Idempotency/concurrency: reuse verified downloads; avoid concurrent GPU benchmarks and partial prediction files.

Sources: [FlashOCC installation](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md), [OccFormer model zoo](https://github.com/zhangyp15/OccFormer#model-zoo), [GaussianOcc setup](https://github.com/GANWANSHUI/GaussianOcc).
