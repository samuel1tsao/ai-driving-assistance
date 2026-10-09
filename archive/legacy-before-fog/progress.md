# Occupancy inference implementation progress

Plan: [real inference setup](docs/occupancy-inference-implementation.md).

| Work | Status | Evidence / next step |
| --- | --- | --- |
| Audit model compatibility | DONE | FlashOCC/GaussianOcc document Occ3D-nuScenes; original OccFormer nuScenes checkpoint is LiDAR segmentation |
| Isolated Linux GPU runtime | DONE | Docker/PyTorch 1.10 CUDA 11.3 sees RTX 2070; MMDetection3D and FlashOCC operations compiled |
| Official sources and checkpoints | DONE | Pinned sources and M0/M1/GaussianOcc weights downloaded; rebuild script records origins |
| Real camera/calibration input subset | DONE | 100 official validation frames, 600 matched images, original calibration and poses |
| FlashOCC inference/export | IN_PROGRESS | M1 produced all 100 predictions; M0 benchmark running |
| GaussianOcc comparison | DONE | 100 real predictions; 0.102 mIoU, ~2596 ms mean; streamed aggregation checked against official calculation |
| Notebook integration | IN_PROGRESS | Simple loop and camera/occupancy examples implemented; execute after all runs |
| Positive behavior test | IN_PROGRESS | M1 and GaussianOcc completed same 100 frames; M0 corrected to official M0 config |
| Negative behavior test | DONE | Invalid counts, wrong dataset/order, bad class IDs and missing predictions checked |
| Performance test | IN_PROGRESS | GaussianOcc batch complete; final sequential Flash runs pending |
| Idempotency/concurrency test | TODO | Verified cache and atomic outputs; sequential GPU runs |

The local GPU is an RTX 2070 with 8 GiB VRAM. First M1 frame scored 0.332 mIoU; this is a smoke test, not the final benchmark. Existing notebook packages remain isolated from research dependencies. No host environment variables were changed.

Compatibility: the published PyTorch container uses Python 3.7. One FlashOCC starred tuple return was parenthesized (no computation change). Headless OpenCV replaces the desktop-dependent build. New notebook helpers follow the existing readable Python style and the user's explicit simplicity priority instead of fragmenting operations into 250-character functions.

GaussianOcc's six simultaneous feature volumes filled VRAM. The adapter now streams the identical masked-mean calculation one camera at a time; a reduced-grid comparison on real features passes at 1e-5 tolerance. Unused image rendering is excluded. The optional PyTorch3D pose-refinement import is deferred; occupancy does not use that feature. M0 initially failed strict checkpoint loading when using the README-linked M1 config; resolved by selecting the repository's dedicated `flashocc-r50-M0.py`, including its narrower neck and coarser depth bins.
