# Use the local Model Lab

The Model Lab helps you try the project's two compute options and understand their processing speed. Start with the included sample; no phone, camera permission or dataset is needed. This is a research tool, not a driving-warning app.

## Open the app

On this already-configured PC, open PowerShell in the project folder and run:

```powershell
./scripts/start_app.ps1
```

Open **http://127.0.0.1:8501** in your browser. Keep the PowerShell window open. Press **Ctrl+C** in it to stop the app. The address is local to this PC; typing it on your phone will not connect to this computer.

On a fresh installation, run the base setup and install the UI dependencies first:

```powershell
./scripts/setup.ps1
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv pip install --python .venv/Scripts/python.exe -r requirements-ui.txt
```

Prepare the two model files using the commands in [Deployment modes](deployment-modes.md). This PC already has both. Initial installation/model acquisition requires internet; sample inference uses local files. The app does not download models during a test.

## Run your first comparison

1. Leave **Compare both** selected.
2. Leave **Use the included street sample** selected. It contains a bus and people.
3. Leave the test length at **100**. This repeats the same image after warmup.
4. Click **Run local test** and wait for completion. The two models run sequentially.
5. Read the table, then use **Download this result (JSON)** to keep the numerical results.

To try a photo, choose **Choose my photo**, select a JPEG or PNG, then run. Limit: 10 MB and 12 million pixels. The form applies your choices when submitted. Changing controls does not change the last completed result, which retains its mode and prediction count. A new submission clears the old result so failures cannot look like fresh success.

## Which mode should I choose?

| Choice | What it does today | What it represents |
| --- | --- | --- |
| Phone candidate | Runs the small ONNX model on this PC's CPU | A portable starting point for on-phone inference |
| Backend | Runs the larger PyTorch model on this PC's NVIDIA GPU | The compute portion of a future server |
| Compare both | Runs both on the same image | A first look at the two profiles |

**The phone candidate is not running on your handset.** A native app and device tests are still required. Backend timing excludes transferring video to a server and sending results back. Neither option starts a camera, a video stream or a remote service.

## Interpret the results

| Field | Plain meaning | How to use it |
| --- | --- | --- |
| Mean (ms) | Average time for one prediction | Lower means faster processing on this computer |
| p95 (ms) | About 95% of predictions finished within this time | Look for slow-frame behavior; this is not the worst case |
| Predictions/sec | 1,000 divided by mean milliseconds | Processing rate only, not finished-app video FPS |

Example: a 25 ms mean corresponds to about 40 predictions/sec. A p95 of 40 ms means roughly 5% took longer than 40 ms. Warmup is excluded. Results include model preprocessing and postprocessing, but exclude camera capture, network transmission and rendering. Background applications can affect timing.

**Speed does not tell us accuracy.** Faster processing can still miss a pedestrian. Accuracy needs labeled images and measurements of correct detections, missed objects and false alarms. This interface currently displays the input and timing results, not annotated predictions or an accuracy score. Existing annotated sample previews are in the earlier deployment report folder.

A box confidence such as 0.90 is the detector's score; it does not mean the system is 90% safe or correct. Distance and time to collision are not calculated here. The two profiles use different model sizes, runtimes and heads, so this is not a controlled CPU-versus-GPU comparison.

## Files and privacy

Submitted images are saved as uniquely named PNGs under `data/local-inputs/`. Results are saved under `reports/deployment/<run-id>/`. The app shows the exact result directory. JSON includes local paths, model hashes, runtime versions and measurements; inspect it before sharing. Downloading JSON does not include the photo itself.

The app has no image-upload integration with an external service. This workspace is inside OneDrive, so OneDrive may independently sync saved images and reports. Images are retained until you delete them; there is no automatic retention cleanup. Do not use sensitive photos if you do not want them stored here.

## If something goes wrong

- **Another test is running:** wait for it to finish. An exclusive lock prevents competing GPU benchmarks. Do not run multiple tabs or the command-line benchmark together.
- **Missing model or package:** follow the setup and export commands in Deployment modes. Phone mode needs `yolo26n.onnx`; backend needs `yolo26s.pt` and CUDA.
- **GPU unavailable:** try Phone candidate for the CPU test. Backend does not silently switch to CPU.
- **Invalid photo:** choose a smaller, valid JPEG/PNG. Renaming a file extension does not convert an image.
- **Timeout:** wait for active model processing to stop before retrying. Individual profiles have a 10-minute timeout; the UI has an overall limit slightly longer than two profiles.
- **Port 8501 already in use:** reuse the existing app tab or stop the previous instance.

For implementation details and the future phone/backend flows, see [Architecture](architecture.md).
