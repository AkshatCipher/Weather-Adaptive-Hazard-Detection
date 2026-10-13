# Confidence-Aware Weather-Adaptive Hazard Detection for Monocular Driving Video

A perception pipeline for dashcam / webcam video that answers two questions **separately** for every frame:

* **Trust**: how far can the detector's output be believed right now? (predicted at runtime, without ground truth)
* **Danger**: how hazardous is the scene, given what was detected?

The two are kept apart because their most dangerous combination is invisible to a single fused score: an
apparently empty road in dense fog is *low apparent danger with low trust*. The pipeline also recognises the
weather (clear, fog, night, rain, snow) and the fog density, adapts image enhancement to it, and is evaluated
across weather conditions with the datasets' own ground-truth weather labels.

Final-year B.Tech project (Robotic Perception / Computer Vision), single member. Runs on one laptop
(RTX 5060 8 GB). All heavy models are frozen; the only trained parts are the weather classifier and a small
gradient-boosted trust model.

## What is in the repository

| Folder | Content |
|---|---|
| `src_core/` | **the pipeline and the application** (video and webcam modes), experiment drivers, tests' subject |
| `src_v4/` | the final weather classifier v4 (training, evaluation, CLIP comparison) |
| `src_v3/`, `src_cityscapes/`, `src/` | earlier weather-model versions, kept frozen as documented fallbacks |
| `tests/` | pytest suite (41 tests) |
| `FINAL_PROJECT_LOG.md` | every issue, mitigation, statistic, training set and test result in one place |
| `docs/` | the presentation (`Weather_Adaptive_Hazard_Detection.pptx`, 21 slides) and the project report (`Weather_Adaptive_Hazard_Detection_Report.docx` and `.pdf`) |
| `final_release/`, `make_release.py` | the clean copy of the project for GitHub and the script that assembles it (`python make_release.py`) |
| `results/` | all results, metrics, figures, checkpoints (large files are git-ignored) |
| `weights/` | YOLO weights (downloaded on first use) |
| `PROJECT_GUIDE.md` | the design, every decision and why, remaining work, viva answers |
| `PROJECT_LOG_REPORT.md` | history, issues and how they were solved |
| `MODEL_DETAILS_v3_v4.txt` | full technical record of the weather models |
| `src_core/README.md` | the core package in detail |

## Pipeline

```
frame
  -> weather + fog level        v4 ensemble (DINOv2 + ConvNeXt), 99.7% on ACDC, ~94% on a camera never seen
  -> adaptive enhancement       luminance-only recipes per weather, switchable ON/OFF (a tested hypothesis)
  -> object detection           frozen YOLO (v8m / 11m / 26m as a flag; the choice is made by measurement)
  -> flip-consistency           detector stability under mirroring (replaces tracking on single frames)
  -> trust                      XGBoost over 5 feature families, trained on clear-day frames ONLY
  -> danger                     weighted hazard formula on the detections; R_true from annotations in experiments
  -> quadrant                   trust x danger readout; warning in the low-trust / low-apparent-danger corner
```

Image-quality features for trust are always computed on the **raw** frame, before enhancement.

## Prerequisites

* Windows 11 (tested), NVIDIA GPU with CUDA 12.8-capable driver (RTX 5060, driver 592 tested). CPU works, slowly.
* Anaconda or Miniconda. Python 3.12.
* About 15 GB of free disk without the datasets; the datasets (see below) take about 60 GB.

## Installation

```powershell
conda create -y -n weather -c conda-forge --override-channels python=3.12
conda activate weather
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print(torch.cuda.get_device_name(0), (x@x).sum().item())"
```

The last line must print the GPU name and a number. `requirements.txt` pins the exact versions this project was
built and tested with (PyTorch 2.11.0 + CUDA 12.8, Ultralytics 8.4.155, XGBoost 3.4.1, OpenCV 5.0).

Environment notes that matter on this laptop: never spawn worker processes from a script that imports PyTorch
(Windows commit memory, about 21 GB, runs out); all scripts use threads or cap loader processes from free memory.
Set `HF_HUB_DISABLE_XET=1` if a Hugging Face download stalls (the scripts set it themselves).

## Running the application

**Desktop interface (recommended)**

```powershell
python src_core\gui.py
```

A window with three sections: source (webcam or video file), pipeline options (detector, enhancement, DINOv2 feature,
confidence, how often the weather model and the mirror pass run, input size, device, fast and accurate presets) and
outputs (annotated mp4, per-frame CSV, JSONL). It shows the exact command it will run, streams the log, and stops a run
cleanly so the output video is finalised.

**Command line**

Two modes. Both need the weather checkpoints in `results/weather_v4/` and download the YOLO weights (about 50 MB)
on first use. The trust model (`results/core/trust/trust_model.json`) is produced by the experiments; without it
the app runs and shows trust as n/a.

**1. A video file on your PC**

```powershell
python src_core\app.py video --input "C:\path\to\drive.mp4" --output results\core\drive_out.mp4 --csv results\core\drive.csv --show
```

**2. The laptop webcam / a USB camera, in real time**

```powershell
python src_core\app.py camera --output results\core\cam.mp4 --csv results\core\cam.csv
```

Press `q` or `Esc` in the window to stop. `--device-index 1` selects another camera. Speed flags:
`--weather-every 5 --flip-every 2 --no-dino` (defaults chosen for about 10 fps at 720p on the RTX 5060);
`--no-enhance` turns enhancement off; `--detector yolo11m` swaps the detector.

Outputs: an annotated mp4, a per-frame CSV (weather, fog level, probabilities, detections count, trust, danger,
quadrant, per-stage milliseconds) and optionally a JSONL with every detection (`--jsonl`).

## Datasets (for the experiments only; the app needs none of them)

All extracted directly under the project root. See `PROJECT_GUIDE.md` section 5 for layout, quirks and roles.
BDD100K (images + the two label JSONs), ACDC (images + detection package), Cityscapes (clean images, Foggy
Cityscapes-DBF, gtFine), DAWN. RainCityscapes is on disk but used only by the weather classifier.

## Running the experiments

```powershell
python src_core\run_experiments.py            # strata -> 3 detector sweeps -> choice -> synthetic + enhancement -> trust -> analysis
```

About 1 to 2 hours on the RTX 5060. Every step is resumable and documented in `src_core/README.md`. Results and
figures land in `results/core/`. The weather-model experiments are in `src_v4/README.md`.

## Measured results (13 October 2026, full tables in `results/core/RESULTS.md`)

* Detector chosen by adverse-weather pedestrian recall on identical frames: YOLO26m (0.443) over YOLOv8m (0.413) and YOLO11m (0.409).
* Trust model, trained on clear day only, beats the confidence-only baseline in all 21 held-out strata: AUROC 0.73-0.76 on the
  same camera in unseen weather, 0.64-0.66 on the ACDC camera, 0.73-0.74 on Cityscapes fog. Acting on the trusted half of
  clear-day frames cuts the failure rate from 36.8% to 19.7%.
* Danger gap is positive in every stratum and largest at night (BDD100K 0.09, ACDC 0.10; 36-46% of night frames underestimated
  by more than 0.1); it grows monotonically with fog density on identical Cityscapes scenes. Weight sensitivity: Spearman >= 0.954.
* Enhancement: +5 to +7 points of pedestrian recall at night, +4 in dense fog, neutral in rain and snow, no confidence inflation.
* Runtime at 720p: 20.6 fps with everything on (p95 71 ms), 34.8 fps in fast mode; GPU peak 439 MB; no memory growth.

## Tests

```powershell
python -m pytest tests -q                 # 39 fast tests (matching, failure labels, enhancement, features, danger, trust, loaders, app modes with a fake pipeline, GUI commands)
python -m pytest tests -q --run-slow      # + real-model integration test
python src_core\bench.py --frames 300     # latency / throughput / memory on this machine -> results/core/benchmark.json
```

## Health checks and troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA is not available` at start | Wrong torch build. Reinstall from the cu128 index (see Installation). |
| `WinError 1455` / `OpenBLAS ... memory allocation failed` | Windows commit memory exhausted. Close other programs; pass `--workers 0` to the training scripts. |
| `cannot open camera 0` | Another app holds the webcam, or Windows camera privacy settings block desktop apps. Try `--device-index 1`. |
| Webcam shows `fog` indoors | The model was trained on driving footage only; indoor scenes are out of its domain. Expected. |
| Downloads stall | Hugging Face Xet backend; `set HF_HUB_DISABLE_XET=1`. GitHub (YOLO weights) is simply slow on this link. |
| `trust: n/a` | No trust model yet. Run the experiments (or `src_core\trust.py train`). |
| Video writer error | Output folder not writable or codec missing; the app uses `mp4v`. |

## Security and privacy

The application is a local, offline program: it reads a video file or the local camera and writes files you name.
There is no network service, no account, no database and no stored credentials. The only network use is the
one-time download of public model weights. Camera frames are never written unless you pass `--output`/`--jsonl`.
Dataset images under the project root are never modified.

## Known limitations

* Rain severity is not available on real rain (reported as `unrated`); fog level is.
* The weather model is strong on driving footage (99.7% ACDC, 96.4% BDD100K) and weaker on arbitrary web images
  (74% on DAWN).
* The detector is frozen; preprocessing can only recover part of the weather-induced loss.
* Experiments mix camera shift and weather shift in the ACDC columns; the Cityscapes clear-vs-fog strata isolate
  the weather effect.
* Single images only: no temporal evidence beyond the per-frame flip-consistency.

See `PROJECT_GUIDE.md` sections 8 and 13 for the methods and the full list.
