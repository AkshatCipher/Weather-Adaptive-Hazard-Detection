# Final Project Log

**Confidence-Aware Weather-Adaptive Hazard Detection for Autonomous Driving Using Monocular Video**
Single-member final-year B.Tech project (Robotic Perception / Computer Vision). Period: 15 September to
13 October 2026. Machine: ASUS TUF F16, i7-14650HX, 15.6 GB RAM, RTX 5060 Laptop 8 GB, Windows 11.
This file consolidates every issue met, how it was handled, the statistics applied, the training data and the
test results. The narrative history is in `PROJECT_LOG_REPORT.md`; the design is in `PROJECT_GUIDE.md`;
all measured core results are in `results/core/RESULTS.md`; the weather models in `MODEL_DETAILS_v3_v4.txt`.

---

## 1. Issues faced and mitigations

| # | Area | Issue | Mitigation | Outcome |
|---|---|---|---|---|
| 1 | Scope | The original 9-module concept (weather, enhancement, detection, tracking, depth, confidence, near-miss, risk fusion, decisions) was a research programme, not a 4-day project; time-to-collision needed metric depth that one camera cannot give; no dataset had near-miss labels | Cut to two evaluable scores (Trust, Danger); near-miss, TTC, depth-as-module, risk fusion and detector training excluded in writing | Every remaining component has a metric |
| 2 | Environment | Windows Smart App Control blocked PyTorch's unsigned DLLs (WinError 4551); only Python 3.14 with a CPU-only torch; no WSL | User turned the setting off; conda env `weather` (Python 3.12, PyTorch 2.11.0 + CUDA 12.8) verified with a real GPU matrix multiply, not just `is_available` | sm_120 kernels run |
| 3 | Data | BDD100K 10k mirror had no labels; official site returns 403 without an account; ETH mirror does not resolve in DNS here | Public Kaggle mirror of the two official label JSONs; later the full 100k images (Kaggle) | 80,000 labels on disk |
| 4 | Data | Only 3,426 of the 10k subset images had labels (it is the segmentation split) | Full BDD100K images downloaded; 54,822 labelled training images for the weather model, 8,193 held-out | Fair held-out BDD100K test |
| 5 | Data | ACDC shipped as images only; its rain training label file is a mis-packaged copy of the combined file | COCO detection package downloaded; loader filters by `file_name` prefix | 500 labelled frames per condition |
| 6 | Data | Cityscapes was present only as the foggy and rainy variants, with no clean frames and no gtFine | gtFine downloaded, then the clean 11 GB package | Paired clear + 3 fog levels with boxes on identical scenes |
| 7 | Memory | Cache building with a process pool exhausted 16 GB: on Windows every spawned worker re-imports PyTorch | Thread pool for image decoding (OpenCV releases the interpreter lock) | No crash |
| 8 | Memory | Training succeeded, then the automatic evaluation crashed and hung the terminal: 27 copies of PyTorch across training and evaluation loader processes | Training loaders shut down before evaluation; evaluation workers capped; maths libraries forced to one thread; later all loader counts derived from free RAM and free Windows commit (about 21 GB limit on this machine) | No further memory crashes |
| 9 | Network | Hugging Face downloads stalled at 67 MB through the Xet transfer backend | Plain HTTP (`HF_HUB_DISABLE_XET=1`), set by the scripts themselves | 605 MB CLIP weights and the depth network downloaded |
| 10 | Correctness | CLIP's OpenAI weights were loaded into a model with the wrong activation (QuickGELU mismatch) | Default model set to the QuickGELU variant; mismatched metrics kept for comparison | ACDC 96.0% -> 96.6% |
| 11 | Correctness | ACDC's official splits share GoPro recordings across train, val and test (28 of 60 recording ids) | Guard changed from "recordings disjoint" to "no identical recording-and-frame pair in two splits"; overlap reported | Honest guard |
| 12 | Correctness | A scene key without the label collapsed ACDC clear reference frames onto their adverse twins | Label added to the scene key | Correct per-scene sampling |
| 13 | Method | v2 scored 100% on every Cityscapes fog and rain row: confidence flat across fog levels, higher than on real fog | Diagnosed as a camera shortcut; clean Cityscapes frames downloaded; proven on 60 clear frames (59 called fog); retrained with clean frames as `clear` | 99.8% of clear frames correct; shortcut gone |
| 14 | Method | Severity labels existed for one camera only | Fog and rain rendered onto clear frames of all cameras with the atmospheric scattering model and Depth Anything V2 depth; renderer calibrated against real Foggy Cityscapes frames (pixel error 10.3 vs 36.6 doing nothing) | Fog level 97.85% on real synthetic-fog Cityscapes, unseen cities |
| 15 | Method | Rain-severity head answers "none" on 99% of real rain (learned from synthetic streaks only) | Reported as `unrated`; never invents a level; documented as unfixable without real labels | Honest output |
| 16 | Data | BDD100K training images nested in `trainA/trainB/testA/testB`; the first scan found 1,160 of 70,000 | Recursive scan | All 70,000 indexed |
| 17 | Process | Upgraded v3 scripts in place; no version control | New code moved to `src_v4`; `src_v3` rebuilt and verified by reproducing every logged v3 number; freeze guard added | v3 preserved as a fallback |
| 18 | Correctness | Trust evaluation scored trust in the wrong direction (AUROC 0.12 for a good model) | Caught by a unit test; failure score = 1 - trust | Correct AUROC |
| 19 | Dependency | `imagecorruptions` 1.1.2 uses `np.float_`, removed in NumPy 2 | Alias shimmed before import | Severity sweep runs |
| 20 | Dependency | Ultralytics 8.4 deprecates the `half` flag | Removed; warm-up at the real input size | Clean logs |
| 21 | Robustness | A camera that changes resolution mid-stream would make the video writer drop frames silently | Frames resized to the writer's size | Verified by test |
| 22 | Process | The experiment chain was interrupted by a session end | Chain is resumable (skips finished steps); resumed without loss | Complete results |
| 23 | Interface | Killing the app subprocess would leave the output mp4 unfinalised | Graceful stop file checked every frame; GUI falls back to terminate after 8 s | Clean outputs |

## 2. Statistics and methods applied

| Question | Method | Statistic |
|---|---|---|
| Weather classification quality | Confusion matrices on held-out splits; macro-F1 for model selection | accuracy, macro-F1, per-class recall |
| Does the weather model generalise beyond its cameras? | Leave-one-camera-out: train on two cameras, test on the third, for every camera and recipe | accuracy on a camera never seen (about 94%) vs 99.7% in-domain |
| Is a perfect score a shortcut? | Confidence across severity levels; counter-case test on clear frames of the same camera | share of clear frames called fog (3.4% -> 99.8% after the fix) |
| Fog level quality | Exact and within-one-level accuracy on real Foggy Cityscapes (unseen cities) and on rendered fog | 97.85% exact, 99.9% within one level |
| Renderer fidelity | Mean absolute pixel error and signed bias against the real foggy frame of the same scene, at three betas | error 7.1 / 10.1 / 12.6 vs 21.4 / 36.5 / 58.5 doing nothing |
| Detector choice | Same frames, three detectors; adverse-weather pedestrian recall with precision within 5 points of the best | YOLO26m 0.443 vs 0.413 / 0.409 |
| Detection quality per stratum | VOC-style AP50 per class, mAP50; precision and recall at the 0.25 operating point; boxes under 12 px ignored | see `results/core/RESULTS.md` section 2 |
| Failure label | Frame F1 < 0.5 or any missed pedestrian (greedy one-to-one IoU 0.5, class-aware matching) | per-stratum failure rates 0.15 to 0.74 |
| Can failure be predicted? | XGBoost (400 trees, depth 4, class-weighted) on 30 meta-features, trained on 3,000 clear-day frames only | AUROC per held-out stratum; mean-confidence baseline |
| Operational value of trust | Risk-coverage: residual failure rate when acting on the most-trusted fraction | fail@100 / fail@70 / fail@50 |
| Which features matter? | Leave-one-family-out ablation; XGBoost gain importance | pooled AUROC 0.739; confidence family most important |
| Is the danger score valid? | Same formula on detections (R_est) and annotations (R_true); gap per stratum | mean gap, share of frames underestimated by > 0.1 |
| Are the danger weights arbitrary? | Each class weight +-50% on 2,000 frames | Spearman rank correlation >= 0.954; tercile agreement >= 0.875 |
| Does enhancement help? | ON vs OFF on identical frames; confidence of true vs false positives to detect miscalibration | delta mAP50, delta pedestrian recall, delta confidence of true positives |
| Severity effect | Synthetic corruptions (fog, frost, motion blur) at severity 1, 3, 5 and real Cityscapes fog at 3 betas | failure rate, danger gap and trust AUROC vs severity |
| Runtime | 300-frame sustained run at 720p | sustained fps, latency p50 / p95 / p99, GPU peak, RSS growth |

## 3. Training data

### Weather classifier v4 (final)

| Source | Train | Val | Test | Role |
|---|---|---|---|---|
| ACDC (fog, night, rain, snow + clear references) | 3,200 | 812 | 4,000 | official splits as shipped |
| BDD100K (full 100k images, 80,000 labels) | 54,822 | 2,848 | 8,193 | official val labels held out; 5% of training videos as validation |
| Cityscapes clean | 2,724 | 251 | 500 | split by city: test = frankfurt, lindau, munster |
| Foggy Cityscapes-DBF (3 betas) | 8,172 | 753 | 1,500 | same city split; fog-level labels |
| RainCityscapes (3 intensities) | 7,272 | 2,160 | 1,188 | same city split; rain-level labels; classifier only |
| Fog / rain rendered on clear frames of all cameras | 30,562 rows | | | severity labels on every camera |
| DAWN (704 fog / rain / snow; 323 sandstorm) | 0 | 0 | 704 (+323 probe) | external test, unknown cameras |

Two models, dinov2_last2 (DINOv2 ViT-S/14, last 2 blocks tuned, 3.95 M trainable) and convnext_full
(ConvNeXt-Tiny, 28.22 M), 30 epochs x 24,000 draws each, 60 and 68 minutes; ensemble with mirrored-image
averaging. Earlier versions: v1 (ACDC + BDD subset), v2 (+ Cityscapes), v3 (3 heads, rendered weather, before the
full BDD100K). CLIP ViT-B-32 zero-shot as the no-training baseline.

### Trust model

XGBoost trained on 3,000 BDD100K clear-day frames only (failure rate 0.355), detector YOLO26m, 30 features in
five families. Tested on 21 strata it never saw: 12,024 real frames (BDD100K val clear / night / rain / snow,
ACDC fog / night / rain / snow, Cityscapes clear and three fog levels on identical scenes) and 1,800 synthetic
frames (fog, frost, motion blur at severity 1, 3, 5).

## 4. Test results

### Automated tests (`python -m pytest tests -q --run-slow`): 41 passed, 0 failed

| File | What is verified |
|---|---|
| `test_common.py` (6) | IoU matrix values; one-to-one class-aware matching; failure labels incl. threshold and min-size rules; AP50; record JSON round trip |
| `test_processing.py` (14) | clear is untouched; every recipe keeps shape/dtype and changes pixels; night never darkens; dehazing restores contrast; quality features track blur and haze; confidence, flip and weather features; kNN distances; danger bounds, monotonicity, zero gap when detections equal ground truth, quadrant, weight sensitivity |
| `test_trust_and_loaders.py` (6) | Cityscapes masks to boxes; BDD100K condition mapping; ACDC rain-file quirk on the real file; XGBoost learns, transfers and round-trips; AUROC and risk-coverage edge cases; the train CLI writes every output |
| `test_app.py` (6) | overlay drawing; CSV columns; video mode end to end with a fake pipeline; missing input rejected; camera mode headless with a fake capture; real-model pipeline on real frames (slow) |
| `test_gui.py` (4) | command building for both modes and all flags; graceful stop file; the window constructs (slow) |

### Weather model (identical real test images; full tables in `MODEL_DETAILS_v3_v4.txt`)

| | v2 | v3 | v4 final | CLIP zero-shot |
|---|---|---|---|---|
| ACDC test accuracy | 98.95% | 99.48% | 99.72% | 96.6% |
| BDD100K test, 7,875 fair images | 94.5% | 95.4% | 96.4% | 89.0% |
| DAWN, unknown cameras | 69.2% | 74.0% | 74.3% | 82.7% |
| Fog level, real Foggy Cityscapes, unseen cities | n/a | 95.0% | 97.85% | n/a |
| Accuracy on a camera never trained on (LODO) | n/a | 90.4% | 93.9% | n/a |

### Core pipeline (full tables in `results/core/RESULTS.md`)

| Result | Value |
|---|---|
| Detector chosen | YOLO26m: adverse pedestrian recall 0.443 vs YOLOv8m 0.413, YOLO11m 0.409 |
| Trust AUROC, same camera, unseen weather | 0.73 to 0.76 (confidence-only baseline 0.59 to 0.65) |
| Trust AUROC, ACDC camera | 0.64 to 0.66 (baseline 0.54 to 0.61) |
| Trust AUROC, Cityscapes fog | 0.73 to 0.74 (baseline 0.65 to 0.68) |
| Risk-coverage, clear day | failure rate 36.8% -> 19.7% acting on the trusted half |
| Danger gap | positive in all 21 strata; night 0.09 to 0.10 with 36 to 46% of frames underestimated by > 0.1; rises with fog density and synthetic severity |
| Weight sensitivity | Spearman >= 0.954, tercile agreement >= 0.875 |
| Enhancement | pedestrian recall +6.5 (BDD night), +5.4 (ACDC night), +4.2 (dense fog); rain and snow within noise; true-positive confidence changes <= +0.012 |
| Runtime, 720p | 20.6 fps default (p95 71 ms), 34.8 fps fast; GPU peak 439 MB; no memory growth over 300 frames |
| Video mode on a 120-frame mixed clip | all 120 frames' weather correct; 9.9 fps with encoding |
| Camera mode | 8.0 fps at 1280x720 with a live window |

## 5. What was not done

* Synthetic strata were not re-run with enhancement ON.
* The trust model keeps the weather-probability features although the ablation shows they add nothing.
* No multi-seed repetition; differences under about 0.01 are noise.
* Tested on Windows only.

## 6. Final deliverables (13 October 2026)

| Deliverable | Where | Status |
|---|---|---|
| Desktop interface (webcam or video file; detector swap, enhancement and DINOv2 toggles, stride, confidence, presets, outputs) | `src_core/gui.py`, run with `python src_core\gui.py` | done, 4 tests |
| Command-line application, two modes | `src_core/app.py video ...`, `src_core/app.py camera ...` | done, verified on screen |
| Presentation, 21 slides, measured numbers and native charts | `docs/Weather_Adaptive_Hazard_Detection.pptx` | done, validated and rendered |
| Report with the required headings (Title Page, Abstract, TOC, List of Figures and Tables, Introduction, Literature Survey, Methodology, Result and Discussion, Conclusion, References) | `docs/Weather_Adaptive_Hazard_Detection_Report.docx` and `.pdf`, 23 pages | done |
| Rebuild scripts for both documents | `docs/build/` | done |
| Clean copy for GitHub (files over 95 MB excluded and documented) | `final_release/`, assembled by `python make_release.py` | done, 99 files, 127 MB |
| Full test suite | `python -m pytest tests -q --run-slow` | 41 passed |
