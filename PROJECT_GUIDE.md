# Project Guide

**Confidence-Aware Weather-Adaptive Hazard Detection for Autonomous Driving Using Monocular Video**
*Separating Perceptual Trust from Scene Danger Under Adverse Weather*

Updated Saturday 19 September 2026, 23:00. This guide replaces the 4-day plan in `PROJECT_HANDOFF.md`.
The scope and the scientific design in that handoff still stand, except where section 4 lists a change.
Companion documents: `PROJECT_LOG_REPORT.md` (history, issues, fixes) and `MODEL_DETAILS_v3_v4.txt`
(full technical record of the weather models).

---

## 1. The project in one paragraph

A monocular driving-image perception pipeline that outputs two separate scores per frame. **Trust** asks how far
the detector's output can be believed. **Danger** asks how hazardous the scene is. The two are kept apart because
their most dangerous combination is invisible to a single fused number: an apparently empty road in dense fog
looks safe exactly because perception may be failing. Everything is evaluated across weather conditions. All
heavy models are frozen. The only trained parts are the weather classifier, now finished, and a small
gradient-boosted trust model.

**Research question.** Can a lightweight introspection layer, trained only on clear-weather data, predict
per-frame detector failure across unseen adverse weather? And does the perception system systematically
underestimate scene danger in adverse weather, measured by the gap between danger computed on detector outputs
and danger computed on ground-truth annotations?

**Constraints.** Single member. Publication not required. One laptop, RTX 5060 8 GB. Whole project due
Monday 21 September 2026.

---

## 2. The pipeline

```
image
  -> [A] weather recognition ........ v4: weather + fog level          DONE
  -> [B] adaptive enhancement ....... switchable ON / OFF, per weather  TO DO
  -> [C] object detection ........... frozen YOLO, chosen by test       TO DO
  -> [D] stability check ............ flip-consistency (replaces tracking) TO DO
  -> +-- [E] Trust score  -----+ .... parallel branches, never serial   TO DO
     +-- [F] Danger score -----+
  -> [G] analysis ................... trust x danger quadrant, tables   TO DO

Raw-frame bypass: image-quality features for Trust are computed on the RAW frame,
before enhancement, because enhancement inflates contrast and sharpness.
```

Rules that never change:
1. Every result table is stratified by the **datasets' own weather labels**, never by the classifier's output.
2. Every experiment runs with enhancement **ON and OFF**.
3. Trust and Danger are **parallel**. Danger does not consume Trust.
4. The detector is **never trained**.

---

## 3. What is excluded, and why

| Excluded | Reason |
|---|---|
| Near-miss prediction | No dataset in use has near-miss labels. It is a separate research field. |
| Time-to-collision | Needs metric depth. Monocular depth has no metric scale. |
| Depth estimation as a pipeline module | Its only consumer was time-to-collision. |
| Risk fusion and decision generation | Nothing to validate them against. Absorbed into Trust, Danger and the quadrant. |
| Any training of the detector | Keeps the study about perception reliability, not detector tuning. |
| KITTI, Waymo, nuScenes, CARLA, accident datasets | No weather labels, or hundreds of GB for no extra experiment. |
| A sandstorm weather class | Its only source is 323 web images. It would teach the dataset's look and would destroy the only unseen-camera test. |

---

## 4. What changed since the handoff, and why

| Topic | Handoff | Now | Why |
|---|---|---|---|
| Weather classifier | CLIP zero-shot first, MobileNetV3 only if time remained | v4: two-model ensemble, weather plus fog level | Requested by the user. It also produced a self-contained domain-shift study. |
| Runtime weather source | CLIP | v4 | v4 beats CLIP by 3 to 7 points on driving footage and gives a fog level. CLIP stays as the baseline and for non-driving imagery. |
| Depth estimation | Excluded | Still excluded from the pipeline. Used **offline** to render fog onto clear frames when building training data for v3 and v4. | It is a data-preparation tool, not a pipeline module. Must be stated this way in the report. |
| RainCityscapes | Excluded | Still excluded from the main experiments. Used only to train the weather classifier's rain-level head. | The only source of rain-intensity labels. |
| Cityscapes | Clear-weather out-of-domain column, or fog if the foggy package existed | Clean, plus three fog levels, plus gtFine boxes, on identical scenes | The clean package was downloaded to prove the v2 shortcut. It also gives the project its best experiment, see section 8. |
| BDD100K | 10k validation subset as the whole labelled core | Full 100k images on disk, 80,000 labels | The subset was the segmentation split: only 3,426 of 10,000 had labels. |
| Fog in BDD100K | Expected 100 to 150 images | 48 in the whole training set | Fog evidence comes from ACDC and Cityscapes instead. |
| Detector | YOLOv8m | YOLOv8m, YOLO11m or YOLO26m, chosen by test | Review I says "a frozen YOLO-family detector". The choice will be measured, not assumed. |
| Enhancement | CLAHE only | A recipe per weather class, keyed on v4's fog level | v4 made severity-aware enhancement possible. Still an ON and OFF hypothesis. |
| DAWN | Not used | External test for the weather classifier, and a possible extra detection stratum | 704 real fog, rain and snow images with vehicle boxes from unknown cameras. |
| Timeline | Friday trust model, Saturday enhancement test | Both days went into the weather component | See section 10. |

Decisions that did **not** change: two scores, the dual danger computation, enhancement as an ON and OFF arm, the
raw-frame bypass, ground-truth labels for stratification, flip-consistency in place of tracking, synthetic
corruptions as a severity sweep, image grids for the demo.

---

## 5. Data on disk

Root: `D:\Projects\Weather analysed object detection`

| Dataset | Folder | Images | Boxes | Role in the main experiments |
|---|---|---|---|---|
| BDD100K full | `bdd100k\bdd100k\bdd100k\images\100k` | 70,000 train, 10,000 val | yes, in both label JSONs | Clear-day training set for Trust. Same-camera night, rain and snow test strata. |
| BDD100K labels | `bdd100k-10k\labels\` | 2 JSON files, 1.45 GB and 208 MB | weather, time of day, scene, 2D boxes | |
| ACDC | `rgb_anon_trainvaltest_ACDC` | 4,006 adverse + 4,006 clear references | adverse frames only, train and val, `gt_detection_trainval` | Real fog, night, rain, snow. 500 labelled frames per condition. |
| Cityscapes clean | `leftImg8bit_trainvaltest` | 3,475 | gtFine instance masks | Clear twin of every fog scene |
| Foggy Cityscapes | `leftImg8bit_trainval_foggyDBF` | 10,425 = 3,475 x 3 fog levels | same gtFine masks | Controlled fog severity on identical scenes |
| gtFine | `gtFine_trainvaltest` | | instance masks, boxes derivable where id is 1000 or more | |
| DAWN | `DAWN` | 704 + 323 sandstorm | YOLO-format vehicle boxes | External, unknown cameras |
| RainCityscapes | `leftImg8bit_trainval_rain` | 10,620 | | Classifier only. Not in the main experiments. |

Data quirks that the next scripts must respect:
- ACDC `rain/instancesonly_rain_train_gt_detection.json` is a copy of the combined 1,600-image file. Filter by the
  `file_name` prefix `rain/`.
- ACDC has **no boxes for the clear reference frames**.
- BDD100K training images are nested in `trainA`, `trainB`, `testA`, `testB`. Scan recursively.
- ACDC's official splits share GoPro recordings across train, val and test, with different frames.
- BDD100K weather tags are noisy: wet roads tagged rainy, snow on the ground under a clear sky tagged snowy.
- Detector classes are COCO. Mapping: person and BDD rider to person; car, truck, bus as is; bicycle to bike;
  motorcycle to motor; traffic light as is. ACDC and Cityscapes use person, rider, car, truck, bus, train,
  motorcycle, bicycle.

### Strata available for the main experiments

| Stratum | Source | Labelled frames | Camera relative to Trust training |
|---|---|---|---|
| Clear day, Trust training set | BDD100K train | about 23,000, subsample as needed | same |
| Clear day, held out | BDD100K val | 3,441 | same |
| Night | BDD100K val | 3,929 | same |
| Rain | BDD100K val | 396 | same |
| Snow | BDD100K val | 422 | same |
| Synthetic fog, frost, blur at severity 1, 3, 5 | `imagecorruptions` on BDD100K clear | as many as generated | same |
| Clear, and fog at three levels, identical scenes | Cityscapes | 3,475 each | different |
| Fog, night, rain, snow | ACDC train + val | 500 each | different |
| Fog, rain, snow | DAWN | 300, 200, 204, vehicles only | different, optional |

**A caveat that must appear in the report.** The Trust model trains on the BDD100K camera. ACDC and Cityscapes are
other cameras, so weather shift and camera shift are mixed in those columns. Lead with the same-camera strata.
Cityscapes is the exception that rescues the design: its clear frames come from the same camera as its fog frames,
so the failure rate on clear Cityscapes is the camera-only baseline, and the rise from clear to each fog level is
the weather effect alone.

---

## 6. Status

### Done

| Item | Where | Key result |
|---|---|---|
| Environment | conda env `weather`, Python 3.12, PyTorch 2.11.0 + CUDA 12.8 | GPU kernels verified |
| Data acquisition and inspection | section 5 | All annotations present |
| Weather classifier v1 | `src`, `results\weather_classifier` | 98.75% ACDC, fails on an unseen camera |
| CLIP baseline | `src\clip_zeroshot_weather.py` | 96.6% ACDC |
| v2 and the shortcut study | `src_cityscapes` | 3.4% then 99.8% on clear Cityscapes |
| v3, frozen fallback | `src_v3`, `results\weather_v3` | 99.48% ACDC, fog level 95.0% |
| **v4, final** | `src_v4`, `results\weather_v4` | **99.72% ACDC, 96.4% BDD100K, fog level 97.85%, about 94% on an unseen camera** |
| CLIP against v4 | `results\weather_v4\comparison_with_clip.csv` | v4 on driving footage, CLIP on web images |
| Enhancement recipes and detector-choice rule | section 7 | Designed, not built |
| Documents | `MODEL_DETAILS_v3_v4.txt`, `PROJECT_LOG_REPORT.md`, this guide | |

### The graded core: DONE on 12-13 October 2026 (`src_core/`, `results/core/`)

Record schema, ground-truth loaders, three-detector sweep (YOLO26m chosen), failure labels, the five trust-feature
families, synthetic corruptions, the clear-only trust model, enhancement ON/OFF, danger gap, weight sensitivity,
quadrant, figures, benchmark, and the runtime app with video-file and webcam modes. Every measured number is in
`results/core/RESULTS.md`. Steps 1 to 11 of section 7 are complete; step 12 (report, paper) remains.

Also open: Review I section 5.4 claims progress that was false when written. Those three items are now true
(environment verified, labels parsed and tabulated, a detector smoke test is part of step 3 below). Delete the
bracketed note when the document is next touched.

---

## 7. What remains, step by step

Each step names its output and its check. Build the measuring code before the model code.

**Step 1. Per-frame record and ground-truth loaders.**
One JSON line per frame and enhancement condition, with the handoff's schema: `frame_id, source, split,
weather_gt, timeofday_gt, corruption, enhancement, detections, gt_boxes, consistency, meta, scores,
weather_pred`. Add `fog_level_pred` and `detector`. Three loaders into one common box format: BDD100K JSON,
ACDC COCO JSON with the rain-file filter, Cityscapes instance masks to boxes.
*Check:* draw ground-truth boxes on 5 frames per source and look at them.

**Step 2. Frame lists per stratum.** Fixed, seeded, saved to CSV, so every later run sees identical frames.
*Check:* counts match section 5.

**Step 3. Detection sweep with the model as a flag.** YOLOv8m, YOLO11m, YOLO26m, enhancement OFF, one fixed
confidence threshold per model, all strata. Write records.
*Check:* mAP50 on BDD100K clear day is in a sane range before anything else is believed.

**Step 4. Choose the detector.** Per stratum: mAP50, per-class recall, pedestrian and rider recall. Pick the best
adverse-weather pedestrian and rider recall at equal precision. On a tie, the smallest drop from clear to adverse.
Then YOLO26 because it has no suppression threshold. Everything after this step uses the winner only.

**Step 5. Failure labels.** Per frame, from ground truth: frame F1 below 0.5, or any missed pedestrian. State
which one is primary. Missed pedestrian is the more safety-relevant.
*Check:* failure rate rises from clear to adverse. If it does not, stop and find out why.

**Step 6. Trust features, four families.**
(a) confidence statistics: mean, max, entropy, spread, count.
(b) flip-consistency: detector on the mirrored frame, fraction of boxes matched by IoU.
(c) feature-space distance: DINOv2-small embedding, k-nearest-neighbour distance to clear-day training frames.
(d) raw-frame image quality: Laplacian variance, RMS contrast, mean brightness, and a dark-channel haze
    estimate, all computed **before** enhancement.
*Check on (d):* on the 3,475 Cityscapes scenes the haze estimate must rise from clear to light to medium to
dense. That validates it as the severity signal and is a figure for the report.

**Step 7. Synthetic corruption set.** `imagecorruptions` fog, frost and motion blur at severity 1, 3, 5 on a few
hundred BDD100K clear-day frames. Run steps 3, 5 and 6 on them.

**Step 8. Trust model.** XGBoost on BDD100K clear-day frames only. Test on every other stratum.
Report AUROC of failure prediction per stratum, risk-coverage curves, and a feature-family ablation.
Baseline to beat: mean detection confidence alone.

**Step 9. Enhancement ON and OFF.** Recipes keyed on v4's `weather` and `fog_level`:

| Weather | Recipe | Arm to compare |
|---|---|---|
| clear | nothing | |
| fog, light | CLAHE on luminance, clip 1.5 | nothing |
| fog, medium | dark-channel dehazing, omega 0.75, floor 0.2, then CLAHE 1.5 | CLAHE 2.0 only |
| fog, dense | dark-channel dehazing, omega 0.9, floor 0.2, CLAHE 2.0, light bilateral denoise | AOD-Net |
| night | adaptive gamma towards mean 0.45, CLAHE 2.0, bilateral 5 px | Zero-DCE++ |
| rain, including `unrated` | CLAHE 1.5 to 2.0 | mild dehazing, omega 0.5 |
| snow | gamma 1.1 to 1.3 if very bright, CLAHE 1.5 | nothing |

Detector-side arms: a confidence threshold per condition chosen for equal precision, inference at 960 or 1280,
test-time augmentation. Rerun detection and Trust. Compare mAP50, pedestrian recall, and the mean confidence of
true and false detections. Confidence rising without correctness is miscalibration, and the Trust layer should
catch it. If a recipe does not beat OFF, the honest setting is nothing.

**Step 10. Danger.** A weighted sum normalised to 0..1: hazard types present with vulnerability weights, object
position and size, and a weather-severity term. Compute it twice per frame: `R_est` from detections, `R_true` from
annotations. Metric: the gap `R_true - R_est` per stratum. Perturb each weight by 50% up and down and show the
ranking of scenes is stable.

**Step 11. Quadrant.** Trust against Danger scatter per condition. The low-trust, low-apparent-danger corner is
the finding.

**Step 12. Figures, report, demo.** A 2 x 3 image grid per condition: raw, detections, readout.

**If time runs out, cut in this order:** DAWN column, learned enhancement arms, weight sensitivity, the corruption
sweep down to one corruption, the three-detector comparison down to YOLOv8m against YOLO26m.
**Never cut:** the raw-frame bypass, ground-truth stratification, the ON and OFF pair, `R_est` against `R_true`,
clear-only Trust training.

---

## 8. Methods and metrics, and why each was chosen

| Question | Method | Metric | Why this and not something else |
|---|---|---|---|
| Does the weather model generalise? | Train on two cameras, test on the third | Accuracy on a camera never seen | An ordinary test split shares the camera. It gave 99.7% where this gives about 94%. |
| Is a perfect score real? | Look at confidence across severity, then test on the counter-case | Share of clear frames called fog | v2 scored 100% by recognising the camera. |
| Are splits clean? | By city, by video, official ACDC splits, asserted on every run | Zero scenes in two splits | File-name parameters are renderings of one scene. |
| Which detector? | Same frames, three models | Pedestrian and rider recall in adverse weather at equal precision | COCO scores say nothing about fog. Those classes carry the most danger weight. |
| Can failure be predicted? | XGBoost on meta-features, clear-only training | AUROC per stratum, risk-coverage curves | Introspection is needed exactly where there was no training data. A negative transfer result is still reportable. |
| Is raw confidence enough? | Confidence-only baseline against the full feature set | AUROC difference | Modern detectors are miscalibrated and more so under shift. |
| Is the danger score valid? | The same formula on detections and on annotations | Gap per stratum | Hand-set weights largely cancel. What is measured is the distortion perception introduces. |
| Are the danger weights arbitrary? | Perturb by 50% | Rank stability | Answers "why these weights". |
| Does enhancement help? | ON against OFF on identical frames | mAP50, pedestrian recall, confidence of true against false detections | Restoration tuned for the eye often fails to help recognition. |
| What is fog severity? | v4 fog level, plus a dark-channel statistic | Monotonic rise over clear and three fog levels on identical scenes | No dataset labels real fog severity. The paired Cityscapes scenes are the only ground truth. |
| Weather or camera? | Clear Cityscapes as the camera-only baseline | Failure rate at clear against each fog level | Separates the two shifts that are otherwise mixed. |

Literature behind these choices: introspective perception (Daftry 2016, Rabiee and Biswas 2019), calibration of
modern networks (Guo 2017), haze removal and recognition (Pei 2018), common corruptions (Hendrycks and Dietterich
2019), fog-density estimation from Foggy Cityscapes (Sakaridis 2018), DINOv2 (Oquab 2024). Full list in
`MODEL_DETAILS_v3_v4.txt` section 16.

---

## 9. Environment and working rules

```powershell
conda activate weather
cd "D:\Projects\Weather analysed object detection"
```

- Conda is at `D:\Softwares\Anaconda` and is not on PATH. Use the Anaconda PowerShell Prompt.
- PyTorch must come from the CUDA 12.8 index. Verify with a real GPU computation.
- Memory on this laptop means Windows commit, about 21 GB. Never spawn processes from a script that imports
  PyTorch. Use threads for image caching. Cap loader processes from free memory. `--workers 0` always works.
- Set `HF_HUB_DISABLE_XET=1` for Hugging Face downloads. The default transfer backend stalls on this network.
- There is no version control. Never edit a finished version's scripts in place. Copy to a new folder.
  Running `git init` before the core work starts is strongly recommended.
- Every script writes `log_<command>.txt` beside its results.
- Test every new script with a tiny end-to-end run on real data before the full run.

Code map: `src` v1 and shared helpers, `src_cityscapes` v2, `src_v3` frozen fallback, `src_v4` final weather model.
Proposed for the core: `src_core`, writing to `results\core`.

Using v4 in the pipeline:

```python
import sys, glob; sys.path.insert(0, "src_v4")
from weather_v4_lib import load_ensemble, predict_image_ensemble
models = load_ensemble(sorted(glob.glob("results/weather_v4/ckpt_*.pt")), "cuda")
r = predict_image_ensemble(models, bgr_frame)   # r["weather"], r["fog_level"]
```

---

## 10. Timeline

The handoff put the trust model on Friday and the enhancement test on Saturday, with all trained components done
by Saturday night. That did not happen. Both days went into the weather component. It is finished and far stronger
than planned, but it is the part the plan called minor. Two days remain.

| Day | Work | Must exist at the end |
|---|---|---|
| Sunday 20 Sept, morning | Steps 1 to 4 | Records for three detectors, detector chosen |
| Sunday 20 Sept, afternoon and evening | Steps 5 to 8 | Trust AUROC and risk-coverage per stratum |
| Monday 21 Sept, morning | Steps 9 and 10 | ON and OFF table, danger gap per stratum |
| Monday 21 Sept, afternoon | Steps 11 and 12 | Figures, report, demo grid |

If the deadline can move, ask for it on Sunday, not Monday.

---

## 11. Figures the report needs

1. Detector comparison per stratum, three models: mAP50 and pedestrian recall.
2. Per-stratum mAP50 and pedestrian recall, enhancement ON against OFF.
3. Trust AUROC per stratum, trained on clear only, with the confidence-only baseline.
4. Risk-coverage curves per stratum.
5. Severity sweep: failure rate and Trust AUROC against synthetic severity, and against the Cityscapes fog levels
   with clear as the baseline.
6. Danger gap per stratum.
7. Weight-sensitivity table.
8. Trust against Danger quadrant, one panel per condition.
9. Weather model: v1, v2, v3, v4 and CLIP on identical images, the two v2 columns showing the shortcut, and the
   leave-one-camera-out table.
10. Fog-level confusion matrix, and the haze statistic rising with fog level.
11. Qualitative image grids for clear, night and fog.

---

## 12. Viva questions and prepared answers

- **How do you validate a risk score?** Trust is validated as a failure predictor against ground truth. Danger is
  validated by the gap between the same formula on detections and on annotations.
- **Why these weights?** They are hand-set and declared as such. A 50% perturbation leaves the ranking stable, and
  the gap metric cancels them on both sides.
- **Does detector confidence not already tell you trust?** Modern detectors are miscalibrated, more so under
  shift. The confidence-only baseline is in the results for exactly this comparison.
- **Your weather model scores 99.7%. Is it overfitted?** On a camera it has never seen it scores about 94%. Both
  numbers are reported. An earlier version scored 100% by recognising the camera. That was detected from flat
  confidence, proven on 60 clear frames, and fixed by adding clear frames from that camera.
- **Why not use CLIP?** On driving footage v4 is 3 to 7 points better and gives a fog level. On web images CLIP
  wins, and that is reported.
- **You excluded depth estimation, yet you use a depth network.** Only offline, to render fog onto clear frames
  for training data. No pipeline module uses depth, and no metric distance is ever claimed.
- **Why is there no rain severity?** No dataset labels real rain intensity. A head trained on synthetic streaks
  answers "none" on real rain, so the system reports `unrated` instead of inventing a level.
- **Why is sandstorm not a class?** One small web dataset is its only source. A class defined by one dataset
  teaches the dataset. The model calls 310 of 323 sandstorm images fog, which is the safe reading.
- **Where is the near-miss prediction?** Out of scope by design: no labels, and time-to-collision needs metric
  depth that one camera cannot give.
- **Why no tracking?** The data is single frames. Flip-consistency measures the same stability property per image.
- **Does enhancement not obviously help?** It is a tested hypothesis. The results show ON against OFF, including
  the cases where it hurts.
- **Why train Trust on clear weather only?** Runtime introspection is needed for the conditions that were not in
  the training data. Transfer is the experiment.
- **Which YOLO and why?** Three generations were run on identical frames and the choice was made by
  adverse-weather pedestrian recall, not by COCO score.

---

## 13. Known limits to state in the report

1. Camera shift and weather shift are mixed in the ACDC columns. Cityscapes separates them.
2. Rain severity is unavailable on real footage.
3. Real fog severity cannot be scored. Fog level is validated on synthetic fog over real scenes and on rendered fog.
4. BDD100K weather tags are noisy, which caps any model's accuracy there.
5. Real fog is scarce: 400 ACDC training images and 48 in BDD100K.
6. Foggy Cityscapes is synthetic fog on real scenes. ACDC fog is the real-fog check.
7. Single images only. No temporal evidence.
8. Benchmarks are single-seed. Differences under about one point are noise.
9. Danger weights are heuristics. Only the gap is claimed to be meaningful.
