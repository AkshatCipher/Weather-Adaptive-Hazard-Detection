# Project Log Report

**Project:** Confidence-Aware Weather-Adaptive Hazard Detection for Autonomous Driving Using Monocular Video
**Course:** Robotic Perception, also serving Computer Vision. Single-member team.
**Period covered:** project origin to Saturday 19 September 2026, 23:00
**Machine:** ASUS TUF F16, i7-14650HX, 15.6 GB RAM, RTX 5060 Laptop 8 GB (Blackwell sm_120), Windows 11
**Status at the end of this log:** the weather-recognition component is finished (v4). The graded core of the
project (detection sweep, trust model, danger gap, enhancement test) has not been started.

Every number in this log was measured on this machine and is traceable to a file under `results/`.
Times are taken from the timestamps of the result files.

---

## 0. One-page summary

| Stage | Date | What happened | Outcome |
|---|---|---|---|
| A. Origin and scoping | up to 17 Sept | 18 project ideas ranked; a 9-module concept cut to a 2-score pipeline; Review I submitted | Frozen scope |
| B. Reality check | 18 Sept morning | Machine and data did not match the plan | 5 blockers found |
| C. Unblocking | 18 Sept | Security setting, labels, annotations, environment | All blockers cleared |
| D. v1 classifier | 18 Sept 14:33 | MobileNetV3 on ACDC + BDD100K subset | 98.75% ACDC, fails on an unseen camera |
| E. CLIP baseline | 18 Sept 15:05 | Zero-shot, no training | 96.6% ACDC, robust to the unseen camera |
| F. v2 classifier | 18 Sept 20:26 | Added Foggy, Rain and clean Cityscapes | A camera shortcut was found, proven and removed |
| G. v3 classifier | 19 Sept 07:56 | 3 heads, rendered weather, leave-one-camera-out benchmark | 99.48% ACDC, fog levels 95.0% |
| H. v4 classifier | 19 Sept 19:08 | Full BDD100K, DAWN, 2-model ensemble | 99.72% ACDC, 96.4% BDD100K, fog levels 97.85% |
| I. CLIP vs v4 | 19 Sept 21:07 | Same 16,085 test images | v4 wins on driving footage, CLIP on web images |
| J. Design decisions | 19 Sept evening | Enhancement recipes per weather; detector choice by measurement | Ready for the core work |

Headline progression of the weather component on identical test images:

| | v1 | v2 | v3 | v4 | CLIP |
|---|---|---|---|---|---|
| ACDC test accuracy | 98.75% | 98.95% | 99.48% | 99.72% | 96.6% |
| BDD100K test accuracy, 7,875 fair images | not measurable | 94.5% | 95.4% | 96.4% | 89.0% |
| DAWN, cameras never seen | not measured | 69.2% | 74.0% | 74.3% | 82.7% |
| Fog level on real Foggy Cityscapes | none | none | 95.0% | 97.85% | none |
| Accuracy on a camera never trained on | not measured | not measured | about 90% | about 94% | not applicable |

---

## A. Origin and scoping (before 18 September)

**Start.** About 18 robotic-perception project ideas were brainstormed and ranked on feasibility 35%, innovation
25%, research value 20% and demo impact 20%. A teammate's monocular visual-odometry proposal was analysed twice
and belongs to the teammate. A second concept arrived: a 9-module "weather-adaptive hazard detection and
near-miss prediction" pipeline.

**Issues found in the 9-module concept.**
- Nine serial modules is a research programme, not a 3 to 4 day project. Errors compound and the ablation plan
  was incoherent for a serial chain.
- Time-to-collision from monocular depth has no metric scale, the same scale problem as monocular odometry.
- No dataset on the list contains near-miss labels, so that experiment could not be run.
- Enhancement was assumed to help detection, which the literature contradicts.
- Risk fusion and decision generation had nothing to be validated against.

**Mitigation.** The scope was cut to a pipeline with two separate scores per frame, Trust and Danger, with every
model frozen except a small weather classifier and a gradient-boosted trust model. Near-miss, time-to-collision,
depth estimation as a pipeline module, risk fusion and detector training were excluded in writing.

**Result.** Review I document written and submitted. `PROJECT_HANDOFF.md` records the frozen scope, the decision
log and a 4-day plan. Datasets had been downloaded but never inspected.

---

## B. Reality check (18 September, morning)

**Start.** First step of the plan: verify the data and the environment before writing any model code.

**Issues found.**

| # | Expected by the plan | What was actually on disk |
|---|---|---|
| 1 | PyTorch with CUDA works | Windows Smart App Control was on and blocked PyTorch's unsigned DLLs. Only Python 3.14 existed, with a CPU-only torch. No WSL. |
| 2 | BDD100K 10k subset with labels and weather tags | 10,000 images and no label file of any kind |
| 3 | ACDC with annotations | Images only |
| 4 | Clean Cityscapes as a clear-weather set | Only the foggy and rainy variants, and no gtFine annotations |
| 5 | Review I "initial progress" section | All three claims were not yet true |

**Result.** Four questions put to the user. Nothing GPU-side could start until the first was answered.

---

## C. Unblocking (18 September)

| Issue | Mitigation | Result |
|---|---|---|
| Smart App Control | The user turned it off. Verified by registry state and a successful torch import. | Cleared |
| No CUDA environment | Conda env `weather`, Python 3.12, PyTorch 2.11.0 from the CUDA 12.8 index. Verified with a real GPU matrix multiply, not just `is_available`. Conda-forge channel used so no licence terms had to be accepted. | sm_120 kernels run |
| BDD100K labels, official site returns 403 | The ETH mirror does not resolve in DNS from this network. A public Kaggle dataset holds the two official label JSONs. | Labels placed in `bdd100k-10k\labels\` |
| Only 3,426 of the 10,000 subset images have labels | The subset is the segmentation split, not the detection split. Accepted, and later solved by the full download in stage H. | Documented |
| ACDC boxes | Separate COCO-format detection package downloaded. Quirk found: the rain training file is a copy of the combined file, so it must be filtered by path. | 500 labelled images per condition |
| Cityscapes boxes | gtFine package downloaded. | Instance masks for all 3,475 scenes |
| A 30-agent workflow to research and write the first scripts stopped half-way | The account's usage limit was reached. The environment and three script drafts had completed. The remaining work was finished by hand from those drafts. | No work lost |
| Wrong assumption of mine: ACDC splits are disjoint by recording | The listing showed the same GoPro recording ids in train, val and test. The guard was changed to "no identical recording-and-frame pair in two splits", with the overlap reported openly (28 of 60 recordings). | Correct guard |

---

## D. v1 weather classifier (18 September, 14:33)

**Start.** MobileNetV3-Large, five classes (clear, fog, night, rain, snow), trained on ACDC and the labelled
BDD100K subset. ACDC's clear reference frames serve as the clear class. Foggy Cityscapes kept out of training as
an unseen-camera probe. No colour or brightness augmentation, because brightness and contrast are the night and
fog cues. Images cached once at 320 px.

**Issues and mitigations.**

| Issue | Cause | Mitigation |
|---|---|---|
| Smoke test ran out of memory during the cache build | On Windows every spawned worker process re-imports the script, PyTorch included. 16 processes exhausted 16 GB. | Cache builder switched to a thread pool. OpenCV releases the interpreter lock while decoding, so threads scale. |
| Full training succeeded, then the automatic evaluation crashed and the terminal hung | 12 new loader processes were started while the 15 from training were still alive: 27 copies of PyTorch. | Training loaders are shut down before evaluation. Evaluation uses 4 processes. Maths libraries forced to one thread per process. The hung process tree was stopped after confirming the checkpoints were on disk. |
| The training summary named the wrong best epoch | Ties were resolved towards the last epoch, but the saved model was the first to reach the top score. | Summary corrected. The saved model had always been the right one. |
| CLIP weight download stalled at 67 MB | Hugging Face's Xet transfer backend stalls on this network. Plain HTTP works. | Plain HTTP made the default. The real file size is 605 MB. |

**Results** (`results/weather_classifier/`).
- ACDC validation 99.63%, ACDC test 98.75%, macro-F1 0.984. Only real confusion: rain against snow.
- Unseen-camera probe on Foggy Cityscapes: fog recognised in 2%, 9% and 25% of light, medium and dense fog frames.
- Reading: near-perfect on the camera it trained on, weak on a camera it never saw. This is the project's thesis
  in miniature.

---

## E. CLIP zero-shot baseline (18 September, 15:05)

**Issue.** The first run printed a QuickGELU mismatch warning. The OpenAI weights were loaded into a model built
with a different activation function, silently degrading it.
**Mitigation.** Default model changed to the QuickGELU variant. The mismatched metrics were kept for comparison.
**Result.** ACDC test 96.6%. Foggy Cityscapes fog recognised in 73%, 89% and 96% of frames. Weak class: snow.
**Decision at that point.** Use CLIP at runtime because it survives a camera change. This was later reversed.

---

## F. v2: adding Cityscapes fog and rain (18 September, 20:26)

**Start.** The user asked to train on Foggy and Rain Cityscapes too. This reopened an exclusion, and one risk was
stated before building: with no clean Cityscapes frames on disk, every Cityscapes-camera image is fog or rain.

**Design.** Split by city with one city list shared by fog and rain, because the beta, alpha, dropsize and pattern
values in the file names are renderings of the same street frame. Test cities are Cityscapes' official validation
cities. Sampling per scene, not per image, because rain has only 202 training scenes drawn 36 ways.

**Issues and mitigations.**

| Issue | Evidence | Mitigation |
|---|---|---|
| A bug in my per-scene key | The scene table showed zero clear ACDC scenes. An ACDC clear reference frame carries the same recording and frame number as its adverse twin. | The label was added to the scene key. |
| 100% on every Cityscapes test row | Confidence was flat at 0.957 for light, medium and dense fog, the ceiling the loss allows, and higher than on real ACDC fog. | Diagnosed as a camera shortcut. |
| The shortcut could not be proven | No clear-weather frame from that camera existed on disk. | The user downloaded the clean Cityscapes package, 11 GB. |
| The package had a different folder name than I had given | Cityscapes calls it `leftImg8bit_trainvaltest`. | The script was taught the real name. |
| Proof | The model trained without clean frames called 59 of 60 clear Cityscapes frames fog, at 0.92 confidence. v1 called 59 of the same 60 clear. | Retrained with the clean frames as clear. |

**Results** (`results/weather_classifier_cityscapes/`), identical test rows.

| | v1 | v2 without clean frames | v2 retrained |
|---|---|---|---|
| ACDC test | 98.8% | 98.65% | 98.95% |
| Clear Cityscapes called clear | 99.4% | 3.4% | 99.8% |
| Light fog called fog | 2% | 100% | 99.2% |

- In all 500 paired scenes the fog frame scored higher than its clear twin. The model now reacts to haze.
- Its confidence still saturated across fog levels, so it detects fog but cannot measure it.
- Both v2 checkpoints were kept. The pair is a clean cautionary result for the report.

---

## G. v3: severity levels without camera overfitting (19 September, 07:56)

**Start.** The user asked for a model that is better than v2, does not overfit, and separates fog types and rain
types, with CLIP as the last choice.

**Research that shaped it.**
- The Foggy Cityscapes authors trained a fog-density estimator on the clean frames plus the three beta levels and
  applied it to real fog. A 4-level fog head is established practice.
- A 2026 recipe study found lightly tuned pretrained backbones generalise best and large fully tuned models
  overfit, that scene-balanced sampling helps and that restoration preprocessing hurts.
- A 2026 fog-density paper renders fog with estimated depth on another dataset to obtain severity labels.

**Design.** One backbone, three heads: weather, fog level, rain level. Severity labels exist for one camera only,
so fog and rain are rendered onto clear frames of all three cameras with the atmospheric scattering model and a
depth map from Depth Anything V2 Small. The renderer is calibrated against the real fog frame of the same scene.
"Does not overfit" is measured by training on two cameras and testing on the third.

**Issues and mitigations.**

| Issue | Mitigation |
|---|---|
| A held-out camera with few test images is evaluated on its training rows too, and rendered training rows have no stored severity. Evaluation crashed. This would have hit the real run. | A reproducible severity is derived from the file path whenever none is stored. |
| A visual check showed the network-free depth fallback fogs near objects as if they were far. | Kept as a fallback only. The real run uses the depth network. |
| Rendered rain streaks were too thick and bright. | Made finer and denser. |
| Rain level fails on real rain: "none" on 496 of 500 real ACDC rain frames, although the weather head says rain on 97.6%. | The predict code, which always picked a level, now reports `unrated`. The cause is structural: rain severity could only be learned from synthetic streaks. |

**Results** (`results/weather_v3/`).
- Benchmark winner: DINOv2 with its last two blocks tuned, 0.752, against ConvNeXt 0.748. A near tie.
- Accuracy on a camera never seen: about 90%, against 99.5% on cameras trained on.
- ACDC test 99.48%, 21 errors against v2's 42. Cityscapes rows tie with v2.
- Fog level: 95.0% exact and 99.8% within one level on real Foggy Cityscapes in unseen cities. 97.8% on rendered
  fog on ACDC and BDD100K. 84.4% on rendered fog on Cityscapes: my renderer read about one level too dense.
- Renderer calibration: pixel error 10.3 against 36.6 for doing nothing.

---

## H. v4: more data, an ensemble, and a version split (19 September, 19:08)

**Start.** The user downloaded the full BDD100K images and DAWN and asked for the best accuracy regardless of
training time.

**Issues and mitigations.**

| Issue | Evidence | Mitigation |
|---|---|---|
| BDD100K looked incomplete: 1,160 training files where 70,000 were expected | A recursive count found 70,000. The package nests them in `trainA`, `trainB`, `testA`, `testB`. | Scanner made recursive. Without this, 98% of the new images would have been ignored silently. |
| A bug I introduced in that fix | My folder loop reused the variable holding the label files. Caught by the first run on real data. | Renamed. |
| "Why so few training images?" | I had capped clear and night at 8,000 each for speed. Rain 2,411, snow 2,714 and fog 43 are all BDD100K has. | Cap removed: 54,822 BDD100K training images instead of 21,168. Each night image is then seen about 5 times instead of 15. |
| Renderer bias from v3 | | Calibrated on all three fog levels at once with a bias penalty and an extra exponent. Bias at the medium level: -0.01. |
| Should sandstorm become a class? | DAWN is its only source, 323 web images. | No. It would teach the dataset's look, the v2 shortcut again, and would destroy the only unseen-camera test. Used as an out-of-distribution probe. |
| v3 scripts had been upgraded in place, and the project has no version control | The v3 model and results were intact. | New code moved to `src_v4`. `src_v3` rebuilt from the earlier version and verified by reproducing every logged v3 number exactly on a copy. A freeze guard stops v3 being retrained over. |

**Design changes.** Two-model ensemble, best recipe plus best recipe of a different backbone, averaged together
with the mirrored image. 30 epochs of 24,000 draws. A recipe tuning four DINOv2 blocks added to the benchmark.
DAWN as an external test. 318 BDD100K test images that v2 had trained on are excluded from every comparison row.

**Results** (`results/weather_v4/`), identical real test images.

| | v2 | v3 | v4 |
|---|---|---|---|
| ACDC test | 98.95% | 99.48% | 99.72%, 11 errors of 4,000 |
| BDD100K test, 7,875 images | 94.5% | 95.4% | 96.4% |
| BDD100K rain and snow recall | 73% and 69% | 72% and 75% | 81% and 79% |
| DAWN | 69.2% | 74.0% | 74.3% |
| Fog level, real Foggy Cityscapes | none | 95.0% | 97.85% |
| Fog level, rendered on Cityscapes | none | 84.4% | 93.8% |
| Accuracy on a camera never seen | not measured | about 90% | about 94% |

- A model that never saw ACDC scored 96.6% on it.
- The ensemble beats both members on BDD100K: 96.0% and 95.9% alone, 96.4% together.
- Sandstorm probe: 310 of 323 images called fog, the sensible answer.
- Rain level still answers "none" on all real rain. More data cannot fix this because no dataset labels it.
- Training time 60 and 68 minutes for the two members.

---

## I. CLIP against v4 (19 September, 21:07)

Same 16,085 real test images. `src_v4/compare_clip.py`.

| | CLIP | v4 |
|---|---|---|
| ACDC | 96.6% | 99.7% |
| BDD100K | 89.0% | 96.4% |
| BDD100K snow recall | 35% | 79% |
| Light fog on Cityscapes | 64% | 99.4% |
| DAWN | 82.7% | 74.3% |
| DAWN snow recall | 99% | 69% |

**Reading.** v4 wins clearly on driving footage. CLIP wins on DAWN, which is web imagery like its pretraining data.
CLIP gives no severity level. The earlier decision to use CLIP at runtime is reversed: v4 is the runtime model.

---

## J. Decisions taken for the next phase (19 September, evening)

- **Enhancement per weather class.** Nothing for clear. Fog dehazing in proportion to v4's fog level. Shadow
  lifting without touching highlights at night. No removal of rain or snow, only mild contrast recovery.
  Every recipe is an ON and OFF experiment, because clearer to the eye is not better for a frozen detector.
- **Detector-side settings** matter as much: a confidence threshold per condition, a larger inference size for
  the small far objects that fog hides, and test-time augmentation.
- **Detector choice.** YOLOv8m, YOLO11m and YOLO26m will be run on identical frames and the winner chosen by
  pedestrian and rider recall in adverse conditions at equal precision, not by COCO score. Review I says
  "a frozen YOLO-family detector", so any of the three is consistent with it.

---

## K. Recurring lessons

1. **Inspect the data before trusting a plan.** Five of the plan's assumptions were false on the first morning.
2. **A perfect score is a warning.** The 100% in v2 was a shortcut. Flat confidence exposed it and a 60-image
   test proved it.
3. **Every dataset is also a camera.** A class that lives in one dataset teaches the dataset. This decided the
   clean Cityscapes download, the rendering on all cameras, and the refusal of a sandstorm class.
4. **Measure generalisation, do not assert it.** The leave-one-camera-out benchmark gave 94% where the ordinary
   test gave 99.7%. Both numbers belong in the report.
5. **Small real runs catch real bugs.** Every tiny end-to-end run on real data found at least one bug that a
   full run would have hit.
6. **On this laptop memory means Windows commit, about 21 GB, not RAM.** Never spawn processes from a script that
   imports PyTorch. Loader processes are capped from free memory automatically.
7. **No version control exists.** A finished version's scripts must never be edited in place. Copy to a new folder.
8. **State limits plainly.** Rain level does not work on real rain. Real fog severity cannot be scored. BDD100K's
   tags are noisy. Saying so is stronger than hiding it.

## L. Honest account of time

The plan reserved Friday for the trust model and Saturday for the enhancement test. Both days went into the
weather component, which the plan rated as minor. The component is far stronger than planned and yields a
self-contained domain-shift study, but the graded core is unstarted with two days left.
`PROJECT_GUIDE.md` gives the compressed plan.
