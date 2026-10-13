"""Assemble final_release/: only the files needed to present, run and reproduce the project. Re-runnable.

    python make_release.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_release"
MAX_MB = 95  # GitHub refuses files over 100 MB

DOCS = ["README.md", "PROJECT_GUIDE.md", "PROJECT_LOG_REPORT.md", "FINAL_PROJECT_LOG.md", "MODEL_DETAILS_v3_v4.txt",
        "requirements.txt", "make_release.py"]
CODE_DIRS = {"src": ["weather_data.py", "train_weather_classifier.py", "clip_zeroshot_weather.py", "README_weather_classifier.md"],
             "src_cityscapes": ["train_weather_cityscapes.py", "README.md"],
             "src_v4": "*.py *.md", "src_core": "*.py *.md", "tests": "*.py"}
RESULT_GLOBS = {
    "results/core": ["RESULTS.md", "strata_counts.csv", "metrics_*.csv", "detector_comparison_off.csv", "detector_choice.json",
                     "danger_gap.csv", "danger_weight_sensitivity.csv", "quadrant_counts.csv", "enhancement_on_vs_off.csv",
                     "benchmark*.json", "summary.json", "analysis_info.json", "knn_reference_bdd_train_clear.npy",
                     "sample_input.mp4", "sample_output.mp4", "sample_output.csv"],
    "results/core/figures": ["*.png", "*.jpg"],
    "results/core/trust": ["*.json", "*.csv", "*.png"],
    "results/weather_v4": ["metrics.json", "comparison.csv", "comparison_with_clip.csv", "lodo_results.csv", "lodo_summary.csv",
                           "lodo_best.json", "fog_render_params.json", "history_*.csv", "sampling_shares.csv", "counts_*.csv",
                           "confusion_*.png", "ckpt_dinov2_last2.pt"],
    "results/weather_classifier_cityscapes": ["comparison_v1_vs_v2.csv"],
    "docs": ["*.pptx", "*.docx", "*.pdf"],
    "docs/build": ["*.js", "*.md"],
}
GITIGNORE = """# datasets live outside the repository (see PROJECT_GUIDE.md section 5)
DAWN/
bdd100k/
bdd100k-10k/
gtFine_trainvaltest/
gt_detection_trainval/
leftImg8bit_trainval_foggyDBF/
leftImg8bit_trainval_rain/
leftImg8bit_trainvaltest/
rgb_anon_trainvaltest_ACDC/
# regenerated at run time
weights/
results/*/cache/
results/core/records/
results/core/.stop_app
results/core/run_*.mp4
results/core/run_*.csv
results/core/run_*.jsonl
__pycache__/
*.pyc
.pytest_cache/
"""


def copy(src: Path, dst: Path) -> int:
    if src.stat().st_size > MAX_MB * 2**20:
        print(f"  skip (over {MAX_MB} MB, GitHub limit): {src.relative_to(ROOT)}")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    n = 0
    for f in DOCS:
        n += copy(ROOT / f, OUT / f)
    for d, spec in CODE_DIRS.items():
        names = spec if isinstance(spec, list) else [p.name for pat in spec.split() for p in (ROOT / d).glob(pat)]
        for name in names:
            n += copy(ROOT / d / name, OUT / d / name)
    for d, pats in RESULT_GLOBS.items():
        for pat in pats:
            for p in sorted((ROOT / d).glob(pat)):
                if p.is_file():
                    n += copy(p, OUT / p.relative_to(ROOT))
    (OUT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (OUT / "results" / "weather_v4" / "MODELS.md").write_text(
        "# Weather model checkpoints\n\n"
        "The final weather model is an ensemble of two checkpoints:\n\n"
        "| file | backbone | size | in this folder |\n|---|---|---|---|\n"
        "| `ckpt_dinov2_last2.pt` | DINOv2 ViT-S/14, last 2 blocks tuned | 88 MB | yes |\n"
        "| `ckpt_convnext_full.pt` | ConvNeXt-Tiny, fully tuned | 113 MB | no: over GitHub's 100 MB file limit |\n\n"
        "The application runs with the one checkpoint present (a single-model prediction instead of the ensemble; the\n"
        "DINOv2 member alone scores 99.68% on ACDC test against 99.72% for the ensemble, see `comparison.csv`). To restore the\n"
        "ensemble, either copy `ckpt_convnext_full.pt` from the original results folder, store it with Git LFS, or retrain it:\n\n"
        "    python src_v4/train_weather_v4.py train --configs convnext_full --no-final-eval\n\n"
        "The trust model (`results/core/trust/trust_model.json`), its feature list, and the kNN reference of clear-day\n"
        "embeddings (`results/core/knn_reference_bdd_train_clear.npy`) are included and are all the runtime needs.\n",
        encoding="utf-8")
    weights = OUT / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "README.md").write_text("YOLO weights (`yolov8m.pt`, `yolo11m.pt`, `yolo26m.pt`, about 50 MB each) are downloaded here\n"
                                       "automatically by Ultralytics on first use. Nothing to do.\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 2**20
    print(f"[release] {n} files, {total:.1f} MB -> {OUT}")
    big = sorted(((p.stat().st_size / 2**20, p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()), reverse=True)[:6]
    print("[release] largest files: " + ", ".join(f"{r} ({s:.1f} MB)" for s, r in big))


if __name__ == "__main__":
    sys.exit(main())
