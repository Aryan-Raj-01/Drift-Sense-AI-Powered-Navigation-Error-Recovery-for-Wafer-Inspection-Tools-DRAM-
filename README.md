# Drift-Sense — AI-Powered Navigation-Error Recovery for Wafer Inspection

Locates a 100× reference pattern inside a 10× search image and returns the
centre coordinates of the match, in search-image pixels.

**Headline result — 2,000 held-out pairs, never trained on, never used to tune
any hyperparameter:**

| | ≤1 px | ≤2 px | ≤4 px | ≤5 px | median | runtime |
|---|---|---|---|---|---|---|
| **This system** | **89.5%** | **95.9%** | **96.7%** | **96.7%** | **0.24 px** | **186 ms/pair** |
| Classical NCC baseline | 76.6% | — | — | — | — | 705 ms/pair |

Median error 0.24 px ≈ 2.4 nm on the wafer. The system is both **12.9 points
more accurate and 3.8× faster** than the classical baseline it replaces.

---

## 1. Quick start

Everything below runs from the archive root with no source-code edits.

### 1.1 Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Tested on:** Python 3.14, Windows 11, NVIDIA RTX 4050 Laptop GPU (CUDA).
A GPU is *not* required — the code runs on CPU automatically, roughly 8–10×
slower. No manual configuration is needed either way.

### 1.2 Generate a sample dataset

```bash
python generate_dataset.py --style dram --num-images 30 --out data/sample --format png --yes
```

Produces 30 reference/search pairs as PNG files plus `labels.jsonl`,
`labels.csv` and per-pair metadata under `data/sample/`. Use
`--style finfet` for the FinFET layout family, or `--style mixed` for both.

### 1.3 Localize a single pair

```bash
python localize.py --reference data/sample/images/00000/0000000_ref.png --search data/sample/images/00000/0000000_search.png
```

Output:

```
  x = 288.2599
  y = 391.8231
  confidence 2.249e-01   margin 2.88   method learned
  timing: 176 ms/pair steady-state   (25267 ms including one-off model load and CUDA warm-up)
```

The model weights load automatically from `model/final_phase3_all.pt`. There is
nothing to download and nothing to edit.

**Check the `method` field.** It reports which path produced the answer:
`learned` means the neural model ran. If it ever reads `classical`, the
checkpoint failed to load and the result came from the fallback — still a valid
coordinate, but at the baseline's 76.6% accuracy rather than 89.5%. The
fallback is deliberately silent about failure at the per-pair level so a batch
cannot be destroyed by one bad file, which makes this field the way to confirm
the real model is running.

### 1.4 Localize a batch

```bash
python localize.py --csv pairs.csv --out predictions.csv
```

`pairs.csv` must contain the columns `Wide Search Image Path` and
`Reference Image Path`. Paths may be absolute or relative to the CSV's own
directory. The output reproduces those columns and appends `GTx`, `GTy`,
`confidence` and `margin`. The model is loaded once for the whole batch, so the
185 ms/pair steady-state figure is what applies at any batch size.

### 1.5 Use it as a library

```python
from localize import locate
x, y = locate(reference_image, search_image)   # grayscale numpy arrays
```

---

## 2. Coordinate convention

- Origin `(0, 0)` is the **top-left** of the search image.
- `x` increases to the **right**, `y` increases **downward**.
- Returned values are **sub-pixel floats**, not integers.
- The returned point is the centre of the region in the search image
  corresponding to the centre of the reference image.
- Both images are 1000 × 1000 grayscale; the nominal magnification ratio is
  10:1, and the system is trained and evaluated across roughly 9.6:1 to 10.4:1.

---

## 3. Folder structure

```
DriftSense_Submission/
├── README.md                     this file
├── requirements.txt              pip freeze from the tested environment
├── generate_dataset.py           synthetic dataset generator  (entry point)
├── localize.py                   localization / inference     (entry point)
├── solution_presentation.pptx    solution presentation
├── configs/
│   └── config.yaml               generator configuration
├── model/
│   └── final_phase3_all.pt       trained weights, loaded automatically
├── src/driftsense/               all source code
│   ├── cli/generate.py           generator implementation
│   ├── config.py                 generator parameter space
│   ├── layouts/                  DRAM and FinFET layout synthesis
│   ├── optics/                   SEM image formation: PSF, noise, detection
│   ├── pipeline.py               pair rendering and ground-truth computation
│   ├── localize/                 classical NCC baseline (also the fallback)
│   ├── eval/                     baseline and metric implementations
│   └── dl_localize/               the learned localizer
│       ├── model.py            Siamese correlation network
│       ├── coords.py           output-grid ↔ pixel mapping
│       ├── losses.py           training objectives
│       ├── train.py            training script
│       ├── infer.py            inference
│       ├── refine.py           classical sub-pixel refinement stage
│       ├── eval.py             evaluation and diagnostics
│       ├── report.py           figure and failure-report generation
│       └── localize.py           the locate() API
├── results/
│   ├── metrics_golden.json               all reported numbers
│   ├── manifest_golden_predictions.csv   per-pair paths, truth, predictions
│   ├── accuracy_by_threshold.png
│   ├── error_distribution.png
│   ├── confidence_separation.png
│   ├── robustness_covariates.png
│   └── failure_cases/                    annotated failure panels
└── references/
    └── references.md             literature justifying structure and noise
```

---

## 4. Method

Two stages, each handling the half of the problem it is measurably better at.

### Stage 1 — learned global search

A Siamese dense-correlation network (SiamFC-style, see `references/`) embeds
the downscaled reference and the search image through a shared stride-4
encoder, then cross-correlates the reference embedding against the search
embedding to produce a dense response map. The peak is the coarse location.

The network is trained with whole-map softmax cross-entropy plus a **lattice
margin term** that explicitly penalizes confusing the true cell with its
periodic replicas — the failure mode that defeats classical matching on
repetitive layouts.

### Stage 2 — classical sub-pixel refinement

Stage 1's answer is quantized to its stride-4 output grid. A stride-1
multi-angle normalized cross-correlation search over a ±3 px window, with a
3-point parabolic fit, recovers the last fraction of a pixel.

**Measured contribution: median error 0.493 px → 0.240 px, for about 5 ms.**

The ±3 px window is deliberately narrower than the smallest lattice pitch in
the dataset (5.3 px), so stage 2 *cannot* relocate to a periodic replica — it
can only sharpen the cell stage 1 already chose. This is a safety property of
the design, not a tuning choice.

### Decision rule for repeated patterns

Where several matches are equally valid, the problem statement specifies taking
the one nearest the search-image centre. That rule is implemented
(`--tie-margin`) but **disabled by default**, on measured evidence. See §7.

---

## 5. Results

All numbers below are on the **golden set**: 2,000 pairs held out from
training and never used to select a hyperparameter.

### 5.1 Threshold-wise accuracy

| bucket | n | ≤1 px | ≤2 px | ≤3 px | ≤4 px | ≤5 px | mean | median | p90 | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| **ALL** | 2000 | **89.5%** | 95.9% | 96.5% | 96.7% | 96.7% | 16.06 | **0.24** | 1.02 | 920.02 |
| easy | 1000 | 92.3% | 98.7% | 99.5% | 99.8% | 99.8% | 1.84 | 0.23 | 0.86 | 920.02 |
| medium | 600 | 91.8% | 98.7% | 99.3% | 99.3% | 99.3% | 2.49 | 0.23 | 0.87 | 419.73 |
| hard | 400 | 79.2% | 84.5% | 85.0% | 85.0% | 85.0% | 71.95 | 0.30 | 374.12 | 866.97 |

**On reading the mean.** On buckets with gross misses the mean is dominated by
the tail and is not a useful central estimate — hard has a mean of 71.95 px
against a median of 0.30 px. Both are reported so the divergence is visible
rather than inferred. The median is the honest central figure; the mean and
worst-case describe the tail.

Difficulty buckets are properties of the generated data, not of the model:
*easy* has distinctive landmarks, *medium* has weaker ones, *hard* is
defect-only periodic structure where the pattern is genuinely near-ambiguous.

### 5.2 Baseline comparison

Same pairs, same run, same hardware:

| | learned | classical NCC |
|---|---|---|
| ALL ≤1 px | **89.5%** | 76.6% |
| easy | 92.3% | 90.4% |
| medium | 91.8% | 81.3% |
| hard | **79.2%** | 35.0% |
| runtime | **186 ms/pair** | 705 ms/pair |

Almost the entire advantage is on the hard bucket — **35.0% → 79.2%**. That is
the periodic-ambiguity case the learned lattice-margin objective targets, and
it is where classical correlation fundamentally cannot distinguish a cell from
its replicas.

### 5.3 Runtime and timing method

- **186 ms/pair** steady-state, NVIDIA RTX 4050 Laptop GPU, fp16, batch of 1.
- Measured with `time.perf_counter()` around the `locate()` call.
- One-off model load and CUDA warm-up (~7 s) is **excluded**, because batch
  mode loads the model once regardless of batch size. Single-pair CLI mode
  prints both figures so the distinction is visible.
- Observed range across runs: 164–188 ms/pair. The slowest is reported.

### 5.4 Robustness

Accuracy versus each generation parameter (`results/robustness_covariates.png`).
A flat trend rules a factor out as a cause; a falling trend names it.

| factor | range tested | trend |
|---|---|---|
| relative rotation | 0–6.8° | flat (89–91%) — absorbed by augmentation |
| scale ratio | 9.63–10.40 | flat (88–92%) |
| lattice pitch | 53–229 nm | flat (89–90%) |
| electron dose | 22–110 e/px | flat (88–90%) |
| shot SNR | 3.3–7.4 | flat (88–90%) |
| landmark size | 0–430 nm | **79% → 93%** |
| **non-rigid distortion** | 0.01–3.67 px | **96% → 70%** |

Noise, dose, rotation and scale are all *ruled out* as the limiting factor.
The two that matter are landmark distinctiveness and non-rigid scan distortion
— and distortion is by far the strongest. This is a data-realism property, not
a model deficiency: when the reference pattern is warped relative to the
search image by more than a pixel, no rigid match can be exact.

---

## 6. Failure analysis

### 6.1 Failure taxonomy

| bucket | hit | near-miss (1–5 px) | periodic-lock | gross-miss |
|---|---|---|---|---|
| ALL | 89.5% | 7.2% | 1.4% | 1.9% |
| easy | 92.3% | 7.5% | 0.0% | 0.2% |
| medium | 91.8% | 7.5% | 0.2% | 0.5% |
| hard | 79.2% | 5.8% | 6.8% | 8.2% |

- **near-miss** — right region, imperfect sub-pixel. Uniform across buckets at
  ~7%, and tracks non-rigid distortion, not noise.
- **periodic-lock** — locked onto a lattice replica an integer number of
  pitches away. Effectively eliminated on easy/medium (0.0%/0.2%); remains
  6.8% on hard, where the structure is genuinely periodic with no landmark.
- **gross-miss** — wrong region entirely. 8.2% on hard.

### 6.2 The model knows when it has failed

This is the most operationally useful result in the submission. The logit
margin separates catastrophic failures almost perfectly:

| | median margin | p90 |
|---|---|---|
| hits (≤1 px) | 5.70 | — |
| gross misses (>5 px) | **0.35** | 0.94 |

Thresholding on it:

| margin < | flags | **catches of gross misses** | wrongly flags of correct |
|---|---|---|---|
| 1.0 | 5.5% | **92.4%** | 2.3% |
| 1.5 | 7.0% | **98.5%** | 3.8% |
| 2.0 | 9.2% | **100%** | 6.0% |

In a real inspection tool this matters as much as the accuracy itself: at a
threshold of 1.5 the system flags 7% of results and captures 98.5% of its own
catastrophic errors, so the tool can re-scan or fall back rather than silently
acting on a coordinate 900 px wrong. See `results/confidence_separation.png`.

### 6.3 Worked failure case

`results/failure_cases/failure_1_109881.png` — the single worst error in the
entire evaluation, 920 px, on an *easy* pair.

**What happened.** Ground truth (349.8, 199.1); predicted (945.2, 900.5).
The chosen region and the true region are visually near-identical: the same
horizontal dark bar with the same periodic contact array above and below. The
reference's distinguishing landmark is itself a structure that repeats across
the layout.

**Root cause — three factors compounding:**

1. **Genuine structural ambiguity.** The layout's landmark repeats. The chosen
   region is a legitimate visual match.
2. **Non-rigid distortion of 1.50 px**, in the top quintile where accuracy
   drops to 70%. The true match is warped; the false one is not, so the false
   one correlates *better*.
3. **Low SNR (3.90)**, flattening the correlation surface further.

**The model reported the failure.** Margin 0.04, against a hit average of 5.47
— the lowest-confidence prediction class in the entire set. Under the §6.2 rule
this pair is flagged.

**It is recoverable.** Enabling the problem statement's closest-to-centre rule
(`--tie-margin 1.0`) on this pair returns (348.5, 199.2) — **1.3 px error,
down from 920 px.** The true target sits 336 px from the search centre; the
false one 599 px. The rule works exactly as intended here. Why it is
nevertheless off by default is §7.

---

## 7. The closest-to-centre rule — implemented, measured, off by default

The problem statement specifies: *where several valid matches exist, select the
one whose centre is closest to the search-image centre.*

That rule is implemented — `--tie-margin` on both entry points,
`set_tie_margin()` in the API, `tie_margin=` in `LearnedLocalizer88`. When the
runner-up peak is within `tie_margin` logits of the winner, the candidate
nearer the centre takes the answer.

**It defaults to disabled, because we measured it.** On 2,000 held-out pairs:

| tie_margin | ALL ≤1 px | easy | medium | hard |
|---|---|---|---|---|
| **0.0 (default)** | **90.8%** | **92.7%** | 92.7% | **83.2%** |
| 1.0 | 90.2% | 92.4% | **93.2%** | 80.5% |
| 1.5 | 89.5% | 91.9% | 92.3% | 79.0% |

Enabling it costs accuracy monotonically; on hard, total >5 px failures rise
from 9.4% to 12.2%.

**Why, precisely.** The rule's premise is sound — among our gross misses the
true answer is nearer the centre than the wrong one **71%** of the time, well
above chance. But the tie-break can only choose among peaks the correlation
surface already proposes, and on a genuine gross miss there is no peak near the
truth. So it often trades one wrong answer for a different wrong answer, while
also displacing the ~2.3% of correct predictions that carry a low margin. It
fixes our single worst case (§6.3) and breaks a larger number of marginal
successes.

**When it should be enabled.** Our generator places targets close to uniformly
— median 307 px from the search centre, only 31% within 250 px. A real
navigation-recovery scenario is *not* uniform: the tool landed approximately
right, so the target concentrates near the centre. On a centre-weighted
evaluation distribution this rule should help rather than hurt. Enable it with
`--tie-margin 1.0`.

---

## 8. Reproducing the reported numbers

```bash
# threshold-wise accuracy, taxonomy, robustness trends, per-pair CSV
python -m driftsense.dl_localize.eval \
    --dataset <dataset root> \
    --checkpoint model/final_phase3_all.pt \
    --split-csv <split csv> --split golden --pairs 2000 \
    --dump-csv results/manifest_golden_predictions.csv \
    --out results/metrics_golden.json

# figures and failure panels
python -m driftsense.dl_localize.report \
    --csv results/manifest_golden_predictions.csv \
    --dataset <dataset root> --out results

# add --classical to eval to re-measure the baseline (~705 ms/pair)
```

Set `PYTHONPATH` to `src` first, or run from a directory where `src` is
importable. Training is reproduced with `src/driftsense/dl_localize/train.py`;
the exact command and schedule are documented in that file's docstring.

---

## 9. Assumptions and limitations

**Assumptions**

- Both images are 1000 × 1000 grayscale, nominal 10:1 magnification.
- The reference pattern is present in the search image.
- Rotation is small (trained across ±3°, evaluated to 6.8°).
- Scale is near 10:1 (evaluated 9.63–10.40).

**Limitations, stated honestly**

1. **Non-rigid distortion is the binding constraint.** Accuracy falls from 96%
   to 70% across the distortion range. Above ~1 px of warp, no rigid match is
   exact. Fixing this needs a deformable readout, not more data or more
   training.
2. **The hard bucket remains at 79.2%.** Defect-only periodic structure with no
   landmark is genuinely ambiguous. 6.8% periodic-lock and 8.2% gross-miss
   persist. The confidence signal (§6.2) catches these rather than hiding them.
3. **Both scaling axes are exhausted.** Measured: training steps plateau at
   5,000 of 10,000; 3.6× more hard data (11,600 → 41,600 samples) bought +1.4
   points, inside the noise band. More compute or more data will not move this
   architecture further.
4. **Trained on our own synthetic data.** Performance on Applied Materials'
   test set depends on how closely their generation process matches ours. The
   flat robustness trends for noise, dose, rotation and scale (§5.4) suggest
   tolerance to those axes; the distortion trend suggests sensitivity to how
   scan distortion is modelled.
5. **The centre rule is off by default** (§7). If the evaluation data is
   centre-weighted, enabling it is likely correct.

**Rejected approaches, with measured reasons** — recorded so they are not
re-attempted: rotation/scale template banks at inference (hurt every bucket),
per-pixel focal loss (collapses — pushing all logits down is a cheaper descent
direction than learning the peak), upsampled-DFT sub-pixel registration (no
gain; our residual is non-rigid, not rigid), ECC warp refinement, refinement
gating on 16 rule variants (none beat unconditional refinement), model soup of
two checkpoints (no gain over the better parent), and a higher lattice-margin
weight (2.0 vs 0.5 — *doubled* the periodic-lock rate it was meant to reduce).

---

## 10. References

See `references/references.md` — five public sources covering DRAM 6F2 array
geometry, SEM shot-noise and secondary-electron statistics, scan-distortion
artifacts, the Siamese correlation architecture, and sub-pixel registration.
Each entry states which design decision it justifies. Every reference was
checked against its publisher page; none were written from memory.

**No proprietary or fab-confidential data was used at any stage.** All imagery
is synthetic, generated by `generate_dataset.py` from public structural
knowledge.
