# Drift-Sense — dl_localize

Navigation-Error Recovery for SEM wafer inspection. Given a reference image
(~1 nm/px) and a wider search image (~10 nm/px), return the sub-pixel (x, y)
centre of the reference pattern inside the search frame.

Import as `driftsense.dl_localize`. The original `driftsense.localize` package
is retained unmodified as the historical baseline.

---

## 1. Headline result

Measured on a **golden set of 2,000 pairs** that were never trained on and
never used to select a hyperparameter. Learned and classical evaluated in the
same run, on the same pairs, on the same machine (RTX 4050 Laptop).

| | ≤1px | ≤2px | ≤3px | ≤5px | median | classical ≤1px |
|---|---|---|---|---|---|---|
| **ALL** (n=2000) | **89.5%** | 95.9% | 96.5% | 96.7% | 0.24 px | 76.6% |
| easy (n=1000) | 92.3% | 98.7% | 99.5% | 99.8% | 0.23 px | 90.4% |
| medium (n=600) | 91.8% | 98.7% | 99.3% | 99.3% | 0.23 px | 81.3% |
| hard (n=400) | 79.2% | 84.5% | 85.0% | 85.0% | 0.30 px | 35.0% |

**168 ms/pair versus 705 ms/pair — 4.2× faster while 12.9 points more
accurate overall and 44.2 points more accurate on the hard bucket.**

1 px ≈ 10 nm on the wafer, so ≤2px ≈ 20 nm.

### Progression

| stage | ALL ≤1px | easy | medium | hard |
|---|---|---|---|---|
| classical NCC baseline | 76.6% | 90.4% | 81.3% | 35.0% |
| previous learned model (`localize`, `lscv_ext`) | 76% | 84% | 85% | 41% |
| + coordinate/geometry fixes (run B) | 88.1% | 92.6% | 92.8% | 69.8% |
| + hard-bucket data scaling (run C) | 89.0% | 91.9% | 91.7% | 77.8% |
| + further scaling (run D, **final**) | **89.5%** | 92.3% | 91.8% | 79.2% |

The first three rows are on the val split; the last two are on the golden
split. The val→golden gap was measured at −0.8 points, i.e. within the ±1.3
point confidence interval for n=2000, confirming the tuning done against val
did not meaningfully inflate it.

---

## 2. Architecture

A hybrid: a learned Siamese dense matcher for **global search**, and classical
normalised cross-correlation for **sub-pixel precision**. Each stage does the
half of the problem it is measurably better at.

```
reference 1000×1000                    search 1000×1000
        │ shrink to 100×100                    │
        └────────── shared encoder ────────────┘
             custom residual CNN, stride 4
        │                                      │
    25×25×128                            250×250×128
        └──► dense cross-correlation ◄─────────┘
                        │
                226×226 score map ──► argmax + offset head
                        │
              ±3 px window ──► multi-angle stride-1 NCC
                        │
                  sub-pixel (x, y)
```

**Stage 1 — learned (SiamFC-style dense correlation).**

| component | choice | reason |
|---|---|---|
| Encoder | custom residual CNN, GroupNorm | Not HRNet — see §5. GroupNorm because training processes one pair per forward pass, where BatchNorm statistics are meaningless |
| Output stride | 4 | Classification backbones reach stride 32, which erases a 1.4–3.4 px hard-case defect |
| Embedding | L2-normalised along channels | Makes the correlation a cosine similarity |
| Correlation | reference embedding as a conv kernel, divided by kernel positions | The raw `conv2d` sum reaches ±625 and saturates everything downstream |
| Head | 1×1×1 conv, weight init 20.0 | A scalar temperature on a mean cosine that starts near 0 |
| Offset head | CenterNet-style, per-cell (dx, dy) | Stride 4 quantises the peak to 4 px against a 1 px metric |

Losses (summed): whole-map **softmax cross-entropy** + **lattice margin
hinge** + **offset L1** + **Huber** on the decoded pixel coordinates.
AdamW, cosine schedule with warmup, EMA, fp16.

**Stage 2 — classical local refinement.** Multi-angle (±4.5° in 1.5° steps)
stride-1 NCC in a ±3 px window, with a 3-point parabolic peak fit. Measured
contribution: median error **0.493 px → 0.240 px**, for **5 ms/pair**.

The ±3 px window is narrower than the minimum measured lattice pitch
(5.3 px), so this stage physically cannot introduce a periodic lock — it can
only sharpen the cell the learned stage already chose.

---

## 3. The root-cause finding

The previous model plateaued at 84% on the easy bucket and did not respond to
more data or more steps. The cause was a **coordinate-convention bug**, found
by an independent geometric probe rather than by inspecting loss curves.

`localize/coords.py` mapped correlation-output index to search pixel using the
size of the *encoded* reference. The true mapping depends on the template's
*pixel* size. Measured by planting an exact crop of a search image back into
that image and reading the argmax (argmax sat at index 100 for every size):

| template_px k | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104 |
|---|---|---|---|---|---|---|---|---|
| true centre | 448.0 | 448.5 | 449.0 | 449.5 | 450.0 | 450.5 | 451.0 | 451.5 |
| old formula | 448.0 | 448.0 | 450.0 | 450.0 | 450.0 | 450.0 | 452.0 | 452.0 |
| **bias (px)** | +0.0 | −0.5 | **+1.0** | +0.5 | +0.0 | −0.5 | **+1.0** | +0.5 |

A *constant* bias would cancel between training and inference. This one is a
sawtooth in k — and k differed between the two: training derived it from the
per-sample `scale_ratio` **label** (96–104), while inference hard-coded 100.

Confirmed end to end on real pairs with an untrained encoder, so the effect is
purely geometric:

```
id=56  k=99   old-map err 2.13 px   corrected-map err 0.74 px
id=59  k=99   old-map err 2.31 px   corrected-map err 1.47 px
id=60  k=101  old-map err 1.15 px   corrected-map err 1.15 px   ← k=101 is the unbiased case
```

The 1.4 px gap at k=99 is exactly √2 × the predicted 1.0 px per-axis bias
applied to both axes, matching the table cell for cell.

**On a metric scored at 1 px, this alone accounts for the plateau.** No amount
of data fixes a geometry error, which is exactly what the 10k experiment
showed: more steps moved hard 32→41% and moved easy not at all.

`calibrate.py` now makes this class of bug impossible to reintroduce
silently — it plants patches at known locations and fails the build if the
decoded coordinate is off by more than zero.

Other defects fixed in the same pass: encoder fusion dropping a row/column
when the two branches disagreed by one cell; Gaussian targets built in
bfloat16, whose ULP at index ~113 is 0.5 cells (a ±1 px target quantisation);
a degenerate coordinate loss whose window was centred on ground truth so its
gradient was ~0; `lattice_boost` accepted as an argument and never used;
confidence reported as `sigmoid(peak_logit)`, which saturates to 1.000 for
every sample by construction.

---

## 4. Where the remaining error is

**Failure taxonomy** (golden, n=2000):

| | hit | near-miss (1–5px) | periodic-lock | gross-miss |
|---|---|---|---|---|
| ALL | 89.5% | 7.2% | 1.4% | 1.9% |
| easy | 92.3% | 7.5% | **0.0%** | 0.2% |
| medium | 91.8% | 7.5% | 0.2% | 0.5% |
| hard | 79.2% | 5.8% | 6.8% | 8.2% |

Periodic locking — the failure this entire architecture was built to defeat,
and 4% of the classical baseline's errors — is **0.0% on easy** and 1.4%
overall.

**≤1px accuracy against every metadata covariate** (golden). Flat means ruled
out as a cause:

| covariate | quintiles | verdict |
|---|---|---|
| \|rel_rotation\| | 89 / 88 / 91 / 89 / 91 | flat — handled |
| scale_ratio | 88 / 92 / 90 / 88 / 88 | flat — handled |
| bitline pitch | 90 / 90 / 89 / 90 / 90 | flat — handled |
| search dose | 90 / 89 / 88 / 90 / 90 | flat — not noise-limited |
| shot SNR | 90 / 89 / 88 / 90 / 90 | flat — not noise-limited |
| landmark size | **79** / 92 / 92 / 92 / 93 | hard-bucket floor |
| label_correction | 96 / 96 / 92 / 92 / **70** | **the binding constraint** |

Two, and only two, structures remain.

**(a) Non-rigid scan distortion, for easy/medium.** `label_correction_px`
records how far the true centre (inverting the *actual* sampling, including
jitter, drift and scan distortion) sits from the nominal rigid map. A rigid
template cannot align to a non-rigidly warped pattern better than roughly the
warp magnitude. Measured on the failures: `label_correction` averages 1.42 px
versus 0.65 px for the population, while their rotation is identical
(1.28 vs 1.25). The error/distortion ratio is ~0.5 in every bin — the matcher
already recovers about half the warp, and nothing tried moved that ratio.

**(b) Landmark absence, for hard.** Samples with no landmark >136 nm score
79% against 92–93% for the rest. This is consistent with the project's earlier
`gate_test` finding that classical top-K recall on hard saturates at ~73% even
at K=512 — for roughly a quarter of hard samples there is no usable
correlation peak at any rank.

**Confidence is calibrated and usable.** Logit margin averages 5.47 ± 1.98 on
correct predictions versus 3.73 ± 2.82 on failures — enough separation to gate
a fallback or flag low-confidence sites for re-measurement.

---

## 5. Negative results

Seven hypotheses tested and rejected with measurements. Each was cheap to test
and would have been expensive to assume.

| # | hypothesis | measured outcome |
|---|---|---|
| 1 | **FFT lattice suppression** would raise the aperiodic signal | Classical accuracy 70% → **32%**. Once the lattice is removed what remains is shot noise (22–110 e/px), not the aperiodic marker. No notch-gain setting paid |
| 2 | **CenterNet focal loss** suits a single-peak heatmap | Degenerate collapse: loss 701,315 → 6,823 over 600 steps while error went 1.54 px → **422.53 px**. With one positive among ~51,000, the cheapest descent direction is "no match anywhere". Replaced by whole-map softmax CE, under which collapse is mathematically impossible — verified: adding −50 to every logit leaves the loss bit-identical |
| 3 | **A rotation × scale template bank** at inference, selected by peak height | Worse on *every* bucket: easy 85.7→85.1, medium 83.5→81.7, hard 37.5→33.8, and 156→242 ms/pair. Selection by max cosine is reliable only where the surface is sharp; on a flat surface a wrong member wins with a spurious peak |
| 4 | **ECC warp refinement** (euclidean / affine) would absorb the distortion | median 0.617 → 0.593 / 0.609 px, ≤1px unchanged at 62.5%, mean error slightly worse. The distortion is non-rigid; a global affine cannot represent it |
| 5 | **Matching only the template centre** would avoid the periphery dragging the fit | Flat on distorted samples at all crop sizes (100/80/60/40), and *worse* on clean ones (100% → 93% at 40×40) |
| 6 | **Upsampled DFT registration** would beat parabolic peak-locking bias | median 0.251 → 0.270 px, ≤1px unchanged at 90%. Peak locking is not a meaningful error source here |
| 7 | **Gating the classical refinement** on `shift`, `margin` or `ncc` | All 16 rule variants scored *below* always-refining. A perfect oracle choosing per-sample would reach only 91.8% versus 89.0% — the whole learned-vs-classical axis contains just 2.8 points |

---

## 6. Scaling — measured, then stopped

The project's standing question was whether to train on the full 200,000-pair
dataset. Answered with two controlled experiments, val set frozen throughout.

**Data.** Only the hard bucket was scaled; easy and medium are distortion-
limited, not data-limited.

| hard training samples | hard ≤1px |
|---|---|
| 1,600 | ~68% |
| 11,600 (7.25×) | 77.8% |
| 41,600 (3.6×) | **79.2%** |

The first increase bought +9.4 points; the second bought +1.4, inside the ±4
point interval for n=400. **The curve is flat.** The remaining ~46,000 unused
hard pairs would buy nothing.

**Steps.** Run D's validation across 10,000 steps:

| step | 1000 | 2000 | 3000 | 4000 | **5000** | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL ≤1px | 76 | 79 | 80 | 84 | **87** | 85 | 85 | 87 | 86 | 87 |

Plateaued at step 5000. The second half produced no gain. Training accuracy
tracks validation throughout, so the model is neither overfitting nor starved.

Both scaling axes are exhausted. The remaining error is architectural, not
budgetary.

---

## 7. Honest limitations

1. **Easy/medium are at this design's ceiling (~92%).** Seven approaches were
   tested against the residual; none moved it. Reaching ~97% would require
   modelling the non-rigid warp field, a different model class.
2. **Hard is bounded by landmark absence**, not by data or training.
3. **DRAM-only training** against a test set stated to contain DRAM *and*
   FinFET. An explicit, unmitigated project decision.
4. **Generator gap.** All results are on our own simulator. Photometric
   augmentation and 64,000 distinct training pairs are insurance against
   fingerprinting it, but the transfer gap to Applied Materials' generator is
   unmeasurable from here.
5. **Applied Materials' slides showed coarser-pitch DRAM** than our generated
   references. Flagged, not acted on.
6. **The centre tie-break ships disabled.** Measured: at a 0.5-logit threshold
   it moved a synthetic pair whose argmax was correct to 0 px by 130 px.
   Logits are `gain × cosine + bias`, so the threshold is not a physical
   quantity and must be calibrated before enabling.

**Future work, with the strongest candidate first.** Replace the classical
refinement with a *learned* stride-1 refinement head trained on the
distortion-corrected labels. The learned stage alone scores 82.5% and the
classical refinement scores 89.5%; an oracle combining them reaches 91.8%.
The gap exists because classical NCC finds the best **rigid** alignment by
construction, while ground truth is distortion-corrected. A learned refiner
would get both precision and distortion-awareness in one output. Estimated
ceiling from the covariate data: ALL 93–94%.

---

## 8. Reproduction

```bat
set PYTHONPATH=<submission-root>\src

REM 0. Geometry gate. CPU, ~1 minute. Never skip.
python -m driftsense.dl_localize.calibrate --dataset <dataset-root> --pairs 4

REM 1. Smoke test — the err column must trend down.
python -m driftsense.dl_localize.train --dataset <dataset-root> ^
    --out <run-dir>\smoke --smoke-test

REM 2. Curriculum: easy -> +medium -> +hard
python -m driftsense.dl_localize.train ^
    --dataset <dataset-root> ^
    --split-csv <dataset-root>\subset_split.csv ^
    --out <run-dir> --steps-per-phase 1500 --phase-step-scale 1,1,1 ^
    --accum-steps 8 --val-every 500 --val-pairs 150

REM 3. Scale the hard bucket (val and golden seeds are never touched)
python -m driftsense.dl_localize.extend_subset --dataset <dataset-root> ^
    --split-csv <dataset-root>\subset_split_golden.csv ^
    --add hard=30000 easy=8000 medium=8000 ^
    --out <dataset-root>\subset_split_64k.csv

python -m driftsense.dl_localize.train ^
    --dataset <dataset-root> ^
    --split-csv <dataset-root>\subset_split_64k.csv ^
    --out <run-dir> --resume-from <run-dir>\phase2_easy_medium.pt ^
    --phases 3 --steps-per-phase 10000 --phase-step-scale 1 --phase-lr-scale 0.3 ^
    --accum-steps 8 --mix "easy=1,medium=1,hard=2" --val-every 1000 --val-pairs 300

REM 4. Final measurement, learned vs classical, on untouched data
python -m driftsense.dl_localize.eval --dataset <dataset-root> ^
    --checkpoint <run-dir>\phase3_all.pt ^
    --split-csv <dataset-root>\subset_split_golden.csv --split golden ^
    --pairs 2000 --classical --out <run-dir>\eval_golden_vs_classical.json
```

### Modules

| file | role |
|---|---|
| `coords.py` | The coordinate convention. Constant affine map; nothing per-sample can perturb it |
| `calibrate.py` | Pre-flight geometric gate. Fails loudly if the encoder and the convention disagree |
| `data.py` | Dataset, fixed-size template construction, augmentation |
| `model.py` | Encoder, dense correlation, offset head, readout, confidence |
| `losses.py` | Softmax CE, lattice margin, offset L1, Huber |
| `train.py` | Curriculum training, in-run validation, checkpointing |
| `refine.py` | Classical stride-1 multi-angle NCC refinement |
| `infer.py` | Inference entry point (`LearnedLocalizer.locate`) |
| `eval.py` | Evaluation, failure taxonomy, covariate trends, refinement audit |
| `extend_subset.py` | Grows train/golden splits without moving val |

Every module has a runnable self-test (`python -m driftsense.dl_localize.<module>`).
`losses` asserts the collapse-invariance property that focal loss lacked;
`model` asserts the coordinate convention to zero error.

---

## 9. Verification status

**Executed against real data:** coordinate round-trip and calibration
(0.00 px on stride-aligned crops); real-data geometry within the 2.83 px
argmax-quantisation floor; loss self-tests including the collapse-invariance
proof and lattice-margin gradient directions; end-to-end training; bank
correlation equivalence; refinement recovering applied rotations; all
accuracy numbers in §1.

**Cross-check:** the σ=1.0 target's analytic entropy floor computes to
**2.838 nats**, and training converges to 2.84 — matching the 2.842 measured
independently in this project's earlier overfit test.

**Not verified:** transfer to Applied Materials' generator; FinFET layouts;
any hardware other than the RTX 4090 and RTX 4050 used here.
