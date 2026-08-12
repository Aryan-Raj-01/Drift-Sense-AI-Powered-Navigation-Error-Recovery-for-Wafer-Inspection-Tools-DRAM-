# References

Public sources supporting the synthetic-structure design, SEM image-formation
and noise model, the geometric transformations applied during augmentation, and
the localization method. Every source below is publicly accessible; no
proprietary or fab-confidential material was used at any stage.

Each entry states **what it justifies in this submission** rather than listing
literature for its own sake.

---

## 1. Semiconductor structure — DRAM array geometry

**T. Schloesser, F. Jakubowski, J. v. Kluge, A. P. Graham, S. Slesazeck,
M. Popp, P. Baars, K. Muemmler, P. Moll, K. Wilson, A. Buerke, D. Koehler,
J. Radecker, E. Erben, U. Zimmermann, T. Vorrath, B. Fischer, G. Aichmayr,
R. Agaiby, W. Pamler, T. Schuster, W. Bergner, W. Mueller.**
"6F2 buried wordline DRAM cell for 40nm and beyond."
*IEEE International Electron Devices Meeting (IEDM)*, 2008.
<https://ieeexplore.ieee.org/abstract/document/4796820>

**Justifies:** the DRAM-style layout generator. The 6F2 cell with buried
wordline is the architecture family our synthetic arrays reproduce: two
orthogonal periodic interconnect sets (bitlines and wordlines) at a fixed
half-pitch F, with the array transistor and storage node repeating on that
lattice. The paper reports a 46 nm technology with a 0.013 µm² cell, which
anchors the physical scale of our generated pitches to a real published node
rather than to an arbitrary choice.

**Specifically used for:**
- Orthogonal bitline/wordline periodicity as the dominant image structure.
- Bitline and wordline pitch as independent, sampled parameters
  (`layout_bl_pitch`, `layout_wl_pitch` in the manifest).
- The rationale that a reference patch is genuinely ambiguous under
  translation by an integer number of pitches — the *periodic-lock* failure
  mode this entire project is built around.

---

## 2. SEM image formation — shot noise and secondary-electron statistics

**"Effect of Shot Noise and Secondary Emission Noise in Scanning Electron
Microscope Images."** *Scanning* (Wiley).
<https://onlinelibrary.wiley.com/doi/epdf/10.1002/sca.4950260106>

**"Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring."**
HAL open archive.
<https://hal.science/hal-01051309/document>

**Justifies:** the noise model. Both sources establish that the dominant
degradation in SEM imaging is Poisson-distributed shot noise arising from
random fluctuation in primary and secondary electron emission, and that the
several distinct noise sources in the detection chain (primary emission,
secondary emission, scintillator, photocathode, photomultiplier) each follow
quantum/Poisson statistics.

**Specifically used for:**
- Applying **Poisson** noise as a function of electron dose, rather than
  additive Gaussian noise. Dose is recorded per pair as
  `search_dose_e_per_px`, and the resulting `search_shot_snr` follows the
  √N relationship this literature describes.
- Sampling the search image at a *lower* dose than the reference, reflecting
  that a fast wide-field 10x scan collects fewer electrons per pixel than a
  slow 100x close-up — the physical reason the search image is the noisier of
  the two.
- Our measured SNR range of roughly 3.3–7.4 is deliberately in the
  low-SNR regime these papers identify as characteristic of CD-SEM metrology.

**Additionally consulted:**
"Shot noise-mitigated secondary electron imaging with ion count-aided
microscopy," *PNAS*, 2024. <https://www.pnas.org/doi/10.1073/pnas.2401246121>
— models incident particles and emitted secondary electrons as Poisson with a
Gaussian per-electron detector response, which is the same two-stage structure
our detector chain implements.

---

## 3. Geometric transformations — drift and scan distortion

**Correction of Scanning Electron Microscope Imaging Artifacts in a Novel
Digital Image Correlation Framework.** *Experimental Mechanics*, Springer.
<https://link.springer.com/article/10.1007/s11340-018-00469-w>

**Justifies:** the drift/jitter and non-rigid distortion augmentations, and —
importantly — an honest account of our own accuracy ceiling.

This work identifies three dominant SEM artifact classes:

| artifact | random? | time-dependent? | our model |
|---|---|---|---|
| spatial distortion | non-random | no | static warp field |
| drift distortion | non-random | yes | slow low-order drift |
| scan line shifts | random | yes | per-line jitter |

**Specifically used for:**
- Modelling drift as a *non-random, time-dependent* low-order deformation
  rather than as a pure rigid translation.
- Modelling scan-line shift as *random* per-line displacement.
- The `label_correction_px` field in our manifest, which records how far the
  true pattern centre moves under the non-rigid component. This is not a
  cosmetic detail: our measured accuracy falls from 96% (≤1px) in the lowest
  distortion quintile to 70% in the highest, which is the single strongest
  covariate in the entire evaluation. The literature above is why we expected
  a non-rigid term to exist at all, and why we instrument it rather than
  treating those samples as unexplained failures.

---

## 4. Localization method — dense correlation via a Siamese network

**L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, P. H. S. Torr.**
"Fully-Convolutional Siamese Networks for Object Tracking."
*ECCV 2016 Workshops*, pp. 850–865.
<https://github.com/bertinetto/siamese-fc>

**Justifies:** the architecture of the learned stage. SiamFC embeds an
exemplar and a search region with a shared fully-convolutional encoder and
cross-correlates the two embeddings to produce a dense response map, taking
the peak as the target location.

**Specifically used for:**
- The shared-encoder + dense cross-correlation design, applied here to
  cross-magnification localization instead of video tracking.
- Correlation implemented as a convolution with the reference embedding as the
  kernel, normalized by kernel position count.
- The known consequence that the response map is quantized to the encoder
  stride — ours is stride 4 — which is precisely why a sub-pixel stage is
  required rather than optional.

---

## 5. Sub-pixel refinement

**M. Guizar-Sicairos, S. T. Thurman, J. R. Fienup.**
"Efficient subpixel image registration algorithms."
*Optics Letters*, vol. 33, no. 2, pp. 156–158, 2008.
<https://opg.optica.org/ol/abstract.cfm?uri=ol-33-2-156> (PMID 18197224)

**Justifies:** the sub-pixel stage, and one of our documented negative results.

**Specifically used for:**
- Establishing that translation registration to a small fraction of a pixel is
  achievable by correlation-based methods, which sets the target for our second
  stage (measured: median error 0.493 px → 0.240 px).
- We implemented and **tested** the upsampled-DFT approach this paper
  describes and measured it as *not* helping on our data, because our residual
  error is dominated by non-rigid distortion (§3) rather than by rigid
  sub-pixel translation. We report this as a negative result rather than
  omitting it. The method our system ships instead is stride-1 multi-angle
  NCC with a 3-point parabolic fit over a ±3 px window, chosen on measured
  performance.

---

## Citation integrity

Every reference above was located and checked against its publisher or archive
page at the URL given. No reference in this document was generated from memory
without verification, and no citation has been included that we did not
actually use in a design decision. Where a source informed a choice we later
*rejected* on measured evidence (§5), that is stated explicitly rather than
being presented as an endorsement.
