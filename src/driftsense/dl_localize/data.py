"""Dataset for the fixed dense-correlation model.

Reads the same ``labels.jsonl`` the old pipeline used -- no regeneration.

WHAT CHANGED AND WHY
====================

1. THE TEMPLATE IS ALWAYS ``TEMPLATE_PX`` PIXELS.  The old ``data.py`` did::

       scale = float(r["scale_ratio"])          # <- a ground-truth LABEL
       k = shrink_size(ref_img.shape[0], scale) # -> 96..104, varies per sample

   while ``infer_dl.locate`` hard-coded ``scale=10.0`` -> k=100 for every
   sample.  Two separate failures came out of that single line:

     (a) Coordinate-convention mismatch.  The old ``out_to_pixel`` keyed off
         the embedding size, which is a step function of k, so the decoded
         pixel differed between training (k from the label) and inference
         (k=100) by 0 to 1.0 px per axis.  Measured; see coords.py.
     (b) Distribution mismatch.  The network was trained on a template that
         was always at exactly the right scale, and evaluated on one that
         could be 4% wrong.  More training makes this worse, not better --
         which is the observed signature: easy sat at 84% across Run 1 and
         Run 2 despite 1500 extra phase-3 steps.

   ``scale_ratio`` is an annotation.  Applied Materials' held-out set will
   not ship one.  It is used here only to steer augmentation, never to set a
   tensor shape.

2. ROTATION IS NOW REPRESENTED AT ALL.  ``rel_rotation_deg`` has sigma 1.6
   deg and abs max 5.83 deg in the manifest.  The classical ``propose.py``
   searches five angles over +/-3 deg and reaches 89% on easy; the learned
   path did a single un-rotated correlation and reached 84%.  A rigid
   correlation kernel has no way to absorb a 3 deg rotation of a 100 px
   template -- the corners move ~3.7 px, which is 4x the scoring tolerance.
   The template is now built through a rotation, and inference searches a
   bank of angles (``infer``).

3. ONE WARP, NOT THREE RESAMPLINGS.  Rotation, scaling and the crop to
   ``TEMPLATE_PX`` happen as: area-resize (correct anti-aliasing for a ~10x
   downscale, which warpAffine alone does not provide) followed by a single
   affine that rotates about the reference centre and lands that centre
   exactly on the template centre.  An integer centre-crop would shift the
   centre by 0.5 px whenever ``m - TEMPLATE_PX`` is odd -- reintroducing a
   smaller copy of the bug this module exists to remove.

4. THE FRACTIONAL OFFSET TARGET is returned alongside the cell index, for
   the sub-pixel offset head.  Stride 4 quantises the peak to 4 px; the
   measured median error of 0.55 px is consistent with interpolation error,
   and on a 1 px metric that is expensive.

Self-test (needs torch + real data)::

    python -m driftsense.dl_localize.data --dataset <dataset-root>
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from driftsense.dl_localize.coords import STRIDE, TEMPLATE_PX, pixel_to_out

#: Nominal demagnification.  Used only as the centre of the augmentation and
#: inference-bank ranges, never read from a label at inference time.
NOMINAL_SCALE: float = 10.0

#: The only manifest fields anything downstream actually reads.  See the
#: comment in ``DriftSensePairs.__init__`` for why the other ~79 are
#: dropped at load time rather than carried around.
_KEEP_FIELDS = (
    "id", "seed", "difficulty", "reference_path", "search_path",
    "gt_x", "gt_y", "scale_ratio", "rel_rotation_deg", "search_px_nm",
    "layout_bl_pitch", "layout_wl_pitch", "search_dose_e_per_px",
    "search_shot_snr", "landmark_size_nm", "defect_size_nm",
)


def _resolve(root: Path, path: str) -> str:
    return path if os.path.isabs(path) else str(root / path)


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """uint8 grayscale -> (1, H, W) float32, roughly zero-mean unit-ish.

    Per-image standardisation, not a fixed affine.  Search frames span
    22-110 e/px of dose and the manifest carries independent contrast,
    brightness and gamma jitter for reference and search, so a fixed
    ``(x/255 - 0.5) / 0.25`` (what the old ``data._normalize`` did) leaves a
    brightness/contrast gap between the two branches of a Siamese network
    whose entire output is a cosine similarity between them.  Standardising
    each image removes that gap for free.
    """
    a = img.astype(np.float32)
    mean = float(a.mean())
    std = float(a.std())
    if std < 1e-6:
        std = 1.0
    return torch.from_numpy((a - mean) / std).unsqueeze(0)


def build_template(reference: np.ndarray, scale: float, angle_deg: float,
                   template_px: int = TEMPLATE_PX) -> np.ndarray:
    """Reference image -> fixed-size template, rotated and scaled.

    The output is always ``(template_px, template_px)``, and the reference's
    geometric centre lands exactly on the template's geometric centre
    ``((template_px - 1) / 2, (template_px - 1) / 2)``.  That exactness is
    what makes ``coords.out_to_pixel`` a constant, sample-independent map.

    Args:
        reference: (N, N) grayscale reference, full resolution.
        scale: Demagnification to apply (search_px_nm / ref_px_nm), ~10.
        angle_deg: Rotation applied to the template, degrees, positive
            counter-clockwise in image coordinates.  To match a reference
            that is rotated by ``rel_rotation_deg`` relative to the search
            frame, pass ``-rel_rotation_deg``.
        template_px: Output side length.

    Returns:
        (template_px, template_px) array, same dtype family as the input,
        as float32.
    """
    n = reference.shape[0]
    m = max(8, int(round(n / float(scale))))

    # Step 1: anti-aliased downscale.  INTER_AREA is a box filter, which is
    # the only interpolation in OpenCV that does not alias at a 10x
    # reduction.  A fine DRAM pitch (53-227 nm, i.e. 5-23 search px) aliases
    # badly under INTER_LINEAR at this ratio and the aliased pattern is
    # phase-dependent -- it would look like real positional signal.
    interp = cv2.INTER_AREA if m < n else cv2.INTER_LINEAR
    small = cv2.resize(reference.astype(np.float32), (m, m),
                       interpolation=interp)

    # Step 2: rotate about the small image's centre and translate that centre
    # onto the template centre, in one affine.  Sub-pixel exact for any m,
    # odd or even, which an integer crop is not.
    src_c = (m - 1) / 2.0
    dst_c = (template_px - 1) / 2.0
    mat = cv2.getRotationMatrix2D((src_c, src_c), float(angle_deg), 1.0)
    mat[0, 2] += dst_c - src_c
    mat[1, 2] += dst_c - src_c

    # BORDER_REFLECT_101 rather than a constant: at scale 10.4 the resized
    # reference is 96 px and the template is 100, so up to 2 px per side is
    # invented, plus ~3 px in the corners at 4 deg of rotation.  A constant
    # border would create a hard artificial edge -- a strong, position-locked
    # feature the encoder would happily learn.  Reflection at least matches
    # the local texture statistics.  Train and inference use the identical
    # code path, so whatever bias remains is common-mode.
    return cv2.warpAffine(small, mat, (template_px, template_px),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def photometric_jitter(img: np.ndarray, rng: np.random.Generator,
                       strength: float = 1.0) -> np.ndarray:
    """Mild gain/gamma/noise jitter on the template only.

    Insurance against the model keying on our own generator's fingerprint
    (the "channel ablation" item flagged repeatedly and never implemented).
    The held-out test set comes from Applied Materials' generator, not ours;
    anything the model learns that is specific to our contrast/gamma/noise
    model is worse than useless there.

    Applied to the TEMPLATE only, never the search image.  The task is a
    similarity between the two branches, so perturbing one of them is what
    forces the encoder to represent structure rather than intensity.
    """
    if strength <= 0.0:
        return img
    a = img.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return a
    u = (a - lo) / (hi - lo)
    gamma = float(np.exp(rng.normal(0.0, 0.18 * strength)))
    u = np.clip(u, 0.0, 1.0) ** gamma
    gain = 1.0 + float(rng.normal(0.0, 0.15 * strength))
    bias = float(rng.normal(0.0, 0.06 * strength))
    u = u * gain + bias
    u = u + rng.normal(0.0, 0.02 * strength, size=u.shape).astype(np.float32)
    if rng.random() < 0.05 * strength:      # polarity flip; the manifest has
        u = 1.0 - u                         # ref_invert / search_invert flags
    return (u * (hi - lo) + lo).astype(np.float32)


class DriftSensePairs(Dataset):
    """One reference/search pair per item, plus everything the losses need.

    Args:
        root: Dataset directory containing ``labels.jsonl``.
        difficulties: Which difficulty buckets to include.
        seed_filter: Keep only rows whose ``seed`` is in this set.  Used for
            the train/val separation.  Pass ``None`` only when you do not
            care that val may leak.
        augment: Enable geometric/photometric augmentation.  Off for
            evaluation.
        aug_rot_deg: Half-width of the uniform template-rotation jitter, in
            degrees, applied AROUND the nominal 0 deg that inference uses.
            The sample's own ``rel_rotation_deg`` already supplies the real
            rotation; this only widens the margin.  Keep it small -- larger
            values recreate the train/inference distribution gap documented
            in ``__getitem__``.
        aug_scale: Half-width of the uniform scale jitter around
            ``NOMINAL_SCALE``.  The measured <=1px-vs-scale_ratio trend is
            FLAT (75/76/72/75/72 across quintiles), so scale is not currently
            a limiting factor and this is insurance, not a fix.
        jitter_strength: Photometric jitter scale, 0 disables.
        seed: RNG seed for augmentation.
    """

    #: Retained only so old checkpoints record something sensible.  The
    #: inference bank is a SINGLE member -- see infer and the measurement
    #: quoted in ``__getitem__``.
    ANGLE_BANK: Tuple[float, ...] = (0.0,)
    SCALE_BANK: Tuple[float, ...] = (NOMINAL_SCALE,)

    def __init__(self, root: str, difficulties: Sequence[str] = ("easy",),
                 seed_filter: Optional[set] = None, augment: bool = True,
                 aug_rot_deg: float = 1.0, aug_scale: float = 0.1,
                 jitter_strength: float = 1.0, seed: int = 88) -> None:
        self.root = Path(root)
        self.augment = augment
        self.aug_rot_deg = float(aug_rot_deg)
        self.aug_scale = float(aug_scale)
        self.jitter_strength = float(jitter_strength)
        self._seed = int(seed)

        want = set(difficulties)
        rows: List[Dict] = []
        with open(self.root / "labels.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["difficulty"] not in want:
                    continue
                if seed_filter is not None and int(r["seed"]) not in seed_filter:
                    continue
                # Keep only the ~16 fields anything downstream reads.  The
                # manifest has 95 fields per row; retaining all of them costs
                # roughly 6x the memory for no benefit, and that cost is paid
                # once per DataLoader worker process because the dataset is
                # pickled to each of them.  On an 8000-row pool with 2 workers
                # that is the difference between ~40 MB and ~250 MB of
                # resident memory before a single image is read.
                rows.append({k: r.get(k) for k in _KEEP_FIELDS})
        if not rows:
            raise ValueError(
                f"No rows matched difficulties={tuple(difficulties)} in {root}"
                + (" under the given seed_filter" if seed_filter else "")
                + ". If a split CSV was passed, check that its 'seed' column "
                  "survived as int64 -- seeds here reach 4.6e18 and a float64 "
                  "round-trip silently corrupts them.")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _nearest(bank: Sequence[float], value: float) -> float:
        return min(bank, key=lambda b: abs(b - value))

    def __getitem__(self, idx: int) -> Dict:
        # A transient MemoryError while decoding a 1000x1000 PNG should not
        # destroy a multi-hour run. Observed in practice: an allocation of
        # 3.81 MB failed at step ~1000 of a run whose identical predecessor
        # completed 10,000 steps, i.e. external memory pressure, not a leak.
        # Collect and retry before giving up; only a genuinely exhausted
        # machine will fail twice.
        for attempt in range(3):
            try:
                return self._load(idx)
            except MemoryError:
                import gc
                gc.collect()
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise MemoryError("unreachable")

    def _load(self, idx: int) -> Dict:
        r = self.rows[idx]
        # Deterministic per (epoch-independent) index so a worker restart
        # does not silently change the augmentation distribution mid-run.
        rng = np.random.default_rng((self._seed * 1000003 + idx) % (2 ** 63))

        ref_img = cv2.imread(_resolve(self.root, r["reference_path"]),
                             cv2.IMREAD_GRAYSCALE)
        srch_img = cv2.imread(_resolve(self.root, r["search_path"]),
                              cv2.IMREAD_GRAYSCALE)
        if ref_img is None or srch_img is None:
            raise FileNotFoundError(
                f"Could not read images for id={r['id']} under {self.root}")

        true_angle = -float(r.get("rel_rotation_deg", 0.0))
        true_scale = float(r.get("scale_ratio", NOMINAL_SCALE))

        # AUGMENTATION IS CENTRED ON THE INFERENCE OPERATING POINT.
        #
        # infer uses a single template at (angle 0, scale NOMINAL_SCALE),
        # because a rotation x scale bank selected by max peak logit was
        # MEASURED to hurt: bank-15 vs bank-1 on 2000 val pairs gave
        # easy 85.1 vs 85.7, medium 81.7 vs 83.5, hard 33.8 vs 37.5.  On a
        # flat correlation surface a wrong bank member wins with a spurious
        # peak, so selection is noise exactly where the model is weakest.
        #
        # Once the bank is gone, the OLD bank-quantised augmentation becomes
        # a train/inference mismatch in its own right -- the same class of
        # bug as the coordinate convention.  Simulated over 200k draws from
        # the measured rotation distribution (sigma 1.6 deg, max 5.83):
        #
        #   residual rotation seen   mean   p90    max
        #   inference (angle 0)      1.28   2.63   5.83
        #   train, snapped half      0.50   0.90   1.83   <- never happens
        #   train, random half       2.72   5.20   9.83   <- never happens
        #   train, actual mix        1.61   4.32   9.83
        #
        # Leaving the template at nominal makes the residual the sample's own
        # rel_rotation -- exactly the inference distribution.  A small
        # CONTINUOUS jitter on top buys robustness margin for Applied
        # Materials' unknown generator without distorting the distribution
        # the way bank quantisation did.
        if not self.augment:
            angle, scale = 0.0, NOMINAL_SCALE
        else:
            angle = float(rng.uniform(-self.aug_rot_deg, self.aug_rot_deg))
            scale = NOMINAL_SCALE + float(
                rng.uniform(-self.aug_scale, self.aug_scale))

        tmpl = build_template(ref_img, scale, angle)
        if self.augment and self.jitter_strength > 0:
            tmpl = photometric_jitter(tmpl, rng, self.jitter_strength)

        gt_x, gt_y = float(r["gt_x"]), float(r["gt_y"])
        gt_out_x = pixel_to_out(gt_x)
        gt_out_y = pixel_to_out(gt_y)
        cell_x = int(round(gt_out_x))
        cell_y = int(round(gt_out_y))

        # Lattice pitch in correlation-grid cells.  bl_pitch spans
        # 53-227 nm and search_px_nm is ~10, so the pitch is 1.3-5.7 CELLS.
        # At the low end two adjacent lattice replicas are barely more than
        # one cell apart -- which is exactly why losses refuses to place a
        # hard negative closer than a minimum separation, and why the
        # sub-pixel readout window must be small (see model).
        px_nm = float(r.get("search_px_nm", 0.0))
        pitch_x_px = (float(r.get("layout_bl_pitch", 0.0)) / px_nm
                      if px_nm > 0 else 0.0)
        pitch_y_px = (float(r.get("layout_wl_pitch", 0.0)) / px_nm
                      if px_nm > 0 else 0.0)

        return {
            "id": int(r["id"]),
            "seed": int(r["seed"]),
            "difficulty": r["difficulty"],
            "ref": torch.from_numpy(
                np.ascontiguousarray(
                    (tmpl - tmpl.mean()) / (tmpl.std() + 1e-6),
                    dtype=np.float32)).unsqueeze(0),
            "search": normalize_image(srch_img),
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_out_x": gt_out_x,
            "gt_out_y": gt_out_y,
            "cell_x": cell_x,
            "cell_y": cell_y,
            "frac_x": gt_out_x - cell_x,      # in [-0.5, 0.5], offset-head target
            "frac_y": gt_out_y - cell_y,
            "pitch_x_out": pitch_x_px / STRIDE,
            "pitch_y_out": pitch_y_px / STRIDE,
            "tmpl_angle": angle,
            "tmpl_scale": scale,
            "true_angle": true_angle,
            "true_scale": true_scale,
            "rel_rotation_deg": float(r.get("rel_rotation_deg", 0.0)),
            "search_dose": float(r.get("search_dose_e_per_px", 0.0)),
            "search_snr": float(r.get("search_shot_snr", 0.0)),
            "landmark_size_nm": float(r.get("landmark_size_nm") or 0.0),
            "defect_size_nm": float(r.get("defect_size_nm") or 0.0),
        }


def load_split_seeds(split_csv: str, split_name: str) -> set:
    """Seeds belonging to one split of a make_split/make_subset CSV.

    ``seed`` is forced to int64 on read.  The manifest's seeds reach
    4.6e18; if pandas ever infers float64 for that column (one malformed
    row is enough) every value silently loses ~3 digits of precision and the
    resulting filter matches nothing -- which surfaces as an empty dataset,
    not as an error, unless it is caught here.
    """
    import pandas as pd
    df = pd.read_csv(split_csv, usecols=["seed", "split"],
                     dtype={"seed": "int64", "split": "string"})
    seeds = set(int(s) for s in df.loc[df["split"] == split_name, "seed"])
    if not seeds:
        raise ValueError(
            f"split '{split_name}' is empty in {split_csv}. "
            f"Present: {sorted(set(df['split'].dropna().unique()))}")
    return seeds


def _self_test() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    ds = DriftSensePairs(args.dataset,
                           difficulties=("easy", "medium", "hard"))
    print(f"  dataset size: {len(ds)}")
    for i in range(min(args.n, len(ds))):
        it = ds[i]
        assert it["ref"].shape == (1, TEMPLATE_PX, TEMPLATE_PX), it["ref"].shape
        assert it["search"].shape == (1, 1000, 1000), it["search"].shape
        assert -0.5 <= it["frac_x"] <= 0.5 and -0.5 <= it["frac_y"] <= 0.5
        from driftsense.dl_localize.coords import out_to_pixel
        back = out_to_pixel(it["cell_x"] + it["frac_x"])
        assert abs(back - it["gt_x"]) < 1e-6, (back, it["gt_x"])
        print(f"  id={it['id']:>7} {it['difficulty']:<7} "
              f"gt=({it['gt_x']:7.2f},{it['gt_y']:7.2f}) "
              f"cell=({it['cell_x']:>3},{it['cell_y']:>3}) "
              f"frac=({it['frac_x']:+.2f},{it['frac_y']:+.2f}) "
              f"tmpl=(ang {it['tmpl_angle']:+.1f}, sc {it['tmpl_scale']:.1f}) "
              f"pitch_out=({it['pitch_x_out']:.2f},{it['pitch_y_out']:.2f})")
    print("  template size fixed, cell+frac round-trips to gt exactly")
    print("  OK")


if __name__ == "__main__":
    _self_test()
