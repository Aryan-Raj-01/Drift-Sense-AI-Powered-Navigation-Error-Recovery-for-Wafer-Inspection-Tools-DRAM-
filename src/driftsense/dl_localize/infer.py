"""Inference for the fixed dense-correlation model.

    from driftsense.dl_localize.infer import LearnedLocalizer
    loc = LearnedLocalizer("<run-dir>/phase3_all.pt")
    out = loc.locate(reference_uint8, search_uint8)   # -> {"x","y",...}

WHAT CHANGED FROM ``localize/infer_dl.py``
==========================================

1. NO ``scale`` ARGUMENT ANY MORE.  The old ``locate(reference, search,
   scale=10.0)`` shrank the reference by a caller-supplied factor and let
   that factor set the template's pixel size.  ``tools/eval_dl.py`` never
   passed one, so every evaluated sample used 10.0 while every trained
   sample had used its own ``scale_ratio`` label -- a train/inference
   mismatch in both geometry (a 0-1 px per-axis coordinate bias, measured;
   see coords) and content (up to 4% template scale error).  The template
   size is now a constant and scale variation is handled by searching a
   bank.

2. TEMPLATE BANK OVER ROTATION x SCALE.  ``rel_rotation_deg`` has sigma 1.6
   and max 5.83 in the manifest.  The classical baseline searches five
   angles and gets 89% on easy; the learned path searched none and got 84%.
   A rigid correlation kernel cannot absorb a 3 degree rotation of a 100 px
   template -- the corners move ~3.7 px against a 1 px tolerance.

   The bank is cheap because the expensive half of the work is encoding the
   1000x1000 search image, and that is done ONCE and shared.  Every template
   is the same fixed size, so the whole bank is a single ``conv2d`` with B
   output channels.  A 5x3 bank costs roughly 15 correlations plus one
   search encode, against a measured budget of ~1000 ms/pair for the
   classical baseline that this must beat on time as well as accuracy.

   Selecting by max cosine needs no extra training: a better-matched
   template genuinely produces a higher correlation.  That is a property of
   the similarity, not something the network has to learn.

3. CALIBRATED CONFIDENCE.  The old confidence was
   ``sigmoid(peak_logit)``.  Logits are ``gain * cosine + bias`` with gain
   initialised to 20, so sigmoid saturates: both 10k runs reported exactly
   1.000 for every one of 2000 samples.  Replaced by the softmax
   probability at the peak plus the LOGIT MARGIN to the best competitor
   outside a suppression radius.  The margin is what separates a clean match
   from a periodic lock, and it is the signal the centre tie-break needs.

4. CENTRE TIE-BREAK IMPLEMENTED.  The challenge statement says: when several
   matching regions are found, return the one nearest the centre of the
   search image.  Nothing in the learned path implemented that.  When the
   runner-up is within ``tie_margin`` logits of the peak, the candidate
   closer to the search centre wins.

Self-test (needs a checkpoint)::

    python -m driftsense.dl_localize.infer --checkpoint <run-dir>/phase3_all.pt ^
        --reference ref.png --search search.png
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from driftsense.dl_localize.coords import STRIDE, TEMPLATE_PX, out_to_pixel
from driftsense.dl_localize.data import (DriftSensePairs, build_template,
                                          normalize_image)
from driftsense.dl_localize.model import (SiameseCorrelationNet, peak_xy,
                                           peak_confidence,
                                           pitch_limited_radius, refine_xy,
                                           windowed_soft_argmax)
from driftsense.dl_localize.refine import (ANGLES as REFINE_ANGLES,
                                            RADIUS as REFINE_RADIUS,
                                            make_template_bank, refine_local)


# Checkpoints record the architecture that produced them. The shipped weights
# were trained before this package was renamed, so their stored string is the
# historical one. Both are accepted: the network definition did not change, only
# the module and class names did.
#
# The historical constant is built from parts on purpose. Written literally it
# would be rewritten by any future rename sweep across this file, and the
# failure is silent -- inference falls back to the classical path and reports a
# plausible but much less accurate coordinate.
_LEGACY_ARCH = "localize" + "88." + "SiameseCorrelationNet" + "88"
_ACCEPTED_ARCH = ("dl_localize.SiameseCorrelationNet", _LEGACY_ARCH)


class LearnedLocalizer:
    """Loads a checkpoint once; call :meth:`locate` per pair."""

    def __init__(self, checkpoint_path: str, device: str = "auto",
                 use_ema: bool = True,
                 angle_bank: Optional[Sequence[float]] = None,
                 scale_bank: Optional[Sequence[float]] = None,
                 readout: str = "offset", tie_margin: float = 0.0,
                 refine: bool = True, refine_radius: int = REFINE_RADIUS,
                 refine_angles: Optional[Sequence[float]] = None) -> None:
        """
        Args:
            checkpoint_path: ``.pt`` written by ``train``.
            device: "auto", "cuda", or "cpu".
            use_ema: Load the EMA-averaged weights.  EMA reduces sub-pixel
                jitter; prefer it unless debugging a training-time issue.
            angle_bank: Template rotations, degrees.  Defaults to the bank
                recorded in the checkpoint, which is the bank training used.
                Overriding it to something wider is legitimate at inference
                (the encoder does not know about the bank) but overriding it
                to something NARROWER than training reintroduces a mismatch.
            scale_bank: Template demagnifications.  Same caveat.
            readout: ``"offset"``, ``"softargmax"`` or ``"both"``.
            tie_margin: Logit gap under which the centre tie-break applies.
                DEFAULT 0.0, i.e. DISABLED, and that default is deliberate.
                The tie-break can only ever move an answer away from the
                model's own best guess, so if the threshold is too generous
                it converts correct predictions into wrong ones.  Measured
                here on an untrained net: with ``tie_margin=0.5`` a synthetic
                pair whose argmax was correct to 0 px was moved 130 px by the
                tie-break, because a weak peak had a 0.097 logit margin.
                Logits are ``gain * cosine + bias``, so "0.5" is not a
                physical quantity -- it depends on the learned gain.
                Calibrate it: run ``eval --diagnose``, read the reported
                mean logit margin for hits versus failures, and set this
                BELOW the hits distribution.  Turning it on without that
                measurement is a way to lose accuracy, not gain it.
        """
        self.device = (torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
                       if device == "auto" else torch.device(device))
        ck = torch.load(checkpoint_path, map_location=self.device,
                        weights_only=False)
        if ck.get("arch") not in _ACCEPTED_ARCH:
            raise ValueError(
                f"{checkpoint_path} is not a dl_localize checkpoint "
                f"(arch={ck.get('arch', 'localize (old)')}). The old package "
                f"used a different encoder fusion and a different coordinate "
                f"convention; loading it here would silently mislocalise.")
        if int(ck.get("template_px", TEMPLATE_PX)) != TEMPLATE_PX:
            raise ValueError(
                f"checkpoint was trained with template_px="
                f"{ck['template_px']} but coords.TEMPLATE_PX is "
                f"{TEMPLATE_PX}. The coordinate mapping would be off by "
                f"{abs(ck['template_px'] - TEMPLATE_PX) / 2:.1f} px.")

        state = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
        self.net = SiameseCorrelationNet()
        self.net.load_state_dict({k: v for k, v in state.items()})
        self.net.to(self.device).eval()

        # A SINGLE template by default.  The rotation x scale bank was
        # measured on 2000 val pairs to make things WORSE on every bucket:
        #
        #            bank-15   bank-1
        #   easy       85.1%    85.7%
        #   medium     81.7%    83.5%
        #   hard       33.8%    37.5%
        #   ms/pair      242      156
        #
        # Selection is by max peak logit.  Where the correlation surface is
        # sharp (easy) that is reliable; where it is flat (hard) a wrong
        # member routinely wins with a spurious peak at a wrong location, so
        # the bank injects noise exactly where the model is weakest.  My
        # original reasoning -- "a better-matched template genuinely scores
        # higher, so selection needs no training" -- is true in expectation
        # and false per-sample, which is what matters.
        #
        # Kept as an override rather than deleted: if the encoder is later
        # trained to make peak height a calibrated match-quality signal, or
        # if bank selection is gated on `margin`, this becomes viable again.
        self.angle_bank = tuple(angle_bank if angle_bank is not None
                                else (0.0,))
        self.scale_bank = tuple(scale_bank if scale_bank is not None
                                else (10.0,))
        self.readout = readout
        self.tie_margin = float(tie_margin)
        self.bank = [(a, s) for s in self.scale_bank for a in self.angle_bank]
        self.refine = bool(refine)
        self.refine_radius = int(refine_radius)
        self.refine_angles = (tuple(refine_angles) if refine_angles is not None
                              else REFINE_ANGLES)

    def _templates(self, reference: np.ndarray) -> torch.Tensor:
        """(B, 1, TEMPLATE_PX, TEMPLATE_PX) normalised template bank."""
        out = []
        for angle, scale in self.bank:
            t = build_template(reference, scale, angle)
            out.append((t - t.mean()) / (t.std() + 1e-6))
        arr = np.stack(out).astype(np.float32)[:, None]
        return torch.from_numpy(arr).to(self.device)

    @torch.no_grad()
    def locate(self, reference: np.ndarray, search: np.ndarray,
               pitch_x_px: float = 0.0, pitch_y_px: float = 0.0,
               return_all: bool = False) -> Dict:
        """Locate the reference pattern inside the search image.

        Args:
            reference: (N, N) grayscale reference, full resolution.
            search: (H, W) grayscale search image.
            pitch_x_px: Optional lattice pitch along x in search pixels.  If
                given, the sub-pixel window is shrunk so it cannot span more
                than one lattice period.  Purely optional -- Applied
                Materials' test set will not supply it, and the default is
                safe without it.
            pitch_y_px: Optional lattice pitch along y.
            return_all: Also return per-bank-member diagnostics.

        Returns:
            Dict with ``x``, ``y`` (sub-pixel, original search pixels),
            ``confidence`` (softmax mass at the peak), ``margin`` (logit gap
            to the best competitor -- the useful one), ``angle``, ``scale``
            (winning bank member), and ``tie_break`` (whether the centre rule
            fired).
        """
        srch_t = normalize_image(search).unsqueeze(0).to(self.device)
        bank_t = self._templates(reference)

        se = self.net.embed(srch_t)
        corr = self.net.correlate_bank(bank_t, se)            # (B, h, w)
        logits_bank = self.net.head(corr.unsqueeze(1).float())[:, 0]

        # Winning bank member = highest peak.  A better-matched template
        # gives a genuinely higher cosine; no training term needed.
        peaks = logits_bank.reshape(logits_bank.shape[0], -1).max(dim=1).values
        b = int(torch.argmax(peaks))
        logits = logits_bank[b]
        offsets = self.net.offset(corr[b][None, None].float())

        radius = pitch_limited_radius(pitch_x_px / STRIDE, pitch_y_px / STRIDE,
                                      default=2)
        prob, margin = peak_confidence(logits, suppress_radius=radius + 1)

        px, py = peak_xy(logits)
        tie_break = False
        if self.tie_margin > 0.0 and margin < self.tie_margin:
            # Several regions match about equally well.  The challenge
            # statement's rule: return the one nearest the centre of the
            # search image.
            cand = self._competitors(logits, radius + 1, self.tie_margin)
            cy_c = (search.shape[0] - 1) / 2.0
            cx_c = (search.shape[1] - 1) / 2.0
            best = min(cand, key=lambda c: (out_to_pixel(c[0]) - cx_c) ** 2
                       + (out_to_pixel(c[1]) - cy_c) ** 2)
            tie_break = (best[0], best[1]) != (px, py)
            px, py = best

        # Sub-pixel refinement at the SELECTED cell (which the tie-break may
        # have moved away from the global argmax).
        off = offsets[:, py, px].float()
        hard = (float(px) + float(off[0]), float(py) + float(off[1]))
        if self.readout == "offset":
            sub = hard
        else:
            sa = windowed_soft_argmax(logits, px, py, radius=radius)
            sa = (float(sa[0]), float(sa[1]))
            sub = sa if self.readout == "softargmax" else (
                0.5 * (hard[0] + sa[0]), 0.5 * (hard[1] + sa[1]))

        res = {
            "x": float(out_to_pixel(sub[0])),
            "y": float(out_to_pixel(sub[1])),
            "confidence": prob,
            "margin": margin,
            "angle": self.bank[b][0],
            "scale": self.bank[b][1],
            "tie_break": tie_break,
        }

        # Stage 3: hand the last pixel to classical stride-1 NCC.  The learned
        # stage is a global search that lands 99.6% of easy samples within
        # 5 px; the +/-3 px window it hands over cannot contain a lattice
        # replica (minimum measured pitch 5.3 px), so this can only sharpen a
        # correct answer, never relocate to a wrong one.  See refine.
        if self.refine:
            res["x_learned"], res["y_learned"] = res["x"], res["y"]
            rf = refine_local(reference, search, res["x"], res["y"],
                              angles=self.refine_angles,
                              radius=self.refine_radius)
            res.update({"x": rf["x"], "y": rf["y"], "ncc": rf["ncc"],
                        "refine_angle": rf["angle"], "shift": rf["shift"]})
        if return_all:
            res["bank_peaks"] = [float(v) for v in peaks]
            res["bank"] = list(self.bank)
        return res

    @staticmethod
    def _competitors(logits: torch.Tensor, suppress: int, margin: float,
                     max_n: int = 8) -> List[Tuple[int, int]]:
        """Local maxima within ``margin`` logits of the global peak."""
        lg = logits.detach().float().clone()
        h, w = lg.shape
        top = float(lg.max())
        out: List[Tuple[int, int]] = []
        for _ in range(max_n):
            v = float(lg.max())
            if v < top - margin:
                break
            x, y = peak_xy(lg)
            out.append((x, y))
            y0, y1 = max(0, y - suppress), min(h, y + suppress + 1)
            x0, x1 = max(0, x - suppress), min(w, x + suppress + 1)
            lg[y0:y1, x0:x1] = float("-inf")
        return out or [peak_xy(logits)]


def _self_test() -> None:
    import argparse
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    loc = LearnedLocalizer(args.checkpoint, device=args.device)
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    out = loc.locate(ref, srch, return_all=True)
    print(f"  x={out['x']:.2f}  y={out['y']:.2f}")
    print(f"  winning bank member: angle {out['angle']:+.1f} deg, "
          f"scale {out['scale']:.2f}")
    print(f"  confidence {out['confidence']:.3e}  margin {out['margin']:.3f}  "
          f"tie_break {out['tie_break']}")
    print(f"  bank peaks: "
          + ", ".join(f"{a:+.0f}/{s:.1f}:{p:.2f}"
                      for (a, s), p in zip(out["bank"], out["bank_peaks"])))


if __name__ == "__main__":
    _self_test()
