"""Inference for the trained learned dense correlation model.

Used by ``tools/eval_dl.py`` for batch evaluation, and is what the final
deliverable ``localize.py`` will call once the classical-vs-learned
comparison is measured and the choice is made for real.

Deliberately does NOT replace the classical ``localize.py`` yet. Per the
project's own rule (an inference script that fails scores zero), a model
this new does not become the only path until it has been measured, on real
hardware, to actually beat the baseline it is meant to replace.

NOTE: syntax-checked only (no torch in the authoring environment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from driftsense.localize.coords import out_to_pixel, shrink_size
from driftsense.localize.data import STRIDE, _normalize
from driftsense.localize.model import (SiameseCorrelationNet,
                                       predict_xy)


class LearnedLocalizer:
    """Loads a checkpoint once; call repeatedly for each pair."""

    def __init__(self, checkpoint_path: str, device: str = "auto",
                use_ema: bool = True) -> None:
        """
        Args:
            checkpoint_path: Path to a ``.pt`` file written by ``train.py``.
            device: "auto", "cuda", or "cpu".
            use_ema: Load the EMA-averaged weights rather than the raw ones.
                EMA measurably reduces sub-pixel jitter; prefer it unless
                debugging a specific training-time issue.
        """
        self.device = (torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu") if device == "auto"
                      else torch.device(device))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]

        self.net = SiameseCorrelationNet()
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def locate(self, reference: np.ndarray, search: np.ndarray,
              scale: float = 10.0) -> dict:
        """Locate the reference pattern inside the search image.

        Args:
            reference: Grayscale reference image (uint8), full resolution.
            search: Grayscale search image (uint8).
            scale: Demagnification; the reference is shrunk by this before
                encoding, matching the classical pipeline's convention.

        Returns:
            Dict with ``x``, ``y`` (sub-pixel, original search pixels) and
            ``confidence`` (sigmoid of the peak logit, in [0, 1] -- how
            certain the model is, usable for the same agreement-gating and
            centre-closest tie-break logic as the classical path).
        """
        k = shrink_size(reference.shape[0], scale)
        ref_small = cv2.resize(reference, (k, k), interpolation=cv2.INTER_AREA)

        ref_t = _normalize(ref_small).unsqueeze(0).to(self.device)
        srch_t = _normalize(search).unsqueeze(0).to(self.device)

        logits, ref_emb_size = self.net(ref_t, srch_t)
        coords_out = predict_xy(logits)

        x = float(out_to_pixel(coords_out[0].item(), ref_emb_size, STRIDE))
        y = float(out_to_pixel(coords_out[1].item(), ref_emb_size, STRIDE))
        peak_logit = float(logits.max().item())
        confidence = float(torch.sigmoid(torch.tensor(peak_logit)))

        return {"x": x, "y": y, "confidence": confidence}


def _self_test() -> None:
    """Requires a trained checkpoint; not runnable without training first.
    Manual smoke check once a checkpoint exists:
        python -m driftsense.localize.infer_dl --checkpoint runs/lscv/smoke.pt \\
            --reference some_ref.png --search some_search.png
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--search", required=True)
    args = ap.parse_args()

    loc = LearnedLocalizer(args.checkpoint)
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    out = loc.locate(ref, srch)
    print(f"  x={out['x']:.2f}  y={out['y']:.2f}  "
          f"confidence={out['confidence']:.3f}")


if __name__ == "__main__":
    _self_test()
