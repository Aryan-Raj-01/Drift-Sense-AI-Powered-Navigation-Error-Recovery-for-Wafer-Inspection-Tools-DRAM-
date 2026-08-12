"""Losses for the learned dense correlation model.

Three terms:

**Heatmap** -- penalty-reduced focal loss (Zhou et al., "Objects as Points" /
CenterNet, 2019, alpha=2, beta=4).  Handles the extreme positive/negative
imbalance of one true peak in a ~226x226 map (~1:51000), and softens the
penalty for negatives near the peak instead of treating a 1 px miss like a
100 px miss.

**Lattice weight** -- plain CenterNet weights every non-peak location
identically.  The measured failure mode of the classical baseline is a
periodic lock: the wrong answer lands at an exact multiple of the lattice
pitch, not at a random position (see ``tools/README_STAGE1.md``).  This term
multiplies the negative-pixel loss by an extra weight at those specific
offsets, so the network is trained specifically against the confusions it
actually makes -- the "twins," not the "lampposts."  This is the single
highest-leverage decision in the training design.

**Coordinate** -- Huber loss on the soft-argmax output, mapped to original
search pixels via ``coords.out_to_pixel`` -- the exact quantity the hackathon
scores, optimised directly rather than only through the heatmap proxy.

NOTE: syntax-checked only (no torch in the authoring environment). See
``model.py`` module docstring.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from driftsense.localize.coords import out_to_pixel


def gaussian_bump(h: int, w: int, cx: float, cy: float, sigma: float,
                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A single 2-D Gaussian on an (h, w) grid, peak value 1.0 at (cx, cy)."""
    ys = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, -1)
    return torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def lattice_weight_map(h: int, w: int, cx: float, cy: float,
                       pitch_x: float, pitch_y: float,
                       device: torch.device, dtype: torch.dtype,
                       boost: float = 4.0, max_k: int = 6,
                       sigma: float = 1.2) -> torch.Tensor:
    """Extra loss weight at (cx + m*pitch_x, cy + n*pitch_y), (m, n) != (0, 0).

    Args:
        h: Heatmap height (correlation-output grid units).
        w: Heatmap width.
        cx: True peak x, in the same grid.
        cy: True peak y.
        pitch_x: Lattice pitch along x, in the same grid units. Non-positive
            disables the boost (falls back to plain CenterNet weighting).
        pitch_y: Lattice pitch along y.
        device: Torch device.
        dtype: Torch dtype.
        boost: Additional weight multiplier at each lattice replica.
        max_k: Replicas checked out to +/- this many pitches.
        sigma: Spread of each boosted bump, in grid cells.

    Returns:
        (h, w) weight map, 1.0 away from any lattice replica.
    """
    w_map = torch.ones(h, w, device=device, dtype=dtype)
    if pitch_x <= 0 or pitch_y <= 0:
        return w_map
    for m in range(-max_k, max_k + 1):
        for n in range(-max_k, max_k + 1):
            if m == 0 and n == 0:
                continue
            ox, oy = cx + m * pitch_x, cy + n * pitch_y
            if -5.0 <= ox < w + 5.0 and -5.0 <= oy < h + 5.0:
                w_map = w_map + boost * gaussian_bump(h, w, ox, oy, sigma,
                                                       device, dtype)
    return w_map


def map_softmax_ce(logits: torch.Tensor, target_dist: torch.Tensor
                   ) -> torch.Tensor:
    """Cross-entropy over the WHOLE correlation map treated as one softmax.

    This replaces the per-pixel focal loss, which collapsed. Focal treats each
    of the ~51000 locations as an independent binary decision, so with a
    single positive the cheapest way to reduce it is to push every logit
    down -- "no match anywhere". The loss falls beautifully (701k -> 6.8k was
    observed) while the map goes flat and the argmax becomes meaningless.

    A softmax over the flattened map makes the locations COMPETE: the
    probabilities sum to one, so lowering everything changes nothing. The only
    way to reduce the loss is to raise the true location *relative to* the
    others. Degenerate collapse becomes impossible by construction.

    This also gives lattice-offset hard-negative mining for free. The gradient
    on each wrong location is proportional to the probability the model
    assigns it, so whichever replicas the model currently confuses with the
    truth -- exactly the periodic locks one pitch away -- receive the largest
    gradient automatically, with no explicit negative sampling.

    Args:
        logits: (h, w) raw logits.
        target_dist: (h, w) non-negative target distribution summing to 1.

    Returns:
        Scalar cross-entropy.
    """
    log_p = F.log_softmax(logits.float().reshape(-1), dim=0)
    return -(target_dist.float().reshape(-1) * log_p).sum()


def build_soft_target(h: int, w: int, gt_out_x: float, gt_out_y: float,
                      device: torch.device, dtype: torch.dtype,
                      sigma: float = 1.0) -> torch.Tensor:
    """Normalised Gaussian target distribution for :func:`map_softmax_ce`.

    A soft target rather than a one-hot index: neighbouring cells are not
    "wrong", they are nearly right, and a small amount of mass on them keeps
    the peak smooth enough for the sub-pixel soft-argmax to interpolate
    within.

    Args:
        h: Map height.
        w: Map width.
        gt_out_x: True location x, in map grid units.
        gt_out_y: True location y, in map grid units.
        device: Torch device.
        dtype: Torch dtype.
        sigma: Spread in grid cells.

    Returns:
        (h, w) tensor summing to 1.
    """
    g = gaussian_bump(h, w, gt_out_x, gt_out_y, sigma, device, dtype).float()
    total = g.sum()
    if float(total) <= 0:
        # Ground truth fell outside the map: fall back to the nearest cell so
        # the loss stays finite rather than producing NaN.
        g = torch.zeros(h, w, device=device, dtype=torch.float32)
        iy = int(min(max(round(gt_out_y), 0), h - 1))
        ix = int(min(max(round(gt_out_x), 0), w - 1))
        g[iy, ix] = 1.0
        return g
    return g / total


def focal_heatmap_loss(logits: torch.Tensor, heatmap: torch.Tensor,
                       weight_map: Optional[torch.Tensor] = None,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss.

    Numerics: the loss is computed in float32 regardless of the autocast
    dtype, and uses ``logsigmoid`` rather than ``log(sigmoid(.))``.  Both are
    required, not stylistic.  In bfloat16 the mantissa is 8 bits, so the
    smallest representable step below 1.0 is about 0.0078 -- a probability
    clamped to 0.999999 rounds to exactly 1.0, making ``log(1 - p)`` evaluate
    to ``log(0) = -inf`` and turning the whole network to NaN on the first
    backward pass.  ``logsigmoid(-logits)`` computes the same quantity
    without ever forming ``1 - p``.

    Args:
        logits: (h, w) raw logits (pre-sigmoid).
        heatmap: (h, w) Gaussian target, peak exactly 1.0 at the true pixel.
        weight_map: Optional (h, w) extra weight on negative pixels; see
            :func:`lattice_weight_map`.
        alpha: Focal exponent on the prediction confidence.
        beta: Exponent controlling how fast the penalty falls off near a
            positive as `heatmap` approaches 1.

    Returns:
        Scalar loss, normalised by the (here, always 1) positive count.
    """
    logits = logits.float()
    heatmap = heatmap.float()

    log_p = F.logsigmoid(logits)
    log_one_minus_p = F.logsigmoid(-logits)
    p = torch.sigmoid(logits)

    is_peak = heatmap.eq(1.0).to(p.dtype)

    pos_loss = -log_p * (1.0 - p) ** alpha * is_peak
    neg_loss = (-log_one_minus_p * p ** alpha
               * (1.0 - heatmap) ** beta * (1.0 - is_peak))
    if weight_map is not None:
        neg_loss = neg_loss * weight_map.float()

    n_pos = is_peak.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def build_target(h: int, w: int, gt_out_x: float, gt_out_y: float,
                 pitch_x_out: float, pitch_y_out: float,
                 device: torch.device, dtype: torch.dtype,
                 sigma: float = 1.5, lattice_boost: float = 4.0
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the Gaussian heatmap target and lattice weight map together.

    Args:
        h: Heatmap height.
        w: Heatmap width.
        gt_out_x: True location, in the correlation-output grid (i.e. after
            ``coords.pixel_to_out``).
        gt_out_y: True location y, same grid.
        pitch_x_out: Lattice pitch along x, same grid units.
        pitch_y_out: Lattice pitch along y.
        device: Torch device.
        dtype: Torch dtype.
        sigma: Heatmap peak spread, in grid cells.
        lattice_boost: Passed through to :func:`lattice_weight_map`.

    Returns:
        ``(heatmap, weight_map)``, both (h, w).
    """
    heatmap = gaussian_bump(h, w, gt_out_x, gt_out_y, sigma, device, dtype)
    # The exact peak pixel must read as 1.0 for focal_heatmap_loss's is_peak
    # mask; a Gaussian sampled off-grid may miss 1.0 by float error.
    iy = int(round(gt_out_y))
    ix = int(round(gt_out_x))
    if 0 <= iy < h and 0 <= ix < w:
        heatmap[iy, ix] = 1.0
    weight_map = lattice_weight_map(h, w, gt_out_x, gt_out_y,
                                    pitch_x_out, pitch_y_out,
                                    device, dtype, boost=lattice_boost)
    return heatmap, weight_map


def coordinate_loss(coords_out: torch.Tensor, ref_emb_size: int, stride: float,
                    gt_px_x: float, gt_px_y: float,
                    huber_delta: float = 3.0) -> torch.Tensor:
    """Huber loss between the decoded prediction and ground truth, in pixels.

    Args:
        coords_out: (2,) soft-argmax output, in correlation-output grid units.
        ref_emb_size: Reference embedding spatial size, from the model
            forward pass -- see ``model.py`` and ``coords.py`` docstrings.
        stride: Encoder output stride (4).
        gt_px_x: True x, in original search pixels.
        gt_px_y: True y, in original search pixels.
        huber_delta: Transition point between quadratic and linear loss, in
            pixels. Above this, outliers (likely periodic locks early in
            training) do not dominate the gradient.

    Returns:
        Scalar loss.
    """
    coords_out = coords_out.float()  # see focal_heatmap_loss on bf16 numerics
    px_x = out_to_pixel(coords_out[0], ref_emb_size, stride)
    px_y = out_to_pixel(coords_out[1], ref_emb_size, stride)
    target = torch.stack([
        torch.as_tensor(gt_px_x, dtype=coords_out.dtype, device=coords_out.device),
        torch.as_tensor(gt_px_y, dtype=coords_out.dtype, device=coords_out.device),
    ])
    pred = torch.stack([px_x, px_y])
    return F.huber_loss(pred, target, delta=huber_delta)
