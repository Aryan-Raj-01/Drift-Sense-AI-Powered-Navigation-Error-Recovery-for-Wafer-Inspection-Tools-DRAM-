"""Losses for the fixed dense-correlation model.

Four terms.  Everything is computed in float32 regardless of the autocast
dtype -- see the numerics note at the bottom of this docstring, which is the
reason two of the four bugs already found on this project happened.

1. ``map_softmax_ce`` -- the whole flattened correlation map as ONE softmax
   classification.  KEPT, unchanged in spirit.  A previous review
   recommended reverting this to ``focal_heatmap_loss``.  That would
   reintroduce a measured, understood, already-fixed failure: with a single
   positive among ~51000 independent binary decisions, the cheapest descent
   direction under focal loss is to push every logit down -- "no match
   anywhere".  Observed on this project: loss 701,315 -> 6,823 over 600
   steps while error went 1.54 px -> 422.53 px.  Confirmed analytically too
   (collapse loss 8.0 < correct-answer loss 92.0).  Under a softmax the
   probabilities sum to one, so lowering everything changes nothing and
   collapse is impossible by construction.  Do not revert this.

2. ``lattice_margin_loss`` -- NEW, and this is the term that finally puts
   the roadmap's central idea into the ACTIVE loss path.

   The idea ("train explicitly against the periodic replicas one pitch away,
   the twins rather than the lampposts") was implemented as
   ``losses.lattice_weight_map`` and wired only into ``focal_heatmap_loss``,
   which is dead code.  ``train.py`` called ``build_soft_target`` and
   ``map_softmax_ce`` and passed ``lattice_boost`` to a function that
   ignored it.  So the single most-emphasised design decision in the project
   has never run.

   The previous review's proposed fix -- multiply the softmax CE by the
   weight map -- does not typecheck as mathematics.  Softmax CE is
   ``-sum_i t_i log p_i``; with a one-hot-ish target the negatives do not
   appear as separate additive terms, so there is nothing to reweight.  Any
   per-pixel weight applied to a softmax CE either does nothing or silently
   breaks the normalisation that prevents collapse in the first place.

   The correct construction inside a softmax framework is an explicit MARGIN
   against the replica set::

       L = relu(margin + logsumexp(logits over lattice replicas)
                       - logit at the true cell)

   ``logsumexp`` is a soft maximum, so the gradient concentrates on whichever
   replica the model currently ranks highest -- automatic hard-negative
   mining over exactly the confusions the classical baseline was measured to
   make (4% periodic lock).  ``relu`` makes the term free once the margin is
   satisfied, so it stops fighting the CE term after it has done its job.

   Replicas closer than ``min_sep`` cells to the true peak are EXCLUDED.
   This is not a detail: measured lattice pitch is 53-227 nm at ~10 nm/px
   and stride 4, i.e. 1.3-5.7 correlation cells.  At the fine end the first
   replica is barely one cell from the truth and sits inside the sigma=1.0
   Gaussian target, so penalising it would penalise the correct answer.

3. ``offset_loss`` -- L1 on the sub-pixel offset head at the true cell.
   Stride 4 quantises the peak to 4 px against a 1 px metric; this is where
   sub-pixel accuracy actually comes from.

4. ``coordinate_loss`` -- Huber in original search pixels, the quantity the
   hackathon scores, optimised directly rather than only through the heatmap
   proxy.

NUMERICS.  All of these run in float32 even under bf16/fp16 autocast.  In
bfloat16 the mantissa is 8 bits, so the smallest step below 1.0 is ~0.0078:
a probability clamped to 0.999999 rounds to exactly 1.0 and ``log(1 - p)``
becomes ``-inf``.  That was bug #1 on this project.  Separately, the
quantity that decides a periodic lock is the gap between the true peak's
cosine and a replica's, which can be a few thousandths -- comparable to one
bfloat16 ULP at these magnitudes.  Nothing that decides an answer is allowed
to live in a low-mantissa dtype.

Self-test (pure torch, CPU, no data)::

    python -m driftsense.dl_localize.losses
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from driftsense.dl_localize.coords import STRIDE, TEMPLATE_PX, out_to_pixel


def gaussian_bump(h: int, w: int, cx: float, cy: float, sigma: float,
                  device: torch.device) -> torch.Tensor:
    """A single 2-D Gaussian on an (h, w) float32 grid, peak 1.0 at (cx, cy).

    float32 unconditionally.  The old version took a ``dtype`` argument and
    ``train.py`` passed ``logits.dtype``, which under autocast is bfloat16.
    The grid coordinates then round to bfloat16: at index ~113 the bfloat16
    spacing is 0.5, so the Gaussian's centre was snapped to the nearest half
    cell -- a +/-1 px quantisation of the training target, on a 1 px metric.
    """
    ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    cxt = torch.as_tensor(float(cx), device=device, dtype=torch.float32)
    cyt = torch.as_tensor(float(cy), device=device, dtype=torch.float32)
    return torch.exp(-((xs - cxt) ** 2 + (ys - cyt) ** 2) / (2.0 * sigma ** 2))


def build_soft_target(h: int, w: int, gt_out_x: float, gt_out_y: float,
                      device: torch.device, sigma: float = 1.0
                      ) -> torch.Tensor:
    """Normalised Gaussian target distribution for :func:`map_softmax_ce`.

    A soft target rather than one-hot: neighbouring cells are not "wrong",
    they are nearly right, and a little mass on them keeps the peak smooth
    enough for the sub-pixel readout to interpolate within.

    Returns:
        (h, w) float32 tensor summing to 1.
    """
    g = gaussian_bump(h, w, gt_out_x, gt_out_y, sigma, device)
    total = g.sum()
    if float(total) <= 0.0:
        # Ground truth fell outside the map.  Fall back to the nearest cell
        # so the loss stays finite instead of producing NaN.
        g = torch.zeros(h, w, device=device, dtype=torch.float32)
        iy = int(min(max(round(gt_out_y), 0), h - 1))
        ix = int(min(max(round(gt_out_x), 0), w - 1))
        g[iy, ix] = 1.0
        return g
    return g / total


def soft_target_entropy(sigma: float = 1.0, half: int = 12) -> float:
    """Entropy floor of the sigma-Gaussian target, in nats.

    ``map_softmax_ce`` cannot go below this even with a perfect prediction.
    Log it next to the training loss: a run that converges to ~this value has
    a correct heatmap, and a run that goes below it is leaking or broken.
    For sigma=1.0 this is ~2.84, which is exactly where the healthy
    overfit test converged (2.842).
    """
    xs = torch.arange(-half, half + 1, dtype=torch.float64)
    g = torch.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma ** 2))
    p = (g / g.sum()).reshape(-1)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def map_softmax_ce(logits: torch.Tensor, target_dist: torch.Tensor
                   ) -> torch.Tensor:
    """Cross-entropy over the whole correlation map as one softmax.

    Args:
        logits: (h, w) raw logits.
        target_dist: (h, w) non-negative, summing to 1.

    Returns:
        Scalar cross-entropy, in nats.  Floor is
        :func:`soft_target_entropy`.
    """
    log_p = F.log_softmax(logits.float().reshape(-1), dim=0)
    return -(target_dist.float().reshape(-1) * log_p).sum()


def lattice_replica_mask(h: int, w: int, cx: float, cy: float,
                         pitch_x: float, pitch_y: float,
                         device: torch.device, tol: float = 0.75,
                         min_sep: float = 2.0, max_k: int = 8
                         ) -> torch.Tensor:
    """Boolean mask of correlation cells sitting on a lattice replica.

    Vectorised: a cell is on a replica when its offset from the true peak is
    close to an integer number of pitches along BOTH axes.  Doing it this way
    rather than stamping a Gaussian per (m, n) avoids a 289-iteration Python
    loop inside the training step.

    Args:
        h: Map height.
        w: Map width.
        cx: True peak x, grid cells.
        cy: True peak y, grid cells.
        pitch_x: Lattice pitch along x, grid cells.  Non-positive -> empty
            mask (the boost is simply disabled for that sample).
        pitch_y: Lattice pitch along y, grid cells.
        device: Torch device.
        tol: Half-width around each replica centre, in cells.
        min_sep: Replicas closer than this to the true peak are excluded.
            Measured pitch reaches 1.3 cells, where the first replica
            overlaps the sigma=1 target and penalising it would penalise the
            correct answer.
        max_k: Replicas considered out to +/- this many pitches per axis.

    Returns:
        (h, w) bool tensor.
    """
    if pitch_x is None or pitch_y is None or pitch_x <= 0 or pitch_y <= 0:
        return torch.zeros(h, w, dtype=torch.bool, device=device)

    ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    dx = xs - float(cx)
    dy = ys - float(cy)

    kx = dx / float(pitch_x)
    ky = dy / float(pitch_y)
    mx, my = kx.round(), ky.round()

    # On-lattice along each axis: within `tol` cells of an exact multiple of
    # the pitch, and no more than `max_k` pitches out.
    on_x = ((kx - mx).abs() * float(pitch_x) <= tol) & (mx.abs() <= max_k)
    on_y = ((ky - my).abs() * float(pitch_y) <= tol) & (my.abs() <= max_k)

    # A replica is any (m, n) != (0, 0) that is on-lattice in BOTH axes and
    # far enough from the true peak not to overlap the Gaussian target.
    nonzero = (mx.abs() >= 1) | (my.abs() >= 1)
    far = (dx ** 2 + dy ** 2) > float(min_sep) ** 2
    return on_x & on_y & nonzero & far


def lattice_margin_loss(logits: torch.Tensor, cx: float, cy: float,
                        pitch_x: float, pitch_y: float,
                        margin: float = 4.0, tol: float = 0.75,
                        min_sep: float = 2.0, max_k: int = 8
                        ) -> Tuple[torch.Tensor, int]:
    """Hinge pushing the true peak above every lattice replica.

    ``relu(margin + logsumexp(replica logits) - logit(true cell))``.

    ``logsumexp`` is a soft max over the replica set, so the gradient goes
    almost entirely to whichever replica currently scores highest -- the
    hard negative -- with no explicit mining pass.  ``relu`` zeroes the term
    once the margin is met, so it does not keep fighting the CE term.

    Args:
        logits: (h, w) raw logits.
        cx: True peak x in grid cells.
        cy: True peak y in grid cells.
        pitch_x: Lattice pitch along x, grid cells.
        pitch_y: Lattice pitch along y, grid cells.
        margin: Required logit gap.  Logits are ``gain * cosine + bias`` with
            gain ~20, so a margin of 4 asks for a cosine gap of ~0.2 -- large
            enough to survive noise, small enough to be reachable.
        tol: See :func:`lattice_replica_mask`.
        min_sep: See :func:`lattice_replica_mask`.
        max_k: See :func:`lattice_replica_mask`.

    Returns:
        ``(loss, n_replica_cells)``.  The count is returned for logging: if
        it is 0 for most samples the pitch metadata is missing and this term
        is silently doing nothing, which is precisely the failure mode this
        function exists to correct.
    """
    lg = logits.float()
    h, w = lg.shape
    mask = lattice_replica_mask(h, w, cx, cy, pitch_x, pitch_y, lg.device,
                                tol=tol, min_sep=min_sep, max_k=max_k)
    n = int(mask.sum())
    if n == 0:
        return lg.new_zeros(()), 0

    iy = int(min(max(round(cy), 0), h - 1))
    ix = int(min(max(round(cx), 0), w - 1))
    true_logit = lg[iy, ix]
    replica_lse = torch.logsumexp(lg[mask], dim=0)
    return F.relu(margin + replica_lse - true_logit), n


def offset_loss(offsets: torch.Tensor, cell_x: int, cell_y: int,
                frac_x: float, frac_y: float, radius: int = 1
                ) -> torch.Tensor:
    """L1 on the sub-pixel offset head over a neighbourhood of the true cell.

    WHY A NEIGHBOURHOOD AND NOT JUST THE PEAK
    -----------------------------------------
    CenterNet supervises its offset head at one location because each object
    has exactly one centre and neighbouring cells belong to no object.  That
    reasoning does not transfer here: the correct offset is DEFINED for every
    cell, as ``gt_out - cell``, so restricting supervision to a single cell
    throws away free labels.

    It also throws away almost all of the gradient.  One supervised cell per
    forward pass over a 1500-step phase is 12,000 supervised points for the
    whole head.  A radius-1 neighbourhood is 9x that, at no extra compute --
    the head already produces a prediction at every cell.

    And it fixes a real inference failure: when the argmax lands one cell off
    the truth, the head is read at a cell it was never trained on.  With
    neighbourhood supervision that cell has a correct target
    (``frac - offset_from_gt_cell``), so the readout degrades gracefully
    instead of returning something arbitrary.

    This is the measured bottleneck for the easy and medium buckets.  On the
    10k run, easy reached 98.3% within 2 px but only 85.7% within 1 px, and
    the head's converged L1 of 0.09 cells is 0.36 px per axis -- about
    0.51 px Euclidean, which is exactly the 0.52 px median error observed.
    Sub-pixel precision, not localisation, is what the 1 px metric is
    costing.

    Args:
        offsets: (2, h, w) offset-head output, in grid cells.
        cell_x: Ground-truth cell index x (``round(gt_out_x)``).
        cell_y: Ground-truth cell index y.
        frac_x: ``gt_out_x - cell_x``, in [-0.5, 0.5].
        frac_y: ``gt_out_y - cell_y``.
        radius: Neighbourhood half-width in cells.  With ``radius=1`` the
            largest target magnitude is 1.5 cells, so ``OffsetHead.max_cells``
            must be at least that or the tanh saturates on the corners.

    Returns:
        Scalar loss, in grid cells.  Multiply by STRIDE for pixels.
    """
    off = offsets.float()
    _, h, w = off.shape
    iy = int(min(max(cell_y, 0), h - 1))
    ix = int(min(max(cell_x, 0), w - 1))
    r = max(0, int(radius))

    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    patch = off[:, y0:y1, x0:x1]

    # Target at cell (x, y) is gt_out - cell, i.e. frac shifted by how far
    # that cell sits from the ground-truth cell.
    ys = torch.arange(y0, y1, device=off.device, dtype=off.dtype) - iy
    xs = torch.arange(x0, x1, device=off.device, dtype=off.dtype) - ix
    tx = float(frac_x) - xs.view(1, -1).expand(patch.shape[1], -1)
    ty = float(frac_y) - ys.view(-1, 1).expand(-1, patch.shape[2])
    target = torch.stack([tx, ty])
    return F.l1_loss(patch, target)


def coordinate_loss(coords_out: torch.Tensor, gt_px_x: float, gt_px_y: float,
                    huber_delta: float = 3.0, stride: float = STRIDE,
                    template_px: int = TEMPLATE_PX) -> torch.Tensor:
    """Huber loss between the decoded prediction and ground truth, in pixels.

    Note the signature: no ``ref_emb_size``.  The old version took one and
    the mapping depended on it, which is how a sample-dependent 0-1 px bias
    entered every decoded coordinate.  ``coords.out_to_pixel`` is now a
    constant affine map and there is nothing per-sample left to pass in.

    Args:
        coords_out: (2,) soft-argmax output, grid units.
        gt_px_x: True x, original search pixels.
        gt_px_y: True y.
        huber_delta: Quadratic/linear transition, in pixels.  Above this,
            outliers (periodic locks early in training) stop dominating.
        stride: Encoder output stride.
        template_px: Template side length.

    Returns:
        Scalar loss.
    """
    c = coords_out.float()
    pred = torch.stack([out_to_pixel(c[0], stride, template_px),
                        out_to_pixel(c[1], stride, template_px)])
    target = torch.stack([
        torch.as_tensor(float(gt_px_x), device=c.device, dtype=c.dtype),
        torch.as_tensor(float(gt_px_y), device=c.device, dtype=c.dtype),
    ])
    return F.huber_loss(pred, target, delta=huber_delta)


def _self_test() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    h = w = 60

    # --- soft target -------------------------------------------------------
    t = build_soft_target(h, w, 30.4, 25.7, dev, sigma=1.0)
    assert abs(float(t.sum()) - 1.0) < 1e-5
    assert t.dtype == torch.float32
    ay, ax = divmod(int(t.argmax()), w)
    assert (ax, ay) == (30, 26), (ax, ay)
    floor = soft_target_entropy(1.0)
    print(f"  soft target sums to 1, argmax at ({ax},{ay}), "
          f"entropy floor {floor:.3f} nats")
    assert 2.7 < floor < 3.0

    # A perfect prediction must reach the entropy floor.
    perfect = (t.log() * 1.0).clamp_min(-60.0)
    ce = float(map_softmax_ce(perfect, t))
    assert abs(ce - floor) < 0.02, (ce, floor)
    print(f"  map_softmax_ce on a perfect prediction = {ce:.3f} == floor")

    # Collapse must NOT reduce the loss (the property focal loss lacked).
    flat = torch.zeros(h, w)
    ce_flat = float(map_softmax_ce(flat, t))
    ce_flat_low = float(map_softmax_ce(flat - 50.0, t))
    assert abs(ce_flat - ce_flat_low) < 1e-3
    print(f"  collapse-invariance: flat {ce_flat:.3f} vs flat-50 "
          f"{ce_flat_low:.3f}  (identical -- collapse is not a descent "
          f"direction)")

    # --- lattice replica mask ---------------------------------------------
    m = lattice_replica_mask(h, w, 30.0, 30.0, 6.0, 6.0, dev,
                             tol=0.75, min_sep=2.0, max_k=3)
    assert bool(m[30, 36]) and bool(m[36, 30]) and bool(m[36, 36])
    assert not bool(m[30, 30])                       # never the true peak
    assert not bool(m[30, 31])                       # not a replica
    print(f"  replica mask: {int(m.sum())} cells at pitch 6, "
          f"true peak excluded, +/-1 pitch marked")

    # Fine pitch: the first replica is inside min_sep and must be dropped.
    m_fine = lattice_replica_mask(h, w, 30.0, 30.0, 1.3, 1.3, dev,
                                  tol=0.4, min_sep=2.0, max_k=8)
    near = m_fine[29:32, 29:32]
    assert not bool(near.any()), "min_sep failed to protect the true peak"
    print(f"  fine pitch 1.3 cells: {int(m_fine.sum())} cells, none within "
          f"min_sep of the peak")

    # Zero pitch disables cleanly rather than dividing by zero.
    assert int(lattice_replica_mask(h, w, 30., 30., 0.0, 0.0, dev).sum()) == 0
    print("  pitch=0 disables the term instead of raising")

    # --- lattice margin loss ----------------------------------------------
    lg = torch.full((h, w), -1.0)
    lg[30, 30] = 10.0
    lg[30, 36] = 9.9                                  # a strong replica
    loss_bad, n = lattice_margin_loss(lg, 30.0, 30.0, 6.0, 6.0, margin=4.0)
    lg2 = lg.clone(); lg2[30, 36] = -1.0              # replica suppressed
    loss_good, _ = lattice_margin_loss(lg2, 30.0, 30.0, 6.0, 6.0, margin=4.0)
    print(f"  margin loss: replica present {float(loss_bad):.3f}  "
          f"replica suppressed {float(loss_good):.3f}  ({n} replica cells)")
    assert float(loss_bad) > float(loss_good) >= 0.0

    # Gradient must push the replica down and the true peak up.
    lg3 = lg.clone().requires_grad_(True)
    l, _ = lattice_margin_loss(lg3, 30.0, 30.0, 6.0, 6.0, margin=4.0)
    l.backward()
    g = lg3.grad
    assert float(g[30, 30]) < 0, "true peak should be pushed UP"
    assert float(g[30, 36]) > 0, "replica should be pushed DOWN"
    print(f"  gradient sign: true peak {float(g[30,30]):+.3f}, "
          f"replica {float(g[30,36]):+.3f}  (correct directions)")
    # Hard-negative concentration: the strongest replica gets the mass.
    strongest = float(g[30, 36])
    others = float(g[36, 36])
    assert strongest > others, "logsumexp did not concentrate on the hardest"
    print(f"  hard-negative mining: strongest replica grad {strongest:.4f} "
          f"> weaker replica grad {others:.4f}")

    # --- offset + coordinate ----------------------------------------------
    off = torch.zeros(2, h, w)
    off[0, 26, 30] = 0.4
    off[1, 26, 30] = -0.3
    ol = float(offset_loss(off, 30, 26, 0.4, -0.3))
    assert ol < 1e-6, ol
    print(f"  offset loss at the exact target = {ol:.2e}")

    c = torch.tensor([10.0, 20.0])
    gx, gy = out_to_pixel(10.0), out_to_pixel(20.0)
    cl = float(coordinate_loss(c, gx, gy))
    assert cl < 1e-9, cl
    print(f"  coordinate loss at ground truth = {cl:.2e}")
    print("  OK")


if __name__ == "__main__":
    _self_test()
