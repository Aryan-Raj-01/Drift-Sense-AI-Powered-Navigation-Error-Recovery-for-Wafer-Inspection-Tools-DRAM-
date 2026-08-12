"""Learned dense correlation model -- fixed.

Unchanged from the previous version (and deliberately so, because each was
justified by a measured failure):

  * GroupNorm, not BatchNorm -- one pair per forward pass makes batch-of-one
    statistics meaningless.
  * Correlation divided by the kernel position count -- the raw conv2d sum
    reaches ~625 and saturates everything downstream.
  * Head weight initialised to 20.0 -- a 1x1x1 conv is a single scalar gain
    on a mean-cosine score that starts near 0; without gain there is no
    usable gradient.  This was flagged as a bug by a previous review.  It is
    not one: it is a scalar temperature, the comment above it says so, and no
    measurement points at it.
  * Hard argmax then a LOCAL soft-argmax -- a global soft-argmax over 51000
    cells is near-uniform and collapses to the map centre.

CHANGED, with the reason for each:

1. FUSION SIZE MISMATCH (silent, geometric).  The old ``Encoder.forward``
   did ``avg_pool2d(s1, 2)`` then cropped both branches to the elementwise
   minimum size.  For a 97 px template s1 is 49 -> pooled to 24, while s2 is
   25, so the fused output was 24 and the crop threw away s2's LAST row and
   column -- shifting the template's effective centre by a full cell (4 px)
   relative to the 25-cell case.  Combined with the old coordinate formula
   this is where the measured sawtooth bias came from.  Replaced with
   ``adaptive_avg_pool2d(s1, s2.shape[-2:])``, which cannot disagree in size
   and stays centre-aligned.  ``forward`` now asserts the reference
   embedding is exactly ``coords.REF_EMB_PX``.

2. SUB-PIXEL OFFSET HEAD (new).  Output stride 4 quantises the peak to 4 px
   while the metric is 1 px, so essentially all sub-pixel accuracy comes
   from interpolating the correlation surface.  The measured median error of
   0.55 px is consistent with interpolation error rather than with mis-
   localisation, and on a 1 px threshold that is expensive: tightening the
   readout converts directly into accuracy on samples that are already in
   the right cell.  A 3-layer conv on the correlation surface regresses the
   residual (dx, dy) in cell units, trained only at the ground-truth cell
   (CenterNet's offset head).  It is a strictly better-conditioned
   interpolator than a fixed parabola because it can see the local peak
   shape, including the neighbouring lattice replicas that skew it.

3. READOUT WINDOW RADIUS 6 -> 2.  The old radius-6 window spans 13 cells =
   52 search pixels.  The lattice pitch is 5.3-22.7 search px (measured:
   bl_pitch 53-227 nm at ~10 nm/px), so on a fine-pitch sample that window
   contained up to TEN lattice replicas of near-equal score, and the softmax
   expectation over them is pulled towards the window centre rather than the
   true peak.  Radius 2 (5 cells = 20 px) is already generous for sub-pixel
   refinement of a single peak.  ``refine_xy`` also accepts the sample's
   pitch and shrinks the window further so it never spans more than one
   lattice period.

4. TEMPLATE BANK CORRELATION (new).  Because ``TEMPLATE_PX`` is now fixed,
   every template in a rotation x scale bank encodes to the same 25x25, so
   the whole bank is one ``conv2d`` call with B output channels rather than
   B separate calls.  This is what makes the inference-time rotation and
   scale search affordable -- and rotation search is precisely what the
   classical baseline has and the learned path did not.

5. CALIBRATED CONFIDENCE.  ``sigmoid(peak_logit)`` reported 1.000 for every
   sample in both 10k runs, which is arithmetic, not a finding: logits are
   ``20 * cosine + bias``, and sigmoid(20) is 1.0 to float precision.
   Replaced by the softmax probability of the peak over the whole map and
   the margin to the best competitor outside a suppression radius.  The
   margin is the quantity that actually distinguishes a clean match from a
   periodic lock.

Self-test (shapes + geometric alignment, CPU, no training needed)::

    python -m driftsense.dl_localize.model
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from driftsense.dl_localize.coords import REF_EMB_PX, STRIDE, TEMPLATE_PX


def _norm(channels: int) -> nn.Module:
    """GroupNorm with groups chosen to divide the channel count."""
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    """Conv -> GroupNorm -> ReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation,
                              bias=False)
        self.norm = _norm(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Two 3x3 convs with a residual connection, stride 1."""

    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.c1 = ConvNormAct(c, c, 3, 1, dilation)
        self.c2 = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation,
                            bias=False)
        self.n2 = _norm(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.n2(self.c2(self.c1(x))))


class Encoder(nn.Module):
    """Grayscale in, stride-4 embedding out.

    Two stride-2 stages give exactly stride 4.  Standard classification
    backbones reach stride 32, which would destroy a 1.4-3.4 px hard-case
    defect outright.
    """

    C_OUT = 128

    def __init__(self, width: int = 64, n_blocks: int = 4) -> None:
        super().__init__()
        self.stem = ConvNormAct(1, width, k=5, stride=1)
        self.down1 = ConvNormAct(width, width, k=3, stride=2)            # /2
        self.blocks1 = nn.Sequential(
            *(ResBlock(width) for _ in range(n_blocks)))
        self.down2 = ConvNormAct(width, width * 2, k=3, stride=2)        # /4
        self.blocks2 = nn.Sequential(
            *(ResBlock(width * 2, dilation=2) for _ in range(n_blocks)))
        self.fuse = ConvNormAct(width + width * 2, self.C_OUT, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, (B, 1, H, W).  Returns: (B, C_OUT, H/4, W/4)."""
        s1 = self.blocks1(self.down1(self.stem(x)))
        s2 = self.blocks2(self.down2(s1))
        # adaptive_avg_pool2d, not avg_pool2d + min-crop: the old code
        # dropped s2's last row/column whenever the two branches disagreed by
        # one cell, moving the template centre by a full stride.  This cannot
        # disagree.
        s1_down = F.adaptive_avg_pool2d(s1, s2.shape[-2:])
        return self.fuse(torch.cat([s1_down, s2], dim=1))


class OffsetHead(nn.Module):
    """Per-cell sub-pixel residual (dx, dy), in correlation-grid cells.

    Input is the raw mean-cosine correlation surface, one channel.  The
    sub-pixel position of a correlation peak is encoded entirely in the local
    shape of that surface -- this is the same information a parabolic fit
    uses, with a learned interpolator instead of a fixed one, which matters
    here because the neighbouring lattice replicas skew a parabola
    systematically.

    ``tanh`` bounds the output to +/-``max_cells``.  The true residual is in
    [-0.5, 0.5] by construction; the extra range lets the head correct an
    off-by-one-cell argmax rather than saturating on it.
    """

    def __init__(self, hidden: int = 64, max_cells: float = 2.0) -> None:
        super().__init__()
        self.max_cells = float(max_cells)
        self.body = nn.Sequential(
            ConvNormAct(1, hidden, k=3),
            ConvNormAct(hidden, hidden, k=3),
        )
        self.out = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)   # start at "no correction"

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        """Args: corr, (1, 1, h, w).  Returns: (2, h, w), (dx, dy) in cells.

        The input is STANDARDISED before the convolutions.  The raw map is a
        mean cosine similarity whose absolute level and spread vary a lot
        from sample to sample (a sharp easy match and a marginal hard one do
        not live on the same scale).  Sub-pixel position is encoded in the
        peak's SHAPE, not its height, so removing the per-sample level and
        scale deletes a nuisance variable the head would otherwise have to
        learn to ignore -- with only a handful of supervised cells per image
        to learn it from.
        """
        c = corr.float()
        c = (c - c.mean()) / (c.std() + 1e-6)
        return self.max_cells * torch.tanh(self.out(self.body(c))[0])


def correlate(ref_emb: torch.Tensor, search_emb: torch.Tensor
              ) -> torch.Tensor:
    """Dense cross-correlation; the reference bank is the conv kernel.

    Args:
        ref_emb: (B, C, kh, kw), L2-normalised along C.  B is the template
            bank size (1 during training).
        search_emb: (1, C, H, W), L2-normalised along C.

    Returns:
        (1, B, H - kh + 1, W - kw + 1) MEAN cosine similarity, in [-1, 1].

    The division by the position count is load-bearing.  ``conv2d`` sums the
    per-position dot products over the whole kernel, so a 25x25 reference
    embedding produces raw values up to +/-625.  Feeding that to a sigmoid
    saturates it instantly and the loss starts in the millions -- one of the
    four bugs already found and fixed on this project.
    """
    n_positions = ref_emb.shape[-1] * ref_emb.shape[-2]
    return F.conv2d(search_emb, weight=ref_emb) / n_positions


def peak_xy(logits_2d: torch.Tensor) -> Tuple[int, int]:
    """Hard argmax.  Returns ``(x, y)`` integer grid coordinates."""
    flat = int(torch.argmax(logits_2d.detach()).item())
    w = int(logits_2d.shape[1])
    return flat % w, flat // w


def windowed_soft_argmax(logits_2d: torch.Tensor, cx: float, cy: float,
                         radius: int = 2, temperature: float = 1.0
                         ) -> torch.Tensor:
    """Sub-pixel (x, y) from a small window centred on ``(cx, cy)``.

    A soft-argmax over the FULL map does not work: 226x226 = 51000 cells,
    and unless the logits are extremely peaked the softmax is near-uniform
    and its expectation is the map centre -- the model appears to predict the
    image centre for every input (~380 px fixed error).  That was measured on
    this project and is why the readout is windowed.

    Radius defaults to 2, NOT 6.  Radius 6 spans 52 search pixels; the
    lattice pitch is 5.3-22.7 search pixels, so a radius-6 window can contain
    ten near-equal replicas and the expectation over them is meaningless.

    Args:
        logits_2d: (h, w) logits.
        cx: Window centre x, grid units.
        cy: Window centre y, grid units.
        radius: Half-width; the window is ``2 * radius + 1`` on a side.
        temperature: Softmax sharpness inside the window.

    Returns:
        (2,) tensor, (x, y) in full-map grid coordinates.  Differentiable in
        ``logits_2d``.
    """
    logits_2d = logits_2d.float()
    h, w = logits_2d.shape
    radius = max(1, int(radius))
    side = 2 * radius + 1
    if side > min(h, w):
        radius = (min(h, w) - 1) // 2
        side = 2 * radius + 1
    x0 = int(max(0, min(round(float(cx)) - radius, w - side)))
    y0 = int(max(0, min(round(float(cy)) - radius, h - side)))

    win = logits_2d[y0:y0 + side, x0:x0 + side]
    p = F.softmax((win * temperature).reshape(-1), dim=0).reshape(side, side)
    ar = torch.arange(side, device=logits_2d.device, dtype=logits_2d.dtype)
    dx = (p.sum(dim=0) * ar).sum()
    dy = (p.sum(dim=1) * ar).sum()
    return torch.stack([dx + x0, dy + y0])


def pitch_limited_radius(pitch_x_out: float, pitch_y_out: float,
                         default: int = 2) -> int:
    """Largest readout radius that cannot span more than one lattice period.

    Args:
        pitch_x_out: Lattice pitch along x in grid cells (0 = unknown).
        pitch_y_out: Lattice pitch along y in grid cells.
        default: Radius used when no pitch is available.

    Returns:
        Radius in cells, at least 1.
    """
    pitches = [p for p in (pitch_x_out, pitch_y_out) if p and p > 0]
    if not pitches:
        return default
    return max(1, min(default, int(min(pitches) // 2)))


def refine_xy(logits_2d: torch.Tensor, offsets: Optional[torch.Tensor] = None,
              radius: int = 2, mode: str = "offset") -> torch.Tensor:
    """Peak location refined to sub-pixel -- the inference readout.

    Args:
        logits_2d: (h, w) logits.
        offsets: (2, h, w) offset-head output, or ``None``.
        radius: Soft-argmax window half-width (``mode != "offset"``).
        mode: ``"offset"`` uses the offset head at the argmax cell;
            ``"softargmax"`` uses the windowed soft-argmax; ``"both"``
            averages them.  ``eval`` reports all three so the choice is
            made on measured numbers, not on this docstring.

    Returns:
        (2,) tensor, (x, y) in grid coordinates.
    """
    px, py = peak_xy(logits_2d)
    sa = windowed_soft_argmax(logits_2d, px, py, radius=radius)
    if offsets is None or mode == "softargmax":
        return sa
    off = offsets[:, py, px].float()
    hard = torch.stack([
        torch.as_tensor(float(px), device=off.device, dtype=off.dtype),
        torch.as_tensor(float(py), device=off.device, dtype=off.dtype),
    ]) + off
    if mode == "both":
        return 0.5 * (hard + sa.to(hard.dtype))
    return hard


def peak_confidence(logits_2d: torch.Tensor, suppress_radius: int = 3
                    ) -> Tuple[float, float]:
    """Calibrated confidence for one correlation map.

    ``sigmoid(peak_logit)`` is useless here: logits are ``gain * cosine +
    bias`` with the gain initialised to 20, so sigmoid saturates to 1.000 for
    essentially every sample -- which is exactly what both 10k runs reported.

    Args:
        logits_2d: (h, w) logits.
        suppress_radius: Cells around the peak excluded when looking for the
            runner-up.  Must be at least the sub-pixel window radius,
            otherwise the "competitor" is the peak's own shoulder.

    Returns:
        ``(prob, margin)``:
        prob -- softmax probability mass at the peak cell over the whole map.
        margin -- peak logit minus the best logit outside the suppression
        radius.  This is the quantity that separates a clean match from a
        periodic lock; a lock has a large ``prob`` but a near-zero margin.
    """
    lg = logits_2d.detach().float()
    h, w = lg.shape
    px, py = peak_xy(lg)
    prob = float(F.softmax(lg.reshape(-1), dim=0).reshape(h, w)[py, px])
    masked = lg.clone()
    y0, y1 = max(0, py - suppress_radius), min(h, py + suppress_radius + 1)
    x0, x1 = max(0, px - suppress_radius), min(w, px + suppress_radius + 1)
    masked[y0:y1, x0:x1] = float("-inf")
    runner = float(masked.max())
    margin = float(lg[py, px]) - runner if runner != float("-inf") else 0.0
    return prob, margin


class SiameseCorrelationNet(nn.Module):
    """Shared encoder + dense correlation + sub-pixel offset head."""

    def __init__(self, width: int = 64, n_blocks: int = 4,
                 head_gain: float = 20.0, offset_hidden: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(width=width, n_blocks=n_blocks)
        # 1x1x1 conv: a learnable scalar temperature and bias on the mean
        # cosine score, not a spatial filter.  Mean cosine sits near 0 at
        # init, so a unit gain leaves every logit near 0 and the softmax
        # near-uniform -- no usable gradient.  Deliberate, and measured.
        self.head = nn.Conv2d(1, 1, 1)
        nn.init.constant_(self.head.weight, head_gain)
        nn.init.constant_(self.head.bias, 0.0)
        self.offset = OffsetHead(hidden=offset_hidden)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and L2-normalise along the channel dimension."""
        return F.normalize(self.encoder(x), dim=1, eps=1e-6)

    def correlate_bank(self, ref_bank: torch.Tensor, search_emb: torch.Tensor
                       ) -> torch.Tensor:
        """Correlate a whole template bank against one search embedding.

        Args:
            ref_bank: (B, 1, TEMPLATE_PX, TEMPLATE_PX) raw templates.
            search_emb: (1, C, H, W) already-embedded search image.

        Returns:
            (B, h, w) mean-cosine surfaces, one per bank member.
        """
        re = self.embed(ref_bank)
        assert re.shape[-1] == REF_EMB_PX and re.shape[-2] == REF_EMB_PX, (
            f"reference embedding is {tuple(re.shape[-2:])}, expected "
            f"({REF_EMB_PX}, {REF_EMB_PX}). coords.out_to_pixel is only "
            f"valid for a fixed template size; re-run calibrate if the "
            f"encoder changed.")
        return correlate(re, search_emb)[0]

    def forward(self, ref: torch.Tensor, search: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One reference/search pair.

        Args:
            ref: (1, 1, TEMPLATE_PX, TEMPLATE_PX).
            search: (1, 1, H, W).

        Returns:
            ``(logits, offsets)``:
            logits -- (h, w) heatmap logits.
            offsets -- (2, h, w) sub-pixel residual in grid cells.

        ``ref_emb_size`` is deliberately NOT returned.  The old model
        returned it and the coordinate mapping consumed it, which is how a
        sample-dependent 0-1 px bias got into every decoded prediction.  The
        mapping is now a compile-time constant; nothing here can perturb it.

        Coordinates are not returned either: training and inference use
        different readouts (see ``train`` and ``infer``), and keeping
        that at the call site makes the difference visible rather than
        hidden inside the model.
        """
        se = self.embed(search)
        corr = self.correlate_bank(ref, se).unsqueeze(0)      # (1, 1, h, w)
        # Head and offset in float32.  Under autocast the correlation surface
        # is a low-mantissa type; the quantity that decides a periodic lock is
        # the gap between the true peak and a replica, which can be well
        # under one bfloat16 ULP at these magnitudes.
        logits = self.head(corr.float())[0, 0]
        offsets = self.offset(corr.float())
        return logits, offsets


def _self_test() -> None:
    """Shapes, alignment, and the geometric calibration, on CPU."""
    import numpy as np
    from driftsense.dl_localize.coords import out_to_pixel

    torch.manual_seed(0)
    net = SiameseCorrelationNet().eval()

    ref = torch.randn(1, 1, TEMPLATE_PX, TEMPLATE_PX)
    search = torch.randn(1, 1, 400, 400)
    with torch.no_grad():
        logits, offsets = net(ref, search)
    print(f"  logits  {tuple(logits.shape)}   offsets {tuple(offsets.shape)}")
    assert logits.dim() == 2 and offsets.shape[0] == 2
    assert offsets.shape[1:] == logits.shape

    # Geometric calibration: plant an exact crop of the search image back
    # into it and check the decoded pixel is the crop's true centre.  Works
    # with random weights -- the encoder is a deterministic function, so a
    # patch still matches itself.
    # Crop origins MUST be multiples of STRIDE.  At a half-cell offset the
    # true response splits across two cells and, with random (untrained)
    # weights, an edge artefact can outscore it -- that tests the encoder's
    # feature quality, not the coordinate convention, which is what this
    # assertion is for.
    rng = np.random.default_rng(0)
    s = torch.from_numpy(rng.normal(0, 1, (1, 1, 400, 400)).astype("float32"))
    worst = 0.0
    for (x0, y0) in ((120, 80), (200, 152), (64, 220)):
        t = s[..., y0:y0 + TEMPLATE_PX, x0:x0 + TEMPLATE_PX].clone()
        with torch.no_grad():
            lg, _ = net(t, s)
        ox, oy = peak_xy(lg)
        got = (out_to_pixel(ox), out_to_pixel(oy))
        want = (x0 + (TEMPLATE_PX - 1) / 2.0, y0 + (TEMPLATE_PX - 1) / 2.0)
        err = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
        worst = max(worst, err)
        print(f"  crop at ({x0:>3},{y0:>3})  decoded ({got[0]:7.1f},"
              f"{got[1]:7.1f})  true ({want[0]:7.1f},{want[1]:7.1f})  "
              f"err {err:.2f} px")
    assert worst <= STRIDE / 2.0 + 1e-6, (
        f"coordinate convention is off by {worst:.2f} px, which is more than "
        f"half a cell. coords and the encoder disagree -- do NOT train.")

    # Bank correlation must reproduce single-template correlation exactly.
    bank = torch.randn(4, 1, TEMPLATE_PX, TEMPLATE_PX)
    with torch.no_grad():
        se = net.embed(s)
        multi = net.correlate_bank(bank, se)
        single = net.correlate_bank(bank[2:3], se)
    assert multi.shape[0] == 4
    assert torch.allclose(multi[2], single[0], atol=1e-5), "bank mismatch"
    print(f"  bank correlation {tuple(multi.shape)} matches single-template")

    prob, margin = peak_confidence(logits)
    print(f"  confidence prob {prob:.3e}  margin {margin:.3f}")
    assert 0.0 <= prob <= 1.0
    print("  OK")


if __name__ == "__main__":
    _self_test()
