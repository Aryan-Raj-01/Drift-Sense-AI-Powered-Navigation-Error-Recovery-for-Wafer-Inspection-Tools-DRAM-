"""Learned dense correlation model -- replaces classical NCC as the proposal
stage.

Why this exists: ``tools/gate_test.py`` measured that raw-intensity NCC's
top-K recall on hard DRAM samples saturates at 73 % even at K=512 (see
``tools/README_STAGE1.md``).  For roughly 27 % of hard samples, no candidate
at any rank is within 5 px of the truth -- uniform-weight pixel correlation
cannot see the signal, because the aperiodic marker (a 1.4-3.4 px defect) is
below the noise floor once averaged against the periodic background at equal
weight.  A learned encoder can weight the marker above the grid; raw pixels
cannot.

Pipeline, end to end, one image pair at a time:

    reference (shrunk to ~100x100)      search (1000x1000)
              |                                |
        shared encoder (stride 4)        shared encoder (stride 4)
              |                                |
          L2-normalise                    L2-normalise
              |________________________________|
                          |
              dense cross-correlation (reference as kernel)
                          |
                    1x1 conv head -> heatmap logits
                          |
              softmax + soft-argmax -> sub-pixel (x, y)

NOT a literal HRNet reimplementation.  HRNet was recommended for one specific
property -- preserving high resolution throughout so a 2 px feature is not
destroyed by downsampling -- and this encoder provides exactly that (output
stride 4, with a stride-2/stride-4 fusion for a taste of multi-scale
context) using a plain residual CNN that can be read, reasoned about, and
debugged without a live training run to catch subtle multi-branch bugs.  If
this proves to be the accuracy bottleneck later, swapping in a full HRNet is
a drop-in replacement -- everything downstream only depends on
``Encoder.forward`` producing a ``(B, C, H/4, W/4)`` tensor.

Batching: correlation kernel size varies slightly per sample (scale ratio
jitter of 9.73-10.27 makes the shrunk reference 97-103 px, hence the
embedding 24-26 px).  Rather than pad templates to a common size within a
batch -- an easy place to introduce a silent shape bug that only shows up on
certain samples -- this module processes one pair per forward call and
relies on gradient accumulation in the training loop for an effective batch
size.  Simpler to verify correct; revisit only if throughput proves
insufficient on measured wall-clock time.

NOTE: torch is not installed in the environment this file was authored in.
Every line has been syntax-checked (``python -m py_compile``) but not
execution-tested.  Run the smoke test in ``train.py --smoke-test`` first, on
real hardware, before committing to a full training run.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    """Batch-independent normalisation.

    GroupNorm, not BatchNorm. This model processes ONE image pair per forward
    pass (see the module docstring on why batching was avoided), and
    BatchNorm with a batch of one estimates its mean and variance from that
    single image -- so the statistics swing wildly step to step, and at
    inference it switches to running averages that never matched any actual
    batch. That mismatch shows up as an unstable predicted peak even while
    the loss falls smoothly. GroupNorm normalises within each sample, so
    training and inference behave identically regardless of batch size.

    Args:
        channels: Channel count; groups are chosen to divide it evenly.

    Returns:
        A GroupNorm module.
    """
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBNAct(nn.Module):
    """Conv -> GroupNorm -> ReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation,
                              bias=False)
        self.bn = _norm(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two 3x3 convs with a residual connection, stride 1."""

    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.c1 = ConvBNAct(c, c, 3, 1, dilation)
        pad = dilation
        self.c2 = nn.Conv2d(c, c, 3, 1, pad, dilation=dilation, bias=False)
        self.bn2 = _norm(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.bn2(self.c2(y))
        return self.act(x + y)


class Encoder(nn.Module):
    """Single-channel grayscale in, stride-4 embedding out.

    Two stride-2 stages give exactly stride 4 -- the constraint that rules
    out standard classification backbones for this task (they reach stride
    32).  The stride-2 feature is fused back in after downsampling to give a
    little multi-scale context without the complexity of a full multi-branch
    architecture.
    """

    C_OUT = 128

    def __init__(self, width: int = 64, n_blocks: int = 4) -> None:
        super().__init__()
        self.stem = ConvBNAct(1, width, k=5, stride=1)
        self.down1 = ConvBNAct(width, width, k=3, stride=2)          # /2
        self.blocks1 = nn.Sequential(*(ResBlock(width) for _ in range(n_blocks)))
        self.down2 = ConvBNAct(width, width * 2, k=3, stride=2)      # /4
        self.blocks2 = nn.Sequential(
            *(ResBlock(width * 2, dilation=2) for _ in range(n_blocks)))
        self.fuse = ConvBNAct(width + width * 2, self.C_OUT, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, (B, 1, H, W). Returns: (B, C_OUT, ~H/4, ~W/4)."""
        s1 = self.blocks1(self.down1(self.stem(x)))
        s2 = self.blocks2(self.down2(s1))
        s1_down = F.avg_pool2d(s1, 2, ceil_mode=False)
        # Stride-2 and stride-4 branches can differ by one pixel from
        # independent rounding; crop to the smaller so concatenation is safe.
        h = min(s1_down.shape[-2], s2.shape[-2])
        w = min(s1_down.shape[-1], s2.shape[-1])
        fused = torch.cat([s1_down[..., :h, :w], s2[..., :h, :w]], dim=1)
        return self.fuse(fused)


def correlate(ref_emb: torch.Tensor, search_emb: torch.Tensor) -> torch.Tensor:
    """Dense cross-correlation, reference embedding used as the kernel.

    Args:
        ref_emb: (1, C, kh, kw), L2-normalised.
        search_emb: (1, C, H, W), L2-normalised.

    Returns:
        (1, 1, H-kh+1, W-kw+1) MEAN cosine-similarity map, values in [-1, 1].

    The division is essential, not cosmetic. ``conv2d`` sums the per-position
    dot products over the whole kernel, so with a 25x25 reference embedding
    the raw output ranges to +/-625. Feeding that into a sigmoid saturates it
    instantly, the focal loss starts in the millions, and gradients are
    unusable. Dividing by the number of kernel positions turns the sum back
    into a mean cosine similarity in [-1, 1], which is what the head and the
    loss both assume.
    """
    n_positions = ref_emb.shape[-1] * ref_emb.shape[-2]
    return F.conv2d(search_emb, weight=ref_emb) / n_positions


def windowed_soft_argmax(logits_2d: torch.Tensor, cx: float, cy: float,
                         radius: int = 6, temperature: float = 1.0
                         ) -> torch.Tensor:
    """Sub-pixel (x, y) from a small window centred on ``(cx, cy)``.

    A soft-argmax over the FULL correlation map does not work here. The map is
    ~226x226 = 51000 cells; unless the logits are extremely peaked, the softmax
    is near-uniform and its expectation is the centre of the map. The model
    then predicts the image centre for every input -- a fixed ~380 px error
    that looks like "not learning" but is really a degenerate readout.

    Restricting to a window makes the softmax well-conditioned: the peak
    dominates its own neighbourhood even when logits are modest, so the output
    is a genuine sub-pixel refinement of the peak location.

    Args:
        logits_2d: (h, w) logits.
        cx: Window centre x, in grid units.
        cy: Window centre y, in grid units.
        radius: Half-width; the window is ``2 * radius + 1`` on a side.
        temperature: Softmax sharpness within the window.

    Returns:
        (2,) tensor, (x, y) in the full map's grid coordinates.
    """
    logits_2d = logits_2d.float()
    h, w = logits_2d.shape
    side = 2 * radius + 1
    x0 = int(max(0, min(round(cx) - radius, w - side)))
    y0 = int(max(0, min(round(cy) - radius, h - side)))

    win = logits_2d[y0:y0 + side, x0:x0 + side]
    p = F.softmax((win * temperature).reshape(-1), dim=0).reshape(side, side)

    ys = torch.arange(side, device=logits_2d.device, dtype=logits_2d.dtype)
    xs = torch.arange(side, device=logits_2d.device, dtype=logits_2d.dtype)
    dx = (p.sum(dim=0) * xs).sum()
    dy = (p.sum(dim=1) * ys).sum()
    return torch.stack([dx + x0, dy + y0])


def peak_xy(logits_2d: torch.Tensor) -> tuple[int, int]:
    """Hard argmax of the logits map.

    Args:
        logits_2d: (h, w) logits.

    Returns:
        ``(x, y)`` integer grid coordinates of the maximum.
    """
    flat = int(torch.argmax(logits_2d.detach()).item())
    w = logits_2d.shape[1]
    return flat % w, flat // w


def predict_xy(logits_2d: torch.Tensor, radius: int = 6) -> torch.Tensor:
    """Peak location refined to sub-pixel -- the inference-time readout.

    Args:
        logits_2d: (h, w) logits.
        radius: Refinement window half-width.

    Returns:
        (2,) tensor, (x, y) in grid coordinates.
    """
    px, py = peak_xy(logits_2d)
    return windowed_soft_argmax(logits_2d, px, py, radius=radius)


class SiameseCorrelationNet(nn.Module):
    """Shared encoder + dense correlation + sub-pixel head.

    Call :meth:`forward` with one reference/search pair (batch size 1 in the
    tensor sense; see module docstring on the batching decision).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        # 1x1 conv: a learnable temperature/bias on the mean cosine score,
        # not a spatial filter -- kernel size 1 is intentional.
        self.head = nn.Conv2d(1, 1, 1)
        # Mean cosine sits near 0 at init, so default init leaves every logit
        # near 0 and the soft-argmax output almost uniform -- no usable
        # gradient. Starting the gain high gives the peak somewhere to grow
        # from immediately.
        nn.init.constant_(self.head.weight, 20.0)
        nn.init.constant_(self.head.bias, 0.0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and L2-normalise along the channel dimension."""
        return F.normalize(self.encoder(x), dim=1, eps=1e-6)

    def forward(self, ref: torch.Tensor, search: torch.Tensor
                ) -> Tuple[torch.Tensor, int]:
        """One reference/search pair.

        Args:
            ref: (1, 1, Kh, Kw) reference, already shrunk to template size.
            search: (1, 1, H, W) search image.

        Returns:
            ``(logits, ref_emb_size)``:
            logits -- (h, w) heatmap logits, ``h = search_emb_h - ref_emb_h + 1``.
            ref_emb_size -- spatial size of the reference embedding, read
            from the tensor at runtime.  Required by
            ``coords.out_to_pixel`` to decode grid coordinates into original
            search pixels -- see that module's docstring for why this must
            never be hard-coded.

        Coordinates are NOT returned here. The readout differs between
        training and inference: training centres the refinement window on
        ground truth (so the sub-pixel loss is well-conditioned from step
        one, before the peak is anywhere near right), while inference centres
        it on the predicted peak. Keeping that choice at the call site makes
        the difference explicit instead of hiding it in the model.
        """
        re = self.embed(ref)
        se = self.embed(search)
        corr = correlate(re, se)
        logits = self.head(corr)[0, 0]
        return logits, re.shape[-1]


def _self_test() -> None:
    """Shape check only -- requires torch, so this does not run in the
    authoring environment. Run manually once torch is available:
        python -m driftsense.localize.model
    """
    torch.manual_seed(0)
    net = SiameseCorrelationNet()
    net.eval()
    ref = torch.randn(1, 1, 100, 100)
    search = torch.randn(1, 1, 1000, 1000)
    with torch.no_grad():
        logits, k = net(ref, search)
        coords = predict_xy(logits)
    print(f"  logits shape   {tuple(logits.shape)}")
    print(f"  coords         {coords.tolist()}")
    print(f"  ref_emb_size   {k}")
    assert logits.dim() == 2
    assert coords.shape == (2,)
    assert 0 <= float(coords[0]) <= logits.shape[1]
    assert 0 <= float(coords[1]) <= logits.shape[0]
    print("  OK")


if __name__ == "__main__":
    _self_test()
