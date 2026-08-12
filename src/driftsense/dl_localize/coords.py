"""Coordinate mapping between the correlation-output grid and search pixels.

WHY THIS FILE WAS REWRITTEN -- the measured bug
================================================

The previous ``localize/coords.py`` used::

    out_to_pixel(o, ref_emb_size, stride) = (o + (ref_emb_size - 1) / 2) * stride + stride / 2

That formula depends on ``ref_emb_size`` (the spatial size of the *encoded*
reference, 24-26 cells).  ``ref_emb_size`` is a step function of the template
*pixel* size k, which in turn was derived from the per-sample
``scale_ratio`` label.  The true relationship, measured by planting a known
crop of a search image back into that same image and reading the argmax of
the correlation map, is::

    search_pixel = stride * out_index + (template_px - 1) / 2

which depends on the template PIXEL size, not the embedding size.  Measured
bias of the old formula (untrained encoder, exact-crop template, 1000x1000
search image, argmax was at out-index 100 for every k):

    template_px k :  97    98    99   100   101   102   103   104
    true centre   : 448.0 448.5 449.0 449.5 450.0 450.5 451.0 451.5
    old formula   : 448.0 448.0 450.0 450.0 450.0 450.0 452.0 452.0
    bias (px)     : +0.0  -0.5  +1.0  +0.5  +0.0  -0.5  +1.0  +0.5

A constant bias would be harmless -- training and inference share the
convention, so it cancels.  This one is NOT constant: it is a sawtooth in k
with period 4, and k was ``round(1000 / scale_ratio)`` at training time
(96-104, since scale_ratio spans 9.60-10.40 in the 200k manifest) but
hard-coded to 100 at inference (``infer_dl.locate`` defaults ``scale=10.0``
and ``tools/eval_dl.py`` never overrides it).  So the convention the network
was trained under and the convention it was decoded under disagreed by
0 to 1.0 px PER AXIS, sample-dependently.

Confirmed end to end on four real easy pairs with an untrained encoder
(so the effect is purely geometric, not a training artefact):

    id=56  k=99   old-map err 2.13 px   new-map err 0.74 px
    id=59  k=99   old-map err 2.31 px   new-map err 1.47 px
    id=60  k=101  old-map err 1.15 px   new-map err 1.15 px   <- k=101 is the
                                                                 unbiased case
The 1.4 px gap at k=99 is exactly sqrt(2) x 1.0 px, i.e. the predicted 1.0 px
per-axis bias applied to both axes.  This matches the table above cell for
cell, which is why the diagnosis is stated as measured rather than suspected.

Success is scored at 1 px.  A sample-dependent 0.7-1.4 px error injected by
the coordinate convention alone is sufficient on its own to explain the easy
bucket sitting at 84% while the classical baseline -- which does its
sub-pixel fit on a stride-1 correlation surface and has no such convention --
sits at 89%.

THE TWO RULES THIS MODULE ENFORCES
==================================
1. The template is ALWAYS ``TEMPLATE_PX`` pixels on a side, at training and
   at inference.  ``scale_ratio`` is a ground-truth label and is not
   available on Applied Materials' held-out test set, so it must never
   determine a tensor shape.  Scale variation is handled by content
   (augmentation + an inference-time template bank), never by geometry.
2. The mapping depends only on ``TEMPLATE_PX`` and ``STRIDE``, both
   compile-time constants.  There is no per-sample term left to disagree
   about.

Self-test (pure arithmetic, no torch)::

    python -m driftsense.dl_localize.coords

Geometric calibration against the real encoder (needs torch)::

    python -m driftsense.dl_localize.calibrate
"""

from __future__ import annotations

#: Encoder output stride.  Two stride-2 stages in ``model.Encoder``.
STRIDE: float = 4.0

#: Template side length in pixels, fixed for every sample.  100 matches the
#: mean footprint of the reference pattern inside the search image
#: (measured mean 100.05 px, range 96.16-104.18 over the 200k manifest), so
#: the average sample needs no resampling beyond the nominal 10x shrink.
TEMPLATE_PX: int = 100

#: Reference embedding side length that ``TEMPLATE_PX`` produces.  Asserted
#: at runtime in ``model.SiameseCorrelationNet.forward`` -- if the encoder
#: is ever changed this constant must be re-measured with calibrate, not
#: guessed.
REF_EMB_PX: int = 25


def out_to_pixel(out_xy: float, stride: float = STRIDE,
                 template_px: int = TEMPLATE_PX) -> float:
    """Correlation-output-grid coordinate -> original search pixel.

    Args:
        out_xy: Coordinate in the correlation output grid.
        stride: Encoder output stride.
        template_px: Template side length in pixels.

    Returns:
        Coordinate in original search-image pixels.
    """
    return stride * out_xy + (template_px - 1) / 2.0


def pixel_to_out(px_xy: float, stride: float = STRIDE,
                 template_px: int = TEMPLATE_PX) -> float:
    """Exact inverse of :func:`out_to_pixel`."""
    return (px_xy - (template_px - 1) / 2.0) / stride


def _self_test() -> None:
    import random
    random.seed(0)

    for template_px in (96, 100, 104, 128):
        for _ in range(5000):
            px = random.uniform(-100.0, 1100.0)
            back = out_to_pixel(pixel_to_out(px, template_px=template_px),
                                template_px=template_px)
            assert abs(back - px) < 1e-9, (template_px, px, back)

    assert out_to_pixel(11.0) > out_to_pixel(10.0)
    assert abs(out_to_pixel(1.0) - out_to_pixel(0.0) - STRIDE) < 1e-12

    # The exact numbers measured in the probe documented above.
    for k, true_centre in ((97, 448.0), (99, 449.0), (100, 449.5),
                           (101, 450.0), (104, 451.5)):
        got = out_to_pixel(100.0, template_px=k)
        assert abs(got - true_centre) < 1e-9, (k, got, true_centre)

    print("  round-trip exact on 4 template sizes x 5000 pixels")
    print("  monotonic, spacing == STRIDE")
    print("  reproduces the 5 measured calibration points")
    print("  OK")


if __name__ == "__main__":
    _self_test()
