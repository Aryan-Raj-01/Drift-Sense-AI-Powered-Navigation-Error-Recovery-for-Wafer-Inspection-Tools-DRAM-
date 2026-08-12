#!/usr/bin/env python3
"""Drift-Sense -- synthetic dataset generator entry point.

Produces paired grayscale SEM-style images: a 1000x1000 reference at nominal
100x magnification and a 1000x1000 search image at nominal 10x, with the
reference pattern placed at a known sub-pixel location inside the search image.

    python generate_dataset.py --style dram   --num-images 200 --out data/my_set --format png
    python generate_dataset.py --style finfet --num-images 200 --out data/my_set --format png

REQUIRED PARAMETERS
===================

    --style {dram,finfet,mixed}   architecture family
    --num-images N                number of pairs to generate
    --out DIR                     output directory

GROUND TRUTH
============

Every pair records the true centre of the reference pattern in search-image
pixels (``gt_x``, ``gt_y``), top-left origin, as sub-pixel floats. Ground truth
is computed from the geometric transform used to place the pattern, not
recovered by matching afterwards, so it is exact up to the documented
``label_correction_px`` term that accounts for non-rigid scan distortion.

Alongside the coordinates each row stores the random seed, architecture style,
applied rotation and scale, the full noise/optics parameter set, lattice pitch,
landmark and defect geometry, and the difficulty bucket -- everything needed to
regenerate that exact pair bit-for-bit.

OUTPUT FORMATS
==============

    --format manifest   labels only (default). ~25 MB per 100k pairs. Pixels
                        are regenerated on demand from the seed, because every
                        sample is a pure function of its seed.
    --format png        materialise the image files. ~1 MB per pair, so 200
                        pairs is ~200 MB and 100k pairs is ~100 GB.
    --format npz        materialise as compressed arrays.

Use ``png`` for anything a human or an evaluator will open; use ``manifest``
for large training sets, where writing the pixels costs far more than
computing them.

REPRODUCIBILITY
===============

``--seed`` fixes the dataset root seed; per-pair seeds are derived from it up
front in the parent process, so the same command reproduces the same dataset
regardless of ``--workers``. See ``src/driftsense/rng.py`` for the derivation.

The literature basis for the structure, noise and degradation choices is in
``references/references.md``.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from driftsense.cli.generate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
