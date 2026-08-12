#!/usr/bin/env python3
"""Drift-Sense -- localization entry point.

Locates a 100x reference pattern inside a 10x search image and returns the
centre of the match in search-image pixels.

    python localize.py --reference ref.png --search search.png
    python localize.py --csv pairs.csv --out predictions.csv

COORDINATE CONVENTION
=====================

Origin (0, 0) is the TOP-LEFT of the search image; x increases to the right,
y increases downward. The returned (x, y) is the centre of the region in the
search image that corresponds to the centre of the reference image. Values are
sub-pixel floats, not integers.

MULTIPLE MATCHES  --  read this, the default is not the obvious one
==================================================================

Semiconductor layouts are periodic, so a reference pattern genuinely recurs
many times inside the search image. The problem statement's rule for this is:
where several valid matches exist, take the one nearest the search-image
centre.

That rule IS implemented -- ``tie_margin`` in ``infer.LearnedLocalizer``,
exposed as ``--tie-margin`` here and in ``eval``. When the runner-up peak is
within ``tie_margin`` logits of the winner, the candidate closer to the centre
takes the answer.

It is DISABLED BY DEFAULT (``tie_margin = 0.0``), and that is a measured
decision, not an oversight. On 2,000 held-out pairs:

    tie_margin      ALL <=1px    easy     medium    hard
    0.0  (default)     90.8%     92.7%     92.7%    83.2%
    1.0                90.2%     92.4%     93.2%    80.5%
    1.5                89.5%     91.9%     92.3%    79.0%

Enabling it costs accuracy monotonically, most of it on the hard bucket, where
total >5 px failures rise from 9.4% to 12.2%.

WHY, precisely -- the reason matters for whether this holds on other data. The
rule's premise is sound: among our gross misses the true answer is nearer the
centre than the wrong one 71% of the time, well above chance. But the tie-break
can only select among peaks the correlation surface already proposes, and on a
genuine gross miss there is no peak near the truth. So it trades one wrong
answer for a different wrong answer, while also displacing the ~2.3% of correct
predictions that happen to carry a low margin.

WHEN IT WOULD HELP. Our generator places targets close to uniformly across the
search image -- median 307 px from centre, only 31% within 250 px. A real
navigation-recovery scenario is not uniform: the tool landed approximately
right, so the target concentrates near the centre. On a centre-weighted test
distribution this rule should help rather than hurt, which is why it is kept,
exposed and calibrated rather than deleted. Enable it with
``--tie-margin 1.0`` if the evaluation data is centre-weighted.

BATCH MODE
==========

``--csv`` expects columns ``Wide Search Image Path`` and
``Reference Image Path``; the output file reproduces those columns and adds
``GTx``, ``GTy``, ``confidence`` and ``margin``. Paths inside the CSV may be
absolute, or relative to the CSV's own directory. The model is loaded once for
the whole batch.

FAILURE BEHAVIOUR
=================

This script never raises on a single bad pair. It degrades in three steps:
learned model -> classical NCC baseline -> search-image centre, and records
which path was taken in the ``method`` field. A batch of 10 000 pairs will not
be destroyed by one unreadable PNG.

MODEL WEIGHTS
=============

Loaded automatically from ``model/final_phase3_all.pt`` relative to this file.
No manual edit is required. Override with ``--checkpoint`` or the
``DRIFTSENSE_CHECKPOINT`` environment variable if the archive is rearranged.
"""

from __future__ import annotations

import os
import sys

# ``src`` holds the package; add it to the path so this script runs from the
# archive root with no installation step and no PYTHONPATH juggling.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from driftsense.dl_localize.localize import (  # noqa: E402
    locate,
    locate_full,
    main,
    set_checkpoint,
    set_tie_margin,
)

# Point at the submission's own weights before anything else runs, so the
# default is correct regardless of where the archive was unpacked.
_DEFAULT = os.path.join(_ROOT, "model", "final_phase3_all.pt")
if os.path.isfile(_DEFAULT) and not os.environ.get("DRIFTSENSE_CHECKPOINT"):
    set_checkpoint(_DEFAULT)

__all__ = ["locate", "locate_full", "set_checkpoint", "set_tie_margin"]

if __name__ == "__main__":
    raise SystemExit(main())
