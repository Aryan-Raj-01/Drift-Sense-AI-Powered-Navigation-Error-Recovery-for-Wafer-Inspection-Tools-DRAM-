"""dl_localize -- fixed rebuild of driftsense.localize.

Import as ``driftsense.dl_localize``.  The original ``driftsense.localize``
package is left untouched so the 10k-subset checkpoints remain loadable and
the two paths can be compared directly.

Checkpoints from ``driftsense.localize`` are NOT loadable here: the encoder
fusion, the head, and the coordinate convention all changed.  Train from
scratch.  See README88.md for the measured evidence behind each change.
"""

__all__ = ["coords", "data", "model", "losses", "train",
           "infer", "calibrate"]
