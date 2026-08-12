"""Dataset for the learned dense correlation model.

Reads the same manifest (``labels.jsonl``) the classical pipeline uses, so no
new data preparation step is needed -- the 200,000 pairs already on disk are
the training set directly.

Curriculum: pass ``difficulties=("easy",)``, then ``("easy", "medium")``, then
``("easy", "medium", "hard")`` across training phases. Starting on hard alone
gives the network almost no signal to learn from (see the roadmap's
discussion of the ~0.05 % aperiodic energy fraction on hard samples) and risks
collapsing to a centre-biased prediction.

Lattice pitch, needed by ``losses.lattice_weight_map``, is read directly from
the manifest (``layout_wl_pitch``, ``layout_bl_pitch``, both nanometres) and
converted to correlation-output grid units via ``search_px_nm`` and the
encoder stride -- not re-measured from the image, since the generator already
recorded it exactly.

NOTE: syntax-checked only (no torch in the authoring environment). See
``model.py`` module docstring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from driftsense.localize.coords import pixel_to_out, shrink_size

#: Encoder output stride. Must match model.Encoder (two stride-2 stages).
STRIDE = 4.0


def _resolve(root: Path, path: str) -> str:
    """Manifest paths may be absolute or relative to the dataset root."""
    return path if os.path.isabs(path) else str(root / path)


def _normalize(img: np.ndarray) -> torch.Tensor:
    """uint8 grayscale -> (1, H, W) float tensor, roughly zero-mean unit-ish.

    Args:
        img: (H, W) uint8 array.

    Returns:
        (1, H, W) float32 tensor.
    """
    a = img.astype(np.float32) / 255.0
    a = (a - 0.5) / 0.25
    return torch.from_numpy(a).unsqueeze(0)


class DriftSensePairs(Dataset):
    """One reference/search pair per item, plus everything the loss needs.

    Returns a dict rather than a tuple -- the training loop needs several
    scalars (ground truth, pitch) alongside the two images, and a dict keeps
    that self-documenting at the call site.
    """

    def __init__(self, root: str, difficulties: Sequence[str] = ("easy",),
                seed_filter: Optional[Sequence[int]] = None) -> None:
        """
        Args:
            root: Dataset directory containing ``labels.jsonl``.
            difficulties: Which difficulty buckets to include. Pass a growing
                tuple across curriculum phases -- see module docstring.
            seed_filter: If given, keep only rows whose ``seed`` is in this
                set. Used to carve out the val/golden split without touching
                files on disk.
        """
        self.root = Path(root)
        rows: List[Dict] = []
        want = set(difficulties)
        with open(self.root / "labels.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["difficulty"] not in want:
                    continue
                if seed_filter is not None and r["seed"] not in seed_filter:
                    continue
                rows.append(r)
        if not rows:
            raise ValueError(
                f"No rows matched difficulties={difficulties} in {root}. "
                "Check the manifest path and the seed_filter, if any.")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]

        ref_img = cv2.imread(_resolve(self.root, r["reference_path"]),
                             cv2.IMREAD_GRAYSCALE)
        srch_img = cv2.imread(_resolve(self.root, r["search_path"]),
                              cv2.IMREAD_GRAYSCALE)
        if ref_img is None or srch_img is None:
            raise FileNotFoundError(
                f"Could not read images for id={r['id']} in {self.root}")

        scale = float(r["scale_ratio"])
        k = shrink_size(ref_img.shape[0], scale)
        ref_small = cv2.resize(ref_img, (k, k), interpolation=cv2.INTER_AREA)

        ref_t = _normalize(ref_small)
        srch_t = _normalize(srch_img)

        # Ground-truth pitch, in original search pixels, then to grid units.
        # A missing or non-positive pitch (should not occur for DRAM, but
        # guarded rather than assumed) disables the lattice-negative boost
        # for that sample -- see losses.lattice_weight_map.
        px_nm = float(r.get("search_px_nm", 0.0))
        pitch_x_px = float(r.get("layout_bl_pitch", 0.0)) / px_nm if px_nm > 0 else 0.0
        pitch_y_px = float(r.get("layout_wl_pitch", 0.0)) / px_nm if px_nm > 0 else 0.0

        return {
            "id": r["id"],
            "difficulty": r["difficulty"],
            "ref": ref_t,
            "search": srch_t,
            "gt_x": float(r["gt_x"]),
            "gt_y": float(r["gt_y"]),
            "pitch_x_px": pitch_x_px,
            "pitch_y_px": pitch_y_px,
            "pitch_x_out": pitch_x_px / STRIDE,
            "pitch_y_out": pitch_y_px / STRIDE,
        }


def gt_to_out_grid(sample: Dict, ref_emb_size: int) -> tuple[float, float]:
    """Ground-truth location in the correlation-output grid.

    Args:
        sample: One item from :class:`DriftSensePairs`.
        ref_emb_size: Reference embedding spatial size, from the model
            forward pass -- read at runtime, never hard-coded.

    Returns:
        ``(gt_out_x, gt_out_y)``.
    """
    return (pixel_to_out(sample["gt_x"], ref_emb_size, STRIDE),
            pixel_to_out(sample["gt_y"], ref_emb_size, STRIDE))


def _self_test() -> None:
    """Requires torch and real data; not runnable in the authoring
    environment. Run manually once torch is available:
        python -m driftsense.localize.data --dataset data/eval
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    ds = DriftSensePairs(args.dataset, difficulties=("easy", "medium", "hard"))
    print(f"  dataset size: {len(ds)}")
    item = ds[0]
    print(f"  ref shape:    {tuple(item['ref'].shape)}")
    print(f"  search shape: {tuple(item['search'].shape)}")
    print(f"  gt:           ({item['gt_x']:.2f}, {item['gt_y']:.2f})")
    print(f"  pitch (px):   ({item['pitch_x_px']:.2f}, {item['pitch_y_px']:.2f})")
    assert item["ref"].shape[0] == 1
    assert item["search"].shape == (1, 1000, 1000)
    print("  OK")


if __name__ == "__main__":
    _self_test()
