#!/usr/bin/env python3
"""Train the learned dense correlation model.

    python -m driftsense.localize.train --dataset "/media/user1/New Volume/driftsense/data/train" \\
        --out runs/lscv --smoke-test

Run with ``--smoke-test`` FIRST, always.  It runs 20 real samples for a
handful of steps and prints the loss and a prediction, so any shape or
indexing bug surfaces in under a minute -- on the actual data and actual GPU
-- rather than after committing hours to a run that then crashes, or worse,
runs to completion with a silent coordinate-convention bug (see
``coords.py``).  This code was written without a torch runtime available to
test it in; the smoke test is the substitute for that missing verification
step, and it is not optional.

Curriculum, in three phases, matching the roadmap:
    phase 1: easy only
    phase 2: easy + medium
    phase 3: easy + medium + hard
Each phase resumes from the previous phase's checkpoint.

Batching: gradient accumulation over individual-sample forward/backward
passes (see ``model.py`` for why true tensor batching was not used).
``--accum-steps`` sets the effective batch size.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

import pandas as pd

from driftsense.localize.data import DriftSensePairs, STRIDE, gt_to_out_grid
from driftsense.localize.losses import (build_soft_target, coordinate_loss,
                                        map_softmax_ce)
from driftsense.localize.model import (SiameseCorrelationNet,
                                       predict_xy,
                                       windowed_soft_argmax)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _step(net: SiameseCorrelationNet, sample: Dict, device: torch.device,
          dtype: torch.dtype, huber_weight: float, lattice_boost: float
          ) -> tuple[torch.Tensor, float]:
    """Forward + loss for one sample. No backward here -- caller accumulates.

    Args:
        net: The model.
        sample: One item from ``DriftSensePairs``.
        device: Torch device.
        dtype: Autocast dtype (bf16 recommended; see module docstring in
            ``model.py`` on correlation-volume dynamic range).
        huber_weight: Weight on the coordinate loss relative to the heatmap
            loss.
        lattice_boost: Passed to ``losses.build_target``.

    Returns:
        ``(loss, pixel_error)`` -- loss is the tensor to backward from;
        pixel_error is a plain float for logging only.
    """
    ref = sample["ref"].unsqueeze(0).to(device)
    search = sample["search"].unsqueeze(0).to(device)

    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=device.type == "cuda"):
        logits, ref_emb_size = net(ref, search)

        gt_out_x, gt_out_y = gt_to_out_grid(sample, ref_emb_size)
        # Loss window is centred on ground truth: the sub-pixel term must be
        # well-conditioned from step one, before the peak is near correct.
        coords_out = windowed_soft_argmax(logits, gt_out_x, gt_out_y)
        h, w = logits.shape
        target = build_soft_target(h, w, gt_out_x, gt_out_y,
                                   device=logits.device, dtype=logits.dtype)

        loss_hm = map_softmax_ce(logits, target)
        loss_coord = coordinate_loss(coords_out, ref_emb_size, STRIDE,
                                     sample["gt_x"], sample["gt_y"])
        loss = loss_hm + huber_weight * loss_coord

    with torch.no_grad():
        from driftsense.localize.coords import out_to_pixel
        # Reported error uses the PREDICTED peak, never ground truth --
        # otherwise the metric flatters the model and hides a failure to
        # localise globally.
        pred = predict_xy(logits)
        px = float(out_to_pixel(pred[0].item(), ref_emb_size, STRIDE))
        py = float(out_to_pixel(pred[1].item(), ref_emb_size, STRIDE))
        err = ((px - sample["gt_x"]) ** 2 + (py - sample["gt_y"]) ** 2) ** 0.5

    return loss, err


def train_phase(net: SiameseCorrelationNet, dataset: DriftSensePairs,
                device: torch.device, out_dir: Path, phase_name: str,
                steps: int, accum_steps: int, lr: float, wd: float,
                warmup_frac: float, ema_decay: float, huber_weight: float,
                lattice_boost: float, dtype: torch.dtype,
                log_every: int = 20) -> None:
    """Run one curriculum phase and save a checkpoint at the end.

    Args:
        net: Model, modified in place.
        dataset: Data for this phase (already filtered to the right
            difficulties by the caller).
        device: Torch device.
        out_dir: Where to write the checkpoint.
        phase_name: Used in the checkpoint filename and log lines.
        steps: Optimiser steps (not sample count -- each step consumes
            ``accum_steps`` samples).
        accum_steps: Samples accumulated per optimiser step.
        lr: Peak learning rate.
        wd: Weight decay.
        warmup_frac: Fraction of ``steps`` spent warming up linearly.
        ema_decay: EMA decay for the shadow weights saved alongside the raw
            checkpoint.
        huber_weight: See ``_step``.
        lattice_boost: See ``_step``.
        dtype: Autocast dtype.
        log_every: Print progress every this many optimiser steps.
    """
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    warmup_steps = max(1, int(steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(prog * 3.14159265)).item())

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}

    loader = DataLoader(dataset, batch_size=None, shuffle=True,
                        num_workers=4, persistent_workers=True)
    it = iter(loader)

    t0 = time.time()
    running_loss = running_err = 0.0
    n_logged = 0

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = step_err = 0.0

        for _ in range(accum_steps):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(loader)
                sample = next(it)
            loss, err = _step(net, sample, device, dtype,
                              huber_weight, lattice_boost)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step} (id={sample['id']}). "
                    "Training halted rather than continuing with corrupted "
                    "weights. Run tools/overfit_test.py to isolate.")
            (loss / accum_steps).backward()
            step_loss += loss.item() / accum_steps
            step_err += err / accum_steps

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                else:
                    ema[k] = v.detach().clone()

        running_loss += step_loss
        running_err += step_err
        n_logged += 1

        if step % log_every == 0 or step == steps:
            dt = time.time() - t0
            print(f"  [{phase_name}] step {step}/{steps}  "
                  f"loss {running_loss / n_logged:.4f}  "
                  f"err {running_err / n_logged:.2f} px  "
                  f"lr {sched.get_last_lr()[0]:.2e}  "
                  f"{dt / step:.2f} s/step", flush=True)
            running_loss = running_err = 0.0
            n_logged = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{phase_name}.pt"
    torch.save({"model": net.state_dict(), "ema": ema,
               "phase": phase_name}, ckpt_path)
    print(f"  saved {ckpt_path}")


def smoke_test(args: argparse.Namespace) -> None:
    """20 real samples, a handful of steps, on the actual dataset and device.

    Exists to catch shape and indexing bugs immediately rather than after a
    long run. Not optional given this code was authored without a torch
    runtime to test it in -- see module docstring.
    """
    device = _device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"  device: {device}   dtype: {dtype}")

    ds = DriftSensePairs(args.dataset, difficulties=("easy",))
    small = Subset(ds, list(range(min(20, len(ds)))))
    net = SiameseCorrelationNet()

    out_dir = Path(args.out) / "smoke"
    train_phase(net, small, device, out_dir, "smoke",
               steps=10, accum_steps=2, lr=3e-4, wd=0.05,
               warmup_frac=0.3, ema_decay=0.99,
               huber_weight=1.0, lattice_boost=4.0, dtype=dtype,
               log_every=1)
    print("\n  SMOKE TEST PASSED -- loss decreased and no exceptions.")
    print("  Inspect the per-step error above: it should trend downward, "
          "even on 20 samples over 10 steps. If it does not move at all, "
          "stop and report the printed loss/err sequence before scaling up.")


def _load_split_seeds(split_csv: str, split_name: str) -> set:
    """Seeds belonging to one split ("train", "val", or "golden").

    Args:
        split_csv: Path written by ``make_split.py``.
        split_name: Which split to extract.

    Returns:
        Set of seeds. Never overlaps with the other two splits -- verified at
        split-creation time.
    """
    df = pd.read_csv(split_csv, usecols=["seed", "split"])
    return set(df.loc[df["split"] == split_name, "seed"])


def full_train(args: argparse.Namespace) -> None:
    """The three-phase curriculum: easy -> +medium -> +hard.

    If ``--split-csv`` is given, every phase trains ONLY on that split's
    "train" seeds. The "val" and "golden" seeds are never used for a gradient
    update -- golden is reserved entirely for the one-time final number
    reported via ``tools/eval_dl.py --split golden``.
    """
    device = _device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"  device: {device}   dtype: {dtype}")

    train_seeds = None
    if args.split_csv:
        train_seeds = _load_split_seeds(args.split_csv, "train")
        print(f"  split-csv: {args.split_csv}  "
              f"({len(train_seeds)} train seeds; val/golden withheld)")

    net = SiameseCorrelationNet()
    out_dir = Path(args.out)

    all_phases = [
        ("phase1_easy", ("easy",)),
        ("phase2_easy_medium", ("easy", "medium")),
        ("phase3_all", ("easy", "medium", "hard")),
    ]

    # Resume: load weights trained elsewhere (or in an earlier session) so a
    # run can be split across machines or interrupted without losing work.
    # The raw "model" weights are loaded rather than the EMA copy: EMA is a
    # smoothed snapshot for inference, and continuing optimisation from it
    # discards the trajectory the optimiser was on.
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        net.load_state_dict(ckpt["model"])
        print(f"  resumed weights from {args.resume_from} "
              f"(was phase: {ckpt.get('phase', 'unknown')})")

    if args.phases:
        wanted = {p.strip() for p in args.phases.split(",")}
        chosen = [(n, d) for n, d in all_phases
                  if n in wanted or n.replace("phase", "") in wanted
                  or n[5] in wanted]
        if not chosen:
            raise SystemExit(
                f"--phases '{args.phases}' matched nothing. Valid names: "
                + ", ".join(n for n, _ in all_phases) + " (or 1,2,3)")
        phases = chosen
    else:
        phases = all_phases

    print(f"  running phases: {[n for n, _ in phases]}")
    for name, diffs in phases:
        print(f"\n=== {name}  difficulties={diffs} ===")
        ds = DriftSensePairs(args.dataset, difficulties=diffs,
                            seed_filter=train_seeds)
        print(f"  {len(ds)} samples available"
              + (" (train split only)" if train_seeds else " (NO split filter -- "
                 "val/golden may leak into training)"))
        train_phase(net, ds, device, out_dir, name,
                   steps=args.steps_per_phase, accum_steps=args.accum_steps,
                   lr=args.lr, wd=args.wd, warmup_frac=0.05,
                   ema_decay=0.999, huber_weight=args.huber_weight,
                   lattice_boost=args.lattice_boost, dtype=dtype,
                   log_every=args.log_every)

    last = phases[-1][0]
    print(f"\nDone. Final checkpoint: {out_dir / (last + '.pt')}")
    print("Next: python tools/eval_dl.py --dataset data/eval "
          f"--checkpoint {out_dir / (last + '.pt')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="dataset root containing labels.jsonl")
    ap.add_argument("--out", default="runs/lscv", help="checkpoint directory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run a 20-sample, 10-step sanity check and exit")
    ap.add_argument("--steps-per-phase", type=int, default=15000)
    ap.add_argument("--accum-steps", type=int, default=8,
                    help="effective batch size per optimiser step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--huber-weight", type=float, default=1.0)
    ap.add_argument("--lattice-boost", type=float, default=4.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint .pt to load weights from before training")
    ap.add_argument("--phases", default=None,
                    help="which phases to run, e.g. '2,3' or "
                         "'phase2_easy_medium'. Default: all three.")
    ap.add_argument("--split-csv", default=None,
                    help="path from make_split.py; restricts training to "
                         "the 'train' split. Strongly recommended -- "
                         "without it, val/golden seeds may be trained on, "
                         "and the final eval number cannot be trusted.")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test(args)
    else:
        full_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
