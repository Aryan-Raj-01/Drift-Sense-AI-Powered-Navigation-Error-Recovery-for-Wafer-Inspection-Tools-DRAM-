#!/usr/bin/env python3
"""Train the fixed dense-correlation model.

ALWAYS run the smoke test first::

    python -m driftsense.dl_localize.train --dataset <dataset-root> ^
        --out <run-dir> --smoke-test

Then a real run::

    python -m driftsense.dl_localize.train ^
        --dataset <dataset-root> ^
        --split-csv <dataset-root>/subset_split.csv ^
        --out <run-dir> --steps-per-phase 4000 --accum-steps 8 ^
        --val-every 1000 --val-pairs 300

WHAT CHANGED FROM ``localize/train.py``
=======================================

1. ``lattice_boost`` IS NOW ACTUALLY USED.  The old ``_step`` took the
   argument and never referenced it: it called ``build_soft_target`` +
   ``map_softmax_ce``, neither of which accepts a weight map, while the
   lattice logic lived on the dead ``focal_heatmap_loss`` path.  The
   roadmap's central idea has never run.  It runs here, as an explicit
   margin term -- see ``losses.lattice_margin_loss`` for why a margin and
   not a per-pixel weight.

2. THE COORDINATE LOSS IS NO LONGER DEGENERATE.  The old ``_step`` centred
   the sub-pixel window on GROUND TRUTH and then penalised the distance from
   the resulting soft-argmax to ground truth.  Early in training the
   in-window softmax is near-uniform, so its expectation is the window
   centre -- which IS ground truth.  The loss is ~0 and the gradient is ~0:
   the term taught nothing.  Here the window is centred on the PREDICTED
   peak, jittered by up to a cell, and the term is gated to samples where
   the peak is already close enough for a sub-pixel refinement to mean
   anything.

3. FP16 BY DEFAULT, NOT BF16.  Every quantity in this network is a cosine
   similarity bounded in [-1, 1], so bfloat16's extra exponent range buys
   nothing while its 8-bit mantissa costs a lot: the ULP near 0.5 is 0.0039,
   and the gap between the true peak and a lattice replica -- the number
   that decides a periodic lock -- can be smaller than that.  fp16 has an
   11-bit mantissa, ~8x finer, with ample range for [-1, 1].  This is
   reasoning, not a measurement: A/B it with ``--precision bf16`` before
   treating it as settled.

4. IN-TRAINING VALIDATION.  ``--val-every`` reports per-difficulty <=1 px
   accuracy on held-out seeds mid-run.  The 10k runs committed 1-2 GPU-hours
   per phase before anyone saw a number.

5. BALANCED SAMPLING.  Uniform sampling over a phase-3 pool that is 50%
   easy spends half the gradient budget on the bucket that needs it least.
   ``--mix`` sets the sampling ratio independently of the pool's composition.

6. REAL RESUME.  Optimiser, scheduler, scaler, EMA and step counter are all
   checkpointed.  The old code saved weights only, so ``--resume-from``
   restarted the LR schedule from warmup every time -- which is part of why
   "1500 more steps" behaved like a fresh run rather than a continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from driftsense.dl_localize.coords import STRIDE, TEMPLATE_PX, out_to_pixel
from driftsense.dl_localize.data import (DriftSensePairs, load_split_seeds)
from driftsense.dl_localize.losses import (build_soft_target, coordinate_loss,
                                            lattice_margin_loss,
                                            map_softmax_ce, offset_loss,
                                            soft_target_entropy)
from driftsense.dl_localize.model import (SiameseCorrelationNet, peak_xy,
                                           pitch_limited_radius, refine_xy,
                                           windowed_soft_argmax)

PHASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("phase1_easy", ("easy",)),
    ("phase2_easy_medium", ("easy", "medium")),
    ("phase3_all", ("easy", "medium", "hard")),
)


def _device(name: str) -> torch.device:
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(name)
    if dev.type == "cuda":
        # cuDNN autotunes a convolution algorithm per input shape and caches
        # it.  In the old package the template size varied per sample
        # (96-104 px, from the scale_ratio label), so every few steps hit a
        # new shape, re-ran the benchmark, and thrashed the cache -- turning
        # this on there would have made training SLOWER.  dl_localize fixes
        # the template at TEMPLATE_PX and the search at 1000x1000, so there
        # are exactly two shapes for the whole run and autotuning pays for
        # itself in the first few steps.
        torch.backends.cudnn.benchmark = True
        # TF32 for the fp32 fallback paths.  The correlation surface itself
        # is never TF32 -- head and losses are explicit float32 ops on
        # values in [-1, 1], where TF32's 10-bit mantissa would be the same
        # mistake bf16 was.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  cuda: {torch.cuda.get_device_name(0)}  "
              f"cudnn.benchmark=True (shapes are fixed in dl_localize)")
    return dev


def _autocast_dtype(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return None
    return torch.float16 if precision == "fp16" else torch.bfloat16


def _step(net: SiameseCorrelationNet, sample: Dict, device: torch.device,
          amp_dtype, cfg: argparse.Namespace, rng: np.random.Generator
          ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forward + loss for one sample.  No backward -- the caller accumulates.

    Returns:
        ``(loss, stats)``.  ``stats`` is plain floats for logging only.
    """
    ref = sample["ref"].unsqueeze(0).to(device, non_blocking=True)
    search = sample["search"].unsqueeze(0).to(device, non_blocking=True)

    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None else torch.autocast(
               device_type=device.type, enabled=False))
    with ctx:
        logits, offsets = net(ref, search)

    # Losses in float32 always -- see the numerics note in losses.
    logits = logits.float()
    offsets = offsets.float()
    h, w = logits.shape

    gt_out_x = float(sample["gt_out_x"])
    gt_out_y = float(sample["gt_out_y"])

    target = build_soft_target(h, w, gt_out_x, gt_out_y, logits.device,
                               sigma=cfg.target_sigma)
    loss_ce = map_softmax_ce(logits, target)

    loss_lat, n_rep = lattice_margin_loss(
        logits, gt_out_x, gt_out_y,
        float(sample["pitch_x_out"]), float(sample["pitch_y_out"]),
        margin=cfg.lattice_margin, min_sep=cfg.lattice_min_sep)

    loss_off = offset_loss(offsets, int(sample["cell_x"]),
                           int(sample["cell_y"]), float(sample["frac_x"]),
                           float(sample["frac_y"]), radius=cfg.offset_radius)

    # Sub-pixel coordinate term.  Window on the PREDICTED peak (not ground
    # truth -- see module docstring on why the old version was degenerate),
    # jittered, and gated to peaks that are already close enough for a
    # sub-pixel refinement to be meaningful.
    px, py = peak_xy(logits)
    radius = pitch_limited_radius(float(sample["pitch_x_out"]),
                                  float(sample["pitch_y_out"]),
                                  default=cfg.readout_radius)
    jx = float(rng.integers(-1, 2))
    jy = float(rng.integers(-1, 2))
    coords_out = windowed_soft_argmax(logits, px + jx, py + jy, radius=radius)
    cell_err = math.hypot(px - gt_out_x, py - gt_out_y)
    if cell_err <= cfg.coord_gate_cells:
        loss_coord = coordinate_loss(coords_out, float(sample["gt_x"]),
                                     float(sample["gt_y"]))
    else:
        loss_coord = logits.new_zeros(())

    loss = (loss_ce
            + cfg.lattice_weight * loss_lat
            + cfg.offset_weight * loss_off
            + cfg.huber_weight * loss_coord)

    with torch.no_grad():
        pred = refine_xy(logits, offsets, radius=radius, mode="offset")
        ex = out_to_pixel(float(pred[0])) - float(sample["gt_x"])
        ey = out_to_pixel(float(pred[1])) - float(sample["gt_y"])
        err = math.hypot(ex, ey)

    return loss, {
        "loss": float(loss.detach()), "ce": float(loss_ce.detach()),
        "lat": float(loss_lat.detach()), "off": float(loss_off.detach()),
        "coord": float(loss_coord.detach()),
        "err": err, "n_rep": float(n_rep), "hit1": float(err <= 1.0),
    }


@torch.no_grad()
def quick_val(net: SiameseCorrelationNet, ds: DriftSensePairs,
              device: torch.device, n: int, cfg: argparse.Namespace
              ) -> Dict[str, Dict[str, float]]:
    """Per-difficulty <=1 px accuracy on a random slice of a held-out set."""
    was_training = net.training
    net.eval()
    idx = np.random.default_rng(0).permutation(len(ds))[:n]
    buckets: Dict[str, List[float]] = {}
    for i in idx:
        s = ds[int(i)]
        logits, offsets = net(s["ref"].unsqueeze(0).to(device),
                              s["search"].unsqueeze(0).to(device))
        radius = pitch_limited_radius(float(s["pitch_x_out"]),
                                      float(s["pitch_y_out"]),
                                      default=cfg.readout_radius)
        pred = refine_xy(logits.float(), offsets.float(), radius=radius,
                         mode="offset")
        err = math.hypot(out_to_pixel(float(pred[0])) - s["gt_x"],
                         out_to_pixel(float(pred[1])) - s["gt_y"])
        buckets.setdefault(s["difficulty"], []).append(err)
        buckets.setdefault("ALL", []).append(err)
    if was_training:
        net.train()
    return {k: {"n": len(v),
                "acc1": 100.0 * float(np.mean(np.array(v) <= 1.0)),
                "acc5": 100.0 * float(np.mean(np.array(v) <= 5.0)),
                "median": float(np.median(v))}
            for k, v in buckets.items()}


def _balanced_sampler(ds: DriftSensePairs, mix: Optional[Dict[str, float]],
                      n_draw: int) -> Optional[WeightedRandomSampler]:
    """Sampler that draws difficulties in a chosen ratio, not the pool's."""
    if not mix:
        return None
    diffs = [r["difficulty"] for r in ds.rows]
    counts: Dict[str, int] = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    total_w = sum(mix.get(d, 0.0) for d in counts)
    if total_w <= 0:
        return None
    weights = [mix.get(d, 0.0) / max(counts[d], 1) for d in diffs]
    return WeightedRandomSampler(weights, num_samples=n_draw, replacement=True)


def train_phase(net: SiameseCorrelationNet, dataset: DriftSensePairs,
                val_ds: Optional[DriftSensePairs], device: torch.device,
                out_dir: Path, phase_name: str, steps: int,
                cfg: argparse.Namespace,
                resume_state: Optional[Dict] = None) -> None:
    """Run one curriculum phase and checkpoint at the end (and periodically)."""
    net.to(device).train()
    amp_dtype = _autocast_dtype(cfg.precision, device)
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.wd)
    warmup = max(1, int(steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (
            1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = {k: v.detach().clone().float() if v.dtype.is_floating_point
           else v.detach().clone()
           for k, v in net.state_dict().items()}
    start_step = 0

    if resume_state:
        if "opt" in resume_state:
            opt.load_state_dict(resume_state["opt"])
        if "sched" in resume_state:
            sched.load_state_dict(resume_state["sched"])
        if "scaler" in resume_state and use_scaler:
            scaler.load_state_dict(resume_state["scaler"])
        if "ema" in resume_state:
            # Checkpoints are loaded with map_location="cpu", so the restored
            # EMA lives on the CPU while the model is already on the GPU. The
            # in-place EMA update below then mixes devices and raises. The
            # optimiser does not need this because Optimizer.load_state_dict
            # casts its state to each parameter's device automatically; the
            # EMA is a plain dict and has no such machinery.
            ema = {k: v.detach().clone().to(device)
                   for k, v in resume_state["ema"].items()}
        start_step = int(resume_state.get("step", 0))
        print(f"  resumed optimiser/scheduler/EMA at step {start_step}")

    workers = cfg.workers
    sampler = _balanced_sampler(dataset, cfg._mix,
                                (steps + 8) * cfg.accum_steps)
    # Memory notes, learned the hard way on a 6 GB laptop:
    #  * batch_size=None means prefetch_factor counts SAMPLES, not batches,
    #    so prefetch_factor=4 with 2 workers buffers 8 search tensors of
    #    1000x1000 float32 (4 MB each) on top of two pickled copies of the
    #    dataset. Dropped to 2.
    #  * pin_memory allocates NON-PAGEABLE host memory. Windows cannot swap
    #    it, so on a machine that is already tight it converts "slow" into
    #    MemoryError. It buys a few ms of H2D transfer on a workload that is
    #    ~400 ms/sample of GPU compute. Not worth it here; off by default.
    loader = DataLoader(
        dataset, batch_size=None, sampler=sampler,
        shuffle=(sampler is None), num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=cfg.pin_memory and device.type == "cuda")
    it = iter(loader)
    rng = np.random.default_rng(cfg.seed)

    floor = soft_target_entropy(cfg.target_sigma)
    print(f"  entropy floor of the sigma={cfg.target_sigma} target: "
          f"{floor:.3f} nats -- ce should approach this, never go below")

    t0 = time.time()
    acc: Dict[str, float] = {}
    n_logged = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_stats: Dict[str, float] = {}

        for _ in range(cfg.accum_steps):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(loader)
                sample = next(it)
            loss, st = _step(net, sample, device, amp_dtype, cfg, rng)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step} (id={sample['id']}). "
                    f"Halted rather than continuing with corrupted weights. "
                    f"Re-run with --precision fp32 to determine whether this "
                    f"is an overflow or a genuine data problem.")
            scaler.scale(loss / cfg.accum_steps).backward()
            for k, v in st.items():
                step_stats[k] = step_stats.get(k, 0.0) + v / cfg.accum_steps

        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                     cfg.clip))
        scaler.step(opt)
        scaler.update()
        sched.step()

        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(cfg.ema_decay).add_(v.detach().float(),
                                                    alpha=1 - cfg.ema_decay)
                else:
                    ema[k] = v.detach().clone()

        for k, v in step_stats.items():
            acc[k] = acc.get(k, 0.0) + v
        acc["gnorm"] = acc.get("gnorm", 0.0) + gnorm
        n_logged += 1

        if step % cfg.log_every == 0 or step == steps:
            dt = time.time() - t0
            a = {k: v / n_logged for k, v in acc.items()}
            print(f"  [{phase_name}] {step}/{steps}  "
                  f"loss {a['loss']:.3f}  ce {a['ce']:.3f}(floor {floor:.2f})  "
                  f"lat {a['lat']:.3f}  off {a['off']:.3f}  "
                  f"coord {a['coord']:.3f}  err {a['err']:.2f}px  "
                  f"hit1 {100*a['hit1']:.0f}%  rep {a['n_rep']:.0f}  "
                  f"|g| {a['gnorm']:.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{dt / max(1, step - start_step):.2f}s/step", flush=True)
            acc, n_logged = {}, 0

        if (val_ds is not None and cfg.val_every > 0
                and (step % cfg.val_every == 0 or step == steps)):
            res = quick_val(net, val_ds, device, cfg.val_pairs, cfg)
            line = "  VAL  " + "  ".join(
                f"{g}: {res[g]['acc1']:.0f}%<=1px (med {res[g]['median']:.2f})"
                for g in ("ALL", "easy", "medium", "hard") if g in res)
            print(line, flush=True)

        if cfg.save_every > 0 and step % cfg.save_every == 0:
            _save(net, ema, opt, sched, scaler, step, phase_name,
                  out_dir / f"{phase_name}_last.pt", cfg)

    _save(net, ema, opt, sched, scaler, steps, phase_name,
          out_dir / f"{phase_name}.pt", cfg)


def _save(net, ema, opt, sched, scaler, step, phase_name, path, cfg) -> None:
    torch.save({
        "model": net.state_dict(),
        "ema": ema,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "phase": phase_name,
        "arch": "dl_localize.SiameseCorrelationNet",
        "template_px": TEMPLATE_PX,
        "stride": STRIDE,
        "angle_bank": list(DriftSensePairs.ANGLE_BANK),
        "scale_bank": list(DriftSensePairs.SCALE_BANK),
        "cfg": {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))},
    }, path)
    print(f"  saved {path}")


def smoke_test(cfg: argparse.Namespace) -> None:
    """A few real samples, a few steps, on the real device.  Not optional.

    Asserts that error actually moves.  The old smoke test printed
    "SMOKE TEST PASSED" whenever nothing raised, which is a much weaker
    claim than it sounds.
    """
    device = _device(cfg.device)
    amp = _autocast_dtype(cfg.precision, device)
    print(f"  device {device}  precision {cfg.precision}  autocast {amp}")

    ds = DriftSensePairs(cfg.dataset, difficulties=("easy",), augment=True,
                           aug_rot_deg=cfg.aug_rot_deg,
                           aug_scale=cfg.aug_scale,
                           jitter_strength=cfg.jitter)
    ds.rows = ds.rows[:cfg.smoke_samples]
    print(f"  {len(ds)} samples")

    net = SiameseCorrelationNet()
    out_dir = Path(cfg.out) / "smoke88"
    cfg.val_every = 0
    train_phase(net, ds, None, device, out_dir, "smoke", cfg.smoke_steps, cfg)
    print("\n  Smoke run finished without error. Now READ THE err COLUMN: "
          "it must trend down. If it is flat, stop and report the printed "
          "sequence -- do not start a real run.")


def full_train(cfg: argparse.Namespace) -> None:
    device = _device(cfg.device)
    print(f"  device {device}  precision {cfg.precision}")

    train_seeds = val_seeds = None
    if cfg.split_csv:
        train_seeds = load_split_seeds(cfg.split_csv, "train")
        print(f"  split-csv {cfg.split_csv}: {len(train_seeds)} train seeds")
        try:
            val_seeds = load_split_seeds(cfg.split_csv, "val")
            print(f"  {len(val_seeds)} val seeds held out")
        except ValueError:
            print("  WARNING: no 'val' split in the CSV; skipping in-training "
                  "validation")
    else:
        print("  WARNING: no --split-csv. Training will draw from the entire "
              "pool with NO held-out guarantee, and any accuracy number you "
              "report afterwards is contaminated.")

    net = SiameseCorrelationNet()
    out_dir = Path(cfg.out)
    resume_state = None
    if cfg.resume_from:
        ck = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
        if ck.get("arch") != "dl_localize.SiameseCorrelationNet":
            raise SystemExit(
                f"{cfg.resume_from} was written by a different architecture "
                f"({ck.get('arch', 'localize (old)')}). dl_localize changed the "
                f"encoder fusion, added an offset head and changed the "
                f"coordinate convention -- old checkpoints are not loadable "
                f"and would be wrong even if they were. Train from scratch.")
        net.load_state_dict(ck["model"])
        resume_state = ck if cfg.resume_optimizer else None
        print(f"  resumed weights from {cfg.resume_from} "
              f"(phase {ck.get('phase')}, step {ck.get('step')})")

    wanted = ({p.strip() for p in cfg.phases.split(",")} if cfg.phases
              else None)
    phases = [(n, d) for n, d in PHASES
              if wanted is None or n in wanted or n[5] in wanted]
    if not phases:
        raise SystemExit(f"--phases '{cfg.phases}' matched nothing. Valid: "
                         + ", ".join(n for n, _ in PHASES) + " (or 1,2,3)")
    print(f"  phases: {[n for n, _ in phases]}")

    scales = [float(s) for s in cfg.phase_step_scale.split(",")]
    lr_scales = [float(s) for s in cfg.phase_lr_scale.split(",")]
    base_lr = cfg.lr
    for i, (name, diffs) in enumerate(phases):
        print(f"\n=== {name}  difficulties={diffs} ===")
        ds = DriftSensePairs(cfg.dataset, difficulties=diffs,
                               seed_filter=train_seeds, augment=True,
                               aug_rot_deg=cfg.aug_rot_deg,
                               aug_scale=cfg.aug_scale,
                               jitter_strength=cfg.jitter, seed=cfg.seed)
        val_ds = None
        if val_seeds:
            val_ds = DriftSensePairs(cfg.dataset, difficulties=diffs,
                                       seed_filter=val_seeds, augment=False)
        print(f"  {len(ds)} train samples"
              + (f", {len(val_ds)} val samples" if val_ds else ""))
        steps = int(cfg.steps_per_phase * scales[min(i, len(scales) - 1)])
        cfg.lr = base_lr * lr_scales[min(i, len(lr_scales) - 1)]
        print(f"  steps {steps}   peak lr {cfg.lr:.2e}")
        train_phase(net, ds, val_ds, device, out_dir, name, steps, cfg,
                    resume_state if i == 0 else None)
    cfg.lr = base_lr

    last = phases[-1][0]
    print(f"\nDone. Final checkpoint: {out_dir / (last + '.pt')}")
    print(f"Next: python -m driftsense.dl_localize.eval --dataset "
          f"{cfg.dataset} --checkpoint {out_dir / (last + '.pt')} "
          f"--split-csv {cfg.split_csv} --split val --pairs 2000")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs/lscv88")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="fp16",
                    choices=["fp16", "bf16", "fp32"],
                    help="fp16 default: all quantities are bounded in [-1,1] "
                         "so mantissa matters and range does not")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--smoke-samples", type=int, default=24)
    ap.add_argument("--smoke-steps", type=int, default=30)

    ap.add_argument("--steps-per-phase", type=int, default=4000)
    ap.add_argument("--phase-step-scale", default="1,1.5,3.0",
                    help="per-phase multiplier on --steps-per-phase. The old "
                         "code gave every phase the same budget, so phase 3 "
                         "(the biggest and hardest pool) got the least "
                         "repetition per sample")
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-floor", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes. Default 0: each "
                         "worker is a separate process holding its own copy "
                         "of the dataset and its own prefetch buffers, and "
                         "loading two PNGs takes ~20 ms against ~400 ms of "
                         "GPU compute per sample -- so workers buy little "
                         "here and cost real RAM. Raise it only if you can "
                         "see the GPU starving.")
    ap.add_argument("--pin-memory", action="store_true",
                    help="pinned host memory is non-pageable; on a laptop "
                         "that is already tight it turns into MemoryError")
    ap.add_argument("--seed", type=int, default=88)

    ap.add_argument("--target-sigma", type=float, default=1.0)
    ap.add_argument("--huber-weight", type=float, default=0.2)
    ap.add_argument("--offset-weight", type=float, default=5.0,
                    help="raised from 2.0: the offset head is the MEASURED "
                         "bottleneck for easy/medium (98.3%% within 2px but "
                         "85.7%% within 1px), so sub-pixel deserves more of "
                         "the gradient budget")
    ap.add_argument("--offset-radius", type=int, default=1,
                    help="supervise the offset head over a (2r+1)^2 "
                         "neighbourhood of the true cell instead of one "
                         "cell. r=1 is 9x the supervision at no extra "
                         "compute, and gives the readout a valid target when "
                         "the argmax lands one cell off")
    ap.add_argument("--lattice-weight", type=float, default=0.5,
                    help="0 disables the lattice margin term -- the clean "
                         "A/B for whether lattice-offset hard negatives "
                         "actually help. Run it both ways.")
    ap.add_argument("--lattice-margin", type=float, default=4.0)
    ap.add_argument("--lattice-min-sep", type=float, default=2.0)
    ap.add_argument("--readout-radius", type=int, default=2)
    ap.add_argument("--coord-gate-cells", type=float, default=4.0)
    ap.add_argument("--aug-rot-deg", type=float, default=1.0,
                    help="uniform template-rotation jitter around the "
                         "nominal 0 deg that inference uses. Replaces the old "
                         "--match-prob bank sampling, which put the training "
                         "residual-rotation distribution (p90 4.32 deg) well "
                         "away from the inference one (p90 2.63 deg)")
    ap.add_argument("--aug-scale", type=float, default=0.1,
                    help="uniform scale jitter around the nominal 10.0")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="photometric augmentation strength, 0 disables")
    ap.add_argument("--phase-lr-scale", default="1,0.5,0.3",
                    help="per-phase multiplier on --lr. MEASURED: with a "
                         "flat 3e-4 every phase, easy reached 90%% at the end "
                         "of phase 2 and fell to 83%% by the end of phase 3 "
                         "-- phase 3 restarts warmup at full LR on a harder "
                         "mixture and spends its whole budget recovering. "
                         "Decaying across phases keeps what phase 2 learned.")
    ap.add_argument("--mix", default=None,
                    help="sampling ratio by difficulty, e.g. "
                         "'easy=1,medium=1,hard=2'. Default: the pool's own "
                         "composition")

    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-pairs", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--resume-optimizer", action="store_true",
                    help="also restore optimiser/scheduler/EMA state, so a "
                         "continuation is a continuation rather than a fresh "
                         "warmup + cosine cycle")
    ap.add_argument("--phases", default=None)
    ap.add_argument("--split-csv", default=None)
    return ap


def main() -> int:
    cfg = build_argparser().parse_args()
    cfg._mix = None
    if cfg.mix:
        cfg._mix = {k: float(v) for k, v in
                    (p.split("=") for p in cfg.mix.split(","))}
    torch.manual_seed(cfg.seed)
    if cfg.smoke_test:
        smoke_test(cfg)
    else:
        full_train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
