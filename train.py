"""Train FDFormer.

Example (laptop GPU):
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --batch_size 2 --img_size 256   # lower VRAM
    python train.py --config configs/default.yaml --accum_steps 4                 # big effective batch, low VRAM

Only the FDFormer modules + decoder are optimised; the VGG-19 encoder stays frozen.
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models import FDFormer
from losses.losses import FDFormerCriterion
from data.dataset import ContentStyleDataset, SeededEpochSampler


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adjust_lr(optimizer, base_lr, decay, step):
    lr = base_lr / (1.0 + decay * step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def seeded_batches(loader, dataset, sampler, used):
    """Yield the seeded example sequence, skipping `used` batches on a resumed run."""
    per_epoch = len(dataset) // loader.batch_size
    epoch, batch = used // per_epoch, used % per_epoch
    while True:
        dataset.epoch = epoch
        sampler.set_position(epoch, batch * loader.batch_size)
        for item in loader:
            yield item
        epoch, batch = epoch + 1, 0


def sample_alphas(cfg):
    """Draw the two blend coefficients for one step.

    Left disabled, this returns (1, 1) and training is exactly as before. Enabled, it
    draws each coefficient uniformly from a range, reserving a fraction of steps for
    the nominal setting so that the operating point the model is reported at stays
    well trained. Without this, the decoder only ever renders one setting, and the
    coefficients are out of distribution the moment a user moves them.
    """
    if not cfg or not cfg.get("enabled", False):
        return 1.0, 1.0
    import random
    if random.random() < float(cfg.get("nominal_prob", 0.25)):
        return 1.0, 1.0
    lo, hi = cfg.get("range", [0.25, 1.75])
    return random.uniform(lo, hi), random.uniform(lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--accum_steps", type=int, default=None,
                    help="gradient accumulation: effective batch = batch_size * accum_steps "
                         "(lets a tiny per-step batch behave like a big one on small GPUs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=None,
                    help="seeds Python, NumPy and PyTorch and fixes the training-example "
                         "sequence (data order, style pairing, crops)")
    ap.add_argument("--save_dir", default=None, help="overrides train.save_dir")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="overrides data.num_workers; a seeded run sees the same examples either way")
    ap.add_argument("--final_only", action="store_true",
                    help="keep only latest.pth (with optimiser state) and the final checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
    if args.img_size:   cfg["data"]["img_size"] = args.img_size
    if args.max_iters:  cfg["train"]["max_iters"] = args.max_iters
    if args.accum_steps: cfg["train"]["accum_steps"] = args.accum_steps
    if args.save_dir: cfg["train"]["save_dir"] = args.save_dir
    if args.num_workers is not None: cfg["data"]["num_workers"] = args.num_workers
    accum_steps = max(1, int(cfg["train"].get("accum_steps", 1)))
    seed = args.seed if args.seed is not None else cfg["train"].get("seed", None)
    cfg["train"]["seed"] = seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    dataset = ContentStyleDataset(cfg["data"]["content_dir"], cfg["data"]["style_dir"],
                                  img_size=cfg["data"]["img_size"], seed=seed)
    sampler = SeededEpochSampler(len(dataset), seed) if seed is not None else None
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"],
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=cfg["data"]["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))

    model = FDFormer(embed_dim=cfg["model"]["embed_dim"], heads=cfg["model"]["heads"],
                     pretrained_vgg=cfg["model"]["pretrained_vgg"],
                     freq_decouple=cfg["model"].get("freq_decouple", True),
                     use_gate=cfg["model"].get("use_gate", True),
                     freq_mode=cfg["model"].get("freq_mode", "learnable")).to(device)
    criterion = FDFormerCriterion(model.encoder, cfg["loss"], cfg["loss"]["use_identity"])
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg["train"]["lr"])

    alpha_cfg = cfg["train"].get("alpha_sampling", None)
    if alpha_cfg and alpha_cfg.get("enabled", False):
        print(f"sampling blend coefficients from {alpha_cfg.get('range', [0.25, 1.75])}, "
              f"nominal setting on {alpha_cfg.get('nominal_prob', 0.25):.0%} of steps")

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Resume from checkpoint if specified
    start_step = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt.get("step", 0) + 1
            print(f"Resumed from step {start_step - 1}")
        else:
            print(f"Checkpoint not found: {args.resume}, starting fresh")

    if sampler is not None:
        data_iter = seeded_batches(loader, dataset, sampler, (start_step - 1) * accum_steps)
    else:
        data_iter = infinite(loader)
    max_iters = cfg["train"]["max_iters"]
    pbar = tqdm(range(start_step, max_iters + 1))
    for step in pbar:
        adjust_lr(optimizer, cfg["train"]["lr"], cfg["train"]["lr_decay"], step)

        optimizer.zero_grad(set_to_none=True)
        # Gradient accumulation: average the loss over accum_steps micro-batches so the
        # update equals one big batch of size batch_size*accum_steps, but only one
        # micro-batch is ever resident in memory (key for small GPUs).
        for _ in range(accum_steps):
            content, style = next(data_iter)
            content, style = content.to(device), style.to(device)
            a_low, a_high = sample_alphas(alpha_cfg)
            with torch.amp.autocast(device.type, enabled=use_amp):
                total, comps = criterion(model, content, style,
                                         alpha_low=a_low, alpha_high=a_high)
                total = total / accum_steps
            scaler.scale(total).backward()
        # Unscale gradients before clipping (required when using AMP)
        scaler.unscale_(optimizer)
        clip_norm = cfg["train"].get("clip_grad_norm", None)
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg["train"]["log_every"] == 0:
            pbar.set_description(
                f"it {step} | total {comps['total']:.3f} c {comps['content']:.3f} "
                f"s {comps['style']:.3f} hf {comps['hf']:.4f} id {comps['id1']:.3f}")

        if step % cfg["train"]["sample_every"] == 0:
            model.eval()
            with torch.no_grad():
                out = model(content[:2], style[:2]).clamp(0, 1)
            grid = torch.cat([content[:2], style[:2], out], dim=0)
            save_image(grid, os.path.join(save_dir, "samples", f"it{step:06d}.png"), nrow=2)
            model.train()

        if step % cfg["train"]["ckpt_every"] == 0 or step == max_iters:
            if not args.final_only or step == max_iters:
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                           os.path.join(save_dir, f"ckpt_{step:06d}.pth"))
            tmp = os.path.join(save_dir, "latest.tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "cfg": cfg, "step": step}, tmp)
            os.replace(tmp, os.path.join(save_dir, "latest.pth"))

    print("Training done. Checkpoints in", save_dir)


if __name__ == "__main__":
    main()
