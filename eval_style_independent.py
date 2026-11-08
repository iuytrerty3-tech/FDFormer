"""A per-pair style metric that shares nothing with the training objective.

The style-statistic distance in the main table uses the same frozen VGG-19 mean and
standard deviation as FDFormer's style loss. This script scores the same outputs by the
sliced Wasserstein distance between 7x7 patches of the output and the style image over a
Laplacian pyramid, in RGB. No network is involved, and because the descriptors are not
recentred per image the coarsest level also sees the palette.

Patches and projection directions are drawn from a generator seeded per pair, so every
method is compared under the same random draw and the score is reproducible.

Usage
-----
python eval_style_independent.py --style_dir eval/default/style --reference fdformer \
    --methods fdformer=eval/v2/stylized fdformer_a140=eval/v2_alpha140/stylized ... \
    --json_out results/style_independent.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_BLUR = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def load(path, size):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def blur(x):
    k = _BLUR.to(x)
    kx, ky = k.view(1, 1, 1, 5).repeat(3, 1, 1, 1), k.view(1, 1, 5, 1).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), kx, groups=3)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), ky, groups=3)


def laplacian_pyramid(x, levels):
    pyr = []
    for _ in range(levels - 1):
        down = F.avg_pool2d(blur(x), 2)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        pyr.append(x - up)
        x = down
    pyr.append(x)
    return pyr


def sample_patches(x, n, patch, gen):
    _, c, h, w = x.shape
    ys = torch.randint(0, h - patch + 1, (n,), generator=gen)
    xs = torch.randint(0, w - patch + 1, (n,), generator=gen)
    grid = x.unfold(2, patch, 1).unfold(3, patch, 1)      # 1, c, H, W, p, p
    return grid[0, :, ys, xs].permute(1, 0, 2, 3).reshape(n, -1)


def sliced_wasserstein(a, b, dirs, gen):
    d = torch.randn(dirs, a.shape[1], generator=gen)
    d = d / d.norm(dim=1, keepdim=True)
    pa, _ = torch.sort(a @ d.T, dim=0)
    pb, _ = torch.sort(b @ d.T, dim=0)
    return float((pa - pb).abs().mean())


def lp_swd(out, style, seed, levels=4, patches=2048, patch=7, dirs=256):
    if out.shape[-2:] != style.shape[-2:]:
        style = F.interpolate(style, size=out.shape[-2:], mode="bicubic", align_corners=False)
    total = []
    gen = torch.Generator().manual_seed(seed)
    for lo, ls in zip(laplacian_pyramid(out, levels), laplacian_pyramid(style, levels)):
        a = sample_patches(lo, patches, patch, gen)
        b = sample_patches(ls, patches, patch, gen)
        total.append(sliced_wasserstein(a, b, dirs, gen))
    return float(np.mean(total)) * 1e3, [t * 1e3 for t in total]


def paired(ours, theirs, boots=10000, seed=0):
    d = ours - theirs
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (boots, len(d)))].mean(axis=1)
    return {"diff_mean": float(d.mean()), "diff_ci_low": float(np.percentile(means, 2.5)),
            "diff_ci_high": float(np.percentile(means, 97.5)),
            "wilcoxon_p": float(wilcoxon(d).pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="name=stylized_dir")
    ap.add_argument("--references", nargs="+", default=["fdformer", "fdformer_a140"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num", type=int, default=0)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    methods = dict(m.split("=", 1) for m in args.methods)
    names = sorted(f for f in os.listdir(args.style_dir) if f.lower().endswith(_EXTS))
    if args.num:
        names = names[:args.num]
    scores, per_level = {}, {}
    for m, d in methods.items():
        vals, lv = [], []
        for i, n in enumerate(names):
            stem = os.path.splitext(n)[0]
            p = next(os.path.join(d, stem + e) for e in (".png", ".jpg")
                     if os.path.exists(os.path.join(d, stem + e)))
            with torch.no_grad():
                v, l = lp_swd(load(p, args.size), load(os.path.join(args.style_dir, n), args.size), i)
            vals.append(v)
            lv.append(l)
        scores[m], per_level[m] = np.array(vals), np.array(lv)
        print(f"{m:16s} LP-SWD {scores[m].mean():.3f}", flush=True)

    rec = {"pairs": len(names), "metric": "Laplacian-pyramid sliced Wasserstein x 1e3",
           "levels": 4, "patches_per_level": 2048, "patch": 7, "directions": 256,
           "size": args.size, "methods": {}}
    rng = np.random.default_rng(0)
    for m, s in scores.items():
        means = s[rng.integers(0, len(s), (10000, len(s)))].mean(axis=1)
        rec["methods"][m] = {"mean": float(s.mean()),
                             "ci_low": float(np.percentile(means, 2.5)),
                             "ci_high": float(np.percentile(means, 97.5)),
                             "per_level_mean": per_level[m].mean(axis=0).tolist()}
    rec["paired"] = {r: {m: paired(scores[r], s) for m, s in scores.items()
                         if not m.startswith("fdformer")}
                     for r in args.references if r in scores}
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    np.savez_compressed(os.path.splitext(args.json_out)[0] + "_per_pair.npz",
                        names=np.array(names), **scores)
    print("wrote", args.json_out)


if __name__ == "__main__":
    main()
