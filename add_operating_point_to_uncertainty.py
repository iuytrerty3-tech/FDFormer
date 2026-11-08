"""Add the reported operating point to the paired comparisons.

eval_uncertainty.py scores every method once and caches each method's per-pair values.
The main table, however, reports FDFormer at the coefficient chosen on held-out data,
not only at the training point, and a reader is entitled to the paired tests at the
setting the paper actually quotes. Rescoring all seven methods to add one column would
be wasteful, so this scores the one new folder and reuses the cached arrays for the
rest. The pairing is preserved because the cache stores the pair names alongside the
values and we score exactly those names, in that order.

Usage
-----
python add_operating_point_to_uncertainty.py --stylized_dir eval/v2_alpha140/stylized \
    --label "fdformer_a140" --json_out results/uncertainty_a140.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from eval_uncertainty import boot_ci, paired_stats
from losses.losses import calc_mean_std
from models.vgg import VGGEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stylized_dir", default="eval/v2_alpha140/stylized")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--cache", default="results/uncertainty_per_pair.npz")
    ap.add_argument("--label", default="fdformer_a140")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/uncertainty_a140.json")
    args = ap.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    baselines = sorted({k.split("__")[0] for k in cache.files if "__" in k})
    print(f"{len(names)} pairs cached for: {', '.join(baselines)}")

    device = torch.device(args.device)
    encoder = VGGEncoder(pretrained=True).to(device).eval()
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])

    def load(d, n):
        return tf(Image.open(os.path.join(d, n)).convert("RGB")).unsqueeze(0).to(device)

    lp, st = [], []
    with torch.no_grad():
        for i, n in enumerate(names):
            out = load(args.stylized_dir, n)
            con = load(args.content_dir, n)
            sty = load(args.style_dir, n)
            lp.append(float(lpips_fn(out * 2 - 1, con * 2 - 1).mean()))
            of, sf = encoder(out), encoder(sty)
            s = 0.0
            for o, t in zip(of, sf):
                om, osd = calc_mean_std(o)
                sm, ssd = calc_mean_std(t)
                s += float(F.mse_loss(om, sm) + F.mse_loss(osd, ssd))
            st.append(s)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
    lp, st = np.asarray(lp), np.asarray(st)

    m_lp, lo_lp, hi_lp = boot_ci(lp)
    m_st, lo_st, hi_st = boot_ci(st)
    rec = {"pairs": len(names), "label": args.label, "stylized_dir": args.stylized_dir,
           "lpips_mean": m_lp, "lpips_ci_low": lo_lp, "lpips_ci_high": hi_lp,
           "style_mean": m_st, "style_ci_low": lo_st, "style_ci_high": hi_st,
           "paired": {}}
    print(f"{args.label}: LPIPS {m_lp:.4f} [{lo_lp:.4f}, {hi_lp:.4f}]   "
          f"style {m_st:.4f} [{lo_st:.4f}, {hi_st:.4f}]")

    for b in baselines:
        if b == "fdformer":
            continue
        rec["paired"][b] = {
            "lpips": paired_stats(lp, cache[f"{b}__lpips"]),
            "style": paired_stats(st, cache[f"{b}__style"]),
        }
        d = rec["paired"][b]
        print(f"  vs {b:10s} LPIPS {d['lpips']['diff_mean']:+.4f} "
              f"[{d['lpips']['diff_ci_low']:+.4f}, {d['lpips']['diff_ci_high']:+.4f}]  "
              f"style {d['style']['diff_mean']:+.4f} "
              f"[{d['style']['diff_ci_low']:+.4f}, {d['style']['diff_ci_high']:+.4f}]")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
