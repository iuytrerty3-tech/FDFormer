"""Uncertainty around every number the comparison table reports.

The main table gives one figure per method per metric. This script says how much of
that figure is signal. It works on the folders the methods have already written, so
it re-scores nothing by hand.

Per-pair metrics.
    Content LPIPS and the style-statistic distance are defined per content-style pair,
    so each method yields 1000 values. We report the mean with a 95% confidence
    interval from 10,000 bootstrap resamples over pairs. Because every method is
    scored on the SAME pairs, the comparison against a baseline is paired: we
    bootstrap the per-pair difference as well, and run a two-sided Wilcoxon signed-rank
    test on it. A confidence interval on the difference that excludes zero is what
    licenses a claim that one method is ahead of another.

ArtFID.
    ArtFID is a set-level quantity: it has no per-pair value to resample. We instead
    split the benchmark into five disjoint subsets of 200 pairs and score each, which
    gives a spread across independent samples of the same size. Set-level Fréchet
    distances are biased upward on smaller samples, so these five values sit above the
    1000-pair figure and are reported as a measure of variability, not as a restatement
    of the headline number.

Usage
-----
python eval_uncertainty.py --methods fdformer=eval/v2 sanet=eval/sanet ... \
    --reference fdformer --json_out results/uncertainty.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from losses.losses import calc_mean_std
from models.vgg import VGGEncoder

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def names_of(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def load(path, size, device):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(img).unsqueeze(0).to(device)


def per_pair_metrics(stylized_dir, content_dir, style_dir, names, size, device,
                     encoder, lpips_fn):
    lp, st = [], []
    with torch.no_grad():
        for n in names:
            out = load(os.path.join(stylized_dir, n), size, device)
            con = load(os.path.join(content_dir, n), size, device)
            sty = load(os.path.join(style_dir, n), size, device)
            lp.append(float(lpips_fn(out * 2 - 1, con * 2 - 1).mean()))
            of, sf = encoder(out), encoder(sty)
            s = 0.0
            for o, t in zip(of, sf):
                om, osd = calc_mean_std(o)
                sm, ssd = calc_mean_std(t)
                s += float(F.mse_loss(om, sm) + F.mse_loss(osd, ssd))
            st.append(s)
    return np.asarray(lp), np.asarray(st)


def boot_ci(x, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def paired_stats(a, b, n_boot=10000, seed=0):
    """a - b, bootstrapped over the shared pairs, plus a Wilcoxon signed-rank test."""
    d = a - b
    mean, lo, hi = boot_ci(d, n_boot=n_boot, seed=seed)
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:                                   # noqa: BLE001
        p = None
    return {"diff_mean": mean, "diff_ci_low": lo, "diff_ci_high": hi, "wilcoxon_p": p}


def artfid_subsets(stylized_dir, content_dir, style_dir, names, n_splits, workdir):
    """ArtFID on disjoint subsets of equal size, via the official package."""
    from art_fid.art_fid import compute_art_fid
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = len(names) // n_splits
    vals = []
    for k in range(n_splits):
        sub = names[k * chunk:(k + 1) * chunk]
        dirs = {}
        for tag, src in (("stylized", stylized_dir), ("content", content_dir),
                         ("style", style_dir)):
            d = os.path.join(workdir, f"split{k}", tag)
            os.makedirs(d, exist_ok=True)
            for n in sub:
                dst = os.path.join(d, n)
                if not os.path.exists(dst):
                    Image.open(os.path.join(src, n)).convert("RGB").save(dst)
            dirs[tag] = d
        v = compute_art_fid(path_to_stylized=dirs["stylized"], path_to_style=dirs["style"],
                            path_to_content=dirs["content"], batch_size=8, device=device,
                            mode="art_fid_inf", content_metric="lpips", num_workers=1)
        vals.append(float(v))
        print(f"    subset {k + 1}/{n_splits} ({len(sub)} pairs): ArtFID {v:.2f}", flush=True)
    v = np.asarray(vals)
    return {"n_splits": n_splits, "pairs_per_split": chunk, "values": vals,
            "mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=root pairs; root holds content/ style/ stylized/")
    ap.add_argument("--reference", default="fdformer",
                    help="method the paired comparisons are made against")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--artfid_splits", type=int, default=5)
    ap.add_argument("--skip_artfid", action="store_true")
    ap.add_argument("--workdir", default="eval/_subsets")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/uncertainty.json")
    args = ap.parse_args()

    roots = dict(m.split("=", 1) for m in args.methods)
    device = torch.device(args.device)
    encoder = VGGEncoder(pretrained=True).to(device).eval()
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # The shared pair list: names present for every method, so comparisons are paired.
    shared = None
    for root in roots.values():
        n = set(names_of(os.path.join(root, "stylized")))
        shared = n if shared is None else (shared & n)
    names = sorted(shared)
    print(f"{len(names)} pairs scored for every method")

    rec = {"pairs": len(names), "methods": {}}
    per = {}
    for name, root in roots.items():
        print(f"[{name}] per-pair metrics ...", flush=True)
        lp, st = per_pair_metrics(os.path.join(root, "stylized"),
                                  os.path.join(root, "content"),
                                  os.path.join(root, "style"),
                                  names, args.size, device, encoder, lpips_fn)
        per[name] = {"lpips": lp, "style": st}
        m_lp, lo_lp, hi_lp = boot_ci(lp)
        m_st, lo_st, hi_st = boot_ci(st)
        rec["methods"][name] = {
            "lpips_mean": m_lp, "lpips_ci_low": lo_lp, "lpips_ci_high": hi_lp,
            "style_mean": m_st, "style_ci_low": lo_st, "style_ci_high": hi_st,
        }
        print(f"    LPIPS {m_lp:.4f} [{lo_lp:.4f}, {hi_lp:.4f}]   "
              f"style {m_st:.4f} [{lo_st:.4f}, {hi_st:.4f}]")

    ref = args.reference
    if ref in per:
        rec["paired_vs_" + ref] = {}
        for name in per:
            if name == ref:
                continue
            rec["paired_vs_" + ref][name] = {
                "lpips": paired_stats(per[ref]["lpips"], per[name]["lpips"]),
                "style": paired_stats(per[ref]["style"], per[name]["style"]),
            }
            d = rec["paired_vs_" + ref][name]["lpips"]
            print(f"[{ref} - {name}] LPIPS diff {d['diff_mean']:+.4f} "
                  f"[{d['diff_ci_low']:+.4f}, {d['diff_ci_high']:+.4f}] p={d['wilcoxon_p']}")

    if not args.skip_artfid:
        rec["artfid_subsets"] = {}
        for name, root in roots.items():
            print(f"[{name}] ArtFID over {args.artfid_splits} disjoint subsets ...", flush=True)
            rec["artfid_subsets"][name] = artfid_subsets(
                os.path.join(root, "stylized"), os.path.join(root, "content"),
                os.path.join(root, "style"), names, args.artfid_splits,
                os.path.join(args.workdir, name))

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    np.savez(os.path.splitext(args.json_out)[0] + "_per_pair.npz",
             names=np.asarray(names),
             **{f"{k}__{m}": per[k][m] for k in per for m in ("lpips", "style")})
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
