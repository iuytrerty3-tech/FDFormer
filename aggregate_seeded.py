"""Summarise results/seeded/*.json: mean and sample SD across seeds, per variant and alpha.

Same-seed differences from the full model are reported too, since every variant under a
seed sees the same sequence of training examples. No run is dropped: an incomplete
variant is reported with the seeds it has, and the seed list is printed beside it. The
single-path variant has no second coefficient, so it appears at alpha=1 only.

Usage
-----
python aggregate_seeded.py --json_out results/seeded_summary.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ALPHAS = ("alpha_1.0", "alpha_1.4")


def summarise(runs, key):
    rec = {}
    for metric in ("artfid", "lpips_c", "style_dist"):
        table = {}
        for r in runs:
            if key in r:
                table.setdefault(r["variant"], {})[r["seed"]] = r[key][metric]
        full = table.get("full", {})
        for v, by_seed in table.items():
            xs = np.array([by_seed[s] for s in sorted(by_seed)])
            e = rec.setdefault(v, {}).setdefault(metric, {})
            e.update(seeds=sorted(by_seed), values=xs.tolist(), mean=float(xs.mean()),
                     sd=float(xs.std(ddof=1)) if len(xs) > 1 else None)
            shared = sorted(set(by_seed) & set(full))
            if v != "full" and shared:
                d = np.array([by_seed[s] - full[s] for s in shared])
                e["diff_from_full"] = {"seeds": shared, "values": d.tolist(), "mean": float(d.mean()),
                                       "sd": float(d.std(ddof=1)) if len(d) > 1 else None}
                if len(d) > 1:
                    # t interval over seeds: with three seeds it is wide, and that is the point.
                    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    e["diff_from_full"]["ci95"] = [float(d.mean() - half), float(d.mean() + half)]
                    e["diff_from_full"]["p_paired_t"] = float(stats.ttest_rel(
                        [by_seed[s] for s in shared], [full[s] for s in shared]).pvalue)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/seeded")
    ap.add_argument("--json_out", default="results/seeded_summary.json")
    args = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "*_s*.json")))]
    rec = {key: summarise(runs, key) for key in ALPHAS}
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    fmt = lambda m, sd, p=2: f"{m:.{p}f}" + (f"+-{sd:.{p}f}" if sd is not None else "")
    for key in ALPHAS:
        print(key)
        print(f"{'variant':20s} {'seeds':10s} {'ArtFID':>16s} {'dArtFID':>16s} {'LPIPS':>14s} {'Style':>14s}")
        for v, e in rec[key].items():
            a = e["artfid"]
            d = a.get("diff_from_full")
            print(f"{v:20s} {str(a['seeds']):10s} {fmt(a['mean'], a['sd']):>16s} "
                  f"{fmt(d['mean'], d['sd']) if d else '-':>16s} "
                  f"{fmt(e['lpips_c']['mean'], e['lpips_c']['sd'], 3):>14s} "
                  f"{fmt(e['style_dist']['mean'], e['style_dist']['sd'], 3):>14s}")


if __name__ == "__main__":
    main()
