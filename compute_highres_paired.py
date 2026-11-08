"""Paired tests on the per-pair high-resolution scores, Holm-corrected over the whole family.

Reads results/highres_<size>_per_pair.npz from eval_highres.py and writes
results/highres_paired.json. Differences are FDFormer at alpha=1 minus the other method, so
a positive value means FDFormer scored worse on that lower-is-better measure.
"""
import json

import numpy as np
from scipy.stats import wilcoxon

MEASURES = ("lpips_crops", "style_dist", "lp_swd")

tests = []
for size in (2048, 1024):
    z = np.load(f"results/highres_{size}_per_pair.npz")
    for m in z.files:
        if m in ("names", "fdformer"):
            continue
        for j, k in enumerate(MEASURES):
            d = z["fdformer"][:, j] - z[m][:, j]
            tests.append({"size": size, "method": m, "measure": k, "diff_mean": float(d.mean()),
                          "wilcoxon_p": float(wilcoxon(d).pvalue)})

# Holm step-down: adjusted values are monotone in the sorted order.
order = sorted(range(len(tests)), key=lambda i: tests[i]["wilcoxon_p"])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(tests) - rank) * tests[i]["wilcoxon_p"]))
    tests[i]["holm_p"] = running

with open("results/highres_paired.json", "w") as f:
    json.dump({"family": len(tests), "tests": tests}, f, indent=2)
for t in tests:
    print(t["size"], t["method"], t["measure"], round(t["diff_mean"], 3),
          "p=%.2e holm=%.2e" % (t["wilcoxon_p"], t["holm_p"]))
