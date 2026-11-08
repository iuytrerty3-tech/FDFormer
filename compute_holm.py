"""Holm correction over the family of paired comparisons, and the alpha=1.4 table.

The family is every baseline on both per-pair measures at both operating points: 6 x 2 x 2
= 24 Wilcoxon tests. The correction uses the p-values the bootstrap script recorded.

Usage
-----
python compute_holm.py --json_out results/holm.json
"""
import argparse
import json


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a100", default="results/uncertainty.json")
    ap.add_argument("--a140", default="results/uncertainty_a140.json")
    ap.add_argument("--json_out", default="results/holm.json")
    args = ap.parse_args()
    rows = []
    for alpha, path in ((1.0, args.a100), (1.4, args.a140)):
        u = json.load(open(path))
        paired = u.get("paired_vs_fdformer") or u["paired"]
        for base, by_metric in paired.items():
            for metric in ("lpips", "style"):
                r = by_metric[metric]
                rows.append({"alpha": alpha, "baseline": base, "metric": metric, **r})
    for r, a in zip(rows, holm([r["wilcoxon_p"] for r in rows])):
        r["holm_p"] = a
        r["significant_holm_05"] = a < 0.05
        r["ci_excludes_zero"] = r["diff_ci_low"] > 0 or r["diff_ci_high"] < 0
    with open(args.json_out, "w") as f:
        json.dump({"family_size": len(rows), "method": "Holm", "rows": rows}, f, indent=2)
    for r in rows:
        print(f"{r['alpha']:.1f} {r['baseline']:9s} {r['metric']:5s} {r['diff_mean']:+.3f} "
              f"[{r['diff_ci_low']:+.3f},{r['diff_ci_high']:+.3f}] p={r['wilcoxon_p']:.2e} "
              f"holm={r['holm_p']:.2e} sig={r['significant_holm_05']} ci={r['ci_excludes_zero']}")


if __name__ == "__main__":
    main()
