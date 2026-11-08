"""Check which orderings of the main table survive retraining under explicit seeds.

The FDFormer checkpoint in the main table comes from one unseeded run. This script sets
each seeded full-model run (results/seeded/full_s*.json) against the released baselines
(results/metrics_<method>.json) on every metric and operating point, and records in how
many seeds FDFormer is ahead. It also applies Holm's correction over every same-seed
comparison in results/seeded_summary.json.

Usage
-----
python aggregate_seeded.py
python compare_seeded_baselines.py --json_out results/seeded_vs_baselines.json
"""
import argparse
import glob
import json
import os

BASELINES = ("sanet", "adaattn", "stytr2", "microast", "aespanet", "samam")
METRICS = ("artfid", "lpips_c", "style_dist")
REPORTED = {"alpha_1.0": "results/metrics_v2.json", "alpha_1.4": "results/metrics_v2_alpha140.json"}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/seeded_summary.json")
    ap.add_argument("--json_out", default="results/seeded_vs_baselines.json")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in sorted(glob.glob("results/seeded/full_s*.json"))]
    base = {b: json.load(open(f"results/metrics_{b}.json")) for b in BASELINES}
    rec = {"seeds": [r["seed"] for r in runs], "orderings": {}, "reported_vs_seeded": {}}
    for key, path in REPORTED.items():
        reported = json.load(open(path))
        rec["reported_vs_seeded"][key] = {
            m: {"reported": reported[m], "seeded": [r[key][m] for r in runs]} for m in METRICS}
        for b in BASELINES:
            for m in METRICS:
                ours = [r[key][m] for r in runs]
                rec["orderings"].setdefault(key, {}).setdefault(b, {})[m] = {
                    "baseline": base[b][m], "seeded": ours,
                    "seeds_ahead": sum(o < base[b][m] for o in ours),
                    "reported_ahead": reported[m] < base[b][m]}

    summary = json.load(open(args.summary))
    tests = [(key, v, m, e[m]["diff_from_full"]["p_paired_t"])
             for key, variants in summary.items() for v, e in variants.items()
             for m in METRICS if "diff_from_full" in e[m]]
    adj = holm([t[3] for t in tests])
    rec["holm_same_seed"] = [{"alpha": k, "variant": v, "metric": m, "p": p, "p_holm": a}
                             for (k, v, m, p), a in zip(tests, adj)]

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)

    for key, by_base in rec["orderings"].items():
        print(key)
        for b, by_m in by_base.items():
            print(f"  {b:9s} " + "  ".join(
                f"{m} {d['baseline']:.3f}: ahead in {d['seeds_ahead']}/{len(runs)} seeds"
                f" (reported {'ahead' if d['reported_ahead'] else 'behind'})" for m, d in by_m.items()))
    print(f"Holm over {len(tests)} same-seed tests: smallest adjusted p "
          f"{min(adj):.3f}; unadjusted p < 0.05: "
          f"{[(t[0], t[1], t[2], round(t[3], 3)) for t in tests if t[3] < 0.05]}")


if __name__ == "__main__":
    main()
