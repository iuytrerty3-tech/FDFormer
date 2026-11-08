"""Merge one baseline's measured efficiency row into results.json.

run_all.py only writes FDFormer's efficiency row. Each baseline's params/latency/
max-res are measured inside its own repo by the *_paired.py --bench commands, which
write results/eff_<key>.json. This helper drops that record into
results.json["efficiency"][<key>] using the SAME field names the paper's Table 3
expects (params_M, flops_G, latency_ms, max_res). It copies the measured JSON
unchanged.

Usage:
    python merge_efficiency.py --key sanet --eff results/eff_sanet.json
    python merge_efficiency.py --key stytr2 --eff results/eff_stytr2.json
"""
import argparse
import json
import os

VALID = ("fdformer", "adain", "sanet", "adaattn", "styleformer", "stytr2", "cast", "styleid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=VALID)
    ap.add_argument("--eff", required=True, help="results/eff_<key>.json produced by *_paired.py --bench")
    ap.add_argument("--results", default="results.json")
    args = ap.parse_args()

    if not os.path.exists(args.eff):
        raise SystemExit(f"{args.eff} not found; run the baseline's --bench step first.")
    with open(args.eff) as f:
        rec = json.load(f)

    results = {}
    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)
    results.setdefault("efficiency", {})
    results["efficiency"][args.key] = rec
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"merged efficiency['{args.key}'] <- {args.eff}")
    print(f"  params_M={rec.get('params_M')}  flops_G={rec.get('flops_G')}  "
          f"latency_ms={rec.get('latency_ms')}  max_res={rec.get('max_res')}  "
          f"device={rec.get('device')}")


if __name__ == "__main__":
    main()
