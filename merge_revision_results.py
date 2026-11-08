"""Fold the revision's measurements into results.json.

Every value written here is copied from a JSON that an evaluation script produced.
Nothing is entered by hand, so the file the figures and tables read from cannot drift
away from what was actually measured.

Usage
-----
python merge_revision_results.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# metrics_<name>.json -> key in results["metrics"]
METRIC_FILES = {
    "microast": "metrics_microast.json",
    "aespanet": "metrics_aespanet.json",
    "samam": "metrics_samam.json",
    "fdformer_v2_freq_fixed": "metrics_v2_freq_fixed.json",
    "fdformer_v2_freq_dwt": "metrics_v2_freq_dwt.json",
}

# eff_<name>.json -> key in results["efficiency"]
EFF_FILES = {
    "fdformer": "eff_fdformer.json",
    "sanet": "eff_sanet.json",
    "adaattn": "eff_adaattn.json",
    "stytr2": "eff_stytr2.json",
    "microast": "eff_microast.json",
    "aespanet": "eff_aespanet.json",
    "samam": "eff_samam.json",
}

# analysis JSONs carried over whole, under their own top-level key
ANALYSIS_FILES = {
    "frequency_evidence": "freq_analysis.json",
    "alpha_selection": "alpha_selection.json",
    "control_generalisation": "control_generalisation.json",
    "uncertainty": "uncertainty.json",
}


def load(path):
    p = os.path.join(ROOT, "results", path)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    rp = os.path.join(ROOT, "results.json")
    with open(rp) as f:
        res = json.load(f)
    res.setdefault("metrics", {})
    res.setdefault("efficiency", {})

    added = []
    for key, fn in METRIC_FILES.items():
        d = load(fn)
        if d and d.get("artfid") is not None:
            res["metrics"][key] = d
            added.append(f"metrics/{key}")
    for key, fn in EFF_FILES.items():
        d = load(fn)
        if d:
            res["efficiency"][key] = d
            added.append(f"efficiency/{key}")
    for key, fn in ANALYSIS_FILES.items():
        d = load(fn)
        if d:
            res[key] = d
            added.append(key)

    with open(rp, "w") as f:
        json.dump(res, f, indent=2)
    print(f"merged into {rp}:")
    for a in added:
        print("  " + a)
    missing = [f"{k} ({v})" for k, v in
               list(METRIC_FILES.items()) + list(EFF_FILES.items()) + list(ANALYSIS_FILES.items())
               if load(v) is None]
    if missing:
        print("still missing:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
