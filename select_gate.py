"""Sweep the inference-time gate scale on held-out pairs.

The gate decides how much style each location admits. A scalar on its output moves
the whole image along a different trade-off than the blend coefficient does: alpha
scales the transferred feature, whereas this changes how much of that feature is
mixed with the content. Whether either is worth using is an empirical question, and
it has to be answered away from the benchmark.

Everything here runs on the validation pairs built by build_val_set.py. Only the
winner is carried to the benchmark, and the whole search grid is reported so that the
size of the search is visible rather than implied.

Usage
-----
python select_gate.py --ckpt experiments/fdformer_v2/ckpt_080000.pth --alpha 1.4 \
    --grid 0.8,0.9,1.0,1.1,1.2,1.3 --json_out results/gate_selection.json
"""
import argparse
import json
import os
import subprocess
import sys


def sh(cmd):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"command failed with code {r.returncode}")


def score(py, ckpt, content_dir, style_dir, root, alpha, gate, size, mj):
    stylized = os.path.join(root, "stylized")
    if not (os.path.isdir(stylized) and len(os.listdir(stylized)) >= 1):
        sh([py, "batch_stylize.py", "--ckpt", ckpt, "--content_dir", content_dir,
            "--style_dir", style_dir, "--out_dir", root, "--size", size,
            "--alpha_low", alpha, "--alpha_high", alpha, "--gate_scale", gate])
    if not os.path.exists(mj):
        sh([py, "eval_metrics.py", "--stylized_dir", stylized,
            "--content_dir", os.path.join(root, "content"),
            "--style_dir", os.path.join(root, "style"),
            "--size", size, "--json_out", mj])
    with open(mj) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--val_dir", default="datasets/val_pairs")
    ap.add_argument("--test_content", default="eval/default/content")
    ap.add_argument("--test_style", default="eval/default/style")
    ap.add_argument("--alpha", type=float, default=1.4)
    ap.add_argument("--grid", default="0.8,0.9,1.0,1.1,1.2,1.3")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--json_out", default="results/gate_selection.json")
    args = ap.parse_args()

    py = sys.executable
    grid = [float(x) for x in args.grid.split(",")]
    val = {}
    for g in grid:
        t = f"{g:.2f}".replace(".", "")
        val[g] = score(py, args.ckpt, os.path.join(args.val_dir, "content"),
                       os.path.join(args.val_dir, "style"),
                       f"eval/valg_{args.tag}_g{t}", args.alpha, g, args.size,
                       f"results/metrics_valg_{args.tag}_g{t}.json")
        print(f"  gate={g:.2f}  val ArtFID {val[g]['artfid']:.2f}  "
              f"LPIPS {val[g]['lpips_c']:.4f}", flush=True)

    best = min(grid, key=lambda g: val[g]["artfid"])
    print(f"selected gate scale = {best:.2f} (validation ArtFID {val[best]['artfid']:.2f})")

    rec = {"checkpoint": args.ckpt, "alpha": args.alpha, "grid": grid,
           "validation": {str(g): val[g] for g in grid}, "selected_gate": best}

    if abs(best - 1.0) > 1e-9:
        t = f"{best:.2f}".replace(".", "")
        print("scoring the benchmark at the selected gate scale")
        rec["benchmark_at_selected_gate"] = score(
            py, args.ckpt, args.test_content, args.test_style,
            f"eval/testg_{args.tag}_g{t}", args.alpha, best, args.size,
            f"results/metrics_testg_{args.tag}_g{t}.json")
        b = rec["benchmark_at_selected_gate"]
        print(f"  benchmark: ArtFID {b['artfid']:.2f}  LPIPS {b['lpips_c']:.4f}")
    else:
        print("the unscaled gate wins on validation; nothing to carry to the benchmark")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
