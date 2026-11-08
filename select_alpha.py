"""Choose the inference coefficient on held-out data, then score the benchmark once.

The two coefficients are free at inference, so the operating point has to be chosen
somehow. Choosing it by reading the benchmark would make the reported figure a best
case over a grid rather than a prediction, and a reader is right to distrust that.

This script keeps the two roles apart. It sweeps the shared setting
alpha_low = alpha_high over a grid on the validation pairs built by build_val_set.py,
which share no content and no style image with the benchmark, and takes the minimiser
of validation ArtFID. Only that single setting is then run on the benchmark. The
sweep over the benchmark itself is still recorded, so the gap between the chosen
point and the benchmark's own minimum is visible rather than hidden.

Usage
-----
python select_alpha.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --val_dir datasets/val_pairs --test_content eval/default/content \
    --test_style eval/default/style --json_out results/alpha_selection.json
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


def stylise_and_score(py, ckpt, content_dir, style_dir, out_root, alpha, size, num,
                      metrics_json):
    if not (os.path.isdir(os.path.join(out_root, "stylized"))
            and len(os.listdir(os.path.join(out_root, "stylized"))) >= (num or 1)):
        sh([py, "batch_stylize.py", "--ckpt", ckpt, "--content_dir", content_dir,
            "--style_dir", style_dir, "--out_dir", out_root, "--size", size,
            "--alpha_low", alpha, "--alpha_high", alpha] + (["--num", num] if num else []))
    if not os.path.exists(metrics_json):
        sh([py, "eval_metrics.py",
            "--stylized_dir", os.path.join(out_root, "stylized"),
            "--content_dir", os.path.join(out_root, "content"),
            "--style_dir", os.path.join(out_root, "style"),
            "--size", size, "--json_out", metrics_json])
    with open(metrics_json) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--val_dir", default="datasets/val_pairs")
    ap.add_argument("--test_content", default="eval/default/content")
    ap.add_argument("--test_style", default="eval/default/style")
    ap.add_argument("--grid", default="1.0,1.1,1.25,1.4,1.5,1.6,1.75,2.0")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val_num", type=int, default=0, help="0 = all validation pairs")
    ap.add_argument("--test_num", type=int, default=0)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--also_sweep_test", action="store_true",
                    help="additionally score the whole grid on the benchmark, for disclosure")
    ap.add_argument("--json_out", default="results/alpha_selection.json")
    args = ap.parse_args()

    py = sys.executable
    grid = [float(x) for x in args.grid.split(",")]

    print("sweeping the validation pairs")
    val = {}
    for a in grid:
        tag = f"{a:.2f}".replace(".", "")
        rec = stylise_and_score(py, args.ckpt,
                                os.path.join(args.val_dir, "content"),
                                os.path.join(args.val_dir, "style"),
                                f"eval/val_{args.tag}_a{tag}", a, args.size,
                                args.val_num, f"results/metrics_val_{args.tag}_a{tag}.json")
        val[a] = rec
        print(f"  alpha={a:.2f}  val ArtFID {rec['artfid']:.2f}  LPIPS {rec['lpips_c']:.4f}")

    best = min(grid, key=lambda a: val[a]["artfid"])
    print(f"selected alpha = {best:.2f} (validation ArtFID {val[best]['artfid']:.2f})")

    print("scoring the benchmark at the selected setting")
    tag = f"{best:.2f}".replace(".", "")
    test_best = stylise_and_score(py, args.ckpt, args.test_content, args.test_style,
                                  f"eval/test_{args.tag}_a{tag}", best, args.size,
                                  args.test_num, f"results/metrics_test_{args.tag}_a{tag}.json")
    print(f"  benchmark at alpha={best:.2f}: ArtFID {test_best['artfid']:.2f}  "
          f"LPIPS {test_best['lpips_c']:.4f}")

    rec = {"checkpoint": args.ckpt, "grid": grid,
           "validation": {str(a): val[a] for a in grid},
           "selected_alpha": best,
           "benchmark_at_selected_alpha": test_best}

    if args.also_sweep_test:
        print("scoring the whole grid on the benchmark, for disclosure")
        test = {}
        for a in grid:
            t = f"{a:.2f}".replace(".", "")
            test[a] = stylise_and_score(py, args.ckpt, args.test_content, args.test_style,
                                        f"eval/test_{args.tag}_a{t}", a, args.size,
                                        args.test_num, f"results/metrics_test_{args.tag}_a{t}.json")
            print(f"  alpha={a:.2f}  benchmark ArtFID {test[a]['artfid']:.2f}")
        rec["benchmark_sweep"] = {str(a): test[a] for a in grid}
        b = min(grid, key=lambda a: test[a]["artfid"])
        rec["benchmark_own_minimum"] = {"alpha": b, "artfid": test[b]["artfid"]}
        print(f"  benchmark's own minimum is at alpha={b:.2f} "
              f"(ArtFID {test[b]['artfid']:.2f}); the selected point gives "
              f"{test_best['artfid']:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
