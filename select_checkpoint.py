"""Choose which training checkpoint to report, on held-out data.

The last checkpoint is not automatically the best one, and picking it by looking at
the benchmark would repeat the mistake the operating point was criticised for. This
scores a set of checkpoints on the validation pairs alone, at a fixed coefficient,
and reports the ordering. Only the winner is then run on the benchmark, by
select_alpha.py or eval_metrics.py.

Usage
-----
python select_checkpoint.py --exp experiments/fdformer_v2 --steps 50000,60000,70000,80000 \
    --alpha 1.4 --json_out results/checkpoint_selection.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/fdformer_v2")
    ap.add_argument("--steps", default="50000,60000,70000,80000")
    ap.add_argument("--val_dir", default="datasets/val_pairs")
    ap.add_argument("--alpha", type=float, default=1.4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--json_out", default="results/checkpoint_selection.json")
    args = ap.parse_args()

    py = sys.executable
    steps = [int(s) for s in args.steps.split(",")]
    out = {}
    for st in steps:
        ck = os.path.join(args.exp, f"ckpt_{st:06d}.pth")
        if not os.path.isfile(ck):
            print(f"[skip] {ck} missing")
            continue
        root = f"eval/ckpt_{args.tag}_{st}"
        mj = f"results/metrics_ckpt_{args.tag}_{st}.json"
        if len(os.listdir(os.path.join(root, "stylized"))) if os.path.isdir(
                os.path.join(root, "stylized")) else 0:
            pass
        if not (os.path.isdir(os.path.join(root, "stylized"))
                and len(os.listdir(os.path.join(root, "stylized"))) >= 1):
            sh([py, "batch_stylize.py", "--ckpt", ck,
                "--content_dir", os.path.join(args.val_dir, "content"),
                "--style_dir", os.path.join(args.val_dir, "style"),
                "--out_dir", root, "--size", args.size,
                "--alpha_low", args.alpha, "--alpha_high", args.alpha])
        if not os.path.exists(mj):
            sh([py, "eval_metrics.py",
                "--stylized_dir", os.path.join(root, "stylized"),
                "--content_dir", os.path.join(root, "content"),
                "--style_dir", os.path.join(root, "style"),
                "--size", args.size, "--json_out", mj])
        with open(mj) as f:
            out[st] = json.load(f)
        print(f"  step {st}: val ArtFID {out[st]['artfid']:.2f}  "
              f"LPIPS {out[st]['lpips_c']:.4f}", flush=True)

    best = min(out, key=lambda s: out[s]["artfid"])
    print(f"selected checkpoint: step {best} (validation ArtFID {out[best]['artfid']:.2f})")
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"experiment": args.exp, "alpha": args.alpha,
                   "validation": {str(k): v for k, v in out.items()},
                   "selected_step": best}, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
