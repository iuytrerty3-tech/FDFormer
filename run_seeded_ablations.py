"""Train, stylise and score every ablation variant under several explicit seeds.

Finished stages are skipped, so the queue can be stopped and restarted at any point and
a training run continues from its last latest.pth. Results land in results/seeded/, one
JSON per run, and aggregate_seeded.py turns them into the table.

Run this on a GPU no other training job is using.

Usage
-----
python run_seeded_ablations.py --root runs/seeded --seeds 0 1 2
python run_seeded_ablations.py --only full single_path --seeds 0      # part of the queue
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

VARIANTS = {
    "full": "configs/v2_hf_rescaled.yaml",
    "single_path": "configs/v2_single_path.yaml",
    "freq_fixed": "configs/v2_freq_fixed.yaml",
    "freq_dwt": "configs/v2_freq_dwt.yaml",
    "two_branch_equal": "configs/v2_two_branch_equal.yaml",
}
# Style pairing of the 1000-pair benchmark.
CONTENT, STYLE = "eval/default/content", "eval/default/style"


def run(cmd, log):
    print(">>", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"failed ({code}), see {log}: {' '.join(cmd)}")


def gpu_is_free():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return sum("python" in l.lower() for l in out.splitlines()) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/seeded")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--only", nargs="+", default=list(VARIANTS))
    ap.add_argument("--max_iters", type=int, default=None, help="for a smoke test only")
    ap.add_argument("--allow_busy_gpu", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    if not args.allow_busy_gpu and not gpu_is_free():
        raise SystemExit("another Python process is on the GPU; refusing to start")
    py = sys.executable
    os.makedirs("results/seeded", exist_ok=True)

    # Seed-by-seed so that a partial queue still gives every variant the same seeds.
    for seed in args.seeds:
        for name in args.only:
            tag = f"{name}_s{seed}"
            out = os.path.join("results/seeded", tag + ".json")
            if os.path.exists(out) and not args.max_iters:
                continue
            d = os.path.join(args.root, tag)
            os.makedirs(d, exist_ok=True)
            log = os.path.join(d, "run.log")
            iters = args.max_iters or 80000
            final = os.path.join(d, f"ckpt_{iters:06d}.pth")
            t0 = time.time()
            if not os.path.exists(final):
                cmd = [py, "train.py", "--config", VARIANTS[name], "--seed", str(seed),
                       "--save_dir", d, "--final_only", "--max_iters", str(iters),
                       "--num_workers", str(args.num_workers)]
                latest = os.path.join(d, "latest.pth")
                if os.path.exists(latest):
                    cmd += ["--resume", latest]
                run(cmd, log)
            train_s = time.time() - t0
            rec = {"variant": name, "seed": seed, "iters": iters, "train_seconds": train_s,
                   "torch": torch.__version__, "config": VARIANTS[name]}
            alphas = [1.0] if name == "single_path" else [1.0, 1.4]
            for a in alphas:
                ev = os.path.join(d, f"eval_a{int(a * 100):03d}")
                run([py, "batch_stylize.py", "--ckpt", final, "--content_dir", CONTENT,
                     "--style_dir", STYLE, "--out_dir", ev, "--size", "256",
                     "--alpha_low", str(a), "--alpha_high", str(a)], log)
                mj = os.path.join(d, f"metrics_a{int(a * 100):03d}.json")
                run([py, "eval_metrics.py", "--stylized_dir", os.path.join(ev, "stylized"),
                     "--content_dir", CONTENT, "--style_dir", STYLE, "--json_out", mj], log)
                with open(mj) as f:
                    rec[f"alpha_{a}"] = json.load(f)
                if rec[f"alpha_{a}"].get("artfid") is None:
                    raise SystemExit(f"ArtFID missing for {tag}: is art_inception.pth present?")
            if not args.max_iters:
                with open(out, "w") as f:
                    json.dump(rec, f, indent=2)
            print(f"finished {tag}", flush=True)


if __name__ == "__main__":
    main()
