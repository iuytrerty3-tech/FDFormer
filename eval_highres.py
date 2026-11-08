"""Score stylisations at the resolution they were produced.

The main benchmark is scored at 256x256. A method's behaviour can change with resolution,
so this script runs each method that can reach a size on an evenly spaced sample of the
benchmark pairs and scores the actual outputs at that size:

  content LPIPS on a 4x4 grid of 512x512 crops, so the network sees native pixels,
  the VGG style-statistic distance over the full image,
  the Laplacian-pyramid sliced Wasserstein distance, which involves no network.

COCO photographs are about 640 pixels on a side, so at 2048 the content input is itself
enlarged by the methods' own preprocessing. The score measures rendering at that size on
that input and says nothing about native 2048-pixel photographs.

Usage
-----
python eval_highres.py prepare --num 100 --out work/highres
python eval_highres.py run --root work/highres --size 2048 \
    --methods fdformer fdformer_a140 microast
python eval_highres.py score --root work/highres --size 2048 \
    --methods fdformer fdformer_a140 microast --json_out results/highres_2048.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_style_independent import lp_swd
from losses.losses import calc_mean_std
from models.vgg import VGGEncoder

BASE = os.environ.get("FDFORMER_BASELINES", "baselines")
CKPT = "experiments/fdformer_v2/ckpt_080000.pth"
WRAPPERS = {
    "microast": ("MicroAST", "microast_paired.py", []),
    "sanet": ("SANET", "sanet_paired.py", ["--vgg", "vgg_normalised.pth", "--decoder",
              "decoder_iter_500000.pth", "--transform", "transformer_iter_500000.pth"]),
    "adaattn": ("AdaAttN", "adaattn_paired.py",
                ["--image_encoder_path", "checkpoints/vgg_normalised.pth"]),
    "stytr2": ("StyTR-2", "stytr2_paired.py", []),
}


def prepare(args):
    """Copy the original files of every k-th pair, named as the wrappers expect."""
    rows = [l.rstrip("\n").split("\t") for l in open("manifests/benchmark_pairs.tsv", encoding="utf-8")][1:]
    step = len(rows) // args.num
    c_dir, s_dir = os.path.join(args.out, "content"), os.path.join(args.out, "style")
    os.makedirs(c_dir, exist_ok=True)
    os.makedirs(s_dir, exist_ok=True)
    for r in rows[::step][:args.num]:
        Image.open(os.path.join("datasets/coco/train2017", r[1])).convert("RGB").save(
            os.path.join(c_dir, r[0] + ".png"))
        Image.open(os.path.join("datasets/style_flat", r[3])).convert("RGB").save(
            os.path.join(s_dir, r[0] + ".png"))
    print("prepared", len(os.listdir(c_dir)), "pairs in", args.out)


def run(args):
    py = sys.executable
    c = os.path.abspath(os.path.join(args.root, "content"))
    s = os.path.abspath(os.path.join(args.root, "style"))
    for m in args.methods:
        out = os.path.abspath(os.path.join(args.root, f"out_{args.size}", m))
        if m.startswith("fdformer"):
            a = "1.4" if m.endswith("a140") else "1.0"
            cmd = [py, "batch_stylize.py", "--ckpt", CKPT, "--content_dir", c, "--style_dir", s,
                   "--out_dir", out, "--size", str(args.size), "--low_memory",
                   "--alpha_low", a, "--alpha_high", a]
            cwd = "."
        else:
            d, script, extra = WRAPPERS[m]
            os.makedirs(os.path.join(out, "stylized"), exist_ok=True)
            cmd = [py, script, "--content_dir", c, "--style_dir", s,
                   "--out_dir", os.path.join(out, "stylized"), "--size", str(args.size), *extra]
            cwd = os.path.join(BASE, d)
        print(">>", " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=cwd)
        if code != 0:
            print(f"[{m}] exited {code}: no result at {args.size}", flush=True)


def to_tensor(path, size, device):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def score(args):
    import lpips
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = VGGEncoder(pretrained=True).to(device).eval()
    lp = lpips.LPIPS(net="vgg").to(device).eval()
    names = sorted(os.listdir(os.path.join(args.root, "content")))
    crop = min(512, args.size // 2)
    rec = {"size": args.size, "pairs": len(names), "crop": crop, "methods": {}}
    per = {}
    for m in args.methods:
        d = os.path.join(args.root, f"out_{args.size}", m, "stylized")
        if not os.path.isdir(d) or len(os.listdir(d)) < len(names):
            print(f"[{m}] incomplete outputs, skipped", flush=True)
            continue
        rows = []
        # batch_stylize numbers its outputs and the wrappers keep the input name; both follow
        # the sorted input order, so position identifies the pair.
        files = sorted(os.listdir(d))
        for i, n in enumerate(names):
            out = to_tensor(os.path.join(d, files[i]), args.size, device)
            produced = Image.open(os.path.join(d, files[i])).size
            c = to_tensor(os.path.join(args.root, "content", n), args.size, device)
            s = to_tensor(os.path.join(args.root, "style", n), args.size, device)
            with torch.no_grad():
                ls = []
                for y in np.linspace(0, args.size - crop, 4).astype(int):
                    for x in np.linspace(0, args.size - crop, 4).astype(int):
                        a = out[..., y:y + crop, x:x + crop]
                        b = c[..., y:y + crop, x:x + crop]
                        ls.append(float(lp(a * 2 - 1, b * 2 - 1).mean()))
                st = 0.0
                x_o, x_s = enc.normalize(out), enc.normalize(s)
                for block in enc.blocks:
                    x_o, x_s = block(x_o), block(x_s)
                    om, osd = calc_mean_std(x_o)
                    sm, ssd = calc_mean_std(x_s)
                    st += float(F.mse_loss(om, sm) + F.mse_loss(osd, ssd))
                del x_o, x_s
            swd, _ = lp_swd(out.cpu(), s.cpu(), i)
            rows.append((float(np.mean(ls)), st, swd))
            assert max(produced) == args.size, f"{m} returned {produced} at size {args.size}"
        arr = np.array(rows)
        per[m] = arr
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(arr), (10000, len(arr)))
        rec["methods"][m] = {
            k: {"mean": float(arr[:, j].mean()),
                "ci_low": float(np.percentile(arr[:, j][idx].mean(axis=1), 2.5)),
                "ci_high": float(np.percentile(arr[:, j][idx].mean(axis=1), 97.5))}
            for j, k in enumerate(("lpips_crops", "style_dist", "lp_swd"))}
        print(m, {k: round(v["mean"], 3) for k, v in rec["methods"][m].items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    np.savez_compressed(os.path.splitext(args.json_out)[0] + "_per_pair.npz", names=np.array(names), **per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "score"])
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--methods", nargs="+", default=["fdformer", "fdformer_a140", "microast"])
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
