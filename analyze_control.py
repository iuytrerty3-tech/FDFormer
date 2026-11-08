"""Does the separation of the two coefficients hold beyond a single example?

The grid figure shows four content-style cases. Four pictures cannot establish that
the behaviour is general, so this script sweeps each coefficient over a large sample
of pairs and records both the direct response and the crossed one.

For every pair and every setting of a coefficient, with the other held at 1, two
disjoint statistics are measured against the style image:

  chroma distance   mean and standard deviation of the a, b channels in CIE Lab
  texture distance  Gram matrices of the high-pass of relu3_1 and relu4_1, taken on
                    the luminance of the image so that nothing chromatic enters

Both are divided by the same distance between content and style, so 0 means the
output matches the style and 1 means it is unchanged from the content. The output
holds the mean and standard error at each setting, for all four combinations of
statistic and coefficient, so the crossed responses can be plotted beside the direct
ones instead of only being asserted to be small.

Usage
-----
python analyze_control.py --ckpt experiments/fdformer_v2/ckpt_080000.pth --num 300 \
    --json_out results/control_generalisation.json
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from analyze_frequency import (_d_chroma, _d_texture, chroma_stats, texture_gram,
                               texture_gram_shape)
from models import FDFormer

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/control_generalisation.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ck.get("cfg", {}).get("model", {"embed_dim": 256, "heads": 4})
    model = FDFormer(embed_dim=mc["embed_dim"], heads=mc["heads"], pretrained_vgg=True,
                     freq_decouple=mc.get("freq_decouple", True),
                     use_gate=mc.get("use_gate", True),
                     freq_mode=mc.get("freq_mode", "learnable"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    alphas = [float(x) for x in args.alphas.split(",")]
    names = sorted(f for f in os.listdir(args.content_dir)
                   if f.lower().endswith(_EXTS))[:args.num]
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])

    metrics = ("colour", "texture", "texture_shape")
    keys = [f"{m}__alpha_{k}" for m in metrics for k in ("low", "high")]
    acc = {k: [[] for _ in alphas] for k in keys}

    with torch.no_grad():
        for i, n in enumerate(names):
            c = tf(Image.open(os.path.join(args.content_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(os.path.join(args.style_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s_ch = chroma_stats(s)
            s_tx = texture_gram(model.encoder, s)
            s_ts = texture_gram_shape(model.encoder, s)
            ref_ch = _d_chroma(chroma_stats(c), s_ch)
            ref_tx = _d_texture(texture_gram(model.encoder, c), s_tx)
            ref_ts = _d_texture(texture_gram_shape(model.encoder, c), s_ts)
            if min(ref_ch, ref_tx, ref_ts) <= 0:
                continue
            for knob in ("low", "high"):
                for j, a in enumerate(alphas):
                    al, ah = (a, 1.0) if knob == "low" else (1.0, a)
                    out = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                    acc[f"colour__alpha_{knob}"][j].append(
                        _d_chroma(chroma_stats(out), s_ch) / ref_ch)
                    acc[f"texture__alpha_{knob}"][j].append(
                        _d_texture(texture_gram(model.encoder, out), s_tx) / ref_tx)
                    acc[f"texture_shape__alpha_{knob}"][j].append(
                        _d_texture(texture_gram_shape(model.encoder, out), s_ts) / ref_ts)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(names)} pairs", flush=True)

    curves = {}
    for k, cols in acc.items():
        mean, sem = [], []
        for col in cols:
            v = np.asarray(col, dtype=float)
            mean.append(float(v.mean()))
            sem.append(float(v.std(ddof=1) / np.sqrt(len(v))))
        curves[k] = {"mean": mean, "sem": sem}

    # Total movement of each statistic across the sweep, direct against crossed.
    span = {k: float(max(v["mean"]) - min(v["mean"])) for k, v in curves.items()}
    rec = {"checkpoint": args.ckpt, "alphas": alphas,
           "pairs": len(acc["colour__alpha_low"][0]), "curves": curves, "span": span,
           "colour_direct_over_crossed": span["colour__alpha_low"] / max(span["colour__alpha_high"], 1e-12),
           "texture_direct_over_crossed": span["texture__alpha_high"] / max(span["texture__alpha_low"], 1e-12),
           "texture_shape_direct_over_crossed": span["texture_shape__alpha_high"] / max(span["texture_shape__alpha_low"], 1e-12)}

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    for k in keys:
        print(f"{k}: span {span[k]:.4f}")
    print(f"colour, direct over crossed: {rec['colour_direct_over_crossed']:.2f}")
    print(f"texture, direct over crossed: {rec['texture_direct_over_crossed']:.2f}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
