"""The cross-effect, measured on the quantities the coefficients are supposed to control.

A distance to the style is the wrong instrument for this question. It is smallest at
the setting the model was trained at and rises on both sides, so a coefficient that
works exactly as intended produces a shallow U and appears to do very little. The
claim being tested is not that a coefficient reduces a distance; it is that one
coefficient governs how much colour the output takes from the painting while leaving
the brushwork alone, and the other the reverse. Those are two absolute properties of
the output, and they are what this script measures.

  colour travel   the fraction of the way the output's mean (a, b) chromaticity has
                  moved from the content's toward the style's, obtained by projecting
                  the output onto that direction. 0 leaves the photograph's palette,
                  1 arrives at the painting's.

  stroke energy   the high-pass energy of the output's luminance divided by the
                  content's. 1 is as smooth as the photograph, larger values carry
                  more visible mark-making.

Both are computed from the image alone, need no network, and share no term, so a
change in one cannot arithmetically induce a change in the other. Each coefficient is
swept with the other held at 1, and the result reports how far each measure travels
under each coefficient, over the whole sweep and over the usable middle of it.

Usage
-----
python analyze_control_absolute.py --num 300 --json_out results/control_absolute.json
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from analyze_frequency import srgb_to_lab
from models import FDFormer

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def to_np(t):
    return np.clip(t[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float64), 0, 1)


def mean_ab(img):
    lab = srgb_to_lab(img)
    return lab[..., 1:].reshape(-1, 2).mean(axis=0)


def highpass_energy(img):
    """Energy of the luminance above a 3x3 box average, per pixel."""
    lum = srgb_to_lab(img)[..., 0]
    k = np.ones((3, 3)) / 9.0
    pad = np.pad(lum, 1, mode="reflect")
    low = sum(k[i, j] * pad[i:i + lum.shape[0], j:j + lum.shape[1]]
              for i in range(3) for j in range(3))
    return float(((lum - low) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--usable", default="0.5,1.5",
                    help="range over which outputs remain sensible images")
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/control_absolute.json")
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
    lo, hi = (float(x) for x in args.usable.split(","))
    usable = [i for i, a in enumerate(alphas) if lo - 1e-9 <= a <= hi + 1e-9]
    names = sorted(f for f in os.listdir(args.content_dir)
                   if f.lower().endswith(_EXTS))[:args.num]
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])

    keys = [f"{m}__alpha_{k}" for m in ("colour_travel", "stroke_energy")
            for k in ("low", "high")]
    acc = {k: [[] for _ in alphas] for k in keys}

    with torch.no_grad():
        for i, n in enumerate(names):
            c = tf(Image.open(os.path.join(args.content_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(os.path.join(args.style_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            cn, sn = to_np(c), to_np(s)
            ab_c, ab_s = mean_ab(cn), mean_ab(sn)
            d = ab_s - ab_c
            denom = float(d @ d)
            e_c = highpass_energy(cn)
            if denom < 1e-6 or e_c <= 0:
                continue
            for knob in ("low", "high"):
                for j, a in enumerate(alphas):
                    al, ah = (a, 1.0) if knob == "low" else (1.0, a)
                    o = to_np(model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1))
                    acc[f"colour_travel__alpha_{knob}"][j].append(float((mean_ab(o) - ab_c) @ d / denom))
                    acc[f"stroke_energy__alpha_{knob}"][j].append(highpass_energy(o) / e_c)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(names)} pairs", flush=True)

    curves, span, span_usable = {}, {}, {}
    for k, cols in acc.items():
        mean = [float(np.mean(col)) for col in cols]
        sem = [float(np.std(col, ddof=1) / np.sqrt(len(col))) for col in cols]
        curves[k] = {"mean": mean, "sem": sem}
        span[k] = float(max(mean) - min(mean))
        sub = [mean[i] for i in usable]
        span_usable[k] = float(max(sub) - min(sub))

    def ratio(a, b, d):
        return float(d[a] / max(d[b], 1e-12))

    rec = {"checkpoint": args.ckpt, "alphas": alphas, "usable_range": [lo, hi],
           "pairs": len(acc["colour_travel__alpha_low"][0]),
           "curves": curves, "span": span, "span_usable": span_usable,
           "colour_selectivity_full": ratio("colour_travel__alpha_low", "colour_travel__alpha_high", span),
           "stroke_selectivity_full": ratio("stroke_energy__alpha_high", "stroke_energy__alpha_low", span),
           "colour_selectivity_usable": ratio("colour_travel__alpha_low", "colour_travel__alpha_high", span_usable),
           "stroke_selectivity_usable": ratio("stroke_energy__alpha_high", "stroke_energy__alpha_low", span_usable)}

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    for k in keys:
        print(f"{k:34s} full span {span[k]:.4f}   usable span {span_usable[k]:.4f}")
    print(f"colour selectivity: full {rec['colour_selectivity_full']:.2f}, "
          f"usable {rec['colour_selectivity_usable']:.2f}")
    print(f"stroke selectivity: full {rec['stroke_selectivity_full']:.2f}, "
          f"usable {rec['stroke_selectivity_usable']:.2f}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
