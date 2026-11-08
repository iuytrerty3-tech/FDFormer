"""Does the decomposition behave the same way on every kind of painting?

The aggregate figures say what the two branches do on average over the benchmark.
They cannot say whether the behaviour holds on a pointillist canvas, where colour
varies on the scale of the brushwork, as well as on a colour-field painting, where it
does not. This script repeats the two measurements that matter, grouped by the art
movement each style image belongs to, using the labels recovered by
label_style_movements.py.

For each movement it reports:

  colour_without_low   how far the chromatic distance to the style rises when the
                       low branch is switched off, relative to leaving both on;
                       a large value means the low branch was carrying the colour
  selectivity_chroma   how much more the colour coefficient moves the chromatic
                       distance than the stroke coefficient does
  selectivity_texture  the same ratio for texture, with the roles exchanged

A premise that held only on the easy movements would show up here as a collapse of
these numbers on the hard ones.

Usage
-----
python analyze_by_movement.py --min_images 25 --per_movement 40 \
    --json_out results/movement_analysis.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from analyze_frequency import _d_chroma, _d_texture, chroma_stats, texture_gram
from models import FDFormer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--movements", default="results/style_movements.json")
    ap.add_argument("--alphas", default="0.5,1.0,1.5")
    ap.add_argument("--min_images", type=int, default=25)
    ap.add_argument("--per_movement", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/movement_analysis.json")
    args = ap.parse_args()

    with open(args.movements) as f:
        labels = json.load(f)["by_pair_name"]
    groups = defaultdict(list)
    for name, mv in labels.items():
        groups[mv].append(name)
    groups = {m: sorted(v)[:args.per_movement] for m, v in groups.items()
              if len(v) >= args.min_images}
    print(f"{len(groups)} movements with at least {args.min_images} images")

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
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    out = {}

    with torch.no_grad():
        for mv, names in sorted(groups.items()):
            rows = defaultdict(list)
            for n in names:
                cp = os.path.join(args.content_dir, n)
                sp = os.path.join(args.style_dir, n)
                if not (os.path.exists(cp) and os.path.exists(sp)):
                    continue
                c = tf(Image.open(cp).convert("RGB")).unsqueeze(0).to(device)
                s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
                s_ch, s_tx = chroma_stats(s), texture_gram(model.encoder, s)
                ref_ch = _d_chroma(chroma_stats(c), s_ch)
                ref_tx = _d_texture(texture_gram(model.encoder, c), s_tx)
                if ref_ch <= 0 or ref_tx <= 0:
                    continue

                def score(al, ah):
                    o = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                    return (_d_chroma(chroma_stats(o), s_ch) / ref_ch,
                            _d_texture(texture_gram(model.encoder, o), s_tx) / ref_tx)

                both_ch, both_tx = score(1.0, 1.0)
                no_low_ch, no_low_tx = score(0.0, 1.0)
                no_high_ch, no_high_tx = score(1.0, 0.0)
                rows["colour_without_low"].append(no_low_ch - both_ch)
                rows["colour_without_high"].append(no_high_ch - both_ch)
                rows["texture_without_low"].append(no_low_tx - both_tx)
                rows["texture_without_high"].append(no_high_tx - both_tx)

                sweeps = {}
                for knob in ("low", "high"):
                    ch, tx = [], []
                    for a in alphas:
                        al, ah = (a, 1.0) if knob == "low" else (1.0, a)
                        x, y = score(al, ah)
                        ch.append(x); tx.append(y)
                    sweeps[knob] = (max(ch) - min(ch), max(tx) - min(tx))
                rows["chroma_span_low"].append(sweeps["low"][0])
                rows["chroma_span_high"].append(sweeps["high"][0])
                rows["texture_span_low"].append(sweeps["low"][1])
                rows["texture_span_high"].append(sweeps["high"][1])

            rec = {"images": len(rows["colour_without_low"])}
            for k, v in rows.items():
                v = np.asarray(v, dtype=float)
                rec[f"{k}_mean"] = float(v.mean())
                rec[f"{k}_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
            rec["selectivity_chroma"] = rec["chroma_span_low_mean"] / max(rec["chroma_span_high_mean"], 1e-12)
            rec["selectivity_texture"] = rec["texture_span_high_mean"] / max(rec["texture_span_low_mean"], 1e-12)
            out[mv] = rec
            print(f"  {mv:30s} n={rec['images']:3d}  "
                  f"colour lost without low branch {rec['colour_without_low_mean']:+.3f}  "
                  f"chroma selectivity {rec['selectivity_chroma']:6.2f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"alphas": alphas, "movements": out}, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
