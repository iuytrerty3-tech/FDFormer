"""Batch-stylise a whole test set, ready for eval_metrics.py / ArtFID.

eval_metrics.py and the official art-fid package pair images by *identical
filename* across the stylized, content and style folders. Independent content
and style collections (e.g. COCO ids vs WikiArt names) never share names, so
this script builds a self-consistent evaluation set: it pairs content[i] with
style[i], stylises the pair, and writes three parallel folders whose files all
share the same name:

    <out_dir>/content/00000.png   <out_dir>/style/00000.png   <out_dir>/stylized/00000.png

Then evaluation is one command:

    python eval_metrics.py --stylized_dir <out_dir>/stylized \
        --content_dir <out_dir>/content --style_dir <out_dir>/style

For a fair table, run each baseline on the SAME <out_dir>/content and
<out_dir>/style pairs to produce its own stylized/ folder, then eval each.

Example:
    python batch_stylize.py --ckpt experiments/fdformer_default/ckpt_080000.pth \
        --content_dir datasets/test/content --style_dir datasets/test/style \
        --out_dir eval/fdformer --num 200 --size 256
"""
import argparse
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models import FDFormer

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="output root; content/ style/ stylized/ are created inside")
    ap.add_argument("--num", type=int, default=0, help="cap on number of pairs (0 = all)")
    ap.add_argument("--size", type=int, default=256, help="square size for every saved image")
    ap.add_argument("--alpha_low", type=float, default=1.0)
    ap.add_argument("--alpha_high", type=float, default=1.0)
    ap.add_argument("--gate_scale", type=float, default=1.0,
                    help="scalar on the predicted gate, clamped to [0,1]")
    ap.add_argument("--low_memory", action="store_true",
                    help="use FDFormer.stylize_low_memory; the output is unchanged")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomly pair content/style instead of index order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    mcfg = ckpt.get("cfg", {}).get("model", {"embed_dim": 256, "heads": 4})
    model = FDFormer(embed_dim=mcfg["embed_dim"], heads=mcfg["heads"], pretrained_vgg=True,
                     freq_decouple=mcfg.get("freq_decouple", True),
                     use_gate=mcfg.get("use_gate", True),
                     freq_mode=mcfg.get("freq_mode", "learnable"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if args.low_memory:
        model.forward = model.stylize_low_memory

    contents = list_images(args.content_dir)
    styles = list_images(args.style_dir)
    if not contents or not styles:
        raise SystemExit("content_dir or style_dir has no images.")

    n = min(len(contents), len(styles))
    if args.num:
        n = min(n, args.num)
    contents = contents[:n]
    styles = list(styles)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(styles)
    styles = styles[:n]

    out_c = os.path.join(args.out_dir, "content")
    out_s = os.path.join(args.out_dir, "style")
    out_t = os.path.join(args.out_dir, "stylized")
    for d in (out_c, out_s, out_t):
        os.makedirs(d, exist_ok=True)

    load = transforms.Compose([transforms.Resize((args.size, args.size)),
                               transforms.ToTensor()])

    for i, (cn, sn) in enumerate(tqdm(list(zip(contents, styles)), desc="stylising")):
        name = f"{i:05d}.png"
        c = load(Image.open(os.path.join(args.content_dir, cn)).convert("RGB")).unsqueeze(0).to(device)
        s = load(Image.open(os.path.join(args.style_dir, sn)).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(c, s, alpha_low=args.alpha_low, alpha_high=args.alpha_high,
                        gate_scale=args.gate_scale).clamp(0, 1)
        save_image(c, os.path.join(out_c, name))
        save_image(s, os.path.join(out_s, name))
        save_image(out, os.path.join(out_t, name))

    print(f"Done. {n} pairs written under {args.out_dir} (content/ style/ stylized/).")
    print("Evaluate with:")
    print(f"  python eval_metrics.py --stylized_dir {out_t} "
          f"--content_dir {out_c} --style_dir {out_s} --size {args.size}")


if __name__ == "__main__":
    main()
