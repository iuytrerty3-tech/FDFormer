"""Stylise a single content/style pair with a trained FDFormer.

The two knobs demonstrate the controllability our frequency decoupling buys:
    --alpha_low   colour / global-palette transfer strength
    --alpha_high  brushstroke / texture transfer strength

Example:
    python stylize.py --ckpt experiments/fdformer_default/ckpt_080000.pth \\
        --content c.jpg --style s.jpg --output out.png --alpha_low 1.0 --alpha_high 1.3
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models import FDFormer


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    tf = [transforms.Resize(size)] if size else []
    tf.append(transforms.ToTensor())
    return transforms.Compose(tf)(img).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--output", default="output.png")
    ap.add_argument("--size", type=int, default=None, help="optional resize of short side")
    ap.add_argument("--alpha_low", type=float, default=1.0)
    ap.add_argument("--alpha_high", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    mcfg = cfg.get("model", {"embed_dim": 256, "heads": 4})
    model = FDFormer(embed_dim=mcfg["embed_dim"], heads=mcfg["heads"], pretrained_vgg=True,
                     freq_decouple=mcfg.get("freq_decouple", True),
                     use_gate=mcfg.get("use_gate", True))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    content = load_image(args.content, args.size).to(device)
    style = load_image(args.style, args.size).to(device)
    with torch.no_grad():
        out = model(content, style, alpha_low=args.alpha_low,
                    alpha_high=args.alpha_high).clamp(0, 1)
    save_image(out, args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
