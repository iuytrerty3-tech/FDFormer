"""Native-resolution crops from the 2048x2048 outputs, for inspecting brushwork.

The pairs are the first, middle and last of the 100 scored pairs and the crop is the image
centre, both fixed in advance, so the figure involves no choice of favourable examples.
Each panel is drawn pixel for pixel, with no resampling in the saved PDF.

Usage
-----
python make_fig_highres.py --root work/highres \
    --out figures/fig_highres.pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CROP, SIZE = 512, 2048
COLS = ["Style", "Content (enlarged)", r"FDFormer ($\alpha=1$)", "MicroAST"]


def centre(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    o = (SIZE - CROP) // 2
    return np.asarray(img.crop((o, o, o + CROP, o + CROP)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = sorted(os.listdir(os.path.join(args.root, "content")))
    out_dir = os.path.join(args.root, f"out_{SIZE}")
    fd = sorted(os.listdir(os.path.join(out_dir, "fdformer", "stylized")))
    rows = (0, len(names) // 2, len(names) - 1)
    fig, axes = plt.subplots(len(rows), 4, figsize=(8, 6.1))
    for r, i in enumerate(rows):
        n = names[i]
        style = Image.open(os.path.join(args.root, "style", n)).convert("RGB").resize((CROP, CROP), Image.BICUBIC)
        panels = [np.asarray(style), centre(os.path.join(args.root, "content", n)),
                  centre(os.path.join(out_dir, "fdformer", "stylized", fd[i])),
                  centre(os.path.join(out_dir, "microast", "stylized", n))]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(COLS[c], fontsize=9)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.005, wspace=0.015, hspace=0.015)
    fig.savefig(args.out, dpi=300)
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "KB")


if __name__ == "__main__":
    main()
