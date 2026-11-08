"""Check that samam_paired.py gives the output of the repository's own test_image.py.

Both code paths load the authors' checkpoint through the plain-PyTorch route, then process
the same content and style files. The first path follows test_image.stylize_image and its
test_utils transforms; the second follows the wrapper's loading and resizing. A pair
matches when the two output tensors are identical, and when the PNGs written by each route
decode to the same bytes.

Run from anywhere; the script enters the SaMam TEST folder.

Usage
-----
python verify_samam_wrapper.py --content_dir eval/default/content --style_dir eval/default/style \
    --num 20 --json_out results/samam_wrapper_check.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--samam_test", default="baselines/SaMam/TEST")
    ap.add_argument("--tmp", default="work/samam_check")
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for a in ("content_dir", "style_dir", "json_out"):
        setattr(args, a, os.path.abspath(getattr(args, a)))
    os.makedirs(args.tmp, exist_ok=True)
    os.chdir(args.samam_test)
    sys.path[:0] = [args.samam_test, os.path.abspath("..")]

    import samam_paired
    import test_image
    import test_utils

    cfg = dict(nVSSMs=2, nSAVSSMs=2, nSAVSSGs=2, embed_dim=256, patch_size=8,
               representation_dim=64, d_state=16, expand=2.0, compress_ratio=8,
               squeeze_factor=8, mamba_from_trion=0)
    device = torch.device(args.device)
    model = samam_paired.build_model(cfg, "./checkpoint/SaMam_patchsize_x8.ckpt", device)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    names = sorted(os.listdir(args.content_dir))[:: 1000 // args.num][: args.num]
    rows = []
    with torch.no_grad():
        for n in names:
            cp, sp = os.path.join(args.content_dir, n), os.path.join(args.style_dir, n)
            repo = test_image.stylize_image(model, cp, sp, 256)
            c = tf(Image.open(cp).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
            ours = model.forward(c, s)[0].cpu()
            a, b = os.path.join(args.tmp, "repo.png"), os.path.join(args.tmp, "ours.png")
            save_image(repo.clamp(0, 1), a)
            save_image(ours.clamp(0, 1), b)
            pa, pb = np.asarray(Image.open(a)), np.asarray(Image.open(b))
            rows.append({"pair": n, "tensor_identical": bool(torch.equal(repo, ours)),
                         "max_abs_tensor_diff": float((repo - ours).abs().max()),
                         "png_identical": bool(np.array_equal(pa, pb)),
                         "sha256": hashlib.sha256(pb.tobytes()).hexdigest()[:16]})
            print(rows[-1], flush=True)
    rec = {"pairs": len(rows), "device": args.device,
           "all_tensor_identical": all(r["tensor_identical"] for r in rows),
           "all_png_identical": all(r["png_identical"] for r in rows), "rows": rows}
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print({k: rec[k] for k in ("pairs", "all_tensor_identical", "all_png_identical")})


if __name__ == "__main__":
    main()
