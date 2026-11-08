"""SaMam (Liu et al., CVPR 2025) - REAL pretrained inference, paired & same-named.

Runs the official SaMam released checkpoint (Chernobyllight/SaMam,
SaMam_patchsize_x8.ckpt) over our canonical paired test set, writing outputs whose
filenames MATCH the content files, which is what eval_metrics.py and the art-fid
package require.

The repository offers two execution paths. One uses the Triton-backed mamba-ssm and
causal-conv1d kernels; the other, selected by --mamba-from-trion 0, runs the same
model in plain PyTorch and is the one the authors document for Windows. Only the
second is available on our machine, so it is what we use; it changes the speed of the
scan, not the arithmetic it performs, and the checkpoint is the authors' own. Because
that path is not the one the authors time, we do not report a latency for SaMam.

Images are loaded and saved exactly as the repository's test_image.py does: tensors in
[0, 1] with no normalisation, the style image resized on its shorter edge.

Place this file inside the TEST folder of a clone of
https://github.com/Chernobyllight/SaMam, then:

  python samam_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/samam/stylized \
      --model_ckpt  ./checkpoint/SaMam_patchsize_x8.ckpt --size 256
"""
import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append(os.path.abspath(".."))

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def build_model(args_dict, ckpt_path, device):
    from TRAIN.lightning_module.lightningmodel import LightningModel
    if args_dict["mamba_from_trion"] == 0:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = LightningModel(**args_dict)
        sd = ck["state_dict"]
        for k in list(sd.keys()):
            if k.startswith("loss_func"):
                del sd[k]
            else:
                sd[k[6:]] = sd.pop(k)
        model.model.load_state_dict(sd)
    else:
        model = LightningModel.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                    map_location="cpu")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_ckpt", default="./checkpoint/SaMam_patchsize_x8.ckpt")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model hyper-parameters, defaults copied from the repository's test_image.py
    ap.add_argument("--nVSSMs", type=int, default=2)
    ap.add_argument("--nSAVSSMs", type=int, default=2)
    ap.add_argument("--nSAVSSGs", type=int, default=2)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--representation_dim", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--expand", type=float, default=2.0)
    ap.add_argument("--compress_ratio", type=int, default=8)
    ap.add_argument("--squeeze_factor", type=int, default=8)
    ap.add_argument("--mamba_from_trion", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = {k: v for k, v in vars(args).items()
           if k not in ("content_dir", "style_dir", "out_dir", "model_ckpt", "size", "device")}
    model = build_model(cfg, args.model_ckpt, device)

    os.makedirs(args.out_dir, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    done = 0
    with torch.no_grad():
        for name in list_images(args.content_dir):
            sp = os.path.join(args.style_dir, name)
            if not os.path.exists(sp):
                continue
            if os.path.exists(os.path.join(args.out_dir, name)):
                done += 1
                continue
            c = tf(Image.open(os.path.join(args.content_dir, name)).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
            out = model.forward(c, s)[0].clamp(0, 1).cpu()
            save_image(out, os.path.join(args.out_dir, name))
            done += 1
            if done % 100 == 0:
                print(f"  {done} done", flush=True)
    print(f"SaMam: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
