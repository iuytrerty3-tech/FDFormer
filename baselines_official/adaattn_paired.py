"""AdaAttN (Liu et al., ICCV 2021): inference with the released weights, on paired inputs.

Copy this file into a clone of the official repository (Huage001/AdaAttN) and run it there.
It builds the official AdaAttN model through the repo's own option/model framework,
loads the released checkpoints (checkpoints/AdaAttN/*.pth), and saves each stylised
image through the repo's own util.util.tensor2im so output is byte-faithful to the
official pipeline. We only change the I/O: paired loop content[i]<->style[i] and
output filename == content filename.

Pretrained weights, as released by the authors:
  checkpoints/vgg_normalised.pth        (image encoder)
  checkpoints/AdaAttN/  (latest_net_decoder.pth, latest_net_transformer.pth, latest_net_adaattn_3.pth)

Run inside the repo:
  python adaattn_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/adaattn/stylized \
      --image_encoder_path checkpoints/vgg_normalised.pth --size 256
Add --bench --eff_out /ABS/FDFormer/results/eff_adaattn.json for the Table-3 row.
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image
from torchvision import transforms

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_params_M(modules):
    return sum(sum(p.numel() for p in m.parameters()) for m in modules) / 1e6


def bench_latency(fwd, device, size, warmup=10, runs=50):
    c = torch.rand(1, 3, size, size, device=device)
    s = torch.rand(1, 3, size, size, device=device)
    cu = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fwd(c, s)
        if cu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fwd(c, s)
        if cu:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0


def bench_max_res(fwd, device):
    if device.type != "cuda":
        return None
    best = None
    for size in (256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048):
        try:
            torch.cuda.empty_cache()
            c = torch.rand(1, 3, size, size, device=device)
            s = torch.rand(1, 3, size, size, device=device)
            with torch.no_grad():
                fwd(c, s)
            torch.cuda.synchronize()
            best = size
            del c, s
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    return f"{best}x{best}" if best else None


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_encoder_path", default="checkpoints/vgg_normalised.pth")
    ap.add_argument("--name", default="AdaAttN")
    ap.add_argument("--checkpoints_dir", default="./checkpoints")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eff_out", default=None)
    args = ap.parse_args()

    # Build the repo's option object by handing its parser the same flags
    # test_adaattn.sh uses (load_size/crop_size set to our --size).
    sys.argv = [
        "test.py",
        "--content_path", args.content_dir,
        "--style_path", args.style_dir,
        "--name", args.name,
        "--model", "adaattn",
        "--dataset_mode", "unaligned",
        "--load_size", str(args.size),
        "--crop_size", str(args.size),
        "--image_encoder_path", args.image_encoder_path,
        "--checkpoints_dir", args.checkpoints_dir,
        "--gpu_ids", args.gpu_id,
        "--skip_connection_3",
        "--shallow_layer",
    ]
    from options.test_options import TestOptions
    from models import create_model
    from util.util import tensor2im

    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    model = create_model(opt)
    model.setup(opt)          # loads pretrained net_decoder / net_transformer / net_adaattn_3
    if opt.eval:
        model.eval()
    device = model.device

    def fwd(c, s):
        model.set_input({"c": c, "s": s, "name": ["bench"]})
        model.test()
        return model.cs

    if args.bench:
        mods = list(model.image_encoder_layers) + [model.net_decoder, model.net_transformer]
        if hasattr(model, "net_adaattn_3"):
            mods.append(model.net_adaattn_3)
        params_M = count_params_M(mods)
        lat = bench_latency(fwd, device, args.size)
        mres = bench_max_res(fwd, device)
        rec = {"params_M": round(params_M, 3), "flops_G": None,
               "latency_ms": round(lat, 2), "max_res": mres, "size": args.size,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        print("AdaAttN efficiency:", rec)
        if args.eff_out:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(args.eff_out)), exist_ok=True)
            json.dump(rec, open(args.eff_out, "w"), indent=2)
            print("wrote", args.eff_out)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    names = list_images(args.content_dir)
    done = 0
    for name in names:
        sp = os.path.join(args.style_dir, name)
        if not os.path.exists(sp):
            print(f"[skip] no matching style for {name}")
            continue
        c = tf(Image.open(os.path.join(args.content_dir, name)).convert("RGB")).unsqueeze(0)
        s = tf(Image.open(sp).convert("RGB")).unsqueeze(0)
        model.set_input({"c": c, "s": s, "name": [name]})
        model.test()
        im = tensor2im(model.get_current_visuals()["cs"])   # official conversion -> uint8 HWC
        Image.fromarray(im).save(os.path.join(args.out_dir, name))
        done += 1
    print(f"AdaAttN: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
