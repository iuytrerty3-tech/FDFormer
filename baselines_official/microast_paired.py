"""MicroAST (Wang et al., AAAI 2023) - REAL pretrained inference, paired & same-named.

Runs the official MicroAST released weights (EndyWon/MicroAST, models/*_iter_160000.pth.tar)
over our canonical paired test set, writing outputs whose filenames MATCH the content
files, which is what eval_metrics.py and the art-fid package require.

Place this file inside a clone of https://github.com/EndyWon/MicroAST so that
net_microAST.py is importable, then:

  python microast_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/microast/stylized --size 256

Add --bench --eff_out /ABS/FDFormer/results/eff_microast.json to measure
params / latency / max-res on THIS box with the same protocol as bench_efficiency.py.
"""
import argparse
import os
import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net_microAST as net

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
    ap.add_argument("--content_encoder", default="models/content_encoder_iter_160000.pth.tar")
    ap.add_argument("--style_encoder", default="models/style_encoder_iter_160000.pth.tar")
    ap.add_argument("--modulator", default="models/modulator_iter_160000.pth.tar")
    ap.add_argument("--decoder", default="models/decoder_iter_160000.pth.tar")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eff_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    content_encoder = net.Encoder().eval()
    style_encoder = net.Encoder().eval()
    modulator = net.Modulator().eval()
    decoder = net.Decoder().eval()
    content_encoder.load_state_dict(torch.load(args.content_encoder, map_location="cpu"))
    style_encoder.load_state_dict(torch.load(args.style_encoder, map_location="cpu"))
    modulator.load_state_dict(torch.load(args.modulator, map_location="cpu"))
    decoder.load_state_dict(torch.load(args.decoder, map_location="cpu"))

    network = net.TestNet(content_encoder, style_encoder, modulator, decoder).to(device).eval()

    def fwd(content, style):
        return network(content, style, args.alpha)

    if args.bench:
        rec = {"params_M": round(count_params_M([network]), 3), "flops_G": None,
               "latency_ms": round(bench_latency(fwd, device, args.size), 2),
               "max_res": bench_max_res(fwd, device), "size": args.size,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        print("MicroAST efficiency:", rec)
        if args.eff_out:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(args.eff_out)), exist_ok=True)
            json.dump(rec, open(args.eff_out, "w"), indent=2)
            print("wrote", args.eff_out)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    done = 0
    for name in list_images(args.content_dir):
        sp = os.path.join(args.style_dir, name)
        if not os.path.exists(sp):
            print(f"[skip] no matching style for {name}")
            continue
        c = tf(Image.open(os.path.join(args.content_dir, name)).convert("RGB")).unsqueeze(0).to(device)
        s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = fwd(c, s).clamp(0, 1).cpu()
        save_image(out, os.path.join(args.out_dir, name))
        done += 1
    print(f"MicroAST: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
