"""StyTr^2 (Deng et al., CVPR 2022): inference with the released weights, on paired inputs.

Copy this file into a clone of the official repository (diyiiyiii/StyTR-2) and run it there,
because it imports that repo's own modules (models.StyTR, models.transformer, function)
and loads the repo's official pretrained weights. We only change the I/O vs the repo's
test.py: (1) device -> cuda:0 (their test.py hardcodes cuda:2), (2) paired loop
content[i]<->style[i] instead of the full N*M cross product, (3) output filename ==
content filename (what eval_metrics.py / ArtFID require).

Official weights, placed in ./experiments/ of that repository:
  vgg_normalised.pth  decoder_iter_160000.pth  transformer_iter_160000.pth  embedding_iter_160000.pth

Run inside the repo:
  python stytr2_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/stytr2/stylized --size 256
Add --bench --eff_out /ABS/FDFormer/results/eff_stytr2.json for the Table-3 row.
"""
import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.transformer as transformer
import models.StyTR as StyTR

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_into(module, path):
    sd = torch.load(path, map_location="cpu")
    new = OrderedDict((k, v) for k, v in sd.items())
    module.load_state_dict(new)


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
    ap.add_argument("--vgg", default="./experiments/vgg_normalised.pth")
    ap.add_argument("--decoder_path", default="experiments/decoder_iter_160000.pth")
    ap.add_argument("--Trans_path", default="experiments/transformer_iter_160000.pth")
    ap.add_argument("--embedding_path", default="experiments/embedding_iter_160000.pth")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--position_embedding", default="sine", choices=("sine", "learned"))
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eff_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg, map_location="cpu"))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()
    decoder.eval(); Trans.eval(); vgg.eval(); embedding.eval()
    _load_into(decoder, args.decoder_path)
    _load_into(Trans, args.Trans_path)
    _load_into(embedding, args.embedding_path)

    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
    network.eval().to(device)

    def fwd(content, style):
        out = network(content, style)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    if args.bench:
        params_M = count_params_M([network])
        flops_G = None
        try:
            from thop import profile
            c = torch.rand(1, 3, args.size, args.size, device=device)
            s = torch.rand(1, 3, args.size, args.size, device=device)
            macs, _ = profile(network, inputs=(c, s), verbose=False)
            flops_G = round(float(macs) * 2 / 1e9, 2)
            del c, s
        except Exception as e:  # noqa: BLE001
            print(f"[FLOPs] thop failed ({e}); leaving null.")
        lat = bench_latency(fwd, device, args.size)
        mres = bench_max_res(fwd, device)
        rec = {"params_M": round(params_M, 3), "flops_G": flops_G,
               "latency_ms": round(lat, 2), "max_res": mres, "size": args.size,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        print("StyTr2 efficiency:", rec)
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
        c = tf(Image.open(os.path.join(args.content_dir, name)).convert("RGB")).unsqueeze(0).to(device)
        s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = fwd(c, s).clamp(0, 1).cpu()
        save_image(out, os.path.join(args.out_dir, name))
        done += 1
    print(f"StyTr2: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
