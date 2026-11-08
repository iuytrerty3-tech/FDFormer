"""SANet (Park & Lee, CVPR 2019): inference with the released weights, on paired inputs.

This is the official-architecture SANet with the *released pretrained weights* from
GlebSBrykin/SANET (https://github.com/GlebSBrykin/SANET). The model definition below
is copied verbatim from that repo's Eval.py; we only change the I/O so it runs over
our canonical paired test set and writes outputs whose filenames MATCH the content
files, as eval_metrics.py and ArtFID require.

  --content_dir eval/default/content   --style_dir eval/default/style
  -> writes <out_dir>/<same_name> for every content image, paired content[i]<->style[i].

Weights (download from the repo's release or the Yandex links in its README):
  vgg_normalised.pth  decoder_iter_500000.pth  transformer_iter_500000.pth

Run (no GPU needed for --bench=off, but use the paper's GPU for real numbers):
  python sanet_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/sanet/stylized \
      --vgg vgg_normalised.pth --decoder decoder_iter_500000.pth \
      --transform transformer_iter_500000.pth --size 256

Add --bench --eff_out /ABS/FDFormer/results/eff_sanet.json to also measure
params / latency / max-res on THIS box (same method as bench_efficiency.py).
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------- model (verbatim from SANET/Eval.py) -----------------
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(size)) / std.expand(size)


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
)


class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) +
            self.upsample5_1(self.sanet5_1(content5_1, style5_1))))


# ----------------------------- shared bench helpers --------------------------------
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
    ap.add_argument("--vgg", default="vgg_normalised.pth")
    ap.add_argument("--decoder", default="decoder_iter_500000.pth")
    ap.add_argument("--transform", default="transformer_iter_500000.pth")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eff_out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    decoder.eval(); transform = Transform(in_planes=512).eval(); vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder, map_location="cpu"))
    transform.load_state_dict(torch.load(args.transform, map_location="cpu"))
    vgg.load_state_dict(torch.load(args.vgg, map_location="cpu"))

    enc_1 = nn.Sequential(*list(vgg.children())[:4]).to(device)
    enc_2 = nn.Sequential(*list(vgg.children())[4:11]).to(device)
    enc_3 = nn.Sequential(*list(vgg.children())[11:18]).to(device)
    enc_4 = nn.Sequential(*list(vgg.children())[18:31]).to(device)
    enc_5 = nn.Sequential(*list(vgg.children())[31:44]).to(device)
    transform.to(device); decoder.to(device)

    def fwd(content, style):
        c4 = enc_4(enc_3(enc_2(enc_1(content))))
        c5 = enc_5(c4)
        s4 = enc_4(enc_3(enc_2(enc_1(style))))
        s5 = enc_5(s4)
        return decoder(transform(c4, s4, c5, s5))

    if args.bench:
        params_M = count_params_M([enc_1, enc_2, enc_3, enc_4, enc_5, transform, decoder])
        lat = bench_latency(fwd, device, args.size)
        mres = bench_max_res(fwd, device)
        rec = {"params_M": round(params_M, 3), "flops_G": None,
               "latency_ms": round(lat, 2), "max_res": mres, "size": args.size,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        print("SANet efficiency:", rec)
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
    print(f"SANet: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
