"""AesPA-Net (Hong et al., ICCV 2023) - REAL pretrained inference, paired & same-named.

Runs the official AesPA-Net released weights (Kibeom-Hong/AesPA-Net: dec_model.pth and
transformer_model.pth) over our canonical paired test set, writing outputs whose
filenames MATCH the content files, which is what eval_metrics.py and the art-fid
package require.

The forward pass reproduces the repository's own Baseline.test() exactly: images pass
through size_arrange, the aesthetic pattern-aware coefficient is obtained from
adaptive_gram_weight over levels 1-3 of the colour and greyscale style image, and the
network is called as network(content, style, alpha, gray_content, style).

Encoder weights. The repository loads the normalised VGG-19 from a Torch7 file,
vgg_normalised_conv5_1.t7, whose Google Drive link is restricted to the repository owner
and cannot be downloaded. We therefore feed the encoder the SAME published normalised
VGG-19 weights from the PyTorch file vgg_normalised.pth that SANet, AdaAttN and StyTr2
use, through a shim exposing them under the Torch7 module indices the encoder reads
(0, 2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42). Every tensor shape is checked
against the layer it is assigned to, so a mismatch fails loudly rather than silently
producing wrong outputs.

Place this file inside a clone of https://github.com/Kibeom-Hong/AesPA-Net, then:

  python aespanet_paired.py \
      --content_dir /ABS/FDFormer/eval/default/content \
      --style_dir   /ABS/FDFormer/eval/default/style \
      --out_dir     /ABS/FDFormer/eval/aespanet/stylized \
      --vgg /ABS/vgg_normalised.pth --size 256

Add --bench --eff_out /ABS/FDFormer/results/eff_aespanet.json to measure
params / latency / max-res on THIS box with the same protocol as bench_efficiency.py.
"""
import argparse
import os
import sys
import time
import types

import torch

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Torch7 module indices the repository's VGGEncoder reads, and the conv shape it
# assigns each one to. Checked at load time so a wrong file cannot pass silently.
_T7_CONV_SHAPES = {
    0: (3, 3), 2: (64, 3), 5: (64, 64), 9: (128, 64), 12: (128, 128),
    16: (256, 128), 19: (256, 256), 22: (256, 256), 25: (256, 256),
    29: (512, 256), 32: (512, 512), 35: (512, 512), 38: (512, 512), 42: (512, 512),
}


class _T7Module:
    """One Torch7 nn.SpatialConvolution as the encoder expects to see it."""

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias


class _T7Model:
    def __init__(self, modules):
        self.modules = modules


def build_vgg_shim(vgg_pth):
    """Expose PyTorch vgg_normalised.pth under the Torch7 indices the encoder uses."""
    sd = torch.load(vgg_pth, map_location="cpu")
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    modules = [_T7Module(None, None) for _ in range(max(_T7_CONV_SHAPES) + 1)]
    for i, (cout, cin) in _T7_CONV_SHAPES.items():
        wk, bk = f"{i}.weight", f"{i}.bias"
        if wk not in sd or bk not in sd:
            raise SystemExit(f"{vgg_pth}: no convolution at index {i}; this is not the "
                             f"normalised VGG-19 the encoder expects.")
        w = sd[wk]
        if tuple(w.shape[:2]) != (cout, cin):
            raise SystemExit(f"{vgg_pth}: index {i} has shape {tuple(w.shape)}, expected "
                             f"({cout}, {cin}, k, k).")
        modules[i] = _T7Module(w.numpy(), sd[bk].numpy())
    return _T7Model(modules)


class _Stub(types.ModuleType):
    """Absorbs any attribute access, so an unused import cannot stop inference."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _Stub(f"{self.__name__}.{name}")
        sys.modules[sub.__name__] = sub
        return sub

    def __call__(self, *a, **k):
        return None


def _stub_missing_modules():
    """Stub repository imports that play no part in inference.

    The repository pulls in experiment-tracking and video helpers at import time
    (wandb, skvideo) and reads its encoder through torchfile. None of them affects
    the stylisation forward pass, so a missing one is replaced by an inert module
    rather than being installed.
    """
    for name in ("wandb", "skvideo", "skvideo.io", "torchfile"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _Stub(name)


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vgg", required=True, help="path to vgg_normalised.pth")
    ap.add_argument("--decoder", default="baseline_checkpoints/dec_model.pth")
    ap.add_argument("--transformer", default="baseline_checkpoints/transformer_model.pth")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--eff_out", default=None)
    args = ap.parse_args()

    import random

    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _stub_missing_modules()
    import torchfile
    shim = build_vgg_shim(args.vgg)
    torchfile.load = lambda *a, **k: shim          # the encoder now reads these weights

    from baseline import Baseline, size_arrange    # noqa: E402  (after the patch)
    import torchvision
    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image

    device = torch.device(args.device)
    cfg = argparse.Namespace(imsize=512, batch_size=1, cencrop=False, cropsize=256,
                             num_workers=0, content_dir=args.content_dir,
                             style_dir=args.style_dir, lr=1e-4,
                             train_result_dir="./train_results", comment="paired",
                             max_iter=1, check_iter=1)
    model = Baseline(cfg)
    model.network.decoder.load_state_dict(
        torch.load(args.decoder, map_location="cpu")["state_dict"])
    model.network.transformer.load_state_dict(
        torch.load(args.transformer, map_location="cpu")["state_dict"])
    model.network.eval()
    model.network.train(False)

    # The repository normalises every input by the ImageNet statistics its encoder was
    # trained under and denormalises the stylisation before writing it, both inside its
    # dataset and imsave helpers. Feeding raw [0, 1] tensors instead would hand the
    # model inputs it never saw, so we apply the same pair of transforms here.
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def normalise(x):
        return (x - _MEAN) / _STD

    def denormalise(x):
        return x * _STD + _MEAN

    def fwd(content, style):
        content = size_arrange(normalise(content))
        style = size_arrange(normalise(style))
        gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1, 3, 1, 1)
        gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1, 3, 1, 1)
        alpha = ((sum(model.adaptive_gram_weight(style, lv, 8) for lv in (1, 2, 3)) / 3
                  ).unsqueeze(1).to(device)
                 + (sum(model.adaptive_gram_weight(gray_style, lv, 8) for lv in (1, 2, 3)) / 3
                    ).unsqueeze(1).to(device)) / 2
        return denormalise(model.network(content, style, alpha, gray_content, style)[0])

    if args.bench:
        params_M = sum(p.numel() for p in model.network.parameters()) / 1e6
        rec = {"params_M": round(params_M, 3), "flops_G": None,
               "latency_ms": round(bench_latency(fwd, device, args.size), 2),
               "max_res": bench_max_res(fwd, device), "size": args.size,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        print("AesPA-Net efficiency:", rec)
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
        if os.path.exists(os.path.join(args.out_dir, name)):
            done += 1
            continue
        c = tf(Image.open(os.path.join(args.content_dir, name)).convert("RGB")).unsqueeze(0).to(device)
        s = tf(Image.open(sp).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = fwd(c, s).clamp(0, 1).cpu()
        save_image(out, os.path.join(args.out_dir, name))
        done += 1
        if done % 100 == 0:
            print(f"  {done} done", flush=True)
            # Releasing cached blocks synchronises the device, so do it rarely; every
            # image would serialise this process against anything else on the GPU.
            torch.cuda.empty_cache()
    print(f"AesPA-Net: wrote {done} stylised images to {args.out_dir}")


if __name__ == "__main__":
    main()
