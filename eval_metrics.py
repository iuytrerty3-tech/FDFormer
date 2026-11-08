"""Quantitative evaluation on a test set.

Primary metric: ArtFID (Wright & Ommer, GCPR 2022), computed with the official
`art-fid` package so that numbers are comparable across methods. Content LPIPS and a
style mean/std distance are reported as well.

Folders must contain matching, same-named files:
    --stylized_dir   stylised outputs
    --content_dir    the content inputs
    --style_dir      the style inputs

Every method, FDFormer and the baselines alike, is scored by this script.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.vgg import VGGEncoder
from losses.losses import calc_mean_std

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _names(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def _load(path, size, device):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return t(img).unsqueeze(0).to(device)


def compute_artfid(stylized_dir, style_dir, content_dir):
    """Compute ArtFID by calling the art_fid Python API directly.

    Uses num_workers=1 to avoid Windows multiprocessing issues that break
    the CLI subprocess approach.  Mode is art_fid_inf (default).
    """
    try:
        from art_fid.art_fid import compute_art_fid
        device = "cuda" if torch.cuda.is_available() else "cpu"
        val = compute_art_fid(
            path_to_stylized=stylized_dir,
            path_to_style=style_dir,
            path_to_content=content_dir,
            batch_size=8,
            device=device,
            mode="art_fid_inf",
            content_metric="lpips",
            num_workers=1,
        )
        print(f"[ArtFID] computed: {val:.4f}")
        return float(val)
    except Exception as e:
        print(f"[ArtFID] API call failed ({e}); falling back to CLI.")
        try:
            import subprocess, sys, re
            result = subprocess.run(
                [sys.executable, "-m", "art_fid",
                 "--style_images", style_dir,
                 "--content_images", content_dir,
                 "--stylized_images", stylized_dir,
                 "--device", "cuda" if torch.cuda.is_available() else "cpu",
                 "--batch_size", "8"],
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace"
            )
            out = result.stdout + result.stderr
            m = re.search(r"ArtFID[:\s]*([\d.]+)", out)
            if m:
                val = float(m.group(1))
                print(f"[ArtFID] computed (CLI): {val:.4f}")
                return val
            print(f"[ArtFID] could not parse CLI output:\n{out[:500]}")
            return None
        except Exception as e2:
            print(f"[ArtFID] CLI also failed ({e2}).")
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stylized_dir", required=True)
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default=None,
                    help="if set, also write {images, artfid, lpips_c, style_dist} as JSON here")
    args = ap.parse_args()
    device = torch.device(args.device)

    artfid = compute_artfid(args.stylized_dir, args.style_dir, args.content_dir)

    # Auxiliary metrics computed here so logs are self-contained.
    encoder = VGGEncoder(pretrained=True).to(device).eval()
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
    except Exception as e:  # noqa: BLE001
        lpips_fn = None
        print(f"[LPIPS] lpips package unavailable ({e}); skipping content-LPIPS.")

    s_names = _names(args.stylized_dir)
    lp_sum, sty_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for name in s_names:
            cp = os.path.join(args.content_dir, name)
            sp = os.path.join(args.style_dir, name)
            if not (os.path.exists(cp) and os.path.exists(sp)):
                continue
            out = _load(os.path.join(args.stylized_dir, name), args.size, device)
            content = _load(cp, args.size, device)
            style = _load(sp, args.size, device)
            if lpips_fn is not None:
                lp_sum += float(lpips_fn(out * 2 - 1, content * 2 - 1).mean())
            of, sf = encoder(out), encoder(style)
            sl = 0.0
            for o, s in zip(of, sf):
                om, osd = calc_mean_std(o); sm, ssd = calc_mean_std(s)
                sl += float(F.mse_loss(om, sm) + F.mse_loss(osd, ssd))
            sty_sum += sl
            n += 1

    print("\n==== Evaluation ====")
    print(f"images evaluated     : {n}")
    print(f"ArtFID (lower better): {artfid if artfid is not None else 'N/A'}")
    print(f"content LPIPS (lower): {lp_sum / n if (lpips_fn and n) else 'N/A'}")
    print(f"style loss   (lower) : {sty_sum / n if n else 'N/A'}")

    if args.json_out:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        rec = {"images": n,
               "artfid": artfid,
               "lpips_c": (lp_sum / n) if (lpips_fn and n) else None,
               "style_dist": (sty_sum / n) if n else None}
        with open(args.json_out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote metrics JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
