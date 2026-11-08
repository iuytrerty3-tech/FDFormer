"""Pick content-style pairs that span the cases a controllability figure should cover.

A grid drawn for one pair says nothing about whether the two coefficients behave the
same way elsewhere. The cases are therefore chosen by measurable properties of the
images rather than by how the stylisations turn out, so the selection cannot be read
as a search for flattering examples.

Content axis
    portrait   a frontal face is detected in the photograph
    landscape  no face, low edge density, and a horizon-like split in the vertical
               luminance profile

Style axis
    textured   the painting sits in the top decile of high-frequency energy in
               luminance, which is what heavy impasto and visible brushwork produce
    overlapping  the painting sits in the top decile of scale overlap, defined as the
               correlation between the local high-pass energy of chroma and of
               luminance: where colour varies on the same spatial scale as texture,
               the premise behind the decomposition is hardest to satisfy, as in
               pointillist and mosaic work

Every score is written out so the choice can be checked.

Usage
-----
python pick_control_cases.py --content_dir eval/default/content \
    --style_dir eval/default/style --json_out results/control_cases.json
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_CASCADE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


def srgb_to_lab(img):
    m = img > 0.04045
    lin = np.where(m, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ mat.T
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def highpass(x, k=5):
    low = cv2.GaussianBlur(x.astype(np.float32), (k, k), 0)
    return x - low


def local_energy(x, win=9):
    return cv2.blur(highpass(x) ** 2, (win, win))


def score_content(path, size=256):
    img = np.asarray(Image.open(path).convert("RGB").resize((size, size)))
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = cv2.CascadeClassifier(_CASCADE).detectMultiScale(grey, 1.1, 5,
                                                             minSize=(32, 32))
    edges = cv2.Canny(grey, 80, 160)
    edge_density = float(edges.mean() / 255.0)
    rows = grey.astype(np.float32).mean(axis=1)
    # A landscape tends to separate into a bright upper band and a darker lower one;
    # the largest step in the smoothed row profile measures how cleanly it does.
    sm = np.convolve(rows, np.ones(9) / 9, mode="valid")
    horizon = float(np.abs(np.diff(sm)).max() / (rows.std() + 1e-6))
    return {"faces": int(len(faces)), "edge_density": edge_density,
            "horizon_step": horizon}


def score_style(path, size=256):
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size))).astype(np.float64) / 255.0
    lab = srgb_to_lab(arr)
    L = lab[..., 0].astype(np.float32)
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2).astype(np.float32)
    hf_luma = float((highpass(L) ** 2).mean() / (L.var() + 1e-6))
    eL, eC = local_energy(L).ravel(), local_energy(C).ravel()
    eL, eC = np.log(eL + 1e-6), np.log(eC + 1e-6)
    overlap = float(np.corrcoef(eL, eC)[0, 1])
    return {"hf_energy_luma": hf_luma, "scale_overlap": overlap,
            "chroma_std": float(C.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--top", type=int, default=6, help="candidates listed per category")
    ap.add_argument("--json_out", default="results/control_cases.json")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.content_dir) if f.lower().endswith(_EXTS))
    rows = []
    for i, n in enumerate(names):
        c = score_content(os.path.join(args.content_dir, n), args.size)
        s = score_style(os.path.join(args.style_dir, n), args.size)
        rows.append({"name": n, **{f"c_{k}": v for k, v in c.items()},
                     **{f"s_{k}": v for k, v in s.items()}})
        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(names)}", flush=True)

    hf = np.array([r["s_hf_energy_luma"] for r in rows])
    ov = np.array([r["s_scale_overlap"] for r in rows])
    ed = np.array([r["c_edge_density"] for r in rows])
    hz = np.array([r["c_horizon_step"] for r in rows])
    hf_hi, ov_hi = np.percentile(hf, 90), np.percentile(ov, 90)
    ed_lo = np.percentile(ed, 25)

    def top(pred, key, reverse=True):
        sel = [r for r in rows if pred(r)]
        sel.sort(key=lambda r: r[key], reverse=reverse)
        return sel[:args.top]

    cats = {
        "portrait": top(lambda r: r["c_faces"] >= 1, "s_chroma_std"),
        "landscape": top(lambda r: r["c_faces"] == 0 and r["c_edge_density"] <= ed_lo,
                         "c_horizon_step"),
        "textured_style": top(lambda r: r["s_hf_energy_luma"] >= hf_hi, "s_hf_energy_luma"),
        "overlapping_style": top(lambda r: r["s_scale_overlap"] >= ov_hi, "s_scale_overlap"),
    }
    for k, v in cats.items():
        print(f"\n{k}:")
        for r in v:
            print(f"   {r['name']}  faces={r['c_faces']} edge={r['c_edge_density']:.3f} "
                  f"horizon={r['c_horizon_step']:.2f} hf={r['s_hf_energy_luma']:.4f} "
                  f"overlap={r['s_scale_overlap']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"thresholds": {"hf_p90": float(hf_hi), "overlap_p90": float(ov_hi),
                                  "edge_p25": float(ed_lo)},
                   "candidates": {k: [r["name"] for r in v] for k, v in cats.items()},
                   "scores": rows}, f, indent=2)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
