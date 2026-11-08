"""Evidence for the frequency assumption behind FDFormer.

The method rests on one premise: in a painting, the low-frequency band of a deep
feature carries mainly the palette and the high-frequency band mainly the brushwork.
This script tests that premise three ways and writes every number it measures to
JSON so the paper can quote them.

A. Response of the learned low-pass.
   Each of the 512 depthwise kernels is zero-padded, transformed, and radially
   averaged, giving a per-channel response H(f) on the relu4_1 grid. We report the
   -3 dB cut-off of the learned kernels and of the Gaussian they started from, so
   the amount the decomposition adapted to the data is visible rather than assumed.

B. Where colour and texture energy actually sit, in real paintings.
   Style images are converted to CIE Lab. The radially averaged power spectrum is
   computed for the chroma channels (a, b), which carry palette and nothing else,
   and for luminance (L), which carries brushwork. We report the share of spectral
   energy each places above the cut-off measured in A.

C. Band attribution, by running the trained model.
   For each pair we decode three outputs from the same weights: both branches on,
   only the colour branch (alpha_high = 0), and only the stroke branch
   (alpha_low = 0). Each output is scored with

     - a purely chromatic distance: mean and standard deviation of the a, b channels
     - a purely textural distance: Gram matrices of the high-pass of relu3_1/relu4_1

   and each distance is normalised by the same distance between content and style,
   so 0 means the output matches the style and 1 means it is unchanged from the
   content. If the premise holds, the colour branch alone should move the chromatic
   distance and leave the textural one, and the stroke branch alone the reverse.

Usage
-----
python analyze_frequency.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --content_dir eval/default/content --style_dir eval/default/style \
    --num 200 --json_out results/freq_analysis.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from losses.losses import gram, high_freq
from models import FDFormer
from models.freq import _gaussian_kernel2d

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- A
def radial_profile(power: np.ndarray) -> np.ndarray:
    """Average a centred 2-D spectrum over rings of equal radius."""
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rbin = r.astype(int)
    nbin = min(h, w) // 2
    out = np.zeros(nbin)
    for i in range(nbin):
        m = rbin == i
        out[i] = power[m].mean() if m.any() else np.nan
    return out


def kernel_response(kernel: np.ndarray, pad: int = 64) -> np.ndarray:
    """Radially averaged magnitude response of one depthwise kernel, normalised to H(0)=1."""
    k = np.zeros((pad, pad), dtype=np.float64)
    ks = kernel.shape[-1]
    off = (pad - ks) // 2
    k[off:off + ks, off:off + ks] = kernel
    mag = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k))))
    prof = radial_profile(mag)
    return prof / prof[0]


def cutoff_frequency(profile: np.ndarray, drop_db: float = 3.0) -> float:
    """First frequency at which the response falls below -drop_db, in cycles/pixel.

    Bin i of the profile corresponds to radial frequency i / pad cycles per pixel.
    """
    thresh = 10.0 ** (-drop_db / 20.0)
    below = np.where(profile < thresh)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    # linear interpolation between the two bracketing bins
    p0, p1 = profile[i - 1], profile[i]
    frac = (p0 - thresh) / (p0 - p1) if p0 != p1 else 0.0
    return float((i - 1 + frac) / (2 * len(profile)))


def analyse_kernels(model, drop_db=3.0):
    if getattr(model, "freq", None) is None or model.freq.weight is None:
        # The Haar split has no kernel to transform: its response is fixed by the
        # transform itself. Report that rather than failing, so this analysis can be
        # run over every decomposition variant without special-casing the call site.
        return {"kernel": None,
                "note": "this decomposition has no learnable low-pass kernel"}
    w = model.freq.weight.detach().cpu().numpy()[:, 0]          # [C, k, k]
    gauss = _gaussian_kernel2d(model.freq.ksize, 1.0).numpy()
    learned_prof = np.stack([kernel_response(k) for k in w])
    gauss_prof = kernel_response(gauss)
    learned_cut = np.array([cutoff_frequency(p, drop_db) for p in learned_prof])
    return {
        "channels": int(w.shape[0]),
        "kernel_size": int(model.freq.ksize),
        "cutoff_db": drop_db,
        "gaussian_init_cutoff_cyc_per_px": cutoff_frequency(gauss_prof, drop_db),
        "learned_cutoff_mean_cyc_per_px": float(np.nanmean(learned_cut)),
        "learned_cutoff_std_cyc_per_px": float(np.nanstd(learned_cut)),
        "learned_cutoff_p05_cyc_per_px": float(np.nanpercentile(learned_cut, 5)),
        "learned_cutoff_p95_cyc_per_px": float(np.nanpercentile(learned_cut, 95)),
        "kernel_l1_drift_from_gaussian": float(np.abs(w - gauss[None]).sum(axis=(1, 2)).mean()),
        "mean_learned_profile": learned_prof.mean(axis=0).tolist(),
        "gaussian_profile": gauss_prof.tolist(),
    }


# --------------------------------------------------------------------------- B
def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    """sRGB in [0,1], HxWx3 -> CIE Lab (D65)."""
    m = img > 0.04045
    lin = np.where(m, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    mat = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ mat.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def radial_power(channel: np.ndarray) -> np.ndarray:
    """Radially averaged power spectrum of one channel, windowed and mean-removed."""
    x = channel - channel.mean()
    x = x * np.hanning(x.shape[0])[:, None] * np.hanning(x.shape[1])[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    return radial_profile(power)


def spectral_slope(prof: np.ndarray, lo=2, hi=None) -> float:
    """Exponent of a 1/f**alpha fit to the radial power spectrum.

    A larger exponent means energy falls off faster with frequency, that is, the
    channel is more concentrated in the low band.
    """
    hi = hi or len(prof)
    f = np.arange(lo, hi, dtype=float)
    p = prof[lo:hi]
    m = np.isfinite(p) & (p > 0)
    if m.sum() < 5:
        return float("nan")
    coef = np.polyfit(np.log(f[m]), np.log(p[m]), 1)
    return float(-coef[0])


def analyse_style_spectra(style_dir, names, size):
    """Compare how fast chroma and luminance energy fall off, in real paintings."""
    slopes = {"chroma": [], "luma": []}
    profiles = {"chroma": [], "luma": []}
    for n in names:
        img = Image.open(os.path.join(style_dir, n)).convert("RGB").resize((size, size))
        lab = srgb_to_lab(np.asarray(img).astype(np.float64) / 255.0)
        pl = radial_power(lab[..., 0])
        pc = (radial_power(lab[..., 1]) + radial_power(lab[..., 2])) / 2
        sl, sc = spectral_slope(pl), spectral_slope(pc)
        if not (np.isfinite(sl) and np.isfinite(sc)):
            continue
        slopes["luma"].append(sl)
        slopes["chroma"].append(sc)
        profiles["luma"].append(pl / np.nanmax(pl))
        profiles["chroma"].append(pc / np.nanmax(pc))
    lu = np.asarray(slopes["luma"])
    ch = np.asarray(slopes["chroma"])
    diff = ch - lu
    return {
        "images": int(len(lu)),
        "spectral_slope_luma_mean": float(lu.mean()),
        "spectral_slope_luma_sem": float(lu.std(ddof=1) / np.sqrt(len(lu))),
        "spectral_slope_chroma_mean": float(ch.mean()),
        "spectral_slope_chroma_sem": float(ch.std(ddof=1) / np.sqrt(len(ch))),
        "slope_difference_mean": float(diff.mean()),
        "slope_difference_sem": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "fraction_of_images_chroma_steeper": float((diff > 0).mean()),
        "mean_profile_luma": np.nanmean(np.stack(profiles["luma"]), axis=0).tolist(),
        "mean_profile_chroma": np.nanmean(np.stack(profiles["chroma"]), axis=0).tolist(),
    }


def _ridge_cv_r2(X, Y, folds=5, lam=1.0):
    """Cross-validated R^2 of a ridge probe from X to Y, averaged over targets."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    preds = np.zeros_like(Y)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
        ym = Y[tr].mean(0)
        Xtr = (X[tr] - xm) / xs
        Xte = (X[te] - xm) / xs
        A = Xtr.T @ Xtr + lam * len(tr) * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ (Y[tr] - ym))
        preds[te] = Xte @ W + ym
    ss_res = ((Y - preds) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def analyse_band_probes(model, style_dir, names, size, device, n_bands=8):
    """How much colour and how much texture each band of the split actually carries.

    Style images are encoded, the deep feature is split by the trained low-pass, and
    each band is pooled to per-channel mean and standard deviation. A ridge probe is
    then fitted from those pooled statistics to two disjoint targets: the chroma
    statistics of the image, and a colour-free description of its texture given by
    the log energy of the greyscale image in equal radial frequency bands. The
    cross-validated R^2 says how much of each target a band linearly determines.
    """
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    X_low, X_high, Y_col, Y_tex = [], [], [], []
    energy = []
    with torch.no_grad():
        for n in names:
            img = Image.open(os.path.join(style_dir, n)).convert("RGB")
            t = tf(img).unsqueeze(0).to(device)
            feat = model.encoder(t)[-1]
            low, high = model.freq(feat)
            # How the energy of the deep feature divides between the two bands. This
            # is why a single attention map cannot serve both: the residual high band
            # is far the smaller of the two, so similarities computed on the full
            # feature are set almost entirely by the low band.
            energy.append(float((high ** 2).sum() / ((low ** 2).sum() + 1e-12)))
            for bank, band in ((X_low, low), (X_high, high)):
                b = band[0]
                bank.append(torch.cat([b.mean(dim=(1, 2)), b.std(dim=(1, 2))]).cpu().numpy())
            arr = t[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
            lab = srgb_to_lab(np.clip(arr, 0, 1))
            ab = lab[..., 1:]
            Y_col.append(np.concatenate([ab.mean(axis=(0, 1)), ab.std(axis=(0, 1))]))
            prof = radial_power(arr.mean(axis=2))
            edges = np.linspace(1, len(prof), n_bands + 1).astype(int)
            Y_tex.append([np.log(np.nansum(prof[edges[i]:edges[i + 1]]) + 1e-12)
                          for i in range(n_bands)])
    e = np.asarray(energy, dtype=float)
    return {
        "images": len(names),
        "high_over_low_energy_mean": float(e.mean()),
        "high_over_low_energy_sem": float(e.std(ddof=1) / np.sqrt(len(e))),
        "high_over_low_energy_median": float(np.median(e)),
        "low_band_predicts_colour_r2": _ridge_cv_r2(X_low, Y_col),
        "high_band_predicts_colour_r2": _ridge_cv_r2(X_high, Y_col),
        "low_band_predicts_texture_r2": _ridge_cv_r2(X_low, Y_tex),
        "high_band_predicts_texture_r2": _ridge_cv_r2(X_high, Y_tex),
    }


# --------------------------------------------------------------------------- C
def chroma_stats(t: torch.Tensor) -> torch.Tensor:
    """Mean and standard deviation of the a, b channels: colour, with no texture."""
    img = t[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float64)
    lab = srgb_to_lab(np.clip(img, 0, 1))
    ab = lab[..., 1:]
    return torch.tensor(np.concatenate([ab.mean(axis=(0, 1)), ab.std(axis=(0, 1))]))


def texture_gram(encoder, t: torch.Tensor, layers=(2, 3)):
    """Gram matrices of the high-pass of deep features: texture, with no palette.

    The image is reduced to luminance first, so that nothing chromatic can reach this
    statistic and the two measures stay disjoint.
    """
    grey = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    feats = encoder(grey)
    return [gram(high_freq(feats[i])) for i in layers]


def texture_gram_shape(encoder, t: torch.Tensor, layers=(2, 3)):
    """Gram matrices of the high-pass, each normalised to unit Frobenius norm.

    Scaling a coefficient changes how much energy the decoder puts into the image,
    and a raw Gram distance registers that change as a change of texture even when
    the pattern of channel correlations is untouched. Dividing each Gram by its own
    norm removes overall contrast and leaves the pattern, so this measure separates
    a genuine change of brushwork from a change of amplitude.
    """
    out = []
    for g in texture_gram(encoder, t, layers):
        out.append(g / (torch.linalg.norm(g.reshape(g.shape[0], -1), dim=1)
                        .view(-1, 1, 1) + 1e-12))
    return out


def _d_chroma(a, b):
    return float(torch.norm(a - b).item())


def _d_texture(a, b):
    return float(sum(F.mse_loss(x, y).item() for x, y in zip(a, b)))


def analyse_bands(model, content_dir, style_dir, names, size, device):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    settings = {"both": (1.0, 1.0), "colour_branch_only": (1.0, 0.0),
                "stroke_branch_only": (0.0, 1.0)}
    acc = {k: {"chroma": [], "texture": []} for k in settings}
    with torch.no_grad():
        for n in names:
            c = tf(Image.open(os.path.join(content_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(os.path.join(style_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            c_ch, s_ch = chroma_stats(c), chroma_stats(s)
            c_tx, s_tx = texture_gram(model.encoder, c), texture_gram(model.encoder, s)
            ref_ch = _d_chroma(c_ch, s_ch)
            ref_tx = _d_texture(c_tx, s_tx)
            if ref_ch <= 0 or ref_tx <= 0:
                continue
            for key, (al, ah) in settings.items():
                out = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                acc[key]["chroma"].append(_d_chroma(chroma_stats(out), s_ch) / ref_ch)
                acc[key]["texture"].append(_d_texture(texture_gram(model.encoder, out), s_tx) / ref_tx)
    out = {"pairs": len(acc["both"]["chroma"])}
    for key in settings:
        for metric in ("chroma", "texture"):
            v = np.asarray(acc[key][metric], dtype=float)
            out[f"{key}__{metric}_ratio_mean"] = float(v.mean())
            out[f"{key}__{metric}_ratio_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    return out


def analyse_sensitivity(model, content_dir, style_dir, names, size, device,
                        sweep=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """How strongly each knob moves each statistic.

    One knob is swept while the other is held at 1. For every pair we fit the slope
    of the normalised distance against the knob by least squares, then average the
    slopes over pairs. Four slopes result: the two direct ones, which the premise
    expects to be large, and the two crossed ones, which it expects to be small.
    Reporting slopes rather than absolute distances removes the fact that the low
    band simply carries more feature energy than the residual high band, which would
    otherwise dominate any direct comparison of the two branches.
    """
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    a = np.asarray(sweep, dtype=float)
    a_centred = a - a.mean()
    denom = float((a_centred ** 2).sum())
    slopes = {f"d_{m}_d_alpha_{k}": [] for m in ("chroma", "texture") for k in ("low", "high")}
    with torch.no_grad():
        for n in names:
            c = tf(Image.open(os.path.join(content_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s = tf(Image.open(os.path.join(style_dir, n)).convert("RGB")).unsqueeze(0).to(device)
            s_ch, s_tx = chroma_stats(s), texture_gram(model.encoder, s)
            ref_ch = _d_chroma(chroma_stats(c), s_ch)
            ref_tx = _d_texture(texture_gram(model.encoder, c), s_tx)
            if ref_ch <= 0 or ref_tx <= 0:
                continue
            for knob in ("low", "high"):
                ch, tx = [], []
                for v in a:
                    al, ah = (v, 1.0) if knob == "low" else (1.0, v)
                    out = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                    ch.append(_d_chroma(chroma_stats(out), s_ch) / ref_ch)
                    tx.append(_d_texture(texture_gram(model.encoder, out), s_tx) / ref_tx)
                ch, tx = np.asarray(ch), np.asarray(tx)
                slopes[f"d_chroma_d_alpha_{knob}"].append(float((a_centred * (ch - ch.mean())).sum() / denom))
                slopes[f"d_texture_d_alpha_{knob}"].append(float((a_centred * (tx - tx.mean())).sum() / denom))
    out = {"pairs": len(slopes["d_chroma_d_alpha_low"]), "sweep": list(sweep)}
    for k, v in slopes.items():
        v = np.asarray(v, dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    # How much more the direct knob moves a statistic than the crossed one.
    eps = 1e-12
    out["selectivity_chroma"] = float(abs(out["d_chroma_d_alpha_low_mean"])
                                      / (abs(out["d_chroma_d_alpha_high_mean"]) + eps))
    out["selectivity_texture"] = float(abs(out["d_texture_d_alpha_high_mean"])
                                       / (abs(out["d_texture_d_alpha_low_mean"]) + eps))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default="results/freq_analysis.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ck.get("cfg", {}).get("model", {"embed_dim": 256, "heads": 4})
    model = FDFormer(embed_dim=mc["embed_dim"], heads=mc["heads"], pretrained_vgg=True,
                     freq_decouple=mc.get("freq_decouple", True),
                     use_gate=mc.get("use_gate", True),
                     freq_mode=mc.get("freq_mode", "learnable"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    names = sorted(f for f in os.listdir(args.content_dir) if f.lower().endswith(_EXTS))[:args.num]

    print("A. response of the learned low-pass ...")
    a = analyse_kernels(model)
    if a.get("kernel", True) is None:
        print("   " + a["note"])
    else:
        print(f"   Gaussian init cut-off {a['gaussian_init_cutoff_cyc_per_px']:.4f} cyc/px; "
              f"learned {a['learned_cutoff_mean_cyc_per_px']:.4f} "
              f"+/- {a['learned_cutoff_std_cyc_per_px']:.4f}")

    # The kernel acts on relu4_1, which sits after three pooling stages, so one
    # feature pixel spans eight image pixels and a cut-off of f cycles per feature
    # pixel is f/8 cycles per image pixel.
    stride = 8
    if a.get("kernel", True) is not None:
        a["encoder_stride_at_relu4_1"] = stride
        a["learned_cutoff_cyc_per_image_px"] = float(
            a["learned_cutoff_mean_cyc_per_px"] / stride)
        print(f"   -> {a['learned_cutoff_cyc_per_image_px']:.4f} cycles per image pixel")

    print("B1. how fast chroma and luminance energy fall off, in real paintings ...")
    b = analyse_style_spectra(args.style_dir, names, args.size)
    print(f"   spectral slope: luminance {b['spectral_slope_luma_mean']:.3f}, "
          f"chroma {b['spectral_slope_chroma_mean']:.3f} "
          f"(chroma steeper in {b['fraction_of_images_chroma_steeper'] * 100:.1f}% of images)")

    print("B2. what each band of the split linearly determines ...")
    bp = analyse_band_probes(model, args.style_dir, names, args.size, device)
    print(f"   high/low band energy: {bp['high_over_low_energy_mean']:.4f} "
          f"(median {bp['high_over_low_energy_median']:.4f})")
    print(f"   colour  R2: low {bp['low_band_predicts_colour_r2']:.3f}  "
          f"high {bp['high_band_predicts_colour_r2']:.3f}")
    print(f"   texture R2: low {bp['low_band_predicts_texture_r2']:.3f}  "
          f"high {bp['high_band_predicts_texture_r2']:.3f}")

    print("C. band attribution over the trained model ...")
    c = analyse_bands(model, args.content_dir, args.style_dir, names, args.size, device)
    for k in sorted(c):
        if k != "pairs":
            print(f"   {k}: {c[k]:.4f}")

    print("D. sensitivity of each statistic to each knob ...")
    d = analyse_sensitivity(model, args.content_dir, args.style_dir, names, args.size, device)
    for k in sorted(d):
        if isinstance(d[k], float):
            print(f"   {k}: {d[k]:.4f}")

    rec = {"checkpoint": args.ckpt, "kernel_response": a, "style_spectra": b,
           "band_probes": bp, "band_attribution": c, "knob_sensitivity": d}
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
