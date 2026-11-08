"""Figures added for the revision.

Each command reads a JSON written by one of the revision experiments and draws from
it; none of them contains a number of its own.

  freqevidence  four panels of evidence that the split separates colour from texture
  controlgrid   alpha grids for four content-style cases chosen to span portrait,
                landscape, high-frequency-dominant style and frequency-overlapping style
  controlgen    how the two coefficients behave across many pairs, not one
  uncertainty   paired differences against each baseline, with confidence intervals
  decomp        the three decomposition operators compared after identical training

Usage
-----
python make_figures_revision.py all --out_dir figures
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from PIL import Image                               # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(ROOT, "figures")
BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#8C8C8C"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _save(fig, out_dir, name):
    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {path}")
    return path


def _load(path):
    path = _abs(path)
    if not os.path.isfile(path):
        print(f"[skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- A
def cmd_freqevidence(args):
    r = _load(args.freq_json)
    if r is None:
        return None
    k, sp, bp, sens = (r["kernel_response"], r["style_spectra"],
                       r["band_probes"], r["knob_sensitivity"])

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))

    # (a) response of the low-pass, learned against the Gaussian it started from
    lp = np.asarray(k["mean_learned_profile"])
    gp = np.asarray(k["gaussian_profile"])
    f = np.arange(len(lp)) / (2 * len(lp))
    ax[0].axvspan(k["learned_cutoff_p05_cyc_per_px"], k["learned_cutoff_p95_cyc_per_px"],
                  color=BLUE, alpha=0.13, lw=0,
                  label="learned cut-offs, 5th–95th pct")
    ax[0].plot(f, gp, "--", color=GREY, label="Gaussian initialisation")
    ax[0].plot(f, lp, "-", color=BLUE, label="learned, channel mean")
    ax[0].axhline(10 ** (-3 / 20), color=RED, lw=0.8, ls=":")
    ax[0].axvline(k["gaussian_init_cutoff_cyc_per_px"], color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel("cycles per feature pixel", fontsize=9)
    ax[0].set_ylabel("response", fontsize=9)
    ax[0].set_title("(a) the low-pass that was learned", fontsize=10)
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    ax[0].grid(True, ls=":", alpha=0.5)
    ax[0].text(0.97, 0.55,
               f"mean cut-off barely moves,\n"
               f"{k['gaussian_init_cutoff_cyc_per_px']:.3f} $\\rightarrow$ "
               f"{k['learned_cutoff_mean_cyc_per_px']:.3f};\n"
               f"the channels spread out instead",
               transform=ax[0].transAxes, ha="right", va="center", fontsize=7)

    # (b) how fast chroma and luminance energy fall off in real paintings
    pl = np.asarray(sp["mean_profile_luma"])
    pc = np.asarray(sp["mean_profile_chroma"])
    fi = np.arange(1, len(pl)) / (2 * len(pl))
    ax[1].loglog(fi, pl[1:], color=BLUE, label=f"luminance, slope {sp['spectral_slope_luma_mean']:.2f}")
    ax[1].loglog(fi, pc[1:], color=GREEN, label=f"chroma, slope {sp['spectral_slope_chroma_mean']:.2f}")
    ax[1].set_xlabel("cycles per image pixel", fontsize=9)
    ax[1].set_ylabel("normalised power", fontsize=9)
    ax[1].set_title(f"(b) {sp['images']} paintings", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(True, ls=":", alpha=0.5, which="both")
    ax[1].text(0.03, 0.06,
               f"chroma steeper in "
               f"{sp['fraction_of_images_chroma_steeper'] * 100:.0f}% of them",
               transform=ax[1].transAxes, fontsize=7)

    # (c) what each band linearly determines
    labels = ["colour", "texture"]
    low = [bp["low_band_predicts_colour_r2"], bp["low_band_predicts_texture_r2"]]
    high = [bp["high_band_predicts_colour_r2"], bp["high_band_predicts_texture_r2"]]
    x = np.arange(2)
    ax[2].bar(x - 0.19, low, 0.36, color=BLUE, label="low band")
    ax[2].bar(x + 0.19, high, 0.36, color=GREEN, label="high band")
    for xi, (a, b) in enumerate(zip(low, high)):
        ax[2].text(xi - 0.19, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
        ax[2].text(xi + 0.19, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_ylabel("cross-validated $R^2$", fontsize=9)
    ax[2].set_title("(c) what each band determines", fontsize=10)
    ax[2].legend(fontsize=7, frameon=False)
    ax[2].grid(True, axis="y", ls=":", alpha=0.5)

    # (d) what each branch actually transfers
    ba = r["band_attribution"]
    settings = [("both", "both branches"), ("colour_branch_only", "low branch only"),
                ("stroke_branch_only", "high branch only")]
    x = np.arange(len(settings))
    ch = [ba[f"{k}__chroma_ratio_mean"] for k, _ in settings]
    che = [ba[f"{k}__chroma_ratio_sem"] for k, _ in settings]
    tx = [ba[f"{k}__texture_ratio_mean"] for k, _ in settings]
    txe = [ba[f"{k}__texture_ratio_sem"] for k, _ in settings]
    ax[3].bar(x - 0.19, ch, 0.36, yerr=che, capsize=3, color=BLUE, label="chromatic")
    ax[3].bar(x + 0.19, tx, 0.36, yerr=txe, capsize=3, color=GREEN, label="textural")
    ax[3].axhline(1.0, color=GREY, lw=0.9, ls="--")
    ax[3].text(2.42, 1.02, "content", fontsize=6.5, color=GREY, ha="right")
    ax[3].set_xticks(x)
    ax[3].set_xticklabels([lab for _, lab in settings], fontsize=8, rotation=12, ha="right")
    ax[3].set_ylabel("distance to style (1 = content)", fontsize=8.5)
    ax[3].set_title(f"(d) what each branch transfers, {ba['pairs']} pairs", fontsize=10)
    ax[3].legend(fontsize=7, frameon=False)
    ax[3].grid(True, axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_freqevidence.pdf")


# --------------------------------------------------------------------------- B
_CASE_LABELS = {
    "00249.png": "portrait",
    "00074.png": "landscape",
    "00310.png": "high-frequency style",
    "00648.png": "frequency-overlapping style",
}


def cmd_controlgrid(args):
    import torch
    from torchvision import transforms
    sys.path.insert(0, ROOT)
    from models import FDFormer

    ck_path = _abs(args.ckpt)
    if not os.path.isfile(ck_path):
        print(f"[skip] fig_controlgrid: no checkpoint at {args.ckpt}")
        return None
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    mc = (ck.get("cfg", {}) or {}).get("model", {})
    model = FDFormer(embed_dim=mc.get("embed_dim", 256), heads=mc.get("heads", 4),
                     pretrained_vgg=True, freq_decouple=mc.get("freq_decouple", True),
                     use_gate=mc.get("use_gate", True),
                     freq_mode=mc.get("freq_mode", "learnable")).to(device).eval()
    model.load_state_dict(ck["model"])

    load = transforms.Compose([transforms.Resize((args.size, args.size)),
                               transforms.ToTensor()])
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    alphas = [float(x) for x in args.alphas.split(",")]
    n = len(alphas)
    ncol = 2 + n

    nrow = 2 * len(cases)
    fig = plt.figure(figsize=(1.45 * ncol, 1.45 * nrow + 0.9))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.05, hspace=0.34,
                          left=0.075, right=0.995, top=0.945, bottom=0.005)
    with torch.no_grad():
        for r, name in enumerate(cases):
            c = load(Image.open(_abs(os.path.join(args.content_dir, name))).convert("RGB")
                     ).unsqueeze(0).to(device)
            s = load(Image.open(_abs(os.path.join(args.style_dir, name))).convert("RGB")
                     ).unsqueeze(0).to(device)
            for k, (img, lab) in enumerate(((c, "content"), (s, "style"))):
                ax = fig.add_subplot(gs[2 * r + k, 0])
                ax.imshow(img[0].permute(1, 2, 0).cpu().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_ylabel(lab, fontsize=7.5)
            # row 1 of the case sweeps the colour coefficient, row 2 the stroke one
            for k, knob in enumerate(("low", "high")):
                for j, a in enumerate(alphas):
                    al, ah = (a, 1.0) if knob == "low" else (1.0, a)
                    out = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                    ax = fig.add_subplot(gs[2 * r + k, j + 1])
                    ax.imshow(out[0].permute(1, 2, 0).cpu().numpy())
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0:
                        pos = ax.get_position()
                        fig.text(pos.x0 - 0.008, pos.y0 + pos.height / 2,
                                 rf"sweeping $\alpha_\mathrm{{{knob}}}$", fontsize=8,
                                 rotation=90, va="center", ha="right")
                    if r == 0 and k == 0:
                        ax.set_title(f"{a:g}", fontsize=9, pad=4)
                    if j == len(alphas) - 1 and k == 0:
                        pos = ax.get_position()
                        fig.text(0.075, pos.y1 + 0.006, _CASE_LABELS.get(name, name),
                                 fontsize=9.5, ha="left", va="bottom")
    return _save(fig, args.out_dir, "fig_controlgrid.pdf")


# --------------------------------------------------------------------------- C
def cmd_controlgen(args):
    """Direct and crossed response of each coefficient, on the absolute quantities
    the coefficients are meant to control rather than on a distance to the style."""
    r = _load(args.control_json)
    if r is None:
        return None
    alphas = np.asarray(r["alphas"], dtype=float)
    lo, hi = r["usable_range"]
    panels = (("colour_travel", "low", "colour travel  (0 = content, 1 = style)", BLUE),
              ("stroke_energy", "high", "stroke energy  (1 = content)", GREEN))
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for p, (metric, direct, ylab, colour) in enumerate(panels):
        crossed = "high" if direct == "low" else "low"
        ax[p].axvspan(lo, hi, color="k", alpha=0.05, lw=0)
        for kk, sty, mk in ((direct, "-", "o"), (crossed, ":", "s")):
            m = np.asarray(r["curves"][f"{metric}__alpha_{kk}"]["mean"])
            e = np.asarray(r["curves"][f"{metric}__alpha_{kk}"]["sem"])
            col = colour if kk == direct else GREY
            span = r["span_usable"][f"{metric}__alpha_{kk}"]
            ax[p].plot(alphas, m, sty, marker=mk, ms=4, color=col,
                       label=rf"$\alpha_\mathrm{{{kk}}}$  (range {span:.3f})")
            ax[p].fill_between(alphas, m - 1.96 * e, m + 1.96 * e, alpha=0.18,
                               color=col, lw=0)
        ax[p].set_xlabel("coefficient value", fontsize=10)
        ax[p].set_ylabel(ylab, fontsize=9)
        ax[p].set_title(f"{metric.replace('_', ' ')}, {r['pairs']} pairs", fontsize=10)
        ax[p].legend(fontsize=8, frameon=False)
        ax[p].grid(True, ls=":", alpha=0.5)
    fig.text(0.5, -0.03, "shaded band marks the range over which outputs stay usable; "
             "the quoted range is measured inside it", ha="center", fontsize=7.5,
             color="black")
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_controlgen.pdf")


# --------------------------------------------------------------------------- D
_PRETTY = {"sanet": "SANet", "adaattn": "AdaAttN", "stytr2": "StyTr$^2$",
           "microast": "MicroAST", "aespanet": "AesPA-Net", "samam": "SaMam"}


def cmd_uncertainty(args):
    r = _load(args.unc_json)
    if r is None:
        return None
    key = next(k for k in r if k.startswith("paired_vs_"))
    ref = key[len("paired_vs_"):]
    names = [n for n in _PRETTY if n in r[key]]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 0.52 * len(names) + 2.0))
    for p, metric in enumerate(("lpips", "style")):
        y = np.arange(len(names))
        m = np.array([r[key][n][metric]["diff_mean"] for n in names])
        lo = np.array([r[key][n][metric]["diff_ci_low"] for n in names])
        hi = np.array([r[key][n][metric]["diff_ci_high"] for n in names])
        colours = [GREEN if h < 0 else (RED if l > 0 else GREY)
                   for l, h in zip(lo, hi)]
        ax[p].errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor=colours, lw=2)
        ax[p].scatter(m, y, c=colours, s=26, zorder=3)
        ax[p].axvline(0, color="k", lw=0.8)
        ax[p].set_yticks(y)
        ax[p].set_yticklabels([_PRETTY[n] for n in names], fontsize=9)
        ax[p].invert_yaxis()
        ax[p].set_xlabel("FDFormer $-$ baseline" +
                         ("  (content LPIPS)" if metric == "lpips" else "  (style distance)"),
                         fontsize=9)
        ax[p].set_title("negative favours FDFormer", fontsize=9)
        ax[p].grid(True, axis="x", ls=":", alpha=0.5)
        # SaMam's LPIPS lead widens the range past +0.07; at the default density
        # the tick labels then run into one another
        ax[p].xaxis.set_major_locator(plt.MaxNLocator(5))
        ax[p].tick_params(axis="x", labelsize=8.5)
    fig.suptitle(f"paired difference over {r['pairs']} shared pairs, 95% bootstrap interval",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, args.out_dir, "fig_uncertainty.pdf")


# --------------------------------------------------------------------------- E
def cmd_decomp(args):
    r = _load(args.results)
    if r is None:
        return None
    m = r.get("metrics", {})
    rows = [("learned blur", "fdformer_v2"), ("fixed Gaussian", "fdformer_v2_freq_fixed"),
            ("Haar wavelet", "fdformer_v2_freq_dwt"), ("no split", "fdformer_v2_single_path")]
    rows = [(lab, k) for lab, k in rows if k in m and m[k].get("artfid") is not None]
    if len(rows) < 2:
        print("[skip] fig_decomp: fewer than two decomposition variants scored")
        return None
    labs = [lab for lab, _ in rows]
    vals = [m[k]["artfid"] for _, k in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    colours = [BLUE] + [GREY] * (len(rows) - 1)
    b = ax.bar(labs, vals, color=colours, width=0.6)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_ylabel("ArtFID $\\downarrow$", fontsize=10)
    ax.set_ylim(min(vals) - 6, max(vals) + 5)
    ax.set_title("low-pass operator, identical training otherwise", fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_decomp.pdf")


_EFF_STYLE = [("sanet", "SANet", "#C44E52", "o"),
              ("adaattn", "AdaAttN", "#8172B2", "s"),
              ("stytr2", "StyTr$^2$", "#CCB974", "^"),
              ("microast", "MicroAST", "#64B5CD", "v"),
              ("aespanet", "AesPA-Net", "#937860", "D"),
              ("fdformer", "FDFormer (ours)", "#4C72B0", "*")]


_METRIC_KEY = {"sanet": "sanet", "adaattn": "adaattn", "stytr2": "stytr2",
               "microast": "microast", "aespanet": "aespanet", "samam": "samam",
               "fdformer": "fdformer_v2_alpha140"}


def cmd_tradeoff(args):
    """Quality against cost, on the two cost axes that can be compared fairly.

    Latency is only comparable for the methods we can run on the path their authors
    time, so SaMam appears on the parameter axis alone; the caption says so.
    """
    r = _load(args.results)
    if r is None:
        return None
    eff, met = r.get("efficiency", {}), r.get("metrics", {})
    pts = []
    for key, lab, col, mk in _EFF_STYLE + [("samam", "SaMam", "#55A868", "P")]:
        mk_key = _METRIC_KEY.get(key)
        if not (isinstance(eff.get(key), dict) and mk_key in met):
            continue
        pts.append((key, lab, col, mk, eff[key], met[mk_key]["artfid"]))
    if len(pts) < 3:
        print("[skip] fig_tradeoff: too few methods measured")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for key, lab, col, mk, e, artfid in pts:
        ours = key == "fdformer"
        ms = 15 if ours else 9
        if e.get("latency_ms"):
            ax[0].scatter(e["latency_ms"], artfid, s=ms ** 2 / 2, marker=mk,
                          color=col, edgecolor="k" if ours else "none",
                          linewidth=1.1 if ours else 0, zorder=3, label=lab)
            ax[0].annotate(lab, (e["latency_ms"], artfid), fontsize=7.5,
                           textcoords="offset points", xytext=(7, 4))
        if e.get("params_M"):
            ax[1].scatter(e["params_M"], artfid, s=ms ** 2 / 2, marker=mk,
                          color=col, edgecolor="k" if ours else "none",
                          linewidth=1.1 if ours else 0, zorder=3)
            ax[1].annotate(lab, (e["params_M"], artfid), fontsize=7.5,
                           textcoords="offset points", xytext=(7, 4))
    ax[0].set_xlabel("latency at $256\\times256$ (ms)", fontsize=9.5)
    ax[1].set_xlabel("parameters (M)", fontsize=9.5)
    for a, t in ((ax[0], "against time"), (ax[1], "against size")):
        a.set_ylabel("ArtFID $\\downarrow$", fontsize=9.5)
        a.set_xscale("log")
        a.set_title(t, fontsize=10)
        a.grid(True, ls=":", alpha=0.5, which="both")
        a.margins(x=0.22, y=0.16)
    fig.text(0.5, -0.02, "lower and further left is better; SaMam runs here only "
             "through a fallback path, so it is shown on the size axis alone",
             ha="center", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_tradeoff.pdf")


def cmd_effscaling(args):
    """How each method's cost grows with resolution, measured by the same code."""
    r = _load(args.results)
    if r is None:
        return None
    eff = r.get("efficiency", {})
    have = [(k, lab, col, mk) for k, lab, col, mk in _EFF_STYLE
            if isinstance(eff.get(k), dict) and eff[k].get("sweep")]
    if len(have) < 2:
        print("[skip] fig_effscaling: fewer than two methods have a resolution sweep")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, lab, col, mk in have:
        sw = eff[key]["sweep"]
        x = np.asarray(sw["sizes"], dtype=float)
        ours = key == "fdformer"
        lw = 2.2 if ours else 1.3
        ms = 9 if ours else 5
        ax[0].plot(x, sw["latency_ms"], "-", marker=mk, ms=ms, lw=lw, color=col, label=lab)
        mem = [m for m in sw["peak_mem_MB"] if m is not None]
        if len(mem) == len(x):
            ax[1].plot(x, np.asarray(mem) / 1000.0, "-", marker=mk, ms=ms, lw=lw,
                       color=col, label=lab)
    for a, ylab, title in ((ax[0], "latency (ms)", "time"),
                           (ax[1], "peak memory (GB)", "memory")):
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xlabel("input resolution (square side, px)", fontsize=9.5)
        a.set_ylabel(ylab, fontsize=9.5)
        a.set_title(f"{title} against resolution", fontsize=10)
        a.grid(True, ls=":", alpha=0.5, which="both")
        a.set_xticks([256, 384, 512, 768, 1024, 1536, 2048])
        a.set_xticklabels(["256", "384", "512", "768", "1024", "1536", "2048"], fontsize=8)
    ax[1].axhline(8.0, color=GREY, lw=1.0, ls="--")
    ax[1].text(260, 8.4, "8 GB card", fontsize=7.5, color=GREY)
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.text(0.5, -0.02, "each curve ends at the last resolution that completed before "
             "the allocator ran out of memory", ha="center", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_effscaling.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "freqevidence", "controlgrid", "controlgen",
                                    "uncertainty", "decomp", "effscaling", "tradeoff"])
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--freq_json", default="results/freq_analysis.json")
    ap.add_argument("--control_json", default="results/control_generalisation.json")
    ap.add_argument("--unc_json", default="results/uncertainty.json")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--ckpt", default="experiments/fdformer_v2/ckpt_080000.pth")
    ap.add_argument("--content_dir", default="eval/default/content")
    ap.add_argument("--style_dir", default="eval/default/style")
    ap.add_argument("--cases", default="00249.png,00074.png,00310.png,00648.png")
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    todo = (["freqevidence", "controlgrid", "controlgen", "uncertainty", "decomp",
             "effscaling", "tradeoff"] if args.cmd == "all" else [args.cmd])
    for name in todo:
        globals()["cmd_" + name](args)


if __name__ == "__main__":
    main()
