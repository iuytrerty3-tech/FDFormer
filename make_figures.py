"""Produce the paper's figures from REAL model outputs and MEASURED results.

Nothing here invents data. Each subcommand reads files your experiments produced
(the name-aligned image folders under eval/, and results.json) and renders a
figure into ../paper/figures/. A figure whose inputs are missing is skipped with
a clear message, so a figure exists only once the experiment behind it has run.

The autofill step (autofill_paper.py) drops each produced figure into the paper:
a \\figbox stand-in becomes \\includegraphics the moment its file appears here.

Subcommands
-----------
  overview     architecture schematic (no data needed)            -> fig_overview.pdf
  qualitative  content|style|baselines|ours grid from eval/        -> fig_qualitative.pdf
  control      alpha_low x alpha_high sweep (loads an FDFormer ckpt)-> fig_control.pdf
  ablation     content|style|full|w/o-freq|w/o-gate|w/o-hf grid    -> fig_ablation.pdf
  efficiency   bars + linear-vs-quadratic scaling curve from JSON  -> fig_efficiency.pdf,
                                                                      fig_effcurve.pdf
  all          every figure whose inputs already exist

Examples
--------
  python make_figures.py overview
  python make_figures.py qualitative --rows 5
  python make_figures.py ablation --rows 5
  python make_figures.py control --ckpt experiments/fdformer_default/ckpt_080000.pth
  python make_figures.py efficiency --results results.json
  python make_figures.py all

Image-folder convention (built by batch_stylize.py / run_all.py):
  eval/default/content   eval/default/style          the canonical paired inputs
  eval/default/stylized  FDFormer outputs (run key "default")
  eval/<key>/stylized    each ablation's outputs (single_path / no_gate / no_hf)
  eval/<baseline>/stylized  put each baseline's outputs here, SAME filenames,
                            so the grid lines up. Or pass --map LABEL=/path/to/dir.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: no display needed on a remote GPU box
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from PIL import Image                     # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(ROOT, "..", "paper", "figures")
_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_TILE = 256  # display size of each image tile in the grids


# ---- small filesystem helpers -------------------------------------------------
def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _names(d):
    d = _abs(d)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_EXTS))


def _load_tile(path, size=_TILE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return img


def _ensure_out(out_dir):
    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---- generic image-grid renderer ---------------------------------------------
def render_grid(col_dirs, col_labels, content_dir, style_dir, rows, out_path,
                title=None, names=None):
    """col_dirs[i] is a stylized folder (or None to skip that column).

    Columns laid out as: Content | Style | <present col_dirs...>. Rows are the
    first `rows` filenames present in content+style+every present column (or the
    explicit `names` list, filtered to available files), so the grid never shows
    a blank cell. Returns the out_path on success, else None.
    """
    present = [(lbl, d) for lbl, d in zip(col_labels, col_dirs)
               if d is not None and os.path.isdir(_abs(d)) and _names(d)]
    if not present:
        print(f"[skip] {os.path.basename(out_path)}: none of the stylised "
              f"folders exist yet ({', '.join(col_labels)}).")
        return None

    common = set(_names(content_dir)) & set(_names(style_dir))
    for _, d in present:
        common &= set(_names(d))
    if not common:
        print(f"[skip] {os.path.basename(out_path)}: no shared filenames across "
              "content/style/stylised folders; were all methods run on the same pairs?")
        return None
    if names:
        common = [n for n in names if n in common]
    else:
        common = sorted(common)
    common = common[:rows]

    headers = ["Content", "Style"] + [lbl for lbl, _ in present]
    ncols, nrows = len(headers), len(common)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.6 * ncols, 1.6 * nrows + 0.3))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for r, name in enumerate(common):
        tiles = [_load_tile(os.path.join(_abs(content_dir), name)),
                 _load_tile(os.path.join(_abs(style_dir), name))]
        for _, d in present:
            tiles.append(_load_tile(os.path.join(_abs(d), name)))
        for c, tile in enumerate(tiles):
            ax = axes[r][c]
            ax.imshow(tile)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(headers[c], fontsize=10, pad=4)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.04,
                        top=0.93 if title else 0.95, bottom=0.01,
                        left=0.01, right=0.99)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out_path}  ({nrows} rows x {len(present)} methods)")
    return out_path


def _parse_map(pairs):
    """--map 'AdaIN=/path/a' 'SANet=/path/b'  ->  {'AdaIN': '/path/a', ...}"""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--map expects LABEL=DIR, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---- qualitative comparison (Figure 2) ---------------------------------------
QUAL_COLUMNS = [("SANet", "eval/sanet/stylized"),
                ("AdaAttN", "eval/adaattn/stylized"),
                ("StyTr$^2$", "eval/stytr2/stylized"),
                ("MicroAST", "eval/microast/stylized"),
                ("AesPA-Net", "eval/aespanet/stylized"),
                ("SaMam", "eval/samam/stylized"),
                ("FDFormer (ours)", "eval/v2/stylized")]


def cmd_qualitative(args):
    override = _parse_map(args.map)
    labels = [lbl for lbl, _ in QUAL_COLUMNS]
    dirs = [override.get(lbl, d) for lbl, d in QUAL_COLUMNS]
    out = os.path.join(_ensure_out(args.out_dir), "fig_qualitative.pdf")
    names = [n.strip() for n in args.names.split(",")] if getattr(args, "names", None) else None
    return render_grid(dirs, labels, args.content_dir, args.style_dir,
                       args.rows, out, names=names)


# ---- ablation comparison (Figure: ablation visuals) --------------------------
# The last two arms replace the learned low-pass with a frozen Gaussian and with a
# Haar split. They are dropped automatically if their folders are absent, so this
# figure can be built before those runs finish.
ABL_COLUMNS = [("Full", "eval/v2/stylized"),
               ("w/o freq.", "eval/v2_single_path/stylized"),
               ("w/o gate", "eval/v2_no_gate/stylized"),
               ("w/o HF loss", "eval/v2_no_hf/stylized"),
               ("fixed Gaussian", "eval/v2_freq_fixed/stylized"),
               ("Haar wavelet", "eval/v2_freq_dwt/stylized")]


def cmd_ablation(args):
    present = [(lbl, d) for lbl, d in ABL_COLUMNS if _names(d)]
    missing = [lbl for lbl, d in ABL_COLUMNS if not _names(d)]
    if missing:
        print(f"[note] fig_ablation: no outputs yet for {', '.join(missing)}; "
              f"drawing the {len(present)} arms that exist")
    if not present:
        print("[skip] fig_ablation: no ablation outputs found")
        return None
    labels = [lbl for lbl, _ in present]
    dirs = [d for _, d in present]
    out = os.path.join(_ensure_out(args.out_dir), "fig_ablation.pdf")
    return render_grid(dirs, labels, args.content_dir, args.style_dir,
                       args.rows, out)


# ---- controllability sweep (Figure 3) ----------------------------------------
def cmd_control(args):
    import torch                                   # lazy: only this cmd needs it
    from torchvision import transforms
    sys.path.insert(0, ROOT)
    from models import FDFormer

    if not os.path.isfile(_abs(args.ckpt)):
        print(f"[skip] fig_control: checkpoint not found: {args.ckpt}")
        return None
    cnames, snames = _names(args.content_dir), _names(args.style_dir)
    if not cnames or not snames:
        print(f"[skip] fig_control: need images in {args.content_dir} and "
              f"{args.style_dir}.")
        return None

    device = torch.device(args.device if (args.device != "cuda" or
                          torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(_abs(args.ckpt), map_location=device, weights_only=False)
    mc = (ckpt.get("cfg", {}) or {}).get("model", {})
    model = FDFormer(embed_dim=mc.get("embed_dim", 256), heads=mc.get("heads", 4),
                     pretrained_vgg=True,
                     freq_decouple=mc.get("freq_decouple", True),
                     use_gate=mc.get("use_gate", True)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    if not model.freq_decouple:
        print("[warn] fig_control: this checkpoint has freq_decouple=False, so "
              "alpha_low/alpha_high do not apply. Use the full-model checkpoint.")

    load = transforms.Compose([transforms.Resize((args.size, args.size)),
                               transforms.ToTensor()])

    def _t(folder, name):
        return load(Image.open(os.path.join(_abs(folder), name)).convert("RGB")
                    ).unsqueeze(0).to(device)

    idx = min(args.pair_index, len(cnames) - 1, len(snames) - 1)
    c = _t(args.content_dir, cnames[idx])
    s = _t(args.style_dir, snames[idx])
    lows = [float(x) for x in args.alphas.split(",")]
    highs = lows

    n = len(lows)
    # layout: left column shows content then style; then an n x n alpha grid
    fig = plt.figure(figsize=(1.6 * (n + 1) + 0.6, 1.6 * n + 0.4))
    gs = fig.add_gridspec(n, n + 1, wspace=0.04, hspace=0.04)
    axc = fig.add_subplot(gs[0, 0]); axs = fig.add_subplot(gs[1 % n, 0])
    with __import__("torch").no_grad():
        for ax, img, lab in ((axc, c, "content"), (axs, s, "style")):
            ax.imshow(img[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            ax.set_title(lab, fontsize=9); ax.axis("off")
        for i, al in enumerate(lows):
            for j, ah in enumerate(highs):
                out = model(c, s, alpha_low=al, alpha_high=ah).clamp(0, 1)
                ax = fig.add_subplot(gs[i, j + 1])
                ax.imshow(out[0].permute(1, 2, 0).cpu().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(rf"$\alpha_h$={ah:g}", fontsize=9, pad=3)
                if j == 0:
                    ax.set_ylabel(rf"$\alpha_l$={al:g}", fontsize=9)
    out_path = os.path.join(_ensure_out(args.out_dir), "fig_control.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out_path}  (pair #{idx}, {n}x{n} alpha grid)")
    return out_path


# ---- efficiency: bars + scaling curve ----------------------------------------
# SaMam is absent by design: it runs here through its plain-PyTorch fallback rather
# than the fused kernels its authors time, so a latency measured on this machine
# would describe our fallback and not their method. Its parameter count is reported
# in the table, where no such caveat applies.
EFF_ORDER = [("SANet", "sanet"), ("AdaAttN", "adaattn"),
             ("StyTr$^2$", "stytr2"), ("MicroAST", "microast"),
             ("AesPA-Net", "aespanet"), ("FDFormer (ours)", "fdformer")]


def cmd_efficiency(args):
    results = _load_results(args.results)
    if results is None:
        print(f"[skip] efficiency figures: {args.results} not found.")
        return None
    eff = results.get("efficiency", {})
    made = []

    # (a) latency bar chart across whatever methods were measured
    bars = [(lbl, eff[key].get("latency_ms")) for lbl, key in EFF_ORDER
            if isinstance(eff.get(key), dict) and eff[key].get("latency_ms") is not None]
    if bars:
        labels, vals = zip(*bars)
        colors = ["#4C72B0"] * (len(bars) - 1) + ["#C44E52"] \
            if bars[-1][0].startswith("FDFormer") else ["#4C72B0"] * len(bars)
        fig, ax = plt.subplots(figsize=(1.1 * len(bars) + 1.5, 3.2))
        ax.bar(range(len(bars)), vals, color=colors)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Latency at 256$\\times$256 (ms)", fontsize=10)
        ax.set_title("Inference latency (lower is better)", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        p = os.path.join(_ensure_out(args.out_dir), "fig_efficiency.pdf")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[ok]   wrote {p}  ({len(bars)} methods)")
        made.append(p)
    else:
        print("[skip] fig_efficiency: no latency numbers in results.json yet.")

    # (b) scaling curve: latency (and peak memory) vs resolution
    # accept either the lifted location (efficiency.sweep) or the nested one
    # (efficiency.fdformer.sweep), so it works however results.json was built
    sweep = eff.get("sweep") or (eff.get("fdformer") or {}).get("sweep")
    if isinstance(sweep, dict) and sweep.get("sizes") and sweep.get("latency_ms"):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(sweep["sizes"], sweep["latency_ms"], "o-", color="#C44E52",
                label="FDFormer latency")
        ax.set_xlabel("Input resolution (px, square)", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=10, color="#C44E52")
        ax.tick_params(axis="y", labelcolor="#C44E52")
        # overlay any baseline sweeps the user merged (key "sweep_<name>")
        for k, v in eff.items():
            if k.startswith("sweep_") and isinstance(v, dict) and v.get("sizes"):
                ax.plot(v["sizes"], v["latency_ms"], "s--",
                        label=f"{k[6:]} latency")
        if sweep.get("peak_mem_MB"):
            ax2 = ax.twinx()
            ax2.plot(sweep["sizes"], sweep["peak_mem_MB"], "^--", color="#55A868",
                     label="FDFormer peak mem")
            ax2.set_ylabel("Peak GPU memory (MB)", fontsize=10, color="#55A868")
            ax2.tick_params(axis="y", labelcolor="#55A868")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        p = os.path.join(_ensure_out(args.out_dir), "fig_effcurve.pdf")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[ok]   wrote {p}  ({len(sweep['sizes'])} resolutions)")
        made.append(p)
    else:
        print("[skip] fig_effcurve: no 'sweep' in results.json "
              "(run: python bench_efficiency.py --sweep ...).")
    return made or None


def _load_results(path):
    p = _abs(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---- architecture overview (Figure 1) ----------------------------------------
def cmd_overview(args):
    """A clean schematic drawn from boxes + arrows. No experimental data."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF0F7", ec="#3B5B8C", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
        return (x, y, w, h)

    def arrow(p1, p2, color="#444"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color=color))

    box(2, 30, 12, 7, "Content $I_c$", fc="#F7F2EA")
    box(2, 7, 12, 7, "Style $I_s$", fc="#F7F2EA")
    enc_c = box(18, 30, 13, 7, "Frozen\nVGG-19", fc="#E8E8E8")
    enc_s = box(18, 7, 13, 7, "Frozen\nVGG-19", fc="#E8E8E8")
    freq = box(35, 18.5, 14, 14, "Frequency\ndecompose\n(low / high)", fc="#EAF3EC")
    al = box(53, 27, 16, 8, "Linear cross-attn\n(low = colour)", fc="#EAF0F7")
    ah = box(53, 9, 16, 8, "Linear cross-attn\n(high = stroke)", fc="#EAF0F7")
    comb = box(73, 18.5, 10, 8, r"$\alpha_l,\alpha_h$" + "\ncombine", fc="#FBEFF2")
    gate = box(73, 30, 10, 7, "Content-\nadaptive gate", fc="#FBEFF2")
    dec = box(87, 18.5, 11, 8, "Decoder", fc="#E8E8E8")
    box(87, 30, 11, 7, "Output\n$I_{out}$", fc="#F7F2EA")

    arrow((14, 33.5), (18, 33.5)); arrow((14, 10.5), (18, 10.5))
    arrow((31, 33.5), (42, 32.5)); arrow((31, 10.5), (42, 18.5))
    arrow((49, 27), (53, 31)); arrow((49, 24), (53, 13))
    arrow((69, 31), (73, 24)); arrow((69, 13), (73, 22))
    arrow((78, 30), (78, 26.5))            # gate -> combine
    arrow((83, 22.5), (87, 22.5))          # combine -> decoder
    arrow((92.5, 26.5), (92.5, 30))        # decoder -> output
    ax.text(50, 41.5, "FDFormer: frequency-decoupled dual-path cross-attention",
            ha="center", fontsize=11, weight="bold")

    out = os.path.join(_ensure_out(args.out_dir), "fig_overview.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out}  (schematic; replace with a polished diagram if desired)")
    return out


# ---- quantitative charts: quality bars, tradeoff, ablation bars --------------
def _save(fig, out_dir, name):
    p = os.path.join(_ensure_out(out_dir), name)
    fig.savefig(p, bbox_inches="tight", dpi=200)
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {p}")
    return p


def cmd_charts(args):
    """Bar/scatter charts from results.json (no model needed). Each chart is a
    different view and is skipped when its numbers are absent."""
    results = _load_results(args.results)
    if results is None:
        print(f"[skip] charts: {args.results} not found.")
        return None
    metrics = results.get("metrics", {})
    eff = results.get("efficiency", {})
    made = []

    def _hl(labels):
        return ["#C44E52" if l.startswith("FDFormer") else "#4C72B0" for l in labels]

    # (1) quality bars: one subplot per metric, each on its own scale
    present = [(l, k) for l, k in EFF_ORDER
               if isinstance(metrics.get(k), dict) and metrics[k].get("artfid") is not None]
    specs = [("artfid", "ArtFID  (lower better)"),
             ("lpips_c", "Content LPIPS  (lower better)"),
             ("style_dist", "Style distance  (lower better)")]
    if present:
        labels = [l for l, _ in present]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (mk, title) in zip(axes, specs):
            vals = [(metrics[k].get(mk) if metrics[k].get(mk) is not None else 0)
                    for _, k in present]
            ax.bar(range(len(labels)), vals, color=_hl(labels))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_title(title, fontsize=9)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        fig.suptitle("Quality metrics across methods", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        made.append(_save(fig, args.out_dir, "fig_qualbars.pdf"))
    else:
        print("[skip] fig_qualbars: no method metrics in results.json yet.")

    # (2) quality vs efficiency tradeoff: ArtFID (y) against latency (x, log),
    # marker area proportional to parameter count
    # Ours is plotted at the operating point the paper reports, the one chosen on
    # held-out data, rather than at the training point.
    quality_key = {"fdformer": "fdformer_v2_alpha140"}

    def _artfid(k):
        src = metrics.get(quality_key.get(k, k)) or metrics.get(k) or {}
        return src.get("artfid")

    pts = [(l, eff[k].get("latency_ms"), _artfid(k), eff[k].get("params_M"))
           for l, k in EFF_ORDER
           if isinstance(eff.get(k), dict) and isinstance(metrics.get(k), dict)
           and eff[k].get("latency_ms") is not None and _artfid(k) is not None]
    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for l, x, y, pm in pts:
            col = "#C44E52" if l.startswith("FDFormer") else "#4C72B0"
            size = 60 if pm is None else 14 * pm
            ax.scatter(x, y, s=size, color=col, alpha=0.75,
                       edgecolors="black", linewidths=0.6, zorder=3)
            tag = l if pm is None else f"{l}\n{pm:.1f} M, {x:.1f} ms"
            # (dx, dy, ha) per method. FDFormer sits between SANet, just above and
            # to its right, and AdaAttN, below and to its right: a label pushed
            # down lands on AdaAttN's marker, so it goes to the empty left side,
            # right-aligned against its own marker.
            offsets = {"FDFormer (ours)": (-9, -3, "right"), "SANet": (12, 2, "left"),
                       "MicroAST": (8, 2, "left"), "AesPA-Net": (-34, 14, "left"),
                       "AdaAttN": (10, -22, "left"), "StyTr$^2$": (-22, 14, "left")}
            dx, dy, ha = offsets.get(l, (8, 5, "left"))
            ax.annotate(tag, (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8, ha=ha, va="center" if ha == "right" else "baseline",
                        fontweight="bold" if l.startswith("FDFormer") else "normal",
                        zorder=4)
        ax.set_xscale("log")
        # Derived from the data rather than fixed: MicroAST sits below 7 ms and a
        # hard-coded left edge would silently drop it off the plot.
        xs = [p[1] for p in pts]
        ax.set_xlim(min(xs) / 1.6, max(xs) * 1.9)
        ymin = min(p[2] for p in pts)
        ymax = max(p[2] for p in pts)
        pad = 0.18 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("Latency at 256$\\times$256 (ms, log scale)", fontsize=10)
        ax.set_ylabel("ArtFID (lower is better)", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        made.append(_save(fig, args.out_dir, "fig_tradeoff.pdf"))
    else:
        print("[skip] fig_tradeoff: need latency + ArtFID for >=2 methods.")

    # (3) ablation bars: ArtFID across the full model and each ablation
    abl = [("Full", "fdformer"), ("w/o freq.", "single_path"),
           ("w/o gate", "no_gate"), ("w/o HF", "no_hf")]
    av = [(l, metrics[k].get("artfid")) for l, k in abl
          if isinstance(metrics.get(k), dict) and metrics[k].get("artfid") is not None]
    if len(av) >= 2:
        labels, vals = zip(*av)
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        colors = ["#C44E52"] + ["#4C72B0"] * (len(av) - 1)
        ax.bar(range(len(av)), vals, color=colors)
        ax.set_xticks(range(len(av))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ArtFID (lower better)", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        made.append(_save(fig, args.out_dir, "fig_ablbars.pdf"))
    else:
        print("[skip] fig_ablbars: need ArtFID for the ablation runs.")
    return made or None


def cmd_controlcurve(args):
    """Quantify the two knobs: sweep alpha_low and alpha_high and measure how a
    colour distance and a high-frequency (stroke) distance respond. A genuine
    measurement from the trained model; skipped if the checkpoint is absent."""
    import torch
    from torchvision import transforms
    sys.path.insert(0, ROOT)
    try:
        from models import FDFormer
        from losses.losses import calc_mean_std, hf_stroke_loss
    except Exception as e:                                   # noqa: BLE001
        print(f"[skip] fig_controlcurve: cannot import model/losses ({e}).")
        return None
    if not os.path.isfile(_abs(args.ckpt)):
        print(f"[skip] fig_controlcurve: checkpoint not found: {args.ckpt}")
        return None
    cnames, snames = _names(args.content_dir), _names(args.style_dir)
    if not cnames or not snames:
        print("[skip] fig_controlcurve: missing content/style images.")
        return None
    device = torch.device(args.device if (args.device != "cuda" or
                          torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(_abs(args.ckpt), map_location=device, weights_only=False)
    mc = (ckpt.get("cfg", {}) or {}).get("model", {})
    model = FDFormer(embed_dim=mc.get("embed_dim", 256), heads=mc.get("heads", 4),
                     pretrained_vgg=True, freq_decouple=mc.get("freq_decouple", True),
                     use_gate=mc.get("use_gate", True)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    if not getattr(model, "freq_decouple", True):
        print("[skip] fig_controlcurve: ckpt has freq_decouple=False; knobs N/A.")
        return None
    enc = model.encoder
    load = transforms.Compose([transforms.Resize((args.size, args.size)),
                               transforms.ToTensor()])

    def _t(folder, name):
        return load(Image.open(os.path.join(_abs(folder), name)).convert("RGB")
                    ).unsqueeze(0).to(device)

    n = max(1, min(args.num, len(cnames), len(snames)))
    alphas = [float(x) for x in args.alphas.split(",")]

    def _style_dist(of, sf):
        d = 0.0
        for o, s in zip(of, sf):
            om, osd = calc_mean_std(o); sm, ssd = calc_mean_std(s)
            d += float(((om - sm) ** 2).mean() + ((osd - ssd) ** 2).mean())
        return d

    colour_y, stroke_y = [], []
    with torch.no_grad():
        for a in alphas:
            sc = ssd = 0.0
            for i in range(n):
                c = _t(args.content_dir, cnames[i]); s = _t(args.style_dir, snames[i])
                o_low = model(c, s, alpha_low=a, alpha_high=1.0).clamp(0, 1)
                o_high = model(c, s, alpha_low=1.0, alpha_high=a).clamp(0, 1)
                sf = enc(s)
                sc += _style_dist(enc(o_low), sf)
                try:
                    ssd += float(hf_stroke_loss(enc(o_high), sf))
                except Exception:                            # noqa: BLE001
                    ssd += _style_dist(enc(o_high), sf)
            colour_y.append(sc / n); stroke_y.append(ssd / n)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(alphas, colour_y, "o-", color="#4C72B0")
    a1.set_xlabel(r"$\alpha_\mathrm{low}$ (colour knob)", fontsize=10)
    a1.set_ylabel("Style-statistic distance", fontsize=9)
    a1.set_title("Colour control", fontsize=10); a1.grid(True, ls=":", alpha=0.5)
    a2.plot(alphas, stroke_y, "s-", color="#55A868")
    a2.set_xlabel(r"$\alpha_\mathrm{high}$ (stroke knob)", fontsize=10)
    a2.set_ylabel("High-frequency distance", fontsize=9)
    a2.set_title("Stroke control", fontsize=10); a2.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_controlcurve.pdf")


# ---- theoretical cost of the fusion block (no data needed) -------------------
def cmd_costcurve(args):
    """Closed-form cost of the fusion block vs joint token length N (T=L=N):
    softmax cross-attention is O(N^2 d), linear attention is O(N d^2). FDFormer
    runs the linear kernel once per band (x2). Computed in closed form."""
    import numpy as np
    d = float(args.dim)
    N = np.array([float(x) for x in args.sizes.split(",")], dtype=float)
    softmax = (N ** 2) * d           # O(N^2 d): full T x L score matrix
    one_band = N * (d ** 2)          # O(N d^2): key/value summary read by the query
    two_band = 2.0 * one_band        # FDFormer transfers two frequency bands
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(N, softmax, "o-", color="#EE6677", label=r"softmax cross-attn  $O(N^2 d)$")
    ax.loglog(N, two_band, "s-", color="#4477AA", label=r"FDFormer, two bands  $O(N d^2)$")
    ax.loglog(N, one_band, "^--", color="#228833", label=r"single linear band  $O(N d^2)$")
    ax.set_xlabel(r"token count $N$  ($N_c=N_s=N$)", fontsize=10)
    ax.set_ylabel("multiply–adds (arb. units)", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, args.out_dir, "fig_costcurve.pdf")


# ---- CLI ----------------------------------------------------------------------
def _add_common(p, need_imgs=True):
    p.add_argument("--out_dir", default=OUT_DEFAULT)
    if need_imgs:
        p.add_argument("--content_dir", default="eval/default/content")
        p.add_argument("--style_dir", default="eval/default/style")
        p.add_argument("--rows", type=int, default=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("overview"); po.add_argument("--out_dir", default=OUT_DEFAULT)
    po.set_defaults(fn=cmd_overview)

    pq = sub.add_parser("qualitative"); _add_common(pq)
    pq.add_argument("--map", nargs="*", help="override a column dir: 'AdaIN=/path'")
    pq.add_argument("--names", default=None,
                    help="comma list of pair filenames to show as rows (curated examples)")
    pq.set_defaults(fn=cmd_qualitative)

    pa = sub.add_parser("ablation"); _add_common(pa)
    pa.set_defaults(fn=cmd_ablation)

    pc = sub.add_parser("control")
    pc.add_argument("--out_dir", default=OUT_DEFAULT)
    pc.add_argument("--ckpt", default="experiments/fdformer_default/ckpt_080000.pth")
    pc.add_argument("--content_dir", default="eval/default/content")
    pc.add_argument("--style_dir", default="eval/default/style")
    pc.add_argument("--size", type=int, default=256)
    pc.add_argument("--pair_index", type=int, default=0)
    pc.add_argument("--alphas", default="0.5,1.0,1.5",
                    help="comma list used for BOTH alpha_low (rows) and alpha_high (cols)")
    pc.add_argument("--device", default="cuda")
    pc.set_defaults(fn=cmd_control)

    pe = sub.add_parser("efficiency")
    pe.add_argument("--out_dir", default=OUT_DEFAULT)
    pe.add_argument("--results", default="results.json")
    pe.set_defaults(fn=cmd_efficiency)

    pch = sub.add_parser("charts")
    pch.add_argument("--out_dir", default=OUT_DEFAULT)
    pch.add_argument("--results", default="results.json")
    pch.set_defaults(fn=cmd_charts)

    pcc = sub.add_parser("controlcurve")
    pcc.add_argument("--out_dir", default=OUT_DEFAULT)
    pcc.add_argument("--ckpt", default="experiments/fdformer_default/ckpt_080000.pth")
    pcc.add_argument("--content_dir", default="eval/default/content")
    pcc.add_argument("--style_dir", default="eval/default/style")
    pcc.add_argument("--size", type=int, default=256)
    pcc.add_argument("--num", type=int, default=8, help="pairs averaged per alpha")
    pcc.add_argument("--alphas", default="0,0.5,1.0,1.5,2.0")
    pcc.add_argument("--device", default="cuda")
    pcc.set_defaults(fn=cmd_controlcurve)

    pcv = sub.add_parser("costcurve")
    pcv.add_argument("--out_dir", default=OUT_DEFAULT)
    pcv.add_argument("--dim", type=int, default=256, help="channel width d")
    pcv.add_argument("--sizes", default="256,512,1024,2048,4096,8192",
                     help="joint token lengths N to plot")
    pcv.set_defaults(fn=cmd_costcurve)

    pall = sub.add_parser("all"); _add_common(pall)
    pall.add_argument("--results", default="results.json")
    pall.add_argument("--ckpt", default="experiments/fdformer_default/ckpt_080000.pth")
    pall.add_argument("--device", default="cuda")
    pall.set_defaults(fn=None)

    args = ap.parse_args()

    if args.cmd == "all":
        cmd_overview(args)
        cmd_qualitative(argparse.Namespace(out_dir=args.out_dir,
            content_dir=args.content_dir, style_dir=args.style_dir,
            rows=args.rows, map=None))
        cmd_ablation(args)
        cmd_efficiency(argparse.Namespace(out_dir=args.out_dir, results=args.results))
        cmd_control(argparse.Namespace(out_dir=args.out_dir, ckpt=args.ckpt,
            content_dir=args.content_dir, style_dir=args.style_dir, size=256,
            pair_index=0, alphas="0.5,1.0,1.5", device=args.device))
        cmd_charts(argparse.Namespace(out_dir=args.out_dir, results=args.results))
        cmd_controlcurve(argparse.Namespace(out_dir=args.out_dir, ckpt=args.ckpt,
            content_dir=args.content_dir, style_dir=args.style_dir, size=256,
            num=8, alphas="0,0.5,1.0,1.5,2.0", device=args.device))
        cmd_costcurve(argparse.Namespace(out_dir=args.out_dir, dim=256,
            sizes="256,512,1024,2048,4096,8192"))
        return
    args.fn(args)


if __name__ == "__main__":
    main()
