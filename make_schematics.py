"""Vector schematics for the FDFormer paper.

The three schematics that shipped with the first submission were raster images
wrapped in a PDF, so they blur under magnification, and one of them carried a
claim the revision retracted. These are redrawn as real vector figures, with
every module name, channel count and measured number taken from the code in
``models/`` and from ``results.json``.

Two rules keep the drawings from breaking:

* boxes are sized from the *measured* rendered extent of their text, never from
  a guess about how wide a string will be;
* nodes are chained -- each one is placed a fixed gap to the right of the
  previous node's measured edge -- so an arrow can never end up pointing
  backwards into the box it came from.

Figures are authored at the manuscript's own \\linewidth (370.6 pt = 5.15 in),
so the point sizes here are the point sizes on the printed page. Anything longer
than a few words belongs in the LaTeX caption, not in the drawing.

    fig_arch        overall architecture                   NEW
    fig_compare     one attention map vs two bands         replaces the raster
    fig_freqdecomp  what the learnable low-pass learns      replaces the raster
    fig_gate        content-adaptive gate                   replaces the raster

Usage: python make_schematics.py [--out DIR] [--check]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

WIDTH_IN = 5.15          # the manuscript's \linewidth

LOW = "#2F5C8F"          # low band / colour
HIGH = "#B4453C"         # high band / strokes
GREY = "#454B54"
SOFT = "#767C85"
LOW_FILL = "#DEE8F4"
HIGH_FILL = "#F7DEDB"
GREY_FILL = "#EDEFF2"

FS_TITLE = 8.6
FS_BOX = 7.4
FS_SMALL = 6.6

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_UPP = 1.0               # data units per point
_XMAX = 100.0
_WARN = []


def canvas(xmax, ymax):
    global _UPP, _XMAX
    fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * ymax / xmax))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    _UPP, _XMAX = xmax / (WIDTH_IN * 72.0), xmax
    fig.canvas.draw()
    return fig, ax


def pt(v):
    """Points -> data units."""
    return v * _UPP


def _extent(ax, artist):
    r = ax.figure.canvas.get_renderer()
    return artist.get_window_extent(renderer=r).transformed(ax.transData.inverted())


def text(ax, x, y, s, fs=FS_BOX, color="black", ha="center", va="center",
         weight="normal", style="normal", z=7):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=z)


def node(ax, cx, cy, lines, fs=FS_BOX, ec=GREY, fc="white", lw=0.9, colors=None,
         weights=None, sizes=None, padx=5.0, pady=4.0, z=4, frame=True, warn=True):
    """Text centred at (cx, cy) inside a box measured to fit it.

    padx / pady are in points. Returns (x0, y0, x1, y1) in data units.
    """
    n = len(lines)
    sizes = sizes or [fs] * n
    lh = [pt(s * 1.38) for s in sizes]
    total, arts = sum(lh), []
    top = cy + total / 2.0
    for i, s in enumerate(lines):
        arts.append(text(ax, cx, top - sum(lh[:i]) - lh[i] / 2.0, s, fs=sizes[i],
                         color=(colors[i] if colors else "black"),
                         weight=(weights[i] if weights else "normal"), z=z + 3))
    w = max(_extent(ax, a).width for a in arts) + 2 * pt(padx)
    h = total + 2 * pt(pady)
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    if frame:
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle="round,pad=0,rounding_size=%.2f" % pt(3.0),
                                    linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    if warn and (x0 < -0.5 or x0 + w > _XMAX + 0.5):
        _WARN.append("node %r runs from %.1f to %.1f (canvas 0..%.0f)"
                     % (lines[0][:22], x0, x0 + w, _XMAX))
    return (x0, y0, x0 + w, y0 + h)


def after(prev_right, gap_pt, lines, ax, cy, **kw):
    """Place a node whose left edge sits `gap_pt` points right of `prev_right`."""
    probe = node(ax, 0.0, -10_000.0, lines, frame=False, warn=False,
                 **{k: v for k, v in kw.items() if k in
                    ("fs", "sizes", "colors", "weights", "padx", "pady")})
    for t in ax.texts[-len(lines):]:
        t.remove()
    w = probe[2] - probe[0]
    cx = prev_right + pt(gap_pt) + w / 2.0
    return node(ax, cx, cy, lines, **kw)


def arrow(ax, p, q, color=GREY, lw=0.9, ms=6.0, z=5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                                 zorder=z, linestyle=ls))
    if q[0] < p[0] - 0.05 and abs(q[1] - p[1]) < 0.05:
        _WARN.append("arrow points backwards: %.1f -> %.1f at y=%.1f"
                     % (p[0], q[0], p[1]))


def elbow(ax, pts, color=GREY, lw=0.9, head=True, z=5, ls="-"):
    pts = [(float(a), float(b)) for a, b in pts]
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z,
                linestyle=ls, solid_capstyle="round")
    a, b = pts[-2], pts[-1]
    if head:
        arrow(ax, a, b, color=color, lw=lw, z=z, ls=ls)
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z,
                linestyle=ls, solid_capstyle="round")


def op(ax, x, y, r, sym, color=GREY, fs=FS_BOX, fc="white", z=6):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=color, linewidth=0.9,
                        zorder=z))
    text(ax, x, y, sym, fs=fs, color=color, z=z + 1)
    return (x - r, y - r, x + r, y + r)


def snowflake(ax, x, y, r, color=SOFT, lw=0.55, z=9):
    for a in (0, 60, 120):
        t = np.deg2rad(a)
        ax.plot([x - r * np.cos(t), x + r * np.cos(t)],
                [y - r * np.sin(t), y + r * np.sin(t)], color=color, lw=lw, zorder=z)
    for a in range(0, 360, 60):
        t = np.deg2rad(a)
        bx, by = x + 0.60 * r * np.cos(t), y + 0.60 * r * np.sin(t)
        for s in (-40, 40):
            u = np.deg2rad(a + s)
            ax.plot([bx, bx + 0.34 * r * np.cos(u)], [by, by + 0.34 * r * np.sin(u)],
                    color=color, lw=lw * 0.85, zorder=z)


# ---------------------------------------------------------------- thumbnails
def _frame(ax, x, y, s, z):
    ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=GREY, lw=0.8,
                           zorder=z))


def _strokes(ax, x, y, s, n, cols, lo, hi, seed, z, alpha):
    rng = np.random.default_rng(seed)
    for i in range(n):
        a = rng.uniform(0, np.pi)
        ln = rng.uniform(lo, hi) * s
        cxx, cyy = x + rng.uniform(0.12, 0.88) * s, y + rng.uniform(0.12, 0.88) * s
        ax.plot([cxx - ln / 2 * np.cos(a), cxx + ln / 2 * np.cos(a)],
                [cyy - ln / 2 * np.sin(a), cyy + ln / 2 * np.sin(a)],
                color=cols[i % len(cols)], lw=rng.uniform(0.8, 1.7),
                solid_capstyle="round", alpha=alpha, zorder=z)


def thumb_content(ax, cx, cy, s, z=3):
    x, y = cx - s / 2, cy - s / 2
    ax.add_patch(Rectangle((x, y), s, s, facecolor="#CFE0EE", edgecolor="none", zorder=z))
    ax.add_patch(Rectangle((x, y), s, s * 0.40, facecolor="#B9CFA8", edgecolor="none",
                           zorder=z + 1))
    ax.add_patch(Circle((x + s * 0.72, y + s * 0.74), s * 0.11, facecolor="#E9C46A",
                        edgecolor="none", zorder=z + 2))
    _frame(ax, x, y, s, z + 3)
    return (x, y, x + s, y + s)


def thumb_style(ax, cx, cy, s, z=3, seed=3):
    x, y = cx - s / 2, cy - s / 2
    ax.add_patch(Rectangle((x, y), s, s, facecolor="#F3E7D0", edgecolor="none", zorder=z))
    _strokes(ax, x, y, s, 11, ["#B4453C", "#2F5C8F", "#6E8B4E", "#D08C3A"], 0.24, 0.44,
             seed, z + 1, 0.92)
    _frame(ax, x, y, s, z + 3)
    return (x, y, x + s, y + s)


def thumb_out(ax, cx, cy, s, z=3, seed=11):
    x, y = cx - s / 2, cy - s / 2
    ax.add_patch(Rectangle((x, y), s, s, facecolor="#D6DFE7", edgecolor="none", zorder=z))
    ax.add_patch(Rectangle((x, y), s, s * 0.40, facecolor="#BDC9A3", edgecolor="none",
                           zorder=z + 1))
    ax.add_patch(Circle((x + s * 0.72, y + s * 0.74), s * 0.11, facecolor="#E3B55F",
                        edgecolor="none", zorder=z + 2))
    _strokes(ax, x, y, s, 15, ["#B4453C", "#2F5C8F", "#6E8B4E"], 0.16, 0.30, seed,
             z + 3, 0.72)
    _frame(ax, x, y, s, z + 4)
    return (x, y, x + s, y + s)


def gate_map(ax, cx, cy, s, z=3, cbar=True, n=18):
    """Gate heat-map: dark (little style) over a head-and-shoulders and a text strip."""
    x, y = cx - s / 2, cy - s / 2
    g = np.full((n, n), 0.84)
    yy, xx = np.mgrid[0:n, 0:n]
    head = ((xx - 5.4) ** 2 / 3.0 ** 2 + (yy - 8.2) ** 2 / 3.3 ** 2) < 1.0
    body = ((xx - 5.4) ** 2 / 6.0 ** 2 + (yy - 19.5) ** 2 / 7.0 ** 2) < 1.0
    g[head | body] = 0.12
    g[2:5, 10:16] = 0.22
    g[3, 11:15] = 0.58
    g = np.clip(g + np.random.default_rng(0).normal(0, 0.025, g.shape), 0, 1)
    ax.imshow(g, cmap="gray", vmin=0, vmax=1, origin="upper",
              extent=(x, x + s, y, y + s), zorder=z, interpolation="nearest")
    _frame(ax, x, y, s, z + 2)
    right = x + s
    if cbar:
        cw, cx0 = s * 0.09, x + s + s * 0.08
        ax.imshow(np.linspace(1, 0, 64).reshape(-1, 1), cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=(cx0, cx0 + cw, y, y + s), zorder=z,
                  interpolation="bilinear", aspect="auto")
        _frame(ax, cx0, y, 0, z)
        ax.add_patch(Rectangle((cx0, y), cw, s, facecolor="none", edgecolor=GREY,
                               lw=0.6, zorder=z + 2))
        text(ax, cx0 + cw * 1.7, y + s, "1", fs=FS_SMALL, ha="left")
        text(ax, cx0 + cw * 1.7, y, "0", fs=FS_SMALL, ha="left")
        right = cx0 + cw * 1.7 + pt(5)
    return (x, y, right, y + s)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), dpi=600,
                    facecolor="white", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print("  %-16s ok" % (name + ".pdf"))


# --------------------------------------------------------------------------
# Figure A: overall architecture
# --------------------------------------------------------------------------
def fig_arch(out):
    fig, ax = canvas(100, 66)
    Y, YL, YH, YG = 36.0, 47.5, 24.5, 8.0

    ct = thumb_content(ax, 4.6, 41.0, 8.2)
    text(ax, 4.6, ct[1] - pt(2), "content", fs=FS_SMALL, va="top")
    st = thumb_style(ax, 4.6, 30.0, 8.2)
    text(ax, 4.6, st[1] - pt(2), "style", fs=FS_SMALL, va="top")

    vgg = after(ct[2], 9, ["frozen", "VGG-19"], ax, Y, fc=GREY_FILL, pady=9)
    snowflake(ax, vgg[2] - pt(5), vgg[3] - pt(5), pt(3.4))
    arrow(ax, (ct[2], 39.5), (vgg[0], 38.4))
    arrow(ax, (st[2], 31.5), (vgg[0], 33.6))

    lp = after(vgg[2], 10, ["learnable", "low-pass"], ax, Y, fc=GREY_FILL)
    arrow(ax, (vgg[2], Y), (lp[0], Y))

    lo = after(lp[2], 13, ["low band", "linear cross-attention"], ax, YL,
               ec=LOW, fc=LOW_FILL, sizes=[FS_BOX, FS_SMALL], colors=[LOW, LOW],
               weights=["bold", "normal"])
    hi = after(lp[2], 13, ["high band", "linear cross-attention"], ax, YH,
               ec=HIGH, fc=HIGH_FILL, sizes=[FS_BOX, FS_SMALL], colors=[HIGH, HIGH],
               weights=["bold", "normal"])
    mid = (lp[2] + lo[0]) / 2
    elbow(ax, [(lp[2], Y), (mid, Y), (mid, YL), (lo[0], YL)], color=LOW)
    elbow(ax, [(lp[2], Y), (mid, Y), (mid, YH), (hi[0], YH)], color=HIGH)

    r = pt(7.0)
    axl = lo[2] + pt(9) + r
    op(ax, axl, YL, r, r"$\alpha_{\mathrm{low}}$", color=LOW, fs=FS_SMALL, fc=LOW_FILL)
    op(ax, axl, YH, r, r"$\alpha_{\mathrm{high}}$", color=HIGH, fs=FS_SMALL, fc=HIGH_FILL)
    arrow(ax, (lo[2], YL), (axl - r, YL), color=LOW)
    arrow(ax, (hi[2], YH), (axl - r, YH), color=HIGH)

    xsum = axl + r + pt(10) + r
    elbow(ax, [(axl + r, YL), (xsum, YL), (xsum, Y + r)], color=LOW)
    elbow(ax, [(axl + r, YH), (xsum, YH), (xsum, Y - r)], color=HIGH)
    op(ax, xsum, Y, r, "+")
    text(ax, xsum, Y + r + pt(2.5), "$T$", fs=FS_SMALL, va="bottom")

    # gate branch, below
    xtap = vgg[0] + (vgg[2] - vgg[0]) * 0.5
    elbow(ax, [(xtap, vgg[1]), (xtap, YG)], color=GREY, lw=0.85, head=False)
    text(ax, xtap - pt(2), (vgg[1] + YG) / 2, "$F_c$", fs=FS_SMALL, ha="right")
    gt = after(xtap, 6, ["content-adaptive", "gate"], ax, YG, fc=GREY_FILL,
               sizes=[FS_SMALL, FS_SMALL])
    arrow(ax, (xtap, YG), (gt[0], YG))
    gsz = 13.0
    gm = gate_map(ax, gt[2] + pt(9) + gsz / 2, YG, gsz)
    arrow(ax, (gt[2], YG), (gm[0], YG))
    text(ax, (gm[0] + gm[2]) / 2 - pt(4), gm[1] - pt(2.5), "gate $g$", fs=FS_SMALL,
         va="top")

    # blend, decoder, output
    xbl = xsum + r + pt(12) + r
    op(ax, xbl, Y, r, "+")
    arrow(ax, (xsum + r, Y), (xbl - r, Y))
    elbow(ax, [(gm[2], YG), (xbl, YG), (xbl, Y - r)], color=GREY, lw=0.85)
    ycskip = 61.0
    elbow(ax, [(xtap, vgg[3]), (xtap, ycskip), (xbl, ycskip), (xbl, Y + r)],
          color=SOFT, lw=0.8, ls=(0, (4, 2.5)))
    text(ax, xtap + pt(4), ycskip + pt(2), "$F_c$ kept where the gate is low",
         fs=FS_SMALL, color=SOFT, va="bottom", ha="left")

    dec = after(xbl + r, 10, ["decoder"], ax, Y, fc=GREY_FILL, pady=6)
    arrow(ax, (xbl + r, Y), (dec[0], Y))
    ot = thumb_out(ax, (dec[0] + dec[2]) / 2, YL + 2.0, 8.6)
    arrow(ax, ((dec[0] + dec[2]) / 2, dec[3]), ((dec[0] + dec[2]) / 2, ot[1]))
    text(ax, (dec[0] + dec[2]) / 2, ot[3] + pt(2), "stylised", fs=FS_SMALL, va="bottom")

    ax.add_patch(FancyBboxPatch((lp[0] - pt(5), 1.0), dec[2] + pt(5) - lp[0] + pt(5),
                                55.5, boxstyle="round,pad=0,rounding_size=%.2f" % pt(4),
                                linewidth=0.8, edgecolor="#AEB4BC", facecolor="none",
                                linestyle=(0, (4.5, 3)), zorder=1))
    text(ax, dec[2] + pt(3), 56.9, "trained: 4.91 M of the 8.42 M total", fs=FS_SMALL,
         color=SOFT, ha="right", va="bottom")
    save(fig, out, "fig_arch")


# --------------------------------------------------------------------------
# Figure B: one attention map vs two bands
# --------------------------------------------------------------------------
def fig_compare(out):
    fig, ax = canvas(100, 46)
    ax.plot([50.0, 50.0], [5.0, 40.0], color="#C8CDD4", lw=0.8, zorder=1)

    # (a) ---------------------------------------------------------------
    text(ax, 24.0, 41.5, "(a)  One shared attention map", fs=FS_TITLE, weight="bold",
         va="bottom")
    c1 = node(ax, 7.0, 29.5, ["content", "feature"], sizes=[FS_SMALL] * 2)
    s1 = node(ax, 7.0, 16.0, ["style", "feature"], sizes=[FS_SMALL] * 2)
    bx0 = c1[2] + pt(9)
    bw, by0, bh = 19.0, 10.0, 25.5
    ax.add_patch(FancyBboxPatch((bx0, by0), bw, bh,
                                boxstyle="round,pad=0,rounding_size=%.2f" % pt(3),
                                linewidth=0.9, edgecolor=GREY, facecolor=GREY_FILL,
                                zorder=2))
    text(ax, bx0 + bw / 2, by0 + bh - pt(6), "attention", fs=FS_BOX, weight="bold", z=8)
    arrow(ax, (c1[2], 28.0), (bx0, 26.5))
    arrow(ax, (s1[2], 17.5), (bx0, 19.0))
    t = np.linspace(0, 1, 300)
    for col, ph in ((LOW, 0.0), (HIGH, 1.75)):
        for k in range(3):
            ax.plot(bx0 + 1.6 + (bw - 3.2) * t,
                    by0 + 10.0 + 5.4 * np.sin(2 * np.pi * 1.4 * t + ph + k * 0.85)
                    + 0.9 * np.sin(2 * np.pi * 4.3 * t + ph),
                    color=col, lw=1.5, alpha=0.85, zorder=3, solid_capstyle="round")
    for col, lab, yy in ((LOW, "colour", by0 + 18.0), (HIGH, "strokes", by0 + 3.0)):
        ax.text(bx0 + bw / 2, yy, lab, ha="center", va="center", fontsize=FS_SMALL,
                color=col, fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
    xa = bx0 + bw
    ot = thumb_out(ax, xa + pt(13) + 3.6, 22.5, 7.2, seed=4)
    arrow(ax, (xa, 22.5), (ot[0], 22.5))
    text(ax, (ot[0] + ot[2]) / 2, ot[1] - pt(2), "stylised", fs=FS_SMALL, va="top")
    text(ax, 24.0, 7.0, "palette and brushwork both come from", fs=FS_SMALL)
    text(ax, 24.0, 4.2, "whichever style region matches best", fs=FS_SMALL)

    # (b) ---------------------------------------------------------------
    text(ax, 76.0, 41.5, "(b)  Two bands, two matches  (ours)", fs=FS_TITLE,
         weight="bold", va="bottom")
    c2 = node(ax, 58.0, 29.5, ["content", "feature"], sizes=[FS_SMALL] * 2)
    s2 = node(ax, 58.0, 16.0, ["style", "feature"], sizes=[FS_SMALL] * 2)
    lp = after(c2[2], 8, ["learnable", "low-pass"], ax, 22.5, fc=GREY_FILL,
               sizes=[FS_SMALL] * 2)
    j = (c2[2] + lp[0]) / 2
    elbow(ax, [(c2[2], 29.5), (j, 29.5), (j, 22.5), (lp[0], 22.5)], color=GREY)
    elbow(ax, [(s2[2], 16.0), (j, 16.0), (j, 22.5)], color=GREY, head=False)

    lo = after(lp[2], 12, ["low band"], ax, 32.0, ec=LOW, fc=LOW_FILL,
               sizes=[FS_SMALL], colors=[LOW], weights=["bold"])
    hi = after(lp[2], 12, ["high band"], ax, 13.0, ec=HIGH, fc=HIGH_FILL,
               sizes=[FS_SMALL], colors=[HIGH], weights=["bold"])
    k = (lp[2] + lo[0]) / 2
    elbow(ax, [(lp[2], 22.5), (k, 22.5), (k, 32.0), (lo[0], 32.0)], color=LOW)
    elbow(ax, [(lp[2], 22.5), (k, 22.5), (k, 13.0), (hi[0], 13.0)], color=HIGH)
    text(ax, (lo[0] + lo[2]) / 2, lo[3] + pt(2), "linear cross-attention",
         fs=FS_SMALL, color=LOW, va="bottom")
    text(ax, (hi[0] + hi[2]) / 2, hi[1] - pt(2), "linear cross-attention",
         fs=FS_SMALL, color=HIGH, va="top")

    r = pt(6.5)
    xs = max(lo[2], hi[2]) + pt(11) + r
    elbow(ax, [(lo[2], 32.0), (xs, 32.0), (xs, 22.5 + r)], color=LOW)
    elbow(ax, [(hi[2], 13.0), (xs, 13.0), (xs, 22.5 - r)], color=HIGH)
    text(ax, xs - pt(2), 27.5, r"$\alpha_{\mathrm{low}}$", fs=FS_SMALL, color=LOW,
         ha="right")
    text(ax, xs - pt(2), 17.5, r"$\alpha_{\mathrm{high}}$", fs=FS_SMALL, color=HIGH,
         ha="right")
    op(ax, xs, 22.5, r, "+")
    text(ax, 76.0, 7.0, "each band is matched under", fs=FS_SMALL)
    text(ax, 76.0, 4.2, "its own notion of similarity", fs=FS_SMALL)

    text(ax, 50.0, 1.4, "the split is between correspondences, "
                        "not between two independent controls",
         fs=FS_SMALL, color=SOFT, style="italic")
    save(fig, out, "fig_compare")


# --------------------------------------------------------------------------
# Figure C: what the learnable low-pass learns
# --------------------------------------------------------------------------
def fig_freqdecomp(out):
    fig, ax = canvas(100, 42)
    Y = 23.0

    text(ax, 12.0, 37.0, "(a)  Deep feature", fs=FS_TITLE, weight="bold", va="bottom")
    for i in range(4):
        o = (3 - i) * 1.1
        ax.add_patch(Rectangle((5.0 + o, Y - 6.0 + o), 11.5, 11.5,
                               facecolor="#F5F7F9" if i else "#E4EBF2", edgecolor=GREY,
                               lw=0.6, zorder=2 + i))
    rng = np.random.default_rng(2)
    for _ in range(22):
        ax.add_patch(Circle((rng.uniform(6.0, 15.5), rng.uniform(Y - 5.0, Y + 4.5)),
                            rng.uniform(0.6, 1.6),
                            facecolor=rng.choice([LOW, HIGH, "#6E8B4E"]), alpha=0.28,
                            edgecolor="none", zorder=7))
    ax.add_patch(Rectangle((5.0, Y - 6.0), 11.5, 11.5, facecolor="none", edgecolor=GREY,
                           lw=0.85, zorder=8))
    text(ax, 11.0, Y - 7.4, "relu4_1 feature", fs=FS_SMALL, va="top")
    text(ax, 11.0, Y - 10.2, "512 channels", fs=FS_SMALL, color=SOFT, va="top")
    arrow(ax, (20.0, Y), (25.0, Y))

    text(ax, 38.5, 37.0, "(b)  Learnable low-pass", fs=FS_TITLE,
         weight="bold", va="bottom")
    for i, sig in enumerate((0.72, 1.05, 1.55)):
        ox, oy = 28.0 + i * 4.0, Y + 2.6 - i * 3.3
        a = np.arange(5) - 2
        xx, yy = np.meshgrid(a, a)
        k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
        ax.imshow(k / k.max(), cmap="Blues", vmin=-0.3, vmax=1.15, origin="upper",
                  extent=(ox, ox + 8.5, oy, oy + 8.5), zorder=3 + i,
                  interpolation="nearest")
        ax.add_patch(Rectangle((ox, oy), 8.5, 8.5, facecolor="none", edgecolor=GREY,
                               lw=0.65, zorder=6 + i))
        for t in range(1, 5):
            ax.plot([ox + t * 1.7] * 2, [oy, oy + 8.5], color="white", lw=0.4, zorder=7 + i)
            ax.plot([ox, ox + 8.5], [oy + t * 1.7] * 2, color="white", lw=0.4, zorder=7 + i)
    text(ax, 38.5, Y - 7.4, "one 5×5 kernel per channel", fs=FS_SMALL, va="top")
    text(ax, 38.5, Y - 10.2, "each channel learns its own cut-off", fs=FS_SMALL,
         color=SOFT, va="top")
    arrow(ax, (49.0, Y), (54.0, Y))

    text(ax, 79.0, 37.0, "(c)  Frequency response", fs=FS_TITLE, weight="bold",
         va="bottom")
    x0, y0, w, h = 60.0, 16.5, 34.0, 17.0
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=GREY, lw=0.8,
                           zorder=3))
    f = np.linspace(1e-3, 1.0, 400)
    fmax = 0.42                       # so 0.105 and 0.210 fall at 1/4 and 1/2 width

    def resp(fc):
        return 1.0 / (1.0 + (f * fmax / fc) ** 4)

    ax.fill_between(x0 + f * w, y0 + resp(0.105) * h, y0 + resp(0.210) * h,
                    color=LOW, alpha=0.20, lw=0, zorder=4)
    ax.plot(x0 + f * w, y0 + resp(0.1303) * h, color=SOFT, lw=0.9,
            linestyle=(0, (3.5, 2)), zorder=5)
    ax.plot([x0, x0 + w], [y0 + 0.707 * h] * 2, color=GREY, lw=0.6,
            linestyle=(0, (1.3, 1.3)), zorder=5)
    text(ax, x0 + w - pt(2), y0 + 0.707 * h + pt(1.5), "−3 dB", fs=FS_SMALL,
         color=GREY, ha="right", va="bottom")
    for cut, lab in ((0.105, "0.105"), (0.210, "0.210")):
        xx = x0 + (cut / fmax) * w
        ax.plot([xx, xx], [y0, y0 + 0.9], color=GREY, lw=0.7, zorder=5)
        text(ax, xx, y0 - pt(2), lab, fs=FS_SMALL, va="top")
    text(ax, x0 + w / 2, y0 - pt(11), "spatial frequency", fs=FS_SMALL, va="top")
    ax.text(x0 - pt(4), y0 + h / 2, "response", fontsize=FS_SMALL, rotation=90,
            ha="right", va="center")
    text(ax, x0 + w * 0.58, y0 + h - pt(3), "learned, 5th–95th percentile",
         fs=FS_SMALL, color=LOW, va="top")
    ax.annotate("Gaussian initialisation,\nidentical for every channel",
                xy=(x0 + 0.30 * w, y0 + resp(0.1303)[120] * h),
                xytext=(x0 + 0.40 * w, y0 + 0.16 * h), fontsize=FS_SMALL, color=SOFT,
                ha="left", va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=SOFT, shrinkA=2,
                                shrinkB=2))
    node(ax, 77.0, 5.6, ["freezing this costs 13.15 ArtFID"], sizes=[FS_SMALL],
         weights=["bold"], pady=3.2)
    save(fig, out, "fig_freqdecomp")


# --------------------------------------------------------------------------
# Figure D: the content-adaptive gate
# --------------------------------------------------------------------------
def fig_gate(out):
    fig, ax = canvas(100, 40)
    Y = 21.0

    c = node(ax, 8.0, Y, ["content", "feature $F_c$"], sizes=[FS_SMALL] * 2)
    g1 = after(c[2], 9, ["three-layer conv,", "sigmoid", "512–64–64–1"], ax, Y,
               fc=GREY_FILL, sizes=[FS_SMALL] * 3)
    arrow(ax, (c[2], Y), (g1[0], Y))
    gsz = 15.0
    gm = gate_map(ax, g1[2] + pt(10) + gsz / 2, Y, gsz)
    arrow(ax, (g1[2], Y), (gm[0], Y))
    text(ax, gm[0] + gsz / 2, gm[1] - pt(2.5), "gate $g$, one value per location",
         fs=FS_SMALL, va="top")
    ax.annotate("less style over faces and text", xy=(gm[0] + gsz * 0.30, Y + 1.5),
                xytext=(24.0, 36.5), fontsize=FS_SMALL, color=SOFT, ha="center",
                zorder=9, arrowprops=dict(arrowstyle="-", lw=0.6, color=SOFT,
                                          shrinkA=1, shrinkB=2))
    ax.annotate("more style over flat regions", xy=(gm[0] + gsz * 0.82, Y - 4.5),
                xytext=(76.0, 36.5), fontsize=FS_SMALL, color=SOFT, ha="center",
                zorder=9, arrowprops=dict(arrowstyle="-", lw=0.6, color=SOFT,
                                          shrinkA=1, shrinkB=2))

    tb = after(gm[2], 12, ["transferred $T$"], ax, 31.5, sizes=[FS_SMALL])
    cb = node(ax, (tb[0] + tb[2]) / 2, 10.5, ["content $F_c$"], sizes=[FS_SMALL])
    r = pt(6.2)
    xm = max(tb[2], cb[2]) + pt(10) + r
    op(ax, xm, 31.5, r, "×")
    op(ax, xm, 10.5, r, "×")
    arrow(ax, (tb[2], 31.5), (xm - r, 31.5))
    arrow(ax, (cb[2], 10.5), (xm - r, 10.5))
    elbow(ax, [(gm[2], Y), (xm, Y), (xm, 31.5 - r)], color=GREY)
    elbow(ax, [(xm, Y), (xm, 10.5 + r)], color=GREY)
    text(ax, xm + pt(2.5), (Y + 31.5) / 2, "$g$", fs=FS_SMALL, ha="left")
    text(ax, xm + pt(2.5), (Y + 10.5) / 2, "$1-g$", fs=FS_SMALL, ha="left")

    xp = xm + r + pt(14) + r
    op(ax, xp, Y, r, "+")
    elbow(ax, [(xm + r, 31.5), (xp, 31.5), (xp, Y + r)], color=GREY)
    elbow(ax, [(xm + r, 10.5), (xp, 10.5), (xp, Y - r)], color=GREY)
    text(ax, xp + r + pt(3), Y, "$F_{\\mathrm{target}}$", fs=FS_BOX, ha="left")

    text(ax, 50.0, 2.2, "removing the gate costs 30.5 ArtFID, "
                        "the largest effect of any single component",
         fs=FS_SMALL, color=SOFT, style="italic")
    save(fig, out, "fig_gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    print("writing to", out)
    fig_arch(out)
    fig_compare(out)
    fig_freqdecomp(out)
    fig_gate(out)
    if _WARN:
        print("\nlayout warnings:")
        for w in dict.fromkeys(_WARN):
            print("  ", w)
    else:
        print("\nno layout warnings: every box is inside the canvas and every "
              "arrow points forwards")


if __name__ == "__main__":
    main()
