"""Do the two branches compute different correspondences, and of what kind?

Linear attention has no stored score matrix, but Eq. (linattn) still defines one. For a
head, the weight content token i places on style token j is

    A[i, j] = phi(q_i)^T phi(k_j) / sum_l phi(q_i)^T phi(k_l)

which this script materialises at 256x256 (N = 1024 per image) from the trained weights
at the inference setting. It reports:

  how concentrated each branch's attention is (effective tokens = exp(entropy)), since a
  near-uniform map is not a correspondence of any kind;
  how much the two branches differ: Jensen-Shannon divergence between A_low and A_high
  for matched queries, against two references, the divergence between heads within a
  branch, and between a branch and uniform attention;
  what each branch selects: for each query, the expected colour and texture distance
  between its content patch and the style patches attended to, divided by the same
  expectation under uniform attention. A ratio below 1 means the branch picks style
  regions more similar on that property than chance. The property values come from the
  images (mean Lab chroma of a patch, log high-pass luminance energy of a patch), not
  from the network.

Usage
-----
python analyze_correspondence.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --content_dir eval/default/content --style_dir eval/default/style --num 300 \
    --json_out results/correspondence.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon
from skimage.color import rgb2lab

from models import FDFormer

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_PATCH = 8    # relu4_1 has stride 8


def load(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def patch_properties(rgb):
    """Per-token chroma (2) and texture (1) descriptors of an image, from pixels alone."""
    lab = rgb2lab(rgb).astype(np.float32)
    t = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0)
    chroma = F.avg_pool2d(t[:, 1:], _PATCH)
    lum = t[:, :1]
    hp = lum - F.avg_pool2d(F.pad(lum, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    tex = torch.log(F.avg_pool2d(hp ** 2, _PATCH) + 1e-3)
    return chroma.flatten(2)[0].T, tex.flatten(2)[0].T     # N x 2, N x 1


def branch_attention(attn, content_band, style_band):
    """Induced attention per head, [heads, Nq, Nk], exactly as the kernel computes it."""
    cf = content_band + attn.pe(content_band)
    sf = style_band + attn.pe(style_band)
    q = F.elu(attn._to_heads(attn.to_q(cf)).double())[0] + 1.0
    k = F.elu(attn._to_heads(attn.to_k(sf)).double())[0] + 1.0
    scores = q @ k.transpose(1, 2)
    return scores / scores.sum(dim=2, keepdim=True)


def entropy(p):
    return -(p * (p + 1e-30).log()).sum(dim=-1)


def js(p, q):
    m = 0.5 * (p + q)
    return entropy(m) - 0.5 * (entropy(p) + entropy(q))


def selection_ratio(A, content_prop, style_prop):
    """Expected property distance under A divided by that under uniform attention."""
    dist = torch.cdist(content_prop.double(), style_prop.double())
    return float((A * dist).sum(dim=1).mean() / dist.mean())


def summarise(x, boots=10000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (boots, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci_low": float(np.percentile(m, 2.5)),
            "ci_high": float(np.percentile(m, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--content_dir", required=True)
    ap.add_argument("--style_dir", required=True)
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = ckpt["cfg"]["model"]
    model = FDFormer(embed_dim=mc["embed_dim"], heads=mc["heads"], pretrained_vgg=True,
                     freq_decouple=mc.get("freq_decouple", True), use_gate=mc.get("use_gate", True),
                     freq_mode=mc.get("freq_mode", "learnable"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    names = sorted(f for f in os.listdir(args.content_dir) if f.lower().endswith(_EXTS))
    names = [n for n in names if os.path.exists(os.path.join(args.style_dir, n))][:args.num]

    keys = ["eff_low", "eff_high", "js_between", "js_within_low", "js_within_high",
            "js_low_uniform", "js_high_uniform", "colour_low", "colour_high",
            "texture_low", "texture_high", "same_top1"]
    rows = {k: [] for k in keys}
    with torch.no_grad():
        for idx, n in enumerate(names):
            c_np = load(os.path.join(args.content_dir, n), args.size)
            s_np = load(os.path.join(args.style_dir, n), args.size)
            c = torch.from_numpy(c_np).permute(2, 0, 1).unsqueeze(0).to(device)
            s = torch.from_numpy(s_np).permute(2, 0, 1).unsqueeze(0).to(device)
            cd, sd = model.encoder.encode_deepest(c), model.encoder.encode_deepest(s)
            cl, ch = model.freq(cd)
            sl, sh = model.freq(sd)
            Al = branch_attention(model.attn_low, cl, sl)
            Ah = branch_attention(model.attn_high, ch, sh)
            heads, Nq, Nk = Al.shape
            uni = torch.full_like(Al, 1.0 / Nk)

            rows["eff_low"].append(float(entropy(Al).exp().mean()))
            rows["eff_high"].append(float(entropy(Ah).exp().mean()))
            # Heads of different branches are not aligned, so compare the head-averaged map.
            ml, mh = Al.mean(dim=0), Ah.mean(dim=0)
            rows["js_between"].append(float(js(ml, mh).mean()))
            pairs = [(a, b) for a in range(heads) for b in range(a + 1, heads)]
            rows["js_within_low"].append(float(np.mean([float(js(Al[a], Al[b]).mean()) for a, b in pairs])))
            rows["js_within_high"].append(float(np.mean([float(js(Ah[a], Ah[b]).mean()) for a, b in pairs])))
            rows["js_low_uniform"].append(float(js(ml, uni[0]).mean()))
            rows["js_high_uniform"].append(float(js(mh, uni[0]).mean()))
            rows["same_top1"].append(float((ml.argmax(1) == mh.argmax(1)).float().mean()))

            cc, ct = patch_properties(c_np)
            sc, st = patch_properties(s_np)
            cc, ct, sc, st = (x.to(device) for x in (cc, ct, sc, st))
            rows["colour_low"].append(selection_ratio(ml, cc, sc))
            rows["colour_high"].append(selection_ratio(mh, cc, sc))
            rows["texture_low"].append(selection_ratio(ml, ct, st))
            rows["texture_high"].append(selection_ratio(mh, ct, st))
            if (idx + 1) % 50 == 0:
                print(f"{idx + 1}/{len(names)}", flush=True)

    rec = {"pairs": len(names), "tokens": int(Nk), "heads": int(heads),
           "max_js_nats": float(np.log(2)), "ckpt_step": ckpt.get("step")}
    rec["summary"] = {k: summarise(v) for k, v in rows.items()}
    a = {k: np.array(v) for k, v in rows.items()}
    rec["tests"] = {}
    for name, x, y in [("colour_low_vs_high", a["colour_low"], a["colour_high"]),
                       ("texture_low_vs_high", a["texture_low"], a["texture_high"]),
                       ("js_between_vs_within_low", a["js_between"], a["js_within_low"]),
                       ("js_between_vs_within_high", a["js_between"], a["js_within_high"])]:
        rec["tests"][name] = {**summarise(x - y), "wilcoxon_p": float(wilcoxon(x - y).pvalue)}
    # Specialisation would mean the low branch is relatively better on colour and the high
    # branch relatively better on texture, which this interaction measures directly.
    inter = (a["colour_low"] - a["colour_high"]) - (a["texture_low"] - a["texture_high"])
    rec["tests"]["specialisation_interaction"] = {**summarise(inter),
                                                  "wilcoxon_p": float(wilcoxon(inter).pvalue)}
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec["summary"], indent=1))
    print(json.dumps(rec["tests"], indent=1))


if __name__ == "__main__":
    main()
