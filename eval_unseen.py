"""A test on photographs and styles that training never saw.

Every benchmark and validation image comes from the collections training draws from. This
script builds a set from outside them:

  content: MS-COCO val2017, which training does not use, taken in sorted order;
  style:   the example style images shipped with the baseline repositories, kept only if
           they are not in the WikiArt training tree.

A candidate style is dropped if it is the same artwork as a WikiArt training image or as
another candidate. Three pieces of evidence are recorded for each candidate: an exact
SHA-256 match, the highest correlation of its 32x32 greyscale thumbnail with any training
thumbnail (0.85 or above drops it), and its five nearest training images by the frozen
VGG-19 relu4_1 mean and standard deviation, which survive changes of size, crop and colour
that defeat the thumbnail. The nearest neighbours are then inspected and every judgement
of "same artwork" is written to manifests/unseen_style_decisions.json with its reason, so
that the exclusions can be checked. A different work by an artist who also appears in
WikiArt is kept: the image itself is unseen. Pair i joins content i with kept style i mod S.

    python eval_unseen.py check --style_dirs <dir> [<dir> ...]
    python eval_unseen.py prepare --out work/unseen --num 200
    python eval_unseen.py run --root work/unseen
    python eval_unseen.py score --root work/unseen      # also runs per_style

`score` reports content LPIPS and the VGG style distance through eval_uncertainty.py and
LP-SWD through eval_style_independent.py, with paired tests against FDFormer and Holm's
correction over the whole family. ArtFID is not reported: with only S distinct styles its
Frechet term would compare a handful of images repeated many times.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

BASE = os.environ.get("FDFORMER_BASELINES", "baselines")
CKPT = "experiments/fdformer_v2/ckpt_080000.pth"
WIKIART = "datasets/wikiart"
VAL = "datasets/coco/val2017"
THRESH = 0.85
EXTS = (".jpg", ".jpeg", ".png")
WRAPPERS = {
    "sanet": ("SANET", "sanet_paired.py", ["--vgg", "vgg_normalised.pth", "--decoder",
              "decoder_iter_500000.pth", "--transform", "transformer_iter_500000.pth"]),
    "adaattn": ("AdaAttN", "adaattn_paired.py", ["--image_encoder_path", "checkpoints/vgg_normalised.pth"]),
    "stytr2": ("StyTR-2", "stytr2_paired.py", []),
    "microast": ("MicroAST", "microast_paired.py", []),
    "aespanet": ("AesPA-Net", "aespanet_paired.py", ["--vgg", "SANET_VGG"]),
    "samam": ("SaMam/TEST", "samam_paired.py", ["--model_ckpt", "./checkpoint/SaMam_patchsize_x8.ckpt"]),
}
METHODS = ["fdformer", "fdformer_a140"] + list(WRAPPERS)


def thumb_and_hash(path):
    data = open(path, "rb").read()
    try:
        img = Image.open(path)
        img.draft("L", (64, 64))
        t = np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    except Exception:  # noqa: BLE001  unreadable files cannot match anything
        t = None
    return hashlib.sha256(data).hexdigest(), t


def standardise(t):
    t = t - t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def candidates(style_dirs):
    files = sorted(f for d in style_dirs for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(EXTS))
    by_hash = {}
    for f in files:
        by_hash.setdefault(hashlib.sha256(open(f, "rb").read()).hexdigest(), f)
    return len(files), sorted(by_hash.values())


def label(path):
    """Path relative to the baselines folder, which names the repository it came from."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(BASE))
    return (os.path.basename(path) if rel.startswith("..") else rel).replace(os.sep, "/")


class _Images:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torchvision.transforms.functional as TF
        try:
            img = Image.open(self.paths[i])
            img.draft("RGB", (448, 448))
            img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
            return TF.to_tensor(img), 1
        except Exception:  # noqa: BLE001
            import torch
            return torch.zeros(3, 224, 224), 0


def vgg_descriptors(paths, workers):
    """relu4_1 channel mean and standard deviation, L2-normalised."""
    import torch
    from torch.utils.data import DataLoader
    from losses.losses import calc_mean_std
    from models.vgg import VGGEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = VGGEncoder(pretrained=True).to(device).eval()
    out, ok = [], []
    for x, good in DataLoader(_Images(paths), batch_size=32, num_workers=workers):
        with torch.no_grad():
            m, s = calc_mean_std(enc(x.to(device))[-1])
            d = torch.cat([m.flatten(1), s.flatten(1)], 1)
            out.append(torch.nn.functional.normalize(d, dim=1).cpu())
        ok.append(good)
    return torch.cat(out).numpy(), torch.cat(ok).numpy().astype(bool)


def training_index(workers):
    # About 340 MB at float16, so it can be placed on another drive with FDFORMER_WORK.
    cache = os.path.join(os.environ.get("FDFORMER_WORK", "work"), "wikiart_index.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return (list(z["paths"]), list(z["hashes"]), z["thumbs"].astype(np.float32),
                z["vgg"].astype(np.float32), z["ok"])
    train = sorted(f for f in glob.glob(os.path.join(WIKIART, "**", "*"), recursive=True)
                   if f.lower().endswith(EXTS))
    print(f"indexing {len(train)} training images", flush=True)
    with Pool(workers) as pool:
        res = pool.map(thumb_and_hash, train, chunksize=64)
    hashes = [h for h, _ in res]
    thumbs = np.stack([standardise(t) if t is not None else np.zeros(1024, np.float32) for _, t in res])
    vgg, ok = vgg_descriptors(train, workers)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, paths=np.array(train, dtype=object), hashes=np.array(hashes, dtype=object),
             thumbs=thumbs.astype(np.float16), vgg=vgg.astype(np.float16), ok=ok)
    return train, hashes, thumbs, vgg, ok


def check(args):
    n_files, cands = candidates(args.style_dirs)
    train, hashes, thumbs, vgg, ok = training_index(args.workers)
    by_hash = dict(zip(hashes, train))
    cv, _ = vgg_descriptors(cands, 0)
    report = []
    sheet_dir = os.path.join(args.review_dir)
    os.makedirs(sheet_dir, exist_ok=True)
    for i, f in enumerate(cands):
        h, t = thumb_and_hash(f)
        corr = thumbs @ standardise(t)
        sim = np.where(ok, vgg @ cv[i], -1.0)
        top = np.argsort(-sim)[:5]
        others = [(float(cv[i] @ cv[j]), label(cands[j])) for j in range(len(cands)) if j != i]
        report.append({
            "file": label(f), "sha256": h,
            "exact_match": os.path.relpath(by_hash[h], WIKIART) if h in by_hash else None,
            "max_thumbnail_corr": float(corr.max()),
            "thumbnail_nearest": os.path.relpath(train[int(corr.argmax())], WIKIART),
            "vgg_nearest": [{"image": os.path.relpath(train[j], WIKIART), "cosine": float(sim[j])} for j in top],
            "nearest_other_candidate": max(others)[1], "nearest_other_candidate_cosine": max(others)[0],
            "auto_drop": h in by_hash or float(corr.max()) >= THRESH})
        tiles = [Image.open(f).convert("RGB").resize((256, 256))]
        tiles += [Image.open(train[j]).convert("RGB").resize((256, 256)) for j in top[:4]]
        sheet = Image.new("RGB", (256 * len(tiles), 256), "white")
        for k, im in enumerate(tiles):
            sheet.paste(im, (256 * k, 0))
        sheet.save(os.path.join(sheet_dir, f"{i:02d}_{os.path.splitext(os.path.basename(f))[0]}.jpg"), quality=85)
    os.makedirs("results", exist_ok=True)
    json.dump({"threshold_thumbnail": THRESH, "training_images": len(train), "candidate_files": n_files,
               "distinct": len(cands), "styles": report}, open("results/unseen_style_check.json", "w"), indent=2)
    print(f"{len(cands)} candidates checked; review sheets in {sheet_dir}")


def prepare(args):
    chk = json.load(open("results/unseen_style_check.json"))
    dec = json.load(open("manifests/unseen_style_decisions.json", encoding="utf-8"))
    _, cands = candidates(args.style_dirs)
    by_label = {label(f): f for f in cands}
    kept = []
    for s in chk["styles"]:
        d = dec[s["file"]]
        s["same_artwork_as"], s["reason"] = d["same_artwork_as"], d["reason"]
        s["kept"] = not s["auto_drop"] and d["same_artwork_as"] is None
        if s["kept"]:
            kept.append(by_label[s["file"]])
    chk["kept"] = len(kept)
    print(f"kept {len(kept)} of {len(cands)} styles", flush=True)
    if not kept:
        raise SystemExit("no style survives the check")
    contents = sorted(f for f in os.listdir(VAL) if f.lower().endswith(EXTS))[:args.num]
    c_dir, s_dir = os.path.join(args.out, "content"), os.path.join(args.out, "style")
    os.makedirs(c_dir, exist_ok=True)
    os.makedirs(s_dir, exist_ok=True)
    rows = ["pair\tcontent_coco_val2017\tcontent_sha256\tstyle\tstyle_sha256"]
    for i, c in enumerate(contents):
        s = kept[i % len(kept)]
        name = f"{i:05d}.png"
        Image.open(os.path.join(VAL, c)).convert("RGB").save(os.path.join(c_dir, name))
        Image.open(s).convert("RGB").save(os.path.join(s_dir, name))
        rows.append("\t".join([f"{i:05d}", c, hashlib.sha256(open(os.path.join(VAL, c), "rb").read()).hexdigest(),
                               label(s), hashlib.sha256(open(s, "rb").read()).hexdigest()]))
    open("manifests/unseen_pairs.tsv", "w", encoding="utf-8").write("\n".join(rows) + "\n")
    chk["pairs"] = len(contents)
    json.dump(chk, open("results/unseen_style_check.json", "w"), indent=2)


def run(args):
    py = sys.executable
    c = os.path.abspath(os.path.join(args.root, "content"))
    s = os.path.abspath(os.path.join(args.root, "style"))
    for m in args.methods:
        out = os.path.abspath(os.path.join(args.root, "out", m))
        if m.startswith("fdformer"):
            a = "1.4" if m.endswith("a140") else "1.0"
            cmd = [py, "batch_stylize.py", "--ckpt", CKPT, "--content_dir", c, "--style_dir", s,
                   "--out_dir", out, "--size", "256", "--alpha_low", a, "--alpha_high", a]
            cwd = "."
        else:
            d, script, extra = WRAPPERS[m]
            extra = [os.path.join(BASE, "SANET", "vgg_normalised.pth") if e == "SANET_VGG" else e for e in extra]
            os.makedirs(os.path.join(out, "stylized"), exist_ok=True)
            cmd = [py, script, "--content_dir", c, "--style_dir", s,
                   "--out_dir", os.path.join(out, "stylized"), "--size", "256", *extra]
            cwd = os.path.join(BASE, d)
        print(">>", m, flush=True)
        if subprocess.call(cmd, cwd=cwd) != 0:
            raise SystemExit(f"{m} failed")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run_max = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run_max
    return adj


def score(args):
    py = sys.executable
    root = os.path.abspath(args.root)
    c, s = os.path.join(root, "content"), os.path.join(root, "style")
    out = lambda m: os.path.join(root, "out", m)
    # Every method is scored against the same original-resolution content and style files;
    # batch_stylize's own resized copies are replaced so that no method is scored differently.
    for m in METHODS:
        for sub, src in (("content", c), ("style", s)):
            dst = os.path.join(out(m), sub)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                Image.open(os.path.join(src, f)).save(os.path.join(dst, f))
    for ref in ("fdformer", "fdformer_a140"):
        subprocess.check_call([py, "eval_uncertainty.py", "--reference", ref, "--skip_artfid",
                               "--methods", *[f"{m}={out(m)}" for m in METHODS],
                               "--json_out", f"results/unseen_pairlevel_{ref}.json"])
    subprocess.check_call([py, "eval_style_independent.py", "--style_dir", s,
                           "--methods", *[f"{m}={os.path.join(out(m), 'stylized')}" for m in METHODS],
                           "--json_out", "results/unseen_lpswd.json"])
    tests = []
    for ref in ("fdformer", "fdformer_a140"):
        u = json.load(open(f"results/unseen_pairlevel_{ref}.json"))
        pv = u["paired_vs_" + ref]
        for b in WRAPPERS:
            for k in ("lpips", "style"):
                tests.append({"ref": ref, "baseline": b, "metric": k, **pv[b][k]})
    l = json.load(open("results/unseen_lpswd.json"))
    for ref in ("fdformer", "fdformer_a140"):
        for b in WRAPPERS:
            tests.append({"ref": ref, "baseline": b, "metric": "lp_swd", **l["paired"][ref][b]})
    for t, a in zip(tests, holm([t["wilcoxon_p"] for t in tests])):
        t["holm_p"] = a
    json.dump({"family": len(tests), "tests": tests}, open("results/unseen_paired.json", "w"), indent=2)
    print(f"{len(tests)} paired tests written to results/unseen_paired.json")
    per_style(args)


def per_style(args):
    """In how many of the kept styles is FDFormer ahead of each baseline, on the style means.

    Pairs that share a style are not independent, so the pair-level intervals understate
    the uncertainty about styles; this count is the coarser, style-level check.
    """
    rows = [l.rstrip("\n").split("\t") for l in open("manifests/unseen_pairs.tsv", encoding="utf-8")][1:]
    style_of = {r[0]: r[3] for r in rows}
    styles = sorted(set(style_of.values()))
    rec = {"styles": styles, "counts": {}}
    lp = np.load("results/unseen_lpswd_per_pair.npz", allow_pickle=True)
    for ref in ("fdformer", "fdformer_a140"):
        pl = np.load(f"results/unseen_pairlevel_{ref}_per_pair.npz", allow_pickle=True)
        idx = {s: [i for i, n in enumerate(pl["names"]) if style_of[os.path.splitext(str(n))[0]] == s] for s in styles}
        idx_l = {s: [i for i, n in enumerate(lp["names"]) if style_of[os.path.splitext(str(n))[0]] == s] for s in styles}
        for b in WRAPPERS:
            for metric, ours, theirs, ix in (("lpips", pl[ref + "__lpips"], pl[b + "__lpips"], idx),
                                             ("style", pl[ref + "__style"], pl[b + "__style"], idx),
                                             ("lp_swd", lp[ref], lp[b], idx_l)):
                d = [float(ours[ix[s]].mean() - theirs[ix[s]].mean()) for s in styles]
                rec["counts"].setdefault(ref, {}).setdefault(b, {})[metric] = {
                    "styles_ahead": sum(x < 0 for x in d), "of": len(styles), "per_style_diff": d}
    json.dump(rec, open("results/unseen_per_style.json", "w"), indent=2)
    for ref, by_b in rec["counts"].items():
        print(ref)
        for b, by_m in by_b.items():
            print(f"  {b:9s} " + "  ".join(f"{m} {v['styles_ahead']}/{v['of']}" for m, v in by_m.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("check")
    k.add_argument("--style_dirs", nargs="+", required=True)
    k.add_argument("--workers", type=int, default=8)
    k.add_argument("--review_dir", default="work/unseen_review")
    p = sub.add_parser("prepare")
    p.add_argument("--out", default="work/unseen")
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--style_dirs", nargs="+", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", default="work/unseen")
    r.add_argument("--methods", nargs="+", default=METHODS)
    q = sub.add_parser("score")
    q.add_argument("--root", default="work/unseen")
    sub.add_parser("per_style")
    a = ap.parse_args()
    {"check": check, "prepare": prepare, "run": run, "score": score, "per_style": per_style}[a.cmd](a)


if __name__ == "__main__":
    main()
