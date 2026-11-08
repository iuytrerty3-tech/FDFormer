"""Build a validation set of content-style pairs disjoint from the benchmark.

The 1000-pair benchmark is formed from the first 1000 MS-COCO train2017 images in
sorted order paired with the first 1000 images of datasets/style_flat, also in sorted
order. Any coefficient chosen by looking at that set would be chosen on the same data
the paper reports, which is exactly the objection a reader should raise. This script
therefore assembles a separate set:

  content  the next MS-COCO images in sorted order, after the 1000 the benchmark took
  style    WikiArt paintings whose file contents match no image in style_flat, so the
           exclusion holds even if the original sampling cannot be replayed exactly

Every candidate style image is hashed and compared against the hashes of the whole
benchmark style pool, so an accidental overlap cannot slip through.

Usage
-----
python build_val_set.py --num 500 --out_dir datasets/val_pairs
"""
import argparse
import glob
import hashlib
import os
import random
import shutil

_EXTS = ("jpg", "jpeg", "png", "bmp")


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", default="datasets/coco/train2017")
    ap.add_argument("--wikiart_dir", default="datasets/wikiart")
    ap.add_argument("--benchmark_style_dir", default="datasets/style_flat")
    ap.add_argument("--benchmark_size", type=int, default=1000,
                    help="how many content images the benchmark already consumed")
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--out_dir", default="datasets/val_pairs")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    contents = sorted(f for f in os.listdir(args.coco_dir)
                      if f.lower().endswith(tuple("." + e for e in _EXTS)))
    held = contents[args.benchmark_size:args.benchmark_size + args.num]
    if len(held) < args.num:
        raise SystemExit(f"only {len(held)} content images left after the benchmark's "
                         f"first {args.benchmark_size}")

    print("hashing the benchmark style pool ...")
    used = {file_hash(os.path.join(args.benchmark_style_dir, f))
            for f in os.listdir(args.benchmark_style_dir)}
    print(f"  {len(used)} distinct benchmark style images")

    cands = []
    for e in _EXTS:
        cands += glob.glob(os.path.join(args.wikiart_dir, "**", f"*.{e}"), recursive=True)
    # glob order differs between file systems, so sort before the seeded shuffle.
    cands.sort(key=lambda p: os.path.relpath(p, args.wikiart_dir).replace(os.sep, "/"))
    random.Random(args.seed).shuffle(cands)

    out_c = os.path.join(args.out_dir, "content")
    out_s = os.path.join(args.out_dir, "style")
    for d in (out_c, out_s):
        os.makedirs(d, exist_ok=True)

    picked = []
    for p in cands:
        if len(picked) >= args.num:
            break
        try:
            if file_hash(p) in used:
                continue
        except OSError:
            continue
        picked.append(p)
    if len(picked) < args.num:
        raise SystemExit(f"only {len(picked)} unused style images found")

    for i, (c, s) in enumerate(zip(held, picked)):
        name = f"{i:05d}.png"
        from PIL import Image
        Image.open(os.path.join(args.coco_dir, c)).convert("RGB").save(os.path.join(out_c, name))
        Image.open(s).convert("RGB").save(os.path.join(out_s, name))
    print(f"wrote {args.num} validation pairs to {args.out_dir}")
    with open(os.path.join(args.out_dir, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("Validation pairs, disjoint from the 1000-pair benchmark.\n")
        f.write(f"content: sorted MS-COCO train2017, entries "
                f"{args.benchmark_size}..{args.benchmark_size + args.num - 1}\n")
        f.write(f"style: WikiArt images whose SHA-1 matches no image in "
                f"{args.benchmark_style_dir} (seed {args.seed})\n\n")
        for i, (c, s) in enumerate(zip(held, picked)):
            f.write(f"{i:05d}\t{c}\t{os.path.relpath(s, args.wikiart_dir)}\n")


if __name__ == "__main__":
    main()
