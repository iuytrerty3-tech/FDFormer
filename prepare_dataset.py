"""Prepare an image folder for FDFormer: optional random subset + resize.

Solves two laptop problems in one pass:
  * disk:  re-encodes everything to JPEG and shrinks the short side, so a ~60 GB
            source folder typically drops to a few GB;
  * speed: smaller images decode faster during training.

It never upscales (only shrinks images larger than --size), converts everything to
RGB JPEG, and can keep just a random subset with --limit/--shuffle.

Examples
--------
# 30k random COCO images, shrunk to 512px short side -> datasets/coco_512
python prepare_dataset.py --src raw/coco/train2017 --dst datasets/coco_512 \
    --size 512 --limit 30000 --shuffle

# resize an entire style folder, keep every image
python prepare_dataset.py --src raw/wikiart --dst datasets/wikiart_512 --size 512

# subset only, no resizing (--size 0 still re-encodes to JPEG to save space)
python prepare_dataset.py --src raw/wikiart --dst datasets/wikiart_small \
    --size 0 --limit 15000 --shuffle

Then point configs/default.yaml at the new folders:
    data: {content_dir: ./datasets/coco_512, style_dir: ./datasets/wikiart_512}
"""
import argparse
import os
import random
from functools import partial
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
try:
    _RESAMPLE = Image.Resampling.LANCZOS      # Pillow >= 9.1
except AttributeError:                        # older Pillow
    _RESAMPLE = Image.LANCZOS


def find_images(src):
    out = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def process_one(pair, size, quality):
    src_path, dst_path = pair
    try:
        img = Image.open(src_path).convert("RGB")
        if size > 0:
            w, h = img.size
            short = min(w, h)
            if short > size:                  # only downscale, never upscale
                scale = size / short
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), _RESAMPLE)
        img.save(dst_path, "JPEG", quality=quality)
        return None
    except Exception as e:  # noqa: BLE001  skip unreadable or corrupt files
        return f"skip {src_path}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source folder (searched recursively)")
    ap.add_argument("--dst", required=True, help="output folder (created)")
    ap.add_argument("--size", type=int, default=512,
                    help="target short side in px; 0 = do not resize (still re-encodes to JPEG)")
    ap.add_argument("--limit", type=int, default=0, help="keep at most this many images (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="random subset instead of first N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-95")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    paths = find_images(args.src)
    if not paths:
        raise SystemExit(f"no images found under {args.src}")
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    if args.limit:
        paths = paths[:args.limit]

    os.makedirs(args.dst, exist_ok=True)
    # resolve collision-free destination names in the main process (workers stay independent)
    pairs, seen = [], {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        n = seen.get(stem, 0)
        seen[stem] = n + 1
        name = stem if n == 0 else f"{stem}_{n}"
        pairs.append((p, os.path.join(args.dst, name + ".jpg")))

    worker = partial(process_one, size=args.size, quality=args.quality)
    skipped = 0
    with Pool(args.workers) as pool:
        for msg in tqdm(pool.imap_unordered(worker, pairs), total=len(pairs), desc="preparing"):
            if msg:
                skipped += 1
                if skipped <= 10:
                    print(msg)

    print(f"Done. wrote {len(pairs) - skipped} images to {args.dst} ({skipped} skipped).")


if __name__ == "__main__":
    main()
