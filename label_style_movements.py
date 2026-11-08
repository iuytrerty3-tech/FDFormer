"""Recover the art movement of each benchmark style image.

The benchmark's style pool was flattened out of the WikiArt directory tree, which
is organised by movement, so the label is recoverable: the flattened files are byte
copies, and matching them back is a matter of identifying which original each one
came from. Sizes are compared first, since that is a single stat call per candidate,
and only files of identical size are hashed.

The result lets the frequency analysis be reported per movement rather than only in
aggregate, which is what shows whether the decomposition behaves the same way on a
pointillist canvas as on a colour-field painting.

Usage
-----
python label_style_movements.py --json_out results/style_movements.json
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sha1(path, chunk=1 << 20):
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
    ap.add_argument("--flat_dir", default="datasets/style_flat")
    ap.add_argument("--wikiart_dir", default="datasets/wikiart")
    ap.add_argument("--json_out", default="results/style_movements.json")
    args = ap.parse_args()

    flat = sorted(f for f in os.listdir(args.flat_dir) if f.lower().endswith(_EXTS))
    print(f"{len(flat)} flattened style images")

    by_size = defaultdict(list)
    for f in flat:
        p = os.path.join(args.flat_dir, f)
        by_size[os.path.getsize(p)].append(f)

    print("scanning the WikiArt tree by file size ...")
    candidates = defaultdict(list)          # size -> [(movement, path)]
    n_seen = 0
    for movement in sorted(os.listdir(args.wikiart_dir)):
        mdir = os.path.join(args.wikiart_dir, movement)
        if not os.path.isdir(mdir):
            continue
        for root, _, files in os.walk(mdir):
            for f in files:
                if not f.lower().endswith(_EXTS):
                    continue
                p = os.path.join(root, f)
                n_seen += 1
                try:
                    s = os.path.getsize(p)
                except OSError:
                    continue
                if s in by_size:
                    candidates[s].append((movement, p))
    print(f"  {n_seen} WikiArt files scanned; {sum(len(v) for v in candidates.values())} "
          f"size matches to hash")

    flat_hash = {f: sha1(os.path.join(args.flat_dir, f)) for f in flat}
    label = {}
    source = {}
    for size, names in by_size.items():
        for movement, p in candidates.get(size, []):
            try:
                h = sha1(p)
            except OSError:
                continue
            for f in names:
                if f not in label and flat_hash[f] == h:
                    label[f] = movement
                    source[f] = os.path.relpath(p, args.wikiart_dir)

    # The benchmark pairs are named by index, so translate the flattened filename to
    # the name the evaluation folders use.
    per_index = {}
    for i, f in enumerate(flat):
        if f in label:
            per_index[f"{i:05d}.png"] = label[f]

    counts = defaultdict(int)
    for m in per_index.values():
        counts[m] += 1
    print(f"labelled {len(per_index)}/{len(flat)}")
    for m, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:32s} {c}")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"by_pair_name": per_index, "counts": dict(counts),
                   "source": {f"{i:05d}.png": source[n]
                              for i, n in enumerate(flat) if n in source}},
                  f, indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
