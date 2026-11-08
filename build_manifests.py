"""Write exact, hashed manifests of the benchmark and validation pairs.

Every image a reported number depends on is listed with its SHA-256, so the sets can be
checked byte for byte without redistributing the images. The script also records how the
evaluation images relate to the collections training drew from.

Usage
-----
python build_manifests.py --out_dir manifests
"""
import argparse
import hashlib
import json
import os

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", default="datasets/coco/train2017")
    ap.add_argument("--wikiart_dir", default="datasets/wikiart")
    ap.add_argument("--style_flat", default="datasets/style_flat")
    ap.add_argument("--val_dir", default="datasets/val_pairs")
    ap.add_argument("--movements", default="results/style_movements.json")
    ap.add_argument("--benchmark_size", type=int, default=1000)
    ap.add_argument("--out_dir", default="manifests")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    coco = sorted(f for f in os.listdir(args.coco_dir) if f.lower().endswith(_EXTS))
    flat = sorted(f for f in os.listdir(args.style_flat) if f.lower().endswith(_EXTS))
    wiki = {}
    for dirpath, _, files in os.walk(args.wikiart_dir):
        for f in files:
            if f.lower().endswith(_EXTS):
                p = os.path.join(dirpath, f)
                wiki[os.path.relpath(p, args.wikiart_dir).replace(os.sep, "/")] = p
    print(f"{len(coco)} COCO, {len(wiki)} WikiArt, hashing WikiArt ...", flush=True)
    wiki_hash = {}
    for rel, p in wiki.items():
        wiki_hash.setdefault(sha256(p), []).append(rel)

    report = {"coco_train2017_images": len(coco), "wikiart_images": len(wiki),
              "wikiart_distinct_hashes": len(wiki_hash)}

    n = args.benchmark_size
    with open(os.path.join(args.out_dir, "benchmark_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("pair\tcontent_coco_train2017\tcontent_sha256\tstyle_flat\tstyle_sha256\tstyle_wikiart\n")
        hits = 0
        for i, (c, s) in enumerate(zip(coco[:n], flat[:n])):
            hs = sha256(os.path.join(args.style_flat, s))
            src = wiki_hash.get(hs, [""])[0]
            hits += bool(src)
            f.write(f"{i:05d}\t{c}\t{sha256(os.path.join(args.coco_dir, c))}\t{s}\t{hs}\t{src}\n")
    report["benchmark"] = {"pairs": n, "content_from_training_collection": n,
                           "style_found_in_training_collection": hits}

    rows = [l.rstrip("\n").split("\t") for l in
            open(os.path.join(args.val_dir, "MANIFEST.txt"), encoding="utf-8")
            if l[:5].isdigit()]
    val_hits = 0
    bench = {sha256(os.path.join(args.style_flat, s)) for s in flat[:n]}
    overlap = 0
    with open(os.path.join(args.out_dir, "validation_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("pair\tcontent_coco_train2017\tcontent_sha256\tstyle_wikiart\tstyle_sha256\n")
        for pair, c, s in rows:
            rel = s.replace("\\", "/")
            hs = sha256(wiki[rel])
            val_hits += hs in wiki_hash
            overlap += hs in bench
            f.write(f"{pair}\t{c}\t{sha256(os.path.join(args.coco_dir, c))}\t{rel}\t{hs}\n")
    report["validation"] = {"pairs": len(rows), "content_from_training_collection": len(rows),
                            "style_found_in_training_collection": val_hits,
                            "style_overlap_with_benchmark": overlap}
    with open(os.path.join(args.out_dir, "overlap_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
