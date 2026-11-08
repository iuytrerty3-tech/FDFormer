"""Rebuild the benchmark's 1,000-painting style pool from the recorded sample.

The pool was drawn once from the WikiArt tree and flattened into datasets/style_flat
as 00000.jpg ... 00999.jpg. results/style_movements.json records, for every benchmark
pair, the WikiArt file its style image was copied from, so the pool can be rebuilt
byte for byte on any machine that has the 27-movement WikiArt collection laid out as
datasets/wikiart/<Movement>/<artist>_<title>.jpg. Pair i then joins the i-th MS-COCO
train2017 image in sorted order to the i-th file here (see batch_stylize.py).

Usage
-----
python build_style_pool.py [--wikiart_dir datasets/wikiart] [--out_dir datasets/style_flat]
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="results/style_movements.json")
    ap.add_argument("--wikiart_dir", default="datasets/wikiart")
    ap.add_argument("--out_dir", default="datasets/style_flat")
    args = ap.parse_args()

    with open(args.record, encoding="utf-8") as f:
        source = json.load(f)["source"]      # pair name "00000.png" -> "Movement\\file.jpg"
    os.makedirs(args.out_dir, exist_ok=True)
    missing = []
    for pair in sorted(source):
        rel = source[pair].replace("\\", "/")
        src = os.path.join(args.wikiart_dir, *rel.split("/"))
        if not os.path.exists(src):
            missing.append(rel)
            continue
        shutil.copyfile(src, os.path.join(args.out_dir, os.path.splitext(pair)[0] + ".jpg"))
    print("copied %d of %d paintings into %s" % (len(source) - len(missing), len(source),
                                                 args.out_dir))
    if missing:
        print("not found under %s (first 10):" % args.wikiart_dir)
        for m in missing[:10]:
            print("   " + m)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
