# FDFormer

Code, configurations, manifests and recorded measurements for

> L. Hao and Y. Wu, *Frequency-Decoupled Cross-Attention for Efficient Arbitrary Style
> Transfer with Inference-Time Strength Control*, submitted to Discover Artificial
> Intelligence.

FDFormer splits each relu4_1 feature of a frozen VGG-19 with a learnable blur into a
low- and a high-frequency band that sum exactly to the feature. Each band is transferred
by its own linear cross-attention branch, a content-adaptive gate sets per location how
much style is admitted, and a decoder renders the image. A coefficient on the low branch,
`alpha_low`, sets stylisation strength at inference without retraining.

## What the measurements show

All figures below are on an RTX 3070 Laptop GPU (8 GB) and are stored under `results/`.

- 8.42 M parameters and 13.3 ms per 256x256 image; StyTr2 needs 5.7x the parameters
  and 7.9x the time.
- From 256 to 1024 pixels on a side, latency rises 14-fold, against 21-fold for SANet.
- Under a hard allocator ceiling, the forward pass stays in GPU memory up to 1536x1536,
  and the low-memory inference path (`--low_memory`) up to 2048x2048 (4,578 MB
  allocated, 945 ms). MicroAST also reaches 2048x2048 with less memory and time.
- At alpha=1, on the 1,000 benchmark pairs, FDFormer has a lower content LPIPS than five
  of the six baselines and a lower VGG style-statistic distance than a different five,
  every difference significant after Holm correction. In three seeded retrainings the
  leads over four and over five baselines hold in every seed. On 200 pairs of COCO
  val2017 photographs and six styles outside the training collections, the same ten leads
  remain significant, eight of them across at least five of the six styles.

The paper also reports where these advantages stop. On the network-free LP-SWD style
distance FDFormer is the farthest of the seven methods from the style. At 2048x2048
MicroAST ties on content LPIPS and leads on both style measures. `alpha_high` does not
control brushwork. In the seeded ablation no difference between the learned split and the
single path, the fixed splits or a capacity-matched two-branch control is resolved.

| Method | ArtFID | LPIPS (content) | Style dist. (VGG) | LP-SWD x1e3 |
|---|---|---|---|---|
| SANet | 137.68 | 0.621 | 1.214 | 14.47 |
| AdaAttN | 133.85 | 0.601 | 1.993 | 20.96 |
| StyTr2 | 132.62 | 0.539 | 1.745 | 14.01 |
| MicroAST | 145.39 | 0.562 | 3.331 | 14.33 |
| AesPA-Net | 147.55 | 0.618 | 2.689 | 20.50 |
| SaMam | 127.20 | 0.451 | 2.261 | 14.76 |
| FDFormer, alpha=1 | 139.68 | 0.530 | 1.684 | 21.87 |
| FDFormer, alpha=1.4 (selected on validation pairs) | 136.92 | 0.559 | 1.475 | 23.74 |

Lower is better in every column. The FDFormer rows are the released checkpoint, trained
before seed control was added. Its three seeded retrainings with the same configuration
have a higher (worse) ArtFID, 143.73 to 146.06 at alpha=1 (`results/seeded_vs_baselines.json`).

## Layout

```
models/              network: frequency split, linear cross-attention, gate, decoder
losses/              training objective, including the high-frequency Gram term
data/                seeded content/style loader
configs/             v2_hf_rescaled.yaml is the reported model; the rest are ablations
baselines_official/  pair-aligned inference scripts for the six baselines
manifests/           benchmark and validation pairs with SHA-256 of every image
results/             every measurement behind the paper's tables and figures
tools/               CPU smoke test
```

## Install

```bash
pip install -r requirements-pinned.txt --extra-index-url https://download.pytorch.org/whl/cu128
python tools/smoke_test.py          # wiring check, no data or GPU needed
```

`requirements-pinned.txt` lists the exact versions behind the reported numbers
(Python 3.13.6, PyTorch 2.11.0 with CUDA 12.8). `requirements.txt` gives looser bounds.

## Checkpoint

The trained model reported in the paper is attached to the GitHub release as
`fdformer_v2_ckpt_080000.pth` (33.7 MB, 80,000 iterations, configuration
`configs/v2_hf_rescaled.yaml`).

```
SHA-256  aef658ec757fd0ad93e2cbb8a6058cddf63ebac0d401fd1b665225e2de4c33d7
```

## Data

- Content: MS-COCO `train2017`, https://cocodataset.org
- Style: WikiArt, https://www.wikiart.org, and Painter by Numbers,
  https://www.kaggle.com/c/painter-by-numbers

The images are not redistributed. Place them under `datasets/coco/train2017/` and
`datasets/wikiart/<Movement>/<file>.jpg`. `manifests/benchmark_pairs.tsv` and
`manifests/validation_pairs.tsv` define the 1,000 benchmark pairs and the 500 validation
pairs by file name and SHA-256, so a local copy can be checked byte for byte.
`manifests/overlap_report.json` records that every benchmark and validation image also
belongs to the collections training draws from, and that the two sets share no style
image. "Held out" in the paper means held out from benchmark evaluation, not from
training.

```bash
python build_style_pool.py          # benchmark styles -> datasets/style_flat/00000.jpg ...
python build_manifests.py --out_dir manifests   # rebuild and compare the manifests
```

## Train

```bash
python train.py --config configs/v2_hf_rescaled.yaml --seed 0
```

`--seed` fixes Python, NumPy and PyTorch and makes the data order, the style pairing and
both crops a pure function of (seed, epoch, image index), so every variant sees the same
examples under a seed, whatever the number of loader workers. A resumed run
(`--resume latest.pth`) skips exactly the examples it has used. About 6 h on the laptop
GPU above with `--num_workers 4`.

## Stylise

```bash
python stylize.py --ckpt fdformer_v2_ckpt_080000.pth \
    --content content.jpg --style style.jpg --alpha_low 1.4 --alpha_high 1.4
```

`alpha_low` sets stylisation strength; 1.4 is the setting chosen on the validation pairs.
For large images, `batch_stylize.py --low_memory` keeps only the relu4_1 maps and gives
the same output as the forward pass.

## Reproducing the paper

The benchmark is built by stylising the 1,000 pairs into name-aligned folders, then
scoring them:

```bash
python batch_stylize.py --ckpt fdformer_v2_ckpt_080000.pth \
    --content_dir datasets/coco/train2017 --style_dir datasets/style_flat \
    --out_dir eval/default --num 1000 --size 256
python eval_metrics.py --stylized_dir eval/default/stylized \
    --content_dir eval/default/content --style_dir eval/default/style \
    --json_out results/metrics_fdformer.json
```

Baselines are run with the scripts in `baselines_official/`, each copied into a clone
of the authors' repository and used with the weights they release. AesPA-Net's encoder
is the substitution described in the paper.

| Result | Script | Output in `results/` |
|---|---|---|
| Main comparison | `eval_metrics.py` | `metrics_*.json` |
| Paired intervals and tests, Holm over 24 tests | `eval_uncertainty.py`, `compute_holm.py` | `uncertainty*.json`, `holm.json` |
| Independent style metric (LP-SWD) | `eval_style_independent.py` | `style_independent.json` |
| Operating point, checkpoint and gate chosen on validation pairs | `build_val_set.py`, `select_alpha.py`, `select_checkpoint.py`, `select_gate.py` | `alpha_selection*.json`, `checkpoint_selection.json`, `gate_selection.json` |
| What each band carries | `analyze_frequency.py`, `analyze_by_movement.py` | `freq_analysis*.json`, `movement_analysis.json` |
| Attention induced by each branch | `analyze_correspondence.py` | `correspondence.json` |
| Response of the two coefficients | `analyze_control.py`, `analyze_control_absolute.py` | `control_*.json` |
| Seeded ablation, 5 variants x seeds 0, 1, 2 | `run_seeded_ablations.py`, `aggregate_seeded.py` | `seeded/`, `seeded_summary.json` |
| Seeded runs against the baselines | `compare_seeded_baselines.py` | `seeded_vs_baselines.json` |
| Cost at 256x256 and largest completed input | `bench_efficiency.py`, `run_efficiency_final.sh` | `eff_*.json` |
| GPU-resident limits under an allocator ceiling | `bench_resident.py` | `resident_memory.json`, `resident_check_*.json` |
| Quality at 1024 and 2048 | `eval_highres.py`, `compute_highres_paired.py` | `highres_*.json`, `highres_paired.json` |
| SaMam wrapper check | `verify_samam_wrapper.py` | `samam_wrapper_check.json` |
| Unseen photographs (COCO val2017) and styles checked against every WikiArt training image | `eval_unseen.py` | `unseen_*.json`, `manifests/unseen_*` |
| Figures | `make_figures.py`, `make_figures_revision.py`, `make_fig_highres.py`, `make_schematics.py` | written to `figures/` |

```bash
python run_seeded_ablations.py --root runs/seeded --seeds 0 1 2
python aggregate_seeded.py
python compare_seeded_baselines.py
```

Scripts that call a baseline read its repository from `FDFORMER_BASELINES`
(default `./baselines`). GPU kernels are not forced to be deterministic, so a retrained
model matches the recorded one in its data sequence, not bit for bit.

## License

MIT, see `LICENSE`. The datasets and the baseline weights keep their own licences.
