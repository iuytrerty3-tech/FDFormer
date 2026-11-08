#!/usr/bin/env bash
# Revision experiments that do not measure time, run one after another so that at
# most one of them competes with the decomposition retrains for the GPU.
# Timing measurements are deliberately excluded: they are taken separately on an
# idle device, as the measurement-protocol section of the paper states.
set -uo pipefail
cd "$(dirname "$0")"
FD="$(pwd)"
PY="${PYTHON:-python}"
mkdir -p logs results

step() { echo -e "\n========== $* ==========" ; }

step "1. AesPA-Net: score it on the benchmark"
mkdir -p eval/aespanet
[ -d eval/aespanet/content ] || cp -r eval/default/content eval/aespanet/content
[ -d eval/aespanet/style   ] || cp -r eval/default/style   eval/aespanet/style
[ -f results/metrics_aespanet.json ] || \
  "$PY" eval_metrics.py --stylized_dir eval/aespanet/stylized \
    --content_dir eval/aespanet/content --style_dir eval/aespanet/style \
    --size 256 --json_out results/metrics_aespanet.json 2>&1 | tail -6

step "2. SaMam: stylise the benchmark pairs"
mkdir -p eval/samam/stylized
if [ "$(ls eval/samam/stylized 2>/dev/null | wc -l)" -lt 1000 ]; then
  ( cd "${FDFORMER_BASELINES:-baselines}/SaMam/TEST" && \
    "$PY" samam_paired.py \
      --content_dir "$FD/eval/default/content" \
      --style_dir   "$FD/eval/default/style" \
      --out_dir     "$FD/eval/samam/stylized" \
      --model_ckpt ./checkpoint/SaMam_patchsize_x8.ckpt --size 256
  ) 2>&1 | tail -5
fi
echo "samam outputs: $(ls eval/samam/stylized | wc -l)"

step "3. SaMam: score it on the benchmark"
[ -d eval/samam/content ] || cp -r eval/default/content eval/samam/content
[ -d eval/samam/style   ] || cp -r eval/default/style   eval/samam/style
[ -f results/metrics_samam.json ] || \
  "$PY" eval_metrics.py --stylized_dir eval/samam/stylized \
    --content_dir eval/samam/content --style_dir eval/samam/style \
    --size 256 --json_out results/metrics_samam.json 2>&1 | tail -6

step "4. Evidence for the frequency assumption"
[ -f results/freq_analysis.json ] || \
  "$PY" analyze_frequency.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --num 500 --json_out results/freq_analysis.json 2>&1 | tail -35

step "5. Choose the inference coefficient on held-out pairs"
[ -f results/alpha_selection.json ] || \
  "$PY" select_alpha.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --val_dir datasets/val_pairs --grid 1.0,1.25,1.4,1.5,1.75 \
    --json_out results/alpha_selection.json 2>&1 | tail -25

step "6. Does the coefficient separation hold across many pairs"
[ -f results/control_generalisation.json ] || \
  "$PY" analyze_control.py --ckpt experiments/fdformer_v2/ckpt_080000.pth \
    --num 300 --json_out results/control_generalisation.json 2>&1 | tail -12

step "7. Uncertainty around every reported figure"
[ -f results/uncertainty.json ] || \
  "$PY" eval_uncertainty.py \
    --methods fdformer=eval/v2 sanet=eval/sanet adaattn=eval/adaattn \
              stytr2=eval/stytr2 microast=eval/microast aespanet=eval/aespanet \
              samam=eval/samam \
    --reference fdformer --artfid_splits 5 \
    --json_out results/uncertainty.json 2>&1 | tail -70

step "chain finished"
