#!/usr/bin/env bash
# Re-measure every efficiency figure in one session, on an idle GPU.
#
# Run this ONLY when nothing else is using the device: latency and peak memory are
# meaningless under contention, and the paper states that all methods were measured
# in a single session under identical conditions.
#
#   bash run_efficiency_final.sh
set -uo pipefail
cd "$(dirname "$0")"
FD="$(pwd)"
PY="${PYTHON:-python}"
B="${FDFORMER_BASELINES:-baselines}"
mkdir -p logs results

# Count only the Python processes we could have started. A laptop GPU usually also
# carries a context we neither own nor can inspect, which nvidia-smi reports as
# "Insufficient Permissions"; that is not a competing training job and must not block
# the measurement. Set ALLOW_BUSY_GPU=1 to run anyway.
busy=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader \
       | grep -ci "python" || true)
if [ "$busy" -gt 0 ] && [ "${ALLOW_BUSY_GPU:-0}" != "1" ]; then
  echo "WARNING: $busy Python process(es) are still on the GPU. Timings would be wrong."
  nvidia-smi --query-compute-apps=pid,process_name --format=csv
  echo "Stop them first, then re-run. Aborting."
  exit 1
fi
echo "GPU is free of competing Python jobs; other contexts reported by nvidia-smi:"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true

step() { echo -e "\n========== $* ==========" ; }

step "environment"
"$PY" - <<'PY'
import platform, torch
p = torch.cuda.get_device_properties(0)
print(f"gpu      {torch.cuda.get_device_name(0)}  {p.total_memory/1024**3:.2f} GB  cc {p.major}.{p.minor}")
print(f"torch    {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}")
print(f"python   {platform.python_version()}  {platform.system()} {platform.release()}")
PY
nvidia-smi --query-gpu=driver_version --format=csv,noheader

step "FDFormer"
"$PY" bench_efficiency.py --config configs/v2_hf_rescaled.yaml \
    --sweep --json_out results/eff_fdformer.json 2>&1 | tail -14

step "SANet"
( cd "$B/SANET" && "$PY" sanet_paired.py \
    --content_dir "$FD/eval/default/content" --style_dir "$FD/eval/default/style" \
    --out_dir "$FD/eval/sanet/stylized" --vgg vgg_normalised.pth \
    --decoder decoder_iter_500000.pth --transform transformer_iter_500000.pth \
    --bench --eff_out "$FD/results/eff_sanet.json" ) 2>&1 | tail -3

step "AdaAttN"
( cd "$B/AdaAttN" && "$PY" adaattn_paired.py \
    --content_dir "$FD/eval/default/content" --style_dir "$FD/eval/default/style" \
    --out_dir "$FD/eval/adaattn/stylized" \
    --image_encoder_path checkpoints/vgg_normalised.pth \
    --bench --eff_out "$FD/results/eff_adaattn.json" ) 2>&1 | tail -3

step "StyTr2"
( cd "$B/StyTR-2" && "$PY" stytr2_paired.py \
    --content_dir "$FD/eval/default/content" --style_dir "$FD/eval/default/style" \
    --out_dir "$FD/eval/stytr2/stylized" \
    --bench --eff_out "$FD/results/eff_stytr2.json" ) 2>&1 | tail -3

step "MicroAST"
( cd "$B/MicroAST" && "$PY" microast_paired.py \
    --content_dir "$FD/eval/default/content" --style_dir "$FD/eval/default/style" \
    --out_dir "$FD/eval/microast/stylized" \
    --bench --eff_out "$FD/results/eff_microast.json" ) 2>&1 | tail -3

step "AesPA-Net"
( cd "$B/AesPA-Net" && "$PY" aespanet_paired.py \
    --content_dir "$FD/eval/default/content" --style_dir "$FD/eval/default/style" \
    --out_dir "$FD/eval/aespanet/stylized" --vgg "$B/SANET/vgg_normalised.pth" \
    --bench --eff_out "$FD/results/eff_aespanet.json" ) 2>&1 | tail -3

step "SaMam parameter count only"
# SaMam runs here through its plain-PyTorch path rather than the fused kernels its
# authors time, so a latency from this machine would describe our fallback and not
# their method. The parameter count does not depend on which path executes.
( cd "$B/SaMam/TEST" && EFF_OUT="$FD/results/eff_samam.json" "$PY" - <<'PY'
import json, os, sys, torch
sys.path.append(os.path.abspath(".."))
from TRAIN.lightning_module.lightningmodel import LightningModel
cfg = dict(nVSSMs=2, nSAVSSMs=2, nSAVSSGs=2, embed_dim=256, patch_size=8,
           representation_dim=64, d_state=16, expand=2.0, compress_ratio=8,
           squeeze_factor=8, mamba_from_trion=0)
m = LightningModel(**cfg)
n = sum(p.numel() for p in m.model.parameters()) / 1e6
rec = {"params_M": round(n, 3), "flops_G": None, "latency_ms": None,
       "max_res": None, "size": 256,
       "note": "run through the plain-PyTorch path; latency not comparable"}
print("SaMam efficiency:", rec)
out = os.environ["EFF_OUT"]
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rec, open(out, "w"), indent=2)
print("wrote", out)
PY
) 2>&1 | tail -3

step "merge"
"$PY" merge_revision_results.py

step "done"
