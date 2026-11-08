"""Measure FDFormer efficiency on the local GPU.

Reports, as JSON:
  * params_M           total parameters (encoder + transfer net + decoder), in millions
  * params_trainable_M trainable params only (everything except the frozen VGG)
  * flops_G            forward GFLOPs at --size (uses thop or fvcore if installed; else null)
  * latency_ms         mean forward latency at --size, batch 1 (CUDA-synchronised)
  * max_res            largest square that runs a forward without OOM (CUDA only)
  * device             the device string these numbers were measured on

These numbers are specific to the hardware they are measured on. The baselines are
measured with the equivalent code inside each baseline's repository.

Example
-------
python bench_efficiency.py --config configs/default.yaml --json_out results/eff_fdformer.json
"""
import argparse
import json
import os
import time

import torch
import yaml

from models import FDFormer


def build_model(cfg_path, device):
    """Build FDFormer with the same dims as the trained model (if a config is given)."""
    m = {"embed_dim": 256, "heads": 4, "freq_decouple": True, "use_gate": True,
         "freq_mode": "learnable"}
    if cfg_path and os.path.exists(cfg_path):
        # Explicit UTF-8: the configs carry comments with non-ASCII punctuation, and on
        # a non-UTF-8 locale the default encoding raises here.
        with open(cfg_path, encoding="utf-8") as f:
            mc = (yaml.safe_load(f) or {}).get("model", {})
        m.update({k: mc[k] for k in m if k in mc})
    model = FDFormer(embed_dim=m["embed_dim"], heads=m["heads"], pretrained_vgg=True,
                     freq_decouple=m["freq_decouple"], use_gate=m["use_gate"],
                     freq_mode=m["freq_mode"]).to(device).eval()
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.trainable_parameters())
    return total / 1e6, train / 1e6


def measure_flops(model, content, style):
    """Best-effort forward GFLOPs. Returns None if no profiler is installed."""
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(content, style), verbose=False)
        return float(macs) * 2 / 1e9            # MACs -> FLOPs -> giga
    except Exception as e:  # noqa: BLE001
        print(f"[FLOPs] thop unavailable/failed ({e}); trying fvcore.")
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, (content, style)).total()) / 1e9
    except Exception as e:  # noqa: BLE001
        print(f"[FLOPs] fvcore unavailable/failed ({e}); leaving FLOPs blank.")
        return None


def measure_latency(model, device, size, runs, warmup):
    c = torch.rand(1, 3, size, size, device=device)
    s = torch.rand(1, 3, size, size, device=device)
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(c, s)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(c, s)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0           # ms per forward


def measure_max_res(model, device):
    """Largest square resolution that runs a forward without OOM (CUDA only)."""
    if device.type != "cuda":
        print("[max_res] not on CUDA; skipping the OOM probe.")
        return None
    best = None
    for size in (256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048):
        try:
            torch.cuda.empty_cache()
            c = torch.rand(1, 3, size, size, device=device)
            s = torch.rand(1, 3, size, size, device=device)
            with torch.no_grad():
                model(c, s)
            torch.cuda.synchronize()
            best = size
            del c, s
        except RuntimeError as e:               # OOM (or any runtime cap) -> stop
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    return f"{best}x{best}" if best else None


def measure_sweep(model, device, sizes, runs=10, warmup=3):
    """Latency + peak memory at each resolution -> the efficiency scaling curve.

    Returns dict with parallel lists {sizes, latency_ms, peak_mem_MB}. Sizes that
    OOM are dropped and the sweep stops there (so the curve ends at max usable res).
    On CPU peak_mem_MB is left as None (no CUDA allocator to query).
    """
    is_cuda = device.type == "cuda"
    out = {"sizes": [], "latency_ms": [], "peak_mem_MB": []}
    for size in sizes:
        try:
            if is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            c = torch.rand(1, 3, size, size, device=device)
            s = torch.rand(1, 3, size, size, device=device)
            with torch.no_grad():
                for _ in range(warmup):
                    model(c, s)
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(runs):
                    model(c, s)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
            lat = (t1 - t0) / runs * 1000.0
            mem = (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else None
            out["sizes"].append(size)
            out["latency_ms"].append(round(lat, 2))
            out["peak_mem_MB"].append(round(mem, 1) if mem is not None else None)
            print(f"[sweep] {size:5d}px  {lat:8.2f} ms  "
                  f"{(str(round(mem, 1)) + ' MB') if mem is not None else 'mem n/a'}")
            del c, s
        except RuntimeError as e:                # OOM -> curve ends here
            if "out of memory" in str(e).lower():
                if is_cuda:
                    torch.cuda.empty_cache()
                print(f"[sweep] {size}px OOM; stopping sweep.")
                break
            raise
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml",
                    help="match the trained model's dims (embed_dim/heads/switches)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=256, help="resolution for latency + FLOPs")
    ap.add_argument("--runs", type=int, default=50, help="timed forward passes")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no_max_res", action="store_true", help="skip the max-resolution OOM probe")
    ap.add_argument("--sweep", action="store_true",
                    help="measure latency + peak memory across resolutions (efficiency curve)")
    ap.add_argument("--sweep_sizes", default="256,384,512,640,768,1024",
                    help="comma-separated square sizes for --sweep")
    ap.add_argument("--sweep_runs", type=int, default=10, help="timed passes per sweep size")
    ap.add_argument("--low_memory", action="store_true",
                    help="time FDFormer.stylize_low_memory, which returns the same image")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(args.config, device)
    if args.low_memory:
        model.forward = model.stylize_low_memory

    p_total, p_train = count_params(model)
    c = torch.rand(1, 3, args.size, args.size, device=device)
    s = torch.rand(1, 3, args.size, args.size, device=device)
    flops_g = measure_flops(model, c, s)
    del c, s
    latency_ms = measure_latency(model, device, args.size, args.runs, args.warmup)
    max_res = None if args.no_max_res else measure_max_res(model, device)

    sweep = None
    if args.sweep:
        sizes = [int(x) for x in str(args.sweep_sizes).split(",") if x.strip()]
        sweep = measure_sweep(model, device, sizes, runs=args.sweep_runs, warmup=3)

    rec = {"params_M": round(p_total, 3),
           "params_trainable_M": round(p_train, 3),
           "flops_G": (round(flops_g, 2) if flops_g is not None else None),
           "latency_ms": round(latency_ms, 2),
           "max_res": max_res,
           "size": args.size,
           "device": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")}
    if sweep is not None:
        rec["sweep"] = sweep

    print("\n==== Efficiency (FDFormer) ====")
    for k, v in rec.items():
        print(f"{k:20s}: {v}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote efficiency JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
