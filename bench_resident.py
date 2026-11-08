"""Which resolutions actually stay resident in GPU memory?

A forward pass finishing does not show it fitted: under Windows WDDM an allocation past
the card's capacity can spill into shared system memory instead of failing. This script
launches one method at a time under a hard allocator ceiling equal to the memory free on
the card, so a pass that would spill fails with out-of-memory instead, and it records:

  physical memory on the card and what other contexts already use,
  peak allocated and peak reserved at each size,
  latency at each size,
  the Windows GPU Process Memory counters (Dedicated Usage, Shared Usage) sampled during
  the run, so that any shared-memory use is observed and not inferred,
  the Sysmem Fallback setting, which only the NVIDIA Control Panel shows, passed in by hand.

Each baseline runs through its own wrapper, unchanged; a preamble sets the ceiling and
records the reserved peak at the moment the wrapper reads the allocated peak.

Run on an otherwise idle GPU.

Usage
-----
python bench_resident.py --fallback "Prefer sysmem fallback (driver default)" \
    --json_out results/resident_memory.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

BASE = os.environ.get("FDFORMER_BASELINES", "baselines")
HERE = os.path.dirname(os.path.abspath(__file__))
C, S = (os.path.join(HERE, "eval/default", d) for d in ("content", "style"))
O = os.environ.get("RESIDENT_TMP", "work/resident_tmp")
COMMON = ["--content_dir", C, "--style_dir", S, "--out_dir", O, "--bench"]
METHODS = {
    "sanet": ("SANET", "sanet_paired.py", ["--vgg", "vgg_normalised.pth", "--decoder",
              "decoder_iter_500000.pth", "--transform", "transformer_iter_500000.pth"]),
    "adaattn": ("AdaAttN", "adaattn_paired.py",
                ["--image_encoder_path", "checkpoints/vgg_normalised.pth"]),
    "stytr2": ("StyTR-2", "stytr2_paired.py", []),
    "microast": ("MicroAST", "microast_paired.py", []),
    "aespanet": ("AesPA-Net", "aespanet_paired.py", ["--vgg", BASE + "/SANET/vgg_normalised.pth"]),
}

PREAMBLE = """
import json, os, runpy, sys, atexit, torch
limit = int(os.environ["RESIDENT_LIMIT_BYTES"])
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(limit / total, 1.0), 0)
peaks = []
_orig = torch.cuda.max_memory_allocated
def _record(*a, **k):
    v = _orig(*a, **k)
    peaks.append([v / 1e6, torch.cuda.max_memory_reserved(0) / 1e6])
    return v
torch.cuda.max_memory_allocated = _record
_reset = torch.cuda.reset_peak_memory_stats
atexit.register(lambda: json.dump(peaks, open(os.environ["RESIDENT_PEAKS"], "w")))
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name="__main__")
"""


def query_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    name, total, used, driver = [x.strip() for x in out.split(",")]
    return {"name": name, "total_MiB": int(total), "used_MiB": int(used), "driver": driver}


class WddmSampler(threading.Thread):
    """Peak Dedicated and Shared usage of a process, read from Windows counters."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop_flag = pid, False
        self.peak = {"dedicated_MB": 0.0, "shared_MB": 0.0}
        self.base_shared = None

    def run(self):
        ps = ("$p=(Get-Counter -ListSet 'GPU Process Memory').PathsWithInstances | "
              f"Where-Object {{$_ -like '*pid_{self.pid}_*'}} | "
              "Where-Object {$_ -match 'Dedicated Usage|Shared Usage'}; "
              "if($p){(Get-Counter -Counter $p -ErrorAction SilentlyContinue).CounterSamples | "
              "ForEach-Object {$_.Path + '=' + $_.CookedValue}}")
        while not self.stop_flag:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True).stdout
            ded = sum(float(m) for m in re.findall(r"dedicated usage=([\d.]+)", out.lower()))
            sh = sum(float(m) for m in re.findall(r"shared usage=([\d.]+)", out.lower()))
            if self.base_shared is None and (ded or sh):
                self.base_shared = sh / 1e6
            self.peak["dedicated_MB"] = max(self.peak["dedicated_MB"], ded / 1e6)
            self.peak["shared_MB"] = max(self.peak["shared_MB"], sh / 1e6)
            time.sleep(0.5)


def launch(cmd, cwd, env, log):
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        sampler = WddmSampler(p.pid)
        sampler.start()
        code = p.wait()
        sampler.stop_flag = True
    return code, sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", required=True,
                    help="CUDA Sysmem Fallback Policy as read from the NVIDIA Control Panel")
    ap.add_argument("--headroom_MiB", type=int, default=64,
                    help="kept free below the card's unused memory")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--json_out", default="results/resident_memory.json")
    ap.add_argument("--wrapper_args", default="{}", help="JSON: method -> wrapper arguments")
    ap.add_argument("--fd_sizes", default="256,512,1024,1280,1536,1792,2048",
                    help="FDFormer sweep; the WDDM counters are per run, so a shorter sweep "
                         "attributes a rise in shared memory to a size")
    args = ap.parse_args()

    gpu = query_gpu()
    limit = (gpu["total_MiB"] - gpu["used_MiB"] - args.headroom_MiB) * 1024 ** 2
    rec = {"gpu": gpu, "sysmem_fallback_policy": args.fallback, "limit_MB": limit / 1e6,
           "methods": {}, "time": time.ctime()}
    extra = json.loads(args.wrapper_args)
    tmp = os.path.join(HERE, "logs", "resident")
    os.makedirs(tmp, exist_ok=True)
    py = sys.executable

    jobs = {
        "fdformer_forward": (HERE, os.path.join(HERE, "bench_efficiency.py"),
                             ["--config", "configs/v2_hf_rescaled.yaml", "--no_max_res", "--sweep",
                              "--sweep_sizes", args.fd_sizes]),
        "fdformer_low_memory": (HERE, os.path.join(HERE, "bench_efficiency.py"),
                                ["--config", "configs/v2_hf_rescaled.yaml", "--no_max_res", "--sweep",
                                 "--low_memory", "--sweep_sizes", args.fd_sizes]),
    }
    for m, (d, script, a) in METHODS.items():
        a = extra.get(m, a)
        if a is None:
            print(f"[skip] {m}: wrapper arguments not supplied")
            continue
        jobs[m] = (os.path.join(BASE, d), script, COMMON + a)

    for name, (cwd, script, a) in jobs.items():
        if args.only and name not in args.only:
            continue
        eff = os.path.join(tmp, f"{name}_eff.json")
        peaks = os.path.join(tmp, f"{name}_peaks.json")
        for f in (eff, peaks):
            if os.path.exists(f):
                os.remove(f)
        out_flag = "--json_out" if name.startswith("fdformer") else "--eff_out"
        env = dict(os.environ, RESIDENT_LIMIT_BYTES=str(limit), RESIDENT_PEAKS=peaks)
        code, sampler = launch([py, "-c", PREAMBLE, script, *a, out_flag, eff], cwd, env,
                               os.path.join(tmp, f"{name}.log"))
        r = {"exit_code": code, "wddm_peak": sampler.peak,
             "wddm_shared_at_start_MB": sampler.base_shared}
        if os.path.exists(eff):
            r["sweep"] = json.load(open(eff)).get("sweep")
        if os.path.exists(peaks):
            r["peaks_allocated_reserved_MB"] = json.load(open(peaks))
        rec["methods"][name] = r
        print(name, json.dumps(r.get("sweep")), r["wddm_peak"], flush=True)
        with open(args.json_out, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
