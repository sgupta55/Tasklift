#!/usr/bin/env python
"""Wall-clock cost of one 5-shot episode with 8 queries at 256x256 (paper Table V),
for the default (degree-2, F=334) and light (degree-1, F=34) configurations.
Records the hardware, which the paper must name.

  python scripts/benchmark.py --device cuda
  python scripts/benchmark.py --device cpu --threads 8
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift import ModelConfig, TaskLift  # noqa: E402
from tasklift.utils import resolve_device, save_json  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@torch.no_grad()
def time_episode(model, device, n_s=5, n_q=8, size=256, repeats=20, warmup=3):
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(n_s, 4, size, size, generator=g).to(device)
    ys = (torch.rand(n_s, 1, size, size, generator=g) > 0.7).float().to(device)
    xq = torch.randn(n_q, 4, size, size, generator=g).to(device)
    times = []
    for i in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(xs, ys, xq)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t)
    times.sort()
    return {"median_s": times[len(times) // 2], "min_s": times[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="runs/benchmark.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    device = resolve_device(a.device)
    hw = {"device": str(device), "cpu": cpu_name(), "torch": torch.__version__,
          "threads": torch.get_num_threads()}
    if device.type == "cuda":
        hw["gpu"] = torch.cuda.get_device_name(device)
    res = {"hardware": hw}
    for name, deg in (("default_F334", 2), ("light_F34", 1)):
        m = TaskLift(ModelConfig(degree=deg)).to(device).eval()
        res[name] = time_episode(m, device, repeats=a.repeats)
        res[name]["meta_parameters"] = m.num_meta_parameters()
        print(name, res[name])
    print(hw)
    save_json(res, a.out)


if __name__ == "__main__":
    main()
