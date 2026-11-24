"""Sweep the temporal threshold k (static) and run the adaptive schedule.

Each run is saved to results/runs/<name>.json; existing runs are skipped, so
the sweep can be interrupted and resumed.

    python sweep.py                         # default grid
    python sweep.py --ks 0 4 8 --seeds 0    # quick look
    python sweep.py --loss-mode count --adaptive-k0   # spike-count loss, static only
"""
import argparse
import json
import os
from dataclasses import asdict

import torch

from tt import RunConfig, load_mnist, train

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "results", "runs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 15])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--adaptive-k0", type=float, nargs="*", default=[3.0],
                   help="starting k for adaptive runs (empty to skip)")
    p.add_argument("--loss-mode", choices=["mem", "count"], default="mem",
                   help="mem: CE on membrane (original); count: CE on spike counts")
    p.add_argument("--out-dir", default=RUN_DIR, help="where run JSONs go (plot.py reads results/runs)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    lm = args.loss_mode
    jobs = [RunConfig("static", k, args.epochs, s, loss_mode=lm) for k in args.ks for s in args.seeds]
    jobs += [RunConfig("adaptive", k0, args.epochs, s, loss_mode=lm)
             for k0 in args.adaptive_k0 for s in args.seeds]

    train_data, test_data = load_mnist()
    for i, cfg in enumerate(jobs):
        name = cfg.name
        path = os.path.join(args.out_dir, name + ".json")
        if os.path.exists(path):
            print(f"[{i+1}/{len(jobs)}] {name}: exists, skipping")
            continue
        print(f"[{i+1}/{len(jobs)}] {name} ...", flush=True)
        res = train(cfg, train_data, test_data, log_every=args.log_every)
        with open(path, "w") as f:
            json.dump(asdict(res), f)
        extra = f" final_k={res.k_trace[-1]:.2f}" if res.k_trace else ""
        print(f"    acc={res.test_acc:.4f} first_spike_acc={res.first_spike_acc:.4f} "
              f"first_t={res.mean_first_spike_t:.2f} no_spike={res.no_spike_frac:.3f} "
              f"({res.train_seconds:.0f}s){extra}", flush=True)


if __name__ == "__main__":
    main()
