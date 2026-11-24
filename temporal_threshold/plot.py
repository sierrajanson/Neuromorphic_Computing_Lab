"""Aggregate results/runs/*.json into figures and a summary table.

    python plot.py
writes results/sweep.png, results/readout.png, results/adaptive_k.png and
results/summary.md
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# validated palette (light surface): categorical slots 1-3 + an ordinal blue ramp
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

LABELS = {
    ("static", "mem"): "Static k, membrane loss (original)",
    ("static", "count"): "Static k, spike-count loss",
    ("adaptive", "mem"): "Adaptive k (start 3), at final k",
}
COLORS = {("static", "mem"): BLUE, ("static", "count"): AQUA, ("adaptive", "mem"): ORANGE}


def load_runs():
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RES, "runs", "*.json"))):
        r = json.load(open(p))
        c = r["config"]
        groups[(c["schedule"], c.get("loss_mode", "mem"), c["k"])].append(r)
    return groups


def series(groups, schedule, loss_mode):
    return sorted((k, rs) for (s, lm, k), rs in groups.items() if s == schedule and lm == loss_mode)


def agg(rs, key, scale=1.0):
    v = np.array([r[key] for r in rs if r.get(key) is not None], dtype=float) * scale
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def main():
    groups = load_runs()
    static = series(groups, "static", "mem")
    count = series(groups, "static", "count")
    adaptive = series(groups, "adaptive", "mem")
    if not static:
        raise SystemExit("no runs found in results/runs/")

    # ---- figure 1: the trade-off across k ------------------------------------
    panels = [
        ("test_acc", 100, "Test accuracy (%)", "Accuracy (spike-count readout)"),
        ("mean_first_spike_t", 1, "Timestep (of 25)", "Mean time to first output spike"),
        ("no_spike_frac", 100, "Test images (%)", "Output layer never spikes"),
        ("out_spikes_per_sample", 1, "Spikes per test image", "Output spikes (energy proxy)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    axes = axes.ravel()
    for ax, (key, sc, ylab, title) in zip(axes, panels):
        for sched_lm, items in [(("static", "mem"), static), (("static", "count"), count)]:
            if not items:
                continue
            ks = np.array([k for k, _ in items])
            m = np.array([agg(rs, key, sc)[0] for _, rs in items])
            s = np.array([agg(rs, key, sc)[1] for _, rs in items])
            c = COLORS[sched_lm]
            ax.fill_between(ks, m - s, m + s, color=c, alpha=0.15, linewidth=0)
            ax.plot(ks, m, color=c, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                    label=LABELS[sched_lm])
        for k0, rs in adaptive:
            fk = np.array([r["k_trace"][-1] for r in rs])
            am, asd = agg(rs, key, sc)
            ax.errorbar(fk.mean(), am, xerr=fk.std(), yerr=asd, fmt="D", ms=8, color=ORANGE,
                        mec=SURFACE, mew=1.5, elinewidth=1.5, capsize=3,
                        label=LABELS[("adaptive", "mem")])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Temporal threshold k")
        ax.set_ylabel(ylab)
    axes[1].plot([0, 15.5], [0, 15.5], color=INK2, lw=1, ls=":", zorder=0)
    axes[1].annotate("first spike = k", (12, 12), xytext=(6, -4), textcoords="offset points",
                     fontsize=8, color=INK2)
    axes[1].set_ylim(bottom=-0.5, top=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RES, "sweep.png"), dpi=150)
    plt.close(fig)

    # ---- figure 2: accuracy if you stop and read out at step t ---------------
    panels2 = [(static, "Membrane loss (original)")] + ([(count, "Spike-count loss")] if count else [])
    fig, axes = plt.subplots(1, len(panels2), figsize=(6.2 * len(panels2), 4.2), squeeze=False)
    for ax, (items, title) in zip(axes[0], panels2):
        d = dict(items)
        show = [k for k in [0, 4, 8, 12, 15] if k in d]
        T = len(items[0][1][0]["acc_vs_readout_t"])
        t = np.arange(1, T + 1)
        ends = []
        for color, k in zip(RAMP[len(RAMP) - len(show):], show):
            curve = np.mean([r["acc_vs_readout_t"] for r in d[k]], axis=0) * 100
            ax.plot(t, curve, color=color, lw=2, label=f"k = {k:g}")
            ends.append([curve[-1], k])
        # direct end labels, nudged apart so they never overlap (min 5 pts);
        # skipped when the curves converge (the legend already names them)
        ends.sort()
        if ends[-1][0] - ends[0][0] < 5 * (len(ends) - 1):
            ends = []
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5)
        for y, k in ends:
            ax.annotate(f"k={k:g}", (t[-1], y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        ax.set_xlim(1, T + 3)
        ax.set_ylim(0, 103)
        ax.set_xlabel("Read out after t timesteps")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title, loc="left")
        ax.legend(loc="center right", fontsize=9, title="Threshold", title_fontsize=9,
                  bbox_to_anchor=(0.9, 0.6), ncol=2)
    fig.suptitle("How early can the network answer?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "readout.png"), dpi=150)
    plt.close(fig)

    # ---- figure 3: adaptive threshold trajectory -----------------------------
    if adaptive:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for k0, rs in adaptive:
            for i, r in enumerate(rs):
                kt = np.array(r["k_trace"])
                ax.plot(np.arange(len(kt)) * 5, kt, color=ORANGE, lw=1.5, alpha=0.5 + 0.2 * i,
                        label=f"start k={k0:g} (one line per seed)" if i == 0 else None)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)
        n_total = len(adaptive[0][1][0]["loss_trace"]) * 5
        ax.text(98, 0.15, f"(stays at 0 for the rest of the {n_total} minibatches)",
                ha="right", fontsize=8, color=INK2)
        ax.set_xlabel("Training minibatch")
        ax.set_ylabel("Temporal threshold k")
        ax.set_title("Adaptive threshold during training (bugs fixed)", loc="left")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "adaptive_k.png"), dpi=150)
        plt.close(fig)

    # ---- summary table --------------------------------------------------------
    rows = ["| Loss | Schedule | k | Seeds | Test acc (%) | Membrane-readout acc (%) | "
            "Mean first-spike step | No output spike (%) | Output spikes / img | Hidden spikes / img |",
            "|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(rs, key, sc=1, nd=2):
        m, s = agg(rs, key, sc)
        if np.isnan(m):
            return "n/a"
        return f"{m:.{nd}f}" if len(rs) == 1 else f"{m:.{nd}f} ± {s:.{nd}f}"

    for (sched, lm), items in [(("static", "mem"), static), (("adaptive", "mem"), adaptive),
                               (("static", "count"), count)]:
        for k, rs in items:
            klabel = f"{k:g}" if sched == "static" else \
                f"{k:g} → {np.mean([r['k_trace'][-1] for r in rs]):.1f}"
            rows.append(f"| {lm} | {sched} | {klabel} | {len(rs)} | {fmt(rs, 'test_acc', 100, 1)} | "
                        f"{fmt(rs, 'mem_readout_acc', 100, 1)} | {fmt(rs, 'mean_first_spike_t')} | "
                        f"{fmt(rs, 'no_spike_frac', 100, 1)} | {fmt(rs, 'out_spikes_per_sample', 1, 1)} | "
                        f"{fmt(rs, 'hidden_spikes_per_sample', 1, 0)} |")
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
