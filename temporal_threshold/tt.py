"""Temporal-threshold loss for spiking neural networks (snnTorch, MNIST).

The idea: in addition to the usual per-timestep cross-entropy on the output
membrane potential, penalise any output spike that happens before timestep
``k`` (the "temporal threshold"). The network is pushed to hold off on
committing to an answer until it has integrated ``k`` steps of input.

Two schedules for ``k`` are supported:

* ``static``   - ``k`` is a fixed hyperparameter.
* ``adaptive`` - ``k`` starts at ``k0`` and is nudged every minibatch based on
  how often the spikes emitted at step ``int(k)`` hit the correct class:
  below 60% -> wait longer (k += 0.2); above 80% -> answer earlier (k -= 0.2).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from torchvision import datasets

NUM_INPUTS = 28 * 28
NUM_HIDDEN = 1000
NUM_OUTPUTS = 10
NUM_STEPS = 25
BETA = 0.95
BATCH_SIZE = 128
LR = 5e-4
DATA_DIR = os.environ.get("MNIST_DIR", "/tmp/data/mnist")


class Net(nn.Module):
    """784-1000-10 fully connected LIF network (snnTorch tutorial 5 topology)."""

    def __init__(self, num_steps: int = NUM_STEPS, beta: float = BETA):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(NUM_HIDDEN, NUM_OUTPUTS)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, spk2_rec, mem2_rec = [], [], []
        cur1 = self.fc1(x)  # static input -> same current every step
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(cur1, mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        # shapes: (T, B, hidden), (T, B, 10), (T, B, 10)
        return torch.stack(spk1_rec), torch.stack(spk2_rec), torch.stack(mem2_rec)


def load_mnist(data_dir: str = DATA_DIR):
    """Return MNIST as in-memory tensors scaled to [0, 1] (same as ToTensor())."""
    tr = datasets.MNIST(data_dir, train=True, download=True)
    te = datasets.MNIST(data_dir, train=False, download=True)
    f = lambda d: d.data.float().div_(255.0).view(len(d), -1)
    return (f(tr), tr.targets.clone()), (f(te), te.targets.clone())


def spike_accuracy_at(spk_t: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Of the samples where some output neuron spiked at this step, the fraction
    where the correct neuron was among those that spiked. None if nothing spiked."""
    spiked = spk_t.sum(dim=1) > 0
    n_spiked = int(spiked.sum())
    if n_spiked == 0:
        return None
    correct = spk_t[torch.arange(len(targets)), targets] == 1
    return float((correct & spiked).sum()) / n_spiked


@dataclass
class RunConfig:
    schedule: str = "static"   # "static" or "adaptive"
    k: float = 0.0             # static threshold, or starting value for adaptive
    epochs: int = 2
    seed: int = 0
    lo: float = 0.6            # adaptive: raise k below this spike accuracy
    hi: float = 0.8            # adaptive: lower k above this spike accuracy
    step_size: float = 0.2
    # "mem":   cross-entropy on the output membrane potential at every step
    #          (the original notebooks; silent-but-correct membranes are free)
    # "count": cross-entropy on output spike counts, so staying silent is penalised
    loss_mode: str = "mem"

    @property
    def name(self) -> str:
        suffix = "" if self.loss_mode == "mem" else f"_{self.loss_mode}"
        return f"{self.schedule}{suffix}_k{self.k:g}_ep{self.epochs}_s{self.seed}"


@dataclass
class RunResult:
    config: dict
    test_acc: float                 # spike-count readout over all T steps
    mem_readout_acc: float          # argmax of summed output membrane (no spikes needed)
    first_spike_acc: float          # class of first output spike (no spike = wrong)
    mean_first_spike_t: float       # over samples that spiked at all
    no_spike_frac: float            # samples whose output layer never spiked
    out_spikes_per_sample: float
    hidden_spikes_per_sample: float
    acc_vs_readout_t: list          # accuracy using spikes from steps [0, t]
    first_spike_hist: list          # count of samples whose first spike is at t (last bin = never)
    k_trace: list = field(default_factory=list)   # adaptive: k after each minibatch
    loss_trace: list = field(default_factory=list)
    train_seconds: float = 0.0


def train(cfg: RunConfig, train_data, test_data, log_every: int = 0) -> RunResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x_tr, y_tr = train_data
    net = Net()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))

    k = float(cfg.k)
    k_trace, loss_trace = [], []
    n_batches = len(x_tr) // BATCH_SIZE  # drop_last, as in the original notebooks
    t0 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(x_tr))
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            data, targets = x_tr[idx], y_tr[idx]
            net.train()
            _, spk_rec, mem_rec = net(data)

            if cfg.schedule == "adaptive":
                # FIX 1: evaluate at the step the threshold currently allows
                #        (the original used a stale loop variable `step`).
                # FIX 2: index the output neuron with `targets`, not `targets - 1`
                #        (MNIST labels are already 0..9).
                check_t = min(int(k), NUM_STEPS - 1)
                acc = spike_accuracy_at(spk_rec[check_t].detach(), targets)
                if acc is not None:  # guard the original ZeroDivisionError
                    if acc < cfg.lo:
                        k += cfg.step_size
                    elif acc > cfg.hi:
                        k -= cfg.step_size
                    k = float(np.clip(k, 0, NUM_STEPS - 1))
                k_trace.append(k)

            loss_val = torch.zeros(1)
            for t in range(NUM_STEPS):
                if t < k:
                    loss_val = loss_val + mse(spk_rec[t], torch.zeros_like(spk_rec[t]))
                if cfg.loss_mode == "mem":
                    loss_val = loss_val + ce(mem_rec[t], targets)
            if cfg.loss_mode == "count":
                # same scale as the mem loss (summed over T steps)
                loss_val = loss_val + NUM_STEPS * ce(spk_rec.sum(0), targets)

            opt.zero_grad()
            loss_val.backward()
            opt.step()
            loss_trace.append(loss_val.item())
            if log_every and b % log_every == 0:
                print(f"  ep{epoch} b{b:4d} loss={loss_val.item():.3f} k={k:.2f}", flush=True)

    secs = time.time() - t0
    metrics = evaluate(net, test_data)
    return RunResult(config=asdict(cfg), k_trace=k_trace[::5], loss_trace=loss_trace[::5],
                     train_seconds=secs, **metrics)


@torch.no_grad()
def evaluate(net: Net, test_data) -> dict:
    net.eval()
    x_te, y_te = test_data
    T = NUM_STEPS
    correct_at_t = torch.zeros(T)
    total = first_correct = no_spike = mem_correct = 0
    first_t_sum = 0.0
    out_spikes = hid_spikes = 0.0
    hist = torch.zeros(T + 1)
    for i in range(0, len(x_te), 1000):
        x, y = x_te[i:i + 1000], y_te[i:i + 1000]
        spk1, spk2, mem2 = net(x)
        n = len(y)
        total += n
        out_spikes += float(spk2.sum())
        hid_spikes += float(spk1.sum())
        mem_correct += int((mem2.sum(0).argmax(1) == y).sum())

        # accuracy if you read out after t steps (cumulative spike count,
        # summed membrane as tie-break; a sample with no spikes yet is wrong)
        cum_spk = spk2.cumsum(0)
        cum_mem = mem2.cumsum(0)
        for t in range(T):
            score = cum_spk[t] + 1e-4 * torch.tanh(cum_mem[t])
            pred = score.argmax(1)
            has = cum_spk[t].sum(1) > 0
            correct_at_t[t] += float(((pred == y) & has).sum())

        # time and identity of the first output spike
        any_spk = spk2.sum(2) > 0                      # (T, B)
        spiked = any_spk.any(0)
        first_t = torch.where(spiked, any_spk.float().argmax(0), torch.full((n,), T))
        hist += torch.bincount(first_t, minlength=T + 1).float()
        no_spike += int((~spiked).sum())
        first_t_sum += float(first_t[spiked].sum())
        ft = first_t.clamp(max=T - 1)
        ar = torch.arange(n)
        # among neurons spiking at the first spike step, take the highest membrane
        s_at, m_at = spk2[ft, ar], mem2[ft, ar]
        first_pred = (s_at * 1e3 + m_at).argmax(1)
        first_correct += int(((first_pred == y) & spiked).sum())

    n_spiked = total - no_spike
    acc_t = (correct_at_t / total).tolist()
    return dict(
        test_acc=acc_t[-1],
        mem_readout_acc=mem_correct / total,
        first_spike_acc=first_correct / total,
        mean_first_spike_t=first_t_sum / max(n_spiked, 1),
        no_spike_frac=no_spike / total,
        out_spikes_per_sample=out_spikes / total,
        hidden_spikes_per_sample=hid_spikes / total,
        acc_vs_readout_t=acc_t,
        first_spike_hist=hist.tolist(),
    )
