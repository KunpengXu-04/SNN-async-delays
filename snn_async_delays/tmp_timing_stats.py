"""Quantify the mechanism difference between timing-ablation conditions
(baseline/tau20/combined, K=3) using the same best-trace selection as
tmp_timing_flow.py, but printing numbers instead of relying on visual
inspection of the (very tall) flow diagrams."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from snn.model import SNNSimultaneousModel
from utils.viz import _extract_run_traces
from utils.seed import set_seed

DEVICE = "cpu"
BASE = "runs/NAND_timing_ablation"
N_SEEDS = 20
RUN_DIRS = {
    "baseline": f"{BASE}/baseline_K3_seed42",
    "tau20":    f"{BASE}/tau20_K3_seed42",
    "combined": f"{BASE}/combined_K3_seed42",
}


def load_model(run_dir, device=DEVICE):
    with open(f"{run_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden_sizes = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)])
    set_seed(cfg.get("seed", 42))
    model = SNNSimultaneousModel(
        n_queries=cfg["K"], n_hidden=hidden_sizes[0], win_len=cfg["win_len"],
        read_len=cfg["read_len"], d_max=cfg["d_max"],
        train_mode=cfg.get("train_mode", "weights_and_delays"),
        delay_param_type=cfg.get("delay_param_type", "sigmoid"),
        delay_step=cfg.get("delay_step", 1.0),
        fixed_delay_value=cfg.get("fixed_delay_value", None),
        lif_tau_m=cfg.get("lif_tau_m", 10.0), lif_threshold=cfg.get("lif_threshold", 1.0),
        lif_reset=cfg.get("lif_reset", 0.0), lif_refractory=cfg.get("lif_refractory", 2),
        dt=cfg.get("dt", 1.0), surrogate_beta=cfg.get("surrogate_beta", 4.0),
        n_input_channels=cfg.get("n_input", 2), readout_type=cfg.get("readout_type", "linear"),
        num_hidden_layers=cfg.get("num_hidden_layers", 1), hidden_sizes=hidden_sizes,
        use_output_spikes=cfg.get("use_output_spikes", False),
        n_output_neurons=cfg.get("n_output_neurons", None),
    ).to(device)
    model.load_state_dict(torch.load(f"{run_dir}/best_model.pt", map_location=device,
                                      weights_only=True))
    model.eval()
    return cfg, model


def stats_for_trace(tr, w_dict, d_dict):
    s_in = tr["input_spikes"]
    s_h1 = tr["hidden1_spikes"]
    win_len = tr["win_len"]
    sub_win = tr.get("sub_win", win_len)
    K = tr.get("K", 1)
    T, h1 = s_h1.shape

    total_spk = int(s_h1.sum())
    active = int((s_h1.sum(axis=0) > 0).sum())
    ro_spk = int(s_h1[win_len:].sum())
    ro_active = int((s_h1[win_len:].sum(axis=0) > 0).sum())

    # per-subwindow spike counts (which sub-window each input-window spike falls in)
    per_sw = []
    for k in range(K):
        lo, hi = k * sub_win, (k + 1) * sub_win
        per_sw.append(int(s_h1[lo:hi].sum()))

    # mean arrival->fire span length for fires in the readout window
    W_ih = w_dict["ih"]
    D_ih = d_dict["ih"]
    tau_m = tr.get("lif_tau_m", 10.0)
    n_in = s_in.shape[1]
    spans = []
    for j in range(h1):
        t_fires = np.where(s_h1[win_len:, j] > 0)[0] + win_len
        for t_fire in t_fires:
            arrivals = []
            for i in range(n_in):
                ts_pre = np.where(s_in[:, i] > 0)[0]
                for t in ts_pre:
                    arr = t + float(D_ih[i, j])
                    if (t_fire - tau_m) <= arr <= t_fire and abs(float(W_ih[i, j])) > 0.08:
                        arrivals.append(arr)
            if arrivals:
                spans.append(t_fire - min(arrivals))

    return dict(total_spk=total_spk, active=active, ro_spk=ro_spk, ro_active=ro_active,
                per_sw=per_sw, mean_span=float(np.mean(spans)) if spans else float("nan"),
                n_ro_fires_with_span=len(spans))


def main():
    for name, run_dir in RUN_DIRS.items():
        cfg, model = load_model(run_dir)
        op = cfg.get("op_name", "NAND")
        best, best_n = None, -1
        for s in range(N_SEEDS):
            tr, w, d = _extract_run_traces(model, cfg, cfg["K"], op, DEVICE, seed=50 * (s + 1))
            n = int(tr["hidden1_spikes"][tr["win_len"]:].sum())
            if n > best_n:
                best, best_n = (tr, w, d), n
        tr, w, d = best
        st = stats_for_trace(tr, w, d)
        print(f"\n=== {name} (tau_m={cfg['lif_tau_m']}, sub_win={cfg['sub_win']}, "
              f"read_len={cfg['read_len']}) ===")
        for k, v in st.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
