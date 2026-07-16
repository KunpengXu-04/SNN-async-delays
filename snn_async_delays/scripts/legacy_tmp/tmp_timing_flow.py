"""
One-off: generate enhanced_flow plots for 3 representative conditions of the
timing-parameter ablation (baseline / tau20 / combined, K=3, L1-h50+MLP+delay),
to visually inspect WHY raising tau_m and shrinking sub_win hurts Max K@90%
(fewer/weaker readout-window fires? worse cross-slot isolation?). Re-runs a
single forward pass from each saved checkpoint -- no retraining, no npz needed,
same approach as tmp_mlp_flow.py.

Usage (from snn_async_delays/):
    python tmp_timing_flow.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, '.')
from snn.model import SNNSimultaneousModel
from utils.viz import _extract_run_traces, plot_enhanced_spike_flow
from utils.seed import set_seed

DEVICE = "cpu"
BASE = "runs/NAND_timing_ablation"
N_SEEDS = 20

RUNS = {
    f"{BASE}/baseline_K3_seed42": ("docs/enhanced/flow/timing_baseline_K3_flow.png",
                                    "Timing ablation: baseline (sw=10,read=10,tau=10) K=3"),
    f"{BASE}/tau20_K3_seed42":    ("docs/enhanced/flow/timing_tau20_K3_flow.png",
                                    "Timing ablation: tau20 (sw=10,read=10,tau=20) K=3"),
    f"{BASE}/combined_K3_seed42": ("docs/enhanced/flow/timing_combined_K3_flow.png",
                                    "Timing ablation: combined (sw=5,read=20,tau=20) K=3"),
}


def load_model(run_dir, device=DEVICE):
    with open(f"{run_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden_sizes = cfg.get("hidden_sizes", [cfg.get("n_hidden", 50)])
    set_seed(cfg.get("seed", 42))
    model = SNNSimultaneousModel(
        n_queries         = cfg["K"],
        n_hidden          = hidden_sizes[0],
        win_len           = cfg["win_len"],
        read_len          = cfg["read_len"],
        d_max             = cfg["d_max"],
        train_mode        = cfg.get("train_mode", "weights_and_delays"),
        delay_param_type  = cfg.get("delay_param_type", "sigmoid"),
        delay_step        = cfg.get("delay_step", 1.0),
        fixed_delay_value = cfg.get("fixed_delay_value", None),
        lif_tau_m         = cfg.get("lif_tau_m", 10.0),
        lif_threshold     = cfg.get("lif_threshold", 1.0),
        lif_reset         = cfg.get("lif_reset", 0.0),
        lif_refractory    = cfg.get("lif_refractory", 2),
        dt                = cfg.get("dt", 1.0),
        surrogate_beta    = cfg.get("surrogate_beta", 4.0),
        n_input_channels  = cfg.get("n_input", 2),
        readout_type      = cfg.get("readout_type", "linear"),
        num_hidden_layers = cfg.get("num_hidden_layers", 1),
        hidden_sizes      = hidden_sizes,
        use_output_spikes = cfg.get("use_output_spikes", False),
        n_output_neurons  = cfg.get("n_output_neurons", None),
    ).to(device)
    model.load_state_dict(
        torch.load(f"{run_dir}/best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return cfg, model


def get_best_traces(model, cfg, n_seeds=N_SEEDS):
    op = cfg.get("op_name", cfg.get("ops_list", ["NAND"])[0])
    best, best_n = None, -1
    for s in range(n_seeds):
        tr, w, d = _extract_run_traces(model, cfg, cfg["K"], op, DEVICE, seed=50 * (s + 1))
        n = int(tr["hidden1_spikes"][tr["win_len"]:].sum())
        if n > best_n:
            best, best_n = (tr, w, d), n
    print(f"  best readout-window hidden spikes = {best_n}")
    return best


def main():
    os.makedirs("docs/enhanced/flow", exist_ok=True)
    for run_dir, (save_path, title) in RUNS.items():
        print(f"\n{run_dir}")
        cfg, model = load_model(run_dir)
        traces, w_dict, d_dict = get_best_traces(model, cfg)
        plot_enhanced_spike_flow(traces, w_dict, d_dict, save_path, title=title)
        print(f"  saved: {save_path}")


if __name__ == "__main__":
    main()
