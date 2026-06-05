"""
Phase-1 example plot generator.

Produces diagnostic figures from existing Plan D runs (no new training required).
After reviewing the outputs, run the h-sweep (run_plan_d_h_sweep.py) to fill in
the missing data points.

Outputs (written to runs/analysis_examples/):
  plot_A_k_vs_neurons_partial.png   -- K vs min-h (partial, existing data only)
  plot_B_efficiency_partial.png     -- Efficiency ratio (partial data)
  plot_C_raster_h20_K2.png          -- Mechanism raster for h=20, K=2
  plot_C_raster_h20_K3.png          -- Mechanism raster for h=20, K=3 (degraded)
  plot_C_raster_h50_K3.png          -- Mechanism raster for h=50, K=3 (MLP)
  plot_D_truth_table_h20_K1.png     -- 2x2 truth table raster for NAND, h=20, K=1

Usage (from snn_async_delays/):
    python -m scripts.plot_analysis_examples
    python -m scripts.plot_analysis_examples --device cuda
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from snn.model import SNNSimultaneousModel
from utils.viz import (
    plot_K_vs_neurons,
    plot_neuron_efficiency,
    plot_neuron_connection_raster,
    plot_truth_table_raster,
)

# ─── Hardcoded results from existing Plan D experiments ──────────────────────
#
# Source: eval_results.json files in runs/step2_planD/
# MLP readout runs use suffix _rtmlp; linear readout runs have no suffix.
#
# Format: (h, K, condition) -> accuracy
KNOWN_ACCS = {
    # h=20, linear readout, seed=42
    (20, 1, "w_and_d"): 0.9490,
    (20, 2, "w_and_d"): 0.9120,
    (20, 3, "w_and_d"): 0.8347,
    # h=20, d=0, linear readout
    (20, 2, "d0"):       0.7640,
    # h=50, MLP readout, seed=42
    (50, 1, "w_and_d_mlp"): 0.9560,   # from step2_sequential_summary.json
    (50, 2, "w_and_d_mlp"): 0.9270,
    (50, 3, "w_and_d_mlp"): 0.9268,   # mean of seeds 42+0 per CLAUDE.md
    (50, 4, "w_and_d_mlp"): 0.8985,
    (50, 5, "w_and_d_mlp"): 0.8629,
    # h=50, d=0, MLP readout
    (50, 2, "d0_mlp"):       0.7815,
    (50, 3, "d0_mlp"):       0.7713,
}

TAU = 0.90   # accuracy threshold for "minimum neurons" metric


def _min_h(K: int, condition: str, tau: float) -> int | None:
    """
    Find minimum h (from tested values) that achieves >= tau for this K+condition.
    Returns None if no tested h achieves tau.
    """
    tested = sorted(
        (h for (h, k, c), acc in KNOWN_ACCS.items()
         if k == K and c == condition and acc >= tau),
    )
    return tested[0] if tested else None


def build_partial_curves(tau: float = TAU):
    """
    Return (min_h_with_delay, min_h_no_delay) dicts keyed by K,
    using only tested h values.
    """
    K_range = list(range(1, 6))

    min_h_wd, min_h_d0 = {}, {}
    for K in K_range:
        # with delay: check both linear (h=20) and MLP (h=50) results
        h_wd = _min_h(K, "w_and_d", tau) or _min_h(K, "w_and_d_mlp", tau)
        h_d0 = _min_h(K, "d0", tau) or _min_h(K, "d0_mlp", tau)
        min_h_wd[K] = h_wd   # None if nothing tested achieves tau
        min_h_d0[K] = h_d0

    return min_h_wd, min_h_d0


# ─── Model loading helpers ────────────────────────────────────────────────────

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "step2_planD")


def _run_dir(h: int, K: int, condition: str, seed: int = 42) -> str:
    suffix_map = {
        "w_and_d":     "w_and_d_continuous",
        "w_and_d_mlp": "w_and_d_continuous",
        "d0":          "weights_only_d0",
    }
    rt_suffix = "_rtmlp" if "mlp" in condition else ""
    name = (f"step2_seq_NAND_{suffix_map[condition]}_h{h}_K{K}_"
            f"sw10_seed{seed}{rt_suffix}")
    return os.path.join(RUNS_DIR, name)


def _load_model(run_path: str, device: str) -> tuple:
    """Load SNNSimultaneousModel from a run directory.  Returns (model, cfg)."""
    cfg_path = os.path.join(run_path, "config.json")
    ckpt_path = os.path.join(run_path, "best_model.pt")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model = SNNSimultaneousModel(
        n_queries       = cfg["K"],
        n_hidden        = cfg["n_hidden"],
        win_len         = cfg["win_len"],
        read_len        = cfg["read_len"],
        d_max           = cfg["d_max"],
        train_mode      = cfg["train_mode"],
        delay_param_type= cfg.get("delay_param_type", "sigmoid"),
        delay_step      = cfg.get("delay_step", 1.0),
        fixed_delay_value= cfg.get("fixed_delay_value", None),
        lif_tau_m       = cfg["lif_tau_m"],
        lif_threshold   = cfg["lif_threshold"],
        lif_reset       = cfg["lif_reset"],
        lif_refractory  = cfg["lif_refractory"],
        dt              = cfg["dt"],
        surrogate_beta  = cfg["surrogate_beta"],
        n_input_channels= cfg.get("n_input", 2),
        readout_type    = cfg.get("readout_type", "linear"),
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, cfg


@torch.no_grad()
def _record_fixed_trial(model, cfg, A_val: float, B_val: float,
                        device: str, seed: int = 42):
    """Run one trial with a fixed (A,B) input pair using deterministic rate coding."""
    from data.encoding import encode_sequential_trial
    K = cfg["K"]
    A_batch = torch.full((1, K), A_val)
    B_batch = torch.full((1, K), B_val)

    torch.manual_seed(seed)
    spike_input = encode_sequential_trial(
        A_batch, B_batch,
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"],
        dt=cfg["dt"], device=device,
    )
    logits, info = model(spike_input.to(device), record=True)
    pred = int(logits[0, 0].item() > 0.0)

    s_in  = spike_input[0].cpu().numpy()
    s_hid = info["hidden_spike_train"][0].numpy()
    return s_in, s_hid, pred


@torch.no_grad()
def _record_trial(model, cfg, K: int, device: str, seed: int = 999):
    """Run one trial with record=True, return spike arrays + model params."""
    ds = MultiQueryDataset(K=K, n_samples=1, same_op=True, op_name="NAND",
                           ops_list=cfg["ops_list"], seed=seed)
    A, B, op_ids, labels = ds[0]

    spike_input = encode_sequential_trial(
        A.unsqueeze(0), B.unsqueeze(0),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"],
        dt=cfg["dt"], device=device,
    )
    logits, info = model(spike_input.to(device), record=True)

    s_in  = spike_input[0].cpu().numpy()              # [T, 2]
    s_hid = info["hidden_spike_train"][0].numpy()     # [T, n_h]
    d_ih  = model.get_delays()["ih"].detach().cpu().numpy()
    w_ih  = model.syn_ih.weight.detach().cpu().numpy()

    # Readout weights: [K, n_h] (linear) or [hidden_r, n_h] for MLP first layer
    if cfg.get("readout_type", "linear") == "linear":
        w_ro = model.readout.weight.detach().cpu().numpy()   # [K, n_h]
    else:
        # For MLP take first linear layer weights as proxy
        w_ro = model.readout[0].weight.detach().cpu().numpy()  # [hidden_r, n_h]

    label_str = "  ".join(
        f"Q{k}={'1' if labels[k].item()>0.5 else '0'}" for k in range(K)
    )
    return s_in, s_hid, d_ih, w_ih, w_ro, label_str


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(base, "runs", "analysis_examples")
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    print("=" * 60)
    print("Phase-1: Example Plot Generation")
    print(f"Output directory: {out_dir}")
    print("=" * 60)

    # ── Plot A: K vs neurons (partial) ────────────────────────────────
    print("\n[1/5] Plot A — K vs Neurons (partial data)")
    min_h_wd, min_h_d0 = build_partial_curves(tau=TAU)
    print(f"  with_delay  : {min_h_wd}")
    print(f"  no_delay    : {min_h_d0}")

    plot_K_vs_neurons(
        K_values=list(range(1, 6)),
        min_h_with_delay=min_h_wd,
        min_h_no_delay=min_h_d0,
        tau=TAU,
        save_path=os.path.join(out_dir, "plot_A_k_vs_neurons_partial.png"),
    )
    print(f"  -> saved plot_A_k_vs_neurons_partial.png")

    # ── Plot B: Efficiency (partial) ──────────────────────────────────
    print("\n[2/5] Plot B — Efficiency (partial data)")
    plot_neuron_efficiency(
        K_values=list(range(1, 6)),
        min_h_with_delay=min_h_wd,
        min_h_no_delay=min_h_d0,
        save_path=os.path.join(out_dir, "plot_B_efficiency_partial.png"),
    )
    print(f"  -> saved plot_B_efficiency_partial.png")

    # ── Plot C: Raster — h=20, K=2 (working) ─────────────────────────
    print("\n[3/5] Plot C — Raster h=20, K=2 (91.2% acc)")
    run_path = _run_dir(h=20, K=2, condition="w_and_d")
    if not os.path.exists(os.path.join(run_path, "best_model.pt")):
        print(f"  [SKIP] model not found at {run_path}")
    else:
        model, cfg = _load_model(run_path, device)
        s_in, s_hid, d_ih, w_ih, w_ro, lbl = _record_trial(model, cfg, K=2, device=device)
        plot_neuron_connection_raster(
            spike_input=s_in, hidden_spikes=s_hid,
            save_path=os.path.join(out_dir, "plot_C_raster_h20_K2.png"),
            win_len=cfg["win_len"], read_len=cfg["read_len"],
            delays_ih=d_ih, weights_ih=w_ih, weights_readout=w_ro,
            title=f"h=20, K=2 (acc≈91.2%)  |  {lbl}",
            K=2, sub_win=cfg["sub_win"],
            n_connections=15,
        )
        print(f"  -> saved plot_C_raster_h20_K2.png  (labels: {lbl})")

    # ── Plot C: Raster — h=20, K=3 (degraded, 83.5%) ─────────────────
    print("\n[4/5] Plot C — Raster h=20, K=3 (83.5% acc, degraded)")
    run_path = _run_dir(h=20, K=3, condition="w_and_d")
    if not os.path.exists(os.path.join(run_path, "best_model.pt")):
        print(f"  [SKIP] model not found at {run_path}")
    else:
        model, cfg = _load_model(run_path, device)
        s_in, s_hid, d_ih, w_ih, w_ro, lbl = _record_trial(model, cfg, K=3, device=device)
        plot_neuron_connection_raster(
            spike_input=s_in, hidden_spikes=s_hid,
            save_path=os.path.join(out_dir, "plot_C_raster_h20_K3.png"),
            win_len=cfg["win_len"], read_len=cfg["read_len"],
            delays_ih=d_ih, weights_ih=w_ih, weights_readout=w_ro,
            title=f"h=20, K=3 (acc≈83.5%, degraded)  |  {lbl}",
            K=3, sub_win=cfg["sub_win"],
            n_connections=15,
        )
        print(f"  -> saved plot_C_raster_h20_K3.png  (labels: {lbl})")

    # ── Plot C: Raster — h=50, K=3 MLP (working, 92.7%) ──────────────
    print("\n[5/5] Plot C — Raster h=50, K=3 MLP (92.7% acc)")
    run_path = _run_dir(h=50, K=3, condition="w_and_d_mlp")
    if not os.path.exists(os.path.join(run_path, "best_model.pt")):
        print(f"  [SKIP] model not found at {run_path}")
    else:
        model, cfg = _load_model(run_path, device)
        s_in, s_hid, d_ih, w_ih, w_ro, lbl = _record_trial(model, cfg, K=3, device=device)
        plot_neuron_connection_raster(
            spike_input=s_in, hidden_spikes=s_hid,
            save_path=os.path.join(out_dir, "plot_C_raster_h50_K3.png"),
            win_len=cfg["win_len"], read_len=cfg["read_len"],
            delays_ih=d_ih, weights_ih=w_ih, weights_readout=w_ro,
            title=f"h=50, K=3 MLP (acc≈92.7%)  |  {lbl}",
            K=3, sub_win=cfg["sub_win"],
            n_connections=20,
        )
        print(f"  -> saved plot_C_raster_h50_K3.png  (labels: {lbl})")

    # ── Plot D: Truth table raster (h=20, K=1) ────────────────────────
    print("\n[6/6] Plot D — NAND Truth Table raster (h=20, K=1)")
    run_path = _run_dir(h=20, K=1, condition="w_and_d")
    if not os.path.exists(os.path.join(run_path, "best_model.pt")):
        print(f"  [SKIP] model not found at {run_path}")
    else:
        model, cfg = _load_model(run_path, device)
        d_ih = model.get_delays()["ih"].detach().cpu().numpy()
        w_ih = model.syn_ih.weight.detach().cpu().numpy()
        if cfg.get("readout_type", "linear") == "linear":
            w_ro = model.readout.weight.detach().cpu().numpy()
        else:
            w_ro = model.readout[0].weight.detach().cpu().numpy()

        # NAND truth table: (A,B) → expected output
        truth_table = [(0.0, 0.0, 1), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]
        panels = []
        for A_val, B_val, expected in truth_table:
            s_in, s_hid, pred = _record_fixed_trial(
                model, cfg, A_val, B_val, device, seed=42
            )
            correct = "✓" if pred == expected else "✗"
            lbl = (f"A={int(A_val)}, B={int(B_val)}  →  NAND={expected}  "
                   f"[pred={pred} {correct}]")
            panels.append({"spike_input": s_in, "hidden_spikes": s_hid, "label": lbl})

        plot_truth_table_raster(
            panels=panels,
            save_path=os.path.join(out_dir, "plot_D_truth_table_h20_K1.png"),
            win_len=cfg["win_len"],
            read_len=cfg["read_len"],
            delays_ih=d_ih,
            weights_ih=w_ih,
            weights_readout=w_ro,
            K=1,
            sub_win=cfg.get("sub_win"),
            n_connections=15,
            suptitle="NAND Truth Table — h=20, K=1  (acc≈94.9%)",
        )
        print(f"  -> saved plot_D_truth_table_h20_K1.png")

    print("\n" + "=" * 60)
    print("Done.  Review PNGs in:", out_dir)
    print("Next step: run scripts/run_plan_d_h_sweep.py to fill complete curves.")
    print("=" * 60)


if __name__ == "__main__":
    main()
