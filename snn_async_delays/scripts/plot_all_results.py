"""
Auto-generate mechanism plots for all completed Plan D runs.

Scans runs/step2_planD/, parses run names, and produces:
  - Plot D (2×2 truth table raster)  for K=1 runs
  - Plot C (mechanism raster)        for K>=2 runs

Outputs go to runs/all_results_plots/ with filenames that encode all metadata.

Usage (from snn_async_delays/):
    python -m scripts.plot_all_results                  # sw=10 only
    python -m scripts.plot_all_results --sw all         # all sub-window sizes
    python -m scripts.plot_all_results --condition w_and_d
    python -m scripts.plot_all_results --device cuda
    python -m scripts.plot_all_results --dry_run        # list runs without plotting
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from data.encoding import encode_sequential_trial
from snn.model import SNNSimultaneousModel
from utils.viz import plot_neuron_connection_raster, plot_truth_table_raster

# ─── Run name parser ──────────────────────────────────────────────────────────
# Handles both naming conventions observed in runs/step2_planD/:
#   step2_seq_NAND_w_and_d_continuous_h{h}_K{K}_sw{sw}_seed{seed}[_rtmlp]
#   step2_seq_NAND_weights_only_d0_h{h}_K{K}_sw{sw}_seed{seed}
#   step2_seq_NAND_weights_only_h{h}_K{K}_sw{sw}_seed{seed}[_rtmlp]
#   step2_seq_NAND_weights_and_delays_h{h}_K{K}_sw{sw}_seed{seed}

_PATTERN = re.compile(
    r"step2_seq_NAND_"
    r"(?P<train>[^_]+(?:_[^_]+)*?)_"   # greedy train suffix up to _h
    r"h(?P<h>\d+)_"
    r"K(?P<K>\d+)_"
    r"sw(?P<sw>\d+)_"
    r"seed(?P<seed>\d+)"
    r"(?P<rtmlp>_rtmlp)?$"
)


def _parse_run_name(name: str) -> dict | None:
    """Return metadata dict or None if name doesn't match."""
    m = _PATTERN.match(name)
    if not m:
        return None

    train = m.group("train")
    rtmlp = bool(m.group("rtmlp"))

    # Map train suffix to canonical condition name
    if "w_and_d_continuous" in train or "weights_and_delays" in train:
        cond = "w_and_d_mlp" if rtmlp else "w_and_d"
    elif "weights_only_d0" in train:
        cond = "d0_mlp" if rtmlp else "d0"
    elif "weights_only" in train:
        # no explicit d0 tag — treat as d0 variant
        cond = "d0_mlp" if rtmlp else "d0"
    else:
        return None   # unknown training mode

    return {
        "name":      name,
        "h":         int(m.group("h")),
        "K":         int(m.group("K")),
        "sw":        int(m.group("sw")),
        "seed":      int(m.group("seed")),
        "condition": cond,
        "readout":   "mlp" if rtmlp else "linear",
    }


def discover_runs(runs_dir: str, sw_filter: int | None = 10,
                  condition_filter: str | None = None) -> list[dict]:
    """Return list of parsed run metadata dicts for runs that have best_model.pt."""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "best_model.pt")):
            continue

        meta = _parse_run_name(name)
        if meta is None:
            continue

        if sw_filter is not None and meta["sw"] != sw_filter:
            continue
        if condition_filter is not None and not meta["condition"].startswith(condition_filter):
            continue

        meta["path"] = path
        runs.append(meta)

    return runs


# ─── Model loading & recording ────────────────────────────────────────────────

def _load_model(run_path: str, device: str):
    cfg_path  = os.path.join(run_path, "config.json")
    ckpt_path = os.path.join(run_path, "best_model.pt")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model = SNNSimultaneousModel(
        n_queries        = cfg["K"],
        n_hidden         = cfg["n_hidden"],
        win_len          = cfg["win_len"],
        read_len         = cfg["read_len"],
        d_max            = cfg["d_max"],
        train_mode       = cfg["train_mode"],
        delay_param_type = cfg.get("delay_param_type", "sigmoid"),
        delay_step       = cfg.get("delay_step", 1.0),
        fixed_delay_value= cfg.get("fixed_delay_value", None),
        lif_tau_m        = cfg["lif_tau_m"],
        lif_threshold    = cfg["lif_threshold"],
        lif_reset        = cfg["lif_reset"],
        lif_refractory   = cfg["lif_refractory"],
        dt               = cfg["dt"],
        surrogate_beta   = cfg["surrogate_beta"],
        n_input_channels = cfg.get("n_input", 2),
        readout_type     = cfg.get("readout_type", "linear"),
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, cfg


def _extract_weights(model, cfg):
    d_ih = model.get_delays()["ih"].detach().cpu().numpy()
    w_ih = model.syn_ih.weight.detach().cpu().numpy()
    if cfg.get("readout_type", "linear") == "linear":
        w_ro = model.readout.weight.detach().cpu().numpy()
    else:
        w_ro = model.readout[0].weight.detach().cpu().numpy()
    return d_ih, w_ih, w_ro


@torch.no_grad()
def _record_fixed(model, cfg, A_val: float, B_val: float,
                  device: str, seed: int = 42):
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
def _record_trial(model, cfg, device: str, seed: int = 999):
    """Record one random trial for K>=2 raster plots."""
    from data.boolean_dataset import MultiQueryDataset
    ds = MultiQueryDataset(K=cfg["K"], n_samples=1, same_op=True,
                           op_name="NAND", ops_list=cfg["ops_list"], seed=seed)
    A, B, op_ids, labels = ds[0]

    torch.manual_seed(seed)
    spike_input = encode_sequential_trial(
        A.unsqueeze(0), B.unsqueeze(0),
        win_len=cfg["win_len"], read_len=cfg["read_len"],
        r_on=cfg["r_on"], r_off=cfg["r_off"],
        dt=cfg["dt"], device=device,
    )
    logits, info = model(spike_input.to(device), record=True)
    s_in  = spike_input[0].cpu().numpy()
    s_hid = info["hidden_spike_train"][0].numpy()
    label_str = "  ".join(
        f"Q{k}={'1' if labels[k].item()>0.5 else '0'}" for k in range(cfg["K"])
    )
    return s_in, s_hid, label_str


# ─── Accuracy helper ──────────────────────────────────────────────────────────

def _load_acc(run_path: str) -> float | None:
    p = os.path.join(run_path, "eval_results.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("accuracy") or d.get("val_acc")


# ─── Plot drivers ─────────────────────────────────────────────────────────────

def _plot_truth_table(meta: dict, out_dir: str, device: str, n_connections: int):
    model, cfg = _load_model(meta["path"], device)
    d_ih, w_ih, w_ro = _extract_weights(model, cfg)
    acc = _load_acc(meta["path"])
    acc_str = f"{acc*100:.1f}%" if acc is not None else "?"

    truth_table = [(0.0, 0.0, 1), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]
    panels = []
    for A_val, B_val, expected in truth_table:
        s_in, s_hid, pred = _record_fixed(model, cfg, A_val, B_val, device)
        ok = "✓" if pred == expected else "✗"
        lbl = f"A={int(A_val)}, B={int(B_val)}  →  NAND={expected}  [pred={pred} {ok}]"
        panels.append({"spike_input": s_in, "hidden_spikes": s_hid, "label": lbl})

    fname = (f"plot_D_{meta['condition']}_h{meta['h']}_K{meta['K']}"
             f"_sw{meta['sw']}_seed{meta['seed']}.png")
    save_path = os.path.join(out_dir, fname)

    plot_truth_table_raster(
        panels=panels,
        save_path=save_path,
        win_len=cfg["win_len"],
        read_len=cfg["read_len"],
        delays_ih=d_ih,
        weights_ih=w_ih,
        weights_readout=w_ro,
        K=1,
        sub_win=cfg.get("sub_win"),
        n_connections=n_connections,
        suptitle=(f"NAND Truth Table — {meta['condition']}  "
                  f"h={meta['h']} K=1 sw={meta['sw']} seed={meta['seed']}  "
                  f"(acc={acc_str})"),
    )
    return fname


def _plot_raster(meta: dict, out_dir: str, device: str, n_connections: int):
    model, cfg = _load_model(meta["path"], device)
    d_ih, w_ih, w_ro = _extract_weights(model, cfg)
    acc = _load_acc(meta["path"])
    acc_str = f"{acc*100:.1f}%" if acc is not None else "?"

    s_in, s_hid, lbl = _record_trial(model, cfg, device)

    fname = (f"plot_C_{meta['condition']}_h{meta['h']}_K{meta['K']}"
             f"_sw{meta['sw']}_seed{meta['seed']}.png")
    save_path = os.path.join(out_dir, fname)

    plot_neuron_connection_raster(
        spike_input=s_in,
        hidden_spikes=s_hid,
        save_path=save_path,
        win_len=cfg["win_len"],
        read_len=cfg["read_len"],
        delays_ih=d_ih,
        weights_ih=w_ih,
        weights_readout=w_ro,
        title=(f"{meta['condition']}  h={meta['h']} K={meta['K']} "
               f"sw={meta['sw']} seed={meta['seed']}  (acc={acc_str})  |  {lbl}"),
        K=meta["K"],
        sub_win=cfg.get("sub_win"),
        n_connections=n_connections,
    )
    return fname


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--runs_dir", default=os.path.join(base, "runs", "step2_planD"))
    parser.add_argument("--out_dir",  default=os.path.join(base, "runs", "all_results_plots"))
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--sw",        default="10",
                        help="Sub-window filter: integer or 'all' (default: 10)")
    parser.add_argument("--condition", default=None,
                        help="Filter by condition prefix, e.g. 'w_and_d' or 'd0'")
    parser.add_argument("--n_connections", type=int, default=12)
    parser.add_argument("--dry_run",   action="store_true",
                        help="List discovered runs without generating plots")
    args = parser.parse_args()

    sw_filter = None if args.sw == "all" else int(args.sw)
    runs = discover_runs(args.runs_dir, sw_filter=sw_filter,
                         condition_filter=args.condition)

    print(f"Discovered {len(runs)} run(s)  [sw={args.sw}, condition={args.condition or 'all'}]")
    print()

    if args.dry_run:
        for r in runs:
            acc = _load_acc(r["path"])
            acc_str = f"{acc*100:.1f}%" if acc is not None else "?"
            print(f"  K={r['K']:2d}  h={r['h']:3d}  sw={r['sw']:2d}  "
                  f"seed={r['seed']:2d}  {r['condition']:12s}  acc={acc_str}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    n_truth = sum(1 for r in runs if r["K"] == 1)
    n_raster = sum(1 for r in runs if r["K"] >= 2)
    print(f"  Truth table plots (K=1): {n_truth}")
    print(f"  Raster plots     (K≥2): {n_raster}")
    print(f"  Output dir: {args.out_dir}")
    print()

    for i, meta in enumerate(runs, 1):
        try:
            if meta["K"] == 1:
                fname = _plot_truth_table(meta, args.out_dir, args.device,
                                          args.n_connections)
                tag = "D"
            else:
                fname = _plot_raster(meta, args.out_dir, args.device,
                                     args.n_connections)
                tag = "C"
            print(f"  [{i:3d}/{len(runs)}] Plot {tag} → {fname}")
        except Exception as e:
            print(f"  [{i:3d}/{len(runs)}] ERROR {meta['name']}: {e}")

    print(f"\nDone. {len(runs)} plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
