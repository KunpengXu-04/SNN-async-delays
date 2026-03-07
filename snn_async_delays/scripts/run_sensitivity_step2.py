"""
Step 2 sensitivity scan for NAND at small K.
Scans tau_m, refractory, d_max combinations.
"""

import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from scripts.run_step2 import run_single


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/step2_multiquery_sameop.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = load_cfg(args.config)
    sens = cfg.get("sensitivity", {})

    tau_values = sens.get("lif_tau_m", [5.0, 10.0, 20.0])
    refr_values = sens.get("lif_refractory", [0, 2])
    dmax_values = sens.get("d_max", [20, 49])
    K_values = sens.get("K_values", [1, 2, 4])

    condition = sens.get(
        "condition",
        {
            "name": "w_and_d_cont_sens",
            "train_mode": "weights_and_delays",
            "delay_param_type": "sigmoid",
            "delay_step": 1.0,
            "fixed_delay_value": None,
        },
    )

    results = []

    for tau in tau_values:
        for refr in refr_values:
            for dmax in dmax_values:
                for K in K_values:
                    run_cfg = copy.deepcopy(cfg)
                    run_cfg["lif_tau_m"] = float(tau)
                    run_cfg["lif_refractory"] = int(refr)
                    run_cfg["d_max"] = int(dmax)
                    run_cfg["K"] = int(K)

                    res = run_single(run_cfg, K=K, condition=condition, device=device, base_runs_dir=args.runs_dir)
                    res.update({
                        "lif_tau_m": tau,
                        "lif_refractory": refr,
                        "d_max": dmax,
                    })
                    results.append(res)

    out_json = os.path.join(args.runs_dir, "step2_sensitivity_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    out_csv = os.path.join(args.runs_dir, "step2_sensitivity_summary.csv")
    cols = [
        "condition",
        "K",
        "lif_tau_m",
        "lif_refractory",
        "d_max",
        "accuracy",
        "throughput_K_per_spk",
        "ops_per_neuron_per_ms",
        "mean_hidden_spikes",
        "mean_active_hidden_fraction",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in cols})

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
