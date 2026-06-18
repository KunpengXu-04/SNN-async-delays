"""
Run NAND K=1 experiment with spiking output layer (use_output_spikes=True).
Usage (from snn_async_delays/):
    python -m scripts.run_spiking_output --config configs/step2_nand_K1_spiking_output.yaml
    python -m scripts.run_spiking_output --config configs/step2_nand_K1_spiking_output.yaml --device cuda
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml, torch
from torch.utils.data import DataLoader

from snn.model import SNNSimultaneousModel
from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from train.trainer import SimultaneousTrainer
from train.eval import evaluate_simultaneous, save_eval_results
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(run_cfg, model_cfg, K, runs_dir, device):
    name     = model_cfg["name"]
    seed     = run_cfg.get("seed", 42)
    sub_win  = run_cfg.get("sub_win", 10)
    n_ops    = run_cfg.get("n_ops", 0)
    ops_list = run_cfg.get("ops_list", ["NAND"])
    n_input  = 2 + (n_ops if n_ops > 0 else 0)
    win_len  = K * sub_win
    d_max    = int(run_cfg.get("d_max", win_len + run_cfg.get("read_len", 10) - 1))

    run_cfg = {**run_cfg, "win_len": win_len, "K": K,
               "n_input": n_input, "sub_win": sub_win}

    set_seed(seed)
    run_dir  = os.path.join(runs_dir, f"{name}_K{K}_seed{seed}")
    eval_path = os.path.join(run_dir, "eval_results.json")
    os.makedirs(run_dir, exist_ok=True)

    def make_loader(split):
        n = run_cfg[f"n_{split}"]
        off = {"train": 0, "val": 1, "test": 2}[split]
        ds = MultiQueryDataset(K=K, n_samples=n, same_op=True, op_name=ops_list[0],
                               ops_list=ops_list, seed=seed + off)
        return DataLoader(ds, batch_size=run_cfg["batch_size"], shuffle=(split == "train"))

    hidden_sizes = model_cfg.get("hidden_sizes", [run_cfg.get("n_hidden", 50)])
    num_layers   = model_cfg.get("num_hidden_layers", 1)

    model = SNNSimultaneousModel(
        n_queries          = K,
        n_hidden           = hidden_sizes[0],
        win_len            = win_len,
        read_len           = run_cfg["read_len"],
        d_max              = d_max,
        train_mode         = model_cfg.get("train_mode", "weights_and_delays"),
        delay_param_type   = run_cfg.get("delay_param_type", "sigmoid"),
        delay_step         = run_cfg.get("delay_step", 1.0),
        fixed_delay_value  = model_cfg.get("fixed_delay_value", None),
        lif_tau_m          = run_cfg.get("lif_tau_m", 10.0),
        lif_threshold      = run_cfg.get("lif_threshold", 1.0),
        lif_reset          = run_cfg.get("lif_reset", 0.0),
        lif_refractory     = run_cfg.get("lif_refractory", 2),
        dt                 = run_cfg.get("dt", 1.0),
        surrogate_beta     = run_cfg.get("surrogate_beta", 4.0),
        n_input_channels   = n_input,
        readout_type       = model_cfg.get("readout_type", "linear"),
        num_hidden_layers  = num_layers,
        hidden_sizes       = hidden_sizes,
        use_output_spikes  = model_cfg.get("use_output_spikes", False),
        n_output_neurons   = model_cfg.get("n_output_neurons", None),
    ).to(device)

    saved_cfg = {**run_cfg, **model_cfg,
                 "K": K, "n_input": n_input, "sub_win": sub_win,
                 "win_len": win_len, "d_max": d_max,
                 "num_hidden_layers": num_layers, "hidden_sizes": hidden_sizes,
                 "model_name": name, "experiment": "spiking_output"}

    trainer = SimultaneousTrainer(model, run_cfg, run_dir, device,
                                  encode_fn=encode_sequential_trial)
    trainer.save_config(saved_cfg)

    log_rows = trainer.fit(make_loader("train"), make_loader("val"), run_cfg["epochs"])

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"),
                   map_location=device, weights_only=True))

    results = evaluate_simultaneous(model, make_loader("test"), run_cfg, device,
                                    encode_fn=encode_sequential_trial)
    results.update({"model_name": name, "K": K, "seed": seed,
                     "use_output_spikes": model_cfg.get("use_output_spikes", False),
                     "condition": name})
    save_eval_results(results, eval_path)

    save_run_diagnostic_plots(
        model=model, cfg=saved_cfg, log_rows=log_rows,
        eval_results=results, run_dir=run_dir,
        K=K, op=ops_list[0], device=device, seed=seed,
    )
    print(f"  acc={results.get('accuracy', 0):.3f}  "
          f"spk={results.get('mean_hidden_spikes', 0):.1f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs_dir", default="runs/spiking_output")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg      = load_cfg(args.config)
    sweep    = cfg.get("sweep", {})
    K_values = sweep.get("K_values", [1])
    models   = sweep.get("models", [])
    os.makedirs(args.runs_dir, exist_ok=True)

    for model_cfg in models:
        for K in K_values:
            print(f"\n=== {model_cfg['name']}  K={K} ===")
            run_one(cfg, model_cfg, K, args.runs_dir, args.device)

    print("\nDone. Check:", args.runs_dir)


if __name__ == "__main__":
    main()
