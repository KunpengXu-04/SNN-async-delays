"""
Continue training spiking output models from their epoch-200 best checkpoints.
Warm-starts so the 90-epoch dead zone is skipped.
Usage (from snn_async_delays/):
    conda run -n snn_async python tmp_continue_spiking.py --epochs 800 --device cuda
"""
import argparse, os, sys, json, csv
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader

from snn.model import SNNSimultaneousModel
from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_sequential_trial
from train.trainer import SimultaneousTrainer
from train.eval import evaluate_simultaneous, save_eval_results
from utils.seed import set_seed
from utils.viz import save_run_diagnostic_plots


SRC_BASE  = "runs/spiking_output"
DST_BASE  = "runs/spiking_output_continued"

RUN_NAMES = [
    "spiking_out_wad_K1_seed42",
    "spiking_out_d0_K1_seed42",
]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_log(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({k: (float(v) if k != "epoch" else int(float(v)))
                              for k, v in row.items()})
    return rows


def build_model(cfg, device):
    return SNNSimultaneousModel(
        n_queries        = cfg["K"],
        n_hidden         = cfg["hidden_sizes"][0],
        win_len          = cfg["win_len"],
        read_len         = cfg["read_len"],
        d_max            = cfg["d_max"],
        train_mode       = cfg.get("train_mode", "weights_and_delays"),
        delay_param_type = cfg.get("delay_param_type", "sigmoid"),
        delay_step       = cfg.get("delay_step", 1.0),
        fixed_delay_value= cfg.get("fixed_delay_value", None),
        lif_tau_m        = cfg.get("lif_tau_m", 10.0),
        lif_threshold    = cfg.get("lif_threshold", 1.0),
        lif_reset        = cfg.get("lif_reset", 0.0),
        lif_refractory   = cfg.get("lif_refractory", 2),
        dt               = cfg.get("dt", 1.0),
        surrogate_beta   = cfg.get("surrogate_beta", 4.0),
        n_input_channels = cfg.get("n_input", 2),
        readout_type     = cfg.get("readout_type", "linear"),
        num_hidden_layers= cfg.get("num_hidden_layers", 1),
        hidden_sizes     = cfg.get("hidden_sizes", [50]),
        use_output_spikes= cfg.get("use_output_spikes", False),
        n_output_neurons = cfg.get("n_output_neurons", None),
    ).to(device)


def make_loader(cfg, split):
    K        = cfg["K"]
    ops_list = cfg.get("ops_list", ["NAND"])
    seed     = cfg.get("seed", 42)
    off      = {"train": 0, "val": 1, "test": 2}[split]
    ds = MultiQueryDataset(K=K, n_samples=cfg[f"n_{split}"], same_op=True,
                           op_name=ops_list[0], ops_list=ops_list, seed=seed + off)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"))


def run(run_name, extra_epochs, device):
    src_dir = os.path.join(SRC_BASE, run_name)
    dst_dir = os.path.join(DST_BASE, run_name)
    os.makedirs(dst_dir, exist_ok=True)

    cfg = load_cfg(os.path.join(src_dir, "config.json"))
    old_log = load_log(os.path.join(src_dir, "train_log.csv"))
    old_best_epoch = max((r["epoch"] for r in old_log), default=0)
    old_epochs = cfg.get("epochs", 200)
    print(f"\n=== {run_name}  (extending {old_epochs} → {old_epochs + extra_epochs} epochs) ===")

    set_seed(cfg.get("seed", 42))
    model = build_model(cfg, device)
    ckpt  = os.path.join(src_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"  Warm-started from {ckpt}  (best epoch was {old_best_epoch})")

    # Trainer reads lr_w / lr_d / lr_readout from cfg
    trainer = SimultaneousTrainer(model, cfg, dst_dir, device,
                                  encode_fn=encode_sequential_trial)

    new_cfg = {**cfg, "epochs": extra_epochs}
    trainer.save_config(new_cfg)

    new_log = trainer.fit(make_loader(cfg, "train"),
                          make_loader(cfg, "val"),
                          extra_epochs)

    # Reload best from this continued run
    model.load_state_dict(torch.load(os.path.join(dst_dir, "best_model.pt"),
                                     map_location=device, weights_only=True))

    results = evaluate_simultaneous(model, make_loader(cfg, "test"), cfg, device,
                                    encode_fn=encode_sequential_trial)
    results.update({
        "model_name" : cfg.get("model_name", run_name),
        "K"          : cfg["K"],
        "seed"       : cfg.get("seed", 42),
        "use_output_spikes": cfg.get("use_output_spikes", False),
        "condition"  : cfg.get("name", run_name),
        "continued_from_epoch": old_epochs,
        "total_epochs": old_epochs + extra_epochs,
    })
    save_eval_results(results, os.path.join(dst_dir, "eval_results.json"))
    print(f"  acc={results.get('accuracy', 0):.3f}  "
          f"spk={results.get('mean_hidden_spikes', 0):.1f}")

    save_run_diagnostic_plots(
        model=model, cfg=new_cfg, log_rows=new_log,
        eval_results=results, run_dir=dst_dir,
        K=cfg["K"], op=cfg.get("ops_list", ["NAND"])[0],
        device=device, seed=cfg.get("seed", 42),
    )
    print(f"  plots -> {dst_dir}/plots/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800,
                    help="Additional epochs to train (default: 800)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(DST_BASE, exist_ok=True)
    for name in RUN_NAMES:
        run(name, args.epochs, args.device)
    print("\nDone. Results in:", DST_BASE)


if __name__ == "__main__":
    main()
