"""Generate diagnostic plots for spiking output runs (rerun after live-run plot failure)."""
import sys, json, csv, torch
sys.path.insert(0, '.')
from snn.model import SNNSimultaneousModel
from utils.viz import save_run_diagnostic_plots

def load_run(run_dir, device='cpu'):
    with open(run_dir + '/config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    with open(run_dir + '/eval_results.json', encoding='utf-8') as f:
        ev = json.load(f)
    log_rows = []
    csv_path = run_dir + '/train_log.csv'
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            log_rows.append({k: (float(v) if k != 'epoch' else int(float(v)))
                              for k, v in row.items()})

    model = SNNSimultaneousModel(
        n_queries=cfg['K'],
        n_hidden=cfg['hidden_sizes'][0],
        win_len=cfg['win_len'],
        read_len=cfg['read_len'],
        d_max=cfg['d_max'],
        train_mode=cfg.get('train_mode', 'weights_and_delays'),
        delay_param_type=cfg.get('delay_param_type', 'sigmoid'),
        delay_step=cfg.get('delay_step', 1.0),
        fixed_delay_value=cfg.get('fixed_delay_value', None),
        lif_tau_m=cfg.get('lif_tau_m', 10.0),
        lif_threshold=cfg.get('lif_threshold', 1.0),
        lif_reset=cfg.get('lif_reset', 0.0),
        lif_refractory=cfg.get('lif_refractory', 2),
        dt=cfg.get('dt', 1.0),
        surrogate_beta=cfg.get('surrogate_beta', 4.0),
        n_input_channels=cfg.get('n_input', 2),
        readout_type=cfg.get('readout_type', 'linear'),
        num_hidden_layers=cfg.get('num_hidden_layers', 1),
        hidden_sizes=cfg.get('hidden_sizes', [50]),
        use_output_spikes=cfg.get('use_output_spikes', False),
        n_output_neurons=cfg.get('n_output_neurons', None),
    ).to(device)
    model.load_state_dict(torch.load(run_dir + '/best_model.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    return cfg, ev, log_rows, model

for run_name in ['spiking_out_wad_K1_seed42', 'spiking_out_d0_K1_seed42']:
    run_dir = f'runs/spiking_output/{run_name}'
    print(f'\n--- {run_name} ---')
    cfg, ev, log_rows, model = load_run(run_dir)
    ops_list = cfg.get('ops_list', ['NAND'])
    save_run_diagnostic_plots(
        model=model, cfg=cfg, log_rows=log_rows,
        eval_results=ev, run_dir=run_dir,
        K=cfg['K'], op=ops_list[0], device='cpu', seed=cfg.get('seed', 42),
    )
    print(f'  plots -> {run_dir}/plots/')

print('\nDone.')
