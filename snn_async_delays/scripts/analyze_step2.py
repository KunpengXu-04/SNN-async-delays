"""
Step 2 diagnostic analysis.

Investigates WHY w_and_d_continuous maintains >95% accuracy even at K=20.

Hypotheses:
  H1 - Temporal alignment: delays align input to readout window (d ~ win_len - tau_m)
  H2 - Slot isolation: LIF membrane decay (tau_m=10) prevents cross-slot bleed
  H3 - Trivial multiplexing: network fires proportionally in each slot with no real
       discrimination, relying on the shared readout to threshold the sum
  H4 - Cross-slot memory: long delays (~40 steps) bring info from prior slot
       into current readout window -- but this should be NOISE for same_op tasks
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.model import SNNModel, make_slots
from data.boolean_dataset import MultiQueryDataset
from data.encoding import encode_trial
from torch.utils.data import DataLoader

RUNS_DIR = "runs"
DEVICE   = "cpu"   # analysis on CPU is fine

# ── helpers ──────────────────────────────────────────────────────────────────

def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "eval_results.json")) as f:
        results = json.load(f)
    return cfg, results


def build_model(cfg):
    model = SNNModel(
        n_input=cfg["n_input"],
        n_hidden=cfg["hidden_size"],
        d_max=cfg["d_max"],
        train_mode=cfg["train_mode"],
        delay_param_type=cfg["delay_param_type"],
        delay_step=cfg.get("delay_step", 1.0),
        fixed_delay_value=cfg.get("fixed_delay_value", None),
        use_output_layer=cfg["use_output_layer"],
        readout_source=cfg["readout_source"],
        lif_tau_m=cfg["lif_tau_m"],
        lif_threshold=cfg["lif_threshold"],
        lif_reset=cfg["lif_reset"],
        lif_refractory=cfg["lif_refractory"],
        dt=cfg["dt"],
        surrogate_beta=cfg["surrogate_beta"],
    )
    return model


def load_model(run_dir, cfg):
    model = build_model(cfg)
    state = torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Analysis 1: delay statistics across K ────────────────────────────────────

def analyze_delays():
    print("\n=== Analysis 1: Delay Statistics Across K ===")
    K_list, means, stds, medians = [], [], [], []
    slot_len = 35  # win_len=20 + read_len=10 + gap_len=5

    for K in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        run_dir = os.path.join(RUNS_DIR,
            f"step2_NAND_w_and_d_continuous_h20_K{K}_seed42")
        if not os.path.exists(run_dir):
            continue
        cfg, _ = load_run(run_dir)
        model = load_model(run_dir, cfg)
        d = model.syn_ih.get_delays().detach().numpy().flatten()

        K_list.append(K)
        means.append(d.mean())
        stds.append(d.std())
        medians.append(np.median(d))

        frac_within_slot = (d < slot_len).mean()
        frac_near_dmax   = (d > 35).mean()
        print(f"  K={K:2d}  mean={d.mean():.1f}  std={d.std():.1f}  "
              f"median={np.median(d):.1f}  "
              f"d<35:{frac_within_slot:.2f}  d>35:{frac_near_dmax:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(K_list, means, yerr=stds, fmt='o-', capsize=4, label='mean±std')
    ax.plot(K_list, medians, 's--', label='median')
    ax.axhline(35, color='red', linestyle=':', label='slot_len=35')
    ax.axhline(20, color='orange', linestyle=':', label='win_len=20')
    ax.set_xlabel("K (queries per trial)")
    ax.set_ylabel("Delay (steps)")
    ax.set_title("Delay Statistics vs K  (w_and_d_continuous, NAND, h=20)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RUNS_DIR, "step2_analysis_delays_vs_K.png")
    fig.savefig(out, dpi=120)
    print(f"  -> saved {out}")
    plt.close()


# ── Analysis 2: per-slot spike counts and readout weights ────────────────────

@torch.no_grad()
def analyze_slot_responses(K=8, n_trials=200):
    print(f"\n=== Analysis 2: Per-Slot Responses (K={K}) ===")
    run_dir = os.path.join(RUNS_DIR,
        f"step2_NAND_w_and_d_continuous_h20_K{K}_seed42")
    cfg, results = load_run(run_dir)
    model = load_model(run_dir, cfg)

    slots = make_slots(K, cfg["win_len"], cfg["read_len"], cfg["gap_len"])

    ds = MultiQueryDataset(K=K, n_samples=n_trials, same_op=True,
                           op_name="NAND", ops_list=cfg["ops_list"], seed=99)
    loader = DataLoader(ds, batch_size=n_trials, shuffle=False)
    A, B, op_ids, labels = next(iter(loader))
    A = A.unsqueeze(1) if A.dim() == 1 else A
    B = B.unsqueeze(1) if B.dim() == 1 else B
    op_ids = op_ids.unsqueeze(1) if op_ids.dim() == 1 else op_ids
    labels = labels.unsqueeze(1) if labels.dim() == 1 else labels

    spike_input = encode_trial(A, B, op_ids, slots,
                               n_input=cfg["n_input"],
                               r_on=cfg["r_on"], r_off=cfg["r_off"],
                               dt=cfg["dt"], device=DEVICE)

    logits, info = model(spike_input, slots)
    preds = (logits > 0).float()
    acc_per_slot = (preds == labels).float().mean(dim=0)

    print(f"  Readout weight: {model.readout.weight.data.numpy().flatten()[:5]} ...")
    print(f"  Readout bias:   {model.readout.bias.item():.4f}")
    print(f"  Per-slot accuracy: min={acc_per_slot.min():.4f}  "
          f"max={acc_per_slot.max():.4f}  mean={acc_per_slot.mean():.4f}")

    # Mean spikes per hidden neuron per slot for each label class
    # We need to re-run with per-slot hidden spike tracking
    # Use a manual forward pass
    d_max_1 = model.d_max + 1
    d_cont  = model.syn_ih.get_delays()
    B_size  = spike_input.shape[0]
    T       = spike_input.shape[1]

    v_h, ref_h = model.lif_h.init_state(B_size, DEVICE)
    buf_in  = torch.zeros(B_size, d_max_1, model.n_input)
    buf_ptr = 0

    readout_at = [[] for _ in range(T)]
    for k, slot in enumerate(slots):
        for t in range(slot.read_start, min(slot.read_end, T)):
            readout_at[t].append(k)

    slot_spike_acc = [torch.zeros(B_size, model.n_hidden) for _ in range(K)]

    for t in range(T):
        x_t = spike_input[:, t, :]
        I_h = model.syn_ih(buf_in, d_cont, buf_ptr)
        spike_h, v_h, ref_h = model.lif_h(I_h, v_h, ref_h)
        buf_in[:, buf_ptr, :] = x_t
        buf_ptr = (buf_ptr + 1) % d_max_1
        for k in readout_at[t]:
            slot_spike_acc[k] += spike_h

    # Per-class mean spike counts
    print(f"\n  Mean total hidden spikes per slot (per sample):")
    for k in range(K):
        mean_spk = slot_spike_acc[k].sum(dim=1).mean().item()
        lab_k = labels[:, k]
        spk_k = slot_spike_acc[k].sum(dim=1)
        mean_pos = spk_k[lab_k == 1].mean().item() if (lab_k == 1).any() else float('nan')
        mean_neg = spk_k[lab_k == 0].mean().item() if (lab_k == 0).any() else float('nan')
        print(f"    slot {k:2d}: total={mean_spk:.2f}  label=1: {mean_pos:.2f}  label=0: {mean_neg:.2f}")


# ── Analysis 3: cross-slot contamination ─────────────────────────────────────

@torch.no_grad()
def analyze_cross_slot_bleed(K=8, n_trials=100):
    """
    Check how much hidden activity from slot k bleeds into slot k+1's readout window.
    Method: zero out input for slot k+1, check if hidden_acc[k+1] changes significantly.
    """
    print(f"\n=== Analysis 3: Cross-Slot Bleed (K={K}) ===")
    run_dir = os.path.join(RUNS_DIR,
        f"step2_NAND_w_and_d_continuous_h20_K{K}_seed42")
    cfg, _ = load_run(run_dir)
    model = load_model(run_dir, cfg)
    slots = make_slots(K, cfg["win_len"], cfg["read_len"], cfg["gap_len"])

    ds = MultiQueryDataset(K=K, n_samples=n_trials, same_op=True,
                           op_name="NAND", ops_list=cfg["ops_list"], seed=99)
    loader = DataLoader(ds, batch_size=n_trials, shuffle=False)
    A, B, op_ids, labels = next(iter(loader))
    A = A.unsqueeze(1) if A.dim() == 1 else A
    B = B.unsqueeze(1) if B.dim() == 1 else B
    op_ids = op_ids.unsqueeze(1) if op_ids.dim() == 1 else op_ids

    spike_input = encode_trial(A, B, op_ids, slots,
                               n_input=cfg["n_input"],
                               r_on=cfg["r_on"], r_off=cfg["r_off"],
                               dt=cfg["dt"], device=DEVICE)

    def run_forward(s_in):
        d_max_1 = model.d_max + 1
        d_cont  = model.syn_ih.get_delays()
        B_size  = s_in.shape[0]; T = s_in.shape[1]
        v_h, ref_h = model.lif_h.init_state(B_size, DEVICE)
        buf_in  = torch.zeros(B_size, d_max_1, model.n_input)
        buf_ptr = 0
        readout_at = [[] for _ in range(T)]
        for k, slot in enumerate(slots):
            for t in range(slot.read_start, min(slot.read_end, T)):
                readout_at[t].append(k)
        accs = [torch.zeros(B_size, model.n_hidden) for _ in range(K)]
        for t in range(T):
            x_t = s_in[:, t, :]
            I_h = model.syn_ih(buf_in, d_cont, buf_ptr)
            spike_h, v_h, ref_h = model.lif_h(I_h, v_h, ref_h)
            buf_in[:, buf_ptr, :] = x_t
            buf_ptr = (buf_ptr + 1) % d_max_1
            for k in readout_at[t]:
                accs[k] += spike_h
        return accs

    accs_full = run_forward(spike_input)

    # For each slot k >= 1, zero out slot k's input and measure the change in slot k's acc
    bleed_ratios = []
    for k_zero in range(1, min(K, 5)):
        s_masked = spike_input.clone()
        slot = slots[k_zero]
        s_masked[:, slot.win_start:slot.win_end, :] = 0.0
        accs_masked = run_forward(s_masked)

        full_mean = accs_full[k_zero].sum(dim=1).mean().item()
        masked_mean = accs_masked[k_zero].sum(dim=1).mean().item()
        bleed = masked_mean / (full_mean + 1e-8)
        bleed_ratios.append(bleed)
        print(f"  Slot {k_zero} zeroed: acc_full={full_mean:.3f}  "
              f"acc_masked={masked_mean:.3f}  "
              f"residual_ratio={bleed:.3f}  "
              f"(= fraction from prior-slot bleed)")

    if bleed_ratios:
        print(f"  Mean residual ratio: {np.mean(bleed_ratios):.3f}")
        if np.mean(bleed_ratios) < 0.15:
            print("  -> LOW cross-slot bleed: LIF decay provides good temporal isolation")
        else:
            print("  -> HIGH cross-slot bleed: significant interference between slots")


# ── Analysis 4: what do long-delay neurons actually do? ──────────────────────

@torch.no_grad()
def analyze_long_delay_neurons(K=20):
    print(f"\n=== Analysis 4: Long-Delay Neurons (K={K}) ===")
    run_dir = os.path.join(RUNS_DIR,
        f"step2_NAND_w_and_d_continuous_h20_K{K}_seed42")
    cfg, _ = load_run(run_dir)
    model = load_model(run_dir, cfg)

    d = model.syn_ih.get_delays().detach().numpy()  # [n_pre, n_post]
    w = model.syn_ih.weight.detach().numpy()         # [n_pre, n_post]
    r_w = model.readout.weight.detach().numpy().flatten()  # [n_hidden]

    mean_delay_per_neuron = d.mean(axis=0)  # avg delay per hidden neuron
    slot_len = 35

    print(f"  Readout weight magnitude per hidden neuron:")
    print(f"  {'Neuron':>6}  {'mean_delay':>10}  {'readout_w':>10}  {'category':>12}")
    for j in range(model.n_hidden):
        cat = "LONG-DELAY" if mean_delay_per_neuron[j] > slot_len else "short-delay"
        print(f"  {j:6d}  {mean_delay_per_neuron[j]:10.1f}  {r_w[j]:10.4f}  {cat:>12}")

    # Correlation between mean delay and |readout weight|
    corr = np.corrcoef(mean_delay_per_neuron, np.abs(r_w))[0, 1]
    print(f"\n  Corr(mean_delay, |readout_weight|) = {corr:.4f}")
    if corr < -0.3:
        print("  -> Long-delay neurons tend to have SMALLER readout weights")
        print("     (they contribute less to the decision despite firing)")
    elif corr > 0.3:
        print("  -> Long-delay neurons tend to have LARGER readout weights")
    else:
        print("  -> No clear correlation between delay and readout importance")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    analyze_delays()
    analyze_slot_responses(K=1)
    analyze_slot_responses(K=8)
    analyze_slot_responses(K=20)
    analyze_cross_slot_bleed(K=4)
    analyze_cross_slot_bleed(K=20)
    analyze_long_delay_neurons(K=20)
    print("\nDone.")
