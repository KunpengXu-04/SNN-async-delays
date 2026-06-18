"""
Rename runs/ subdirectories to descriptive characteristic-first names.
Experiment codes (step1/2/3, planA/B/C/D) are moved to parenthetical annotations.

Usage (from snn_async_delays/):
    python -m scripts.rename_runs            # dry-run: show all renames
    python -m scripts.rename_runs --execute  # apply renames
    python -m scripts.rename_runs --runs_dir runs

No config.json patching is needed: run_dir is always passed at call-time
by training/eval scripts, never stored inside config files.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Naming convention ─────────────────────────────────────────────────────
# Format: {scientific_description}_(step_plan_code)
#
# Mode abbreviations:
#   wad = weights_and_delays
#   d0  = weights_only + fixed delay = 0
#   wo  = weights_only (non-zero delay)
#   do  = delays_only

# ── Group-level renames  (top-level dirs under runs/) ─────────────────────
GROUP_RENAMES = {
    "step1":                        "single_op_(step1)",
    "step2_planA":                  "NAND_serial_slots_(step2_planA)",
    "step2_planC":                  "NAND_simul_channels_(step2_planC)",
    "step2_planD":                  "NAND_time_mux_(step2_planD)",
    "step3_planD":                  "8op_mixed_(step3)",
    "step3_planD_4ops":             "4op_mixed_1k_(step3)",
    "step3_planD_4ops_16k":         "4op_mixed_16k_(step3)",
    "step3_planD_4ops_16k_h100":    "4op_mixed_16k_h100_(step3)",
    "step3_planD_4ops_16k_h100_L2": "4op_mixed_16k_2layer_(step3)",
    "planD_h_sweep":                "NAND_neuron_sweep_(planD)",
    "depth_ablation":               "NAND_depth_ablation",
    "timing_ablation":              "NAND_timing_ablation",
    "spiking_output":               "NAND_spiking_out",
    "spiking_output_continued":     "NAND_spiking_out_1k",
    # folders already descriptive — not renamed:
    # smoke_diag_test, all_results_plots, all_results_plots_test,
    # analysis_examples, docs (not a runs folder)
}

# ── Run-level renames  (children of group folders) ───────────────────────
# Key: (group_old_name, run_old_name) → run_new_name
#   group_old_name is used as the key so rename logic is unambiguous.
#   Runs not listed keep their current name unchanged.

RUN_RENAMES = {}

# ── step1 runs ──
_step1_mode = {
    "weights_and_delays": "wad",
    "weights_only":       "wo",
    "delays_only":        "do",
}
_step1_ops = ["AND","OR","XOR","XNOR","NAND","NOR","A_IMP_B","B_IMP_A"]
_step1_hs  = [10, 20, 50]
for _op in _step1_ops:
    for _mode_long, _mode_short in _step1_mode.items():
        for _h in _step1_hs:
            _old = f"step1_{_op}_{_mode_long}_sigmoid_h{_h}_seed42"
            _new = f"{_op}_{_mode_short}_h{_h}_seed42"
            RUN_RENAMES[("step1", _old)] = _new

# ── step2_planA runs ──
for _K in [1,2,4,6,8,10,12,14,16,18,20]:
    _old = f"step2_NAND_w_and_d_continuous_h20_K{_K}_seed42"
    RUN_RENAMES[("step2_planA", _old)] = f"wad_h20_K{_K}_seed42"
for _K in [1,2,3,4,5,6,8,10,12]:
    _old = f"step2_NAND_weights_only_d0_h20_K{_K}_seed42"
    RUN_RENAMES[("step2_planA", _old)] = f"d0_h20_K{_K}_seed42"
RUN_RENAMES[("step2_planA", "step2_XOR_weights_and_delays_sigmoid_h20_K1_seed42")] = \
    "XOR_wad_h20_K1_seed42"

# ── step2_planC runs ──
for _K in [1,2,3,4,5,6,8,10,12]:
    for _sfx, _new_pfx in [("w_and_d_continuous", "wad"), ("weights_only_d0", "d0"),
                             ("weights_only_d20",  "wo_d20")]:
        _old = f"step2_simul_NAND_{_sfx}_h20_K{_K}_seed42"
        if os.path.isdir(f"runs/step2_planC/{_old}"):  # best-effort
            RUN_RENAMES[("step2_planC", _old)] = f"{_new_pfx}_h20_K{_K}_seed42"

# ── step2_planD runs ──
# h=20 wad seeds
for _K in [1,2,3,4,5,6,8,10,12]:
    for _seed in [0,1,2,42]:
        _old = f"step2_seq_NAND_w_and_d_continuous_h20_K{_K}_sw10_seed{_seed}"
        RUN_RENAMES[("step2_planD", _old)] = f"wad_h20_K{_K}_sw10_seed{_seed}"
# h=20 wad non-standard sw
for _K, _sw in [(1,20),(1,60),(2,30),(3,6),(3,20),(4,5),(4,15),(5,4),(5,12)]:
    _old = f"step2_seq_NAND_w_and_d_continuous_h20_K{_K}_sw{_sw}_seed42"
    RUN_RENAMES[("step2_planD", _old)] = f"wad_h20_K{_K}_sw{_sw}_seed42"
# h=20 d0
for _K in [1,2,3,4,5,6]:
    for _sw in [10,20,30,60,4,5,6,12,15]:
        _old = f"step2_seq_NAND_weights_only_d0_h20_K{_K}_sw{_sw}_seed42"
        RUN_RENAMES[("step2_planD", _old)] = f"d0_h20_K{_K}_sw{_sw}_seed42"
# h=20 other
RUN_RENAMES[("step2_planD",
    "step2_seq_NAND_weights_and_delays_h20_K3_sw10_seed42")] = "wad_h20_K3_sw10_seed42_v2"
# h=50 wad linear, various seeds
for _K in [1,2,3,4,5,6,8,10,12]:
    for _seed in [0,1,2,42]:
        _old = f"step2_seq_NAND_w_and_d_continuous_h50_K{_K}_sw10_seed{_seed}"
        RUN_RENAMES[("step2_planD", _old)] = f"wad_h50_K{_K}_sw10_seed{_seed}"
    _old = f"step2_seq_NAND_w_and_d_continuous_h50_K{_K}_sw10_seed42_rtmlp"
    RUN_RENAMES[("step2_planD", _old)] = f"wad_h50_K{_K}_sw10_seed42_mlp"
# h=50 d0 mlp
for _K in [2,3,4,5,6]:
    for _seed in [0,42]:
        _old = f"step2_seq_NAND_weights_only_h50_K{_K}_sw10_seed{_seed}_rtmlp"
        RUN_RENAMES[("step2_planD", _old)] = f"d0_h50_K{_K}_sw10_seed{_seed}_mlp"
RUN_RENAMES[("step2_planD",
    "step2_seq_NAND_weights_only_h50_K2_sw10_seed42_rtmlp")] = "d0_h50_K2_sw10_seed42_mlp"

# ── step3_planD runs ──
for _K in [1,2,3,4]:
    for _seed in [0,42]:
        RUN_RENAMES[("step3_planD", f"w_and_d_K{_K}_seed{_seed}")] = f"wad_K{_K}_seed{_seed}"
        RUN_RENAMES[("step3_planD", f"d0_control_K{_K}_seed{_seed}")] = f"d0_K{_K}_seed{_seed}"

# ── step3_planD_4ops ──
for _K in [1,2,3,4]:
    for _seed in [0,42]:
        RUN_RENAMES[("step3_planD_4ops", f"w_and_d_K{_K}_seed{_seed}")] = f"wad_K{_K}_seed{_seed}"
        RUN_RENAMES[("step3_planD_4ops", f"d0_control_K{_K}_seed{_seed}")] = f"d0_K{_K}_seed{_seed}"

# ── step3_planD_4ops_16k ──
for _K in [1,2,3,4]:
    for _seed in [0,42]:
        RUN_RENAMES[("step3_planD_4ops_16k", f"w_and_d_K{_K}_seed{_seed}")] = f"wad_K{_K}_seed{_seed}"
        RUN_RENAMES[("step3_planD_4ops_16k", f"d0_control_K{_K}_seed{_seed}")] = f"d0_K{_K}_seed{_seed}"

# ── step3_planD_4ops_16k_h100 ──
for _K in [1,2,3,4]:
    for _seed in [0,42]:
        RUN_RENAMES[("step3_planD_4ops_16k_h100", f"w_and_d_K{_K}_seed{_seed}")] = f"wad_K{_K}_seed{_seed}"
        RUN_RENAMES[("step3_planD_4ops_16k_h100", f"d0_control_K{_K}_seed{_seed}")] = f"d0_K{_K}_seed{_seed}"

# ── step3_planD_4ops_16k_h100_L2 ──
for _K in [1,2,3,4]:
    for _seed in [0,42]:
        RUN_RENAMES[("step3_planD_4ops_16k_h100_L2", f"w_and_d_K{_K}_seed{_seed}")] = f"wad_K{_K}_seed{_seed}"
        RUN_RENAMES[("step3_planD_4ops_16k_h100_L2", f"d0_control_K{_K}_seed{_seed}")] = f"d0_K{_K}_seed{_seed}"

# ── planD_h_sweep ──
for _h in [10,20,30,50]:
    for _K in [1,2,3,4,5]:
        for _seed in [0,42]:
            RUN_RENAMES[("planD_h_sweep",
                f"planD_sweep_NAND_w_and_d_h{_h}_K{_K}_sw10_seed{_seed}")] = \
                f"wad_h{_h}_K{_K}_sw10_seed{_seed}"
            RUN_RENAMES[("planD_h_sweep",
                f"planD_sweep_NAND_d0_h{_h}_K{_K}_sw10_seed{_seed}")] = \
                f"d0_h{_h}_K{_K}_sw10_seed{_seed}"

# ── depth_ablation ──
for _arch, _new_pfx in [
    ("L1-h50-linear",     "L1_wad_linear"),
    ("L2-h25h25-linear",  "L2_wad_linear"),
    ("L2-h25h25-linear-d0", "L2_d0_linear"),
    ("L2-h25h25-mlp",     "L2_wad_mlp"),
]:
    for _K in [2,3,4]:
        for _seed in [0,42]:
            _old = f"{_arch}_K{_K}_seed{_seed}"
            _new = f"{_new_pfx}_K{_K}_seed{_seed}"
            RUN_RENAMES[("depth_ablation", _old)] = _new

# ── timing_ablation ──
for _cond in ["baseline","read_len20","subwin5","tau20","combined"]:
    for _K in [3,4]:
        for _seed in [0,42]:
            # already clean — no change needed
            pass  # keep as-is

# ── spiking_output ──
RUN_RENAMES[("spiking_output", "spiking_out_wad_K1_seed42")] = "wad_K1_seed42"
RUN_RENAMES[("spiking_output", "spiking_out_d0_K1_seed42")]  = "d0_K1_seed42"

# ── spiking_output_continued ──
RUN_RENAMES[("spiking_output_continued", "spiking_out_wad_K1_seed42")] = "wad_K1_seed42"
RUN_RENAMES[("spiking_output_continued", "spiking_out_d0_K1_seed42")]  = "d0_K1_seed42"


# ── Rename logic ──────────────────────────────────────────────────────────

def build_rename_ops(runs_dir):
    """Return list of (src_abs, dst_abs) pairs in safe execution order.
    Inner renames come first so group-folder renames don't break their paths.
    """
    ops = []

    # 1. Inner run renames (resolve using OLD group names)
    for (group_old, run_old), run_new in RUN_RENAMES.items():
        src = os.path.join(runs_dir, group_old, run_old)
        dst = os.path.join(runs_dir, group_old, run_new)
        if os.path.exists(src) and src != dst:
            ops.append((src, dst))

    # 2. Group-folder renames
    for group_old, group_new in GROUP_RENAMES.items():
        src = os.path.join(runs_dir, group_old)
        dst = os.path.join(runs_dir, group_new)
        if os.path.exists(src) and src != dst:
            ops.append((src, dst))

    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--execute", action="store_true",
                    help="Actually rename (default: dry-run)")
    args = ap.parse_args()

    ops = build_rename_ops(args.runs_dir)

    if not ops:
        print("Nothing to rename.")
        return

    if not args.execute:
        print(f"DRY-RUN — {len(ops)} renames (pass --execute to apply):\n")

    inner = [(s, d) for s, d in ops if os.path.basename(os.path.dirname(s))
             != args.runs_dir]
    group = [(s, d) for s, d in ops if (s, d) not in inner]

    # print inner renames first (run-level)
    for src, dst in ops:
        rel_src = os.path.relpath(src, args.runs_dir)
        rel_dst = os.path.relpath(dst, args.runs_dir)
        print(f"  {rel_src:<70s}  →  {rel_dst}")
        if args.execute:
            if not os.path.exists(src):
                print(f"    [skip] source already gone (previous partial run?)")
                continue
            if os.path.exists(dst):
                print(f"    [skip] destination already exists")
                continue
            try:
                import shutil
                shutil.move(src, dst)
            except Exception as e:
                print(f"    [ERROR] {e}")

    if args.execute:
        print(f"\nApplied (attempted) {len(ops)} renames — check [ERROR] lines above.")
    else:
        print(f"\nRun with --execute to apply.")


if __name__ == "__main__":
    main()
