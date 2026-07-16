"""删除没有 diagnostic_data.npz 的 run 里的 enhanced_raster.png 和 enhanced_flow.png"""
import os, sys
sys.path.insert(0, '.')

RUNS_DIR = "runs"
TARGETS  = {"enhanced_raster.png", "enhanced_flow.png"}
dry_run  = "--execute" not in sys.argv

deleted = 0
for group in sorted(os.listdir(RUNS_DIR)):
    group_path = os.path.join(RUNS_DIR, group)
    if not os.path.isdir(group_path): continue
    for run in sorted(os.listdir(group_path)):
        run_path  = os.path.join(group_path, run)
        plot_dir  = os.path.join(run_path, "plots")
        if not os.path.isdir(plot_dir): continue
        has_npz = os.path.exists(os.path.join(plot_dir, "diagnostic_data.npz"))
        if has_npz: continue   # 有 npz 的是正确生成的，保留
        for fname in TARGETS:
            fpath = os.path.join(plot_dir, fname)
            if os.path.exists(fpath):
                rel = os.path.relpath(fpath, RUNS_DIR)
                print(f"  {'DEL' if not dry_run else 'dry'} {rel}")
                if not dry_run:
                    os.remove(fpath)
                deleted += 1

action = "Would delete" if dry_run else "Deleted"
print(f"\n{action} {deleted} files.")
if dry_run:
    print("Pass --execute to actually delete.")
