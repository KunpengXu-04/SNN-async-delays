"""
make_docs.py
============
文档管理脚本，从 snn_async_delays/ 目录运行。

功能：
  1. 验证所有文档文件是否存在并打印检查清单
  2. 打印重新生成文档的操作指南
  3. 可选：打印 Step1 汇总报告（若 runs/step1_sweep_summary.json 存在）

用法：
  cd snn_async_delays
  python make_docs.py
  python make_docs.py --check-only
  python make_docs.py --show-step1-summary
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# 文档文件清单
# ---------------------------------------------------------------------------

DOC_FILES = [
    ("CODE_WALKTHROUGH.md",                "代码全面解读：每文件深度讲解 + 关键路径分析"),
    ("ARCHITECTURE.md",                    "系统架构与数据流：Mermaid 图 + 张量形状追踪"),
    ("API_REFERENCE.md",                   "API 参考手册：所有公开函数/类的签名与参数"),
    ("GLOSSARY.md",                        "术语表：LIF/代理梯度/延迟/时间槽等核心概念"),
    ("doc_assets/architecture_overview.md","架构概览图集：6 张 Mermaid 图（依赖/流程/延迟机制）"),
]

SOURCE_FILES = [
    "configs/step1_singleop.yaml",
    "configs/step2_multiquery_sameop.yaml",
    "configs/step3_multiquery_multiop.yaml",
    "data/__init__.py",
    "data/boolean_dataset.py",
    "data/encoding.py",
    "snn/__init__.py",
    "snn/neurons.py",
    "snn/synapses.py",
    "snn/model.py",
    "train/__init__.py",
    "train/trainer.py",
    "train/eval.py",
    "scripts/__init__.py",
    "scripts/run_step1.py",
    "scripts/run_step2.py",
    "scripts/run_step3.py",
    "utils/__init__.py",
    "utils/seed.py",
    "utils/logger.py",
    "utils/viz.py",
]

RESULT_FILES = [
    ("runs/step1_sweep_summary.json", "Step1 全扫描结果汇总"),
    ("runs/step2_sweep_summary.json", "Step2 全扫描结果汇总"),
    ("runs/step3_sweep_summary.json", "Step3 全扫描结果汇总"),
]


# ---------------------------------------------------------------------------

def check_exists(path: str) -> bool:
    return os.path.isfile(path)


def print_checklist(title: str, items: list, base: str = "."):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    all_ok = True
    for item in items:
        if isinstance(item, tuple):
            rel_path, desc = item
        else:
            rel_path, desc = item, ""
        full_path = os.path.join(base, rel_path)
        exists = check_exists(full_path)
        status = "OK  " if exists else "MISS"
        if not exists:
            all_ok = False
        size_str = ""
        if exists:
            size = os.path.getsize(full_path)
            size_str = f"  ({size:,} bytes)"
        desc_str = f"  # {desc}" if desc else ""
        print(f"  [{status}]  {rel_path}{size_str}{desc_str}")
    return all_ok


def print_source_check(base: str = "."):
    print(f"\n{'=' * 60}")
    print(f"  源文件完整性检查")
    print(f"{'=' * 60}")
    missing = []
    for f in SOURCE_FILES:
        full = os.path.join(base, f)
        if not os.path.isfile(full):
            missing.append(f)
            print(f"  [MISS]  {f}")
        else:
            print(f"  [OK  ]  {f}")
    return len(missing) == 0


def show_step1_summary(base: str = "."):
    path = os.path.join(base, "runs/step1_sweep_summary.json")
    if not os.path.isfile(path):
        print(f"\n[INFO] Step1 汇总文件不存在: {path}")
        print("       运行以下命令生成:")
        print("       python -m scripts.run_step1 --config configs/step1_singleop.yaml --sweep")
        return

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Step1 扫描结果汇总（共 {len(results)} 次运行）")
    print(f"{'=' * 60}")

    # 按 train_mode 分组
    by_mode = {}
    for r in results:
        mode = r.get("train_mode", "unknown")
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r.get("accuracy", 0.0))

    print(f"\n  {'训练模式':<30} {'平均准确率':>12} {'最低':>8} {'最高':>8} {'≥95%':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for mode, accs in sorted(by_mode.items()):
        n_pass = sum(1 for a in accs if a >= 0.95)
        print(
            f"  {mode:<30} {sum(accs)/len(accs):>12.4f} "
            f"{min(accs):>8.4f} {max(accs):>8.4f} {n_pass:>5}/{len(accs)}"
        )


def print_regen_guide():
    print(f"""
{'=' * 60}
  重新生成文档的操作指南
{'=' * 60}

文档通过 Claude Code 生成（读取所有源文件后手工编写）。
若要更新文档，请在修改源代码后重新运行 Claude Code 并提供如下指令：

  "请重新读取所有源文件并更新 CODE_WALKTHROUGH.md / API_REFERENCE.md / GLOSSARY.md"

以下是各文档的内容来源说明：

  CODE_WALKTHROUGH.md  ← 基于所有源文件逐行分析
  ARCHITECTURE.md      ← 基于 snn/model.py 的 forward 逻辑 + Mermaid 手工绘制
  API_REFERENCE.md     ← 基于函数签名 + docstring 提取
  GLOSSARY.md          ← 基于代码注释 + CLAUDE.md 概念整理
  doc_assets/architecture_overview.md ← Mermaid 图集合

重新生成实验结果图表：
  # Step1 扫描
  python -m scripts.run_step1 --config configs/step1_singleop.yaml --sweep

  # Step2 扫描
  python -m scripts.run_step2 --config configs/step2_multiquery_sameop.yaml --sweep

  # Step3 扫描
  python -m scripts.run_step3 --config configs/step3_multiquery_multiop.yaml --sweep
""")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="文档管理脚本：验证文档文件完整性并打印操作指南"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查文件是否存在，不打印操作指南",
    )
    parser.add_argument(
        "--show-step1-summary",
        action="store_true",
        help="额外打印 Step1 扫描结果汇总",
    )
    parser.add_argument(
        "--base",
        default=".",
        help="项目根目录（默认当前目录）",
    )
    args = parser.parse_args()

    base = args.base

    print(f"\n{'*' * 60}")
    print(f"  snn_async_delays 文档检查工具")
    print(f"  工作目录: {os.path.abspath(base)}")
    print(f"{'*' * 60}")

    # 文档文件检查
    docs_ok = print_checklist("文档文件检查", DOC_FILES, base)

    # 源文件检查
    src_ok = print_source_check(base)

    # 结果文件检查（可选）
    print_checklist("实验结果文件检查", RESULT_FILES, base)

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  检查摘要")
    print(f"{'=' * 60}")
    print(f"  文档文件:  {'全部存在' if docs_ok else '部分缺失（见上方 MISS 标记）'}")
    print(f"  源代码:    {'全部存在' if src_ok else '部分缺失（请检查项目完整性）'}")

    if not args.check_only:
        print_regen_guide()

    if args.show_step1_summary:
        show_step1_summary(base)

    # 退出码
    if not docs_ok:
        print("\n[WARNING] 部分文档文件缺失，请检查生成是否完整。")
        sys.exit(1)
    else:
        print("\n[OK] 所有文档文件已就绪。")
        sys.exit(0)


if __name__ == "__main__":
    main()
