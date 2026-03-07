# SNN Async Project Plan

更新日期：2026-03-07

## 1. 项目主问题

我们已知 delays 可以提升精度，但本项目要回答更强的问题：

在固定神经元规模和时间预算下，delays 是否能通过异步时序复用（temporal multiplexing）提升可承载并行计算量，而不只是提分。

## 2. 当前状态

已完成：
- Step1 主实验代码打通，结果已生成并归档。
- 训练模式已标准化：`weights_only / delays_only / weights_and_delays`。
- 同步基线已修正为 `weights_only + fixed delay=0ms`。
- 延迟机制支持：连续延迟与量化延迟（STE）。
- 指标体系扩展到：`accuracy`, `K/spike`, `ops/neuron/ms`, `active_hidden_fraction`。
- Step2/Step3 脚本支持 condition sweep。

待执行：
- 用当前新口径重跑 Step1（建议先 1 seed，再扩到多 seed）。
- 按 NAND 主线推进 Step2 容量曲线。
- 推进 Step3 mixed-op 并分析 operation-level 干扰与延迟分布。

## 3. 实验路线

### Step1：Single-op solvability
目标：验证 8 个布尔操作在三种训练模式下的可解性与基本能效。

设置：
- `K=1`
- 模式：
  - `weights_only(0ms)`
  - `delays_only`
  - `weights_and_delays`
- 推荐先跑单 seed（42），随后补多 seed（42/43/44）。

输出：
- `step1_sweep_summary.json`
- `step1_summary.csv`（可由汇总脚本生成）
- 图：训练曲线、delay heatmap、delay histogram

验收：
- 排序趋势应接近：`weights_and_delays > weights_only >> delays_only`。
- XOR/XNOR 通常最难。

### Step2：Same op, many ops（先 NAND）
目标：在精度阈值约束下比较最大可承载 K 与能效吞吐。

设置：
- op 固定为 `NAND`
- `K=1..16`
- 条件组：
  - `weights_only_d0`
  - `weights_and_delays_continuous`
  - `weights_and_delays_quantized(step=1/2/5)`

主指标：
- `accuracy@K`
- `throughput_K_per_spk`
- `ops_per_neuron_per_ms`
- `max_K@tau`（建议 `tau=0.95` 和 `0.90`）

附加扫描（小范围）：
- `tau_m in {5,10,20}`
- `refractory in {0,2}`
- `d_max in {20,49}`

### Step3：Different ops, many ops
目标：验证 delay 优势是否在 mixed-op 下仍成立。

设置：
- 每个 query 的 `op_id` 可不同
- 采样策略：
  - `uniform`
  - `hard_weighted`（提高 XOR/XNOR 比重）
- `K` 递增扫描

重点分析：
- operation-wise accuracy
- binary confusion
- 不同条件下的容量与能效对比
- 延迟分布是否出现与任务分工相关的结构

## 4. 绘图与报告最小集合

Step1 必备：
- 训练 loss/acc 曲线
- 延迟热图 + 直方图

Step2/3 必备：
- 三联图：`accuracy-vs-K`、`K/spike-vs-K`、`ops/neuron/ms-vs-K`
- Step3 的 confusion matrix 与 operation-wise accuracy

建议补充：
- 活跃隐藏神经元比例随 K 变化
- 不同条件下延迟分布对比图

## 5. 工程与可复现要求

- 每个 run 保存：`config.json`, `train_log.csv`, `best_model.pt`, `eval_results.json`, `plots/`
- 保留日期化汇总与归档目录，不覆盖历史结果。
- 对外报告区分清楚：
  - 同步基线：`weights_only + delay=0ms`
  - 连续延迟：`delay_param_type=sigmoid`
  - 量化延迟：`delay_param_type=quantized + delay_step`

## 6. 近期执行优先级（建议）

1. 用当前修正版配置重跑 Step1（先 1 seed 快速确认）。
2. 跑 Step2 NAND 条件 sweep，输出 `max_K@0.95/0.90` 与三联图。
3. 跑 Step3（uniform + hard_weighted）并做 operation-wise 分析。
4. 整理成阶段报告：先结论，再证据图，再参数敏感性。
