# Architecture (`snn_async_delays`)

## 1. 项目目标

核心问题：延迟除了提升精度，是否还能通过异步时序复用，在固定神经元规模下提高并行计算密度。

三步实验：
- Step1：单操作可解性（8 个布尔操作，三种训练模式）
- Step2：同操作多查询（NAND，扫描 K）
- Step3：混合操作多查询（mixed-op，扫描 K）

## 2. 系统分层

- `configs/`：实验配置与 sweep 条件
- `data/`：布尔数据与脉冲编码
- `snn/`：神经元、延迟突触、模型
- `train/`：训练与评估
- `scripts/`：Step1/2/3 入口
- `utils/`：可视化、seed、日志

## 3. 网络结构

默认网络为 2 层计算路径：
- 第 1 层：`Input -> DelayedSynapticLayer -> LIF hidden`
- 第 2 层：`Linear readout`

可选分支（默认关闭）：
- 额外输出脉冲层：`hidden -> DelayedSynapticLayer -> LIF output -> readout`

因此你当前默认配置下是“一个隐藏脉冲层 + 一个线性读出层”。

## 4. 延迟机制

`DelayedSynapticLayer` 支持三种延迟参数化：
- `sigmoid`：连续延迟
- `direct`：直接裁剪
- `quantized`：量化延迟 + STE

前向采用 floor/ceil 插值，支持连续可微延迟。

同步基线策略：
- `weights_only` 且未覆盖时，自动固定 `delay=0ms`。

## 5. 时间结构与多查询

每个 query 对应一个 slot：
- 输入窗口：`[win_start, win_end)`
- 读出窗口：`[read_start, read_end)`
- 可选间隔：`gap_len`

`make_slots(K, win_len, read_len, gap_len)` 统一生成时隙。

Step2/3 的关键是同一网络跨 K 个 slot 复用状态与资源，而不是复制 K 份网络。

## 6. 训练模式

- `weights_only`：训权重，延迟冻结（默认固定 0ms）
- `delays_only`：权重冻结，训延迟
- `weights_and_delays`：全部可训

优化器分组学习率：`lr_w`, `lr_d`, `lr_readout`。

## 7. 指标体系

评估重点：
- `accuracy`
- `throughput_K_per_spk`（K/spike）
- `ops_per_neuron_per_ms`
- `mean_active_hidden_fraction`
- Step3：`binary_confusion` 与 `op_accuracy`

## 8. 输出产物

单次 run 目录通常包含：
- `config.json`
- `train_log.csv`
- `best_model.pt`
- `eval_results.json`
- `plots/*.png`

sweep 会额外输出：
- `step1_sweep_summary.json`
- `step2_sweep_summary.json`
- `step3_sweep_summary.json`
- 汇总图（如 `K_metric_triplet.png`）
