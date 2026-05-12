# Code Walkthrough (`snn_async_delays`)

## 1. Step1 运行流程

入口：`scripts/run_step1.py`

典型命令：
```bash
python -m scripts.run_step1 --config configs/step1_singleop.yaml --sweep
```

执行顺序：
1. 读取配置并合并 CLI 参数。
2. 生成单 slot（`K=1`）时间结构。
3. 构建 `MultiQueryDataset(K=1, same_op=True)` 的 train/val/test。
4. 构建 `SNNModel`。
5. `Trainer.fit()` 训练并保存 `best_model.pt`。
6. `evaluate()` 输出指标到 `eval_results.json`。
7. 绘图：训练曲线、delay heatmap、delay histogram。

## 2. Step2 运行流程

入口：`scripts/run_step2.py`

核心差异：
- 扫描 `K_values`。
- `same_op=True`（默认 NAND）。
- 使用 `sweep.conditions` 对比：
  - `weights_only_d0`
  - `weights_and_delays (continuous)`
  - `weights_and_delays (quantized step=1/2/5)`
- 自动计算 `max_K_by_tau`（支持多个阈值）。
- 生成三联图：`accuracy / K-spike / ops-neuron-time`。

## 3. Step3 运行流程

入口：`scripts/run_step3.py`

核心差异：
- `same_op=False`，每个 query 独立采样 op。
- `n_input` 自动设为 `2 + n_ops`。
- 支持 `op_sampling`：
  - `uniform`
  - `hard_weighted`（可加权 XOR/XNOR）
- 输出混淆矩阵与 operation-wise accuracy。

## 4. 前向计算关键路径

`SNNModel.forward(spike_input, slots)`：
1. 初始化状态：`v_h/ref_h`，以及输入环形延迟缓冲 `buf_in`。
2. 每个时间步：
   - `I_h = syn_ih(buf_in)`
   - `spike_h, v_h, ref_h = lif_h(I_h, v_h, ref_h)`
   - 更新 `buf_in`
3. 在 slot 的 read 窗累积 `hidden_acc[k]`。
4. 对每个 `k` 用 `readout(hidden_acc[k])` 得到 `logit_k`。
5. 返回 `logits[B,K]` 与统计信息 `info`。

## 5. 延迟是如何训练的

在 `synapses.py`：
- 从 `delay_raw` 映射到 `d_cont`（连续）或 `d_quant`（量化）。
- 前向对 `buf[floor]` 与 `buf[ceil]` 插值。
- 反向梯度通过 `alpha` 传给 delay 参数。
- 量化分支通过 STE 保证可训练。

## 6. 评估指标由哪里产生

在 `train/eval.py` 的 `evaluate()`：
- 分类指标：`accuracy`, `per_query_acc`, `binary_confusion`
- 能效/密度：`throughput_K_per_spk`, `ops_per_neuron_per_ms`
- 活跃度：`mean_active_hidden_fraction`
- mixed-op：`op_accuracy`, `op_confusions`

## 7. 你最常改的入口

- 改实验条件：`configs/*.yaml`
- 改模型机制：`snn/synapses.py`, `snn/model.py`, `snn/neurons.py`
- 改指标：`train/eval.py`
- 改图：`utils/viz.py`
- 改 sweep 逻辑：`scripts/run_step2.py`, `scripts/run_step3.py`
