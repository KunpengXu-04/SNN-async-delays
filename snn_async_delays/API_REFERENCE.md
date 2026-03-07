# API Reference (`snn_async_delays`)

本文档对应当前代码实现（2026-03-07），重点覆盖 Step1/2/3 主流程会调用的接口。

## 1. `data.boolean_dataset`

### `OPS_LIST`
固定操作顺序：`AND, OR, XOR, XNOR, NAND, NOR, A_IMP_B, B_IMP_A`。

### `compute_label(op_name, A, B) -> int`
输入布尔操作名和二值输入，返回 `0/1` 标签。

### `class BooleanDataset`
用于单查询样本（Step1 可用）。

参数：
- `n_samples: int`
- `op_name: Optional[str] = "NAND"`
- `ops_list: List[str] = OPS_LIST`
- `seed: int = 42`

返回单样本：`(A, B, op_id, label)`，均为标量 tensor。

### `class MultiQueryDataset`
用于多查询样本（Step1/2/3 主用）。

参数：
- `K: int`
- `n_samples: int`
- `same_op: bool = True`
- `op_name: Optional[str] = "NAND"`
- `ops_list: List[str] = OPS_LIST`
- `seed: int = 42`
- `op_sampling: str = "uniform"` (`uniform | hard_weighted`)
- `hard_ops: Optional[List[str]] = None`
- `hard_weight: float = 2.0`

返回单样本：`(A[K], B[K], op_ids[K], labels[K])`。

## 2. `data.encoding`

### `encode_trial(...) -> torch.Tensor`
将 batch 输入编码为脉冲序列。

签名：
- `A_batch: [B, K]`
- `B_batch: [B, K]`
- `op_ids: [B, K]`
- `slots: List[SlotBoundaries]`
- `n_input: int`
- `r_on: float = 400.0`
- `r_off: float = 10.0`
- `dt: float = 1.0`
- `device: str = "cpu"`

输出：`spike_input [B, T, n_input]`，其中 `T = slots[-1].read_end`。

规则：
- 通道 0/1：A/B 的 rate coding。
- Step3 下通道 2..：`op_id` one-hot rate coding。

## 3. `snn.neurons`

### `spike_fn(v_minus_thr, beta=4.0)`
前向 Heaviside，反向 surrogate gradient（fast sigmoid derivative）。

### `class LIFNeurons`
参数：
- `n_neurons`
- `tau_m=10.0`
- `v_threshold=1.0`
- `v_reset=0.0`
- `refractory_steps=2`
- `dt=1.0`
- `surrogate_beta=4.0`

方法：
- `init_state(batch_size, device) -> (v, ref)`
- `forward(I_syn, v, ref) -> (spike, v_new, ref_new)`

## 4. `snn.synapses`

### `class DelayedSynapticLayer`
参数：
- `n_pre, n_post`
- `d_max=50`
- `delay_param_type="sigmoid"` (`sigmoid | direct | quantized`)
- `delay_step=1.0`（仅量化延迟模式）
- `fixed_delay_value: float | None = None`
- `train_weights=True`
- `train_delays=True`

关键方法：
- `get_delays()`
  - `sigmoid`: `d_max * sigmoid(delay_raw)`
  - `direct`: `clamp(delay_raw, 0, d_max)`
  - `quantized`: 连续值量化后用 STE 回传
  - 当 `train_delays=False` 且给定 `fixed_delay_value`：返回常数延迟（用于同步基线）
- `forward(buf)`
  - `buf: [B, d_max+1, n_pre]`
  - 返回 `I_syn: [B, n_post]`
  - floor/ceil 线性插值实现可微连续延迟。

## 5. `snn.model`

### `SlotBoundaries`
字段：`win_start, win_end, read_start, read_end`。

### `make_slots(K, win_len, read_len, gap_len=0)`
生成 K 个时隙边界。

### `class SNNModel`
主模型默认结构：
- 输入 -> `syn_ih` -> `lif_h` -> `readout`（默认）
- 可选输出层：`syn_ho + lif_o`

关键参数：
- `n_input, n_hidden, d_max`
- `train_mode`: `weights_only | delays_only | weights_and_delays`
- `delay_param_type, delay_step, fixed_delay_value`
- `use_output_layer=False, n_output=1, readout_source="hidden"`
- `lif_*`, `dt`, `surrogate_beta`

关键行为：
- `weights_only` 且未显式指定时，自动设置 `fixed_delay_value=0.0`（同步基线）。

方法：
- `weight_params()`
- `delay_params()`
- `readout_params()`
- `get_delays()`
- `delay_regularization()`
- `forward(spike_input, slots) -> (logits, info)`

`info` 字段：
- `total_hidden_spikes`
- `total_output_spikes`
- `active_hidden_neurons`
- `active_hidden_fraction`
- `trial_steps`

## 6. `train.trainer`

### `build_optimizer(model, cfg)`
Adam 分组学习率：`lr_w / lr_d / lr_readout`。

### `class Trainer`
方法：
- `_forward_batch(A, B, op_ids, labels)`
- `train_epoch(loader)`
- `eval_epoch(loader)`
- `fit(train_loader, val_loader, epochs)`
- `save_config(cfg)`

损失：
- 主损失：`BCEWithLogitsLoss`
- 可选：`spike_penalty`, `delay_penalty`

## 7. `train.eval`

### `evaluate(model, loader, slots, cfg, device)`
返回核心指标：
- `accuracy`
- `per_query_acc`
- `mean_hidden_spikes`
- `throughput_K_per_spk`
- `ops_per_neuron_per_ms`
- `mean_active_hidden_fraction`
- `binary_confusion`
- `op_accuracy`, `op_confusions`（Step3 mixed-op）
- `K`

### 其他
- `max_K_at_threshold(results_by_K, tau)`
- `save_eval_results(results, path)`
- `load_eval_results(path)`
- `summarize_sweep(results, key_fields=None)`

## 8. `utils.viz`

主要函数：
- `plot_training_curves`
- `plot_delay_distribution`
- `plot_delay_histogram`
- `plot_K_accuracy`
- `plot_throughput`
- `plot_metric_triplet`
- `plot_spike_raster`
- `plot_confusion_matrix`
- `plot_opwise_accuracy`
