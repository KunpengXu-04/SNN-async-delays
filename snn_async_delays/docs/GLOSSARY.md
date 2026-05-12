# Glossary (`snn_async_delays`)

## A

### Active Hidden Fraction
每个 trial 至少发放过一次的隐藏神经元占比。对应指标：`mean_active_hidden_fraction`。

## B

### BCEWithLogitsLoss
二分类主损失，直接吃 logits，不需要先 sigmoid。

### Binary Confusion
二分类混淆矩阵：`[[TN, FP], [FN, TP]]`。

## D

### `d_max`
最大延迟（单位：时间步）。缓冲长度为 `d_max + 1`。

### Delay Parameterization
延迟参数化方式：
- `sigmoid`
- `direct`
- `quantized`

### Delay Quantization
将连续延迟按 `delay_step` 量化；反向采用 STE。

### Delay Penalty
对平均延迟施加正则：`loss += lambda_d * mean(delay)`。

## K

### `K`
每个 trial 内并行处理的 query 数（时间复用数量）。

### `K/spike`
能效吞吐指标，对应 `throughput_K_per_spk`。

## L

### LIF Neuron
Leaky Integrate-and-Fire 神经元模型，带 refractory。

## O

### `ops_per_neuron_per_ms`
计算密度指标：`K / (n_hidden * trial_ms)`。

### Operation-wise Accuracy
Step3 下按操作类别分别统计准确率。

## R

### Readout Window
每个 slot 中用于累积 hidden spike 并输出该 query 结果的时间窗口。

### Refractory Steps
发放后抑制时长（时间步）。默认 `2`。

## S

### Slot
一个 query 对应的时间片：输入窗口 + 读出窗口 + 可选 gap。

### Step1 / Step2 / Step3
- Step1：单操作可解性
- Step2：同操作多查询容量
- Step3：混合操作多查询容量

### Surrogate Gradient
对脉冲函数不可导问题的近似梯度方法。

## T

### Train Mode
参数可训练性开关：
- `weights_only`
- `delays_only`
- `weights_and_delays`

### Temporal Multiplexing
同一组神经元在不同时间 slot 复用，承载多个查询计算。

## W

### `weights_only (0ms baseline)`
当前代码里同步基线定义为：权重可训练、延迟固定为 `0ms`。
