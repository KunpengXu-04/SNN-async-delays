# SNN 异步延迟项目计划（Spec + Next Steps）

下面是一份**可直接丢给 Claude Code 生成代码**的“项目计划 + 实施规格说明（spec）”。内容覆盖背景、总体问题、三步实验路线、网络/数据/训练/评估指标、以及建议的代码结构。

---

## 0. 项目背景与动机（Background & Motivation）

我们关注的不是“延迟（synaptic delays）能不能让 SNN 学得更好”——这点已有工作证明（可训练/可进化延迟能提升任务性能，且在低精度权重下仍鲁棒）。本项目要问更强的问题：

**Overarching question**  
> *We know delays improve performance, but can they help beyond just that?*  
特别是：延迟带来的**异步性（asynchrony）**是否提供额外的**计算优势**，例如在固定神经元规模/能耗预算下，通过时间复用（temporal interleaving / multiplexing）“塞进更多计算”。

**核心假设（Hypothesis）**  
- 同步/分层（layered, synchronous）推理：吞吐受层间传播与等待限制。  
- 引入可学习延迟后：网络进入异步事件驱动（event-based, asynchronous）模式，不同子计算可在时间轴上重叠；神经元不必在 refractory/空档期“闲置”，从而出现“**同一网络、同一试次、更多独立计算**”的可能。  
- 目标是把“延迟=更高吞吐/更高容量”的现象用**严格可测指标**验证，而不仅是 qualitative spike raster。

---

## 1. 总体实验路线（三步）

### Step 1：单任务可解性基线（Single-op solvability）
对每个布尔运算（你写“7个”，但图里像“8个”；建议代码里参数化 `ops_list`，随时可选 7 或 8）分别训练小网络，比较三类模型：

- **Weight-only network**：只训练权重，延迟固定（例如全 0 或小常数）。
- **Delay-only network**：只训练延迟，权重固定（随机初始化后冻结，或用某个固定分布冻结）。
- （可选但强烈建议同时跑）**Weight+Delay network**：权重+延迟都训练，作为上限对照。

> 目的：证明“仅靠延迟/仅靠权重”在简单逻辑上能到什么程度；建立后续容量实验的可重复 baseline。

---

### Step 2：同一运算，多份计算（Same operation, many operations）
固定一个布尔运算（例如 NAND），在**一次试次（trial）内**让网络处理 **K 个独立输入对**（K 从 1 往上扫），看在固定网络规模（固定 hidden 神经元数、固定 trial 长度）下，网络最多能稳定处理多少份。

关键点是“多份计算”要定义成**严格的吞吐任务**，推荐两种实现方式（代码里二选一，或都支持）：

**方案 A：时间分片（temporal slots / packetized queries）**  
- 一个 trial 长度例如 50 ms，划分成 K 个输入窗口，每个窗口注入一对 (A,B)。  
- 输出也对应 K 个窗口：在每个窗口后给一个 readout 时间段，要求输出该窗口对应的结果。  
- 这样“同一网络在一个 trial 内执行 K 次 forward-like 计算”，且可测 throughput。

**方案 B：并行通道（multi-channel inputs）**  
- 输入层复制 K 份通道（每份通道包含 A_i, B_i），同一时段同时输入；输出层也有 K 个输出头。  
- 这更像空间并行，不完全是你要的“时间复用”；因此更推荐方案 A 作为主实验。

> Step 2 的核心指标：在给定能耗/脉冲预算下，**max K** 能到多少；对比 delay-only / weight-only / weight+delay 的差异。

---

### Step 3：不同运算，多份计算（Different operations, many operations）
在 Step 2 的框架上，把“每份计算”从同一运算变成**不同运算混合**。定义为：

- 每个 query 不仅有 (A,B)，还有 op_id（比如 NAND/NOR/XOR…）。  
- 同一个 trial 内注入 K 个 query，每个 query 的 op 可能不同。  
- 输出要求对每个 query 给出对应运算结果。

依然推荐“时间分片方案 A”，因为它最贴近你的“异步多路复用”故事：**不同运算的计算图可以在时间轴上交错**。

---

## 2. 任务与数据定义（Task & Data Spec）

### 2.1 Boolean 运算集合
建议在代码中用可配置列表：
- `ops_list = ["AND","OR","XOR","XNOR","NAND","NOR","A_IMP_B","B_IMP_A"]`（示例）
- 如果你确实只做 7 个，就删一个或按你导师定义的 7 个为准，但**不要写死**。

### 2.2 输入/输出
每个 query 的原始输入为：
- `A ∈ {0,1}, B ∈ {0,1}, op_id ∈ ops_list`

标签：
- `y = op(A,B) ∈ {0,1}`

### 2.3 脉冲编码（建议先固定，后续再换）
先用简单可靠方案，贴合你当前设置（dt=1ms, 50ms, 1kHz）：

- **rate coding**：  
  - A=1 时，A通道在输入窗口内以 `r_on` 发放（如 200–500 Hz）；A=0 时以 `r_off`（如 0–20 Hz）。  
  - B 同理。  
- op_id 编码（Step 3 才需要）：  
  - one-hot 输入通道（每个 op 一个输入神经元），在该 query 窗口内给高频率 `r_op_on`。  

输出解码：
- **spike count / mean membrane** 在 readout 窗口内做二分类，`p(y=1)`。

---

## 3. SNN 模型规格（Model Spec）

### 3.1 神经元模型
- LIF（或你当前代码用的模型），包含 refractory。  
- 可训练延迟：离散延迟（单位 = dt），`d_ij ∈ [0, d_max]`，例如 `d_max = 50ms`。

### 3.2 网络结构（先最小可行）
- 输入层：  
  - Step 1/2：2 个输入通道（A,B）或 4 通道（A,¬A,B,¬B）二选一（建议先 2 通道）。  
  - Step 3：再加 `|ops|` 个 op 通道。  
- 隐层：小网络（例如 10–50 hidden neurons），对齐你图里 50 hidden。  
- 输出层：  
  - Step 1：1 个输出神经元（binary）。  
  - Step 2/3（时间分片）：仍可用 1 个输出，但需要按窗口读取 K 次；或用 1 个线性 readout 对 hidden trace 逐窗口分类（更稳定）。  

### 3.3 三种训练模式（做成配置项）
- `train_mode = "weights_only" | "delays_only" | "weights_and_delays"`
- weights_only：延迟固定为 0 或常数 `d0`
- delays_only：权重固定（初始化后 freeze）
- weights_and_delays：全训练

> **注意**：delay-only 可能更难收敛。建议延迟参数用 `sigmoid` 映射到 `[0,d_max]`，或使用 round 的 straight-through estimator；代码要支持切换参数化方式。

---

## 4. 训练与损失（Training Spec）

### 4.1 Trial 设置
- `dt = 1 ms`
- `T = 50 ms`（Step 2/3 若用 K slots，可扩到 `T = K*slot_len + buffer`，或固定 T 让 slot_len 自动变短）
- 输入窗口：`[t0, t0+win_len]`  
- readout窗口：`[t0+win_len, t0+win_len+read_len]`  
- 可选 `gap_len` 作为窗口隔离。

### 4.2 损失
- 二分类：对每个 query 的 readout 计算 logits：  
  - `logit = α * spike_count + β`（或 mean membrane）  
  - `loss = BCEWithLogitsLoss(logit, y)`
- Step 2/3：一个 trial 有 K 个 query，loss 做平均或加权平均。

### 4.3 正则（先留开关）
- spike penalty：鼓励低发放（能耗 proxy）  
- delay penalty：鼓励更小/更稀疏延迟（避免 trivial solution 把所有输出拖到最后）
- weight decay（可选）

---

## 5. 评估指标与对比方式（Metrics & Evaluation）

### 5.1 Step 1 指标
- per-op accuracy（test set）
- 收敛速度（epochs-to-threshold）
- spike count / firing rate（能耗 proxy）
- learned delays 分布（可视化）

### 5.2 Step 2/3 核心指标（回答“是否能塞更多计算”）
- **Max K @ accuracy ≥ τ**：固定网络规模与 trial 预算下，最大 K，使平均准确率 ≥ τ（如 95%）。
- **Energy-normalized throughput**：  
  - `E = total_spikes_hidden + total_spikes_out`（或加权）  
  - `Throughput = K / E`
- **Latency**：每个 query 从输入结束到 readout 决策的有效延迟。

对比维度：
- weight-only vs delay-only vs weight+delay  
- hidden neurons 扫描（10, 20, 50, 100）  
- 固定 compute budget（例如限制 hidden spikes）下的 max K

---

## 6. 实验矩阵（推荐直接照此跑）

### Step 1：每个 op，三种训练模式
- ops: N（7或8）
- modes: 2（必须）或 3（建议）
- hidden: 10 / 20 / 50（至少一个）
输出：结果表 + 每个 op 最佳 checkpoint + raster/延迟热图（可选）

### Step 2：选一个 op（如 NAND），扫 K
- K: 1..K_max（直到性能崩）  
- modes: weight-only / delay-only / weight+delay  
- hidden 固定（如 50），再做 hidden sweep（可选）
输出：K-accuracy 曲线、K-throughput 曲线、spike/能耗统计

### Step 3：混合 ops，扫 K
- 每个 query 的 op 随机采样（均匀）  
- 其余同 Step 2  
输出：同上 + per-op accuracy（看是否出现共享/干扰）

---

## 7. 代码实现要求（工程规格）

### 7.1 项目结构（建议）
```text
snn_async_delays/
  configs/
    step1_singleop.yaml
    step2_multiquery_sameop.yaml
    step3_multiquery_multiop.yaml
  snn/
    neurons.py
    synapses.py
    model.py
  data/
    boolean_dataset.py
    encoding.py
  train/
    trainer.py
    eval.py
  scripts/
    run_step1.py
    run_step2.py
    run_step3.py
  utils/
    seed.py
    logging.py
    viz.py
```

### 7.2 配置项（必须可配置）
- random seed
- dt, T, slot_len, read_len, gap_len
- ops_list
- hidden_size
- train_mode
- delay_max, delay_init, delay_param_type（sigmoid/softplus/STE等）
- lr_w, lr_d（分开）
- batch_size, epochs
- thresholds: τ accuracy
- energy penalty权重（可选）

### 7.3 输出与复现
- 保存：best checkpoint、训练曲线、最终评估 json/csv
- 记录：每次 run 的 config 原样 dump
- 支持：一键 sweep（for-loop/argparse），输出汇总表

---

## 8. 关键实现细节提醒（避免“跑起来但不回答问题”）

1. **Step 2/3 必须把“多份计算”落到同一 trial 内**，否则只是数据集变大。  
2. 输出解码必须**按窗口读取**，避免模型把所有信息拖到 trial 末尾一次性输出。  
3. delay-only 可能需要更强的初始化/轻微正则避免全静默或全爆发。  
4. 你的核心 claim 是“**同等 compute budget 下，delay-enabled 能支持更高 K**”，所以必须实现能耗 proxy（spike count）并进入对比表。

---
