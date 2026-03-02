# 代码全面解读：snn_async_delays

> 本文档基于项目实际源代码逐文件编写，所有函数签名、参数名称、张量形状均直接来自源文件，不含推测内容。

---

## 1. 项目总览

`snn_async_delays` 是一个研究**脉冲神经网络（SNN）突触延迟是否提供超越精度之外的计算优势**的实验项目。核心假说：在固定神经元数量和能量预算下，延迟使能的 SNN 能够在同一试次（trial）内处理更多并行查询（更高的 K），即"异步时间复用"。

**研究设计分三步：**

| 步骤 | 任务 | 关键变量 |
|------|------|----------|
| Step 1 | 单布尔运算可解性基线 | op × train_mode × hidden_size |
| Step 2 | 同一运算，K 查询并发，扫描最大 K | K × train_mode |
| Step 3 | 混合运算，K 查询并发 | K × train_mode，op 作为 one-hot 输入 |

**三种训练模式（train_mode）：**
- `weights_only`：权重可训练，延迟固定为初始值
- `delays_only`：权重随机初始化后冻结，仅延迟可训练
- `weights_and_delays`：权重和延迟均可训练

---

## 2. 从命令到结果：端到端调用链

### 单次运行（Step 1）

```
python -m scripts.run_step1 --config configs/step1_singleop.yaml --op NAND --train_mode weights_and_delays
```

调用链：

```
main()
 └── load_cfg()              # 读取 YAML 配置（UTF-8）
 └── merge_cli()             # CLI 参数覆盖 cfg
 └── run_single(cfg, device, runs_dir)
      ├── set_seed(42)        # 全局随机种子
      ├── make_slots(K=1, win_len=40, read_len=10, gap_len=0)
      │     → [SlotBoundaries(0,40,40,50)]
      ├── MultiQueryDataset(K=1, n=4000, same_op=True, op="NAND")
      ├── DataLoader(batch_size=256)
      ├── SNNModel(n_input=2, n_hidden=50, d_max=49, ...)
      ├── Trainer(model, slots, cfg, run_dir, device)
      │    └── fit(train_loader, val_loader, epochs=200)
      │         ├── train_epoch()
      │         │    └── _forward_batch()
      │         │         ├── encode_trial()       # [B,T,2] spike trains
      │         │         ├── model.forward()      # [B,K] logits
      │         │         └── BCEWithLogitsLoss
      │         └── eval_epoch()  (验证集，@torch.no_grad)
      ├── evaluate(model, test_loader, slots, cfg, device)
      ├── save_eval_results()   → runs/.../eval_results.json
      ├── plot_training_curves()
      └── plot_delay_distribution()
```

**数据流摘要（一个训练批次）：**

```
A[B,K], B[B,K], op_ids[B,K], labels[B,K]
         |
    encode_trial()
         |
spike_input[B, T, n_input]   T = slots[-1].read_end
         |
    SNNModel.forward()
    ├── 时间步循环 t=0..T-1:
    │     buf_in → syn_ih → I_h[B,N_h] → lif_h → spike_h[B,N_h]
    │     buf_in = concat([x_t, buf_in[:,:-1,:]])
    │     readout_at[t] 中的 slot k: hidden_acc[k] += spike_h
    └── logits[B,K] = stack(readout(hidden_acc[k]) for k in K)
         |
    BCEWithLogitsLoss(logits.reshape(-1), labels.reshape(-1))
         |
    loss.backward()  →  optimizer.step()
```

---

## 3. 目录与模块概览

```
snn_async_delays/
├── configs/
│   ├── step1_singleop.yaml          # Step1 超参配置
│   ├── step2_multiquery_sameop.yaml # Step2 超参配置
│   └── step3_multiquery_multiop.yaml # Step3 超参配置
│
├── data/
│   ├── __init__.py                  # 导出 BooleanDataset, MultiQueryDataset, encode_trial
│   ├── boolean_dataset.py           # 布尔运算数据集（Step1/2/3）
│   └── encoding.py                  # 速率编码：(A,B,op_id) → spike trains
│
├── snn/
│   ├── __init__.py                  # 导出 LIFNeurons, DelayedSynapticLayer, SNNModel
│   ├── neurons.py                   # LIF 神经元 + 代理梯度
│   ├── synapses.py                  # 延迟突触层（ring buffer + 线性插值）
│   └── model.py                     # SNNModel 主模型 + SlotBoundaries + make_slots
│
├── train/
│   ├── __init__.py                  # 导出 Trainer, build_optimizer, evaluate 等
│   ├── trainer.py                   # 训练引擎（含分组学习率、checkpoint）
│   └── eval.py                      # 评估指标（accuracy, throughput, max-K）
│
├── scripts/
│   ├── __init__.py                  # 空文件（使 scripts 成为包）
│   ├── run_step1.py                 # Step1 入口脚本
│   ├── run_step2.py                 # Step2 入口脚本
│   └── run_step3.py                 # Step3 入口脚本
│
└── utils/
    ├── __init__.py                  # 导出 set_seed, setup_logger
    ├── seed.py                      # 全局随机种子设置
    ├── logger.py                    # logging 封装
    └── viz.py                       # 可视化工具（训练曲线、延迟分布、K-accuracy）
```

**模块依赖关系（有向）：**

```
scripts/* ──imports──> data, snn, train, utils
train/*   ──imports──> snn, data
data/encoding.py ──imports──> snn.model (SlotBoundaries)
snn/model.py ──imports──> snn.neurons, snn.synapses
```

---

## 4. 逐文件深度讲解

### 4.1 configs/

#### 职责

YAML 格式超参文件，作为所有脚本的唯一配置源。CLI 参数可在运行时覆盖任何字段。

#### step1_singleop.yaml 关键字段

| 字段 | 值 | 含义 |
|------|----|------|
| `dt` | 1.0 | 仿真时间步长（ms） |
| `win_len` | 40 | 每个 slot 的输入窗口长度（时间步） |
| `read_len` | 10 | 每个 slot 的读出窗口长度 |
| `gap_len` | 0 | slot 间隔（Step1 无需间隔） |
| `d_max` | 49 | 最大延迟索引（buffer 大小 = 50） |
| `lif_tau_m` | 10.0 | 膜时间常数（ms） |
| `lif_refractory` | 2 | 不应期（时间步） |
| `surrogate_beta` | 4.0 | 代理梯度陡峭度 |
| `delay_param_type` | sigmoid | 延迟参数化方式 |
| `train_mode` | weights_and_delays | 训练模式 |
| `r_on` / `r_off` | 400.0 / 10.0 | 速率编码的发放率（Hz） |

#### step2/step3 差异

- Step2：`win_len=20, read_len=10, gap_len=5`，slot 长度 35ms；K 范围 1-8
- Step3：`n_input` 自动设为 `2 + n_ops`（加入 one-hot op 通道）；`same_op=false`

---

### 4.2 data/boolean_dataset.py

#### 职责

定义布尔运算标签逻辑和两个 Dataset 类，为所有三步实验提供数据。

#### 关键数据结构

```python
OPS_LIST = ["AND", "OR", "XOR", "XNOR", "NAND", "NOR", "A_IMP_B", "B_IMP_A"]

_OPS = {
    "AND":     lambda a, b: int(bool(a) and bool(b)),
    "NAND":    lambda a, b: int(not (bool(a) and bool(b))),
    "XOR":     lambda a, b: int(bool(a) != bool(b)),
    "A_IMP_B": lambda a, b: int((not bool(a)) or bool(b)),  # A 蕴含 B
    ...
}
```

#### BooleanDataset（Step 1）

```python
class BooleanDataset(Dataset):
    def __init__(self, n_samples, op_name="NAND", ops_list=OPS_LIST, seed=42):
        rng = np.random.RandomState(seed)
        A = rng.randint(0, 2, n_samples).astype(np.float32)
        B = rng.randint(0, 2, n_samples).astype(np.float32)
        op_ids = np.full(n_samples, ops_list.index(op_name), dtype=np.int64)
        labels = np.array([compute_label(ops_list[op_ids[i]], int(A[i]), int(B[i]))
                           for i in range(n_samples)], dtype=np.float32)
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.op_ids[idx], self.labels[idx]
```

- **输入**：n_samples 个样本；op_name 固定时所有样本同一运算
- **输出（每个样本）**：`(A: scalar, B: scalar, op_id: scalar, label: scalar)`
- **与 delay 的关系**：Step1 中 `__getitem__` 返回标量，Trainer 中自动 unsqueeze 为 `[B,1]`

#### MultiQueryDataset（Step 2/3）

```python
class MultiQueryDataset(Dataset):
    def __init__(self, K, n_samples, same_op=True, op_name="NAND", ...):
        A = rng.randint(0, 2, (n_samples, K)).astype(np.float32)
        B = rng.randint(0, 2, (n_samples, K)).astype(np.float32)
        if same_op:
            op_ids = np.full((n_samples, K), ops_list.index(op_name), dtype=np.int64)
        else:  # Step 3: 每个 slot 随机选一个 op
            op_ids = rng.randint(0, n_ops, (n_samples, K)).astype(np.int64)
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.op_ids[idx], self.labels[idx]
        # A: [K], B: [K], op_ids: [K], labels: [K]
```

- **输出（每个样本）**：`(A[K], B[K], op_ids[K], labels[K])`——K 个查询打包在一个样本中
- **与 delay/异步的关系**：K 个查询对应 K 个时间槽（temporal slots），是"时间复用"的数据层表示

**依赖者**：所有三个 `scripts/run_step*.py` 均使用 `MultiQueryDataset`

---

### 4.3 data/encoding.py

#### 职责

将离散布尔输入 `(A, B, op_id)` 转换为脉冲神经网络可处理的 Poisson 脉冲串，形状为 `[B, T, n_input]`。

#### 编码规则

- **速率编码**：A=1 → Poisson 过程，每步发放概率 `p_on = r_on * dt / 1000`；A=0 → `p_off = r_off * dt / 1000`
- **时间槽结构**：每个槽的输入窗口 `[win_start, win_end)` 独立编码，其余时间步置零
- **Step3 one-hot op 通道**：通道 `2..2+n_ops` 中，当前 op 对应通道以 `p_on` 发放

```python
def encode_trial(A_batch, B_batch, op_ids, slots, n_input,
                 r_on=400.0, r_off=10.0, dt=1.0, device="cpu"):
    T = slots[-1].read_end           # 总时间步数
    spike_input = torch.zeros(B, T, n_input, device=device)

    p_on  = r_on  * dt / 1000.0     # 每步发放概率（r_on=400Hz, dt=1ms → p=0.4）
    p_off = r_off * dt / 1000.0     # (r_off=10Hz → p=0.01)

    for k, slot in enumerate(slots):
        pA = torch.where(A_k > 0.5, p_on, p_off)   # [B]
        pA_exp = pA.unsqueeze(1).expand(B, wl)
        spike_input[:, slot.win_start:slot.win_end, 0] = torch.bernoulli(pA_exp)
        # ... 同理处理 B 和 op one-hot
    return spike_input  # [B, T, n_input]
```

| 参数 | 类型/形状 | 含义 |
|------|-----------|------|
| `A_batch` | `[B, K]` float | 批量输入 A（0 或 1） |
| `B_batch` | `[B, K]` float | 批量输入 B（0 或 1） |
| `op_ids` | `[B, K]` long | 运算类型索引（Step3 使用） |
| `slots` | List[SlotBoundaries] | K 个时间槽边界 |
| **返回** | `[B, T, n_input]` | 全试次脉冲矩阵 |

**与 delay 的关系**：encode_trial 产生的稀疏脉冲矩阵直接进入 `SNNModel.forward()`，并经过 `buf_in` 延迟缓冲区以延迟 D 时步后到达隐藏层。

**依赖者**：`train/trainer.py`（`_forward_batch`），`train/eval.py`（`evaluate`）

---

### 4.4 snn/neurons.py

#### 职责

实现 LIF（Leaky Integrate-and-Fire）神经元的单时间步更新，以及可微分的代理梯度脉冲函数。

#### 代理梯度函数 `_SurrogateSpike`

```python
class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0.0).float()    # Heaviside 阶跃（不可微）

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        sg = beta / (1.0 + beta * x.abs()) ** 2   # 快速 sigmoid 导数
        return grad_out * sg, None
```

**数学关系：**

正向：
$$s = \Theta(V - V_{thr}) = \begin{cases} 1 & V \geq V_{thr} \\ 0 & V < V_{thr} \end{cases}$$

反向（代理梯度）：
$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial s} \cdot \frac{\beta}{(1 + \beta |x|)^2}$$

其中 `x = V - V_thr`，`beta=4.0`（默认）控制代理函数的陡峭程度。

#### LIFNeurons 类

```python
class LIFNeurons(nn.Module):
    def forward(self, I_syn, v, ref):
        # 1. 不应期门控
        not_ref = (ref <= 0.0).float()

        # 2. 欧拉衰减膜电位更新
        v_new = self.decay * v + (1.0 - self.decay) * I_syn * not_ref

        # 3. 代理梯度脉冲
        spike = spike_fn(v_new - self.v_threshold, self.surrogate_beta)

        # 4. 硬复位 + 不应期计数（detached，不干扰梯度路径）
        spike_d = spike.detach()
        v_new = v_new * (1.0 - spike_d) + self.v_reset * spike_d
        ref_new = torch.clamp(ref - 1.0, min=0.0) + spike_d * self.refractory_steps

        return spike, v_new, ref_new
```

**数学公式（欧拉离散化）：**

$$V(t) = \alpha \cdot V(t-1) + (1-\alpha) \cdot I_{syn}(t) \cdot \mathbb{1}_{ref \leq 0}$$

其中衰减因子 $\alpha = e^{-dt/\tau_m}$（代码中 `self.decay`，在 `__init__` 中预计算并注册为 buffer）。

**张量形状：**

| 参数/返回 | 形状 | 说明 |
|-----------|------|------|
| `I_syn` | `[B, N]` | 突触电流输入 |
| `v` | `[B, N]` | 膜电位状态 |
| `ref` | `[B, N]` | 不应期倒计时（float，>0 表示仍在不应期） |
| `spike` (返回) | `[B, N]` | 0/1 脉冲 |
| `v_new` (返回) | `[B, N]` | 更新后膜电位 |
| `ref_new` (返回) | `[B, N]` | 更新后不应期计数 |

**关键设计细节：**
- `spike_d = spike.detach()`：复位操作使用 detach 的副本，确保梯度只流经代理梯度路径（`spike_fn`），不被硬复位截断
- `decay` 注册为 buffer（非参数），随模型保存但不参与优化

**依赖者**：`snn/model.py`（`SNNModel.__init__` 实例化 `LIFNeurons`）

---

### 4.5 snn/synapses.py

#### 职责

实现带每突触可训练延迟的全连接突触层，通过环形缓冲区（ring buffer）和线性插值实现连续可微的延迟机制。

#### 延迟缓冲区时序约定

```
buf[:, 0, :] = 1 步前的脉冲（最新）
buf[:, 1, :] = 2 步前的脉冲
...
buf[:, d, :] = (d+1) 步前的脉冲
```

更新规则（每步执行）：
```python
buf = torch.cat([x_t.unsqueeze(1), buf[:, :-1, :]], dim=1)
# 等价于：将整个 buffer 向右移一位，在左端插入当前脉冲
```

#### 参数初始化

```python
w_init = torch.randn(n_pre, n_post) * math.sqrt(2.0 / n_pre)  # He 初始化
d_init = torch.full((n_pre, n_post), -2.0)                     # sigmoid(-2) ≈ 0.12 * d_max
```

- 权重：He 初始化（适合 ReLU 类激活），形状 `[N_pre, N_post]`
- 延迟原始参数：初始值 -2.0，经 sigmoid 后约为 `0.12 * d_max`（偏向较小延迟）

#### 延迟参数化与梯度流

```python
def get_delays(self):
    if self.delay_param_type == "sigmoid":
        return self.d_max * torch.sigmoid(self.delay_raw)  # ∈ [0, d_max]
    else:
        return torch.clamp(self.delay_raw, 0.0, float(self.d_max))

def forward(self, buf):
    d_cont = self.get_delays()           # [N_pre, N_post]，连续值，梯度流经此处
    d_floor = d_cont.detach().floor().long()  # 取整（detached，不影响梯度）
    d_ceil  = d_floor + 1
    alpha   = d_cont - d_floor.float()   # 小数部分，梯度流经 alpha → d_cont → d_raw

    for i in range(n_pre):
        buf_i = buf[:, :, i]             # [B, d_max+1]
        s_f = gather(buf_i, d_floor[i])  # floor 处的历史脉冲
        s_c = gather(buf_i, d_ceil[i])   # ceil 处的历史脉冲
        s_i = (1 - alpha[i]) * s_f + alpha[i] * s_c   # 线性插值
        I_syn += s_i * weight[i]
```

**梯度路径分析：**

$$\frac{\partial I_{syn}}{\partial d_{raw}^{(i,j)}} = \frac{\partial I_{syn}}{\partial \alpha_{ij}} \cdot \frac{\partial \alpha_{ij}}{\partial d_{cont}^{(i,j)}} \cdot \frac{\partial d_{cont}^{(i,j)}}{\partial d_{raw}^{(i,j)}}$$

其中：
- $\partial I_{syn}/\partial \alpha_{ij} = (s_c - s_f) \cdot w_{ij}$（当 floor/ceil 处脉冲不同时非零）
- $\partial d_{cont}/\partial d_{raw} = d_{max} \cdot \sigma(d_{raw}) \cdot (1 - \sigma(d_{raw}))$（sigmoid 导数）

**张量形状：**

| 参数/返回 | 形状 | 说明 |
|-----------|------|------|
| `buf` (输入) | `[B, d_max+1, N_pre]` | 历史脉冲缓冲区 |
| `weight` | `[N_pre, N_post]` | 突触权重 |
| `delay_raw` | `[N_pre, N_post]` | 原始延迟参数（sigmoid 前） |
| `I_syn` (返回) | `[B, N_post]` | 突触电流 |

**与 delay/异步的关系**：这是整个延迟机制的核心。不同突触的延迟参数 `d_raw[i,j]` 独立优化，允许网络学习到不同的时序依赖。

**依赖者**：`snn/model.py`（`DelayedSynapticLayer` 被实例化为 `syn_ih` 和可选的 `syn_ho`）

---

### 4.6 snn/model.py

#### 职责

组合 `LIFNeurons` 和 `DelayedSynapticLayer` 构成完整的 SNN 模型，管理时间步仿真循环、多槽读出累积和 train_mode 参数冻结。

#### 关键数据结构

```python
@dataclass
class SlotBoundaries:
    win_start:  int   # 输入编码窗口起始（含）
    win_end:    int   # 输入编码窗口终止（不含）
    read_start: int   # 读出窗口起始（含）
    read_end:   int   # 读出窗口终止（不含）
```

#### make_slots 函数

```python
def make_slots(K, win_len, read_len, gap_len=0):
    slot_len = win_len + read_len + gap_len
    for k in range(K):
        off = k * slot_len
        slots.append(SlotBoundaries(
            win_start=off,
            win_end=off + win_len,
            read_start=off + win_len,
            read_end=off + win_len + read_len,
        ))
```

以 Step2 配置（win_len=20, read_len=10, gap_len=5）、K=3 为例：

```
时间轴 (ms): 0    20    30   35    55    65   70    90    100 105
Slot 0:     [input][readout][gap]
Slot 1:                          [input][readout][gap]
Slot 2:                                               [input][readout][gap]
```

#### SNNModel 架构

```python
class SNNModel(nn.Module):
    # 参数冻结逻辑
    train_w = train_mode in ("weights_only", "weights_and_delays")
    train_d = train_mode in ("delays_only", "weights_and_delays")

    # 层定义
    self.syn_ih  = DelayedSynapticLayer(n_input, n_hidden, ...)  # 输入→隐藏
    self.lif_h   = LIFNeurons(n_hidden, ...)
    self.syn_ho  = DelayedSynapticLayer(n_hidden, n_output, ...)  # 可选
    self.lif_o   = LIFNeurons(n_output, ...)                      # 可选
    self.readout = nn.Linear(readout_in, 1)                       # 始终可训练
```

#### forward 方法（核心仿真循环）

```python
def forward(self, spike_input, slots):
    # 初始化
    v_h, ref_h = self.lif_h.init_state(B, device)
    buf_in = torch.zeros(B, self.d_max + 1, self.n_input, device=device)
    hidden_acc = [torch.zeros(B, self.n_hidden) for _ in range(K)]

    # 预计算读出时间查找表
    readout_at = [[] for _ in range(T)]
    for k, slot in enumerate(slots):
        for t in range(slot.read_start, slot.read_end):
            readout_at[t].append(k)

    # 时间步循环
    for t in range(T):
        x_t = spike_input[:, t, :]           # [B, N_input]
        I_h = self.syn_ih(buf_in)             # 从缓冲区取延迟后电流
        spike_h, v_h, ref_h = self.lif_h(I_h, v_h, ref_h)

        buf_in = torch.cat([x_t.unsqueeze(1), buf_in[:, :-1, :]], dim=1)

        for k in readout_at[t]:
            hidden_acc[k] += spike_h         # 在读出窗口累积脉冲

    # 线性读出
    logits = torch.stack([self.readout(hidden_acc[k]).squeeze(-1) for k in range(K)], dim=1)
    return logits, info   # logits: [B, K]
```

**关键设计：`readout_at` 预计算查找表**

通过预先构建 `readout_at[t]`（时间步 t 属于哪些 slot 的读出窗口），避免每步都遍历所有 slot，提升循环效率。

**张量形状追踪：**

| 变量 | 形状 | 说明 |
|------|------|------|
| `spike_input` | `[B, T, n_input]` | 全试次输入脉冲 |
| `buf_in` | `[B, d_max+1, n_input]` | 输入延迟缓冲 |
| `I_h` | `[B, n_hidden]` | 隐藏层突触电流 |
| `spike_h` | `[B, n_hidden]` | 隐藏层脉冲 |
| `hidden_acc[k]` | `[B, n_hidden]` | 第 k 个槽的脉冲累积 |
| `logits` | `[B, K]` | 各槽输出 logit |

**train_mode 参数冻结机制：**

| train_mode | `syn_ih.weight` | `syn_ih.delay_raw` | `readout` |
|------------|-----------------|--------------------|-----------|
| `weights_only` | `nn.Parameter` | `buffer`（冻结） | `nn.Parameter` |
| `delays_only` | `buffer`（冻结） | `nn.Parameter` | `nn.Parameter` |
| `weights_and_delays` | `nn.Parameter` | `nn.Parameter` | `nn.Parameter` |

**依赖者**：`train/trainer.py`, `train/eval.py`, `scripts/run_step*.py`

---

### 4.7 train/trainer.py

#### 职责

封装完整训练流程：分组优化器、前向传播、损失计算、梯度裁剪、验证、checkpoint 保存、日志记录。

#### build_optimizer

```python
def build_optimizer(model: SNNModel, cfg) -> torch.optim.Optimizer:
    param_groups = []
    if w_params: param_groups.append({"params": w_params, "lr": cfg["lr_w"]})
    if d_params: param_groups.append({"params": d_params, "lr": cfg["lr_d"]})
    if r_params: param_groups.append({"params": r_params, "lr": cfg["lr_readout"]})
    return torch.optim.Adam(param_groups)
```

通过 `model.weight_params()`、`model.delay_params()`、`model.readout_params()` 分别获取三组参数，允许为权重和延迟设置不同的学习率（默认均为 `1e-3`）。

#### _forward_batch

```python
def _forward_batch(self, A, B, op_ids, labels):
    spike_input = encode_trial(A, B, op_ids, self.slots, ...)    # [B, T, n_input]
    logits, info = self.model(spike_input, self.slots)            # [B, K]
    loss = F.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.reshape(-1)                   # 展平 K 维
    )
    if spike_penalty > 0: loss += spike_penalty * spk_mean       # 能量惩罚
    if delay_penalty > 0: loss += delay_penalty * delay_reg      # 延迟 L1
    acc = (logits.detach() > 0) == labels.detach()  # [B, K]
    return loss, acc.float().mean().item(), info
```

**Step1 兼容性处理：**

```python
# Promote Step-1 tensors [B] → [B, 1]
if labels.dim() == 1:
    A, B, op_ids, labels = (A.unsqueeze(1), B.unsqueeze(1), ...)
```

BooleanDataset 返回标量（[B]），MultiQueryDataset 返回 [B,K]。Trainer 统一转换为 [B,K]，与 K=1 的 slots 兼容。

#### fit 训练循环

```python
def fit(self, train_loader, val_loader, epochs):
    for epoch in range(1, epochs + 1):
        tr  = self.train_epoch(train_loader)
        val = self.eval_epoch(val_loader)

        # Checkpoint：保存最优验证准确率模型
        if val["acc"] > self.best_val:
            self.best_val = val["acc"]
            torch.save(self.model.state_dict(), "best_model.pt")

        # 每 10 个 epoch 打印
        if epoch % 10 == 0:
            print(f"[{epoch}/{epochs}] val acc={val['acc']:.4f} spk={spk:.1f}")
```

**输出文件：**
- `best_model.pt`：验证准确率最高时的权重（含 delays）
- `train_log.csv`：每 epoch 的 loss/acc/spikes/time

#### eval_epoch

与 `train_epoch` 相同的前向传播，但包裹在 `@torch.no_grad()` 下，额外返回 `mean_hidden_spikes`（能量代理指标）。

---

### 4.8 train/eval.py

#### 职责

提供最终测试集评估、max-K 计算、结果 JSON 保存和 sweep 汇总打印。

#### evaluate 函数

```python
@torch.no_grad()
def evaluate(model, loader, slots, cfg, device):
    all_preds, all_labels, all_h_spk = [], [], []
    for A, B, op_ids, labels in loader:
        spike_input = encode_trial(...)
        logits, info = model(spike_input, slots)
        preds = (logits > 0).float()
        all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
        all_h_spk.append(info["total_hidden_spikes"].cpu())

    preds  = torch.cat(all_preds, dim=0)    # [N, K]
    h_spk  = torch.cat(all_h_spk, dim=0)   # [N]

    overall_acc   = (preds == labels).float().mean()
    per_query_acc = correct.mean(dim=0).tolist()      # K 个值
    throughput    = K / mean_h_spk   # 核心指标：K/脉冲数

    return {"accuracy": overall_acc, "per_query_acc": per_query_acc,
            "mean_hidden_spikes": mean_h_spk, "throughput_K_per_spk": throughput, "K": K}
```

**核心指标说明：**

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `accuracy` | 所有 (sample, query) 正确率均值 | 整体准确率 |
| `per_query_acc` | 每个 slot 的准确率 | 检查各槽是否均衡 |
| `mean_hidden_spikes` | 每个 trial 的隐藏层总脉冲数均值 | 能量代理 |
| `throughput_K_per_spk` | K / mean_h_spk | 能量归一化吞吐量 |

#### max_K_at_threshold

```python
def max_K_at_threshold(results_by_K: Dict[int, Dict], tau=0.95) -> int:
    max_k = 0
    for K, res in sorted(results_by_K.items()):
        if res["accuracy"] >= tau:
            max_k = K
    return max_k
```

扫描所有 K 值，返回准确率 ≥ τ（默认 0.95）的最大 K。这是 Step2/3 的核心评估指标。

**依赖者**：`scripts/run_step2.py`, `scripts/run_step3.py`

---

### 4.9 scripts/run_step1.py

#### 职责

Step 1 的可执行入口，支持单次运行和全面扫描（8 ops × 3 modes × 3 hidden sizes = 72 次运行）。

#### 关键函数

```python
def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:    # Windows GBK 修复
        return yaml.safe_load(f)

def merge_cli(cfg, args) -> dict:
    if args.op          is not None: cfg["op_name"]    = args.op
    if args.train_mode  is not None: cfg["train_mode"] = args.train_mode
    if args.hidden_size is not None: cfg["hidden_size"]= args.hidden_size
    return cfg
```

#### run_single 流程

1. `set_seed(cfg["seed"])` — 确保可复现
2. `make_slots(K=1, ...)` — Step1 始终 K=1
3. `MultiQueryDataset(K=1, same_op=True)` — 使用 MultiQueryDataset 而非 BooleanDataset，便于统一接口
4. 构建 SNNModel、Trainer，调用 `fit()`
5. 加载 best_model.pt，运行 `evaluate()` on test set
6. 保存 `eval_results.json`，绘制训练曲线和延迟分布图

#### 扫描模式

```python
for op in sweep["ops"]:
    for mode in sweep["train_modes"]:
        for h in sweep["hidden_sizes"]:
            cfg = copy.deepcopy(base_cfg)
            cfg.update({"op_name": op, "train_mode": mode, "hidden_size": h})
            res = run_single(cfg, device, args.runs_dir)
            all_results.append(res)

with open("runs/step1_sweep_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

---

### 4.10 scripts/run_step2.py

#### 职责

Step 2 的入口：固定单一运算（NAND），扫描 K=1..8 和三种训练模式，找到每种模式的 max-K @ 95% 准确率。

#### 关键差异

相比 Step1：
- `K` 从 CLI 参数或 sweep 配置获取（不再固定为 1）
- `slots = make_slots(K, win_len=20, read_len=10, gap_len=5)` — 多槽时序结构
- sweep 后额外调用 `max_K_at_threshold()` 汇总每种模式的最大 K
- 生成 `K_accuracy.png` 和 `K_throughput.png` 对比图

#### sweep 结构

```python
# 结果字典: {mode: {K: eval_result}}
all_results = {}
for mode in sweep["train_modes"]:
    all_results[mode] = {}
    for K in sweep["K_values"]:   # [1,2,3,4,5,6,7,8]
        res = run_single(cfg, K, device, runs_dir)
        all_results[mode][K] = res

# 汇总 max-K
for mode, k_results in all_results.items():
    mk = max_K_at_threshold(k_results, tau=0.95)
    summary[mode] = {"max_K": mk, "results_by_K": k_results}
```

---

### 4.11 scripts/run_step3.py

#### 职责

Step 3 的入口：混合运算（8 种 op 随机采样），每查询独立选择 op，测试 SNN 能否在时间复用中同时处理不同逻辑运算。

#### 关键差异

相比 Step2：
- `n_input = 2 + n_ops`（2 个数据通道 + 8 个 one-hot op 通道 = 10）
- `MultiQueryDataset(same_op=False, op_name=None)` — 每槽随机选 op
- op 信息通过 `encode_trial` 注入 one-hot 通道（channels 2..9）

```python
n_ops   = len(ops_list)   # 8
n_input = 2 + n_ops       # 10
# 数据集
ds = MultiQueryDataset(K=K, same_op=False, op_name=None, ops_list=ops_list)
# 模型
model = SNNModel(n_input=n_input, ...)
```

sweep 扫描 K=1..6（比 Step2 少，因为任务更难）。

---

### 4.12 utils/

#### utils/seed.py

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
```

同时设置 Python、NumPy、PyTorch（CPU/GPU）随机种子，并禁用 cuDNN 非确定性优化。每次 `run_single` 开始时调用。

#### utils/logger.py

```python
def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers: return logger    # 防止重复添加 handler
    fmt = "[%(asctime)s] %(levelname)s  %(message)s"
    logger.addHandler(StreamHandler(sys.stdout))
    if log_file: logger.addHandler(FileHandler(log_file))
    return logger
```

返回标准 Python `Logger` 对象，格式为 `[HH:MM:SS] LEVEL  message`。

#### utils/viz.py

| 函数 | 输入 | 输出 |
|------|------|------|
| `plot_training_curves(log_rows, save_path)` | 训练日志列表 | loss/acc 双子图 PNG |
| `plot_delay_distribution(delays, title, save_path)` | `[N_pre, N_post]` 延迟矩阵 | 热力图 PNG |
| `plot_K_accuracy(K_values, results_by_mode, tau, save_path)` | K 值列表和各模式准确率 | K-accuracy 折线图 |
| `plot_throughput(K_values, results_by_mode, save_path)` | K 值列表和吞吐量 | 吞吐量折线图 |
| `plot_spike_raster(spike_train, title, save_path, slot_boundaries)` | `[T, N]` 脉冲矩阵 | 点阵图 |

所有函数使用 `matplotlib.use("Agg")` 非交互后端，可在无显示器环境中运行。目录不存在时自动创建。

---

## 5. 关键路径深潜

### 5.1 单时间步神经元更新（数学→代码映射）

**完整数学推导：**

LIF 神经元的连续时间方程：
$$\tau_m \frac{dV}{dt} = -V(t) + I_{syn}(t)$$

欧拉离散化（步长 $dt$）：
$$V(t+dt) = \left(1 - \frac{dt}{\tau_m}\right) V(t) + \frac{dt}{\tau_m} I_{syn}(t)$$

令 $\alpha = e^{-dt/\tau_m}$（精确离散化更好地保持时间常数），则：
$$V(t+dt) = \alpha \cdot V(t) + (1-\alpha) \cdot I_{syn}(t)$$

加入不应期门控（$\mathbb{1}_{ref \leq 0}$）：
$$V(t+dt) = \alpha \cdot V(t) + (1-\alpha) \cdot I_{syn}(t) \cdot \mathbb{1}_{ref \leq 0}$$

**代码映射（`snn/neurons.py` 第 93-96 行）：**

```python
not_ref = (ref <= 0.0).float()          # = 𝟙_{ref ≤ 0}
v_new   = self.decay * v                 # = α · V(t)
        + (1.0 - self.decay) * I_syn * not_ref  # + (1-α) · I_syn · gate
spike   = spike_fn(v_new - self.v_threshold, self.surrogate_beta)
```

- `self.decay` = $\alpha = e^{-1/10} \approx 0.905$（dt=1ms, tau_m=10ms）
- 不应期期间 `not_ref=0`，神经元完全不接受输入（硬门控）

**代理梯度数学（`_SurrogateSpike.backward`）：**

$$\frac{\partial \mathcal{L}}{\partial V} = \frac{\partial \mathcal{L}}{\partial s} \cdot \underbrace{\frac{\beta}{(1 + \beta|V - V_{thr}|)^2}}_{\text{代理导数}}$$

当 $V = V_{thr}$ 时代理导数最大（$= \beta$），随距离增大而快速衰减。

---

### 5.2 Delay buffer/queue 机制

**缓冲区结构（`snn/model.py` 第 201 行）：**

```python
buf_in = torch.zeros(B, self.d_max + 1, self.n_input, device=device)
```

形状 `[B, 50, 2]`（d_max=49）。

**每步更新（`snn/model.py` 第 233 行）：**

```python
buf_in = torch.cat([x_t.unsqueeze(1), buf_in[:, :-1, :]], dim=1)
#                   [B, 1, n_input]     [B, d_max, n_input]
#          结果：   [B, d_max+1, n_input]
```

这相当于一个先进先出（FIFO）队列：
- `buf_in[:, 0, :]` = 刚刚发生的脉冲（$t$ 时刻，实际 1 步延迟）
- `buf_in[:, d, :]` = $d+1$ 步前的脉冲

**从缓冲区读取（`snn/synapses.py`）：**

延迟 $d_{ij}$ 对应读取 `buf_in[:, floor(d_ij), i]` 和 `buf_in[:, ceil(d_ij), i]` 并线性插值，提供连续可微的延迟值。

**时序示例（d_ij = 5.3）：**

```
t=10: x_in 发放脉冲
t=11: buf_in[:,0,:] 更新 (t=10的脉冲进入)
...
t=15: buf_in[:,4,:] 包含 t=10 的脉冲 (5步延迟)
t=16: buf_in[:,5,:] 包含 t=10 的脉冲 (6步延迟)
→ I_syn[i,j](t=16) = 0.7 * buf[4] * w + 0.3 * buf[5] * w (alpha=0.3)
```

---

### 5.3 多查询读出机制（Step2/3）

**核心思路：** 一次前向传播覆盖整个多槽试次（总时长 T = K × slot_len），通过 `readout_at` 查找表，在每个槽的读出窗口期间累积隐藏层脉冲。

**读出查找表构建（`snn/model.py` 第 211-213 行）：**

```python
readout_at = [[] for _ in range(T)]
for k, slot in enumerate(slots):
    for t in range(slot.read_start, min(slot.read_end, T)):
        readout_at[t].append(k)
# 结果：readout_at[t] = 在时刻 t 处于读出窗口的所有槽索引
```

**仿真循环中的累积：**

```python
for t in range(T):
    spike_h, v_h, ref_h = self.lif_h(I_h, v_h, ref_h)
    for k in readout_at[t]:
        hidden_acc[k] = hidden_acc[k] + spike_h  # [B, N_hidden]
```

**为什么是每槽读出而非全局读出？**

如果在试次末尾统一读出所有脉冲，网络可以简单地把所有计算堆积到最后时刻，无法测试真正的"每查询独立推理"能力。逐槽读出强制网络在各槽结束时已完成该查询的计算，确实测试了时间复用。

**损失计算（`train/trainer.py` 第 108-111 行）：**

```python
loss = F.binary_cross_entropy_with_logits(
    logits.reshape(-1),   # [B*K]
    labels.reshape(-1)    # [B*K]
)
```

K 个查询的损失在同一批次内平均，梯度同时更新所有槽的预测能力。

---

### 5.4 train_mode 参数冻结机制

参数冻结通过 `nn.Parameter` vs `register_buffer` 的切换实现：

**`snn/synapses.py` 第 57-68 行：**

```python
w_init = torch.randn(n_pre, n_post) * math.sqrt(2.0 / n_pre)
if train_weights:
    self.weight = nn.Parameter(w_init)      # 加入计算图，优化器可见
else:
    self.register_buffer("weight", w_init)  # 不加入参数，随模型保存但不优化

d_init = torch.full((n_pre, n_post), -2.0)
if train_delays:
    self.delay_raw = nn.Parameter(d_init)
else:
    self.register_buffer("delay_raw", d_init)
```

**`snn/model.py` 第 111-112 行控制开关：**

```python
train_w = train_mode in ("weights_only", "weights_and_delays")
train_d = train_mode in ("delays_only", "weights_and_delays")
```

**优化器参数分组（`train/trainer.py`）：**

```python
# model.weight_params(): 过滤 named_parameters 中名含 "weight"（非 readout）且 requires_grad=True
# model.delay_params():  过滤名含 "delay_raw" 且 requires_grad=True
# model.readout_params(): readout.parameters()（始终）
```

三组参数可以设置不同学习率，且在 `delays_only` 模式下 `weight_params()` 返回空列表，优化器的对应参数组不存在，不会造成错误（`build_optimizer` 中有 `if w_params:` 检查）。

---

## 6. 常见问题与排障

### Q1: Windows 上运行报 `UnicodeDecodeError`

**原因**：Windows 默认编码为 GBK，YAML 配置文件含中文注释。

**解决**：所有 `load_cfg()` 函数已修复为 `open(path, encoding="utf-8")`。如果自行添加脚本，务必使用此模式。

### Q2: `delays_only` 模式在 XOR/XNOR 上准确率接近随机水平（~0.5）

**原因**：权重随机初始化后冻结，仅靠延迟调整无法学习 XOR 的非线性分类边界。延迟改变时序但不改变权重空间的分离能力。

**建议**：可尝试增大 `d_max` 或添加 `delay_penalty` 正则化，或在 `delays_only` 模式下使用更大的 `hidden_size`。

### Q3: 训练曲线震荡剧烈

**建议**：减小 `lr_d`（延迟学习率），或增大 `batch_size`。延迟参数的梯度信号较弱（只有 floor/ceil 不同时才非零），较大批次有助于稳定梯度估计。

### Q4: GPU 训练速度慢于预期

**原因**：`snn/synapses.py` 的 `forward` 中有 `for i in range(n_pre)` 的 Python 循环，无法充分利用 GPU 并行。

**建议**：对于 `n_input=2` 的情况（Step1/2），循环仅 2 次，影响较小。Step3 `n_input=10` 时性能开销稍大。未来可用 `torch.gather` 的批量化实现替换。

### Q5: 模型加载失败（`best_model.pt` 不存在）

**原因**：可能第一个 epoch 就崩溃（NaN 损失）。检查 `r_on/r_off` 配置是否合理，以及 `grad_clip` 是否足够小。

---

## 7. 快速修改指南

### 添加新布尔运算

1. 在 `data/boolean_dataset.py` 的 `_OPS` 字典中添加：
   ```python
   "MY_OP": lambda a, b: int(...),
   ```
2. 在 `OPS_LIST` 中追加 `"MY_OP"`
3. 如果是 Step3（混合 op），`n_input` 会自动变为 `2 + len(ops_list)`，无需其他修改

### 更改最大延迟

修改 YAML 中的 `d_max`。注意：`d_max` 越大，`buf_in` 越大（内存 = `B × (d_max+1) × n_input × 4 bytes`）。

### 添加输出层（output LIF layer）

在 YAML 中设置 `use_output_layer: true`，并将 `readout_source: output`。SNNModel 会自动创建 `syn_ho` 和 `lif_o`。

### 修改代理梯度函数

在 `snn/neurons.py` 的 `_SurrogateSpike.backward` 中替换计算式。常见备选：
- 矩形窗：`sg = (x.abs() < 0.5).float()`
- 多角函数：`sg = torch.clamp(1 - x.abs(), min=0)`

### 添加自定义惩罚项

在 `train/trainer.py` 的 `_forward_batch` 中添加：
```python
if self.cfg.get("my_penalty", 0.0) > 0:
    loss = loss + self.cfg["my_penalty"] * my_penalty_fn(self.model)
```
并在 YAML 中添加 `my_penalty: 0.01`。
