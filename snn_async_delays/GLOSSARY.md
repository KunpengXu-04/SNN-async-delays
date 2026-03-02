# 术语表（GLOSSARY）

> 本术语表按字母顺序排列，所有定义均与 `snn_async_delays` 项目的实际代码实现精确对应。

---

## A

### A_IMP_B（A 蕴含 B）

布尔运算之一，逻辑含义为"若 A 则 B"（A implies B）。

**真值表：** `(A=0,B=0)→1, (A=0,B=1)→1, (A=1,B=0)→0, (A=1,B=1)→1`

**代码（`data/boolean_dataset.py`）：**
```python
"A_IMP_B": lambda a, b: int((not bool(a)) or bool(b))
```

等价于 `¬A ∨ B`。`B_IMP_A` 对称地定义为 `bool(a) or (not bool(b))`。

---

## B

### BCEWithLogitsLoss（带 logit 的二元交叉熵损失）

训练中使用的损失函数，公式为：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \sigma(z_i) + (1-y_i) \log(1-\sigma(z_i)) \right]$$

其中 $z_i$ 是模型输出的 logit，$\sigma(\cdot)$ 是 sigmoid 函数。

**代码（`train/trainer.py`）：**
```python
loss = F.binary_cross_entropy_with_logits(
    logits.reshape(-1),   # [B*K]  原始 logit，未经 sigmoid
    labels.reshape(-1)    # [B*K]  0.0 或 1.0
)
```

数值稳定性优于先 sigmoid 再取 log。K 个查询的损失在同一批次展平后平均。

### Boolean ops（布尔运算）

本项目研究的 8 种二元布尔运算，统一存储在 `OPS_LIST` 中：

| 名称 | 符号 | `(0,0)` | `(0,1)` | `(1,0)` | `(1,1)` |
|------|------|---------|---------|---------|---------|
| AND | $A \wedge B$ | 0 | 0 | 0 | 1 |
| OR | $A \vee B$ | 0 | 1 | 1 | 1 |
| XOR | $A \oplus B$ | 0 | 1 | 1 | 0 |
| XNOR | $\overline{A \oplus B}$ | 1 | 0 | 0 | 1 |
| NAND | $\overline{A \wedge B}$ | 1 | 1 | 1 | 0 |
| NOR | $\overline{A \vee B}$ | 1 | 0 | 0 | 0 |
| A_IMP_B | $\neg A \vee B$ | 1 | 1 | 0 | 1 |
| B_IMP_A | $A \vee \neg B$ | 1 | 0 | 1 | 1 |

XOR/XNOR 是线性不可分的，是所有运算中最难学习的（需要 `weights_and_delays` + 足够大的隐藏层）。

---

## D

### d_max（最大延迟）

延迟参数的上界，单位为时间步（ms）。在 `sigmoid` 延迟参数化模式下，连续延迟值 $d \in [0, d_{max}]$。

**配置（`configs/step1_singleop.yaml`）：** `d_max: 49`，有效延迟范围 0–49ms。

**代码（`snn/synapses.py`）：**
```python
# 缓冲区大小 = d_max + 1
buf_in = torch.zeros(B, self.d_max + 1, self.n_input, device=device)
# sigmoid 映射
return self.d_max * torch.sigmoid(self.delay_raw)  # ∈ [0, d_max]
```

缓冲区大小为 `d_max+1`，因为 `buf[:,0,:]` 对应 1 步前（最小有效延迟），`buf[:,d_max,:]` 对应 `d_max+1` 步前。

### Delay parameterization（延迟参数化）

将无约束实数参数 $d_{raw}$ 映射到合法延迟范围 $[0, d_{max}]$ 的方法。本项目支持两种：

**sigmoid 模式（默认）：**
$$d_{cont} = d_{max} \cdot \sigma(d_{raw}) = d_{max} \cdot \frac{1}{1 + e^{-d_{raw}}}$$

优点：值域天然有界，梯度处处存在；初始值 `d_raw=-2.0` 对应 $d_{cont} \approx 0.12 \cdot d_{max}$（约 6ms）。

**direct 模式：**
$$d_{cont} = \text{clamp}(d_{raw}, 0, d_{max})$$

在边界处梯度为零，初始化需要更谨慎。

**代码（`snn/synapses.py`）：**
```python
def get_delays(self) -> torch.Tensor:
    if self.delay_param_type == "sigmoid":
        return self.d_max * torch.sigmoid(self.delay_raw)
    else:
        return torch.clamp(self.delay_raw, 0.0, float(self.d_max))
```

### Delay penalty（延迟惩罚）

可选的正则化项，惩罚过大的平均延迟，鼓励网络使用较短延迟。

$$\mathcal{L}_{delay} = \lambda_d \cdot \frac{1}{N_{pre} \cdot N_{post}} \sum_{i,j} d_{cont}^{(i,j)}$$

**代码（`train/trainer.py`）：**
```python
if self.cfg.get("delay_penalty", 0.0) > 0:
    loss = loss + self.cfg["delay_penalty"] * self.model.delay_regularization()
```

`delay_regularization()` 返回所有突触层延迟均值（`syn_ih.get_delays().mean()`）。默认 `delay_penalty=0.0`（不使用）。

---

## L

### LIF（Leaky Integrate-and-Fire，泄漏积分发放）

SNN 中最常用的神经元模型。膜电位随时间衰减，当超过阈值 $V_{thr}$ 时发放脉冲并复位。

**连续时间方程：**
$$\tau_m \frac{dV}{dt} = -V(t) + I_{syn}(t)$$

**欧拉离散化（本项目实现）：**
$$V(t+1) = \underbrace{e^{-dt/\tau_m}}_{\alpha} \cdot V(t) + (1 - \alpha) \cdot I_{syn}(t) \cdot \mathbb{1}_{ref \leq 0}$$

其中 $\alpha$ 称为衰减因子（decay）。$dt=1\text{ms}$，$\tau_m=10\text{ms}$ 时 $\alpha \approx 0.905$。

**代码（`snn/neurons.py`）：**
```python
decay_val = float(torch.exp(torch.tensor(-dt / tau_m)).item())
self.register_buffer("decay", torch.tensor(decay_val))
# ...在 forward 中:
v_new = self.decay * v + (1.0 - self.decay) * I_syn * not_ref
```

---

## R

### Rate coding（速率编码）

将连续值信息编码为脉冲发放率的方式。本项目中，布尔输入的"1"和"0"分别编码为不同的 Poisson 发放率。

**编码规则：**
- 输入 A=1（或 B=1）：每步发放概率 $p_{on} = r_{on} \cdot dt / 1000$
- 输入 A=0（或 B=0）：每步发放概率 $p_{off} = r_{off} \cdot dt / 1000$

**默认参数（Step1 config）：**
- $r_{on} = 400$ Hz → $p_{on} = 0.4$（dt=1ms）
- $r_{off} = 10$ Hz → $p_{off} = 0.01$

**代码（`data/encoding.py`）：**
```python
p_on  = r_on  * dt / 1000.0   # 0.4
p_off = r_off * dt / 1000.0   # 0.01
pA = torch.where(A_k > 0.5, torch.full_like(A_k, p_on), torch.full_like(A_k, p_off))
spike_input[..., 0] = torch.bernoulli(pA_exp)
```

### Readout window（读出窗口）

每个时间槽结尾处的一段时间区间，在此期间累积隐藏层脉冲数，最终通过线性层转换为输出 logit。

**配置：** `read_len=10`（Step1），意味着每次读出窗口持续 10ms。

**代码（`snn/model.py`）：**
```python
# 在读出窗口内累积
for k in readout_at[t]:
    hidden_acc[k] = hidden_acc[k] + spike_h   # [B, n_hidden]
# 读出结束后
logit_k = self.readout(hidden_acc[k]).squeeze(-1)   # [B]
```

**重要性**：读出必须是每槽独立的（per-window），而非全局累积，否则网络可以"懒惰地"把所有计算推到试次末尾，无法真正测试时间复用能力。

### Refractory period（不应期）

神经元发放脉冲后，在固定的时间步数内不接受任何输入的生理限制，防止异常高频发放。

**代码（`snn/neurons.py`）：**
```python
# 发放后设置不应期倒计时
ref_new = torch.clamp(ref - 1.0, min=0.0) + spike_d * self.refractory_steps

# 不应期门控（不应期内 not_ref=0，阻断输入）
not_ref = (ref <= 0.0).float()
v_new = self.decay * v + (1.0 - self.decay) * I_syn * not_ref
```

`spike_d = spike.detach()` 确保不应期更新不干扰代理梯度路径。

默认 `refractory_steps=2`，即发放后 2 个时步内神经元被抑制。

---

## S

### Spike penalty（脉冲惩罚）

可选的正则化项，惩罚隐藏层过多的脉冲发放，作为能量消耗的代理指标。

$$\mathcal{L}_{spike} = \lambda_s \cdot \frac{1}{B} \sum_b \sum_t \sum_n s_h^{(b)}(t, n)$$

**代码（`train/trainer.py`）：**
```python
if self.cfg.get("spike_penalty", 0.0) > 0:
    spk_mean = info["total_hidden_spikes"].mean()
    loss = loss + self.cfg["spike_penalty"] * spk_mean
```

`info["total_hidden_spikes"]` 形状为 `[B]`，来自 `SNNModel.forward()` 中的 `total_h_spk` 累加。默认 `spike_penalty=0.0`。

### STE（Straight-Through Estimator，直通估计器）

一种处理不可微离散运算（如 Heaviside 阶跃函数）的梯度技术。正向使用真实的不可微函数，反向将梯度"直通"（pass-through），即假装前向函数是单位函数。

**本项目中的 STE（代理梯度版本）：** 不是严格意义上的 STE（后者反向梯度恒为 1），而是使用快速 sigmoid 导数作为代理：

$$\frac{\partial \hat{s}}{\partial x} = \frac{\beta}{(1 + \beta|x|)^2}$$

在 `x=0`（阈值处）时梯度为 $\beta$，而非 1。严格 STE 对应 $\beta \to \infty$。

**代码（`snn/neurons.py`）：**
```python
class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        return (x >= 0.0).float()       # 正向：真实 Heaviside

    @staticmethod
    def backward(ctx, grad_out):
        sg = beta / (1.0 + beta * x.abs()) ** 2   # 代理梯度（非严格 STE）
        return grad_out * sg, None
```

### Surrogate gradient（代理梯度）

为解决脉冲函数（Heaviside 阶跃）的梯度不存在问题而引入的平滑近似。在反向传播中用一个连续可微函数替代真实梯度。

**本项目使用的代理函数**（快速 sigmoid 导数）：

正向：$s = \Theta(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$

反向（代理）：$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial s} \cdot \frac{\beta}{(1 + \beta|x|)^2}$

参数 `surrogate_beta=4.0` 控制峰值高度和宽度的权衡：越大越接近真实阈值函数，但梯度消失区域越大。

---

## T

### Temporal slot / Multi-query（时间槽 / 多查询）

将一次试次（trial）划分为 K 个时间槽的机制，每个槽处理一个独立的布尔查询。这是测试"异步时间复用"的核心架构。

**结构（每个槽）：**
```
[输入窗口 win_len ms] [读出窗口 read_len ms] [间隔 gap_len ms]
```

**代码（`snn/model.py`）：**
```python
@dataclass
class SlotBoundaries:
    win_start: int; win_end: int; read_start: int; read_end: int

def make_slots(K, win_len, read_len, gap_len=0) -> List[SlotBoundaries]:
    slot_len = win_len + read_len + gap_len
    for k in range(K):
        off = k * slot_len
        slots.append(SlotBoundaries(off, off+win_len, off+win_len, off+win_len+read_len))
```

**设计约束**：K 个查询在同一组神经元上串行处理（时间复用），不使用空间并行通道。神经元状态（`v_h`, `ref_h`, `buf_in`）在槽间连续，不重置。

### Throughput K/spk（能量归一化吞吐量）

核心研究指标，度量单位能量（脉冲数）内处理的查询数。

$$\text{throughput} = \frac{K}{\overline{\text{spikes}}}$$

其中 $\overline{\text{spikes}}$ 是每试次平均隐藏层总脉冲数。

**代码（`train/eval.py`）：**
```python
throughput = K / mean_h_spk if mean_h_spk > 0 else float("nan")
```

**研究意义**：在固定精度阈值（τ=95%）下，`weights_and_delays` 期望比 `weights_only` 有更高的 throughput，证明延迟在能量效率上的优势。

### train_mode（训练模式）

控制哪些参数参与优化的全局开关，通过 `nn.Parameter` vs `register_buffer` 实现冻结。

**三种模式：**

| 模式 | W_ih 权重 | D_ih 延迟 | readout |
|------|-----------|-----------|---------|
| `weights_only` | 可训练 | 冻结（初始约 6ms） | 可训练 |
| `delays_only` | 冻结（随机初始化） | 可训练 | 可训练 |
| `weights_and_delays` | 可训练 | 可训练 | 可训练 |

**代码（`snn/model.py`）：**
```python
train_w = train_mode in ("weights_only", "weights_and_delays")
train_d = train_mode in ("delays_only", "weights_and_delays")
# ...传递给 DelayedSynapticLayer
self.syn_ih = DelayedSynapticLayer(..., train_weights=train_w, train_delays=train_d)
```

---

## W

### W_ih（输入→隐藏权重矩阵）

输入神经元到隐藏层神经元的突触权重，形状 `[n_input, n_hidden]`，存储在 `SNNModel.syn_ih.weight`。

**初始化：** He 初始化，`randn(n_pre, n_post) * sqrt(2/n_pre)`，适合 ReLU 类激活（SNN 脉冲在 0-1 之间，近似 ReLU）。

**在延迟层中的作用：**
$$I_{syn}^{(j)} = \sum_{i} s_{delayed}^{(i)} \cdot w_{ij}$$

其中 $s_{delayed}^{(i)}$ 是经延迟缓冲区读出的突触前脉冲（已经过线性插值）。

**代码（`snn/synapses.py`）：**
```python
I_syn = I_syn + s_i * self.weight[i].unsqueeze(0)   # s_i: [B, n_post], weight[i]: [n_post]
```

`train_mode="delays_only"` 时 `weight` 为 `register_buffer`，随机初始化后冻结；`weights_only` 或 `weights_and_delays` 时为 `nn.Parameter`，参与优化。
