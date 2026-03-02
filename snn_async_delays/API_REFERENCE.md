# API 参考手册

> 本文件列出 `snn_async_delays` 项目所有公开函数和类的完整签名、参数、返回值、副作用和异常说明。所有信息均直接来自源代码。

---

## 模块：`data.boolean_dataset`

### `compute_label(op_name, A, B)`

```python
def compute_label(op_name: str, A: int, B: int) -> int
```

**描述**：计算布尔运算 `op_name(A, B)` 的结果标签。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `op_name` | `str` | 运算名称，必须是 `OPS_LIST` 中的一个 | 无默认 |
| `A` | `int` | 输入 A（0 或 1） | 无默认 |
| `B` | `int` | 输入 B（0 或 1） | 无默认 |

**返回：** `int`，值为 0 或 1。

**异常：** `KeyError` — 当 `op_name` 不在 `_OPS` 字典中时。

---

### `class BooleanDataset(Dataset)`

```python
class BooleanDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        op_name: Optional[str] = "NAND",
        ops_list: List[str] = OPS_LIST,
        seed: int = 42,
    )
```

**描述**：Step 1 数据集，每个样本是单个 (A, B) 对，配合一种布尔运算。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `n_samples` | `int` | 样本总数 | 无默认 |
| `op_name` | `Optional[str]` | 固定运算名称；`None` 表示随机采样 | `"NAND"` |
| `ops_list` | `List[str]` | 合法运算名称列表（决定 op_id 空间） | `OPS_LIST` |
| `seed` | `int` | NumPy 随机种子，用于 A/B/op 采样 | `42` |

**属性：**

| 属性 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `self.A` | `[n_samples]` | float32 | 输入 A |
| `self.B` | `[n_samples]` | float32 | 输入 B |
| `self.op_ids` | `[n_samples]` | int64 | 运算索引 |
| `self.labels` | `[n_samples]` | float32 | 目标标签 |

**`__getitem__(idx)` 返回：** `(A: scalar, B: scalar, op_id: scalar, label: scalar)`

---

### `class MultiQueryDataset(Dataset)`

```python
class MultiQueryDataset(Dataset):
    def __init__(
        self,
        K: int,
        n_samples: int,
        same_op: bool = True,
        op_name: Optional[str] = "NAND",
        ops_list: List[str] = OPS_LIST,
        seed: int = 42,
    )
```

**描述**：Step 2/3 数据集，每个样本包含 K 个查询（(A,B,op,label) 的 K 元组）。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `K` | `int` | 每个样本的查询数量 | 无默认 |
| `n_samples` | `int` | 样本总数 | 无默认 |
| `same_op` | `bool` | `True`=所有 K 查询同一运算（Step2）；`False`=每查询独立采样（Step3） | `True` |
| `op_name` | `Optional[str]` | 当 `same_op=True` 时使用的运算名称；`same_op=False` 时忽略 | `"NAND"` |
| `ops_list` | `List[str]` | 运算列表 | `OPS_LIST` |
| `seed` | `int` | 随机种子 | `42` |

**异常：** `AssertionError` — `same_op=True` 且 `op_name=None` 时。

**属性：**

| 属性 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `self.A` | `[n_samples, K]` | float32 | 所有查询的输入 A |
| `self.B` | `[n_samples, K]` | float32 | 所有查询的输入 B |
| `self.op_ids` | `[n_samples, K]` | int64 | 各查询的运算索引 |
| `self.labels` | `[n_samples, K]` | float32 | 各查询的目标标签 |
| `self.K` | `int` | — | 查询数 |

**`__getitem__(idx)` 返回：** `(A[K], B[K], op_ids[K], labels[K])`

---

## 模块：`data.encoding`

### `encode_trial(...)`

```python
def encode_trial(
    A_batch: torch.Tensor,    # [B, K]  float
    B_batch: torch.Tensor,    # [B, K]  float
    op_ids:  torch.Tensor,    # [B, K]  long
    slots:   List[SlotBoundaries],
    n_input: int,
    r_on:    float = 400.0,
    r_off:   float = 10.0,
    dt:      float = 1.0,
    device:  str   = "cpu",
) -> torch.Tensor
```

**描述**：将批量布尔输入转换为全试次 Poisson 脉冲矩阵。

**参数：**

| 参数 | 类型/形状 | 含义 | 默认值 |
|------|-----------|------|--------|
| `A_batch` | `[B, K]` float | 批量输入 A（每槽一个值） | 无默认 |
| `B_batch` | `[B, K]` float | 批量输入 B | 无默认 |
| `op_ids` | `[B, K]` long | 运算索引（Step3 one-hot 编码用） | 无默认 |
| `slots` | `List[SlotBoundaries]` | K 个时间槽边界定义 | 无默认 |
| `n_input` | `int` | 输入通道数（Step1/2=2，Step3=2+n_ops） | 无默认 |
| `r_on` | `float` | 逻辑 1 对应的 Poisson 发放率（Hz） | `400.0` |
| `r_off` | `float` | 逻辑 0 对应的发放率（Hz） | `10.0` |
| `dt` | `float` | 时间步长（ms），用于 Hz→概率换算 | `1.0` |
| `device` | `str` | PyTorch 设备字符串 | `"cpu"` |

**返回：** `torch.Tensor`，形状 `[B, T, n_input]`，其中 `T = slots[-1].read_end`。值为 0.0 或 1.0（Bernoulli 采样）。

**副作用**：在 `device` 上分配张量内存；调用 `torch.bernoulli`（随机采样，受全局随机种子影响）。

---

## 模块：`snn.neurons`

### `spike_fn(v_minus_thr, beta)`

```python
def spike_fn(v_minus_thr: torch.Tensor, beta: float = 4.0) -> torch.Tensor
```

**描述**：代理梯度脉冲函数。正向为 Heaviside 阶跃，反向为快速 sigmoid 导数。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `v_minus_thr` | `Tensor [B, N]` | 膜电位减去阈值（= V - V_thr） | 无默认 |
| `beta` | `float` | 代理函数陡峭度（较大=更接近真实不可微阶跃） | `4.0` |

**返回：** `Tensor [B, N]`，正向为 0.0 或 1.0；反向梯度为 `beta / (1 + beta*|x|)^2`。

---

### `class LIFNeurons(nn.Module)`

```python
class LIFNeurons(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 10.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        refractory_steps: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
    )
```

**描述**：批量 LIF 神经元层，实现带不应期的欧拉积分、代理梯度脉冲和硬复位。

**构造参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `n_neurons` | `int` | 神经元数量（层宽） | 无默认 |
| `tau_m` | `float` | 膜时间常数（ms） | `10.0` |
| `v_threshold` | `float` | 脉冲阈值 | `1.0` |
| `v_reset` | `float` | 脉冲后复位电位 | `0.0` |
| `refractory_steps` | `int` | 不应期时步数（发放后禁止接收输入的时步数） | `2` |
| `dt` | `float` | 仿真时间步长（ms），用于计算 decay = exp(-dt/tau_m) | `1.0` |
| `surrogate_beta` | `float` | 代理梯度参数，传给 `spike_fn` | `4.0` |

**注册的 buffer：**

| buffer | 形状 | 说明 |
|--------|------|------|
| `self.decay` | 标量 tensor | `exp(-dt/tau_m)`，随模型保存，不参与优化 |

#### `init_state(batch_size, device)`

```python
def init_state(self, batch_size: int, device="cpu") -> Tuple[Tensor, Tensor]
```

**描述**：返回零初始化的神经元状态张量。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `batch_size` | `int` | 批大小 | 无默认 |
| `device` | `str` | 设备 | `"cpu"` |

**返回：** `(v, ref)` — 均为形状 `[batch_size, n_neurons]` 的零张量。

#### `forward(I_syn, v, ref)`

```python
def forward(
    self,
    I_syn: torch.Tensor,  # [B, N]
    v:     torch.Tensor,  # [B, N]
    ref:   torch.Tensor,  # [B, N]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**描述**：执行一个时间步的 LIF 更新。

**参数：**

| 参数 | 形状 | 含义 |
|------|------|------|
| `I_syn` | `[B, N]` | 突触电流（来自 `DelayedSynapticLayer.forward`） |
| `v` | `[B, N]` | 当前膜电位 |
| `ref` | `[B, N]` | 不应期倒计时（≤0 表示可发放） |

**返回：** `(spike, v_new, ref_new)` — 均为 `[B, N]`

| 返回值 | 形状 | 说明 |
|--------|------|------|
| `spike` | `[B, N]` | 0/1 脉冲（正向 Heaviside，反向代理梯度） |
| `v_new` | `[B, N]` | 更新后膜电位（发放后复位到 `v_reset`） |
| `ref_new` | `[B, N]` | 更新后不应期（发放后置为 `refractory_steps`，每步递减） |

---

## 模块：`snn.synapses`

### `class DelayedSynapticLayer(nn.Module)`

```python
class DelayedSynapticLayer(nn.Module):
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        d_max: int = 50,
        delay_param_type: str = "sigmoid",
        train_weights: bool = True,
        train_delays: bool = True,
    )
```

**描述**：带每突触可训练延迟的全连接突触层。延迟通过环形缓冲区实现，使用线性插值提供连续可微的梯度流。

**构造参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `n_pre` | `int` | 前突触神经元数 | 无默认 |
| `n_post` | `int` | 后突触神经元数 | 无默认 |
| `d_max` | `int` | 最大延迟索引；缓冲区大小 = d_max+1 | `50` |
| `delay_param_type` | `str` | `"sigmoid"`：`d = d_max * sigmoid(d_raw)`；`"direct"`：`d = clamp(d_raw, 0, d_max)` | `"sigmoid"` |
| `train_weights` | `bool` | `True`=权重为 `nn.Parameter`；`False`=`register_buffer`（冻结） | `True` |
| `train_delays` | `bool` | `True`=延迟原始参数为 `nn.Parameter`；`False`=`register_buffer` | `True` |

**参数/Buffer：**

| 名称 | 形状 | 类型 | 初始化 | 说明 |
|------|------|------|--------|------|
| `weight` | `[n_pre, n_post]` | `Parameter` 或 `buffer` | He: `randn * sqrt(2/n_pre)` | 突触权重 |
| `delay_raw` | `[n_pre, n_post]` | `Parameter` 或 `buffer` | 全 `-2.0` | sigmoid 前的原始延迟参数 |

#### `get_delays()`

```python
def get_delays(self) -> torch.Tensor
```

**描述**：返回连续延迟值（单位：时间步）。

**返回：** `Tensor [n_pre, n_post]`，值域 `[0, d_max]`（sigmoid 模式）或 `[0, d_max]`（direct 模式，clamp）。

**注意**：此函数返回的张量携带梯度（通过 `delay_raw`），用于可视化时应 `.detach()`。

#### `forward(buf)`

```python
def forward(self, buf: torch.Tensor) -> torch.Tensor
```

**描述**：从历史脉冲缓冲区计算突触电流，应用延迟和权重。

**参数：**

| 参数 | 形状 | 含义 |
|------|------|------|
| `buf` | `[B, d_max+1, n_pre]` | 历史脉冲缓冲区；`buf[:,d,:]` = (d+1) 步前的脉冲 |

**返回：** `Tensor [B, n_post]`，突触电流。

**副作用**：循环遍历 `n_pre` 个前突触神经元（Python for 循环，大 `n_pre` 时性能瓶颈）。

---

## 模块：`snn.model`

### `@dataclass SlotBoundaries`

```python
@dataclass
class SlotBoundaries:
    win_start:  int   # 输入编码窗口起始（含）
    win_end:    int   # 输入编码窗口终止（不含）
    read_start: int   # 读出窗口起始（含）
    read_end:   int   # 读出窗口终止（不含）
```

**描述**：描述一个时间槽的四个边界时间步。

---

### `make_slots(K, win_len, read_len, gap_len=0)`

```python
def make_slots(
    K: int,
    win_len: int,
    read_len: int,
    gap_len: int = 0,
) -> List[SlotBoundaries]
```

**描述**：构建 K 个等间距时间槽，返回 SlotBoundaries 列表。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `K` | `int` | 查询/槽数量 | 无默认 |
| `win_len` | `int` | 每槽输入窗口长度（时步） | 无默认 |
| `read_len` | `int` | 每槽读出窗口长度（时步） | 无默认 |
| `gap_len` | `int` | 槽间间隔时步数 | `0` |

**返回：** `List[SlotBoundaries]`，长度为 K。

**计算逻辑：**
- `slot_len = win_len + read_len + gap_len`
- 槽 k 的 `win_start = k * slot_len`
- `read_start = win_start + win_len`
- `read_end = read_start + read_len`

---

### `class SNNModel(nn.Module)`

```python
class SNNModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        d_max: int = 50,
        train_mode: str = "weights_and_delays",
        delay_param_type: str = "sigmoid",
        use_output_layer: bool = False,
        n_output: int = 1,
        readout_source: str = "hidden",
        lif_tau_m: float = 10.0,
        lif_threshold: float = 1.0,
        lif_reset: float = 0.0,
        lif_refractory: int = 2,
        dt: float = 1.0,
        surrogate_beta: float = 4.0,
    )
```

**描述**：单隐藏层 SNN 主模型，组合延迟突触层和 LIF 神经元，支持三种 train_mode 和多槽读出。

**构造参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `n_input` | `int` | 输入神经元数（Step1/2=2，Step3=2+n_ops） | 无默认 |
| `n_hidden` | `int` | 隐藏层神经元数 | 无默认 |
| `d_max` | `int` | 最大延迟（时步），缓冲区大小 = d_max+1 | `50` |
| `train_mode` | `str` | `"weights_only"` / `"delays_only"` / `"weights_and_delays"` | `"weights_and_delays"` |
| `delay_param_type` | `str` | `"sigmoid"` 或 `"direct"` | `"sigmoid"` |
| `use_output_layer` | `bool` | 是否使用 hidden→output LIF 层 | `False` |
| `n_output` | `int` | 输出神经元数（仅 `use_output_layer=True` 时有效） | `1` |
| `readout_source` | `str` | `"hidden"` 或 `"output"` | `"hidden"` |
| `lif_tau_m` | `float` | LIF 膜时间常数（ms） | `10.0` |
| `lif_threshold` | `float` | LIF 脉冲阈值 | `1.0` |
| `lif_reset` | `float` | LIF 脉冲后复位电位 | `0.0` |
| `lif_refractory` | `int` | LIF 不应期时步数 | `2` |
| `dt` | `float` | 仿真时间步长（ms） | `1.0` |
| `surrogate_beta` | `float` | 代理梯度陡峭度 | `4.0` |

**子模块：**

| 属性 | 类型 | 条件 | 说明 |
|------|------|------|------|
| `self.syn_ih` | `DelayedSynapticLayer` | 始终存在 | 输入→隐藏突触层 |
| `self.lif_h` | `LIFNeurons` | 始终存在 | 隐藏层 LIF 神经元 |
| `self.syn_ho` | `DelayedSynapticLayer` | 仅 `use_output_layer=True` | 隐藏→输出突触层 |
| `self.lif_o` | `LIFNeurons` | 仅 `use_output_layer=True` | 输出层 LIF 神经元 |
| `self.readout` | `nn.Linear(readout_in, 1)` | 始终存在 | 线性读出（始终可训练） |

#### `weight_params()`

```python
def weight_params(self) -> List[torch.nn.Parameter]
```

**描述**：返回所有可训练的权重参数（名称含 `"weight"` 且不含 `"readout"` 且 `requires_grad=True`）。

**返回：** 参数列表（`delays_only` 模式下为空列表）。

#### `delay_params()`

```python
def delay_params(self) -> List[torch.nn.Parameter]
```

**描述**：返回所有可训练的延迟原始参数（名称含 `"delay_raw"` 且 `requires_grad=True`）。

**返回：** 参数列表（`weights_only` 模式下为空列表）。

#### `readout_params()`

```python
def readout_params(self) -> List[torch.nn.Parameter]
```

**描述**：返回线性读出层的参数（始终非空）。

**返回：** `list(self.readout.parameters())`

#### `get_delays()`

```python
def get_delays(self) -> Dict[str, torch.Tensor]
```

**描述**：返回所有突触层的连续延迟值（用于可视化和分析）。

**返回：** `{"ih": Tensor[n_input, n_hidden]}`，若 `use_output_layer=True` 则额外含 `"ho": Tensor[n_hidden, n_output]`。

**注意**：返回值携带梯度，可视化时需 `.detach().cpu().numpy()`。

#### `delay_regularization()`

```python
def delay_regularization(self) -> torch.Tensor
```

**描述**：计算延迟 L1 正则化项（所有延迟的均值）。

**返回：** 标量 tensor，参与反向传播。用于 `delay_penalty` 惩罚项。

#### `forward(spike_input, slots)`

```python
def forward(
    self,
    spike_input: torch.Tensor,    # [B, T, n_input]
    slots: List[SlotBoundaries],
) -> Tuple[torch.Tensor, Dict]
```

**描述**：执行完整 T 步时间仿真，按槽累积脉冲，输出每槽 logit。

**参数：**

| 参数 | 形状/类型 | 含义 |
|------|-----------|------|
| `spike_input` | `[B, T, n_input]` | 预生成的全试次脉冲矩阵 |
| `slots` | `List[SlotBoundaries]` | K 个槽的边界定义 |

**返回：** `(logits, info)` 的元组

| 返回值 | 类型/形状 | 说明 |
|--------|-----------|------|
| `logits` | `Tensor [B, K]` | 每槽的原始输出 logit（传入 BCEWithLogitsLoss） |
| `info["total_hidden_spikes"]` | `Tensor [B]` | 每样本的隐藏层总脉冲数（能量代理） |
| `info["total_output_spikes"]` | `Tensor [B]` | 每样本输出层总脉冲数（仅 `use_output_layer=True` 时非零） |

---

## 模块：`train.trainer`

### `build_optimizer(model, cfg)`

```python
def build_optimizer(model: SNNModel, cfg) -> torch.optim.Optimizer
```

**描述**：构建带三个独立学习率参数组的 Adam 优化器。

**参数：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `model` | `SNNModel` | SNN 模型实例 |
| `cfg` | `dict` | 必须含 `"lr_w"`, `"lr_d"`, `"lr_readout"` 键 |

**返回：** `torch.optim.Adam`，参数组：
- weights (lr = cfg["lr_w"])
- delays (lr = cfg["lr_d"])
- readout (lr = cfg["lr_readout"])

**注意**：空参数组（如 `delays_only` 模式下的 `w_params=[]`）不会加入优化器，不会报错。

---

### `class Trainer`

```python
class Trainer:
    def __init__(
        self,
        model: SNNModel,
        slots: List[SlotBoundaries],
        cfg: Dict[str, Any],
        run_dir: str,
        device: str = "cpu",
    )
```

**描述**：封装完整训练运行，管理优化器、checkpoint、日志。

**构造参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `model` | `SNNModel` | SNN 模型（会 `.to(device)`） | 无默认 |
| `slots` | `List[SlotBoundaries]` | 试次时序结构 | 无默认 |
| `cfg` | `Dict` | 超参字典，必须含 `r_on, r_off, dt, spike_penalty, delay_penalty, grad_clip` | 无默认 |
| `run_dir` | `str` | 输出目录（自动创建） | 无默认 |
| `device` | `str` | PyTorch 设备字符串 | `"cpu"` |

**副作用（构造时）**：调用 `os.makedirs(run_dir, exist_ok=True)`；调用 `build_optimizer`。

#### `train_epoch(loader)`

```python
def train_epoch(self, loader: DataLoader) -> Dict[str, float]
```

**描述**：运行一个训练 epoch（启用梯度，执行 backward + step）。

**返回：** `{"loss": float, "acc": float}`

#### `eval_epoch(loader)`

```python
@torch.no_grad()
def eval_epoch(self, loader: DataLoader) -> Dict[str, float]
```

**描述**：在验证集上运行评估（无梯度）。

**返回：** `{"loss": float, "acc": float, "mean_hidden_spikes": float}`

#### `fit(train_loader, val_loader, epochs, verbose=True)`

```python
def fit(
    self,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    verbose:      bool = True,
) -> List[Dict]
```

**描述**：执行完整训练循环，保存 checkpoint 和日志。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `train_loader` | `DataLoader` | 训练集加载器 | 无默认 |
| `val_loader` | `DataLoader` | 验证集加载器 | 无默认 |
| `epochs` | `int` | 训练轮数 | 无默认 |
| `verbose` | `bool` | 每 10 epoch 打印进度 | `True` |

**返回：** 每个 epoch 的日志字典列表，包含键：`epoch, train_loss, train_acc, val_loss, val_acc, mean_hidden_spikes, time_s`。

**副作用**：
- 写入 `{run_dir}/best_model.pt`（验证准确率改善时更新）
- 训练结束后调用 `_save_log()` 写入 `{run_dir}/train_log.csv`

#### `save_config(cfg)`

```python
def save_config(self, cfg: Dict)
```

**描述**：将配置字典保存为 JSON。

**副作用**：写入 `{run_dir}/config.json`

---

## 模块：`train.eval`

### `evaluate(model, loader, slots, cfg, device="cpu")`

```python
@torch.no_grad()
def evaluate(
    model:   SNNModel,
    loader:  DataLoader,
    slots:   List[SlotBoundaries],
    cfg:     Dict,
    device:  str = "cpu",
) -> Dict
```

**描述**：在给定数据集上执行完整评估，计算所有核心指标。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `model` | `SNNModel` | 评估模型（将调用 `model.eval()`） | 无默认 |
| `loader` | `DataLoader` | 测试/验证集加载器 | 无默认 |
| `slots` | `List[SlotBoundaries]` | 时序槽定义 | 无默认 |
| `cfg` | `Dict` | 必须含 `r_on, r_off, dt` | 无默认 |
| `device` | `str` | 设备 | `"cpu"` |

**返回：** 字典，包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `"accuracy"` | `float` | 所有样本和查询的平均准确率 |
| `"per_query_acc"` | `List[float]` | 每个槽（query）的准确率，长度 K |
| `"mean_hidden_spikes"` | `float` | 每试次平均隐藏层脉冲数 |
| `"throughput_K_per_spk"` | `float` | K / mean_hidden_spikes（能量归一化吞吐量） |
| `"K"` | `int` | 查询数 |

**注意**：若 `mean_hidden_spikes=0`（无脉冲），`throughput_K_per_spk` 返回 `float("nan")`。

---

### `max_K_at_threshold(results_by_K, tau=0.95)`

```python
def max_K_at_threshold(
    results_by_K: Dict[int, Dict],
    tau: float = 0.95,
) -> int
```

**描述**：从 K 扫描结果中找到准确率满足阈值的最大 K。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `results_by_K` | `Dict[int, Dict]` | `{K: evaluate() 返回值}` | 无默认 |
| `tau` | `float` | 准确率阈值 | `0.95` |

**返回：** `int`，最大满足 `accuracy >= tau` 的 K；若无满足条件则返回 `0`。

---

### `save_eval_results(results, path)`

```python
def save_eval_results(results: Dict, path: str)
```

**描述**：将评估结果字典保存为 JSON 文件。

**副作用**：调用 `os.makedirs`（自动创建父目录）；写入 JSON 文件。

---

### `load_eval_results(path)`

```python
def load_eval_results(path: str) -> Dict
```

**描述**：从 JSON 文件加载评估结果。

**返回：** 字典。**异常**：`FileNotFoundError`（文件不存在时）。

---

### `summarize_sweep(results, key_fields=None)`

```python
def summarize_sweep(
    results: List[Dict],
    key_fields: List[str] = None,
) -> str
```

**描述**：将多个评估结果列表格式化为 Markdown 表格字符串。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `results` | `List[Dict]` | 评估结果列表 | 无默认 |
| `key_fields` | `Optional[List[str]]` | 要显示的列名；`None` 表示显示所有键 | `None` |

**返回：** `str`，Markdown 表格字符串（列宽 20 字符）。

---

## 模块：`utils.seed`

### `set_seed(seed)`

```python
def set_seed(seed: int)
```

**描述**：设置所有随机数生成器的种子，确保跨运行的可复现性。

**参数：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `seed` | `int` | 随机种子值（通常为 42） |

**副作用**：修改全局状态 — `random.seed(seed)`、`np.random.seed(seed)`、`torch.manual_seed(seed)`、`torch.cuda.manual_seed_all(seed)`（若 CUDA 可用）；设置 `cudnn.deterministic=True, cudnn.benchmark=False`。

---

## 模块：`utils.logger`

### `setup_logger(name, log_file=None, level=logging.INFO)`

```python
def setup_logger(
    name: str = "snn",
    log_file: str = None,
    level = logging.INFO,
) -> logging.Logger
```

**描述**：获取或创建命名 logger，配置格式化输出到标准输出（可选文件）。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `name` | `str` | logger 名称（`logging.getLogger(name)`） | `"snn"` |
| `log_file` | `Optional[str]` | 日志文件路径；`None` 表示仅输出到控制台 | `None` |
| `level` | `int` | 日志级别 | `logging.INFO` |

**返回：** `logging.Logger` 实例。

**副作用**：首次调用时添加 `StreamHandler` 到 `sys.stdout`（格式 `[HH:MM:SS] LEVEL  message`）；若指定 `log_file` 则添加 `FileHandler`（自动创建父目录）。幂等：若 logger 已有 handler 则直接返回，不重复添加。

---

## 模块：`utils.viz`

### `plot_training_curves(log_rows, save_path)`

```python
def plot_training_curves(log_rows: List[Dict], save_path: str)
```

**描述**：绘制训练/验证 loss 和 accuracy 曲线，保存为 PNG。

**参数：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `log_rows` | `List[Dict]` | `Trainer.fit()` 返回的日志列表，需含键 `epoch, train_acc, val_acc, train_loss, val_loss` |
| `save_path` | `str` | 输出 PNG 路径（自动创建父目录） |

**副作用**：写入 PNG 文件（dpi=120，10×4 英寸双子图）；accuracy 图含 0.95 参考水平线。

---

### `plot_delay_distribution(delays, title, save_path)`

```python
def plot_delay_distribution(delays: np.ndarray, title: str, save_path: str)
```

**描述**：将延迟矩阵显示为 imshow 热力图。

**参数：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `delays` | `np.ndarray [N_pre, N_post]` | 延迟值矩阵（`model.get_delays()["ih"].detach().cpu().numpy()`） |
| `title` | `str` | 图标题 |
| `save_path` | `str` | 输出 PNG 路径 |

**副作用**：写入 PNG（dpi=120，6×4 英寸，viridis colormap）。

---

### `plot_K_accuracy(K_values, results_by_mode, tau, save_path)`

```python
def plot_K_accuracy(
    K_values:         List[int],
    results_by_mode:  Dict[str, List[float]],
    tau:              float = 0.95,
    save_path:        str   = "k_accuracy.png",
)
```

**描述**：绘制各训练模式的 accuracy vs K 折线图，含阈值水平线。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `K_values` | `List[int]` | K 的取值列表（x 轴） | 无默认 |
| `results_by_mode` | `Dict[str, List[float]]` | `{mode: [acc for K in K_values]}` | 无默认 |
| `tau` | `float` | 阈值（水平参考线） | `0.95` |
| `save_path` | `str` | 输出 PNG 路径 | `"k_accuracy.png"` |

---

### `plot_throughput(K_values, results_by_mode, save_path)`

```python
def plot_throughput(
    K_values:         List[int],
    results_by_mode:  Dict[str, List[float]],
    save_path:        str = "throughput.png",
)
```

**描述**：绘制各训练模式的能量归一化吞吐量（K/spikes）vs K 折线图。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `K_values` | `List[int]` | K 的取值列表 | 无默认 |
| `results_by_mode` | `Dict[str, List[float]]` | `{mode: [throughput for K in K_values]}` | 无默认 |
| `save_path` | `str` | 输出 PNG 路径 | `"throughput.png"` |

---

### `plot_spike_raster(spike_train, title, save_path, slot_boundaries=None)`

```python
def plot_spike_raster(
    spike_train:     np.ndarray,
    title:           str,
    save_path:       str,
    slot_boundaries: Optional[List] = None,
)
```

**描述**：绘制脉冲点阵图（raster plot），可选高亮时间槽窗口。

**参数：**

| 参数 | 类型 | 含义 | 默认值 |
|------|------|------|--------|
| `spike_train` | `np.ndarray [T, N]` | 脉冲矩阵（0/1） | 无默认 |
| `title` | `str` | 图标题 | 无默认 |
| `save_path` | `str` | 输出 PNG 路径 | 无默认 |
| `slot_boundaries` | `Optional[List[SlotBoundaries]]` | 若提供则高亮输入窗口（蓝色）和读出窗口（红色） | `None` |

---

## 常量

### `OPS_LIST`（`data.boolean_dataset`）

```python
OPS_LIST: List[str] = ["AND", "OR", "XOR", "XNOR", "NAND", "NOR", "A_IMP_B", "B_IMP_A"]
```

8 种布尔运算的标准顺序。用于 one-hot op 编码的索引映射（Step3 中通道 2+i 对应 `OPS_LIST[i]`）。
