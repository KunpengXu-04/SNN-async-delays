# 架构概览图集（Mermaid）

> 本文件包含所有 Mermaid 图表，可在支持 Mermaid 渲染的 Markdown 查看器（如 VS Code、GitHub、Obsidian）中显示。

---

## 图 1：模块依赖图

```mermaid
graph TD
    subgraph "入口脚本 scripts/"
        S1[run_step1.py]
        S2[run_step2.py]
        S3[run_step3.py]
    end

    subgraph "数据层 data/"
        BD[boolean_dataset.py\nBooleanDataset\nMultiQueryDataset]
        ENC[encoding.py\nencode_trial]
    end

    subgraph "SNN 核心 snn/"
        NEU[neurons.py\nLIFNeurons\nspike_fn]
        SYN[synapses.py\nDelayedSynapticLayer]
        MOD[model.py\nSNNModel\nSlotBoundaries\nmake_slots]
    end

    subgraph "训练层 train/"
        TR[trainer.py\nTrainer\nbuild_optimizer]
        EV[eval.py\nevaluate\nmax_K_at_threshold]
    end

    subgraph "工具层 utils/"
        SEED[seed.py\nset_seed]
        LOG[logger.py\nsetup_logger]
        VIZ[viz.py\nplot_*]
    end

    S1 --> BD; S1 --> MOD; S1 --> TR; S1 --> EV; S1 --> SEED; S1 --> LOG; S1 --> VIZ
    S2 --> BD; S2 --> MOD; S2 --> TR; S2 --> EV; S2 --> SEED; S2 --> LOG; S2 --> VIZ
    S3 --> BD; S3 --> MOD; S3 --> TR; S3 --> EV; S3 --> SEED; S3 --> LOG; S3 --> VIZ

    TR --> MOD; TR --> ENC
    EV --> MOD; EV --> ENC
    ENC --> MOD

    MOD --> NEU
    MOD --> SYN
```

---

## 图 2：前向传播数据流

```mermaid
flowchart LR
    subgraph "输入编码"
        A["A[B,K] B[B,K] op_ids[B,K]"]
        ENC["encode_trial()\nPoisson 采样"]
        SI["spike_input\n[B, T, n_input]"]
        A --> ENC --> SI
    end

    subgraph "SNN 仿真 (T 步循环)"
        BUF["buf_in\n[B, d_max+1, n_input]\n延迟环形缓冲区"]
        SYN_IH["syn_ih.forward(buf)\nW_ih + D_ih\n线性插值延迟"]
        IH["I_h\n[B, n_hidden]\n突触电流"]
        LIF_H["lif_h.forward(I_h, v, ref)\nLIF 更新\n代理梯度"]
        SPK["spike_h\n[B, n_hidden]"]
        ACC["hidden_acc[k]\n[B, n_hidden]\n读出窗口累积"]

        BUF --> SYN_IH --> IH --> LIF_H --> SPK
        SI -- "x_t = [:,t,:]" --> BUF
        SPK -- "cat 更新 buf_in" --> BUF
        SPK -- "若 t ∈ readout_at[k]" --> ACC
    end

    subgraph "线性读出"
        RO["readout\n[n_hidden → 1]"]
        LOGIT["logits\n[B, K]"]
        ACC --> RO --> LOGIT
    end

    subgraph "损失"
        LOSS["BCEWithLogitsLoss\nlogits.reshape(-1)\nlabels.reshape(-1)"]
        LOGIT --> LOSS
    end
```

---

## 图 3：训练循环流程

```mermaid
flowchart TD
    START([开始 fit\nephochs 轮]) --> EPOCH{epoch loop\nepoch = 1..N}

    EPOCH --> TRAIN_LOOP[train_epoch\n遍历 DataLoader]
    TRAIN_LOOP --> FWD[_forward_batch\nencode_trial\nmodel.forward]
    FWD --> LOSS[BCEWithLogitsLoss\n+ spike_penalty\n+ delay_penalty]
    LOSS --> BWD[loss.backward]
    BWD --> CLIP[grad_clip_norm]
    CLIP --> STEP[optimizer.step\n分组 lr_w / lr_d / lr_readout]
    STEP --> TRAIN_LOOP

    EPOCH --> VAL_LOOP[eval_epoch @no_grad\n遍历 val DataLoader]
    VAL_LOOP --> VAL_METRICS[计算 val_acc\nmean_hidden_spikes]
    VAL_METRICS --> CKPT{val_acc >\nbest_val?}
    CKPT -- 是 --> SAVE[save best_model.pt\nbest_val = val_acc]
    CKPT -- 否 --> LOG_ROW
    SAVE --> LOG_ROW[记录 log_rows\n含 epoch/loss/acc/spk/time]
    LOG_ROW --> EPOCH

    EPOCH -- 完成 --> SAVE_CSV[保存 train_log.csv]
    SAVE_CSV --> END([返回 log_rows])
```

---

## 图 4：延迟缓冲区机制

```mermaid
sequenceDiagram
    participant Input as spike_input[B,T,n_input]
    participant Buf as buf_in[B, d_max+1, n_input]
    participant Syn as DelayedSynapticLayer
    participant LIF as LIFNeurons

    Note over Buf: 初始化为全零 [B, 50, n_input]

    loop t = 0, 1, 2, ..., T-1
        Input->>Syn: x_t = spike_input[:,t,:]  [B, n_input]

        Note over Syn: buf_in[:,0,:] = 1步前脉冲<br/>buf_in[:,d,:] = (d+1)步前脉冲

        Syn->>Syn: d_cont = d_max * sigmoid(delay_raw)  [n_pre, n_post]
        Syn->>Syn: d_floor = floor(d_cont).detach()
        Syn->>Syn: alpha = d_cont - d_floor  (梯度流经此处)

        loop i = 0..n_pre-1
            Syn->>Syn: s_f = gather(buf[:,d_floor[i],:], dim=1)  [B, n_post]
            Syn->>Syn: s_c = gather(buf[:,d_ceil[i], :], dim=1)
            Syn->>Syn: s_i = (1-alpha[i])*s_f + alpha[i]*s_c  (线性插值)
            Syn->>Syn: I_syn += s_i * weight[i]
        end

        Syn->>LIF: I_h [B, n_hidden]
        LIF->>LIF: V_new = decay*V + (1-decay)*I_h*not_ref
        LIF->>LIF: spike_h = Heaviside(V_new - thr)  [代理梯度]
        LIF->>LIF: 复位 V, 更新 ref (detached)

        Note over Buf: 更新缓冲区 (FIFO)
        Input->>Buf: buf_in = cat([x_t.unsqueeze(1), buf_in[:,:-1,:]], dim=1)
    end
```

---

## 图 5：多查询时间槽结构（K=3 示例）

```mermaid
gantt
    title K=3 试次时间结构 (Step2: win=20, read=10, gap=5 ms)
    dateFormat X
    axisFormat %s ms

    section Slot 0
    输入窗口 (A0,B0) : 0, 20
    读出窗口 → logit0 : 20, 30
    间隔 : 30, 35

    section Slot 1
    输入窗口 (A1,B1) : 35, 55
    读出窗口 → logit1 : 55, 65
    间隔 : 65, 70

    section Slot 2
    输入窗口 (A2,B2) : 70, 90
    读出窗口 → logit2 : 90, 100
    间隔 : 100, 105
```

**关键特性：**
- 神经元状态（`v_h`, `ref_h`, `buf_in`）跨槽连续，不重置
- 每槽有独立读出累积器 `hidden_acc[k]`
- 损失在 K=3 个查询上联合优化：`BCELoss(logits[B,3].reshape(-1), labels[B,3].reshape(-1))`

---

## 图 6：train_mode 参数冻结机制

```mermaid
graph LR
    CFG["train_mode\n(配置)"] --> LOGIC

    subgraph LOGIC["参数控制逻辑 (snn/model.py)"]
        W_COND{"weights_only OR\nweights_and_delays?"}
        D_COND{"delays_only OR\nweights_and_delays?"}
    end

    W_COND -- "是 train_w=True" --> W_PARAM["syn_ih.weight\n= nn.Parameter\n(参与优化)"]
    W_COND -- "否 train_w=False" --> W_BUF["syn_ih.weight\n= register_buffer\n(随机冻结)"]

    D_COND -- "是 train_d=True" --> D_PARAM["syn_ih.delay_raw\n= nn.Parameter\n(参与优化)"]
    D_COND -- "否 train_d=False" --> D_BUF["syn_ih.delay_raw\n= register_buffer\n(固定 ≈ 6ms)"]

    ALWAYS["readout.weight\nreadout.bias\n始终 nn.Parameter"] --> OPT

    W_PARAM --> OPT["Adam 优化器\n三个参数组\nlr_w / lr_d / lr_readout"]
    D_PARAM --> OPT
    W_BUF -.->|"不加入优化器"| X[" "]
    D_BUF -.->|"不加入优化器"| X
    ALWAYS --> OPT
```
