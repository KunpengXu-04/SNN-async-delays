**Subject:** SNN Temporal Multiplexing — Progress Report (Step 1 & Step 2)

---

Dear [Supervisor's Name],

I hope this email finds you well. I am writing to give you an update on my progress on the SNN temporal multiplexing project. I have completed Step 1 (single-query baseline) and three experimental iterations of Step 2 (multi-query scaling). A detailed technical appendix with all results tables and figures is attached (`progress_report_appendix.md`).

---

**Step 1 — Single-Query Baseline**

Step 1 evaluated three training modes (`weights_only`, `delays_only`, `weights_and_delays`) across eight boolean operations and three hidden sizes (h = 10, 20, 50). The main finding is that joint training of weights and delays (`weights_and_delays`) consistently outperforms the other two modes, achieving a mean accuracy of 95.4% vs. 87.8% for `weights_only` and 73.7% for `delays_only`. This confirms that trainable synaptic delays are beneficial at the single-query level and establishes NAND (the most stable operation across all modes) as the focal operation for Step 2.

---

**Step 2 — Three Experimental Designs**

Step 2 was conducted in three successive iterations. Each iteration was motivated by a gap discovered in the previous one, and all three were framed around the core question: *do trainable synaptic delays enable genuine temporal multiplexing?*

**Plan A — Serial Time-Slot Multiplexing.** The first design divided each trial into K sequential slots, one query per slot, with per-slot readout windows. `weights_and_delays` maintained ≥95% accuracy from K = 1 all the way to K = 20, while `weights_only` (fixed d = 0) plateaued at ~80% even at K = 1. However, further analysis revealed that the LIF membrane time constant (τ_m = 10 steps) relative to slot length (35 steps) produces near-perfect natural slot isolation (~3% cross-slot residual). Each slot is effectively an independent single-query problem — no genuine multiplexing is required.

**Plan C — Simultaneous Multi-Query (2K Input Channels).** To force genuine resource competition, the second design injected all K queries simultaneously over a fixed T = 30 step trial, using 2K dedicated input channels (A_k, B_k per query). A three-condition comparison decomposed the delay contribution into an *alignment effect* (fixed d = 20 vs. d = 0: +11.8%) and a *capacity effect* (trainable delays vs. fixed d = 20: +2.7%). The alignment effect dominated by a factor of 4.3×. The modest capacity gain occurs because dedicated per-query input channels allow weights to separate queries spatially, reducing the role of temporal routing.

**Plan D — Sequential Shared-Channel Injection.** The third design used only 2 shared input channels for all K queries, injecting query k within sub-window [k × sub_win, (k+1) × sub_win). This makes delays structurally necessary: a fixed d = 0 weight cannot distinguish which time window an input came from. At K = 1 and K = 2 (sub_win ≥ 10 steps), `weights_and_delays` achieved +19.3% and +14.8% over d = 0, providing the clearest demonstration of temporal routing to date. However, at K ≥ 3 (sub_win < 10 steps), signal sparsity causes both conditions to converge toward the label prior (~75%), limiting how far K can currently be tested.

---

**Open Question — Framing for the Thesis**

Across all three designs, the primary mechanism of trainable delays appears to be *time alignment* (shifting input spikes to arrive within the readout window) rather than *temporal multiplexing capacity* in the sense of routing distinct queries to distinct neuron firing times. The capacity advantage of trainable delays over a well-chosen fixed delay is real but small (~2–3%). Plan D offers the cleanest evidence of genuine temporal routing, but is currently limited by signal sparsity at K ≥ 3.

I would appreciate your guidance on two points:
1. Is the framing of "necessary alignment mechanism" (rather than "scalable multiplexing capacity") a defensible thesis contribution, or should we push further to demonstrate capacity scaling?
2. For Plan D, would extending `win_len` from 20 to 60 steps (so that K = 3 has sub_win = 20 steps, matching the signal density of K = 1) be a worthwhile next experiment?

---

**Planned Next Steps**

- **Isolation sweep (Experiment A):** Vary τ_m and gap_len to map the conditions under which slot isolation breaks down and delays become genuinely necessary for Plan A.
- **Plan D with longer win_len:** Run Plan D with win_len = 60 to test temporal routing at K = 3–5 with sufficient signal.
- **Step 3 (mixed operations):** Extend to trials where different queries test different boolean operations, increasing task heterogeneity.

Please let me know if you would like to discuss any of this further.

Best regards,  
Kunpeng Xu
