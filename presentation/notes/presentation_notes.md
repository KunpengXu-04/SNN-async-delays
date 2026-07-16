# Presentation Speaker Notes
**Temporal Multiplexing via Trainable Synaptic Delays in SNNs**
*Estimated total time: ~25–30 minutes*

---

## How to use this document

- Each section below corresponds to one slide, in order.
- **Key points** tells you the 2–3 things you *must* say.
- **Script** is the full spoken version — read it out loud while practising, then paraphrase naturally when presenting.
- Numbers in bold match what's on the slide. Never contradict the slide's numbers.
- Suggested timing is in parentheses at the top of each section.

---

---

## Slide 1 — Title
*(~1 minute — this is just the opening, keep it short)*

**Key points:**
- Introduce the project in one sentence
- Tell the supervisor what to expect from the talk
- Set the right scope: this is a Step 1 + Step 2 progress report

**Script:**

"Thank you for taking the time to meet. Today I'll be presenting a progress report on my project investigating whether trainable synaptic delays in spiking neural networks can provide a genuine computational advantage — specifically, whether they allow the same network to process more independent queries at once.

The talk covers two completed experimental stages. First, a baseline study of single-task performance. Then a series of multi-query experiments — I'll call them Plans A, C, and D — that progressively isolate and test the role of delays. By the end I'll have three concrete claims I think are defensible for a paper submission.

The whole talk should take about twenty-five minutes, and I'm happy to take questions at any point."

---

---

## Slide 2 — Research Question
*(~3 minutes)*

**Key points:**
- State the question precisely: fixed h, fixed energy, maximize K
- Explain the three training modes clearly
- Define Max K @ 90% — this is the primary metric throughout

**Script:**

"The core question is this: given a fixed number of hidden neurons and a fixed energy budget — measured in spikes per trial — can trainable synaptic delays allow the network to handle more parallel queries simultaneously, without sacrificing accuracy?

More concretely, I define K as the number of independent boolean queries the network must answer in a single trial. The primary metric I'll use throughout is 'Max K at ninety percent accuracy' — the largest K where the network consistently hits ninety percent accuracy. The threshold is ninety rather than the typical ninety-five because Plan D, which I'll get to, involves a genuinely hard multiplexing problem, and ninety is a more informative boundary.

The experimental platform is intentionally minimal: a two-input LIF network performing boolean operations — NAND, AND, OR, XOR, and so on — with rate-coded inputs. I compare three training configurations shown in the table. Weights-only is the standard baseline with no delays. Delays-only freezes the weights and trains only the timing. Weights-and-delays trains both, and this is the main condition we care about.

The reason I use this simple platform is that it gives clean ground truth — I know exactly what the network should output — and it lets me isolate mechanism rather than fighting dataset complexity."

---

---

## Slide 3 — Step 1: Single-Op Baseline
*(~3 minutes)*

**Key points:**
- Weights-and-delays dominates at every h — not just at the best setting
- No mode reaches 95% without delays — delays are necessary even for K=1
- NAND is selected for Step 2 because it's the most stable
- Energy bonus is a free result: 19% fewer spikes at *higher* accuracy

**Script:**

"Before testing multiplexing, I needed to confirm the network can reliably solve a single query. This is Step 1. I ran eight boolean operations across three training modes and three hidden sizes — h equals ten, twenty, and fifty.

Looking at the accuracy table, the pattern is very consistent. Weights-and-delays achieves **0.974** at h=50 and averages **0.954** across all settings. Weights-only tops out at **0.893** — never reaching ninety-five percent in any of the twenty-four runs. Delays-only is the weakest — it collapses on nonlinear operations like XOR and XNOR, averaging **0.737**. So delays alone are not sufficient; you need both weights and delays trained together.

One thing I want to highlight here: this gap of roughly **seven to eight percent** between weights-and-delays and weights-only is not just at h=50 — it holds at h=10 and h=20 as well. Delays provide a consistent, robust improvement.

For Step 2 I chose NAND as the test operation because it achieves **98.9%** at h=50 with weights-and-delays — it's stable across all conditions and gives a strong baseline.

There's also an energy result that comes for free: the weights-and-delays mode uses **27.7 spikes per trial** at h=50, compared to **34.1** for weights-only. That's a nineteen percent reduction in energy at higher accuracy. This becomes relevant later when I talk about energy-normalised throughput."

---

---

## Slide 4 — Plan A: Time Alignment
*(~3 minutes)*

**Key points:**
- Plan A design: K serial slots, per-slot readout — this is the natural starting point
- The result looks amazing (K=20 at 95%) but the mechanism is not multiplexing
- Explain *why* weights-only fails and *why* delays fix it — the alignment story
- This is an important negative result: Plan A doesn't test true multiplexing

**Script:**

"Plan A is the most natural multi-query design. I divide the trial into K serial time slots — each slot is thirty-five milliseconds long, consisting of a twenty-millisecond input window, a ten-millisecond readout window, and a five-millisecond gap. Each query gets its own slot, and the readout for each query happens independently within that slot. So the total trial length scales as K times thirty-five.

The accuracy results look extremely impressive at first glance. With weights-and-delays, the network maintains above ninety-five percent accuracy all the way to K equals twenty — I couldn't find a ceiling. Weights-only, by contrast, can't even solve K equals one reliably — **79.7%** at K=1.

But here's the critical insight, and this took me some time to figure out. The seventeen percentage point advantage of delays over no-delays at K equals one has nothing to do with multiplexing. It's pure time alignment. The weights-only baseline fails because the input window ends at twenty milliseconds, but the readout window starts at twenty milliseconds. By the time the readout begins, the membrane potential from a weights-only network has already decayed to about thirteen percent of its peak — it's almost gone. Delays fix this: the trained delays shift spikes into the readout window, so the network sees a clear high-versus-low firing rate signal exactly when it needs to read out.

Between slots, the LIF decay with tau equal to ten milliseconds and a slot length of thirty-five means only about three percent residual leaks across — so slots are effectively independent.

The lesson from Plan A: delays are valuable, but the benefit here is alignment, not multiplexing. I can't use Plan A to test whether delays allow one network to answer multiple queries from a shared representation. For that, I needed a different design."

---

---

## Slide 5 — Plan C: Alignment vs Capacity
*(~3 minutes)*

**Key points:**
- Plan C isolates the two contributions with a three-condition comparison
- Alignment dominates: +11.8%, which is 4.3× the capacity effect of +2.7%
- But Plan C still doesn't test true multiplexing because of the 2K dedicated channels

**Script:**

"Plan C is designed to quantify exactly how much of delays' benefit is alignment versus genuine capacity expansion. The design change is: instead of serial slots, all K queries are injected simultaneously, and I give each query its own dedicated pair of input channels — so the network has two K input neurons. The trial length is fixed at thirty milliseconds regardless of K.

I compare three conditions. Condition A is fully trainable — weights and delays. Condition B uses trainable weights but fixes all delays to a constant twenty milliseconds — so there is alignment but no adaptive timing. Condition C is weights-only with delays fixed to zero — this is the no-delay baseline.

Looking at the table, the B-minus-C column is the alignment effect, and it averages **plus eleven-point-eight percent** across all K values. The A-minus-B column is the capacity effect — what trainable delays add on top of fixed alignment — and it averages just **plus two-point-seven percent**. The ratio is **4.3 times**: alignment is the dominant contribution, four times larger than the capacity gain.

In terms of Max K at ninety percent, the fully trainable model reaches K equals five, fixed alignment reaches K equals three, and zero delay reaches zero.

Now, the reason the capacity effect is small is that Plan C already gives the network an easy way to separate queries — dedicated input channels. Weights alone can learn to route the right channel to the right readout. So trainable delays only need to polish an already-good solution. This motivated Plan D: what happens if I take away those dedicated channels and force the network to rely entirely on timing?"

---

---

## Slide 6 — Plan D: Design
*(~3 minutes)*

**Key points:**
- Plan D: 2 shared channels only — weights structurally cannot distinguish queries
- Delays are *not just helpful* in Plan D — they are the *only possible mechanism*
- LIF residual (37%) is the fundamental challenge — it's not a bug, it's the constraint we measure against
- Contrast with Plans A and C to make the design motivation clear

**Script:**

"Plan D is the central experiment of this project. The key design change is simple but drastic: I go back to just two input channels — one for A, one for B — shared across all K queries. Queries are injected sequentially in sub-windows of ten milliseconds each. After all K sub-windows, there is a single shared readout window where the network must output all K answers simultaneously.

Why is this the critical test? Because with only two shared input channels, the weights cannot distinguish one query from another. The weight from input A to hidden neuron j is a single scalar — it fires identically when A equals one, regardless of whether that happened in sub-window zero, one, or two. The only thing that *can* encode which sub-window a spike came from is its *timing*. So delays are not merely helpful in Plan D — they are the only structural mechanism available. Weights-only with d equals zero is guaranteed to fail before I even run the experiment.

The challenge I'm measuring against is the LIF membrane residual. With tau equals ten milliseconds and sub-window length also ten milliseconds, the residual from one sub-window to the next is e to the negative one — about thirty-seven percent. So when query two arrives, thirty-seven percent of query one's membrane potential is still present. The representations of different queries overlap. That's the interference the network needs to overcome.

This is different from Plans A and C in an important way. Plan A has independent per-slot readouts — there is no interference. Plan C has 2K dedicated channels — weights can separate queries spatially. Plan D has neither of these escapes. It is the cleanest test of temporal routing via delays."

---

---

## Slide 7 — Plan D: Linear Readout Results
*(~3 minutes)*

**Key points:**
- Max K@90% = 2 with linear readout, stable across 4 seeds
- Increasing h from 20 to 50 (+150% neurons) only gives +1–4% — NOT a neuron count problem
- The bottleneck is the shared linear decoder, not the SNN representation
- This sets up the MLP experiment perfectly

**Script:**

"Here are the Plan D results with a standard linear readout — a single linear layer from hidden to K outputs.

At h=20, the network gets **94.9%** at K=1 and **91.2%** at K=2, just above the ninety percent threshold. K=3 drops to **83.5%**, which is below threshold. So Max K at ninety percent is 2 for h=20.

To test whether this is simply a capacity problem — not enough neurons — I ran the same experiment at h=50, which is a hundred-and-fifty percent more neurons, and I validated across four seeds. At h=50, K=1 reaches **95.05 plus or minus 0.21** and K=2 reaches **92.20 plus or minus 0.75** — both comfortably above ninety percent across all four seeds. But K=3 only reaches **87.52 plus or minus 1.04** — still below the threshold at every seed.

The crucial observation is that going from h=20 to h=50 only improves K=3 accuracy by about four percentage points — from 83.5 to 87.5. If the bottleneck were neurons, we'd expect a much larger gain. Instead, the gain is small and the ceiling stays at K=2.

This suggests the bottleneck is not in the SNN representation — the delays are likely creating separable temporal features — but in the readout. The linear layer Linear(h, K) must simultaneously extract K independent binary answers from a shared population of neurons whose membrane potentials carry a mixture of all K query signals, each decayed version of the previous. A linear decoder may simply not have enough expressive power to separate those signals. This is exactly the hypothesis I tested next with an MLP readout."

---

---

## Slide 8 — Plan D: MLP Readout
*(~3 minutes)*

**Key points:**
- MLP architecture: Linear(h,h) → ReLU → Linear(h,K) — one hidden layer, same parameter budget relative to h
- K=3 jumps from 87.52% to 92.68% — crosses the 90% threshold for the first time
- Max K@90% advances from 2 to 3
- K=4 is only 0.15% below 90% — the ceiling is soft

**Script:**

"To test the hypothesis that the bottleneck is the linear decoder, I replaced it with a two-layer MLP: a linear layer from h to h neurons, then a ReLU nonlinearity, then a linear layer from h to K outputs. This gives the decoder the ability to learn nonlinear decision boundaries over the hidden representations.

Looking at the table, the improvement is most pronounced at K=3: the MLP achieves **92.68%**, compared to **87.52%** for the linear readout — a jump of **5.2 percentage points**, and crucially this crosses the ninety percent threshold. Both seeds confirm this result. Max K at ninety percent advances from 2 to 3.

You can see the pattern clearly in the chart on the right. The orange MLP line stays above the ninety percent threshold through K=3 and only drops below at K=4, where it reaches **89.85%** — just fifteen hundredths of a percent below the threshold. The linear readout falls below ninety at K=3 and stays there.

This tells me two things. First, the delay-trained SNN is creating representations at K=3 that are non-linearly separable — a linear decoder can't access them but an MLP can. Second, the ceiling at K=3 under MLP is soft: K=4 is extremely close to passing. With slight hyperparameter tuning — longer sub-windows, more training — K=4 might be reachable."

---

---

## Slide 9 — 2×2 Ablation
*(~3 minutes)*

**Key points:**
- This is the key causal result — two rows of red (d=0 fails regardless of readout), two rows of green (delays succeed)
- MLP + d=0 ≈ Linear + d=0 ≈ 77% — the MLP is not the reason for improvement
- The failure is representational, not a decoder limitation
- State the conclusion clearly and directly

**Script:**

"This slide is the most important one in the talk. When I showed that MLP improves over linear readout, the natural question is: is it the delays that matter, or could MLP be powerful enough to succeed on its own, even without delays? To answer this cleanly, I ran a two-by-two ablation.

The two factors are readout type — linear or MLP — and delay configuration — d equals zero or trainable. This gives four conditions.

Look at the red rows first — the two d-equals-zero conditions. Linear plus d=0 achieves about **76%** at K=3. MLP plus d=0 achieves **77.13%** — essentially identical. Both are near the majority-class baseline. The MLP, despite its expressive power, cannot extract anything useful when delays are zero.

Now look at the green rows — both use trainable delays. Linear plus delays reaches **87.52%**. MLP plus delays reaches **92.68%**. Both are substantially above the d=0 conditions.

The conclusion is clear. The delay effect — the jump from d=0 to trainable delays — is about **sixteen percentage points**, and this is identical whether you use a linear or MLP readout. The MLP adds an additional **five percentage points** on top of that. But the MLP contribution is zero without delays.

This means the failure of the d=0 conditions is representational. The SNN is not creating separable features when delays are zero, and no decoder can fix that. Trainable delays are the necessary ingredient — they structure the hidden representations in time so that a decoder can separate them. The MLP is then a better decoder of those delay-structured representations, but it is not the cause."

---

---

## Slide 10 — Mechanism Summary
*(~2 minutes)*

**Key points:**
- Tie all three plans into a single coherent narrative
- The three plans are not contradictory — they test different things
- State the unified mechanism: alignment is primary, routing is real but bounded

**Script:**

"Let me now step back and give a unified view of what delays do, across all three experiments.

Plan A shows that delays are essential for time alignment — they shift input spikes into the readout window. Without delays, even a single query fails because the signal has decayed away by readout time. With delays, the network can solve twenty queries with no ceiling. But this is alignment, not multiplexing, because each query has its own independent readout.

Plan C lets me decompose the two contributions precisely. The alignment effect — measured by comparing fixed delay-20 to no delay — is plus eleven-point-eight percent. The adaptive capacity effect — measured by comparing trainable to fixed delays — is plus two-point-seven percent. Alignment is 4.3 times larger.

Plan D tests whether delays can enable true temporal routing — one readout, one set of channels, K queries. The answer is yes, but with limits. With a linear readout, the limit is K=2. With an MLP readout, it extends to K=3. The bottleneck is the shared decoder's ability to separate overlapping membrane-potential representations.

These three results are entirely consistent. They paint a clear picture: delays' primary value is alignment, their secondary value is temporal routing capacity, and that routing capacity is bounded by both the representation quality and the decoder's expressiveness."

---

---

## Slide 11 — Conclusions
*(~2 minutes + questions)*

**Key points:**
- State all three claims confidently — these are what will go in the paper
- Mention the energy result as a bonus
- Invite questions naturally

**Script:**

"To summarise, I have three claims I believe are strongly supported by the experiments.

First: trainable delays are structurally necessary for shared-channel temporal routing. In Plan D with only two input channels, d equals zero fails regardless of whether you use a linear or MLP readout — both plateau around seventy-seven percent at K=3. This is a representational failure, not a decoder failure. Delays are the necessary mechanism.

Second: the dominant contribution of trainable delays is time alignment, not multiplexing capacity. Across all three experimental designs, the alignment effect — fifteen to nineteen percent — is consistently four to five times larger than the capacity effect. This is an important qualification: delays are valuable primarily because they move signals to where the readout needs them, not because they create vast new representational capacity.

Third: delays do create temporally structured representations that exceed linear decodability. Max K at ninety percent goes from zero with d=0, to two with a linear readout plus delays, to three with an MLP readout plus delays. The ablation confirms the gain is attributable to delay-created representations, not to MLP capacity on its own.

As a bonus, weights-and-delays achieves all of this with about nineteen percent fewer spikes than weights-only at equivalent K, which is a free energy efficiency gain.

The open question is whether K=4 can be reached with longer sub-windows or deeper readouts — I showed that K=4 with MLP is only 0.15% below the ninety percent threshold, so it seems reachable.

I'm happy to take questions on any part of this."

---

---

## Timing guide

| Slide | Topic | Target time |
|-------|-------|-------------|
| 1 | Title | 1 min |
| 2 | Research Question | 3 min |
| 3 | Step 1 Baseline | 3 min |
| 4 | Plan A | 3 min |
| 5 | Plan C | 3 min |
| 6 | Plan D Design | 3 min |
| 7 | Plan D Linear | 3 min |
| 8 | Plan D MLP | 3 min |
| 9 | 2×2 Ablation | 3 min |
| 10 | Mechanism Summary | 2 min |
| 11 | Conclusions | 2 min |
| **Total** | | **~29 min** |

---

## Common questions to prepare for

**Q: Why not test larger h, like h=100 or h=200?**
"I already showed that going from h=20 to h=50 — a 150% increase — only gave 1–4% improvement at K=3. The trend suggests diminishing returns. The bottleneck is the decoder and the cross-slot residual, not neuron count."

**Q: Is the 37% LIF residual a fundamental limit?**
"It's a soft constraint, not a hard ceiling. If I increase the sub-window length or decrease tau, the residual drops — e.g., sub-window=20ms gives e^{-2} ≈ 13.5%. K=4 might be achievable with longer sub-windows. The current setup was chosen to demonstrate the phenomenon under realistic constraints."

**Q: Why is NAND used for Step 2 rather than XOR?**
"XOR is the hardest operation — even at h=50 with weights-and-delays it only reaches 93.2% at K=1. I needed a stable, high-accuracy single-query baseline before pushing to K>1. NAND at 98.9% gave me a clean starting point."

**Q: Could a recurrent SNN do better?**
"Possibly. The current architecture is feedforward. Recurrence could allow internal working memory to hold earlier query states. That's a natural next step, but it would confound the delay routing story I'm trying to tell."

**Q: How does this relate to biological neural coding?**
"Biological axonal delays vary from less than 1ms to tens of milliseconds depending on axon diameter and myelination. The finding that delays shift computation in time — rather than just improving accuracy — is consistent with theories of temporal coding in biological circuits, though I've been careful not to overclaim biological plausibility."
