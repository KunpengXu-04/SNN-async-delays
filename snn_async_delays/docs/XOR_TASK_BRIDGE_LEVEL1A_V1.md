# XOR task bridge — Level 1A v1

**Status:** formal Stage I and Stage II complete; Level 1A passed. The locked
selection is `eta=0`, `lr_w=.01`, `lambda=.01`, `lr_d=.01`. Level 1B
preregistration is authorized; K greater than one remains locked.

## 1. Scientific question

Levels 0A-0D used one event and one fixed-weight synapse. Level 0D showed that
a continuous synaptic-current timing coordinate can move a production sigmoid
delay while the evaluated endpoint remains a hard LIF spike. It did **not**
show that the same bridge survives trainable weights, two classes or XOR.

Level 1A asks whether that mechanism survives the smallest complete Boolean
task. It deliberately does not ask whether delays improve XOR accuracy: XOR is
already solvable without delays. The question is whether a learned delay can
reach a declared interior timing schedule while the network simultaneously
learns all four XOR input-output mappings and emits a valid hard-spike code.

## 2. Why the protocol has two gates

The failed Phase-0 Stage A mixed together hidden representation, output
conversion and temporal training. Seven cells lost hidden activity and two
cells produced opponent collisions. A learned-delay failure under the same
interface would therefore be uninterpretable.

Level 1A separates them:

1. **Stage I: fixed-schedule interface gate.** Train weights under both a d0
   causal schedule and a fixed delay-4 oracle. Select one weight learning rate
   and one global voltage-envelope weight only if both schedules produce exact
   hard-spike truth tables in all five seeds.
2. **Stage II: learned-delay gate.** Freeze the Stage-I selection and train one
   global shared input-to-hidden delay from both an early and a late
   initialization. Compare task loss alone with three strengths of an explicit
   arrival-centroid teacher signal.

If Stage I fails, no conclusion about delay learning is permitted. If Stage I
passes but Stage II fails, the failure is localized to joint task/delay
optimization rather than basic XOR capacity or output feasibility.

## 3. Task, encoding and model

The batch is the exhaustive truth table

| A | B | XOR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

and every optimizer update sees all four patterns. This is optimization
calibration on the complete finite domain, not held-out generalization.

Input uses `[A0,A1,B0,B1]`. At `t=9`, exactly one A-value channel and one
B-value channel spike. Hence every trial contains exactly two input events;
neither bit value nor XOR class changes event count. The consecutive two-event
micro-burst from the failed Stage A is intentionally absent because it caused
an early `t=10` leakage/collision mode. Reintroducing two-event bursts is a
Level-1B robustness question, not a Level-1A tuning option.

The network is

```text
4 one-hot input neurons -> 16 LIF hidden neurons -> 2 opponent LIF outputs
```

Both weight matrices are trainable. Hidden-to-output delay is always d0. The
learned condition has one raw scalar `r`, broadcast over all `4 x 16`
input-to-hidden synapses:

\[
d(r)=8\,\sigma(r).
\]

This is deliberately simpler than per-neuron or per-synapse WAD. Level 1A asks
whether one task can preserve the scalar mechanism already established in
Level 0D. Adding 16 or 64 delay degrees of freedom now would make a failure
harder, not easier, to diagnose. Delay-granularity comparison is downstream.

Thresholds `.2` (hidden) and `.03` (output) are dimensionless simulator
arbitrary units, not volts or millivolts. `dt=1` is one simulator step; it may
be described as 1 ms only under the current nominal simulation convention.

## 4. Timing mathematics

The circular buffer implements an effective delay of `d+1` steps. If an input
spike is emitted at `t_in`, then

\[
t_h=t_{in}+d_{ih}+1,
\qquad
t_o=t_h+d_{ho}+1.
\]

With `t_in=9` and `d_ho=0`,

\[
t_o=11+d_{ih}.
\]

Therefore the d0 causal control targets `t=11`; the fixed oracle and learned
conditions target the interior delay `d*=4`, hence `t=15`. This avoids a
boundary optimum and tests both early-to-late (`raw=-2`, `d≈.954`) and
late-to-early (`raw=2`, `d≈7.046`) movement.

## 5. Forward equations

For input channel `i` and hidden neuron `j`, fractional delay interpolation is

\[
\tilde x_{ij}(t;d)=
(1-\alpha)x_i(t-1-\lfloor d\rfloor)
+\alpha x_i(t-1-\lceil d\rceil),
\quad \alpha=d-\lfloor d\rfloor .
\]

The hidden current and LIF update are

\[
I^h_j(t)=\sum_i W^{ih}_{ij}\tilde x_{ij}(t;d),
\]

\[
v^{pre}_j(t)=\rho v^{post}_j(t-1)+(1-\rho)I^h_j(t)
\mathbf 1[r_j(t)=0],
\]

\[
s_j(t)=H(v^{pre}_j(t)-\theta_h).
\]

The output layer uses the same equations with `d_ho=0`. Forward spikes are
hard Heaviside events; backward gradients use the existing fast-sigmoid
surrogate

\[
\frac{\partial s}{\partial(v-\theta)}
\approx \frac{\beta}{(1+\beta|v-\theta|)^2},\qquad\beta=4.
\]

## 6. Three loss components

For trial `b`, target train `Y` contains one spike at the declared target time
in output neuron `y_b=A_b\oplus B_b`; every other output-time element is zero.

### 6.1 Hard filtered-spike task loss

Let

\[
F_t(S)=\gamma F_{t-1}(S)+S_t,
\qquad \gamma=e^{-1/3}.
\]

Then

\[
L_{spk}=\operatorname{mean}_{b,t,c}
\left(F_t(S_{b,:,c})-F_t(Y_{b,:,c})\right)^2.
\]

Unlike signed-count BCE, this penalizes silence, the wrong opponent, early
spikes, late spikes and additional spikes over the complete trial.

### 6.2 Global pre-reset voltage envelope

The failed Stage-A warmup supervised only the target time and did not directly
suppress early opponent spikes. Here every output-time position is supervised.
With

\[
u_{btc}=\beta(v^{pre}_{btc}-\theta_o),
\]

the balanced envelope is

\[
L_{env}=\frac12\operatorname{mean}_{Y=1}\operatorname{softplus}(-u)
+\frac12\operatorname{mean}_{Y=0}\operatorname{softplus}(u).
\]

The positive and negative sets receive equal total weight despite there being
only one positive element per trial. Stage I compares
`eta in {0,.1,1}` rather than assuming this continuous interface term is
necessary.

### 6.3 Shared input-arrival centroid auxiliary

Before signed weights, the shared delayed arrival mass is nonnegative:

\[
A_b(t;d)=\sum_i \tilde x_{bi}(t;d).
\]

Define

\[
\mu(A_b)=\frac{\sum_t tA_b(t)}{\sum_t A_b(t)+\epsilon}
\]

and a fixed target trace generated at `d*=4`. The delay bridge is

\[
L_{arr}=\frac12\operatorname{mean}_b
\left(\mu(A_b(d))-\mu(A_b(d^*))\right)^2.
\]

This is deliberately an oracle timing teacher. It is not learned routing and
not a task-derived mechanism. Its purpose is to test whether the Level-0D
continuous timing coordinate remains compatible with XOR weight learning.

The total loss is

\[
L=L_{spk}+\eta L_{env}+\lambda L_{arr}.
\]

Stage I chooses `eta` and the weight LR from fixed controls. Stage II freezes
them and compares `lambda in {0,.01,.1,1}` and delay LR in `{.01,.05}`. A
`lambda=0` success is allowed and would show that the explicit arrival teacher
is unnecessary; the protocol does not force a positive auxiliary conclusion.

## 7. Matrices and locked selection

Stage I contains

\[
2\text{ schedules}\times3\eta\times3\text{ weight LRs}\times5\text{ seeds}
=90\text{ cells}.
\]

A candidate is one `(eta, lr_w)` pair across both schedules and all seeds: ten
cells. It passes only if all ten pass. Choose the lowest passing `eta`, then
the lowest weight LR.

Stage II contains one five-seed d0/wrong-target negative control plus

\[
4\lambda\times2\text{ delay LRs}\times2\text{ initial directions}
\times5\text{ seeds}=80
\]

learned cells, for 85 total. A learned candidate is one `(lambda, lr_d)` pair
across both initial directions and all seeds: ten cells. Choose the lowest
passing `lambda`, then the lowest delay LR.

No formal subset may be launched selectively. Stage II reads the complete
Stage-I decision file and refuses to run if Stage I did not pass.

## 8. Per-cell gates

Every gating cell must satisfy all of the following at the final checkpoint:

- all four XOR classifications correct and balanced accuracy `1.0`;
- output spike train exactly equals the four one-target templates;
- zero silent and zero opponent-collision patterns;
- exactly one output spike per pattern, at the declared target time;
- nonzero hidden activity for all four patterns.

Learned-delay cells additionally require `|d_final-4|<=.1` step and a correct
nonzero initial total delay-gradient direction. For gradient descent, the sign
test is

\[
\left.\frac{\partial L}{\partial r}\right|_0(d_0-d^*)>0.
\]

All task, envelope, arrival and total delay-gradient components are stored, as
are input-hidden and hidden-output weight-gradient norms. This distinguishes
wrong delay credit from absent weight credit.

## 9. Diagnostic panel and artifacts

Every run must save NPZ and a diagnostic panel during training. The panel
contains loss components, exact-interface trajectory, delay and arrival
centroid trajectories, component delay gradients, both weight-gradient norms,
truth-table input/hidden/output rasters, output pre-reset voltage versus
threshold, final weights and initial/final/target arrival traces. It is designed
to expose silence, collision, wrong timing, dead hidden representation and
wrong delay direction without choosing a visually favourable trial afterward.

Resource output remains a vector: parameters, trainable delay scalars, neuron
updates, dense MACs, measured synaptic events, delay-buffer memory and output
events. Level 1A cannot turn any of these into a Pareto or energy claim.

## 10. Claim boundary and next decision

A Level-1A pass would establish only that one shared scalar timing parameter
can be learned under complete K=1 XOR while final outputs remain exact hard
spikes. It would authorize a versioned Level 1B comparison of global,
per-neuron and per-synapse delays plus two-event-burst robustness at K=1.

K>1, learned temporal multiplexing and `n_hid x T` Pareto surfaces remain
locked. A Stage-I failure returns to representation/output design; a Stage-II
failure after Stage-I success is a negative task-level delay-optimization
result.
