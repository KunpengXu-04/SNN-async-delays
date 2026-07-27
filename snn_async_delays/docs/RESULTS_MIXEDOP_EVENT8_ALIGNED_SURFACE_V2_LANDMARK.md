# Event8-aligned mixed-operation surface v2: landmark results

Status: all 25 single-seed exploratory cells are technically valid. The
preregistered full-sweep gate fails and the full surface remains locked.

## Registered decision

All three required fixed-oracle points pass. The curriculum-assisted model
passes the joint function/activity/oracle-vector gate only at `lowN_lowT` and
`highN_lowT`. It fails at `lowN_highT`, `highN_highT`, and the geometric
center, including the required center. Thus only two of the required three
curriculum landmarks pass and `full_sweep_authorized_by_results=false`.

This negative gate decision is binding. No result below retroactively changes
the registered endpoint or authorizes the 95 remaining grid cells.

## Reliability results

Independent spatial d0 and the fixed temporal oracle are already essentially
perfect at the smallest point `(N=20,w=4,T=30)` and remain saturated at every
landmark. The curriculum-assisted temporal model is also essentially perfect
functionally at all five points. Consequently these conditions contain no
observable width or latency transition from which to infer a factor law.

Shared d0 remains at `.50` worst-query balanced accuracy at all five points.
This shows that the current shared-window decoder needs temporal separation,
but it is not a general proof that every shared representation requires delay:
the d0 model concentrates activity in the first window and lacks an alternative
query-position interface.

Task-only WAD is `.50` at four points. At `highN_lowT` it reaches `.9243`
worst-query balanced accuracy and `.9053` exact-trial accuracy, but its maximum
oracle-vector error is `14.23` steps. Its learned delays
`[.95,8.02,1.57,.77,16.60]` are not the declared routing schedule. This is
another direct example of functional compensation without identifiable timing.

## Endpoint inconsistency discovered after the frozen run

The registered curriculum loss and schedule gate are mathematically
inconsistent when the output-window length changes. With event8 at steps 6--9
and the simulator's one-step arrival offset, the event centroid before delay is
`8.5`. The centroid loss therefore targets

\[
d_q^{center}=10+qw+\frac{w}{2}-8.5
             =1.5+\frac{w}{2}+qw.
\]

The fixed oracle and gate instead use

\[
d_q^{oracle}=3+qw.
\]

Their deterministic separation is `w/2-1.5`: `.5` step at `w=4`, `2.5`
steps at `w=8`, and `4.5` steps at `w=12`. The observed curriculum oracle
errors are `.36/.40` at `w=4`, `2.45` at `w=8`, and `4.10/4.20` at `w=12`.
This pattern is therefore partly designed into the protocol. The three failed
curriculum schedule gates cannot be interpreted simply as optimization or WAD
failure.

The inconsistency does not imply that all high-T routing is correct. At the two
`w=12` points, the final q4 delay is slightly earlier than the lower bound
needed to place the entire four-step packet in q4's window. A future protocol
must choose one mechanism endpoint in advance: exact oracle vector, event-
centroid target, or full-packet window containment. They are not interchangeable.

## Resource interpretation

At `(N=20,w=4,T=30)`, spatial d0, oracle and curriculum are all essentially
perfect, so the observed hidden-neuron compression ratio is one, not less than
one. The grid starts too high to demonstrate hidden compression.

At that same matched point, the temporal oracle has five times as many input-
hidden synapses, dense synapse MACs and measured input-hidden events as the
independent spatial model. It has about four times as many trainable parameters,
4.2 times the decoder MACs, and 24 times the delay-buffer elements. Neuron
updates are equal because both use the same total hidden count and T. Hence
`N_hidden*T` alone materially understates the temporal model's resource cost.

## Scientific conclusion

The landmark supports a narrow architectural statement: hand-scheduled event8
routing and explicit timing curriculum can solve this K=5 workload with 20
shared hidden neurons. It does not demonstrate autonomous learned routing,
hidden compression, a better resource Pareto frontier, or whether time or
neurons are generally more valuable.

The existing full grid should not run. Besides failing its registered gate, it
contains mostly larger widths after all passing baselines have already
saturated. The appropriate next action is an endpoint-reconciliation audit and
then a new transition-region protocol below 20 total hidden neurons, not an
expansion of v2.

The post-decision YAML SHA-256 is
`A3F1CDBF782F061F7AC171E4DC292753FBC625281B25623018B50FB5246E699A`.
Only status/authorization fields advanced after the frozen run; the full-sweep
authorization remains false.
