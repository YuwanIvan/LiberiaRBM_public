# Analysis conventions and historical records

These notes describe the numerical definitions used by the analysis notebook.
The filename is retained for compatibility. For current commands and manuscript
output paths, see the [reproduction guide](../../docs/reproduction.md) and
[asset index](../../docs/paper-assets.csv).

## Time and uncertainty

Weekly demand and shipment values refer to intervals `(n-1, n]`, with boundaries
0 through 22. Inventory is observed at weeks 0 through 22. Q denotes warehouse
stock after dispatch; HF and CHA values are end-of-week inventories. Step lines
connect discrete observations and do not imply constant within-week inventory.

The manuscript inventory and shipment figures show pointwise empirical 5–95%
path bands. Additional mean-confidence plots use `mean ± 1.96 SE` for a fixed
trained policy. Total-shipment SE is calculated from the total rather than by
adding location-specific SEs. These intervals describe Monte Carlo uncertainty
in the mean, not path quantiles, simultaneous coverage or training-seed variation.

Coverage bands show marginal 1–99% state quantiles. Their time coordinates are
fine-interval starts, ending at 21.98 weeks; there is no terminal query at week 22.
Travel unmet-demand values retain the recorded two-decimal source precision,
as documented in `provenance/audited_travel_precision_audit.csv`.

## Two-stage training

The training table reads the baseline coarse and fine configurations. Historical
code snapshots in `implementation_audit/historical_code/` document block mapping,
backward training, Adam, optimizer resets, checkpoint restoration and stopping
rules for those baseline versions.

Zero-based coarse interval j initializes fine interval `10j+9`. Training then
proceeds backward within the block. Each fine interval has a new optimizer,
scheduler and early-stopping state. Terminal coarse/fine iteration budgets are
50,000/30,000; interior limits are 8,000/2,500. Validation occurs every 25/50
iterations. Terminal early stopping is disabled in these historical versions.
Interior early stopping requires at least 1,200 iterations and uses validation
check counters with strict thresholds `counter > 60/30`, corresponding to the
configured patience of 1,500 iterations.

Nonterminal fine block entries use the inner schedule. The general fallback to
a final schedule on parameter changes applies when coarse initialization is
absent. Warm starts within a block require adjacent mu, Gamma and eta to agree
within tolerance.

The R4 pool contains R0:R1:R2 counts of 16,384:8,192:8,192, with a rollout training
sampler. Initial-state perturbation has probability 0.3; conditional on
perturbation, Q is uniform on `[0.95 Q0, 1.05 Q0]` and each Zi is independently
uniform on `[0, 0.0125 Q0]`.

The configuration audit covers all 41 coarse and 41 fine dependencies. Coarse
seeds 11/12/13/14 occur 17/13/10/1 times, and all fine models use seed 16. The
baseline coarse seed is 14. Inventory normalization uses each scenario's Q0;
comparisons allow an absolute tolerance of `1e-8` for floating-point differences.
Implementation-level checks cover the two baseline historical commits. Other
recorded models are checked for configuration consistency.

## Marginal reference-state coverage

Coverage analysis uses the baseline model with lambda 0.25 and fine seed 16.
For each of Q and Z0 through Z7, the reference is the `training_sampler` sample
in `state_support_summary.csv`: 16,384 validation states per fine interval,
drawn from a 32,768-path rollout pool. The separately recorded `reference_rollout`
sample is not used for these comparisons.

Policy query states use `discrete_review_deep_splitting`, aligned by
`periodic_virtual_state_exact_fine_interval`, with 10,000 queries per interval
and 1,100 intervals. These are virtual E3 network query states, distinct from
period-end physical inventories and fine-grid policy evaluation.

Recorded outside-range fractions are averaged with query-count weights. For
the reference min–max range, the mean fractions are approximately 0.091627%
for Q and 0.001936% for Z0, and zero for the other coordinates. For the 1–99%
range they are approximately 1.257318% and 1.232573%, respectively, and zero
for the other coordinates. These fractions come from path-level logs, not
from reconstruction using marginal quantiles. Repeated queries are correlated.

At t=0.98 weeks, Q queries equal 142 while the recorded reference maximum is
about 141.995255. All queries exceed that maximum, by approximately 0.004745
units. The diagnostic retains the original thresholds and fractions; both
frequency and magnitude matter when interpreting this boundary case.

These checks describe empirical marginal coverage. They do not establish
population or joint coverage, or coverage of an unknown optimal policy.
The saved diagnostics do not include a joint-support analysis.
