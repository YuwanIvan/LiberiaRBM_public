# Paper experiment design

## Baseline

The baseline uses `chi_eta=1.25`, `lambda=0.25`, `alpha=0.5`, `Q0=142`,
`T=22` weeks and seven community health assistants (CHAs).
The coarse stage has `M=110` and training seed 14; the fine stage has
`M=1100` and training seed 16.

## Recorded models

The package identifies 41 fine-grid models and 41 coarse initialization models.
All fine-grid models use seed 16. The coarse models use seeds 11, 12, 13 and 14
in counts 17, 13, 10 and 1, respectively. These models cover different settings
and are not independent replications of a single experiment.

[`code/configs/paper/index.csv`](../code/configs/paper/index.csv) lists the
selected runs and configuration hashes. Run metadata records scenario-specific
demand, travel, supply and lambda settings, original code commits and input
artifacts. The generic `code/configs/fangonda_weekly/base.json` contains
development defaults and is not the paper baseline.

## Evaluation and uncertainty

The primary reported policy is periodic-review E3. Policy comparisons use
paired random inputs. Frontiers use the raw lambda parameter; normalized-lambda
explorations are separate experiments.

The inventory and shipment figures show pointwise empirical path quantiles.
Additional notebook plots show Monte Carlo confidence intervals for the mean
of a fixed trained policy. Neither describes variation across training seeds.
Coverage plots compare the recorded marginal reference-state distributions
with E3 network query states; they do not establish joint or population coverage.
