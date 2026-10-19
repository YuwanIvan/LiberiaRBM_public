# Recorded paper configurations

[`index.csv`](index.csv) lists 41 coarse and 41 fine runs. Each selected JSON
retains the final numerical settings from `config_used.json`. External paths
use opaque artifact identifiers. The index provides public and original
configuration hashes and a link to the recorded runtime.

The baseline uses `chi_eta=1.25`, `lambda=0.25`, `alpha=0.5`, `Q0=142` and
`T=22`. Its coarse stage uses `M=110`, seed 14; its fine stage uses `M=1100`,
seed 16.

These configurations record completed experiments. Fields such as `config_dir`
and fluid-plan paths identify external artifacts that are not included.
Running them elsewhere requires locating those artifacts, adapting paths and
validating the parameter files. The generic `fangonda_weekly/base.json` contains
development defaults that differ from the paper baseline.

See [experiment design](../../../docs/experiment-design.md),
[reproduction instructions](../../../docs/reproduction.md) and
[model dependencies](../../../models/README.md).
