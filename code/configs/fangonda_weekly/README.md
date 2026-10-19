# Fangonda configurations

`base.json` provides common defaults for scenario materialization and development
runs. It is not a complete specification of a paper experiment. In particular,
its `chi_eta=1.75` default differs from the paper value of `chi_eta=1.25`.

A recorded experiment is specified by its scenario, frozen input manifest and
complete resolved configuration. These records identify the lambda parameter,
reference mixture, training and sampling seeds, evaluation seed and path count,
evaluation bank, calibration artifacts and output directory.

The selected paper configurations and their source records are indexed in
[`../paper/index.csv`](../paper/index.csv). See the
[paper configuration guide](../paper/README.md) for their scope and external
dependencies.
