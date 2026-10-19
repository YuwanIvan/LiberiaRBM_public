# LiberiaRBM

Code and computational materials for **A Dynamic Inventory Rationing Model for
Last-Mile Delivery of Medication in Liberia's Community Health Program**.

The analysis notebook reproduces the paper's computational figures and table
values from saved evaluation results. The repository also includes the model
implementation and recorded experiment configurations.

## Quick start

Use Python 3.13; the analysis has been tested with Python 3.13.5. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/analysis-requirements.txt
python code/scripts/reproduce_paper.py
```

Figure and table reproduction runs locally without a GPU, PyTorch, Gurobi or
model weights. You can also run all cells in
[`code/notebooks/paper_results.ipynb`](code/notebooks/paper_results.ipynb)
using the same environment.

Outputs are saved under `results/paper_release_v1/exports/`, including an
executed notebook and HTML at `exports/notebook_v2/`.
[`docs/paper-assets.csv`](docs/paper-assets.csv) maps each manuscript figure
and table label to its reference file and generated output. The notebook also
produces additional diagnostics, identified in the section descriptions.

## Contents

| Directory | Contents |
| --- | --- |
| `code/` | Model implementation, recorded configurations and analysis notebook |
| `results/paper_release_v1/` | Frozen analysis inputs and source records |
| `reference_outputs/` | Nine manuscript figure files and eight tables, with reference CSVs |
| `models/` | Model and fluid-input manifests; weights are not included |
| `environment/` | Analysis dependencies and a training environment template |
| `docs/` | Reproduction instructions, experiment design and provenance |

## Reproduction scope

| Task | Availability |
| --- | --- |
| Regenerate computational figures and table values | Included; follow the quick start |
| Recreate the two schematic diagrams | Reference images only; editable originals are not included |
| Reevaluate pretrained models | Requires external weights, fluid artifacts and evaluation banks |
| Retrain the paper models | Requires external inputs, environment setup and the recorded historical code versions |

The 82 recorded run configurations use external artifact identifiers and
require adaptation before use on another system. The development defaults in
`code/configs/fangonda_weekly/base.json` differ from the paper baseline.
See [reproduction instructions](docs/reproduction.md) and
[model dependencies](models/README.md) for details.

## Data availability

Calibrated parameters, simulation results, distance data and inventory data are
included. Original CHA-level records and notebook outputs displaying those
records are excluded. Figure and table reproduction uses the retained inputs;
recalibration from original observations requires separate access to the data.
See [data sources](docs/data-sources.md).

## Validation

The analysis checks input hashes, recomputes table values and verifies numerical
consistency of the saved trajectories and coverage summaries. The quick start
passed in a fresh Python 3.13.5 environment on macOS arm64, using an independent
source copy with no previous outputs. All 20 code cells completed, all eight
table CSVs matched their references, and the six analysis tests passed.
Full model retraining has not been verified.
See [validation details](docs/reproduction.md#validation), the
[clean-environment report](docs/clean-environment-validation.json) and the
[validation summary](docs/validation-report.json).

## Citation and licensing

Citation metadata are provided in [CITATION.cff](CITATION.cff).
The project's source code and software documentation are licensed under the
[MIT License](LICENSE). This license does not cover the datasets, experiment
records, manuscript materials or reference outputs bundled with the repository.
Original CHA-level records are not distributed, and no separate data license
is assigned here. See the [license scope](docs/licensing.md) and
[data sources](docs/data-sources.md).
