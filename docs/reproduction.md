# Reproducing the paper's figures and tables

## Running the analysis

Install the pinned dependencies in
[`environment/analysis-requirements.txt`](../environment/analysis-requirements.txt),
then run from the repository root:

```bash
python code/scripts/reproduce_paper.py
```

The runner uses the active Python interpreter for the notebook kernel. It saves
an executed notebook and HTML under
`results/paper_release_v1/exports/notebook_v2/`, leaving the source notebook
available for editing. This analysis uses saved results and requires no model
weights or cluster connection.

## Finding the paper outputs

[`paper-assets.csv`](paper-assets.csv) maps manuscript labels to reference files
and generated outputs. The inventory and shipment figures in the paper use
pointwise empirical 5–95% path bands. Their filenames end in `p05_p95.pdf`
under `exports/trajectory_distributions/`.

The notebook also produces mean confidence intervals, 1–99% path bands, a
heatmap and coverage diagnostics. These are additional analyses. The eight
table CSVs provide the numerical source views; the corresponding manuscript
LaTeX tables are in `reference_outputs/tables/`. The training table's output
filename is `appendix_e_two_stage_training.csv`, mapped to the manuscript
label `tab:two_stage_training`.

## Validation

The notebook verifies 626 registered frozen-input hashes, recomputes 144 recorded
numeric table cells, checks reference output hashes, and validates coverage
alignment and baseline path reconstruction. The recorded checks also include
3,280 comparisons of training-configuration fields, with no mismatches, and
agreement of all eight exported table CSVs with their references.

Run the six analysis tests with:

```bash
cd code
python -m unittest paper_analysis.test_release paper_analysis.test_coverage
```

The README workflow also passed in a fresh Python 3.13.5 virtual environment on
macOS 15.7.3 (arm64), using the final public inputs. The test used an independent source copy without existing
outputs or runtime caches. Dependencies installed in approximately 23 seconds;
the complete notebook ran in approximately 17 seconds. All 20 code cells
completed, all seven generated manuscript figure files and eight table CSVs
were present, and all eight CSVs matched their references byte for byte. The
six analysis tests also passed. Times depend on the machine and network.

The [clean-environment report](clean-environment-validation.json) records the
installed package versions, checks and limitations. Other operating systems
and Python versions have not been tested by this check.

## Pretrained evaluation and training

Model weights and download links are not yet available in this repository.
Each recorded run depends on its parameter CSVs, fluid benchmark, frozen
evaluation bank and, for fine-grid models, its coarse initialization model.
External artifact identifiers and code versions are recorded in the run metadata. These dependencies
must be available before a run can be reproduced on another system.

The baseline uses coarse seed 14 and fine seed 16. Evaluations use paired random
inputs for policy comparisons. The reported policy is periodic-review E3;
fine-grid evaluations are separate diagnostics. The paper's frontiers use the
raw lambda parameter. See [experiment design](experiment-design.md).

The included training and evaluation scripts provide the implementation, but
cluster submission tools and a complete portable training workflow are not
included. `environment/training-environment.yml` is a starting template rather
than a verified environment lock for all historical runs. Each model's recorded
commit identifies the code version used to train it.

The optional demand-correlation bundle utility uses the current home directory
by default. Set `LIBERIARBM_SHARED_ROOT` to use a different storage root.

## Calibration

The public workflow uses saved calibration parameters. The historical calibration
helper reads `CHA_weekly.xlsx` and `distance.xlsx`; the CHA workbook is excluded
from this package, so recalibration requires separately authorized access to the
original observations. Distance and inventory data remain included.

Generic calibration and development defaults include historical eta conventions;
the recorded paper configurations and scenario materializer specify the paper
settings. See [data sources](data-sources.md) for the included data and the
record of exclusions.
