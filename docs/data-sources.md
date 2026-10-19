# Data included in the public package

The package includes calibrated model parameters, simulation results, distance
data and inventory data. Original CHA-level records are not distributed.
The retained inputs support the paper's figure and table reproduction workflow.

| Material | Availability | Location |
| --- | --- | --- |
| Original CHA weekly observations | Excluded, including the frozen workbook copy and notebook output showing individual records | Not distributed |
| Calibrated model parameters | Included | `code/configs/` and frozen calibration sources |
| Simulation results and analysis inputs | Included | `results/paper_release_v1/data/` |
| CHA-to-HF distances | Included | `code/data/distance.xlsx` and its frozen source copy |
| Inventory and consumption data | Included | `results/paper_release_v1/data/sources/code/data/stock_comsumption.xlsx` |

## Calibration and reproduction

The public analysis reads the saved calibration parameters and evaluation
results. It does not require the original CHA workbook. Re-running the original
calibration procedure requires separately authorized access to the withheld
records. Historical scripts and calibration metadata may still name
`CHA_weekly.xlsx` to document the input used in the research; that filename is
not an available download.

The historical `parameter.ipynb` retains its calibration code and parameter
outputs, with the output displaying CHA-level observations removed. It is a
source reference, not the public figure-reproduction entry point.

## Simulation result scope

The baseline comparison source contains the seven policies evaluated with
stream 16, as used by the paper tables. Sensitivity-analysis inputs retain the
reported parameter settings and policy comparisons. Final training seeds,
evaluation settings and source-quality flags are retained. Cluster job IDs
and branch labels are omitted from the published result tables.

## Source records

[`source_manifest.csv`](../results/paper_release_v1/provenance/source_manifest.csv)
lists the 626 retained frozen inputs and their current checksums.
[`public_data_scope.json`](../results/paper_release_v1/provenance/public_data_scope.json)
records the excluded files, notebook adaptation and manifest revision. Historical
source hashes are retained for provenance; they do not imply that excluded
records are distributed. The supported workflow is described in
[reproduction instructions](reproduction.md).

## Licensing

The MIT software license does not cover these datasets or research outputs.
No separate data license is assigned by this package. See
[license scope](licensing.md).
