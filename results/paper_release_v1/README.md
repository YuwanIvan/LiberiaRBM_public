# Frozen analysis inputs

This directory contains the saved data and source records used to reproduce the
paper's computational figures and tables. Run the analysis from the repository
root:

```bash
python code/scripts/reproduce_paper.py
```

The `data/`, `provenance/` and `implementation_audit/` directories
retain the public analysis inputs and their checksums. Original CHA records
are excluded, and the historical calibration notebook has its record-level
output removed. These changes are documented in
[`public_data_scope.json`](provenance/public_data_scope.json). The notebook
validates the 626 retained inputs before generating outputs.

## Outputs

- `exports/notebook_v2/`: executed notebook, HTML, figures, tables and validation
  reports.
- `exports/trajectory_distributions/`: inventory and shipment path-band
  figures. The `p05_p95` files are the manuscript outputs; the other variants
  are additional diagnostics.

The current manuscript mapping is
[`docs/paper-assets.csv`](../../docs/paper-assets.csv). 
Reference figures and tables are in
[`reference_outputs/`](../../reference_outputs/).

[Analysis notes](README.historical.md) explain the statistical conventions,
training records and coverage interpretation used by the notebook.
