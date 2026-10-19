# Sources and checksums

The public package contains the implementation, final configurations and saved
analysis inputs used for the reported results. The implementation was assembled
from research-code commit `09845123c8a3bc309f6020dc8bac1b391ebd2907`, with the
analysis notebook and supporting code added subsequently. Each recorded model
retains its training commit and seed. The two baseline implementation snapshots
record the versions used for the training table. Public copies use reader-facing
comments, messages and file names; affected snapshots record both original and public hashes. Workbook metadata excludes
local save paths and editor names. Worksheet contents are unchanged.

## File indexes

- [copy-manifest.csv](copy-manifest.csv) records original source hashes and
  current public-file hashes. Source identifiers refer to original file hashes.
- [paper-assets.csv](paper-assets.csv) maps manuscript figures and tables to
  their reference files and generated outputs.
- `results/paper_release_v1/provenance/source_manifest.csv` lists retained
  inputs and their current public checksums.
- `results/paper_release_v1/provenance/run_metadata/` contains final model
  settings, implementation versions and initialization records.

Original machine paths are represented by `external/artifact-…` identifiers.
These identify external dependencies; they are not local paths or download
links. Model weights and full evaluation banks are not included. Where a public
configuration has an adapted path, the index records both its public checksum
and the original configuration checksum. Numerical settings are unchanged.

The [data-scope record](../results/paper_release_v1/provenance/public_data_scope.json)
documents exclusion of original CHA observations. The travel-cost table uses
the recorded two-decimal values in
[travel_unmet_totals.md](../results/paper_release_v1/data/sources/analysis/travel_unmet_totals.md).
Manuscript table extracts are retained; the complete working manuscript is not
part of this package.

The release-content check covers hidden workbook properties, notebook outputs
and embedded image metadata, PDF metadata and attachments, and generated files.
The static training diagram retains seven MathML equation attachments describing
its labels. Legacy certificate and storage-alias identifiers remain compatible
with existing external artifacts. See [validation details](validation-report.json).
