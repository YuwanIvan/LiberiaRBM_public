# Model dependencies

[manifest.csv](manifest.csv) identifies the 41 fine-grid models and 41 coarse
initialization models used for the reported analyses. Their final settings and
training seeds are included. [fluid-inputs.csv](fluid-inputs.csv) identifies the
63 associated fluid-input directories.

Weights and public download links are not included. `external/artifact-…`
values are source identifiers, not accessible filesystem paths. The supplied
configuration hashes distinguish the original records from public copies with
adapted paths.

Pretrained evaluation additionally requires the weights, frozen evaluation bank
and corresponding fluid benchmark. Training requires the stage-specific
parameter files, initialization models and recorded code versions. The baseline
fine model uses the coarse model with seed 14.

Local figure and table reproduction uses the included saved results and does
not load these external artifacts. See [reproduction](../docs/reproduction.md).
