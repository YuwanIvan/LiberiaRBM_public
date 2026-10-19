# Paper analysis

From the repository root, run:

```bash
python code/scripts/reproduce_paper.py
```

The editable entry point is [paper_results.ipynb](../notebooks/paper_results.ipynb).
Install the tested dependencies from
[analysis-requirements.txt](../../environment/analysis-requirements.txt).
The local `requirements.txt` gives broader dependency ranges.

Plotting code is in the notebook. `source_values.py`, `tables.py`, `coverage.py`
and `validation.py` read the retained inputs, calculate table values and check
numerical consistency. `data.py` resolves paths within this checkout. `style.py`
provides plotting defaults. `figures.py` and `baseline.py` provide additional
plotting helpers.

See the [reproduction guide](../../docs/reproduction.md) for output paths and
validation scope.
