"""Read numerical table sources from the public analysis bundle."""
import csv
import re
from pathlib import Path
from .data import RELEASE, WORKSPACE as ROOT

def csvrows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def get_one(rows, **selection):
    found = [r for r in rows if all(str(r[k]) == str(v) for k, v in selection.items())]
    if len(found) != 1:
        raise ValueError(f'Expected one row for {selection}, got {len(found)}')
    return found[0]


def table_value(cell):
    """Independently re-read the frozen source, never use audit source_value as input."""
    relative = cell.get('source_relative') or str(Path(cell['source_file']).relative_to(ROOT))
    path = RELEASE / 'data/sources' / relative
    field, row = cell['source_field_or_formula'], cell['row']
    if path.suffix == '.md':
        values = re.search(r'Total CHA local unmet is ([\d.]+)/([\d.]+)/([\d.]+);', path.read_text())
        if not values:
            raise ValueError('Accepted travel precision passage changed')
        idx = ['kappa0p05_lam0p25', 'nominal_lam0p25', 'kappa0p25_lam0p25'].index(row)
        return float(values.group(idx + 1))
    rows = csvrows(path)
    if path.name == 'baseline_policy_seed_rows.csv':
        policy = cell['row_filter'].split('policy=')[1]
        selected = get_one(rows, stream_id='16', policy=policy)
    elif path.name == 'diagnostic_metrics.csv':
        selected = get_one(rows, setting_id=row, policy_key='E3_mean_rollout')
    elif 'mismatch_metrics' in path.name:
        selected = get_one(rows, policy='E3_mean_rollout')
    elif path.name == 'periodic_review_evaluation_results.csv':
        selected = get_one(rows, Policy='E3_mean_rollout')
        if field == 'E3_mean_rollout.Total objective - Scaled_Static.Total objective':
            return float(selected['Total objective']) - float(get_one(rows, Policy='Scaled_Static')['Total objective'])
    elif path.name == 'periodic_review_auxiliary_metrics.csv':
        selected = get_one(rows, **{'': 'E3_mean_rollout'})
    elif path.name == 'hf_demand_summary.csv':
        selected = get_one(rows, Nh=row)
    elif path.name == 'initial_inventory_summary.csv':
        selected_rows = [r for r in rows if float(r['Q0']) == float(row)]
        if len(selected_rows) != 1:
            raise ValueError('Q0 selection is not unique')
        selected = selected_rows[0]
        if field == '100 * week8_release_fraction':
            return 100 * float(selected['week8_release_fraction'])
    else:
        raise ValueError(f'Unmapped source: {relative}')
    return float(selected[field])
