"""Read existing baseline marginal diagnostics; no model loading or simulation."""
import hashlib
import json
import numpy as np
import pandas as pd
from .data import RELEASE

VARIABLES = ['Q', *[f'Z_{i}' for i in range(8)]]
QUANTILES = ['min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max']
RATES = ['outside_training_min_max_fraction', 'outside_training_p01_p99_fraction']


def align_support(training, evaluation, intervals=1100):
    """Require exact fine-interval E3 alignment before joining marginal summaries."""
    training = training[(training.source == 'training_sampler') & training.variable.isin(VARIABLES)].copy()
    evaluation = evaluation[(evaluation.source == 'discrete_review_deep_splitting') & evaluation.variable.isin(VARIABLES)].copy()
    expected = {(i, v) for i in range(intervals) for v in VARIABLES}
    for label, frame, count in [('R4 training sampler', training, 16384), ('E3 query', evaluation, 10000)]:
        if frame.duplicated(['interval', 'variable']).any() or set(zip(frame.interval, frame.variable)) != expected:
            raise ValueError(f'{label}: duplicate or incomplete interval/variable grid')
        values = frame[QUANTILES].to_numpy(float)
        if not np.isfinite(values).all() or (np.diff(values, axis=1) < -1e-10).any():
            raise ValueError(f'{label}: non-finite or unordered quantiles')
        if not frame['count'].eq(count).all():
            raise ValueError(f'{label}: unexpected sample count')
    if not evaluation.alignment_mode.eq('periodic_virtual_state_exact_fine_interval').all():
        raise ValueError('Expected E3 virtual query states, not observed weekly inventory')
    if not (evaluation.training_interval.eq(evaluation.interval).all()
            and evaluation.review_interval.eq(evaluation.interval // 50).all()
            and evaluation.fine_offset.eq(evaluation.interval % 50).all()):
        raise ValueError('E3 query/training intervals are misaligned')
    rates = evaluation[RATES].to_numpy(float)
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError('Invalid recorded out-of-range fraction')
    if (rates[:, 0] > rates[:, 1] + 1e-12).any():
        raise ValueError('Min/max exceedance must not exceed p01/p99 exceedance')
    return training.merge(evaluation, on=['interval', 'variable'], suffixes=('_r4', '_e3'), validate='one_to_one').sort_values(['variable', 'interval']).reset_index(drop=True)


def load_coverage():
    source = RELEASE / 'data/baseline_coverage'
    manifest = json.loads((source / 'source_manifest.json').read_text())
    for item in manifest:
        if hashlib.sha256((source / item['name']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f"Coverage input checksum mismatch: {item['name']}")
    config = json.loads((source / 'config_used.json').read_text())
    eq, net = config['eqn_config'], config['net_config']
    if (eq['reference_mixture'], eq['lambda_param'], eq['num_time_interval'], net['train_seed']) != ('R4', .25, 1100, 16):
        raise ValueError('Not the baseline R4 / M1100 seed16 run')
    if (net['sampling_mode'], net['mix_rollout'], net['mix_box'], net['mix_shortage']) != ('rollout', 1., 0., 0.):
        raise ValueError('Training sampler is not pure reference rollout')
    joined = align_support(pd.read_csv(source / 'state_support_summary.csv'), pd.read_csv(source / 'evaluation_state_support_summary.csv'))
    joined['time_weeks'] = joined.interval * eq['total_time'] / eq['num_time_interval']
    rows = []
    for variable in VARIABLES:
        d = joined[joined.variable == variable]
        row = {'variable': variable, 'intervals': len(d), 'query_count_per_interval': int(d.count_e3.iloc[0])}
        for band, rate in zip(['min_max', 'p01_p99'], RATES):
            row[f'outside_{band}_pct'] = float(100 * np.average(d[rate], weights=d.count_e3))
            row[f'covered_{band}_pct'] = 100 - row[f'outside_{band}_pct']
            peak = d.loc[d[rate].idxmax()]
            row[f'peak_outside_{band}_pct'] = float(100 * peak[rate])
            row[f'first_peak_{band}_week'] = float(peak.time_weeks)
        rows.append(row)
    report = {'status': 'PASS', 'input_files_checked': len(manifest), 'aligned_coordinate_intervals': len(joined),
              'reference_source': 'training_sampler: logged R4 validation-state sample, 16384 per interval',
              'query_source': 'E3 periodic virtual query states, 10000 per fine interval',
              'aggregation': 'query-count-weighted recorded marginal fractions over 1100 intervals',
              'limits': 'Empirical marginal coverage only; not joint coverage or coverage of the unknown true optimum. Quantiles are not mean CIs.'}
    return joined, pd.DataFrame(rows), report
