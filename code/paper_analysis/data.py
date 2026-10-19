"""Paths resolve relative to this checkout, not to a user's absolute home path."""
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
RELEASE = WORKSPACE / 'results/paper_release_v1'
SOURCES = RELEASE / 'data/sources'
STREAM = SOURCES / 'results/policy_evaluation/settings'
BASELINE = STREAM / 'nominal_lam0p25'
POLICIES = {
    'E3_mean_rollout': 'Our policy', 'Scaled_Static': 'Fairness-only',
    'Status_Quo': '80–20 Rule', 'Static': 'Static', 'Myopic': 'Myopic',
    'Fluid_Optimal': 'Fluid-optimal', 'E0_frozen_average': 'E0 diagnostic',
}


def read(relative):
    return pd.read_csv(RELEASE / relative, encoding='utf-8-sig')


def export_figure(fig, name, style):
    output = RELEASE / 'exports/figures'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'{name}.{suffix}', dpi=style['dpi'], bbox_inches='tight', facecolor='white')
    return output / f'{name}.pdf'
