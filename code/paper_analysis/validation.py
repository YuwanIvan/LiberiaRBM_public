"""Local integrity and scientific consistency checks; never loads a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import RELEASE, BASELINE, STREAM, SOURCES, read
from .source_values import table_value, csvrows


def validate_inputs():
    manifest=read('provenance/source_manifest.csv')
    for row in manifest.itertuples():
        data=(RELEASE/row.bundle_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row.sha256:
            raise ValueError(f'Frozen source checksum mismatch: {row.bundle_path}')
    # Recompute audited cells again so accidental edits to the derived table fail.
    audited=csvrows(RELEASE/'provenance/audited_table_cell_audit.csv')
    long=read('data/table_values_long.csv')
    if len(long)!=144 or long.duplicated(['asset_id','row','column']).any():
        raise ValueError('Unexpected numerical table cell set')
    for cell in audited:
        r=long[(long.asset_id==cell['asset_id'])&(long.row==cell['row'])&(long.column==cell['column'])]
        if len(r)==1:
            cell = {**cell, 'source_relative': str(Path(r.iloc[0].source).relative_to('data/sources'))}
        if len(r)!=1 or not np.isclose(r.iloc[0].value,table_value(cell),rtol=0,atol=1e-9):
            raise ValueError(f'Table source mismatch: {cell}')
        if f'{r.iloc[0].value:.{int(r.iloc[0].decimals)}f}'!=cell['paper_display']:
            raise ValueError(f'Table rounding changed: {cell}')
    inv=pd.read_csv(BASELINE/'inventory_trajectory_by_policy_location.csv')
    shipments=pd.read_csv(BASELINE/'weekly_shipments_by_policy_location.csv')
    total=pd.read_csv(BASELINE/'total_shipment_trajectory_by_policy.csv')
    warehouse=pd.read_csv(BASELINE/'warehouse_inventory_trajectory_by_policy.csv')
    if inv.duplicated(['policy_key','location_index','week']).any():
        raise ValueError('Duplicate inventory coordinates')
    if not (inv.num_paths.eq(10000).all() and inv.stream_id.eq(16).all()):
        raise ValueError('Unexpected baseline path count/training stream')
    for policy in ['E3_mean_rollout','Scaled_Static']:
        d=inv[inv.policy_key==policy]
        if len(d)!=9*23 or set(d.location_index)!=set(range(-1,8)):
            raise ValueError('Baseline location/week grid incomplete')
        if not np.allclose(d.inventory_se,d.inventory_sample_sd/np.sqrt(d.num_paths),atol=1e-10):
            raise ValueError('Inventory SE is inconsistent with sample SD')
        for sign,column in [(-1,'inventory_ci95_lower'),(1,'inventory_ci95_upper')]:
            if not np.allclose(d[column],d.inventory_mean+sign*1.96*d.inventory_se,atol=1e-9):
                raise ValueError('Inventory CI is inconsistent with mean ± 1.96 SE')
        s=shipments[shipments.policy==policy]
        t=total[total.policy==policy].sort_values('week')
        q=warehouse[warehouse.policy==policy].sort_values('week')
        if len(s)!=8*22 or len(t)!=22 or len(q)!=23:
            raise ValueError('Shipment/warehouse time grid incomplete')
        sums=s.groupby('week').mean_shipment.sum().sort_index().to_numpy()
        if not np.allclose(sums,t.mean_total_shipment,atol=1e-5):
            raise ValueError('Location shipments do not sum to total shipments')
        q_inv=d[d.location_index==-1].sort_values('week')
        if not np.allclose(q.mean_warehouse_inventory,q_inv.inventory_mean,atol=1e-5):
            raise ValueError('Warehouse sources disagree')
        if not np.allclose(-np.diff(q.mean_warehouse_inventory),t.mean_total_shipment,atol=2e-4):
            raise ValueError('Warehouse balance does not agree with shipments')
    frontier=pd.read_csv(SOURCES/'analysis/baseline_frontier/fairness_weight_frontier_stream16.csv')
    for lam,setting in [(.1,'nominal_lam0p10'),(.25,'nominal_lam0p25'),(.5,'nominal_lam0p50'),(1.,'nominal_lam1p00')]:
        source=pd.read_csv(STREAM/setting/'periodic_review_evaluation_results.csv').set_index('Policy').loc['E3_mean_rollout']
        row=frontier[frontier['lambda']==lam].iloc[0]
        if not np.allclose([row.raw_fairness,row.operating_cost],[source.Fairness_Utility_Mean,source.Operational_Cost_Mean],atol=1e-9,rtol=0):
            raise ValueError('Fairness frontier differs from evaluation')
    result={'status':'PASS','frozen_files_checked':len(manifest),'table_cells_recomputed':len(long),
        'baseline_policies_checked':['E3_mean_rollout','Scaled_Static'],
        'baseline_inventory_coordinates_checked':2*9*23,
        'ci_definition':'pointwise normal 95% CI of Monte Carlo mean; fixed training seed16',
        'checks':['source SHA256','table recomputation and rounding','inventory SD/SE/CI','location shipment sums','warehouse balance','fairness frontier evaluation identity'],
        'limitations':['Appendix config audit covers selected initialization dependencies; notebook adds baseline historical-code checks',
                      'Baseline marginal coverage is separately checked by paper_analysis.coverage; joint coverage is outside current scope',
                      'PPT model schematic is externally maintained; calibration rerender is not old PDF byte replica',
                      'Travel unmet retains manuscript accepted two-decimal source precision']}
    out=RELEASE/'exports/validation.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result
