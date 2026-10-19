"""Display/export all seven manuscript tables from the frozen local data bundle."""
import json
import re
import pandas as pd
from .data import RELEASE, read

HEADERS = {
    'combined_objective':'Total objective', 'operational_cost':'Operating cost',
    'fairness_utility':'Fairness utility', 'coverage':'Coverage', 'local_fill':'Mean CHA local fill',
    'hf_shipment_share':'HF shipment share', 'kappa':r'$\kappa$',
    'CHA shipment share':'CHA shipment share', 'CHA local unmet':'CHA local unmet',
    'release fraction week8':'Week 8 release fraction', 'MismatchFreq':'Mismatch frequency',
    'S_bar':r'$\bar S$', 'S_positive':r'$S_+$', 'gap':'Objective gap vs. fairness-only',
    'Operational_Cost_Mean':'Operating cost', 'Fairness_Utility_Mean':'Fairness utility',
    'LocalFill':'Mean CHA local fill', 'cha_share_from_weekly_means':'CHA shipment share',
    'Nh':r'$N_h$', 'total_expected_demand':'Total expected demand',
    'objective_gap_vs_FairnessOnly':'Objective gap vs. fairness-only',
    'HF_shipment_share':'HF shipment share', 'HF_load_fulfillment':'HF-load fulfillment',
    'CHA_local_fill':'Mean CHA local fill', 'Q0':r'$Q(0)$',
    'objective gap':'Objective gap vs. fairness-only', 'week8 units':'Week 8 release units',
    'week8 percentage':r'Week 8 release (\%)', 'overall_coverage':'Coverage',
    'HF_load_fulfillment_rate':'HF-load fulfillment', 'total_HF_shipment':'HF shipment',
    'total_CHA_shipment':'CHA shipment',
}
ROW_LABELS = {'Fairness Only':'Fairness-only','Fluid Optimal':'Fluid-optimal',
    'nominal_lam0p25':'1', 'variance2x_lam0p25':'2', 'variance5x_lam0p25':'5',
    'rrho0p50_lam0p25':'0.50', 'rrho0p70_lam0p25':'0.70'}


def formatted_tables(style):
    long = read('data/table_values_long.csv')
    result = {'tab:baseline_parameters': read('data/baseline_parameters.csv')}
    for label,group in long.groupby('asset_id',sort=False):
        row_ids,cols = list(dict.fromkeys(group.row)),list(dict.fromkeys(group.column))
        output=[]
        for row in row_ids:
            values={}
            if label=='tab:baseline_compact_comparison': values['Policy']=ROW_LABELS.get(row,row)
            if label=='tab:demand_overdispersion_performance': values['Variance multiplier']=ROW_LABELS[row]
            if label=='tab:demand_correlation_summary': values[r'$\rho$']='0.00' if row=='nominal_lam0p25' else ROW_LABELS[row]
            for col in cols:
                cell=group[(group.row==row)&(group.column==col)].iloc[0]
                precision=style['table_precision_overrides'].get(label,{}).get(col,int(cell.decimals))
                values[HEADERS.get(col,col)]=f'{cell.value:.{precision}f}'
            output.append(values)
        result[label]=pd.DataFrame(output)
    return result


def export_tables(style, output=None):
    output=RELEASE/'exports/tables' if output is None else output
    output.mkdir(parents=True,exist_ok=True)
    tables=formatted_tables(style)
    blocks=json.loads((RELEASE/'provenance/manuscript_table_blocks.json').read_text())
    for label,df in tables.items():
        name=label.replace('tab:','')
        df.to_csv(output/f'{name}.csv',index=False)
        # Captions in the current seven tables have no nested braces.
        caption=re.search(r'\\caption\{([^}]+)\}',blocks[label]).group(1)
        latex_df=df.copy()
        def wrap_header(header):
            words=header.split()
            if len(header)>15 and len(words)>1 and '$' not in header:
                middle=len(words)//2
                return r'\makecell{' + ' '.join(words[:middle]) + r'\\' + ' '.join(words[middle:]) + '}'
            return header
        latex_df.columns=[wrap_header(c) for c in latex_df.columns]
        column_format=(r'p{0.13\textwidth}p{0.25\textwidth}p{0.49\textwidth}'
                       if label=='tab:baseline_parameters' else 'l'+'r'*(len(df.columns)-1))
        tex=latex_df.to_latex(index=False,escape=False,caption=caption,label=label,position='!ht',column_format=column_format)
        font=style.get('table_latex_font_size','small')
        if font not in ('tiny','scriptsize','footnotesize','small','normalsize','large'):
            raise ValueError('Unsupported table LaTeX font size')
        tex=tex.replace('\\begin{table}[!ht]', '\\begin{table}[!ht]\n\\centering\n\\'+font)
        if style.get('table_fit_width',True) and label!='tab:baseline_parameters':
            tex=tex.replace('\\begin{tabular}', '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}')
            tex=tex.replace('\\end{tabular}', '\\end{tabular}\n}')
        (output/f'{name}.tex').write_text(tex)
    long=read('data/table_values_long.csv')
    long.to_csv(output/'unrounded_table_values.csv',index=False)
    return tables
