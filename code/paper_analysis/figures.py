"""Editable Matplotlib renderers using the manuscript's frozen numerical inputs.

Frontier/calibration styling is a new rendering, not a byte-identical PDF replica.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from .data import SOURCES


def model_diagram(style):
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    ax.imshow(plt.imread(SOURCES / 'paper/figure/Liberia_model_simplified.png'))
    ax.axis('off')
    return fig


def theta_calibration(style):
    values = np.loadtxt(SOURCES / 'code/configs/fangonda_weekly/theta.csv', delimiter=',').reshape(-1)
    if len(values) != 22:
        raise ValueError('Expected 22 calibrated weeks')
    fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
    ax.plot(np.arange(1, 23), values, marker='o', ms=4, color=style['colors'][0])
    ax.set(xlabel='Model week', ylabel=r'Weekly incidence probability, $\theta(n)$', xlim=(1,22))
    ax.set_xticks([1,4,8,12,16,20,22])
    return fig


def fairness_frontier(style):
    df = pd.read_csv(SOURCES / 'analysis/baseline_frontier/fairness_weight_frontier_stream16.csv')
    ours = df[df.point == 'Our Policy'].sort_values('lambda')
    fo = df[df.point == 'Fairness Only']
    fig, ax = plt.subplots(figsize=(7,4.8), layout='constrained')
    ax.plot(ours.raw_fairness, ours.operating_cost, '-o', color=style['colors'][0], label='Our policy')
    ax.scatter(fo.raw_fairness, fo.operating_cost, marker='D', color=style['colors'][1], label='Fairness-only')
    for row in ours.to_dict('records'):
        ax.annotate(r'$\lambda=' + f"{row['lambda']:.2f}" + '$', (row['raw_fairness'],row['operating_cost']), xytext=(5,7), textcoords='offset points', fontsize=9)
    ax.set(xlabel='Fairness utility (higher is better)', ylabel='Operating cost (lower is better)')
    ax.invert_yaxis()
    ax.margins(x=.10,y=.2)
    ax.legend(frameon=False,loc='lower left')
    return fig


def common_raw_frontier(style):
    df = pd.read_csv(SOURCES / 'results/common_raw_frontier/common_raw_frontier_extended_with_historical_endpoints.csv')
    fig, axes = plt.subplots(1,2,figsize=(12.4,5.1))
    for ax, key, values, fixed_key, fixed in [
        (axes[0],'Nh',[199,219,243],'Q0',142),
        (axes[1],'Q0',[124.5,142,166],'Nh',219),
    ]:
        for val,color in zip(values,style['colors']):
            group = df[(df[key]==val)&(df[fixed_key]==fixed)].sort_values('raw_lambda')
            if not np.allclose(group.raw_lambda,[.1,.15,.25,.35,.5,.7,1.]):
                raise ValueError(f'Incomplete lambda grid: {key}={val}')
            label = (r'$N_h=' if key=='Nh' else r'$Q_0=') + f'{val:g}$'
            ax.plot(group.fairness_raw,group.operational_cost_raw,'-o',color=color,ms=4,label=label)
            ax.scatter(group.iloc[0].U_FO,group.iloc[0].C_FO,marker='*',s=130,color=color,edgecolors='white',zorder=5)
        ax.invert_yaxis()
        ax.set_xlabel('Fairness utility, $U$')
        ax.legend(frameon=False,fontsize=9)
        ax.margins(.06,.1)
    axes[0].set(title=r'(a) HF demand exposure ($Q_0=142$)',ylabel='Operating cost, $C$ (lower is better)')
    axes[1].set_title(r'(b) Initial inventory ($N_h=219$)')
    handles=[Line2D([],[],marker='o',ls='',color='#555',label='Our policy'),Line2D([],[],marker='*',ls='',ms=10,color='#555',label='Fairness-only')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.90,wspace=.20)
    return fig
