"""Pointwise Monte Carlo mean CIs conditional on the fixed trained seed-16 model.

These are neither simultaneous confidence bands nor path-distribution quantiles.
No aggregate CI is obtained by summing location standard errors.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import BASELINE, POLICIES


def trajectory_data():
    inventory = pd.read_csv(BASELINE / 'inventory_trajectory_by_policy_location.csv')
    shipment = pd.read_csv(BASELINE / 'weekly_shipments_by_policy_location.csv')
    total = pd.read_csv(BASELINE / 'total_shipment_trajectory_by_policy.csv')
    return inventory, shipment, total


def _band(ax, x, mean, lower, upper, color, label, style):
    x, mean, lower, upper = [np.asarray(v, dtype=float) for v in (x,mean,lower,upper)]
    ax.plot(x,mean,color=color,label=label)
    ax.fill_between(x,lower,upper,color=color,alpha=style['ci_alpha'],linewidth=0)


def inventory_trajectories(style):
    inventory,_,_ = trajectory_data()
    locations = inventory[['location_index','location_display_name']].drop_duplicates().sort_values('location_index')
    fig, axes = plt.subplots(3,3,figsize=(12,9),sharex=True,layout='constrained')
    for ax,loc in zip(axes.flat,locations.itertuples(index=False)):
        for policy,color in zip(style['baseline_policies'],style['colors']):
            d=inventory[(inventory.policy_key==policy)&(inventory.location_index==loc.location_index)].sort_values('week')
            if len(d)!=23:
                raise ValueError(f'Expected weeks 0–22: {policy}/{loc.location_index}')
            _band(ax,d.week,d.inventory_mean,d.inventory_ci95_lower,d.inventory_ci95_upper,color,POLICIES[policy],style)
        ax.set_title(loc.location_display_name)
        ax.set_ylabel('Inventory (units)')
        ax.set_xticks([0,5,10,15,20,22])
    for ax in axes[-1]: ax.set_xlabel('Week')
    axes.flat[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Baseline inventory: mean and pointwise 95% CI',fontsize=13)
    return fig


def shipment_trajectories(style):
    _,shipment,total = trajectory_data()
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,layout='constrained')
    series=[('Total',total,'mean_total_shipment','se_total_shipment')]
    for idx in range(8):
        d=shipment[shipment.location_index==idx]
        series.append(('HF' if idx==0 else f'CHA {idx}',d,'mean_shipment','se_shipment'))
    for ax,(name,data,mean,se) in zip(axes.flat,series):
        for policy,color in zip(style['baseline_policies'],style['colors']):
            d=data[data.policy==policy].sort_values('week')
            if len(d)!=22: raise ValueError(f'Expected weeks 1–22: {policy}/{name}')
            _band(ax,d.week,d[mean],d[mean]-1.96*d[se],d[mean]+1.96*d[se],color,POLICIES[policy],style)
        ax.set_title(name)
        ax.set_ylabel('Weekly shipment (units)')
        ax.set_xticks([1,5,10,15,20,22])
    for ax in axes[-1]: ax.set_xlabel('Week')
    axes.flat[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Baseline allocation: mean and pointwise 95% CI',fontsize=13)
    return fig
