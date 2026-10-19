"""Shared display settings, editable from the notebook."""
import json
from pathlib import Path
import matplotlib as mpl


def load_style():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/paper_analysis/display.json').read_text())


def apply_style(style):
    mpl.rcParams.update({
        'font.family': style['font_family'], 'font.size': style['font_size'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .22, 'axes.axisbelow': True,
        'lines.linewidth': style['line_width'], 'figure.dpi': 110,
        'savefig.dpi': style['dpi'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
