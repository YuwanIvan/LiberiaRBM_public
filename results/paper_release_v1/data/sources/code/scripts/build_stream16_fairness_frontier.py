#!/usr/bin/env python3
"""Build the paper-facing Stream-16 fairness frontier (Panel-A orientation).

This presentation-only builder reads frozen 10,000-path evaluation CSVs and
renders one PGFPlots panel. It does not run training, simulation, policy
evaluation, or experiments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if ROOT.name != "LiberiaRBM-active":
    raise RuntimeError(f"Expected LiberiaRBM-active project root, got {ROOT}")
RESULTS_ROOT = ROOT / "results"
SOURCE_ROOT = (
    RESULTS_ROOT
    / "policy_evaluation"
    / "settings"
)
OUTPUT = ROOT / "analysis/baseline_frontier"
SETTINGS = (
    (0.10, "nominal_lam0p10"),
    (0.25, "nominal_lam0p25"),
    (0.50, "nominal_lam0p50"),
    (1.00, "nominal_lam1p00"),
)
PDFLATEX = Path("/Library/TeX/texbin/pdflatex")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_row(path: Path, policy: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["Policy"] == policy]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {policy} row in {path}")
    return matches[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_records = []
    rows = []
    for fairness_weight, setting in SETTINGS:
        path = SOURCE_ROOT / setting / "periodic_review_evaluation_results.csv"
        row = policy_row(path, "E3_mean_rollout")
        rows.append(
            {
                "lambda": fairness_weight,
                "raw_fairness": float(row["Fairness_Utility_Mean"]),
                "operating_cost": float(row["Operational_Cost_Mean"]),
            }
        )
        source_records.append(
            {
                "setting": setting,
                "path": str(path),
                "sha256": sha256(path),
                "policy": "E3_mean_rollout",
                "stream": 16,
                "evaluation_paths": 10000,
            }
        )

    benchmark_path = SOURCE_ROOT / "nominal_lam0p25" / "periodic_review_evaluation_results.csv"
    benchmark_row = policy_row(benchmark_path, "Scaled_Static")
    benchmark = {
        "raw_fairness": float(benchmark_row["Fairness_Utility_Mean"]),
        "operating_cost": float(benchmark_row["Operational_Cost_Mean"]),
    }

    if [row["lambda"] for row in rows] != [0.10, 0.25, 0.50, 1.00]:
        raise ValueError("Unexpected fairness-weight ordering")
    if not all(rows[i]["raw_fairness"] < rows[i + 1]["raw_fairness"] for i in range(3)):
        raise ValueError("Stream-16 fairness is not increasing over the expected grid")
    if not all(rows[i]["operating_cost"] < rows[i + 1]["operating_cost"] for i in range(3)):
        raise ValueError("Stream-16 operating cost is not increasing over the expected grid")

    data_path = OUTPUT / "fairness_weight_frontier_stream16.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("point", "lambda", "raw_fairness", "operating_cost"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"point": "Our Policy", **row})
        writer.writerow({"point": "Fairness Only", "lambda": "", **benchmark})

    coordinates = " ".join(
        f"({row['raw_fairness']:.15g},{row['operating_cost']:.15g})" for row in rows
    )
    annotation_lines = []
    shifts = ((2, 2), (2, 2), (2, 7), (2, 2))
    for row, shift in zip(rows, shifts):
        label = "0.10" if row["lambda"] == 0.10 else f"{row['lambda']:.2f}"
        weight = ",font=\\bfseries\\scriptsize" if row["lambda"] == 0.25 else ",font=\\scriptsize"
        annotation_lines.append(
            "\\node[anchor=south west" + weight + f",xshift={shift[0]}pt,yshift={shift[1]}pt] "
            f"at (axis cs:{row['raw_fairness']:.15g},{row['operating_cost']:.15g}) "
            "{$\\lambda=" + label + "$};"
        )

    tex_path = OUTPUT / "fairness_weight_frontier_stream16_panel_a.tex"
    tex = rf"""\documentclass[tikz,border=3pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{frontierblue}}{{HTML}}{{1F4E79}}
\definecolor{{benchmarkorange}}{{HTML}}{{D77B28}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
  width=6.5in,
  height=4.5in,
  title={{Observed cost--fairness trade-off}},
  xlabel={{Raw fairness utility (higher is better)}},
  ylabel={{Operating cost (lower is better)}},
  xmin={min(row['raw_fairness'] for row in rows) - 2:.8f},
  xmax={benchmark['raw_fairness'] + 4:.8f},
  ymin={min(row['operating_cost'] for row in rows) - 0.3:.8f},
  ymax={benchmark['operating_cost'] + 0.5:.8f},
  y dir=reverse,
  grid=major,
  grid style={{black!18}},
  axis line style={{black!70}},
  tick label style={{font=\small}},
  label style={{font=\small}},
  title style={{font=\small}},
  legend style={{draw=none,fill=none,font=\small,at={{(0.02,0.02)}},anchor=south west}},
]
\addplot+[frontierblue,mark=*,mark size=2.5pt,thick] coordinates {{{coordinates}}};
\addlegendentry{{Our Policy}}
\addplot+[only marks,mark=diamond*,mark size=3.3pt,benchmarkorange]
  coordinates {{({benchmark['raw_fairness']:.15g},{benchmark['operating_cost']:.15g})}};
\addlegendentry{{Fairness Only}}
{chr(10).join(annotation_lines)}
\node[anchor=south west,font=\scriptsize,xshift=2pt,yshift=2pt]
  at (axis cs:{benchmark['raw_fairness']:.15g},{benchmark['operating_cost']:.15g}) {{Fairness Only}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")

    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "1789776000"
    subprocess.run(
        [
            str(PDFLATEX),
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={OUTPUT}",
            str(tex_path),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    figure_path = OUTPUT / "fairness_weight_frontier_stream16_panel_a.pdf"

    manifest = {
        "convention": "Stream 16; reported configuration; frozen 10,000-path evaluation",
        "orientation": "Panel A: fairness improves right; operating cost improves upward",
        "source_records": source_records,
        "fairness_only_source": {
            "path": str(benchmark_path),
            "sha256": sha256(benchmark_path),
            "policy": "Scaled_Static",
        },
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "figure": str(figure_path),
        "figure_sha256": sha256(figure_path),
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
