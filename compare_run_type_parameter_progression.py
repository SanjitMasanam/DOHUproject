#!/usr/bin/env python3
"""Plot parameter progression across run types using fitted_model_params.csv files."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_paths import comparison_dir, params_csv
from plot_style import format_ax, make_model_grid

# The variants this script compares run types across. Both write their
# parameter tables to figures/<variant>/<run_type>/tables/.
VARIANTS = ["2013a", "2013b"]


def load_run_type_dfs(variant):
    """Read one dataframe per run type. params_csv() resolves the table name,
    so this no longer has to know that 2013b writes params_final.csv where
    2013a writes fitted_model_params.csv."""
    dfs = {}
    for run_type in (1, 2, 3):
        dfs[run_type] = pd.read_csv(params_csv(variant, run_type))
        dfs[run_type]["run_type"] = run_type
    return dfs


EXCLUDED_MODELS = {"ECEARTH", "MIROC3"}


def plot_parameter_progression(dfs, output_dir):
    # Use the list of models from the first run type as the shared model ordering,
    # dropping any models we don't want represented in these comparison plots.
    models = [m for m in dfs[1]["model"].tolist() if m not in EXCLUDED_MODELS]
    # Parameters to compare across fitting procedures.
    params = ["C", "C0", "gamma", "tau_f", "tau_s", "F_ref", "lambda", "T_eq", "a_f", "a_s"]
    param_labels = [
        r"$C$",
        r"$C_0$",
        r"$\gamma$",
        r"$\tau_f$",
        r"$\tau_s$",
        r"$F_{\rm ref}$",
        r"$\lambda$",
        r"$T_{\rm eq}$",
        r"$a_f$",
        r"$a_s$",
    ]
    run_types = [1, 2, 3]
    label_dct = {1: 'Replicate Geoffroy', 2: '50-yr Avg T_eq', 3: '50-yr Avg + LR Fit'}
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    markers = {1: "o", 2: "s", 3: "^"}
    offsets = {1: -0.2, 2: 0.0, 3: 0.2}

    # auto_rows: unlike the *_allModels.py scripts (always exactly 8 fixed
    # models), this script's model list is read back from a CSV and can vary in
    # size, so the grid must be sized to fit however many models are present.
    fig, axs = make_model_grid(models, title="Parameter Progression during Fitting Procedure Changes",
                               ylabel='A.U.', auto_rows=True, hide_unused=True)

    # Compute one shared log-scale y-axis range so every subplot uses the same visual scale.
    all_values = np.concatenate(
        [dfs[rt][params].to_numpy(dtype=float).ravel() for rt in run_types]
    )
    y_min = all_values.min() * 0.8
    y_max = all_values.max() * 1.2

    # Create one subplot per model and plot how that model's fitted parameters evolve.
    for imod, model in enumerate(models):
        ax = axs[imod]
        x = np.arange(len(params))

        # Extract each run type's parameter values once, skipping (and reusing
        # below) any run type that doesn't have this model.
        model_values = {}
        for run_type in run_types:
            df = dfs[run_type]
            row = df[df["model"] == model]
            if row.empty:
                print(f"Model {model} not found in run_type {run_type} data. Continuing...")
                continue
            y = row[params].iloc[0].astype(float).values
            model_values[run_type] = y
            ax.scatter(x, y, marker=markers[run_type], color=colors[run_type], label=label_dct[run_type])

        # Connect consecutive fitting-procedure values with arrows to show the
        # progression, only where both endpoints have data for this model.
        for i in range(len(params)):
            for rt_from, rt_to in zip(run_types[:-1], run_types[1:]):
                if rt_from not in model_values or rt_to not in model_values:
                    continue
                ax.annotate(
                    "",
                    xy=(x[i], model_values[rt_to][i]),
                    xytext=(x[i], model_values[rt_from][i]),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=1, lw=1.2),
                )

        format_ax(ax, text=model, xscale="linear", yscale="log", ylim=(y_min, y_max), legend=False, grid=False)
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45, ha="right")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
        ax.legend(title="Fitting Procedure", loc='upper right', prop={'weight': 'bold', 'size': 11})

    # Save both raster and vector versions.
    out_png = output_dir / "parameter_progression_by_model.png"
    out_pdf = output_dir / "parameter_progression_by_model.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plots to {out_png} and {out_pdf}")

if __name__ == "__main__":
    for variant in VARIANTS:
        dfs = load_run_type_dfs(variant)
        plot_parameter_progression(dfs, comparison_dir(variant))