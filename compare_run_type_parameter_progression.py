#!/usr/bin/env python3
"""Plot parameter progression across run types using fitted_model_params.csv files."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_run_type_dfs(base_dir):
    # Map each fitting procedure version to the directory that contains its CSV output.
    run_type_dirs = {
        1: "geoffroy_replicate_results",
        2: "50-yr_avg_forcing_results",
        3: "50-yr_avg_tau_s_LR_fit_results",
    }
    # Read each CSV into a dataframe so the parameter values for all run types can be compared.
    dfs = {}
    for run_type, subdir in run_type_dirs.items():
        csv_path = base_dir / subdir / "fitted_model_params.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for run_type {run_type}: {csv_path}")
        dfs[run_type] = pd.read_csv(csv_path)
        dfs[run_type]["run_type"] = run_type
    return dfs


def plot_parameter_progression(dfs, output_dir):
    # Use the list of models from the first run type as the shared model ordering.
    models = dfs[1]["model"].tolist()
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
    label_dct = {1: 'Replicate 2013a', 2: '50-yr Avg T_eq', 3: '50-yr Avg + LR Fit'}
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    markers = {1: "o", 2: "s", 3: "^"}
    offsets = {1: -0.2, 2: 0.0, 3: 0.2}

    nmodels = len(models)
    ncols = 4
    nrows = int(np.ceil(nmodels / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(24, 12), constrained_layout=True)
    axs = axs.ravel()

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

        for run_type in run_types:
            df = dfs[run_type]
            row = df[df["model"] == model]
            if row.empty:
                raise ValueError(f"Model {model} not found in run_type {run_type} data")
            # Extract the parameter values for this model at the current fitting procedure.
            y = row[params].iloc[0].astype(float).values
            x_vals = x
            ax.scatter(x_vals, y, marker=markers[run_type], color=colors[run_type], label=label_dct[run_type])

        # Connect the three fitting-procedure values for each parameter with arrows to show the progression.
        for i in range(len(params)):
            y1 = dfs[1][dfs[1]["model"] == model][params].iloc[0].astype(float).values[i]
            y2 = dfs[2][dfs[2]["model"] == model][params].iloc[0].astype(float).values[i]
            y3 = dfs[3][dfs[3]["model"] == model][params].iloc[0].astype(float).values[i]
            x1 = x[i]
            x2 = x[i]
            x3 = x[i]
            # Draw directional progression arrows between run types
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=1, lw=1.2),
            )
            ax.annotate(
                "",
                xy=(x3, y3),
                xytext=(x2, y2),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=1, lw=1.2),
            )

        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, title="Fitting Procedure")

    # Hide any unused subplot slots so the figure remains compact and aligned.
    for ax in axs[nmodels:]:
        ax.set_visible(False)

    # Add a figure-level title and save both raster and vector versions.
    fig.suptitle("Parameter Progression during Fitting Procedure Changes", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "parameter_progression_by_model.png"
    out_pdf = output_dir / "parameter_progression_by_model.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plots to {out_png} and {out_pdf}")


if __name__ == "__main__":
    base_dir = Path("./2013a_figures_report")
    output_dir = base_dir / "comparison_plots"
    dfs = load_run_type_dfs(base_dir)
    plot_parameter_progression(dfs, output_dir)
