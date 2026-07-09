#!/usr/bin/env python3
"""Plot parameter progression across run types using fitted_model_params.csv files."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def format_ax(ax, title="", xlabel="", ylabel="", text="",
            xscale="linear", yscale="linear",
            xlim=None, ylim=None,
            xticks=None, yticks=None,
            xspacing=True, yspacing=True,
            legend=True, legend_loc='upper right', grid=True):

   ax.set(title=title, xlabel=xlabel, ylabel=ylabel,
         xscale=xscale, yscale=yscale,
         xlim=xlim, ylim=ylim)

   if text:
      ax.text(0.02, 0.98, text, transform=ax.transAxes, weight='bold',
               fontsize=15, va="top", ha="left",
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4))

   if xticks is not None: ax.set_xticks(xticks)
   if yticks is not None: ax.set_yticks(yticks)

   ax.tick_params(labelsize=14, width=2, length=8, direction="in")
   if grid: ax.grid(alpha=0.3)
   if legend: ax.legend(loc=legend_loc, prop={'weight': 'bold', 'size': 12})


def make_model_grid(
   models,
   width_per_ax=6,
   height_per_ax=6,
   dpi=None,
   title=None,
   xlabel=None,
   ylabel=None,
   ncols=4,
   nrows=2,
   right=0.98,
   wspace=0.12,
   hspace=0.18,
):
   # Unlike the *_allModels.py scripts (always exactly 8 fixed models), this
   # script's model list is read back from a CSV and can vary in size, so the
   # grid must be sized to fit however many models are actually present.
   nmodels = len(models)
   nrows = int(np.ceil(nmodels / ncols))

   fig, axs = plt.subplots(
      nrows,
      ncols,
      figsize=(width_per_ax * ncols, height_per_ax * nrows),
      dpi=dpi,
      constrained_layout=False,
   )

   fig.subplots_adjust(left=0.05, right=right, bottom=0.075, top=0.95, wspace=wspace, hspace=hspace)

   if title:
      fig.suptitle(title, fontsize=20, fontweight="bold")

   if xlabel:
      fig.text(0.5, 0.02, xlabel, ha='center', fontsize=18, fontweight="bold")

   if ylabel:
      fig.text(0.02, 0.5, ylabel, ha='center', va='center', fontsize=18, fontweight="bold", rotation=90.)

   axs = np.asarray(axs).ravel()
   for ax in axs[nmodels:]:
      ax.set_visible(False)

   return fig, axs


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
        if base_dir == Path("./2013a_figures"): csv_path = base_dir / subdir / "fitted_model_params.csv"
        else: csv_path = base_dir / subdir / 'tables' / 'params_final.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for run_type {run_type}: {csv_path}")
        dfs[run_type] = pd.read_csv(csv_path)
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
    label_dct = {1: 'Replicate 2013a', 2: '50-yr Avg T_eq', 3: '50-yr Avg + LR Fit'}
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    markers = {1: "o", 2: "s", 3: "^"}
    offsets = {1: -0.2, 2: 0.0, 3: 0.2}

    fig, axs = make_model_grid(models, title="Parameter Progression during Fitting Procedure Changes", ylabel='A.U.')

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
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "parameter_progression_by_model.png"
    out_pdf = output_dir / "parameter_progression_by_model.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plots to {out_png} and {out_pdf}")

if __name__ == "__main__":
    for base_dir in [Path("./2013a_figures"), Path("./2013b_figures")]:
        output_dir = base_dir / "comparison_plots"
        dfs = load_run_type_dfs(base_dir)
        plot_parameter_progression(dfs, output_dir)