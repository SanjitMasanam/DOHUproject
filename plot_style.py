#!/usr/bin/env python3
"""Shared matplotlib style and panel-grid constructors for the longrun scripts.

Every `longrun_data_param_fit_script_*.py` used to carry its own copy of these
constants and helpers under a comment reading "keep identical across all longrun
scripts". They were identical only by convention, and had already drifted:
2013a/2013b grew `sharex`/`sharey`, 2013b grew five text/legend knobs, and
compare_run_type_parameter_progression.py hardcoded font sizes where the others
used the constants. Importing this module makes them identical by construction.

Each helper below is the *superset* of the variants it replaces, with defaults
chosen so the majority render unchanged. `format_ax` and `make_model_grid` were
byte-identical across all five diffusive scripts (verified by hash), so those
five are the baseline and the extra keyword arguments are opt-in.

Importing this module applies the project's rcParams as a side effect, which is
why the path layout lives in the separate, stdlib-only figure_paths module.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

# Font used for every label/tick/legend on every plot in these scripts -- change
# this one value to switch fonts everywhere (falls back to matplotlib's default
# sans font if the named font isn't installed on this machine).
PLOT_FONT_FAMILY = "Verdana"
mpl.rcParams["font.family"] = PLOT_FONT_FAMILY

# --- shared plot-styling sizes ---
AXIS_LABEL_FONTSIZE  = 22   # figure-level x/y axis + side labels
PANEL_AXIS_FONTSIZE  = 13   # small per-panel axis labels: 3D x/y/z, format_ax
MODEL_LABEL_FONTSIZE = 15   # bold top-left model tag
EXTRA_TEXT_FONTSIZE  = 12   # param boxes under the model tag
# NOTE: plot titles are intentionally left at their current sizes.

SUPTITLE_FONTSIZE = 20


def format_ax(ax, title="", xlabel="", ylabel="", text="",
              xscale="linear", yscale="linear",
              xlim=None, ylim=None,
              xticks=None, yticks=None,
              xspacing=True, yspacing=True,
              legend=True, legend_loc='upper right', grid=True,
              text_fontsize=None, text_x=0.02, text_ha='left',
              tick_labelsize=14, legend_fontsize=12):
   """Apply the house style to one axis.

   The last five arguments were 2013b-only; their defaults reproduce the other
   scripts' behaviour exactly.
   """
   ax.set(xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)
   if title:  ax.set_title(title, fontweight="bold")
   if xlabel: ax.set_xlabel(xlabel, fontweight="bold", fontsize=PANEL_AXIS_FONTSIZE)
   if ylabel: ax.set_ylabel(ylabel, fontweight="bold", fontsize=PANEL_AXIS_FONTSIZE)

   if text:
      ax.text(text_x, 0.98, text, transform=ax.transAxes, weight='bold',
              fontsize=(text_fontsize if text_fontsize is not None else MODEL_LABEL_FONTSIZE),
              va="top", ha=text_ha,
              bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4))

   if xticks is not None: ax.set_xticks(xticks)
   if yticks is not None: ax.set_yticks(yticks)

   ax.tick_params(labelsize=tick_labelsize, width=2, length=8, direction="in")
   if grid: ax.grid(alpha=0.3)
   if legend: ax.legend(loc=legend_loc, prop={'weight': 'bold', 'size': legend_fontsize})


def _add_grid_labels(fig, title, xlabel, ylabel):
   """Figure-level suptitle and the two big shared axis labels."""
   if title:
      fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
   if xlabel:
      fig.text(0.5, 0.02, xlabel, ha='center',
               fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
   if ylabel:
      fig.text(0.02, 0.5, ylabel, ha='center', va='center',
               fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=90.)


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
   sharex=False,
   sharey=False,
   auto_rows=False,
   hide_unused=False,
):
   """One panel per model on a shared figure, with big figure-level axis labels.

   `auto_rows=False` (the default) keeps the longrun scripts' fixed 4x2 layout
   and their even-model-count check. `auto_rows=True` sizes the grid to however
   many models are present, for compare_run_type_parameter_progression.py, whose
   model list is read back from a CSV and can vary; pair it with
   `hide_unused=True` to blank any trailing empty panels.
   """
   nmodels = len(models)
   if auto_rows:
      nrows = int(np.ceil(nmodels / ncols))
   else:
      if nmodels % 2 != 0:
         raise ValueError(f"Expected an even number of models, got {nmodels}.")
      ncols, nrows = 4, 2

   fig, axs = plt.subplots(
      nrows,
      ncols,
      figsize=(width_per_ax * ncols, height_per_ax * nrows),
      dpi=dpi,
      sharex=sharex,
      sharey=sharey,
      constrained_layout=False,
   )

   fig.subplots_adjust(left=0.05, right=right, bottom=0.075, top=0.95,
                       wspace=wspace, hspace=hspace)
   _add_grid_labels(fig, title, xlabel, ylabel)

   axs = np.asarray(axs).ravel()
   if hide_unused:
      for ax in axs[nmodels:]:
         ax.set_visible(False)
   return fig, axs


def make_model_grid_with_ratio(
   models,
   width_per_ax=6,
   height_per_ax=6,
   ratio_height_frac=0.32,
   dpi=None,
   title=None,
   xlabel=None,
   ylabel=None,
   ratio_ylabel=None,
   ncols=4,
   right=0.98,
   wspace=0.12,
   hspace=0.12,
):
   """Like make_model_grid, but each model panel is a (main, ratio) pair of
   axes stacked vertically and sharing the x-axis -- the ratio axis is
   short and sits directly beneath its main axis. Rows are added as needed for
   any number of models; slots past the last model are left empty."""
   nmodels = len(models)
   nmodel_rows = int(np.ceil(nmodels / ncols))

   fig = plt.figure(
      figsize=(width_per_ax * ncols, height_per_ax * nmodel_rows * (1 + ratio_height_frac)),
      dpi=dpi,
   )

   height_ratios = [1.0, ratio_height_frac] * nmodel_rows
   gs = fig.add_gridspec(
      nmodel_rows * 2, ncols,
      height_ratios=height_ratios,
      left=0.05, right=right, bottom=0.075, top=0.95,
      wspace=wspace, hspace=hspace,
   )

   main_axs = []
   ratio_axs = []
   ratio_ax0 = None                       # all ratio axes share one y-axis
   for i in range(nmodels):
      row_block, col = divmod(i, ncols)
      main_ax = fig.add_subplot(gs[row_block * 2, col])
      ratio_ax = fig.add_subplot(gs[row_block * 2 + 1, col], sharex=main_ax,
                                 sharey=ratio_ax0)
      if ratio_ax0 is None:
         ratio_ax0 = ratio_ax
      plt.setp(main_ax.get_xticklabels(), visible=False)
      if ratio_ylabel and col == 0:
         ratio_ax.set_ylabel(ratio_ylabel, fontweight="bold", fontsize=EXTRA_TEXT_FONTSIZE)
      main_axs.append(main_ax)
      ratio_axs.append(ratio_ax)

   _add_grid_labels(fig, title, xlabel, ylabel)
   return fig, np.asarray(main_axs), np.asarray(ratio_axs)


def make_model_grid_3d(models, title=None, width_per_ax=6, height_per_ax=5, ncols=4, nrows=None):
   """Grid of 3D axes (4x2 for 8 models, mirroring make_model_grid), for the
   (T, H, N) radiative-regression scatter + fit-plane panels. nrows=None sizes
   the grid to the model count and hides the trailing unused axes."""
   nmodels = len(models)
   if nrows is None:
      nrows = int(np.ceil(nmodels / ncols))
   fig, axs = plt.subplots(
      nrows, ncols,
      figsize=(width_per_ax * ncols, height_per_ax * nrows),
      subplot_kw={"projection": "3d"},
   )
   fig.subplots_adjust(left=0.02, right=0.95, bottom=0.03, top=0.92, wspace=0.10, hspace=0.10)
   if title:
      fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
   axs = np.asarray(axs).ravel()
   for ax in axs[nmodels:]:
      ax.set_visible(False)
   return fig, axs


def plot_regression_3d(ax, T, H, N, F, lam, eps, model, cmap="RdBu_r", resid_norm=None, s=12):
   """Scatter the AOGCM (T, H, N) points and overlay the multilinear regression
   plane N = F - lam*T - (eps-1)*H. Points are colored by signed residual
   (obs - plane), so the panel shows both the geometry of the fit and how
   tightly the data hug it. Returns (scatter, residuals) so the caller can put
   every model's panel on one shared symmetric color scale. H is the ocean heat
   uptake -- from the PDE in the diffusive scripts, from the analytic two-mode
   solution in the 2013 ones -- over the same rows the efficacy regression fit.
   """
   T = np.asarray(T, dtype=float); H = np.asarray(H, dtype=float); N = np.asarray(N, dtype=float)
   Nfit = F - lam * T - (eps - 1.0) * H
   resid = N - Nfit
   ss_res = float(np.sum(resid ** 2))
   ss_tot = float(np.sum((N - np.mean(N)) ** 2))
   r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
   rmse = float(np.sqrt(np.mean(resid ** 2)))

   sc = ax.scatter(T, H, N, c=resid, cmap=cmap, norm=resid_norm, s=s, depthshade=False)

   Tg, Hg = np.meshgrid(
      np.linspace(np.min(T), np.max(T), 12),
      np.linspace(np.min(H), np.max(H), 12),
   )
   Ng = F - lam * Tg - (eps - 1.0) * Hg
   ax.plot_surface(Tg, Hg, Ng, color="0.5", alpha=0.25, linewidth=0, antialiased=True)

   ax.set_xlabel("T (K)", fontsize=PANEL_AXIS_FONTSIZE, fontweight="bold")
   ax.set_ylabel(r"H (W m$^{-2}$)", fontsize=PANEL_AXIS_FONTSIZE, fontweight="bold")
   ax.set_zlabel("N (W m$^{-2}$)", fontsize=PANEL_AXIS_FONTSIZE, fontweight="bold")
   ax.tick_params(labelsize=7)
   ax.set_title(rf"{model}: $\epsilon$={eps:.2f}, $R^2$={r2:.3f}, RMSE={rmse:.3f}",
                fontsize=10, fontweight="bold")
   return sc, resid
