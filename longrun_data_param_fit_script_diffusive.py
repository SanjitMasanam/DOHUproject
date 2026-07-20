#!/usr/bin/env python3

import time
import multiprocessing
import rpy2.robjects as ro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sympy as sp
from matplotlib.lines import Line2D
from tqdm import tqdm
from scipy.special import erfcx
from scipy.optimize import curve_fit
from pde_solver_1lyrEBM_diffusive import Params as PDEParams, solve_model as pde_solve_model, YEAR

# Font used for every label/tick/legend on every plot in this script -- change
# this one value to switch fonts everywhere (falls back to matplotlib's
# default sans font if the named font isn't installed on this machine).
PLOT_FONT_FAMILY = "Verdana"
mpl.rcParams["font.family"] = PLOT_FONT_FAMILY

# --- shared plot-styling sizes (keep identical across all longrun scripts) ---
AXIS_LABEL_FONTSIZE  = 22   # figure-level x/y axis + side labels (was 18)
PANEL_AXIS_FONTSIZE  = 13   # small per-panel axis labels: 3D x/y/z, format_ax (was 9/default)
MODEL_LABEL_FONTSIZE = 15   # bold top-left model tag
EXTRA_TEXT_FONTSIZE  = 12   # param boxes under the model tag (was 8)
# NOTE: plot titles are intentionally left at their current sizes.

# "Purely diffusive" limit of the 1-box + diffusive model: h_ml -> 0 makes
# the mixed-layer heat capacity vanish, which is singular in the PDE solver
# (division by c_ml = rho_cp*h_ml), so approximate the limit with a tiny but
# numerically well-behaved mixed layer instead of exactly zero.
H_ML_PURELY_DIFFUSIVE = 1.0e-6  # m

# Sensitivity sweep of the fitted kappa/h_ml around each model's best fit.
SENSITIVITY_KAPPA_FACTOR_RANGE = (0.2, 5.0)   # multiplicative range around kappa_pde
SENSITIVITY_HML_FACTOR_RANGE = (0.25, 50.0)     # multiplicative range around h_ml_pde
SENSITIVITY_N_1D = 13                         # points in each 1-D spaghetti sweep
SENSITIVITY_N_2D = 9                          # points per axis in the joint grid (N_2D^2 solves/model)
SENSITIVITY_KAPPA_BOUNDS = (1e-7, 1e-3)       # matches the curve_fit bounds below
SENSITIVITY_HML_BOUNDS = (10.0, 2000.0)

# Dedicated h_ml sweep for the ocean-heat-uptake diagnostic: a FIXED, wide,
# log-spaced range (independent of the per-model best-fit h_ml), over which the
# ocean heat uptake H is plotted vs time. The plain diffusive model has no
# efficacy (eps=1), so this is H itself rather than the (eps-1)*H of the eps runs.
SENSITIVITY_UPTAKE_HML_RANGE = (1e-3, 1000.0)  # h_ml sweep range [m], log-spaced
SENSITIVITY_UPTAKE_N = 25                       # number of h_ml curves in that sweep

# Grid-resolution sensitivity sweep: target vertical spacing dz, with kappa
# and h_ml held at their best-fit values (unlike the kappa/h_ml sweeps above,
# this isn't a physical-parameter sweep -- it's a numerical-convergence check
# on the fit itself).
SENSITIVITY_DZ_RANGE = (10.0, 1000.0)          # target dz range [m] for the sweep

# Diagnostic-only solver settings for the sensitivity sweep: looser tolerance
# and fewer saved points than the actual kappa_pde/h_ml_pde fit (n_save=250,
# rtol=1e-8), since these solves only feed spaghetti/heatmap plots and never
# touch the fitted parameters themselves.
SENSITIVITY_N_SAVE = 100
SENSITIVITY_RTOL = 1e-6
SENSITIVITY_ATOL = 1e-8

# Number of solver-saved points for the main kappa_pde/h_ml_pde fit and the
# resulting "Best Fit" curve (both go through make_pde_dT's pde_dT closure).
PDE_FIT_N_SAVE = 250

# Target vertical grid spacing [m] for the main kappa_pde/h_ml_pde fit's PDE
# solves (make_pde_dT below). Nz is derived from z_max/PDE_FIT_DZ_TARGET and
# clamped to [121, 401] points, so shrinking this refines the grid (up to the
# clamp) to probe grid-resolution sensitivity of the fit.
PDE_FIT_DZ_TARGET = 15.0

# Progress/ETA tracking across the whole script (all run_types x results x models).
RUN_TYPES = [3, 2, 1]
RESULT_KINDS = ["unblinded"]
_SCRIPT_START_TIME = time.time()

def _log_spaced_t_eval(t_final_seconds, n_save):
   """Log-spaced solver save grid: t=0, then year 1 to t_final log-spaced.

   solve_model's default np.linspace(0, t_final, n_save) puts most of its
   resolution at late times; over a multi-century t_final that leaves early
   years (1-50) spanned by a single linear interpolation segment, which
   badly understates the fast early rise wherever the result gets viewed on
   a log-t axis (or fit against early-year data). Log-spacing puts the
   saved points where the log-t curve actually needs them.
   """
   t_eval = np.concatenate(([0.0], np.geomspace(YEAR, t_final_seconds, n_save - 1)))
   t_eval[-1] = t_final_seconds  # guard against fp round-off landing outside t_span
   return t_eval


def _sensitivity_solve_dT(kappa, h_ml, F_ref, T_eq, t_final_years, t_eval, dz_target=15.0):
   """PDE surface response for one sensitivity-sweep (kappa, h_ml, dz) point.

   Mirrors the per-point evaluation inside the main curve_fit target
   (make_pde_dT below), but at loosened tolerance and as a module-level
   function rather than a closure, so it can be dispatched to the
   multiprocessing pool -- sweep points only populate diagnostic plots,
   never the fitted kappa_pde/h_ml_pde themselves. dz_target defaults to the
   same spacing as the main fit; the dz sensitivity sweep overrides it
   directly (unclamped, so the full requested range from coarse to fine is
   actually exercised) while the kappa/h_ml sweeps leave it at the default.
   """
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref,
      z_max=z_max, Nz=nz, t_final=t_final_years * YEAR,
   )
   sol = pde_solve_model(
      pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, SENSITIVITY_N_SAVE),
      rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
   )
   return np.interp(t_eval, sol["t"] / YEAR, sol["dT"])


def _sensitivity_solve_H(kappa, h_ml, F_ref, T_eq, t_final_years, t_eval, dz_target=15.0):
   """Diffusive ocean heat uptake H(t) = rho_cp*kappa*dtheta/dz|_0 [W/m^2] for one
   sensitivity-sweep point. Mirrors _sensitivity_solve_dT (loosened tolerance,
   module-level so it can be dispatched to _SENS_POOL) but returns the surface
   heat-uptake flux instead of the temperature, via a 2nd-order one-sided stencil
   (-3*theta0 + 4*theta1 - theta2)/(2*dz), with D = rho_cp*kappa. The plain
   diffusive model has no efficacy, so this is H itself (eps=1); H > 0 when the
   surface is warmer than the water just below."""
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref,
      z_max=z_max, Nz=nz, t_final=t_final_years * YEAR,
   )
   try:
      sol = pde_solve_model(
         pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, SENSITIVITY_N_SAVE),
         rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
      )
      theta, dz = sol["theta"], sol["dz"]
      dtheta_dz0 = (-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dz)
      H = -pde_p.D * dtheta_dz0       # D = rho_cp*kappa; H > 0 = heat leaving surface down
      out = np.interp(t_eval, sol["t"] / YEAR, H)
   except Exception:
      out = np.zeros(np.shape(t_eval), dtype=float)
   return np.where(np.isfinite(out), out, 0.0)


def _pde_grid(kappa, t_final_years, dz_target):
   """Shared z_max/Nz sizing used by make_pde_dT and the budget/OHC diagnostics,
   so every consumer reads off the SAME grid as the fit."""
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   return z_max, nz


# Memoized full PDE solutions, keyed by every input that affects the solve.
# The best-fit solve gets re-read several times after fitting (the reported
# temperature curve, the budget plot); caching collapses those into ONE solve
# with bit-identical results. Bounded so curve_fit's many trial evaluations
# can't grow it without limit.
_PDE_SOL_CACHE = {}
_PDE_SOL_CACHE_MAX = 8


def _cached_pde_solve(kappa, h_ml, F_ref_, T_eq_, t_final_yrs_,
                      dz_target_, n_save, rtol=None, atol=None):
   """One (possibly cached) PDE solve; returns (pde_params, sol). rtol/atol
   None means the solver's tight defaults (the main-fit setting)."""
   key = (float(kappa), float(h_ml), float(F_ref_), float(T_eq_),
          float(t_final_yrs_), float(dz_target_), int(n_save),
          None if rtol is None else float(rtol),
          None if atol is None else float(atol))
   hit = _PDE_SOL_CACHE.get(key)
   if hit is not None:
      return hit
   z_max, nz = _pde_grid(kappa, t_final_yrs_, dz_target_)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq_, F0=F_ref_,
      z_max=z_max, Nz=nz, t_final=t_final_yrs_ * YEAR,
   )
   kw = {} if rtol is None else {"rtol": rtol, "atol": atol}
   sol = pde_solve_model(pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, n_save), **kw)
   if len(_PDE_SOL_CACHE) >= _PDE_SOL_CACHE_MAX:
      _PDE_SOL_CACHE.clear()
   _PDE_SOL_CACHE[key] = (pde_p, sol)
   return pde_p, sol


def plot_surface_budget_bars(model, kappa, h_ml, F_ref, T_eq, t_final_years,
                             png_path, dz_target=PDE_FIT_DZ_TARGET):
   """Stacked-bar decomposition of the surface tendency dT_s/dt for the best-fit
   PDE solution, one bar per saved time step (the eps=1 special case of the
   EBM-epsilon scripts' budget plot). The surface node the solver integrates is

       c_ml * dT_s/dt = F - lambda*T_s - H ,   H = D*(T_s - theta_1)/dz ,

   so dividing by the mixed-layer heat capacity c_ml = rho_cp*h_ml gives three
   additive contributions [K/yr] that sum EXACTLY to dT_s/dt:

       +F/c_ml            constant CO2 forcing (flat in time),
       -lambda*T_s/c_ml   radiative restoring,
       -H/c_ml            diffusive heat loss to the thermocline.

   The top panel plots the |magnitude| of each term as stacked bars on a
   logarithmic y-axis (negative-sign terms are shown by magnitude so they fit the
   log scale), with the black line the |net dT_s/dt| and the dashed line the
   equilibration ratio T_s/T_eq on a right-hand axis. The lower panel repeats the
   same magnitudes as line plots. The surface gradient uses the SAME first-order
   forward stencil (theta_1 - T_s)/dz as the solver's surface node, so the terms
   close on the integrated tendency rather than a re-estimate. Time is drawn on a
   linear axis with each bar spanning the arithmetic midpoints to its neighbours.
   """
   pde_p, sol = _cached_pde_solve(kappa, h_ml, F_ref, T_eq,
                                  t_final_years, dz_target, PDE_FIT_N_SAVE)
   t_yr = sol["t"] / YEAR
   theta, dz = sol["theta"], sol["dz"]
   T_s = theta[0]
   c_ml, lam, D = pde_p.c_ml, pde_p.lam, pde_p.D      # lam = F_ref/T_eq exactly

   # Per-second tendencies -> K/yr for readability; all three sum to dT_s/dt.
   forcing   = np.full_like(T_s, F_ref) / c_ml * YEAR
   H_flux    = D * (T_s - theta[1]) / dz             # W/m^2, >0 when surface warmer
   restoring = (-lam * T_s) / c_ml * YEAR
   uptake    = (-H_flux) / c_ml * YEAR
   net       = forcing + restoring + uptake          # = dT_s/dt [K/yr]

   comps = [
      (r"$F/c_{ml}$",             forcing,   "#9467bd"),
      (r"$-\lambda T_s/c_{ml}$",  restoring, "#1f77b4"),
      (r"$-H/c_{ml}$",            uptake,    "#2ca02c"),
   ]
   # Linear time axis: place bars at their true year positions with widths
   # spanning the arithmetic midpoints to the neighbouring steps so the bar areas
   # tile the axis (samples are log-spaced, so early bars are narrow).
   tv = t_yr
   mid = 0.5 * (tv[:-1] + tv[1:])
   left = np.concatenate(([max(0.0, tv[0] - (mid[0] - tv[0]))], mid))
   right = np.concatenate((mid, [tv[-1] + (tv[-1] - mid[-1])]))
   width = right - left

   # Two stacked panels sharing the time axis (ratio-plot layout): the top panel
   # stacks the |magnitude| of the three terms as bars on a LOG y-axis (negatives
   # shown by magnitude so they fit the log scale); the lower panel repeats the
   # same magnitudes as line plots.
   fig, (ax, ax2) = plt.subplots(
      2, 1, figsize=(13, 8), sharex=True,
      gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
   )
   net_abs = np.abs(net)
   floor = max(1e-3, 0.5 * float(np.min([np.abs(v).min() for _, v, _ in comps] + [net_abs.min()])))
   base = np.full_like(net, floor)
   for label, vals, color in comps:
      mag = np.abs(vals)
      ax.bar(left, mag, bottom=base, width=width, align="edge", color=color,
             label=label, edgecolor="none")
      base = base + mag
   ax.plot(tv, net_abs, color="black", lw=1.4, label=r"$|$net $dT_s/dt|$")
   ax.set_yscale("log")
   ax.set_ylim(floor, base.max() * 1.3)
   ax.set_ylabel(r"$|dT_s/dt$ contribution$|$ (K yr$^{-1}$)", fontsize=14, fontweight="bold")
   ax.set_title(rf"{model}: surface tendency budget "
                rf"($\kappa$={kappa:.1e}, $h_{{ml}}$={h_ml:.0f} m)",
                fontsize=13, fontweight="bold")

   # Overlay the surface warming itself as the equilibration ratio T_s/T_eq on a
   # right-hand axis, so the correspondence between how far the surface has
   # equilibrated and its instantaneous tendency dT_s/dt is visible at a glance.
   ax_r = ax.twinx()
   ratio_line, = ax_r.plot(tv, T_s / T_eq, color="red", lw=2.0, ls="--",
                           label=r"$T_s/T_{eq}$")
   ax_r.set_ylabel(r"Equilibrium Ratio $T_s/T_{eq}$", fontsize=14,
                   fontweight="bold", color="red")
   ax_r.tick_params(axis="y", colors="red")
   ax_r.spines["right"].set_color("red")
   ax_r.set_ylim(0.0, 1.05)

   handles, labels = ax.get_legend_handles_labels()
   handles.append(ratio_line); labels.append(ratio_line.get_label())
   ax.legend(handles, labels, loc="lower right", prop={"weight": "bold", "size": 10})
   ax.grid(True, axis="y", which="both", alpha=0.3)

   # Lower panel: magnitude (|.|) of each term vs time, as line plots.
   for label, vals, color in comps:
      ax2.plot(tv, np.abs(vals), color=color, lw=1.6, label=label)
   ax2.set_ylabel(r"$|$contribution$|$ (K yr$^{-1}$)", fontsize=14, fontweight="bold")
   ax2.grid(True, alpha=0.3)

   ax2.set_xlim(left[0], right[-1])
   ax2.set_xlabel("Time (years)", fontsize=14, fontweight="bold")
   fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
   plt.close(fig)


# Sensitivity-sweep points are independent (kappa, h_ml) solves, so farm them
# out across cores instead of running all ~100/model sequentially. Opened
# once and reused for the whole script; closed at the very end.
_SENS_POOL = multiprocessing.get_context("fork").Pool()

# Load the .Rdata file once and reuse it across every run_type/results
# iteration below -- the source data never changes between iterations.
rdata_file = Path("./data/int_netToa_longrun.Rdata")
ro.r["load"](str(rdata_file))

data = ro.globalenv["int_nettoa_longrun_data"]
models = ['CCSM3', 'CESM1', 'CNRMCM6', 'ECHAM5', 'GISSE2R', 'IPSLCM5A', 'HadGEM2', 'MPIESM11']
expts = list(ro.globalenv["expts"])

tcr = {
    "CCSM3": 1.7,      # no direct CMIP6 equivalent
    "CESM1": 2.0,       # CESM2
    "CNRMCM6": 2.1,     # CNRM-CM6-1
    "ECHAM5": 3.0,     # no direct CMIP6 equivalent
    "GISSE2R": 1.5,     # GISS-E2-1-G (or 1.9 for GISS-E2-1-H)
    "IPSLCM5A": 2.3,    # IPSL-CM6A-LR
    "HadGEM2": 2.5,     # HadGEM3-GC31-LL
    "MPIESM11": 2.0,    # MPI-ESM1-LR
}

# Output directory (created once; per-run_type subdirs are made inside the loop)
outdir = Path("./figures_diffusive")
outdir.mkdir(exist_ok=True)

for run_type in RUN_TYPES:
   for results in RESULT_KINDS:
      dir_list = [
         "replicate_results",
         "50-yr_avg_forcing_results",
         "50-yr_avg_LR_fit_results",
      ]
      current_dir = dir_list[run_type - 1]
      print("==================================")
      print(f"Current Dir: {current_dir}\nType of output: {results}")
      print("==================================")

      extra_text = {
         1: "",
         2: "50yrAvg",
         3: "50yrAvg_LRparamFit",
      }[run_type]
      suffix = f"_{extra_text}" if extra_text else ""

      def sympy_prop_unc(expr, values, uncertainties):
         """Propagate uncertainties through a SymPy expression (first-order, uncorrelated)."""
         variance = 0.0
         for sym, unc in uncertainties.items():
            deriv = sp.diff(expr, sym)
            deriv_val = float(deriv.evalf(subs=values))
            variance += (deriv_val * unc) ** 2
         return float(np.sqrt(variance))


      def format_ax(ax, title="", xlabel="", ylabel="", text="",
                  xscale="linear", yscale="linear",
                  xlim=None, ylim=None,
                  xticks=None, yticks=None,
                  xspacing=True, yspacing=True,
                  legend=True, legend_loc='upper right', grid=True):

         ax.set(xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim)
         if title:  ax.set_title(title, fontweight="bold")
         if xlabel: ax.set_xlabel(xlabel, fontweight="bold", fontsize=PANEL_AXIS_FONTSIZE)
         if ylabel: ax.set_ylabel(ylabel, fontweight="bold", fontsize=PANEL_AXIS_FONTSIZE)

         if text:
            ax.text(0.02, 0.98, text, transform=ax.transAxes, weight='bold',
                     fontsize=MODEL_LABEL_FONTSIZE, va="top", ha="left",
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
         nmodels = len(models)
         if nmodels % 2 != 0:
            raise ValueError(f"Expected an even number of models, got {nmodels}.")

         ncols, nrows = 4, 2

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
            fig.text(0.5, 0.02, xlabel, ha='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

         if ylabel:
            fig.text(0.02, 0.5, ylabel, ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=90.)

         return fig, np.asarray(axs).ravel()


      def ensure_dirs(outdir, current_dir, sections):
         for section in sections:
            (outdir / current_dir / section / "png").mkdir(parents=True, exist_ok=True)

      # Create dataframe to store model parameters
      param_cols = [
         "model",
         "tau",
         "F_ref",
         "lambda",
         "T_eq",
         "a_f",
         "a_s",
         "tau_unc",
         "F_ref_unc",
         "lambda_unc",
         "T_eq_unc",
      ]
      df = pd.DataFrame(columns=param_cols)
      df["model"] = models

      ensure_dirs(outdir, current_dir, ["step1", "validation", "unblinded", "budget"])

      # Prepare combined figures for each experiment
      step1_figs = {}
      step1_axs = {}
      step1_idx = {}
      step1_HT_figs = {}
      step1_HT_axs = {}
      step1_HT_idx = {}
      final_figs = {}
      final_axs = {}
      final_idx = {}
      final_xmax = {}
      nettoa_figs = {}
      nettoa_axs = {}
      nettoa_idx = {}
      ohc_ts_figs = {}
      ohc_ts_axs = {}
      ohc_ts_idx = {}
      sens_kappa_figs = {}
      sens_kappa_axs = {}
      sens_kappa_idx = {}
      sens_hml_figs = {}
      sens_hml_axs = {}
      sens_hml_idx = {}
      sens_hml_uptake_figs = {}
      sens_hml_uptake_axs = {}
      sens_hml_uptake_idx = {}
      sens_dz_figs = {}
      sens_dz_axs = {}
      sens_dz_idx = {}
      sens_t63_figs = {}
      sens_t63_axs = {}
      sens_t63_idx = {}
      sens_rmse_figs = {}
      sens_rmse_axs = {}
      sens_rmse_idx = {}
      # Artists accumulated across the model loop so each sensitivity figure can
      # be put on a single figure-wide color scale with one shared colorbar
      # (populated per subplot below, normalized/colorbarred after all models).
      sens_kappa_lines = {}   # list of (Line2D, kappa_value) per expt
      sens_hml_lines = {}     # list of (Line2D, h_ml_value) per expt
      sens_hml_uptake_lines = {}  # list of (Line2D, h_ml_value) per expt (uptake sweep)
      sens_dz_lines = {}      # list of (Line2D, dz_value) per expt
      sens_t63_meshes = {}    # list of (QuadMesh, grid) per expt
      sens_rmse_meshes = {}   # list of (QuadMesh, grid) per expt

      # Create the shared per-experiment figures (one 8-panel grid per model set)
      for expt in ['4xCO2']:
         step1_figs[expt], step1_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs T$_{2M}$", xlabel="2-meter Air Temperature Anomaly (K)", ylabel=r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)")
         step1_idx[expt] = 0

         final_figs[expt], final_axs[expt] = make_model_grid(models, dpi=120, title=r"4xCO$_{2}$ T$_{2M}$ vs. Time w/ Diffusive Fit", xlabel=r"Time (years)", ylabel=r"Temperature Anomaly (K)", right=0.95, wspace=0.28)
         final_idx[expt] = 0
         final_xmax[expt] = []
         final_figs[expt].text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=-90.)

         nettoa_figs[expt], nettoa_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs. Time", xlabel="Time (years)", ylabel=r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)")
         nettoa_idx[expt] = 0

         step1_HT_figs[expt], step1_HT_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Ocean Heat Uptake H vs T$_{2M}$", xlabel="2-meter Air Temperature Anomaly (K)", ylabel=r"Ocean Heat Uptake H ($W*m^{-2}$)")
         step1_HT_idx[expt] = 0

         ohc_ts_figs[expt], ohc_ts_axs[expt] = make_model_grid(models, title=r"4xCO$_2$ OHU vs. Surface Temp (Normalized)", xlabel=r"$T_s/2\, \mathrm{ECS}$", ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$")
         ohc_ts_idx[expt] = 0

         sens_kappa_figs[expt], sens_kappa_axs[expt] = make_model_grid(models, title=r"Sensitivity to $\kappa$ (h$_{ml}$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_kappa_idx[expt] = 0

         sens_hml_figs[expt], sens_hml_axs[expt] = make_model_grid(models, title=r"Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_hml_idx[expt] = 0

         sens_hml_uptake_figs[expt], sens_hml_uptake_axs[expt] = make_model_grid(models, title=r"Ocean Heat Uptake Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)", xlabel="Time (years)", ylabel=r"Ocean Heat Uptake $H$ (W m$^{-2}$)")
         sens_hml_uptake_idx[expt] = 0

         sens_dz_figs[expt], sens_dz_axs[expt] = make_model_grid(models, title=r"Sensitivity to Grid Spacing dz ($\kappa$, h$_{ml}$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_dz_idx[expt] = 0

         sens_t63_figs[expt], sens_t63_axs[expt] = make_model_grid(models, title=r"Years to Reach 63% of T$_{eq}$ vs. $\kappa$/h$_{ml}$", xlabel=r"$\kappa$ (m$^2$/s)", ylabel=r"h$_{ml}$ (m)")
         sens_t63_idx[expt] = 0

         sens_rmse_figs[expt], sens_rmse_axs[expt] = make_model_grid(models, title=r"Fit RMSE vs. $\kappa$/h$_{ml}$ (Parameter Degeneracy)", xlabel=r"$\kappa$ (m$^2$/s)", ylabel=r"h$_{ml}$ (m)")
         sens_rmse_idx[expt] = 0

         sens_kappa_lines[expt] = []
         sens_hml_lines[expt] = []
         sens_hml_uptake_lines[expt] = []
         sens_dz_lines[expt] = []
         sens_t63_meshes[expt] = []
         sens_rmse_meshes[expt] = []

      # ----- STEP 1: fit F_ref/lambda/T_eq from the 4xCO2 T2M-vs-NETTOA regression -----

      for model in models:
         # Extract model data from the R dataset
         model_data = data.rx2(model)
         t2m_mean = 0.0
         nettoa_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Flatten T2M and NETTOA arrays from R data
            if run_type == 1:
               t2m = np.array(expt_data.rx2("T2M")).ravel()[:150]
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:150]
            else:
               t2m = np.array(expt_data.rx2("T2M")).ravel()
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()

            # Filter non-nan data
            quality_filter = np.isfinite(t2m) & np.isfinite(nettoa)
            if np.sum(quality_filter) / len(quality_filter) != 1:
               print(f"{model} {expt} valid/all years:", np.sum(quality_filter) / len(quality_filter))
            t2m = t2m[quality_filter]
            nettoa = nettoa[quality_filter]

            # Convert variables to anomalies using piControl baseline
            if expt == "piControl":
               t2m_mean = np.mean(t2m)
               nettoa_mean = np.mean(nettoa)
            elif expt == "4xCO2":
               t2m = t2m - t2m_mean
               nettoa = nettoa - nettoa_mean

               # For non-150-year runs, average the data into 50-year bins before fitting
               if run_type != 1:
                  tmp_t2m_list = []
                  tmp_nettoa_list = []
                  for i in range(50, t2m.shape[0], 50):
                     tmp_t2m_list.append(np.mean(t2m[i : i + 50]))
                     tmp_nettoa_list.append(np.mean(nettoa[i : i + 50]))
                  t2m = np.array(tmp_t2m_list)
                  nettoa = np.array(tmp_nettoa_list)

               # Fit a linear relationship between T2M and NETTOA
               [m, b], cov = np.polyfit(t2m, nettoa, 1, cov=True)
               m_unc = np.sqrt(cov[0, 0])
               b_unc = np.sqrt(cov[1, 1])

               # Build the best-fit line for plotting
               xfit = np.linspace(t2m.min(), t2m.max(), 100)
               yfit = m * xfit + b

               # Draw the Step 1 scatter + fit on the shared figure
               ax = step1_axs[expt][step1_idx[expt]]
               ax.scatter(t2m, nettoa, s=8, alpha=0.5, label="Data")
               ax.plot(xfit, yfit, linewidth=2, label=f"Fit: F_ref={b:.3f}, -lambda={m:.3f}")

               # Compute and annotate R^2 and RMSE for the fit on this panel
               y_pred = m * t2m + b
               ss_res = np.sum((nettoa - y_pred) ** 2)
               ss_tot = np.sum((nettoa - np.mean(nettoa)) ** 2)
               r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
               rmse = np.sqrt(np.mean((nettoa - y_pred) ** 2))
               ax.text(
                  0.02,
                  0.92,
                  f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )

               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear")
               step1_idx[expt] += 1

               # Define symbolic expressions for uncertainty propagation
               sp_m, sp_b = sp.symbols("m b")
               F_expr = sp_b
               lambda_expr = -sp_m
               T_eq_expr = sp_b / (-sp_m)

               # Compute uncertainties via propagation
               F_ref_unc = sympy_prop_unc(F_expr, {sp_b: b}, {sp_b: b_unc})
               lambda_unc = sympy_prop_unc(lambda_expr, {sp_m: m}, {sp_m: m_unc})
               T_eq_unc = sympy_prop_unc(
                  T_eq_expr,
                  {sp_m: m, sp_b: b},
                  {sp_m: m_unc, sp_b: b_unc},
               )

               # Record 4xCO2 fit results and uncertainties
               if expt == "4xCO2":
                  df.loc[df["model"] == model, "F_ref"] = b
                  df.loc[df["model"] == model, "lambda"] = -m
                  df.loc[df["model"] == model, "T_eq"] = b / (-m)
                  df.loc[df["model"] == model, "F_ref_unc"] = F_ref_unc
                  df.loc[df["model"] == model, "lambda_unc"] = lambda_unc
                  df.loc[df["model"] == model, "T_eq_unc"] = T_eq_unc

      # Save the combined Step 1 and Net TOA timeseries figures
      for expt in ["4xCO2"]:
         step1_figs[expt].savefig(
            outdir / current_dir / "step1" / "png" / f"{expt}_all_models_T2M_vs_NETTOA{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_figs[expt])

      print("Finished Step 1: saved combined Step 1 figures & params to df")

      # ----- STEP 2: fit the box/diffusive/PDE models and plot final results -----

      for model in models:
         # Extract model data for plotting results
         model_data = data.rx2(model)
         t2m_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Flatten T2M for the chosen results window
            t2m = np.array(expt_data.rx2("T2M")).ravel()
            nettoa = np.array(expt_data.rx2("NETTOA")).ravel()

            # Drop invalid values before plotting
            quality_filter = np.isfinite(t2m)
            if np.sum(quality_filter) / len(quality_filter) != 1:
               print(f"{model} {expt} valid/all years:", np.sum(quality_filter) / len(quality_filter))
            t2m = t2m[quality_filter]

            # Subtract piControl mean for 4xCO2 anomaly
            if expt == "piControl":
               t2m_mean = np.mean(t2m)
               nettoa_mean = np.mean(nettoa)
            elif expt == "4xCO2":
               t2m = t2m - t2m_mean
               nettoa = nettoa - nettoa_mean

               F_ref = df.loc[df["model"] == model, "F_ref"].iloc[0]
               lmbda = df.loc[df["model"] == model, "lambda"].iloc[0]

               T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]
               T_eq_unc = df.loc[df["model"] == model, "T_eq_unc"].iloc[0]

               def fit_func_diff(x, T):
                  return 1-erfcx(np.sqrt(x/T))

               def fit_func_ode(x, tau):
                  return 1-np.exp(-x/tau)

               if results == "validation" and run_type in (1, 2):
                  fit_T = t2m[:151]/T_eq
                  plot_T = t2m[:151]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "validation" and run_type == 3:
                  fit_T = t2m/T_eq
                  plot_T = t2m[300:]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(300, 300 + plot_T.shape[0], 1)
               if results == "unblinded" and run_type in (1, 2):
                  fit_T = t2m[:151]/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "unblinded" and run_type == 3:
                  fit_T = t2m/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)

               # NETTOA sliced to the same window as plot_T/plot_t above, so
               # the observed Net TOA data lines up index-for-index with the
               # PDE fit curves evaluated on plot_t.
               if results == "validation" and run_type in (1, 2):
                  nettoa_plot = nettoa[:151]
               elif results == "validation" and run_type == 3:
                  nettoa_plot = nettoa[300:]
               elif results == "unblinded":
                  nettoa_plot = nettoa

               # Compute the fitted temperature curve & nettoa/OHC
               popt, pcov = curve_fit(fit_func_diff, fit_t, fit_T)
               perr = np.sqrt(np.diag(pcov))

               T = popt[0]
               T_unc = perr[0]

               # Diffusivity implied by the analytic (h_ml=0) fit's timescale
               # T, so it can be reused below as the PDE's kappa in the
               # h_ml->0 numerical limit (see pde_p_h0), keeping that curve
               # on the same diffusivity as "Diffusive Analytical" rather
               # than the joint fit's kappa_pde (which trades off against
               # a nonzero h_ml_pde and need not match D).
               D = T * 31536000 * (lmbda/((10**3)*(4.22*10**3)))**2

               popt, pcov = curve_fit(fit_func_ode, fit_t, fit_T)
               perr = np.sqrt(np.diag(pcov))

               tau = popt[0]
               tau_unc = perr[0]

               # Solve the 1-layer EBM + diffusive thermocline PDE, using
               # F_ref/lambda/T_eq already fit above and treating the
               # thermocline diffusivity kappa AND the mixed-layer depth
               # h_ml (i.e. its heat capacity, c_ml = rho_cp*h_ml) as free
               # parameters.
               t_final_years = max(fit_t.max(), plot_t.max())

               def make_pde_dT(F_ref_, T_eq_, t_final_yrs_, dz_target_=15.0):
                  def pde_dT(t_years, kappa, h_ml):
                     _, sol = _cached_pde_solve(kappa, h_ml, F_ref_, T_eq_,
                                                t_final_yrs_, dz_target_, PDE_FIT_N_SAVE)
                     return np.interp(t_years, sol["t"] / YEAR, sol["dT"])
                  return pde_dT

               pde_dT_func = make_pde_dT(F_ref, T_eq, t_final_years, dz_target_=PDE_FIT_DZ_TARGET)
               popt, pcov = curve_fit(
                  pde_dT_func, fit_t, fit_T * T_eq,
                  p0=[1.0e-4, 100.0], bounds=([1e-7, 1e-3], [1e-3, 300.0]), max_nfev=60,
               )
               perr = np.sqrt(np.diag(pcov))

               kappa_pde, h_ml_pde = popt
               kappa_pde_unc, h_ml_pde_unc = perr
               pde_T = pde_dT_func(plot_t, kappa_pde, h_ml_pde) / T_eq

               # Stacked-bar decomposition of the surface tendency dT_s/dt into
               # F/c_ml, -lambda*T/c_ml and -H/c_ml for the best-fit solve
               # (eps=1 special case of the EBM-epsilon scripts' budget plot).
               plot_surface_budget_bars(
                  model, kappa_pde, h_ml_pde, F_ref, T_eq, t_final_years,
                  outdir / current_dir / "budget" / "png" / f"{expt}_{model}_surface_budget_bars{suffix}.png",
               )

               # Purely diffusive counterpart (h_ml -> 0, infinite ocean): uses
               # D (the diffusivity implied by the analytic h_ml=0 fit above),
               # not kappa_pde, so this curve is directly comparable to
               # "Diffusive Analytical" -- kappa_pde comes from the joint
               # two-parameter fit where it trades off against a nonzero
               # h_ml_pde (equifinality) and so need not equal the diffusivity
               # that actually fits the data in the h_ml->0 limit. The
               # mixed-layer heat capacity is taken to ~0 so the surface
               # responds essentially instantaneously to the flux balance at
               # the top of the thermocline, and the domain is auto-deepened
               # (semi_infinite) so the deep ocean keeps absorbing heat
               # instead of hitting an insulated floor.
               pde_p_h0 = PDEParams(kappa=D, h_ml=H_ML_PURELY_DIFFUSIVE,
                                     dT_eq=T_eq, F0=F_ref, t_final=t_final_years * YEAR,
                                     semi_infinite=True)
               sol_h0 = pde_solve_model(pde_p_h0, n_save=250)
               pde_T_h0 = np.interp(plot_t, sol_h0["t"] / YEAR, sol_h0["dT"]) / T_eq

               # "Just a box" counterpart (kappa -> 0): removes the diffusive
               # thermocline entirely, leaving a bare mixed-layer slab. h_ml
               # is independently re-fit with kappa fixed at 0 (rather than
               # reusing kappa_pde/h_ml_pde), so this is its own fit to the
               # data, not the two-parameter fit merely evaluated at kappa=0.
               def pde_dT_box(t_years, h_ml):
                  return pde_dT_func(t_years, 0.0, h_ml)

               popt, pcov = curve_fit(
                  pde_dT_box, fit_t, fit_T * T_eq,
                  p0=[110.0], bounds=(5.0, 2000.0), max_nfev=60,
               )
               h_ml_box = popt[0]
               pde_T_box = pde_dT_box(plot_t, h_ml_box) / T_eq

               # ----- OHC vs. T_s: AOGCM data vs. 1-Box + Diffusion fit -----
               # Uses only the joint kappa_pde/h_ml_pde PDE fit (pde_T, the
               # curve labeled "1-Box + Diffusion Fit" below) -- no 2-box
               # model or mixed-layer-only reference lines. pde_T is only
               # evaluated over plot_t (not extended to equilibrium), same as
               # the AOGCM data it's compared against.
               if results == 'unblinded':
                  # Prediction reference = rho_cp * z_max * A, the heat capacity of
                  # the fit's ACTUAL grid column (which warms to T_eq at equilibrium),
                  # so normalized_OHC_pred/T_eq -> 1 at equilibrium. The AOGCM data
                  # keeps the real-ocean normalization (1.37e21 kg x 3850 J/kg/K).
                  z_max, _ = _pde_grid(kappa_pde, t_final_years, PDE_FIT_DZ_TARGET)
                  ohc_ref = 5.1e14 * 4.186e6 * z_max          # OHC_eq / T_eq  [J/K]
                  normalized_OHC = (5.1e14 * 31536000 * np.cumsum(nettoa)) / (1.37e21 * 3850)

                  N_pde = F_ref - lmbda * (T_eq * pde_T)
                  normalized_OHC_pred = (5.1e14 * 31536000 * np.cumsum(N_pde)) / ohc_ref

                  cmap = plt.cm.turbo
                  norm = mpl.colors.Normalize(vmin=0, vmax=6000)
                  ax = ohc_ts_axs[expt][ohc_ts_idx[expt]]
                  sc = ax.scatter(t2m/T_eq, normalized_OHC/T_eq, c=np.arange(1, 1+normalized_OHC.shape[0], 1), cmap=cmap, norm=norm)
                  ax.plot(pde_T, normalized_OHC_pred/T_eq, color='green', label='1-Box + Diffusion Fit')
                  ax.axvline(1.0, color="0.55", ls='--', lw=0.8)
                  format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", ylim=(-0.05, 1.2))
                  ohc_ts_idx[expt] += 1

               # ----- Sensitivity sweep: kappa & h_ml -----
               # Sweep the two free PDE parameters around this model's best fit to
               # see how sensitive/well-constrained the fit is. Every sweep point
               # is an independent PDE solve, so they're dispatched to _SENS_POOL
               # (loosened tolerance -- see SENSITIVITY_N_SAVE/RTOL/ATOL) rather
               # than run sequentially through pde_dT_func's tight-tolerance solver.
               kappa_lo = max(SENSITIVITY_KAPPA_BOUNDS[0], kappa_pde * SENSITIVITY_KAPPA_FACTOR_RANGE[0])
               kappa_hi = min(SENSITIVITY_KAPPA_BOUNDS[1], kappa_pde * SENSITIVITY_KAPPA_FACTOR_RANGE[1])
               h_ml_lo = max(SENSITIVITY_HML_BOUNDS[0], h_ml_pde * SENSITIVITY_HML_FACTOR_RANGE[0])
               h_ml_hi = min(SENSITIVITY_HML_BOUNDS[1], h_ml_pde * SENSITIVITY_HML_FACTOR_RANGE[1])

               # 1-D sweeps (other parameter held at its best-fit value) for the
               # spaghetti plots.
               kappa_grid_1d = np.geomspace(kappa_lo, kappa_hi, SENSITIVITY_N_1D)
               h_ml_grid_1d = np.linspace(h_ml_lo, h_ml_hi, SENSITIVITY_N_1D)
               kappa_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(k, h_ml_pde, F_ref, T_eq, t_final_years, plot_t) for k in kappa_grid_1d],
                  )
               ]
               h_ml_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(kappa_pde, h, F_ref, T_eq, t_final_years, plot_t) for h in h_ml_grid_1d],
                  )
               ]

               # Ocean-heat-uptake sensitivity to h_ml: sweep h_ml over a FIXED
               # log-spaced range [1e-3, 1000] m (kappa held at best fit) and record
               # the ocean heat uptake H(t) = rho_cp*kappa*dtheta/dz|_0 [W/m^2] (the
               # plain diffusive model has no efficacy, so this is H itself).
               h_ml_uptake_grid = np.geomspace(
                  SENSITIVITY_UPTAKE_HML_RANGE[0], SENSITIVITY_UPTAKE_HML_RANGE[1],
                  SENSITIVITY_UPTAKE_N,
               )
               h_ml_uptake_curves = _SENS_POOL.starmap(
                  _sensitivity_solve_H,
                  [(kappa_pde, h, F_ref, T_eq, t_final_years, plot_t) for h in h_ml_uptake_grid],
               )

               # Grid-resolution sweep: kappa/h_ml held at their best fit, only
               # the target vertical spacing dz varies (see SENSITIVITY_DZ_RANGE).
               # Unlike the two sweeps above, this isn't probing equifinality in
               # the physical parameters -- it's checking whether the fitted
               # kappa_pde/h_ml_pde curve has actually converged w.r.t. the PDE's
               # numerical grid, or whether coarser dz would have changed it.
               dz_grid_1d = np.geomspace(SENSITIVITY_DZ_RANGE[0], SENSITIVITY_DZ_RANGE[1], SENSITIVITY_N_1D)
               dz_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(kappa_pde, h_ml_pde, F_ref, T_eq, t_final_years, plot_t, dz) for dz in dz_grid_1d],
                  )
               ]

               # Joint 2-D grid, shared by both heatmaps below: one PDE solve per
               # cell (on a common fine time grid), then both metrics (time to 63%
               # of T_eq, and RMSE against the AOGCM fit data) are derived from
               # that single solve by interpolation. Cells are independent, so
               # solved in parallel and reshaped back into the grid afterward.
               kappa_grid_2d = np.geomspace(kappa_lo, kappa_hi, SENSITIVITY_N_2D)
               h_ml_grid_2d = np.linspace(h_ml_lo, h_ml_hi, SENSITIVITY_N_2D)
               target_63 = (1.0 - np.exp(-1.0)) * T_eq
               t_grid_common = np.linspace(1.0, t_final_years, 400)

               grid_2d_args = [
                  (k_val, h_val, F_ref, T_eq, t_final_years, t_grid_common)
                  for h_val in h_ml_grid_2d
                  for k_val in kappa_grid_2d
               ]
               grid_2d_curves = _SENS_POOL.starmap(_sensitivity_solve_dT, grid_2d_args)

               t63_grid = np.full((SENSITIVITY_N_2D, SENSITIVITY_N_2D), np.nan)
               rmse_grid = np.full((SENSITIVITY_N_2D, SENSITIVITY_N_2D), np.nan)
               for idx, dT_common in enumerate(grid_2d_curves):
                  ih, ik = divmod(idx, SENSITIVITY_N_2D)
                  if dT_common[-1] >= target_63:
                     t63_grid[ih, ik] = np.interp(target_63, dT_common, t_grid_common)
                  dT_at_fit = np.interp(fit_t, t_grid_common, dT_common)
                  rmse_grid[ih, ik] = np.sqrt(np.mean((dT_at_fit - fit_T * T_eq) ** 2))

               # --- Panel 1: kappa spaghetti (sequential blue ramp = kappa) ---
               # Curves are recolored to the figure-wide shared norm once every
               # model is drawn (see the shared-colorbar block after the loop).
               ax = sens_kappa_axs[expt][sens_kappa_idx[expt]]
               for k_val, curve in zip(kappa_grid_1d, kappa_sweep_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_kappa_lines[expt].append((line, k_val))
               ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
               ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="lower right")
               ax.text(
                  0.02,
                  0.92,
                  rf"h$_{{ml}}$ = {h_ml_pde:.0f} m (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_kappa_idx[expt] += 1

               # --- Panel 2: h_ml spaghetti (sequential green ramp = h_ml) ---
               ax = sens_hml_axs[expt][sens_hml_idx[expt]]
               for h_val, curve in zip(h_ml_grid_1d, h_ml_sweep_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_hml_lines[expt].append((line, h_val))
               ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
               ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="lower right")
               ax.text(
                  0.02,
                  0.92,
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_hml_idx[expt] += 1

               # --- Panel 2b: ocean-heat-uptake sensitivity to h_ml ---
               # H(t) as h_ml is swept log-uniformly over [1e-3, 1000] m; curves
               # are recolored by h_ml (LogNorm) after all models are drawn. The
               # best-fit curve is one extra solve at kappa_pde/h_ml_pde.
               ax = sens_hml_uptake_axs[expt][sens_hml_uptake_idx[expt]]
               for h_val, curve in zip(h_ml_uptake_grid, h_ml_uptake_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_hml_uptake_lines[expt].append((line, h_val))
               ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
               H_bestfit = _sensitivity_solve_H(kappa_pde, h_ml_pde, F_ref, T_eq, t_final_years, plot_t)
               ax.plot(plot_t, H_bestfit, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="upper right")
               ax.text(
                  0.02,
                  0.92,
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s, h$_{{ml}}^*$ = {h_ml_pde:.0f} m",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_hml_uptake_idx[expt] += 1

               # --- Panel 3: dz spaghetti (sequential purple ramp = dz) ---
               ax = sens_dz_axs[expt][sens_dz_idx[expt]]
               for dz_val, curve in zip(dz_grid_1d, dz_sweep_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_dz_lines[expt].append((line, dz_val))
               ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
               ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="lower right")
               ax.text(
                  0.02,
                  0.92,
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s, h$_{{ml}}$ = {h_ml_pde:.0f} m (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_dz_idx[expt] += 1

               # --- Panel 4: years-to-63%-of-T_eq heatmap over (kappa, h_ml) ---
               # Norm is applied figure-wide after the loop so every model shares
               # one color scale and one colorbar.
               ax = sens_t63_axs[expt][sens_t63_idx[expt]]
               mesh = ax.pcolormesh(kappa_grid_2d, h_ml_grid_2d, t63_grid, cmap="viridis", shading="nearest")
               ax.scatter([kappa_pde], [h_ml_pde], marker="*", s=180, color="red",
                          edgecolor="black", linewidth=0.8, zorder=5, label="Best Fit")
               sens_t63_meshes[expt].append((mesh, t63_grid))
               format_ax(ax, text=f"{model}", xscale="log", yscale="linear", legend=False, grid=False)
               sens_t63_idx[expt] += 1

               # --- Panel 5: RMSE-vs-data heatmap over (kappa, h_ml) ---
               # This is the cost surface curve_fit minimized: a sharp minimum at
               # the star means kappa/h_ml are well-constrained; a shallow ridge
               # through the star means the two parameters trade off (equifinality).
               ax = sens_rmse_axs[expt][sens_rmse_idx[expt]]
               mesh = ax.pcolormesh(kappa_grid_2d, h_ml_grid_2d, rmse_grid, cmap="viridis", shading="nearest")
               ax.scatter([kappa_pde], [h_ml_pde], marker="*", s=180, color="red",
                          edgecolor="black", linewidth=0.8, zorder=5, label="Best Fit")
               sens_rmse_meshes[expt].append((mesh, rmse_grid))
               format_ax(ax, text=f"{model}", xscale="log", yscale="linear", legend=False, grid=False)
               sens_rmse_idx[expt] += 1

               # Draw observed and fitted temperature curves
               ax = final_axs[expt][final_idx[expt]]
               ax.scatter(plot_t, T_eq*plot_T, s=4, color="red")
               ax.plot(plot_t, T_eq*plot_T, color="red", label="2-m Surface Temp.")
               # ax.plot(plot_t, T_eq*fit_func_ode(plot_t, tau), color="green", label="1-Box Analytical")
               # ax.plot(plot_t, T_eq*fit_func_diff(plot_t, T), color="blue", label="Diffusive Analytical")
               ax.plot(plot_t, T_eq*pde_T, color="purple", label="1-Box + Diffusion Fit")
               # ax.plot(plot_t, T_eq*pde_T_h0, color="purple", ls="-.",
               #         label=r"Diffusive Numerical (h$_{ml}\to0$, $\infty$ ocean)")
               # ax.plot(plot_t, T_eq*pde_T_box, color="purple", ls="--",
               #         label=r"1-Box Numerical ($\kappa=0$)")

               # Add the slow-timescale parameter and a reference line at 150 years.
               ax.text(
                  0.02,
                  0.92,
                  rf"$\tau_{{1b}}$ = {tau:.0f} yrs" + "\n" + rf"T$_d$ = {T:.0f} yrs, D$_d$ =  {D:.2e} m^2/s" + "\n" + rf"$\kappa_{{1b+d}}$ = {kappa_pde:.2e} m^2/s, h$_{{ml}}$ = {h_ml_pde:.0f} m",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  weight="bold",
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               ax.axvline(151, color="orange", linestyle=":", linewidth=1, alpha=0.7)

               y_bottom, y_top = 1, 1.25 * T_eq
               ratio_ticks = np.arange(0.2, 1.21, 0.2)
               format_ax(
                  ax,
                  text=f"{model}",
                  xscale="linear",
                  yscale="linear",
                  ylim=(y_bottom, y_top),
                  yticks=ratio_ticks * T_eq,
                  legend_loc='lower right',
               )
               final_xmax[expt].append(np.max(plot_t))

               # Secondary y-axis showing T/T_eq (Equilibrium Ratio) on the same scale;
               # ratio ticks match the left axis's fractions so gridlines line up exactly.
               ax2 = ax.twinx()
               ax2.set_ylim(y_bottom / T_eq, y_top / T_eq)
               ax2.set_yticks(ratio_ticks)
               ax2.tick_params(labelsize=14, width=2, length=8, direction="in")

               final_idx[expt] += 1

               # ----- Net TOA vs. time: AOGCM data vs. 1-Box + Diffusion fit -----
               # Predicted N follows directly from the Step-1 linear feedback
               # relation N = F_ref - lambda*T_s, evaluated on the joint
               # kappa_pde/h_ml_pde fit's own temperature curve (pde_T) rather
               # than the raw AOGCM data.
               nettoa_rollingMu = np.array([
                  np.mean(nettoa_plot[i:i + 10]) for i in range(nettoa_plot.shape[0] - 9)
               ])
               t_rollingMu = plot_t[: nettoa_rollingMu.shape[0]]
               N_pde = F_ref - lmbda * (T_eq * pde_T)

               # TOA imbalance from the same radiative relation but fed the ACTUAL
               # AOGCM T2m timeseries (T_eq*plot_T) instead of the PDE-fit
               # temperature: N = F - lambda*T. T2m is 10-yr rolling-averaged to
               # match the (smoothed) black AOGCM curve, otherwise raw year-to-year
               # noise makes this line unreadable on the log axis. This should hug
               # the observed NETTOA if the fit is good.
               T_aogcm_abs = T_eq * plot_T
               T_aogcm_roll = np.array([
                  np.mean(T_aogcm_abs[i:i + 10]) for i in range(T_aogcm_abs.shape[0] - 9)
               ])
               N_aogcm = F_ref - lmbda * T_aogcm_roll

               ax = nettoa_axs[expt][nettoa_idx[expt]]
               ax.plot(t_rollingMu, nettoa_rollingMu, color="black", label="AOGCM")
               ax.plot(plot_t, N_pde, color="purple", label="1-Box + Diffusion Fit")
               ax.plot(t_rollingMu, N_aogcm, color="darkorange", ls="--",
                       label=r"$F-\lambda T_{AOGCM}$ (10-yr mean)")
               format_ax(ax, text=f"{model}", xscale="linear", yscale="log", legend_loc="lower left")
               nettoa_idx[expt] += 1

               # ----- STEP 1 (H vs T) panel: ocean heat uptake H vs surface -----
               # temperature. Data = the fitted PDE's ocean heat uptake H(t) [W/m^2];
               # curve = F - N - lambda*T from the observed Net TOA over the same
               # plot window. The plain diffusive model has NO efficacy, so there is
               # no 1/(eps-1) factor here (unlike the EBM-epsilon scripts).
               T_surf = T_eq * plot_T
               H_model = _sensitivity_solve_H(kappa_pde, h_ml_pde, F_ref, T_eq,
                                              t_final_years, plot_t)
               curve_HT = F_ref - nettoa_plot - lmbda * T_surf
               order_HT = np.argsort(T_surf)
               ax = step1_HT_axs[expt][step1_HT_idx[expt]]
               ax.scatter(T_surf, H_model, s=8, alpha=0.5, label="H (PDE fit)")
               ax.plot(T_surf[order_HT], curve_HT[order_HT], color="green", lw=2, ls="--",
                       label=r"$F-N-\lambda T$")
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="upper right")
               step1_HT_idx[expt] += 1

               # ----- Progress / ETA -----
               # Printed once per model (this is where the sensitivity sweep's
               # PDE solves happen, so it dominates the runtime); the ETA gets
               # more accurate as more models complete.
               total_model_iters = len(RUN_TYPES) * len(RESULT_KINDS) * len(models)
               completed_model_iters = (
                  RUN_TYPES.index(run_type) * len(RESULT_KINDS) * len(models)
                  + RESULT_KINDS.index(results) * len(models)
                  + (models.index(model) + 1)
               )
               elapsed = time.time() - _SCRIPT_START_TIME
               avg_per_model = elapsed / completed_model_iters
               eta_remaining = avg_per_model * (total_model_iters - completed_model_iters)
               print(
                  f"[Progress] run_type={run_type} results={results} model={model} "
                  f"({completed_model_iters}/{total_model_iters} model-runs done) "
                  f"elapsed={elapsed/60:.1f} min, avg={avg_per_model:.1f} s/model, "
                  f"ETA remaining={eta_remaining/60:.1f} min"
               )

      # Save combined final figures in both linear and log x-scale variants
      for expt in ['4xCO2']:
         for scale in ["linear", "log"]:
            for ax, xmax in zip(final_axs[expt], final_xmax[expt]):
               ax.set_xscale(scale)
               ax.set_xlim(1, xmax + 1)
               if scale == "linear":
                  # Add a highlighted "150" tick at the orange reference line
                  # (drawn at x=151) alongside the usual evenly-spaced ticks.
                  base_ticks = np.linspace(1, xmax + 1, 5)
                  all_ticks = sorted(set(base_ticks.tolist()) | {151.0})
                  ax.set_xticks(all_ticks)
                  ax.set_xticklabels(["150" if abs(tv - 151.0) < 1e-9 else f"{tv:.0f}" for tv in all_ticks])
                  for tick_val, label in zip(all_ticks, ax.get_xticklabels()):
                     if abs(tick_val - 151.0) < 1e-9:
                        label.set_color("orange")
                        label.set_fontsize(10)
               else:
                  ax.xaxis.set_major_locator(mpl.ticker.LogLocator())

            final_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_T2m_vs_t_{results}_{scale}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )

            for ax, xmax in zip(nettoa_axs[expt], final_xmax[expt]):
               ax.set_xscale(scale)
               ax.set_xlim(1, xmax + 1)

            nettoa_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_NETTOA_timeseries_{results}_{scale}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )

         step1_HT_figs[expt].savefig(
            outdir / current_dir / results / "png" / f"{expt}_all_models_H_vs_T2M_{results}{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_HT_figs[expt])

         plt.close(final_figs[expt])
         plt.close(nettoa_figs[expt])

         # --- Shared colorbars for the 5 sensitivity figures ---
         # Put every subplot of a given figure on one common color scale, then
         # draw a single bolded colorbar spanning all panels on the right. The
         # two spaghetti figures are recolored to their figure-wide norm here.
         kappa_vals = np.array([v for _, v in sens_kappa_lines[expt]])
         kappa_norm = mpl.colors.LogNorm(vmin=kappa_vals.min(), vmax=kappa_vals.max())
         for line, v in sens_kappa_lines[expt]:
            line.set_color(plt.cm.Blues(0.25 + 0.65 * kappa_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=kappa_norm, cmap=plt.cm.Blues)
         sm.set_array([])
         cbar = sens_kappa_figs[expt].colorbar(sm, ax=list(sens_kappa_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label(r"$\kappa$ (m$^2$/s)", fontsize=15, fontweight="bold")

         hml_vals = np.array([v for _, v in sens_hml_lines[expt]])
         hml_norm = mpl.colors.Normalize(vmin=hml_vals.min(), vmax=hml_vals.max())
         for line, v in sens_hml_lines[expt]:
            line.set_color(plt.cm.Greens(0.25 + 0.65 * hml_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=hml_norm, cmap=plt.cm.Greens)
         sm.set_array([])
         cbar = sens_hml_figs[expt].colorbar(sm, ax=list(sens_hml_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label(r"h$_{ml}$ (m)", fontsize=15, fontweight="bold")

         # Uptake sweep: h_ml is log-spaced over 6 decades, so use a LogNorm.
         hml_up_vals = np.array([v for _, v in sens_hml_uptake_lines[expt]])
         hml_up_norm = mpl.colors.LogNorm(vmin=hml_up_vals.min(), vmax=hml_up_vals.max())
         for line, v in sens_hml_uptake_lines[expt]:
            line.set_color(plt.cm.Oranges(0.25 + 0.65 * hml_up_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=hml_up_norm, cmap=plt.cm.Oranges)
         sm.set_array([])
         cbar = sens_hml_uptake_figs[expt].colorbar(sm, ax=list(sens_hml_uptake_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label(r"h$_{ml}$ (m)", fontsize=15, fontweight="bold")

         dz_vals = np.array([v for _, v in sens_dz_lines[expt]])
         dz_norm = mpl.colors.LogNorm(vmin=dz_vals.min(), vmax=dz_vals.max())
         for line, v in sens_dz_lines[expt]:
            line.set_color(plt.cm.Purples(0.25 + 0.65 * dz_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=dz_norm, cmap=plt.cm.Purples)
         sm.set_array([])
         cbar = sens_dz_figs[expt].colorbar(sm, ax=list(sens_dz_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label("dz (m)", fontsize=15, fontweight="bold")

         t63_min = np.nanmin([np.nanmin(g) for _, g in sens_t63_meshes[expt]])
         t63_max = np.nanmax([np.nanmax(g) for _, g in sens_t63_meshes[expt]])
         t63_norm = mpl.colors.Normalize(vmin=t63_min, vmax=t63_max)
         for mesh, _ in sens_t63_meshes[expt]:
            mesh.set_norm(t63_norm)
         sm = mpl.cm.ScalarMappable(norm=t63_norm, cmap=plt.get_cmap("viridis"))
         sm.set_array([])
         cbar = sens_t63_figs[expt].colorbar(sm, ax=list(sens_t63_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label("Years to 63% of $T_{eq}$", fontsize=15, fontweight="bold")

         rmse_min = np.nanmin([np.nanmin(g) for _, g in sens_rmse_meshes[expt]])
         rmse_max = np.nanmax([np.nanmax(g) for _, g in sens_rmse_meshes[expt]])
         rmse_norm = mpl.colors.LogNorm(vmin=max(rmse_min, 1e-6), vmax=rmse_max)
         for mesh, _ in sens_rmse_meshes[expt]:
            mesh.set_norm(rmse_norm)
         sm = mpl.cm.ScalarMappable(norm=rmse_norm, cmap=plt.get_cmap("viridis"))
         sm.set_array([])
         cbar = sens_rmse_figs[expt].colorbar(sm, ax=list(sens_rmse_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label("RMSE vs. AOGCM Data (K)", fontsize=15, fontweight="bold")

         # Save the kappa/h_ml/dz spaghetti figures in both linear and log x-scale
         # variants (mirroring the combined final figures above); the heatmaps
         # have no time axis so they're saved once, as-is.
         spaghetti_figs_to_save = [
            (sens_kappa_figs[expt], sens_kappa_axs[expt], "sensitivity_spaghetti_kappa"),
            (sens_hml_figs[expt], sens_hml_axs[expt], "sensitivity_spaghetti_h_ml"),
            (sens_hml_uptake_figs[expt], sens_hml_uptake_axs[expt], "sensitivity_uptake_h_ml"),
            (sens_dz_figs[expt], sens_dz_axs[expt], "sensitivity_spaghetti_dz"),
         ]
         for fig, axs, name in spaghetti_figs_to_save:
            for scale in ["linear", "log"]:
               for ax, xmax in zip(axs, final_xmax[expt]):
                  ax.set_xscale(scale)
                  ax.set_xlim(1, xmax + 1)
                  if scale == "log":
                     ax.xaxis.set_major_locator(mpl.ticker.LogLocator())

               fig.savefig(
                  outdir / current_dir / results / "png" / f"{expt}_all_models_{name}_{results}_{scale}{suffix}.png",
                  dpi=200,
                  bbox_inches="tight",
               )
            plt.close(fig)

         heatmap_figs_to_save = [
            (sens_t63_figs[expt], "sensitivity_heatmap_t63"),
            (sens_rmse_figs[expt], "sensitivity_heatmap_rmse"),
         ]
         for fig, name in heatmap_figs_to_save:
            fig.savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_{name}_{results}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            plt.close(fig)

         if results == 'unblinded':
            cbar = ohc_ts_figs[expt].colorbar(sc, ax=ohc_ts_axs[expt].ravel().tolist(), fraction=0.025, pad=0.025)
            cbar.set_label("Year")

            ohc_ts_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_ohc_ts{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            plt.close(ohc_ts_figs[expt])

      print("Finished final val/result plots")

_SENS_POOL.close()
_SENS_POOL.join()