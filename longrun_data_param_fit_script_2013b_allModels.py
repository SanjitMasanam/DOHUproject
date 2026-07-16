#!/usr/bin/env python3
"""
Edited Geoffroy et al. (2013b) EBM-epsilon replication script.

Key fixes relative to the pasted script:
  1. The EBM-epsilon radiative regression uses H from the previous
     analytical EBM solution, not gamma * (T_AOGCM - T0_EBM).
  2. The late-time temperature fit uses a_s = exp(intercept), not
     exp(intercept) / epsilon.
  3. Thermal mode inversion first gives C0_prime and gamma_prime, then
     converts to physical C0 and gamma by dividing by epsilon.
  5. The anomaly baseline is kept consistent within each model.
  6. Time indexing for the late fit is explicit: years 30--150 for
     run_type == 1.

Expected input:
  ./data/int_netToa_longrun.Rdata containing:
    - int_nettoa_longrun_data
    - models
    - expts

Outputs:
  ./2013b_figures/<run_dir>/...
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
import rpy2.robjects as ro
import sympy as sp
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# ------------------------- User settings -------------------------

# Font used for every label/tick/legend on every plot in this script -- change
# this one value to switch fonts everywhere (falls back to matplotlib's
# default sans font if the named font isn't installed on this machine).
PLOT_FONT_FAMILY = "Tahoma"
mpl.rcParams["font.family"] = PLOT_FONT_FAMILY

# --- shared plot-styling sizes (keep identical across all longrun scripts) ---
AXIS_LABEL_FONTSIZE  = 22   # figure-level x/y axis + side labels (was 18)
PANEL_AXIS_FONTSIZE  = 13   # small per-panel axis labels: 3D x/y/z, format_ax (was 9/default)
MODEL_LABEL_FONTSIZE = 15   # bold top-left model tag
EXTRA_TEXT_FONTSIZE  = 12   # param boxes under the model tag (was 8)
# NOTE: plot titles are intentionally left at their current sizes.

# Use at least 3 iterations to match your original script, but allow convergence.
MIN_ITERATIONS = 3
MAX_ITERATIONS = 10
CONVERGENCE_RTOL = 1.0e-4

# If True, saves Step 1 and Step 2 diagnostic figures for each iteration.
SAVE_ITERATION_PLOTS = True

# If non-zero, print per-iteration progress/diagnostics from the calibration loop.
VERBOSE = 0

# If your R data are already anomalies, set this to False.
SUBTRACT_PICONTROL_BASELINE = True

# The Geoffroy2013b replicate case should use years 1--150 for the radiative fit.
N_REPLICATE_YEARS = 150

# Late-time fit used to infer tau_s and a_s.  These are 1-indexed years.
REPLICATE_LATE_YEAR_START = 30
REPLICATE_LATE_YEAR_END = 150

# Early-time fit used to infer tau_f.  These are 1-indexed years.
EARLY_YEAR_START = 1
EARLY_YEAR_END = 10

for run_type in [2, 1, 3]:
   for results in ["unblinded", "validation"]:
      dir_list = [
         "geoffroy_replicate_results",
         "50-yr_avg_forcing_results",
         "50-yr_avg_tau_s_LR_fit_results",
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
      run_type_suffix = f"_{extra_text}" if extra_text else ""


      # ------------------------- Data structures -------------------------

      PARAM_COLS = [
         "model",
         "C",
         "C0",
         "gamma",
         "C0_prime",
         "gamma_prime",
         "tau_f",
         "tau_s",
         "F_ref",
         "lambda",
         "T_eq",
         "a_f",
         "a_s",
         "epsilon",
         "C_unc",
         "C0_unc",
         "gamma_unc",
         "C0_prime_unc",
         "gamma_prime_unc",
         "tau_f_unc",
         "tau_s_unc",
         "F_ref_unc",
         "lambda_unc",
         "T_eq_unc",
         "a_f_unc",
         "a_s_unc",
         "epsilon_unc",
      ]


      @dataclass
      class SeriesBundle:
         t_full: np.ndarray
         T_full: np.ndarray
         N_full: np.ndarray
         t_late: np.ndarray
         T_late: np.ndarray
         N_late: np.ndarray
         t_early: np.ndarray
         T_early: np.ndarray


      # ------------------------- Helper functions -------------------------


      def sympy_prop_unc(expr, values, uncertainties):
         """First-order independent-variable uncertainty propagation."""
         variance = 0.0
         for sym, unc in uncertainties.items():
            if unc is None or not np.isfinite(float(unc)):
                  continue
            deriv = sp.diff(expr, sym)
            deriv_val = float(deriv.evalf(subs=values))
            variance += (deriv_val * float(unc)) ** 2
         return float(np.sqrt(max(variance, 0.0)))


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
         sharex=False,
         sharey=False,
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
            sharex=sharex,
            sharey=sharey,
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
            (outdir / current_dir / section / "pdf").mkdir(parents=True, exist_ok=True)
         (outdir / current_dir / "tables").mkdir(parents=True, exist_ok=True)


      def make_model_grid_3d(models, title=None, width_per_ax=6, height_per_ax=5, ncols=4, nrows=2):
         """4x2 grid of 3D axes (mirrors make_model_grid's layout), for the
         (T, H, N) radiative-regression scatter + fit-plane panels."""
         fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(width_per_ax * ncols, height_per_ax * nrows),
            subplot_kw={"projection": "3d"},
         )
         fig.subplots_adjust(left=0.02, right=0.95, bottom=0.03, top=0.92, wspace=0.10, hspace=0.10)
         if title:
            fig.suptitle(title, fontsize=20, fontweight="bold")
         return fig, np.asarray(axs).ravel()


      def plot_regression_3d(ax, T, H, N, F, lam, eps, model, cmap="RdBu_r", resid_norm=None):
         """Scatter the AOGCM (T, H, N) points and overlay the multilinear
         regression plane N = F - lam*T - (eps-1)*H. Points are colored by the
         signed residual (obs - plane) so the panel shows both the geometry of
         the fit and how tightly the data hug it. Returns (scatter, residuals)
         so the caller can apply one shared symmetric color scale across models.
         """
         Nfit = F - lam * T - (eps - 1.0) * H
         resid = N - Nfit
         ss_res = float(np.sum(resid ** 2))
         ss_tot = float(np.sum((N - np.mean(N)) ** 2))
         r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
         rmse = float(np.sqrt(np.mean(resid ** 2)))

         sc = ax.scatter(T, H, N, c=resid, cmap=cmap, norm=resid_norm, s=10, depthshade=False)

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


      def as_1d_float(x):
         return np.asarray(x, dtype=float).ravel()


      def finite_pair_mask(*arrays):
         mask = np.ones_like(arrays[0], dtype=bool)
         for arr in arrays:
            mask &= np.isfinite(arr)
         return mask


      def covariance_from_lstsq(X, y, coeffs):
         """OLS coefficient covariance estimate sigma^2 (X'X)^-1."""
         n, p = X.shape
         resid = y - X @ coeffs
         dof = max(n - p, 1)
         sigma2 = float(np.sum(resid**2) / dof)
         return sigma2 * np.linalg.pinv(X.T @ X)


      def fit_gregory_initial(T_obs, N_obs):
         """Initial EBM-1 / Gregory fit: N = F - lambda T."""
         good = finite_pair_mask(T_obs, N_obs)
         T_fit = T_obs[good]
         N_fit = N_obs[good]

         (m, b), cov = np.polyfit(T_fit, N_fit, 1, cov=True)
         F_ref = float(b)
         lmbda = float(-m)
         epsilon = 1.0
         T_eq = F_ref / lmbda

         F_ref_unc = float(np.sqrt(max(cov[1, 1], 0.0)))
         lambda_unc = float(np.sqrt(max(cov[0, 0], 0.0)))

         sp_m, sp_b = sp.symbols("m b")
         T_eq_unc = sympy_prop_unc(
            sp_b / (-sp_m),
            {sp_m: m, sp_b: b},
            {sp_m: lambda_unc, sp_b: F_ref_unc},
         )

         return {
            "F_ref": F_ref,
            "lambda": lmbda,
            "epsilon": epsilon,
            "T_eq": T_eq,
            "F_ref_unc": F_ref_unc,
            "lambda_unc": lambda_unc,
            "epsilon_unc": 0.0,
            "T_eq_unc": T_eq_unc,
         }


      def T_model(t, T_eq, a_f, a_s, tau_f, tau_s):
         """Step-forcing temperature response."""
         t = np.asarray(t, dtype=float)
         return T_eq * (1.0 - a_f * np.exp(-t / tau_f) - a_s * np.exp(-t / tau_s))


      def dTdt_model(t, T_eq, a_f, a_s, tau_f, tau_s):
         """Time derivative of the step-forcing temperature response."""
         t = np.asarray(t, dtype=float)
         return T_eq * (
            (a_f / tau_f) * np.exp(-t / tau_f)
            + (a_s / tau_s) * np.exp(-t / tau_s)
         )


      def H_physical_from_previous_solution(t, F_ref, lmbda, C, epsilon, T_eq, a_f, a_s, tau_f, tau_s):
         """
         Physical deep-ocean heat uptake H from the previous EBM solution.

         In the primed EBM-1-equivalent system,
            C dT/dt = F - lambda T - gamma_prime (T - T0).
         Thus gamma_prime(T - T0) = F - lambda T - C dT/dt = epsilon H.
         Eq. (3) in Geoffroy2013b uses physical H, so divide by epsilon.
         """
         Tm = T_model(t, T_eq, a_f, a_s, tau_f, tau_s)
         dTm = dTdt_model(t, T_eq, a_f, a_s, tau_f, tau_s)
         H_prime = F_ref - lmbda * Tm - C * dTm
         return H_prime / epsilon


      def fit_radiative_epsilon(T_obs, N_obs, H_prev):
         """
         Iterative EBM-epsilon radiative fit:
            N = F - lambda*T - (epsilon - 1)*H_prev.
         """
         good = finite_pair_mask(T_obs, N_obs, H_prev)
         T_fit = T_obs[good]
         N_fit = N_obs[good]
         H_fit = H_prev[good]

         X = np.column_stack([np.ones_like(T_fit), -T_fit, -H_fit])
         coeffs, *_ = np.linalg.lstsq(X, N_fit, rcond=None)
         pcov = covariance_from_lstsq(X, N_fit, coeffs)

         F_ref = float(coeffs[0])
         lmbda = float(coeffs[1])
         eps_minus_1 = float(coeffs[2])
         epsilon = 1.0 + eps_minus_1
         T_eq = F_ref / lmbda

         F_ref_unc = float(np.sqrt(max(pcov[0, 0], 0.0)))
         lambda_unc = float(np.sqrt(max(pcov[1, 1], 0.0)))
         epsilon_unc = float(np.sqrt(max(pcov[2, 2], 0.0)))

         sp_F, sp_lmbda = sp.symbols("F lambda")
         T_eq_unc = sympy_prop_unc(
            sp_F / sp_lmbda,
            {sp_F: F_ref, sp_lmbda: lmbda},
            {sp_F: F_ref_unc, sp_lmbda: lambda_unc},
         )

         return {
            "F_ref": F_ref,
            "lambda": lmbda,
            "epsilon": epsilon,
            "T_eq": T_eq,
            "F_ref_unc": F_ref_unc,
            "lambda_unc": lambda_unc,
            "epsilon_unc": epsilon_unc,
            "T_eq_unc": T_eq_unc,
         }


      def fit_slow_mode(t_late, T_late, T_eq):
         """
         Fit log(T_eq - T) - log(T_eq) = log(a_s) - t/tau_s.

         Important: the intercept is log(a_s), not log(epsilon*a_s).
         """
         t_late = np.asarray(t_late, dtype=float)
         T_late = np.asarray(T_late, dtype=float)
         good = np.isfinite(t_late) & np.isfinite(T_late) & np.isfinite(T_eq) & ((T_eq - T_late) > 0) & (T_eq > 0)

         if np.sum(good) < 3:
            raise RuntimeError("Not enough valid late-time points for tau_s/a_s fit.")

         x = t_late[good]
         y = np.log(T_eq - T_late[good]) - np.log(T_eq)

         (m, b), cov = np.polyfit(x, y, 1, cov=True)
         tau_s = float(-1.0 / m)
         a_s = float(np.exp(b))
         a_f = float(1.0 - a_s)

         m_unc = float(np.sqrt(max(cov[0, 0], 0.0)))
         b_unc = float(np.sqrt(max(cov[1, 1], 0.0)))

         sp_m, sp_b = sp.symbols("m b")
         tau_s_unc = sympy_prop_unc(-1 / sp_m, {sp_m: m}, {sp_m: m_unc})
         a_s_unc = sympy_prop_unc(sp.exp(sp_b), {sp_b: b}, {sp_b: b_unc})
         a_f_unc = a_s_unc

         return {
            "tau_s": tau_s,
            "a_s": a_s,
            "a_f": a_f,
            "tau_s_unc": tau_s_unc,
            "a_s_unc": a_s_unc,
            "a_f_unc": a_f_unc,
            "m": float(m),
            "b": float(b),
            "m_unc": m_unc,
            "b_unc": b_unc,
            "x": x,
            "y": y,
            "yfit": m * x + b,
         }


      # Paper-faithful tau_f estimate:
      # compute Eq. (18) year-by-year over years 1--10, then average.
      def fit_fast_mode(t_early, T_early, T_eq, a_f, a_s, tau_s):
         """
         Geoffroy2013a Eq. (18) tau_f estimate.

         The paper does not fit tau_f by nonlinear least squares.
         It computes tau_f(t) from Eq. (18) for each of the first
         10 years and then averages those values.
         """
         t_early = np.asarray(t_early, dtype=float)
         T_early = np.asarray(T_early, dtype=float)

         z = 1.0 - T_early / T_eq - a_s * np.exp(-t_early / tau_s)

         good = (
            np.isfinite(t_early)
            & np.isfinite(T_early)
            & np.isfinite(z)
            & (z > 0.0)
            & np.isfinite(T_eq)
            & (T_eq > 0.0)
            & np.isfinite(a_f)
            & (a_f > 0.0)
            & np.isfinite(tau_s)
            & (tau_s > 0.0)
         )

         if np.sum(good) < 2:
            return np.nan, np.nan

         denom = np.log(a_f) - np.log(z[good])
         tau_vals = t_early[good] / denom

         tau_vals = tau_vals[np.isfinite(tau_vals) & (tau_vals > 0.0)]

         if len(tau_vals) == 0:
            return np.nan, np.nan

         tau_f = float(np.mean(tau_vals))

         # The paper does not define tau_f uncertainty.
         # This is just the scatter of the 1-10 yr implied values.
         tau_f_unc = float(np.std(tau_vals, ddof=1)) if len(tau_vals) > 1 else np.nan

         return tau_f, tau_f_unc



      def thermal_params_from_modes(lmbda, a_f, a_s, tau_f, tau_s, epsilon):
         """
         Convert mode parameters to EBM-epsilon physical parameters.

         The mode inversion gives C0_prime and gamma_prime.  Geoffroy2013b uses
            C0_prime = epsilon*C0 and gamma_prime = epsilon*gamma,
         so the physical C0 and gamma are obtained by dividing by epsilon.
         """
         C = lmbda / (a_f / tau_f + a_s / tau_s)
         C0_prime = lmbda * (a_f * tau_f + a_s * tau_s) - C
         gamma_prime = C0_prime / (tau_f * a_s + tau_s * a_f)

         C0 = C0_prime / epsilon
         gamma = gamma_prime / epsilon

         return {
            "C": float(C),
            "C0": float(C0),
            "gamma": float(gamma),
            "C0_prime": float(C0_prime),
            "gamma_prime": float(gamma_prime),
         }


      def thermal_uncertainties(lmbda, lambda_unc, a_s, a_s_unc, tau_s, tau_s_unc, tau_f, tau_f_unc, epsilon, epsilon_unc):
         """First-order uncertainties for C, C0, gamma and primed versions."""
         sp_lambda, sp_as, sp_taus, sp_tauf, sp_eps = sp.symbols("lambda a_s tau_s tau_f epsilon")
         sp_af = 1 - sp_as

         C_expr = sp_lambda / ((sp_af / sp_tauf) + (sp_as / sp_taus))
         C0p_expr = sp_lambda * (sp_tauf * sp_af + sp_taus * sp_as) - C_expr
         gammap_expr = C0p_expr / (sp_tauf * sp_as + sp_taus * sp_af)
         C0_expr = C0p_expr / sp_eps
         gamma_expr = gammap_expr / sp_eps

         values = {
            sp_lambda: lmbda,
            sp_as: a_s,
            sp_taus: tau_s,
            sp_tauf: tau_f,
            sp_eps: epsilon,
         }
         uncs = {
            sp_lambda: lambda_unc,
            sp_as: a_s_unc,
            sp_taus: tau_s_unc,
            sp_tauf: tau_f_unc,
            sp_eps: epsilon_unc,
         }

         return {
            "C_unc": sympy_prop_unc(C_expr, values, uncs),
            "C0_prime_unc": sympy_prop_unc(C0p_expr, values, uncs),
            "gamma_prime_unc": sympy_prop_unc(gammap_expr, values, uncs),
            "C0_unc": sympy_prop_unc(C0_expr, values, uncs),
            "gamma_unc": sympy_prop_unc(gamma_expr, values, uncs),
         }


      def get_r_array(model_data, expt, var):
         return as_1d_float(model_data.rx2(expt).rx2(var))


      def select_years(arr, start_year, end_year=None):
         """Select 1-indexed inclusive years from an array."""
         if end_year is None:
            return arr[start_year - 1 :]
         return arr[start_year - 1 : end_year]


      def make_series_bundle(model_data, expts):
         """Read piControl and 4xCO2 series and return consistently baselined slices."""
         if "piControl" not in expts or "4xCO2" not in expts:
            raise RuntimeError(f"Expected expts to include 'piControl' and '4xCO2'; got {expts}")

         T_ctrl_raw = get_r_array(model_data, "piControl", "T2M")
         N_ctrl_raw = get_r_array(model_data, "piControl", "NETTOA")
         T_4x_raw = get_r_array(model_data, "4xCO2", "T2M")
         N_4x_raw = get_r_array(model_data, "4xCO2", "NETTOA")

         if run_type == 1:
            full_end = N_REPLICATE_YEARS
            T_base_arr = select_years(T_ctrl_raw, 1, full_end)
            N_base_arr = select_years(N_ctrl_raw, 1, full_end)
            T_full_raw = select_years(T_4x_raw, 1, full_end)
            N_full_raw = select_years(N_4x_raw, 1, full_end)
            t_full = np.arange(1, full_end + 1, dtype=float)

            T_late_raw = select_years(T_4x_raw, REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END)
            N_late_raw = select_years(N_4x_raw, REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END)
            t_late = np.arange(REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END + 1, dtype=float)
         else:
            # For non-replication cases, keep the original broad behavior but still use
            # a consistent baseline.
            T_base_arr = T_ctrl_raw
            N_base_arr = N_ctrl_raw
            T_full_raw = T_4x_raw
            N_full_raw = N_4x_raw
            t_full = np.arange(1, len(T_full_raw) + 1, dtype=float)

            if run_type == 3:
                  start = 50
                  T_late_raw = select_years(T_4x_raw, start, None)
                  N_late_raw = select_years(N_4x_raw, start, None)
                  t_late = np.arange(start, start + len(T_late_raw), dtype=float)
            else:
                  # run_type == 2: matches 2013a_allModels.py's Step 2, which
                  # fits tau_s/a_s over years 30-150 for run_type in (1, 2)
                  # even though T_eq (t_full/T_full/N_full above) is fit to
                  # the entire run for run_type 2.
                  T_late_raw = select_years(T_4x_raw, REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END)
                  N_late_raw = select_years(N_4x_raw, REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END)
                  t_late = np.arange(REPLICATE_LATE_YEAR_START, REPLICATE_LATE_YEAR_END + 1, dtype=float)

         T_early_raw = select_years(T_4x_raw, EARLY_YEAR_START, EARLY_YEAR_END)
         t_early = np.arange(EARLY_YEAR_START, EARLY_YEAR_END + 1, dtype=float)

         if SUBTRACT_PICONTROL_BASELINE:
            T_base = float(np.nanmean(T_base_arr))
            N_base = float(np.nanmean(N_base_arr))
         else:
            T_base = 0.0
            N_base = 0.0

         T_full = T_full_raw - T_base
         N_full = N_full_raw - N_base
         T_late = T_late_raw - T_base
         N_late = N_late_raw - N_base
         T_early = T_early_raw - T_base

         # Apply finite masks while preserving matching time coordinates.
         m_full = finite_pair_mask(t_full, T_full, N_full)
         m_late = finite_pair_mask(t_late, T_late, N_late)
         m_early = finite_pair_mask(t_early, T_early)

         return SeriesBundle(
            t_full=t_full[m_full],
            T_full=T_full[m_full],
            N_full=N_full[m_full],
            t_late=t_late[m_late],
            T_late=T_late[m_late],
            N_late=N_late[m_late],
            t_early=t_early[m_early],
            T_early=T_early[m_early],
         )


      def numeric_params_for_convergence(df):
         cols = ["F_ref", "lambda", "epsilon", "T_eq", "tau_f", "tau_s", "C", "C0", "gamma"]
         return df[cols].apply(pd.to_numeric, errors="coerce")


      def update_row(df, model, values):
         for key, val in values.items():
            if key in df.columns:
                  df.loc[df["model"] == model, key] = val


      def plot_radiative_fit(ax, bundle, fit_params, H_prev, model, iteration):
         T = bundle.T_full
         N = bundle.N_full
         t = bundle.t_full
         F_ref = fit_params["F_ref"]
         lmbda = fit_params["lambda"]
         epsilon = fit_params["epsilon"]

         ax.scatter(T, N, s=8, alpha=0.5, label="AOGCM")

         order = np.argsort(T)
         if H_prev is None:
            yfit = F_ref - lmbda * T[order]
            label = f"Gregory: F={F_ref:.3g}, λ={lmbda:.3g}"
         else:
            yfit_all = F_ref - lmbda * T - (epsilon - 1.0) * H_prev
            yfit = yfit_all[order]
            label = f"EBM-ε: F={F_ref:.3g}, λ={lmbda:.3g}, ε={epsilon:.3g}"

         ax.plot(T[order], yfit, linewidth=2, label=label)
         format_ax(ax, text=model, xscale="linear", yscale="linear")


      def plot_radiative_fit_H(ax, bundle, fit_params, H_prev, model):
         """Net TOA vs. ocean heat uptake H, with the pure-efficacy term
         N = -(eps-1)*H and the full regression evaluated at each point's own
         T, N = F - lambda*T - (eps-1)*H. Points are plotted in series order (H
         is not sorted). No H_prev exists before the first EBM-epsilon iteration
         (Gregory-only), so the panel is left blank then.
         """
         if H_prev is None:
            format_ax(ax, text=model, xscale="linear", yscale="linear", legend=False)
            return

         T = bundle.T_full
         N = bundle.N_full
         F_ref = fit_params["F_ref"]
         lmbda = fit_params["lambda"]
         epsilon = fit_params["epsilon"]

         ax.scatter(H_prev, N, s=8, alpha=0.5, label="AOGCM")

         one_one = np.array([H_prev.min(), H_prev.max()])
         ax.plot(one_one, one_one, color="0.5", lw=1.0, ls="-.", label="1:1")
         ax.plot(H_prev, -(epsilon - 1.0) * H_prev, color="0.4", lw=2, ls=":",
                 label=rf"$N=-(\epsilon-1)H$ ($\epsilon$={epsilon:.2f})")
         ax.plot(H_prev, F_ref - lmbda * T - (epsilon - 1.0) * H_prev,
                 color="green", lw=2, ls="--",
                 label=r"$N=F-\lambda T-(\epsilon-1)H$")
         format_ax(ax, text=model, xscale="linear", yscale="linear")


      def plot_slow_fit(ax, slow, model, iteration):
         ax.scatter(slow["x"], slow["y"], s=8, alpha=0.5, label="AOGCM")
         ax.plot(slow["x"], slow["yfit"], linewidth=2, label=f"τs={slow['tau_s']:.3g}, as={slow['a_s']:.3g}")
         format_ax(ax, text=model, xscale="linear", yscale="linear", legend_loc='lower left')


      # ------------------------- Load R data -------------------------

      rdata_file = Path("./data/int_netToa_longrun.Rdata")
      ro.r["load"](str(rdata_file))

      data = ro.globalenv["int_nettoa_longrun_data"]
      models = list(ro.globalenv["models"])
      expts = list(ro.globalenv["expts"])

      # Convert R strings to plain Python strings if needed.
      models = ['CCSM3', 'CESM1', 'CNRMCM6', 'ECHAM5', 'GISSE2R', 'IPSLCM5A', 'HadGEM2', 'MPIESM11']
      expts = [str(x) for x in expts]

      df = pd.DataFrame(columns=PARAM_COLS)
      df["model"] = models

      outdir = Path("./figures_2013b")
      outdir.mkdir(exist_ok=True)
      ensure_dirs(outdir, current_dir, ["step1", "step2", "validation", "unblinded", "budget"])

      # Cache all model series so every step uses identical baselines and slices.
      series_by_model: Dict[str, SeriesBundle] = {}
      for model in models:
         series_by_model[model] = make_series_bundle(data.rx2(model), expts)


      # ------------------------- Iterative calibration -------------------------

      converged = False
      last_iteration = 0

      for iteration in range(MAX_ITERATIONS):
         if VERBOSE:
            print(f"\n================ ITERATION {iteration + 1} ================")
         old_params = numeric_params_for_convergence(df).copy()

         # Create diagnostic figures for this iteration.
         step1_fig, step1_axs = make_model_grid(
            models,
            title=rf"4xCO$_2$ Net TOA vs T$_{{2M}}$ (radiative fit, iter {iteration + 1})",
            xlabel="2-meter Air Temperature Anomaly (K)",
            ylabel=r"Net TOA Radiative Flux Anomaly ($W\,m^{-2}$)",
         )
         step1_NH_fig, step1_NH_axs = make_model_grid(
            models,
            title=rf"4xCO$_2$ Net TOA vs Ocean Heat Uptake H (radiative fit, iter {iteration + 1})",
            xlabel=r"Ocean Heat Uptake H ($W\,m^{-2}$)",
            ylabel=r"Net TOA Radiative Flux Anomaly ($W\,m^{-2}$)",
         )
         step2_fig, step2_axs = make_model_grid(
            models,
            title=rf"4xCO$_2$ log(T$_{{eq}}$-T) - log(T$_{{eq}}$) vs. Time (slow mode, iter {iteration + 1})",
            xlabel="Time (years)",
            ylabel=r"log($T_{eq}$-T)-log($T_{eq}$)",
         )

         # ----------------- STEP 1: F, lambda, epsilon -----------------
         for imodel, model in enumerate(models):
            bundle = series_by_model[model]

            if iteration == 0:
                  fit = fit_gregory_initial(bundle.T_full, bundle.N_full)
                  H_prev = None
            else:
                  prev = df.loc[df["model"] == model].iloc[0]
                  H_prev = H_physical_from_previous_solution(
                     bundle.t_full,
                     F_ref=float(prev["F_ref"]),
                     lmbda=float(prev["lambda"]),
                     C=float(prev["C"]),
                     epsilon=float(prev["epsilon"]),
                     T_eq=float(prev["T_eq"]),
                     a_f=float(prev["a_f"]),
                     a_s=float(prev["a_s"]),
                     tau_f=float(prev["tau_f"]),
                     tau_s=float(prev["tau_s"]),
                  )
                  fit = fit_radiative_epsilon(bundle.T_full, bundle.N_full, H_prev)

            update_row(df, model, fit)
            plot_radiative_fit(step1_axs[imodel], bundle, fit, H_prev, model, iteration)
            plot_radiative_fit_H(step1_NH_axs[imodel], bundle, fit, H_prev, model)

         if SAVE_ITERATION_PLOTS:
            suffix = f"iter{iteration + 1:02d}"
            step1_fig.savefig(
                  outdir / current_dir / "step1" / "png" / f"4xCO2_all_models_T2M_vs_NETTOA_{suffix}.png",
                  dpi=200,
                  bbox_inches="tight",
            )
            step1_fig.savefig(
                  outdir / current_dir / "step1" / "pdf" / f"4xCO2_all_models_T2M_vs_NETTOA_{suffix}.pdf",
                  bbox_inches="tight",
            )
            step1_NH_fig.savefig(
                  outdir / current_dir / "step1" / "png" / f"4xCO2_all_models_H_vs_NETTOA_{suffix}.png",
                  dpi=200,
                  bbox_inches="tight",
            )
            step1_NH_fig.savefig(
                  outdir / current_dir / "step1" / "pdf" / f"4xCO2_all_models_H_vs_NETTOA_{suffix}.pdf",
                  bbox_inches="tight",
            )
         plt.close(step1_fig)
         plt.close(step1_NH_fig)

         if VERBOSE:
            print("Finished Step 1: F_ref, lambda, epsilon, T_eq updated")

         # ----------------- STEP 2: tau_s, a_s, tau_f, C, C0, gamma -----------------
         for imodel, model in enumerate(models):
            bundle = series_by_model[model]
            row = df.loc[df["model"] == model].iloc[0]

            F_ref = float(row["F_ref"])
            lmbda = float(row["lambda"])
            lambda_unc = float(row["lambda_unc"])
            T_eq = float(row["T_eq"])
            epsilon = float(row["epsilon"])
            epsilon_unc = float(row["epsilon_unc"])

            slow = fit_slow_mode(bundle.t_late, bundle.T_late, T_eq)
            tau_s = slow["tau_s"]
            a_s = slow["a_s"]
            a_f = slow["a_f"]
            tau_s_unc = slow["tau_s_unc"]
            a_s_unc = slow["a_s_unc"]
            a_f_unc = slow["a_f_unc"]

            if a_f <= 0 or a_s <= 0:
                  if VERBOSE:
                     print(f"WARNING: {model} has non-positive mode amplitude(s): a_f={a_f:.4g}, a_s={a_s:.4g}")

            tau_f, tau_f_unc = fit_fast_mode(bundle.t_early, bundle.T_early, T_eq, a_f, a_s, tau_s)
            if not np.isfinite(tau_f):
                  if VERBOSE:
                     print(f"WARNING: {model} tau_f fit failed; using 4 years as fallback.")
                  tau_f = 4.0
                  tau_f_unc = np.nan

            thermal = thermal_params_from_modes(lmbda, a_f, a_s, tau_f, tau_s, epsilon)
            if thermal["C"] <= 0 or thermal["C0"] <= 0 or thermal["gamma"] <= 0:
                  if VERBOSE:
                     print(
                        f"WARNING: {model} non-physical thermal parameters: "
                        f"C={thermal['C']:.3g}, C0={thermal['C0']:.3g}, gamma={thermal['gamma']:.3g}"
                     )

            therm_unc = thermal_uncertainties(
                  lmbda=lmbda,
                  lambda_unc=lambda_unc,
                  a_s=a_s,
                  a_s_unc=a_s_unc,
                  tau_s=tau_s,
                  tau_s_unc=tau_s_unc,
                  tau_f=tau_f,
                  tau_f_unc=tau_f_unc,
                  epsilon=epsilon,
                  epsilon_unc=epsilon_unc,
            )

            update_row(
                  df,
                  model,
                  {
                     **thermal,
                     **therm_unc,
                     "tau_f": tau_f,
                     "tau_s": tau_s,
                     "a_f": a_f,
                     "a_s": a_s,
                     "tau_f_unc": tau_f_unc,
                     "tau_s_unc": tau_s_unc,
                     "a_f_unc": a_f_unc,
                     "a_s_unc": a_s_unc,
                  },
            )

            plot_slow_fit(step2_axs[imodel], slow, model, iteration)

         if SAVE_ITERATION_PLOTS:
            suffix = f"iter{iteration + 1:02d}"
            step2_fig.savefig(
                  outdir / current_dir / "step2" / "png" / f"4xCO2_all_models_log_Teq_minus_T_vs_t_{suffix}.png",
                  dpi=200,
                  bbox_inches="tight",
            )
            step2_fig.savefig(
                  outdir / current_dir / "step2" / "pdf" / f"4xCO2_all_models_log_Teq_minus_T_vs_t_{suffix}.pdf",
                  bbox_inches="tight",
            )
         plt.close(step2_fig)

         if VERBOSE:
            print("Finished Step 2: thermal parameters updated")

         # Save a table every iteration for debugging.
         df.to_csv(outdir / current_dir / "tables" / f"params_iter{iteration + 1:02d}.csv", index=False)

         # Check convergence after the minimum number of iterations.
         new_params = numeric_params_for_convergence(df)
         if iteration + 1 >= MIN_ITERATIONS:
            denom = old_params.abs().replace(0, np.nan)
            rel_change = ((new_params - old_params).abs() / denom).replace([np.inf, -np.inf], np.nan)
            max_rel_change = float(np.nanmax(rel_change.to_numpy()))
            if VERBOSE:
               print(f"Max relative parameter change: {max_rel_change:.3e}")
            if np.isfinite(max_rel_change) and max_rel_change < CONVERGENCE_RTOL:
                  converged = True
                  last_iteration = iteration
                  if VERBOSE:
                     print(f"Converged after {iteration + 1} iterations.")
                  break

         last_iteration = iteration

      if not converged:
         print(f"Did not meet convergence tolerance after {MAX_ITERATIONS} iterations; using final iteration.")

      # Final parameter table.
      df.to_csv(outdir / current_dir / "tables" / "params_final.csv", index=False)
      # 2013a-compatible location, so the tau_s-vs-calibration-time plot below
      # can cross-reference tau_s from the other two run_types' results.
      df.to_csv(outdir / current_dir / "fitted_model_params.csv", index=False)
      print(df)

      # ------------------------- 3D view of the radiative regression -------------------------
      # Visualize the converged multilinear fit N = F - lambda*T - (eps-1)*H as
      # a plane in (T, H, N) space, per model, with the AOGCM data scattered on
      # top and colored by signed residual (obs - plane). This is the exact
      # relationship fit_radiative_epsilon regresses; the plot shows how planar
      # the data really are and how tightly they hug the fit.
      fig3d, axs3d = make_model_grid_3d(
         models,
         title=r"EBM-$\epsilon$ radiative fit: N = F $-\ \lambda$T $-\ (\epsilon-1)$H",
      )
      reg3d_scatters = []
      for imodel, model in enumerate(models):
         bundle = series_by_model[model]
         row = df.loc[df["model"] == model].iloc[0]
         F = float(row["F_ref"]); lam = float(row["lambda"]); eps = float(row["epsilon"])
         # H self-consistent with the final params (same call the last iteration used).
         H = H_physical_from_previous_solution(
            bundle.t_full,
            F_ref=F, lmbda=lam, C=float(row["C"]), epsilon=eps,
            T_eq=float(row["T_eq"]), a_f=float(row["a_f"]), a_s=float(row["a_s"]),
            tau_f=float(row["tau_f"]), tau_s=float(row["tau_s"]),
         )
         T, N = bundle.T_full, bundle.N_full
         good = finite_pair_mask(T, N, H)
         sc, resid = plot_regression_3d(axs3d[imodel], T[good], H[good], N[good], F, lam, eps, model)
         reg3d_scatters.append((sc, resid))

      # One shared, symmetric (diverging) residual color scale across all panels.
      all_resid = np.concatenate([r for _, r in reg3d_scatters]) if reg3d_scatters else np.array([0.0])
      rmax = float(np.nanmax(np.abs(all_resid))) or 1.0
      resid_norm = mpl.colors.Normalize(vmin=-rmax, vmax=rmax)
      for sc, _ in reg3d_scatters:
         sc.set_norm(resid_norm)
      smap = mpl.cm.ScalarMappable(norm=resid_norm, cmap="RdBu_r")
      smap.set_array([])
      cbar = fig3d.colorbar(smap, ax=list(axs3d), fraction=0.015, pad=0.02)
      cbar.set_label(r"N residual (obs $-$ fit) [W m$^{-2}$]", fontsize=13, fontweight="bold")
      for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
         fig3d.savefig(
            outdir / current_dir / "step1" / ext / f"4xCO2_all_models_N_T_H_regression3d{run_type_suffix}.{ext}",
            bbox_inches="tight", **kw,
         )
      plt.close(fig3d)

      # ------------------------- Compare parameters to paper values -------------------------

      model_paperParams = {
         "GISSE2R": {
            "F_ref": 9.1,
            "lambda": 2.03,
            "epsilon": 1.44,
            "T_eq": 4.5,
            "C": 6.1,
            "C0": 134,
            "gamma": 1.06,
            "tau_f": 1.7,
            "tau_s": 224,
         },
         "HadGEM2": {
            "F_ref": 6.8,
            "lambda": 0.61,
            "epsilon": 1.54,
            "T_eq": 11.1,
            "C": 7.5,
            "C0": 98,
            "gamma": 0.49,
            "tau_f": 5.4,
            "tau_s": 457,
         },
         "IPSLCM5A": {
            "F_ref": 6.7,
            "lambda": 0.79,
            "epsilon": 1.14,
            "T_eq": 8.5,
            "C": 8.1,
            "C0": 100,
            "gamma": 0.57,
            "tau_f": 5.5,
            "tau_s": 327,
         },
         "MPIESM11": {
            "F_ref": 9.4,
            "lambda": 1.21,
            "epsilon": 1.42,
            "T_eq": 7.8,
            "C": 8.5,
            "C0": 78,
            "gamma": 0.62,
            "tau_f": 4.0,
            "tau_s": 220,
         },
      }

      colors = [
         ("#ef9a9a", "#b71c1c"),
         ("#90caf9", "#1565c0"),
         ("#a5d6a7", "#2e7d32"),
         ("#ffcc80", "#ef6c00"),
      ]

      available_validation_models = [m for m in model_paperParams if m in set(df["model"])]
      if len(available_validation_models) > 0:
         validation_vars = list(model_paperParams[available_validation_models[0]].keys())
         nvars = len(validation_vars)
         # NOTE: model_paperParams has 9 keys (includes epsilon), which doesn't fit
         # make_model_grid's fixed 4x2/8-panel assumption, so this grid is built
         # manually with the same spacing/styling conventions instead.
         ncols = 5
         nrows = int(np.ceil(nvars / ncols))
         fig_val, axs_val = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 6 * nrows),
            constrained_layout=False,
         )
         fig_val.subplots_adjust(left=0.05, right=0.98, bottom=0.075, top=0.95, wspace=0.12, hspace=0.18)
         fig_val.suptitle("Geoffroy vs. replication (w/ 95% CI)", fontsize=20, fontweight="bold")
         fig_val.text(0.02, 0.5, "A.U.", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=90.)
         axs_val = np.asarray(axs_val).ravel()

         for ivar, var in enumerate(validation_vars):
            ax = axs_val[ivar]
            handles = []
            values_for_range = []

            for model, (face, edge) in zip(available_validation_models, colors):
                  mu_GF = float(model_paperParams[model][var])
                  mu_SN = float(df.loc[df["model"] == model, var].iloc[0])
                  unc_col = f"{var}_unc"
                  mu_unc = float(df.loc[df["model"] == model, unc_col].iloc[0]) if unc_col in df.columns else np.nan
                  xerr = None if not np.isfinite(mu_unc) else mu_unc * 1.96

                  ax.errorbar(
                     mu_SN,
                     mu_GF,
                     xerr=xerr,
                     marker=".",
                     ms=13,
                     mfc=face,
                     mec=edge,
                     ecolor=edge,
                     linewidth=1.5,
                     zorder=3,
                  )
                  values_for_range.extend([mu_SN, mu_GF])
                  handles.append(
                     Line2D(
                        [],
                        [],
                        marker=".",
                        linestyle="None",
                        markersize=13,
                        markerfacecolor=face,
                        markeredgecolor=edge,
                        markeredgewidth=1.5,
                        label=model,
                     )
                  )

            arr = np.asarray(values_for_range, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                  lo, hi = np.min(arr), np.max(arr)
                  pad = 0.05 * (hi - lo) if hi > lo else 1.0
                  one_to_one = np.linspace(lo - pad, hi + pad, 200)
                  ax.plot(one_to_one, one_to_one, "k--", label=r"$\mu_{\rm GF}=\mu_{\rm SN}$")

            format_ax(ax, text=var, xlabel=r"$\mu_{\rm SN}$", xscale="linear", yscale="linear", legend=False)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(handles=handles, loc='lower left', prop={'weight': 'bold', 'size': 10})

         for ax in axs_val[nvars:]:
            ax.set_visible(False)

         fig_val.savefig(
            outdir / current_dir / "validation" / "png" / "all_validation_params_GF_vs_SN.png",
            dpi=200,
            bbox_inches="tight",
         )
         fig_val.savefig(
            outdir / current_dir / "validation" / "pdf" / "all_validation_params_GF_vs_SN.pdf",
            bbox_inches="tight",
         )
         plt.close(fig_val)
      else:
         print("Skipping validation plot: none of the hard-coded paper model names are present in df.")


      # ------------------------- Final result plots -------------------------

      # These plots intentionally mirror the final plotting block in
      # 2013a_allModels.py, while using the EBM-epsilon parameterization.
      #
      # IMPORTANT: this is separate from the calibration bundle above.
      # In 2013a_allModels.py, changing results from "validation" to
      # "unblinded" changes how much of the 4xCO2/piControl data is shown:
      #   - validation: first 151 values
      #   - unblinded: full available time series
      # The helper below preserves that behavior for the 2013b script.

      VALIDATION_PLOT_YEARS = 151

      # Approximate transient climate response values used by the OHC-vs-Ts
      # unblinded plot in 2013a_allModels.py.
      tcr = {
         "CCSM3": 1.7,
         "CESM1": 2.0,
         "CNRMCM6": 2.1,
         "ECHAM5": 3.0,
         "GISSE2R": 1.5,
         "IPSLCM5A": 2.3,
         "HadGEM2": 2.5,
         "MPIESM11": 2.0,
      }


      def make_result_series(model_data, results):
         """
         Return the plotting time series with the same validation/unblinded
         selection logic as 2013a_allModels.py.

         The calibration code uses a fixed, internally consistent calibration
         bundle. This helper is only for final plotting and intentionally
         follows the result-window behavior of the 2013a script.
         """
         T_ctrl_raw = get_r_array(model_data, "piControl", "T2M")
         N_ctrl_raw = get_r_array(model_data, "piControl", "NETTOA")
         T_4x_raw = get_r_array(model_data, "4xCO2", "T2M")
         N_4x_raw = get_r_array(model_data, "4xCO2", "NETTOA")

         if results == "validation":
            T_ctrl = T_ctrl_raw[:VALIDATION_PLOT_YEARS]
            N_ctrl = N_ctrl_raw[:VALIDATION_PLOT_YEARS]
            T_4x = T_4x_raw[:VALIDATION_PLOT_YEARS]
            N_4x = N_4x_raw[:VALIDATION_PLOT_YEARS]
         elif results == "unblinded":
            T_ctrl = T_ctrl_raw
            N_ctrl = N_ctrl_raw
            T_4x = T_4x_raw
            N_4x = N_4x_raw
         else:
            raise ValueError("results must be either 'validation' or 'unblinded'.")

         if SUBTRACT_PICONTROL_BASELINE:
            T_base = float(np.nanmean(T_ctrl))
            N_base = float(np.nanmean(N_ctrl))
         else:
            T_base = 0.0
            N_base = 0.0

         T = T_4x - T_base
         N = N_4x - N_base
         t = np.arange(1, 1 + len(T), dtype=float)

         good = finite_pair_mask(t, T, N)
         return t[good], T[good], N[good]


      def ebm_epsilon_nettoa(t, F_ref, lmbda, epsilon, C, T_eq, a_f, a_s, tau_f, tau_s):
         """TOA imbalance from the final EBM-epsilon solution."""
         T_fit = T_model(t, T_eq, a_f, a_s, tau_f, tau_s)
         H_fit = H_physical_from_previous_solution(
            t,
            F_ref=F_ref,
            lmbda=lmbda,
            C=C,
            epsilon=epsilon,
            T_eq=T_eq,
            a_f=a_f,
            a_s=a_s,
            tau_f=tau_f,
            tau_s=tau_s,
         )
         return F_ref - lmbda * T_fit - (epsilon - 1.0) * H_fit


      def rolling_mean(x, window=10):
         x = np.asarray(x, dtype=float)
         if len(x) < window:
            return np.array([], dtype=float)
         return np.array([np.nanmean(x[i : i + window]) for i in range(len(x) - window + 1)], dtype=float)


      def plot_surface_budget_bars(model, F_ref, lmbda, epsilon, C, T_eq,
                                   a_f, a_s, tau_f, tau_s, t_final_years,
                                   png_path, pdf_path, n_save=60):
         """Stacked-bar decomposition of the surface tendency dT/dt for the final
         EBM-epsilon (Geoffroy 2013b) fit, one bar per (log-spaced) time step. The
         upper-layer surface node obeys

             C * dT/dt = F - lambda*T - eps*H ,   H = gamma*(T - T0) ,

         with H the physical deep-ocean heat uptake reconstructed from the analytic
         two-mode solution. Regrouping the efficacy so the TOA-imbalance term
         N = F - lambda*T - (eps-1)*H is kept together and dividing by the
         upper-layer heat capacity C gives three additive contributions [K/yr] that
         sum EXACTLY to dT/dt:

             +F/C                         constant CO2 forcing (flat in time),
             -(lambda*T + (eps-1)*H)/C    radiative restoring + efficacy loss,
             -H/C                         plain deep-ocean heat loss.

         The top panel plots the |magnitude| of each term as stacked bars on a
         logarithmic y-axis (negative-sign terms are shown by magnitude so they fit
         the log scale), with the black line the |net dT/dt| and the dashed red line
         the equilibration ratio T/T_eq on a right-hand axis. The lower panel repeats
         the same magnitudes as line plots. Time is drawn on a linear axis with each
         bar spanning the arithmetic midpoints to its neighbours (samples are
         log-spaced, so early bars are narrow). C carries the year unit
         (W yr m^-2 K^-1), so F/C etc. are already in K/yr with no extra conversion.
         """
         tv = np.logspace(0.0, np.log10(max(t_final_years, 2.0)), n_save)
         T_s = T_model(tv, T_eq, a_f, a_s, tau_f, tau_s)
         H_flux = H_physical_from_previous_solution(
            tv, F_ref=F_ref, lmbda=lmbda, C=C, epsilon=epsilon, T_eq=T_eq,
            a_f=a_f, a_s=a_s, tau_f=tau_f, tau_s=tau_s,
         )

         # All three sum to dT/dt [K/yr]; C = W yr m^-2 K^-1 absorbs the time unit.
         forcing   = np.full_like(T_s, F_ref) / C
         restoring = (-lmbda * T_s - (epsilon - 1.0) * H_flux) / C   # = (N - F)/C
         uptake    = (-H_flux) / C
         net       = forcing + restoring + uptake                    # = dT/dt [K/yr]

         comps = [
            (r"$F/C$",                                    forcing,   "#9467bd"),
            (r"$-(\lambda T + (\epsilon-1)H)/C$",         restoring, "#1f77b4"),
            (r"$-H/C$",                                   uptake,    "#2ca02c"),
         ]
         # Linear time axis: place bars at their true year positions with widths
         # spanning the arithmetic midpoints to the neighbouring log-spaced steps.
         mid = 0.5 * (tv[:-1] + tv[1:])
         left = np.concatenate(([max(0.0, tv[0] - (mid[0] - tv[0]))], mid))
         right = np.concatenate((mid, [tv[-1] + (tv[-1] - mid[-1])]))
         width = right - left

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
         ax.plot(tv, net_abs, color="black", lw=1.4, label=r"$|$net $dT/dt|$")
         ax.set_yscale("log")
         ax.set_ylim(floor, base.max() * 1.3)
         ax.set_ylabel(r"$|dT/dt$ contribution$|$ (K yr$^{-1}$)", fontsize=14, fontweight="bold")
         ax.set_title(rf"{model}: surface tendency budget "
                      rf"($\epsilon$={epsilon:.2f}, $\tau_s$={tau_s:.0f} yr, $C$={C:.1f})",
                      fontsize=13, fontweight="bold")

         # Overlay the surface warming itself as the equilibration ratio T/T_eq on a
         # right-hand axis, so the correspondence between how far the surface has
         # equilibrated and its instantaneous tendency dT/dt is visible at a glance.
         ax_r = ax.twinx()
         ratio_line, = ax_r.plot(tv, T_s / T_eq, color="red", lw=2.0, ls="--",
                                 label=r"$T/T_{eq}$")
         ax_r.set_ylabel(r"Equilibrium Ratio $T/T_{eq}$", fontsize=14,
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
         fig.savefig(str(pdf_path), bbox_inches="tight")
         plt.close(fig)


      final_fig, final_axs = make_model_grid(
         models, dpi=120,
         title=r"4xCO$_{2}$ T$_{2M}$ vs. Time w/ EBM-$\epsilon$ Fit",
         xlabel=r"Time (years)", ylabel=r"Temperature Anomaly (K)",
         right=0.95, wspace=0.28,
      )
      final_xmax = []
      final_fig.text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=-90.)

      nettoa_fig, nettoa_axs = make_model_grid(
         models,
         title=r"4xCO$_{2}$ Net TOA vs. Time",
         xlabel="Time (years)", ylabel=r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)",
      )

      tau_s_fig, tau_s_axs = make_model_grid(
         models,
         title=r"4xCO$_{2}$ $\tau_s$ vs. Calibration Time",
         xlabel=r"Calibration Time (years)", ylabel=r"$\tau_s$ (years)",
         sharex=True, sharey=True,
      )

      # These two are only populated/saved for unblinded runs, matching 2013a_allModels.py.
      ohct_fig = None
      ohct_axs = None
      assmpt_fig = None
      assmpt_axs = None
      if results == "unblinded":
         ohct_fig, ohct_axs = make_model_grid(
            models,
            title=r"4xCO$_2$ OHU vs. Surface Temp (Normalized)",
            xlabel=r"$T_s/2\, \mathrm{ECS}$", ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$",
         )
         assmpt_fig, assmpt_axs = make_model_grid(
            models,
            title=r"4xCO$_{2}$ Assumption 2 Test",
            xlabel="Time (years)", ylabel=r"$C_{u}\frac{dT_{u}}{dt}$",
         )

      sc_for_cbar = None
      rng = np.random.default_rng(12345)

      for imodel, model in enumerate(models):
         t, T_obs, N_obs = make_result_series(data.rx2(model), results)
         row = df.loc[df["model"] == model].iloc[0]

         F_ref = float(row["F_ref"])
         lmbda = float(row["lambda"])
         epsilon = float(row["epsilon"])
         C = float(row["C"])
         T_eq = float(row["T_eq"])
         a_f = float(row["a_f"])
         a_s = float(row["a_s"])
         tau_f = float(row["tau_f"])
         tau_s = float(row["tau_s"])

         T_eq_unc = float(row["T_eq_unc"])
         a_s_unc = float(row["a_s_unc"])
         tau_f_unc = float(row["tau_f_unc"])
         tau_s_unc = float(row["tau_s_unc"])

         T_fit = T_model(t, T_eq, a_f, a_s, tau_f, tau_s)
         N_fit = ebm_epsilon_nettoa(t, F_ref, lmbda, epsilon, C, T_eq, a_f, a_s, tau_f, tau_s)

         # ----------------- surface tendency budget (stacked bars) -----------------
         # F/C, -(lambda*T + (eps-1)*H)/C, -H/C summing to dT/dt for the final fit.
         plot_surface_budget_bars(
            model, F_ref, lmbda, epsilon, C, T_eq, a_f, a_s, tau_f, tau_s,
            float(np.max(t)),
            outdir / current_dir / "budget" / "png" / f"{model}_surface_budget_bars.png",
            outdir / current_dir / "budget" / "pdf" / f"{model}_surface_budget_bars.pdf",
         )

         # ----------------- T2M time-series plot -----------------
         ax = final_axs[imodel]
         ax.scatter(t, T_obs, s=4, color="red")
         ax.plot(t, T_obs, color="red", label="2-m Surface Temp.")
         ax.plot(t, T_fit, color="blue", label="EBM-ε Fit")

         iterations = 1000
         if np.all(np.isfinite([T_eq_unc, a_s_unc, tau_f_unc, tau_s_unc])):
            param_mean = np.array([T_eq, a_s, tau_f, tau_s], dtype=float)
            param_cov = np.diag(np.array([T_eq_unc**2, a_s_unc**2, tau_f_unc**2, tau_s_unc**2], dtype=float))
            params = rng.multivariate_normal(param_mean, param_cov, size=iterations)

            T_ensemble = []
            for T_eq_i, a_s_i, tau_f_i, tau_s_i in params:
                  if tau_f_i <= 0 or tau_s_i <= 0 or a_s_i <= 0 or a_s_i >= 1:
                     continue
                  T_ensemble.append(
                     T_model(t, T_eq_i, 1.0 - a_s_i, a_s_i, tau_f_i, tau_s_i)
                  )

            if len(T_ensemble) > 0:
                  T_ensemble = np.vstack(T_ensemble)
                  T_mean = np.nanmean(T_ensemble, axis=0)
                  T_std = np.nanstd(T_ensemble, axis=0)
                  ax.fill_between(t, T_mean - 2 * T_std, T_mean + 2 * T_std, color="blue", alpha=0.08, label="±2σ")
                  ax.fill_between(t, T_mean - T_std, T_mean + T_std, color="blue", alpha=0.20, label="±1σ")

         ax.text(
            0.02,
            0.92,
            f"$\\tau_s$ = {tau_s:.1f} yr\n$\\epsilon$ = {epsilon:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=EXTRA_TEXT_FONTSIZE,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
         )
         ax.axvline(150, color="orange", linestyle=":", linewidth=1, alpha=0.7)

         y_bottom, y_top = 1, 1.25 * T_eq
         ratio_ticks = np.arange(0.2, 1.21, 0.2)
         format_ax(
            ax,
            text=model,
            xscale="linear",
            yscale="linear",
            ylim=(y_bottom, y_top),
            yticks=ratio_ticks * T_eq,
            legend_loc='lower right',
         )
         final_xmax.append(np.max(t))

         # Secondary y-axis showing T/T_eq (Equilibrium Ratio) on the same scale;
         # ratio ticks match the left axis's fractions so gridlines line up exactly.
         ax2 = ax.twinx()
         ax2.set_ylim(y_bottom / T_eq, y_top / T_eq)
         ax2.set_yticks(ratio_ticks)
         ax2.tick_params(labelsize=14, width=2, length=8, direction="in")

         # ----------------- tau_s vs. calibration length plot -----------------
         # Only meaningful for run_type == 3 (50-yr avg + LR param fit), where we
         # re-derive tau_s using progressively longer chunks of the raw record to
         # see how the slow-mode estimate depends on run length. Mirrors the
         # (approximate, non-iterative) Gregory + slow-mode fit used in the
         # 2013a script's equivalent plot.
         if run_type == 3:
            model_data_raw = data.rx2(model)
            T_ctrl_raw = get_r_array(model_data_raw, "piControl", "T2M")
            N_ctrl_raw = get_r_array(model_data_raw, "piControl", "NETTOA")
            T_4x_raw = get_r_array(model_data_raw, "4xCO2", "T2M")
            N_4x_raw = get_r_array(model_data_raw, "4xCO2", "NETTOA")
            T_base_full = float(np.nanmean(T_ctrl_raw))
            N_base_full = float(np.nanmean(N_ctrl_raw))

            cal_lengths = np.arange(151, len(T_4x_raw), 1)
            tau_s_lst = []
            for i in cal_lengths:
               t2m_tauRun = T_4x_raw[:i] - T_base_full
               nettoa_tauRun = N_4x_raw[:i] - N_base_full
               t_tauRun = np.arange(0, t2m_tauRun.shape[0], 1)

               [m_step1, b_step1] = np.polyfit(t2m_tauRun, nettoa_tauRun, 1)
               T_eq_tauRun = b_step1 / (-m_step1)

               mask_tauRun = (T_eq_tauRun - t2m_tauRun) > 0
               t_tauRun = t_tauRun[mask_tauRun]
               t2m_tauRun = t2m_tauRun[mask_tauRun]

               y_tauRun = np.log(T_eq_tauRun - t2m_tauRun[30:]) - np.log(T_eq_tauRun)
               [m_slope, b_slope] = np.polyfit(t_tauRun[30:], y_tauRun, 1)
               tau_s_lst.append(-1 / m_slope)

            ax = tau_s_axs[imodel]
            ax.plot(cal_lengths, tau_s_lst)
            # Only extends to this model's own last data point, so the line's
            # length visually encodes how long each model's run was.
            ax.plot(cal_lengths, cal_lengths, label='y=x', color='black')

            fitted_params_path1 = outdir / dir_list[0] / "fitted_model_params.csv"
            fitted_params_path2 = outdir / dir_list[1] / "fitted_model_params.csv"
            fitted_params_path3 = outdir / dir_list[2] / "fitted_model_params.csv"

            if fitted_params_path1.is_file() and fitted_params_path2.is_file() and fitted_params_path3.is_file():
               df_runType1 = pd.read_csv(fitted_params_path1)
               df_runType2 = pd.read_csv(fitted_params_path2)
               df_runType3 = pd.read_csv(fitted_params_path3)

               tau_s_runType1 = df_runType1.loc[df_runType1['model'] == model, "tau_s"].iloc[0]
               tau_s_runType2 = df_runType2.loc[df_runType2['model'] == model, "tau_s"].iloc[0]
               tau_s_runType3 = df_runType3.loc[df_runType3['model'] == model, "tau_s"].iloc[0]

               ax.scatter(151, tau_s_runType1, s=14, color='red', label=r'Geoffroy 2013b')
               ax.scatter(151, tau_s_runType2, s=14, color='yellow', label=r'50-yr Avg T$_{eq}$')
               ax.scatter(len(T_4x_raw), tau_s_runType3, s=14, color='green', label=r'50-yr Avg + LR Fit')

            # How close this model actually got to equilibrium by the end
            # of its own run (last-10-yr mean T2M anomaly / T_eq), shown as
            # extra text under the model-name label.
            eq_ratio = float((np.mean(T_4x_raw[-10:]) - T_base_full) / T_eq)

            ax.text(
               0.02,
               0.92,
               f"$Reached {eq_ratio * 100:.0f}% of $T_{{eq}}$",
               transform=ax.transAxes,
               va="top",
               ha="left",
               fontsize=EXTRA_TEXT_FONTSIZE,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
            )

            format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc='upper right')

         # ----------------- NETTOA time-series plot -----------------
         T_obs_roll = rolling_mean(T_obs, window=10)
         N_obs_roll = rolling_mean(N_obs, window=10)
            
         H_fit = H_physical_from_previous_solution(
             t,
             F_ref=F_ref,
             lmbda=lmbda,
             C=C,
             epsilon=epsilon,
             T_eq=T_eq,
             a_f=a_f,
             a_s=a_s,
             tau_f=tau_f,
             tau_s=tau_s,
         )
         H_fit_roll = rolling_mean(H_fit, window=10)
            
         t_roll = t[: len(N_obs_roll)]
            
         ax = nettoa_axs[imodel]
         if len(N_obs_roll) > 0:
             ax.plot(t_roll, N_obs_roll, color="black", label="AOGCM")
            
             ax.plot(
                 t_roll,
                 F_ref - lmbda * T_obs_roll - (epsilon - 1.0) * H_fit_roll,
                 label=r"$F-\lambda T-(\epsilon-1)H$",
             )

         if results == "unblinded":
             ax.plot(t, N_fit, label="EBM-ε fit")

         format_ax(ax, text=model, xscale="linear", yscale="log", legend_loc='lower left')

         if results == "unblinded":
            # ----------------- OHC vs. T_s plot -----------------
            t_long = np.arange(1, 1 + 100000, dtype=float)
            T_long = T_model(t_long, T_eq, a_f, a_s, tau_f, tau_s)
            N_long = ebm_epsilon_nettoa(t_long, F_ref, lmbda, epsilon, C, T_eq, a_f, a_s, tau_f, tau_s)

            normalized_OHC_pred = (5.1e14 * 31536000 * np.cumsum(N_long)) / (1.37e21 * 3850)
            normalized_OHC = (5.1e14 * 31536000 * np.cumsum(N_obs)) / (1.37e21 * 3850)

            ax = ohct_axs[imodel]
            cmap = plt.cm.turbo
            norm = mpl.colors.Normalize(vmin=0, vmax=max(6000, len(T_obs)))
            sc_for_cbar = ax.scatter(T_obs / T_eq, normalized_OHC / T_eq, c=t, cmap=cmap, norm=norm)

            if len(T_obs) >= 5:
                  m_ohcTs, b_ohcTs = np.polyfit(T_obs[:5] / T_eq, normalized_OHC[:5] / T_eq, 1)
                  t_val = np.arange(0, 1.1, 0.1)
                  ax.plot(t_val, m_ohcTs * t_val + b_ohcTs, ls="--", color="red", label=f"Mixed Layer Depth = {(m_ohcTs * 2500):.0f} m")
            else:
                  t_val = np.arange(0, 1.1, 0.1)

            A = np.nan
            if model in tcr and np.isfinite(T_eq) and T_eq != 0:
                  A = 2 * tcr[model] / T_eq
                  ax.plot(t_val, (t_val - A) / (1 - A), ls="--", color="black", label="2-box Asymptotic Pred.")
                  ax.axvline(A, color="0.55", ls="--", lw=0.8)

            ax.plot(T_long / T_eq, normalized_OHC_pred / T_eq, color="green", label="EBM-ε Pred.")
            ax.axvline(1.0, color="0.55", ls="--", lw=0.8)
            format_ax(ax, text=model, xscale="linear", yscale="linear", ylim=(-0.05, 1.2))

            # ----------------- C dT/dt assumption-test plot -----------------
            dT_dt_fit = np.gradient(T_fit)
            dT_dt_true = np.gradient(T_obs_roll) if len(T_obs_roll) > 1 else np.array([], dtype=float)
            F_thresh = F_ref / 100.0

            ax = assmpt_axs[imodel]
            if len(dT_dt_true) > 0:
                  ax.plot(t[: len(dT_dt_true)], C * dT_dt_true, color="black", label="AOGCM (10 yr running mean)")
            ax.plot(t, C * dT_dt_fit, ls="--", color="blue", label="EBM-ε")
            ax.axhline(F_thresh, color="0.55", ls="--", lw=0.8, label="Order of Mag. Threshold")
            format_ax(ax, text=model, xscale='log', yscale='linear')


      # Save combined final figures in both linear and log x-scale variants, with the
      # same naming/placement pattern as 2013a_allModels.py.
      for scale in ["linear", "log"]:
         for ax, xmax in zip(final_axs, final_xmax):
            ax.set_xscale(scale)
            ax.set_xlim(1, xmax + 1)
            if scale == "linear":
               ax.set_xticks(np.linspace(1, xmax + 1, 5))
            else:
               ax.xaxis.set_major_locator(mpl.ticker.LogLocator())

         for ax, xmax in zip(nettoa_axs, final_xmax):
            ax.set_xscale(scale)
            ax.set_xlim(1, xmax + 1)

         final_fig.savefig(
            outdir / current_dir / results / "png" / f"4xCO2_all_models_T2m_vs_t_{results}_{scale}.png",
            dpi=200,
            bbox_inches="tight",
         )
         final_fig.savefig(
            outdir / current_dir / results / "pdf" / f"4xCO2_all_models_T2m_vs_t_{results}_{scale}.pdf",
            bbox_inches="tight",
         )

         nettoa_fig.savefig(
            outdir / current_dir / results / "png" / f"4xCO2_all_models_NETTOA_timeseries_{scale}.png",
            dpi=200,
            bbox_inches="tight",
         )
         nettoa_fig.savefig(
            outdir / current_dir / results / "pdf" / f"4xCO2_all_models_NETTOA_timeseries_{scale}.pdf",
            bbox_inches="tight",
         )

      plt.close(final_fig)
      plt.close(nettoa_fig)

      # Explicitly span every panel to the full extent of all plotted data
      # (across all models) rather than relying on sharex/sharey autoscale,
      # so the longest run's data is never clipped in any panel.
      populated_axs = [ax for ax in tau_s_axs if ax.has_data()]
      if populated_axs:
         xmax = max(ax.dataLim.intervalx[1] for ax in populated_axs)
         ymax = max(ax.dataLim.intervaly[1] for ax in populated_axs)
         for ax in tau_s_axs:
            ax.set_xlim(0, xmax + 100)
            ax.set_ylim(0, ymax + 100)
            # Denser tick marks than the default locator gives.
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))

      tau_s_fig.savefig(
         outdir / current_dir / "step2" / "png" / f"4xCO2_all_models_tau_s_vs_calibration_t{run_type_suffix}.png",
         dpi=200,
         bbox_inches="tight",
      )
      tau_s_fig.savefig(
         outdir / current_dir / "step2" / "pdf" / f"4xCO2_all_models_tau_s_vs_calibration_t{run_type_suffix}.pdf",
         bbox_inches="tight",
      )
      plt.close(tau_s_fig)

      if results == "unblinded":
         assmpt_fig.savefig(
            outdir / current_dir / results / "png" / f"4xCO2_all_models_cdTdt_t_{results}.png",
            dpi=200,
            bbox_inches="tight",
         )
         assmpt_fig.savefig(
            outdir / current_dir / results / "pdf" / f"4xCO2_all_models_cdTdt_t_{results}.pdf",
            bbox_inches="tight",
         )
         plt.close(assmpt_fig)

         if sc_for_cbar is not None:
            cbar = ohct_fig.colorbar(sc_for_cbar, ax=ohct_axs.ravel().tolist(), fraction=0.025, pad=0.025)
            cbar.set_label("Year")

         ohct_fig.savefig(
            outdir / current_dir / results / "png" / "4xCO2_all_models_ohc_ts.png",
            dpi=200,
            bbox_inches="tight",
         )
         ohct_fig.savefig(
            outdir / current_dir / results / "pdf" / "4xCO2_all_models_ohc_ts.pdf",
            bbox_inches="tight",
         )
         plt.close(ohct_fig)

      print("Finished final val/result plots")
      print("Done.")
      print(f"Final parameter table: {outdir / current_dir / 'tables' / 'params_final.csv'}")
      print(f"2013a-compatible parameter table: {outdir / current_dir / 'fitted_model_params.csv'}")
