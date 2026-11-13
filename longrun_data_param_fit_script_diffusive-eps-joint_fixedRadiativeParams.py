#!/usr/bin/env python3
"""Slab + diffusive thermocline with efficacy: fit (kappa, h_ml) at fixed
radiative parameters.

Step 1 sets F_ref, lambda, eps and T_eq per model from two Net-TOA-vs-T
regressions. Step 2 fits log10(kappa) and h_ml by least squares against the
temperature series and the Net TOA balance N = F - lambda*T - (eps-1)*H, on a
closed 3700 m column.

Models: the 8 LongRunMIP abrupt-4xCO2 runs plus GFDL CM3 and ESM2M (abrupt
2xCO2) and ESM2G (1%/yr ramp to 4x). Run types set the temperature fit window:
years 1-151 for 1 and 2, the whole run for 3 (Step 1 is the same for all).

Flags: --test [N | M1,M2], --high-res (41x41 heatmaps), --li (ECHAM5 ocean
temperature profiles against Li et al. 2012 Fig. 3).
Output: figures/diffusive-eps-joint_fixedRadiativeParams/<run type>/.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
from scipy.special import erfcx

import diffusive_common as dc
from diffusive_common import (
   EXPT, KAPPA_HML_BOUNDS_HI, KAPPA_HML_BOUNDS_LO, NETTOA_SCALE, OHC_TS_EQUILIBRIUM_YEARS,
   PDE_FIT_DZ_TARGET, PDE_Z_MAX, RHO_CP, Case, ramp_frac,
)
from figure_paths import BUDGET, DPI_WIDE, RunLayout, figname
from longrun_common import (
   EARTH_AREA, MODELS as LONGRUN_MODELS, OCEAN_HEAT_CAPACITY, SECONDS_PER_YEAR,
   add_test_argument, announce_test_mode, load_longrun_data, perturbed_anomalies,
   select_models, sympy_prop_unc,
)
from pde_solver_1lyrEBM_diffusive import Params as PDEParams, solve_model as pde_solve_model, YEAR
from plot_style import AXIS_LABEL_FONTSIZE, SUPTITLE_FONTSIZE, format_ax

VARIANT = "diffusive-eps-joint_fixedRadiativeParams"
RUN_TYPES = [3, 2, 1]
MODELS = LONGRUN_MODELS + ['CM3', 'ESM2M', 'ESM2G']
TITLE_PREFIX = r"CO$_2$ runs:"     # panels mix experiments; each tag names its own

# Floors on the Step 1 parameters, so a degenerate regression cannot hand the
# PDE a non-positive forcing, feedback or efficacy.
FREF_FLOOR = 0.1                   # W/m^2
LAMBDA_FLOOR = 0.05                # W/m^2/K
EPS_FLOOR = 1e-3

# Heatmap window, as multiples of the fitted kappa and h_ml.
KAPPA_FACTOR_RANGE_2D = (0.5, 2.0)
HML_FACTOR_RANGE_2D = (0.5, 2.0)
N_2D_HIGH_RES = 41

# Ocean temperature profiles (--li): the Li et al. (2012) model levels, as absolute depths.
PROFILE_AOGCM_COLUMN_Z = 5170.0    # m
PROFILE_DEPTHS_M = (6.0, 1525.0, 2080.0, 3070.0, 4195.0, PROFILE_AOGCM_COLUMN_Z)
LI2012_FIG3_CURVES_CSV = Path(__file__).resolve().parent / "data" / "li2012_fig3_curves.csv"
LI2012_FIG3_YEAR = 1000.0          # overlay and model/Li ratio stop here

PARAM_COLS = [
   "model", "F_ref", "lambda", "T_eq", "epsilon", "kappa_pde", "h_ml_pde",
   "F_ref_unc", "lambda_unc", "T_eq_unc", "epsilon_unc",
]


# ---- experiment metadata ----

def model_meta(model_data):
   """(co2_multiple, ramp_years) of the perturbed run; abrupt 4xCO2 if absent."""
   names = list(model_data.names)
   co2 = float(model_data.rx2("co2_multiple")[0]) if "co2_multiple" in names else 4.0
   ramp = float(model_data.rx2("ramp_years")[0]) if "ramp_years" in names else 0.0
   return co2, ramp


def model_label(model, co2_multiple, ramp_years):
   """Panel tag: the model, plus its experiment when it is not abrupt 4xCO2."""
   if ramp_years > 0:
      return rf"{model} ({ramp_years:.0f}-yr ramp$\to${co2_multiple:g}xCO$_2$)"
   if co2_multiple != 4:
      return rf"{model} ({co2_multiple:g}xCO$_2$)"
   return model


def bin_start(ramp_years, width=50):
   """First year of the 50-yr regression bins: skips the first bin and every
   bin the ramp reaches into."""
   return width * (1 + int(np.ceil(max(ramp_years, 0.0) / width)))


# ---- Step 1: radiative parameters ----

def fit_radiative(ax, df, model, label, t2m, nettoa, ramp_years):
   """Set F_ref, lambda, eps and T_eq from annual anomalies and draw the panel.

      F_ref   intercept of N vs T over the first 20 years at constant forcing
      T_eq    x-intercept of the 50-yr-binned N vs T fit (slope mb, intercept bb)
      lambda  F_ref / T_eq
      eps     F_ref / bb        (bb = F/eps and mb = -lambda/eps)

   Returns the values and the regression rows Step 2 reuses.
   """
   years = np.arange(1, 1 + t2m.shape[0])
   greg = (years > ramp_years) & (years <= ramp_years + 20)
   [m20, b20], cov20 = np.polyfit(t2m[greg], nettoa[greg], 1, cov=True)
   b20_unc = float(np.sqrt(cov20[1, 1]))

   start = bin_start(ramp_years)
   t2m_bin = np.array([t2m[j:j + 50].mean() for j in range(start, t2m.shape[0], 50)])
   nettoa_bin = np.array([nettoa[j:j + 50].mean() for j in range(start, nettoa.shape[0], 50)])
   [mb, bb], covb = np.polyfit(t2m_bin, nettoa_bin, 1, cov=True)
   mb_unc = float(np.sqrt(covb[0, 0]))
   bb_unc = float(np.sqrt(covb[1, 1]))

   F_ref = max(float(b20), FREF_FLOOR)
   T_eq_raw = float(-bb / mb)
   T_eq = T_eq_raw if T_eq_raw > 0 else 1.0     # guard a degenerate x-intercept
   lam = max(F_ref / T_eq, LAMBDA_FLOOR)
   eps = max(F_ref / bb, EPS_FLOOR)

   sp_b20, sp_mb, sp_bb = sp.symbols("b20 mb bb")
   dc.set_row(df, model, {
      "F_ref": F_ref, "lambda": lam, "T_eq": T_eq, "epsilon": eps,
      "F_ref_unc": b20_unc,
      "lambda_unc": sympy_prop_unc(-sp_b20 * sp_mb / sp_bb,
                                   {sp_b20: b20, sp_mb: mb, sp_bb: bb},
                                   {sp_b20: b20_unc, sp_mb: mb_unc, sp_bb: bb_unc}),
      "T_eq_unc": sympy_prop_unc(-sp_bb / sp_mb, {sp_bb: bb, sp_mb: mb},
                                 {sp_bb: bb_unc, sp_mb: mb_unc}),
      "epsilon_unc": sympy_prop_unc(sp_b20 / sp_bb, {sp_b20: b20, sp_bb: bb},
                                    {sp_b20: b20_unc, sp_bb: bb_unc}),
   })

   # Panel: the binned fit drawn out to T_eq, and the 20-yr fit.
   ax.scatter(t2m_bin, nettoa_bin, s=10, alpha=0.6, color="tab:blue", label="50-yr bins (data)")
   xb = np.linspace(min(float(t2m_bin.min()), 0.0), T_eq, 100)
   ax.plot(xb, mb * xb + bb, color="tab:blue", lw=2,
           label=rf"50-yr fit: $T_{{eq}}$={T_eq:.1f} K, $\epsilon$={eps:.2f}")
   ax.scatter(t2m[greg], nettoa[greg], s=8, alpha=0.5, color="0.5",
              label=f"yrs {years[greg][0]}-{years[greg][-1]} (annual)")
   x20 = np.linspace(0.0, float(t2m[greg].max()), 100)
   ax.plot(x20, m20 * x20 + b20, color="0.35", lw=2,
           label=rf"20-yr fit: $F_{{ref}}$={F_ref:.3f}")
   ax.axhline(0.0, color="0.7", lw=0.8, ls=":")

   resid = nettoa_bin - (mb * t2m_bin + bb)
   ss_tot = np.sum((nettoa_bin - np.mean(nettoa_bin)) ** 2)
   r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
   dc.annotate(ax, f"$R^2$={r2:.3f}\nRMSE={np.sqrt(np.mean(resid ** 2)):.3f}", y=0.80)
   format_ax(ax, text=f"{label}", xscale="linear", yscale="linear")

   return {
      "F_ref": F_ref, "lam": lam, "T_eq": T_eq, "eps": float(eps),
      "T_bin": t2m_bin, "N_bin": nettoa_bin, "bin_start": start,
      "T_ann": t2m, "N_ann": nettoa, "years": years,
   }


# ---- Step 2: (kappa, h_ml) fit ----

def fit_kappa_hml(case, lmbda, fit_t, fit_T_abs, rad):
   """Joint fit of (kappa, h_ml) to the temperature series (fit_t, fit_T_abs)
   and the annual Net TOA balance. Returns dc.run_joint_fit's summary."""
   lo, hi = KAPPA_HML_BOUNDS_LO, KAPPA_HML_BOUNDS_HI
   T_ann, N_ann, years = rad["T_ann"], rad["N_ann"], rad["years"]
   F_ann = case.F_ref * ramp_frac(years, case.ramp_years)

   # Seed from a temperature-only fit; its RMSE and the annual radiative RMSE
   # set the weights of the two residuals.
   def seed_dT(t, kappa, h_ml):
      return dc.surface_dT(case, t, kappa, h_ml, rtol=dc.SENSITIVITY_RTOL, atol=dc.SENSITIVITY_ATOL)

   (kappa0, h_ml0), _ = curve_fit(
      seed_dT, fit_t, fit_T_abs, p0=[1.0e-4, 100.0],
      bounds=([10.0**lo[0], lo[1]], [10.0**hi[0], hi[1]]), max_nfev=60,
   )
   sigma_T = max(1e-2, float(np.sqrt(np.mean((seed_dT(fit_t, kappa0, h_ml0) - fit_T_abs) ** 2))))
   sigma_N = max(5e-2, float(np.sqrt(np.mean((N_ann - (F_ann - lmbda * T_ann)) ** 2))))

   data = {
      "fit_t": np.asarray(fit_t, float), "T_obs": np.asarray(fit_T_abs, float),
      "T_reg": T_ann, "N_obs": N_ann,
      "H_years": np.asarray(years, float), "bin_H": False,
      "F_frac": ramp_frac(years, case.ramp_years),
      "T_eq": float(case.T_eq), "eps": float(case.eps), "lam": float(lmbda),
      "t_final_years": float(case.t_final_years), "ramp_years": float(case.ramp_years),
      "z_max": case.z_max,
      "sigma_T": sigma_T, "sigma_N": sigma_N,
      "big": np.full(int(fit_t.shape[0] + N_ann.shape[0]), 1e3),
   }
   theta0 = np.array([
      np.log10(min(max(kappa0, 10.0**lo[0]), 10.0**hi[0])),
      min(max(h_ml0, lo[1]), hi[1]),
   ])
   return dc.run_joint_fit(theta0, data, lo, hi)


def plot_radiative_overlay(ax, ratio_ax, rad, lmbda, eps, H_annual, ramp_years):
   """On the Step 1 panel: the annual data, N = F - lambda*T and the same with
   the -(eps-1)*H term; beneath it, the ratio of the two lines."""
   T_ann, N_ann, F_ref = rad["T_ann"], rad["N_ann"], rad["F_ref"]
   N_line = F_ref * ramp_frac(rad["years"], ramp_years) - lmbda * T_ann
   N_eps = N_line - (eps - 1.0) * H_annual
   rmse_eps = float(np.sqrt(np.mean((N_ann - N_eps) ** 2)))
   order = np.argsort(T_ann)
   ax.scatter(T_ann, N_ann, s=5, alpha=0.15, color="tab:red", zorder=0, label="annual data")
   ax.plot(T_ann[order], N_line[order], color="tab:orange", lw=2, ls="-.",
           label=rf"$F-\lambda T$ (fixed): $F$={F_ref:.3f}, $\lambda$={lmbda:.3f}")
   ax.plot(T_ann[order], N_eps[order], color="green", lw=2, ls="--",
           label=rf"w/ $(\epsilon-1)H$: $\epsilon$={eps:.2f} (RMSE={rmse_eps:.3f})")
   ax.legend(loc="upper right", prop={"weight": "bold", "size": 8})

   # Mask the ratio where the denominator crosses zero (T near T_eq).
   denom = np.where(np.abs(N_line) > 0.02 * abs(F_ref), N_line, np.nan)
   ratio_ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
   ratio_ax.plot(T_ann[order], (N_eps / denom)[order], color="green", lw=1.2)
   ratio_ax.tick_params(labelsize=12, width=2, length=6, direction="in")
   ratio_ax.grid(alpha=0.3)


def plot_ohc_vs_ts(figs, i, model, label, case, lmbda, kappa_pde, h_ml_pde, t2m, nettoa):
   """Normalized OHC against T/T_eq: AOGCM years, and the fit integrated out to
   OHC_TS_EQUILIBRIUM_YEARS so it reaches (1, 1)."""
   T_eq, eps = case.T_eq, case.eps
   to_joules = EARTH_AREA * SECONDS_PER_YEAR          # annual W/m^2 -> J
   ohc = (to_joules * np.cumsum(nettoa)) / (OCEAN_HEAT_CAPACITY * NETTOA_SCALE)

   # Annual grid, so the cumulative sum of N is a time integral.
   eq_t = np.arange(1.0, OHC_TS_EQUILIBRIUM_YEARS + 1.0)
   case_eq = replace(case, t_final_years=OHC_TS_EQUILIBRIUM_YEARS)
   pde_T_eq = dc.surface_dT(case_eq, eq_t, kappa_pde, h_ml_pde) / T_eq
   H_eq = dc.heat_uptake(case_eq, eq_t, kappa_pde, h_ml_pde)
   ohc_ref = EARTH_AREA * RHO_CP * PDE_Z_MAX     # model column OHC_eq / T_eq [J/K]
   N_pde = (case.F_ref * ramp_frac(eq_t, case.ramp_years) - lmbda * (T_eq * pde_T_eq)
            - (eps - 1.0) * H_eq)
   ohc_pred = (to_joules * np.cumsum(N_pde)) / ohc_ref
   print(f"{model} OHC_eq diff-eps / reference:", ohc_ref / OCEAN_HEAT_CAPACITY)

   dc.plot_ohc_panel(figs, i, label, t2m/T_eq, ohc/T_eq, pde_T_eq, ohc_pred/T_eq)


def fit_and_plot_model(figs, i, df, layout, run_type, model, label, ramp_years, rad,
                       t2m, nettoa, n_2d, li_profiles):
   """Step 2 for one model: fit (kappa, h_ml), record it, draw every panel."""
   F_ref, lmbda, T_eq, eps = rad["F_ref"], rad["lam"], rad["T_eq"], rad["eps"]

   # Fit window by run type; the plots always cover the whole run. Temperatures
   # are carried as T/T_eq.
   fit_T = t2m[:151]/T_eq if run_type in (1, 2) else t2m/T_eq
   plot_T = t2m/T_eq
   fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
   plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
   t_final_years = max(fit_t.max(), plot_t.max())
   case = Case(F_ref, T_eq, eps, t_final_years, ramp_years, PDE_Z_MAX)

   joint = fit_kappa_hml(case, lmbda, fit_t, fit_T * T_eq, rad)
   kappa_pde, h_ml_pde = joint["kappa"], joint["h_ml"]
   dc.joint_fit_check(model, joint)
   print(f"   [joint] {model}: kappa={kappa_pde:.2e} h_ml={h_ml_pde:.0f} | "
         f"fixed F={F_ref:.2f} lam={lmbda:.2f} T_eq={T_eq:.2f} eps={eps:.3f}")
   dc.set_row(df, model, {
      "kappa_pde": kappa_pde, "h_ml_pde": h_ml_pde,
      "kappa_pde_unc": joint["kappa_err"], "h_ml_pde_unc": joint["h_ml_err"],
      **dc.joint_diagnostics(joint),
   })

   # Radiative diagnostics: annual rows on the Step 1 panel, 50-yr bins elsewhere.
   H_annual = dc.heat_uptake(case, rad["years"], kappa_pde, h_ml_pde)
   H_bin = np.array([H_annual[j:j + 50].mean()
                     for j in range(rad["bin_start"], H_annual.shape[0], 50)])
   plot_radiative_overlay(figs.step1.axs[i], figs.step1.ratio_axs[i], rad, lmbda, eps,
                          H_annual, ramp_years)
   dc.plot_uptake_panels(figs, i, model, label, rad["T_bin"], rad["N_bin"], H_bin, True,
                         F_ref, lmbda, eps, ramp_years)

   dc.plot_surface_budget_bars(
      case, model, kappa_pde, h_ml_pde,
      layout.reserve(BUDGET, figname(EXPT, model, "surface_budget_bars")))
   if model == "ECHAM5" and li_profiles:
      plot_ocean_temperature_profiles(
         case, model, kappa_pde, h_ml_pde,
         layout.reserve(BUDGET, figname(EXPT, model, "ocean_temp_profiles")))

   pde_T = dc.surface_dT(case, plot_t, kappa_pde, h_ml_pde) / T_eq
   H_plot = dc.heat_uptake(case, plot_t, kappa_pde, h_ml_pde)

   # h_ml -> 0 on a semi-infinite column (eps = 1): the numerical counterpart
   # of Held's analytic step response below. The one solve not on PDE_Z_MAX.
   pde_p_h0 = PDEParams(
      kappa=kappa_pde, h_ml=dc.H_ML_PURELY_DIFFUSIVE, dT_eq=T_eq, F0=F_ref,
      t_final=t_final_years * YEAR, z_max=PDE_Z_MAX,
      Nz=dc.pde_grid(PDE_FIT_DZ_TARGET, kappa_pde, t_final_years, PDE_Z_MAX)[1],
      semi_infinite=True, forcing=dc.ramp_forcing(F_ref, ramp_years))
   sol_h0 = pde_solve_model(pde_p_h0, n_save=250)
   pde_T_h0 = np.interp(plot_t, sol_h0["t"] / YEAR, sol_h0["dT"]) / T_eq

   plot_ohc_vs_ts(figs, i, model, label, case, lmbda, kappa_pde, h_ml_pde, t2m, nettoa)
   dc.plot_sensitivity_panels(
      figs, i, label, case, kappa_pde, h_ml_pde, lmbda, plot_t, plot_T, fit_t, fit_T,
      pde_T, H_plot, kappa_factor_2d=KAPPA_FACTOR_RANGE_2D, hml_factor_2d=HML_FACTOR_RANGE_2D,
      n_2d=n_2d)

   # Held's diffusive timescale T = kappa*rho_cp^2/lambda^2 and depth H = kappa*rho_cp/lambda.
   T_held = kappa_pde * RHO_CP**2 / lmbda**2 / YEAR
   dc.plot_temperature_panel(
      figs, i, label, plot_t, plot_T, pde_T, T_eq,
      note=(rf"$\kappa_{{1b+d}}$ = {kappa_pde:.1e} m^2/s, h$_{{ml}}$ = {h_ml_pde:.0f} m, "
            rf"$\epsilon$ = {eps:.2f}" + "\n" +
            rf"$\mathcal{{T}}$ = {T_held:.0f} yrs, $\mathcal{{H}}$ = {kappa_pde * RHO_CP / lmbda:.0f} m"),
      analytic=1-erfcx(np.sqrt(plot_t/T_held)) if ramp_years <= 0 else None,
      pde_T_h0=pde_T_h0)
   dc.plot_nettoa_panel(figs, i, label, case, lmbda, plot_t, plot_T, pde_T, H_plot, nettoa)


# ---- ocean temperature profiles (--li) ----

def load_li2012_fig3():
   """Digitized Li et al. (2012) Fig. 3 as {depth [m]: (years, dT [K])}, or
   None if extract_li2012_fig3.py has not written the CSV."""
   if not LI2012_FIG3_CURVES_CSV.exists():
      return None
   ref = pd.read_csv(LI2012_FIG3_CURVES_CSV)
   tol = 0.01 * PROFILE_AOGCM_COLUMN_Z
   out = {}
   for depth in PROFILE_DEPTHS_M:
      rows = ref[np.abs(ref["depth_m"] - depth) <= tol].sort_values("year")
      if rows.empty:
         raise ValueError(f"{LI2012_FIG3_CURVES_CSV} has no rows within "
                          f"{tol:.0f} m of the {depth:.0f} m level "
                          f"(PROFILE_DEPTHS_M)")
      out[depth] = (rows["year"].to_numpy(), rows["dT_K"].to_numpy())
   return out


def plot_ocean_temperature_profiles(case, model, kappa, h_ml, png_path):
   """theta(z, t) of the best-fit solve at PROFILE_DEPTHS_M, one panel per
   depth, with the Li et al. (2012) curves overlaid.

   Each depth is drawn at the nearest solver level below the surface slab.
   Depths below the column floor show only the reference.
   """
   _, sol = dc.solve(case, kappa, h_ml)
   t_yr, z, theta = sol["t"] / YEAR, sol["z"], sol["theta"]
   z_max = z[-1]
   ref = load_li2012_fig3()

   fig, axs = plt.subplots(len(PROFILE_DEPTHS_M), 1, figsize=(11, 14),
                           sharex=True, layout="constrained")
   for k, (ax, depth) in enumerate(zip(axs, PROFILE_DEPTHS_M)):
      below_floor = depth > z_max
      if below_floor:
         ax.set_facecolor("0.93")
         ax.text(0.5, 0.5, f"below the model column floor "
                           f"(z$_{{max}}$ = {z_max:.0f} m)\nno diff-eps temperature at this depth",
                 transform=ax.transAxes, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="0.35")
         tag = f"z = {depth:.0f} m  (not in column)"
      else:
         # z[0] is the mixed-layer slab (the surface temperature), not an ocean level.
         iz = 1 + int(np.argmin(np.abs(z[1:] - depth)))
         ax.plot(t_yr, theta[iz], color="#1f77b4", lw=2)
         tag = f"z = {depth:.0f} m  (model level {z[iz]:.0f} m)"
      if ref is not None:
         yr_ref, dT_ref = ref[depth]
         span = yr_ref <= LI2012_FIG3_YEAR
         ax.plot(yr_ref[span], dT_ref[span], color="red", lw=1.6,
                 label=f"Li et al. (2012) Fig. 3 (to yr {LI2012_FIG3_YEAR:.0f})")
         # model / reference at LI2012_FIG3_YEAR, if the run reaches it
         if not below_floor and t_yr[-1] >= LI2012_FIG3_YEAR:
            model_at = float(np.interp(LI2012_FIG3_YEAR, t_yr, theta[iz]))
            li_at = float(np.interp(LI2012_FIG3_YEAR, yr_ref, dT_ref))
            tag += f"\ndiff-eps/Li = {model_at / li_at:.2f}"
      format_ax(ax, text=tag, ylim=(0, 10),
                legend=(ref is not None and k == 0), legend_loc="lower right")
   axs[-1].set_xlim(t_yr[0], t_yr[-1])
   axs[-1].set_xlabel("Time (years)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
   fig.suptitle("Global Mean Ocean Temperature Change [K]",
                fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
   fig.savefig(str(png_path), dpi=DPI_WIDE, bbox_inches="tight")
   plt.close(fig)


# ---- main ----

def main():
   cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
   add_test_argument(cli)
   cli.add_argument("--high-res", action="store_true",
                    help=f"{N_2D_HIGH_RES}x{N_2D_HIGH_RES} sensitivity heatmaps instead of "
                         f"{dc.SENSITIVITY_N_2D}x{dc.SENSITIVITY_N_2D} (much slower).")
   cli.add_argument("--li", action="store_true",
                    help="Also plot the ECHAM5 ocean temperature profiles with the "
                         "Li et al. (2012) Fig. 3 overlay.")
   args = cli.parse_known_args()[0]
   n_2d = N_2D_HIGH_RES if args.high_res else dc.SENSITIVITY_N_2D

   dc.start_pool()
   data, expts = load_longrun_data()
   fit_models = select_models(MODELS, args.test)
   announce_test_mode(fit_models, MODELS, args.test)
   progress = dc.Progress(len(RUN_TYPES) * len(fit_models))

   for run_type in RUN_TYPES:
      layout = RunLayout(VARIANT, run_type)
      print("==================================")
      print(f"Output: {layout.label}")
      print("==================================")
      df = pd.DataFrame(columns=PARAM_COLS)
      df["model"] = fit_models
      # Grids are sized to the full model list; --test fills the first panels.
      figs = dc.make_figures(MODELS, TITLE_PREFIX, r"$T_s/T_{eq}$",
                             auto_rows=True, hide_unused=True)

      meta, rad = {}, {}
      for i, model in enumerate(fit_models):
         model_data = data.rx2(model)
         co2_multiple, ramp_years = model_meta(model_data)
         meta[model] = (model_label(model, co2_multiple, ramp_years), ramp_years)
         t2m, nettoa = perturbed_anomalies(model, model_data, expts, NETTOA_SCALE)
         rad[model] = fit_radiative(figs.step1.axs[i], df, model, meta[model][0],
                                    t2m, nettoa, ramp_years)
      print("Finished Step 1: Gregory params to df (Step 1 figures saved after Step 2)")

      for i, model in enumerate(fit_models):
         label, ramp_years = meta[model]
         t2m, nettoa = perturbed_anomalies(model, data.rx2(model), expts, NETTOA_SCALE,
                                              drop_nan_nettoa=False)
         fit_and_plot_model(figs, i, df, layout, run_type, model, label, ramp_years,
                            rad[model], t2m, nettoa, n_2d, args.li)
         progress.step(run_type, model)

      df.to_csv(layout.csv_path("fitted_model_params"), index=False)
      dc.finish_figures(figs, layout)
      print("Finished final val/result plots")

   dc.close_pool()


if __name__ == "__main__":
   main()
