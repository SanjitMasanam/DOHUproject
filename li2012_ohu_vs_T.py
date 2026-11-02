#!/usr/bin/env python3
"""Normalized ocean heat uptake against normalized warming for the Li et al.
(2012) ECHAM5/MPIOM 4xCO2 run, next to LongRunMIP's ECHAM5 abrupt-4xCO2 run and
the ECHAM5 diffusive-eps fit.

The companion of figures/li2012_fig1_N_vs_T.png: same digitized series
(data/li2012_fig1_N_vs_T.csv, written by extract_li2012_fig1.py), drawn in the
OHC-vs-T_s plane of the fit scripts' `ohc_ts` panel rather than the Gregory
plane:

    x = T / T_eq        y = OHC / OHC_eq = A * int N dt / (C_ocean * T_eq)

with A = 5.1e14 m^2 and C_ocean = 1.37e21 kg * 3850 J kg^-1 K^-1 the full-ocean
heat capacity, exactly the normalization the fit script applies to the
LongRunMIP data. For the Li data T_eq is the N = 0 intercept of the Gregory fit
over the paper's transient period (140-1200 yr), the same fit and number the
N-vs-T figure draws (~12.1 K).

The fit is ECHAM5's row of the diffusive-eps-joint_fixedRadiativeParams table
for one run type (default 03_lrfit-50yravg); nothing is refit. It is drawn two
ways:

  * as fitted: the abrupt-4xCO2 step response, against the LongRunMIP run it
    was fitted to (gray), both normalized by the fit's own T_eq, i.e. the green
    curve and scatter of the fit script's ohc_ts ECHAM5 panel (solid purple);
  * under Li et al.'s scenario, CO2 +1 %/yr to 4x over 140 yr and then held:
    forcing ~ ln(CO2) rises linearly to F_ref at year 140, with the fitted
    (kappa, h_ml, eps, lambda) unchanged, also normalized by the fit's own
    T_eq (green, drawn narrower on top of a wider purple: the two coincide
    once the ramp is over).

Each fit curve is solid over its run's length and dashed beyond, out to
OHC_TS_EQUILIBRIUM_YEARS, so both end at (1, 1).

A second figure (--out-qe) replicates the first with EVERY series -- Li data,
LongRunMIP data and both fit curves -- divided by one T_eq instead: the N = 0
intercept of the Gregory fit over Li's quasi-equilibrium period (1200-4600 yr,
~11.2 K), the other fit the N-vs-T figure draws. The stored fit's own T_eq
(12.0 K) is warmer than that and would run past (1, 1) on this scale, so the
fit curves there come from a REFIT pinned to it (refit_fixed_teq): the fit
script's STEP-2 joint fit of (kappa, h_ml) to the LongRunMIP abrupt run, re-run
with F_ref and eps unchanged and lambda = F_ref/11.2 K. It is restated here
because the fit script cannot be imported; at the stored T_eq it reproduces the
stored kappa/h_ml to within their 1-sigma errors. The refit is used for this
figure only -- nothing is written back to the parameter table.

The PDE settings (closed PDE_Z_MAX column, PDE_FIT_DZ_TARGET spacing, model
heat-content reference rho_cp*z_max, the (eps-1)*H term in N) restate the fit
script's, which cannot be imported (see CLAUDE.md). Its F_ref is in the
NETTOA_SCALE-scaled units the fit used, and its heat-content reference is the
matching 3700 m column, so the curves' y is on the same scale as the data's.

Run from the repo root (needs the science env: extract_li2012_fig1 pulls in
pymupdf, compare_ohc_ts_across_models rpy2):

    python3 li2012_ohu_vs_T.py [--run-type 3] [--out figures/li2012_fig1_OHU_vs_T.png]
                               [--out-qe figures/li2012_fig1_OHU_vs_T_quasiEq.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from compare_ohc_ts_across_models import load_aogcm_series
from extract_li2012_fig1 import PERIOD_YEARS, gregory_fit
from figure_paths import RUN_TYPE_DIRS, params_csv
from pde_solver_1lyrEBM_diffusive import Params as PDEParams, solve_model as pde_solve_model, YEAR

VARIANT = "diffusive-eps-joint_fixedRadiativeParams"
MODEL = "ECHAM5"
TRANSIENT = "transient (140-1200 yr)"
QUASI_EQ = "quasi-equilibrium (1200-4600 yr)"

# Li et al. (2012) scenario: CO2 +1 %/yr for 140 yr (1.01^140 = 4.0x), then held.
RAMP_YEARS = 140.0

# Restated from longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py.
PDE_Z_MAX = 3700.0                 # m, closed insulated floor
PDE_FIT_DZ_TARGET = 15.0           # m
OHC_TS_EQUILIBRIUM_YEARS = 100000.0
NETTOA_SCALE = PDE_Z_MAX / 2500.0  # LongRunMIP N -> the 3700 m column's units (F_ref's)
# ... and its STEP-2 joint fit, for refit_fixed_teq.
PDE_FIT_N_SAVE = 250
SENSITIVITY_RTOL, SENSITIVITY_ATOL = 1e-6, 1e-8
JOINT_MAX_NFEV, JOINT_FTOL, JOINT_XTOL, JOINT_FD_REL_STEP = 200, 1e-8, 1e-8, 1e-6
JOINT_BOUNDS_LO = np.array([np.log10(1e-7), 1e-3])    # [log10(kappa), h_ml]
JOINT_BOUNDS_HI = np.array([np.log10(1e-3), 300.0])

EARTH_AREA = 5.1e14                # m^2
SECONDS_PER_YEAR = 31536000.0
OCEAN_HEAT_CAPACITY = 1.37e21 * 3850.0   # J K^-1, full ocean

YEAR_COLOR_MAX = 6000              # turbo colorbar range, as in the ohc_ts panel
C_RAMP, C_STEP, C_LRM = "green", "tab:purple", "0.6"


def ohc_kelvin(N):
   """A * cumsum(N) * 1 yr / C_ocean [K] for an annual N series [W m^-2]."""
   return EARTH_AREA * SECONDS_PER_YEAR * np.cumsum(N) / OCEAN_HEAT_CAPACITY


def load_li2012(csv_path=Path("data/li2012_fig1_N_vs_T.csv")):
   """(years, dT, N, {period: T_eq}) of the digitized run, T_eq from the
   transient and quasi-equilibrium Gregory fits."""
   df = pd.read_csv(csv_path)
   years, dT, N = df["year"].to_numpy(float), df["dT_K"].to_numpy(float), df["N_W_m2"].to_numpy(float)
   T_eqs = {label: gregory_fit(years, dT, N, *PERIOD_YEARS[label])[2] for label in (TRANSIENT, QUASI_EQ)}
   return years, dT, N, T_eqs


def fit_curve(row, run_years, ramp_years=None):
   """(t [yr], T [K], OHC/C [K]) of the fitted PDE, abrupt forcing or, with
   ramp_years, forcing rising linearly to F_ref over ramp_years then held.

   Annual output over the run (so the integral of N is sampled like the data),
   then log-spaced out to OHC_TS_EQUILIBRIUM_YEARS. N = F(t) - lam*T - (eps-1)*H
   with H the diffusive flux out of the slab, as in the fit script's ohc_ts
   panel; its time integral is the model ocean's heat content.
   """
   kappa, h_ml = float(row["kappa_pde"]), float(row["h_ml_pde"])
   T_eq, F_ref, eps = float(row["T_eq"]), float(row["F_ref"]), float(row["epsilon"])
   ramp = (lambda t_yr: np.minimum(t_yr / ramp_years, 1.0)) if ramp_years else (lambda t_yr: np.ones_like(t_yr))
   p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref, epsilon=eps,
      z_max=PDE_Z_MAX, Nz=max(4, int(PDE_Z_MAX / PDE_FIT_DZ_TARGET + 1)),
      t_final=OHC_TS_EQUILIBRIUM_YEARS * YEAR,
      forcing=(lambda t: F_ref * min(t / (ramp_years * YEAR), 1.0)) if ramp_years else None,
   )
   t_yr = np.concatenate((np.arange(0.0, run_years + 1.0),
                          np.geomspace(run_years + 1.0, OHC_TS_EQUILIBRIUM_YEARS, 300)))
   sol = pde_solve_model(p, t_eval=t_yr * YEAR)
   theta, dz = sol["theta"], sol["dz"]
   # H > 0 = heat leaving the slab downward; 2nd-order one-sided gradient, as pde_H_at.
   H = -p.D * (-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dz)
   N = F_ref * ramp(t_yr) - p.lam * sol["dT"] - (eps - 1.0) * H
   ohc = cumulative_trapezoid(N, t_yr * SECONDS_PER_YEAR, initial=0.0)   # J m^-2
   return t_yr, sol["dT"], ohc / (p.rho_cp * PDE_Z_MAX)


def _joint_forward(theta, T_eq, eps, lam, fit_t, t_final_years):
   """(T at fit_t, H at fit_t) for theta = [log10(kappa), h_ml], or None if the
   solve fails -- the fit script's _joint_forward with annual, unbinned H."""
   p = PDEParams(
      kappa=10.0 ** theta[0], h_ml=theta[1], dT_eq=T_eq, F0=lam * T_eq, epsilon=eps,
      z_max=PDE_Z_MAX, Nz=max(4, int(PDE_Z_MAX / PDE_FIT_DZ_TARGET + 1)),
      t_final=t_final_years * YEAR,
   )
   t_eval = np.concatenate(([0.0], np.geomspace(YEAR, p.t_final, PDE_FIT_N_SAVE - 1)))
   t_eval[-1] = p.t_final
   try:
      sol = pde_solve_model(p, t_eval=t_eval, rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL)
   except Exception:
      return None
   t_yr, theta_z, dz = sol["t"] / YEAR, sol["theta"], sol["dz"]
   H = -p.D * (-3.0 * theta_z[0] + 4.0 * theta_z[1] - theta_z[2]) / (2.0 * dz)
   T_model, H_model = np.interp(fit_t, t_yr, sol["dT"]), np.interp(fit_t, t_yr, H)
   if not (np.all(np.isfinite(T_model)) and np.all(np.isfinite(H_model))):
      return None
   return T_model, H_model


def refit_fixed_teq(row, lrm, T_eq):
   """Re-run the fit script's STEP-2 joint fit of (kappa, h_ml) to the LongRunMIP
   abrupt run, with T_eq pinned: F_ref and eps stay at the row's values and
   lambda = F_ref/T_eq. Restated rather than imported (see CLAUDE.md), from
   _joint_forward / _joint_residuals / run_joint_fit and the seed + sigma setup
   that precedes them. Returns a copy of `row` carrying the refit parameters."""
   from scipy.optimize import curve_fit, least_squares
   F_ref, eps = float(row["F_ref"]), float(row["epsilon"])
   lam = F_ref / T_eq
   fit_t, T_obs, N_obs = lrm.t, lrm.T, lrm.N * NETTOA_SCALE
   t_final = float(fit_t.max())

   # Seed (kappa, h_ml) from a temperature-only fit; its RMSE sets sigma_T and
   # the annual Gregory residual sets sigma_N, the two streams' weights.
   def seed_T(t, kappa, h_ml):
      out = _joint_forward([np.log10(kappa), h_ml], T_eq, eps, lam, t, t_final)
      return np.full(np.shape(t), 1e6) if out is None else out[0]
   (kappa0, h_ml0), _ = curve_fit(seed_T, fit_t, T_obs, p0=[1.0e-4, 100.0], max_nfev=60,
                                  bounds=([10.0 ** JOINT_BOUNDS_LO[0], JOINT_BOUNDS_LO[1]],
                                          [10.0 ** JOINT_BOUNDS_HI[0], JOINT_BOUNDS_HI[1]]))
   sigma_T = max(1e-2, float(np.sqrt(np.mean((seed_T(fit_t, kappa0, h_ml0) - T_obs) ** 2))))
   sigma_N = max(5e-2, float(np.sqrt(np.mean((N_obs - (F_ref - lam * T_obs)) ** 2))))

   def residuals(theta):
      out = _joint_forward(theta, T_eq, eps, lam, fit_t, t_final)
      if out is None:
         return np.full(2 * fit_t.size, 1e3)
      T_model, H = out
      N_model = lam * (T_eq - T_obs) - (eps - 1.0) * H
      return np.concatenate([(T_model - T_obs) / sigma_T, (N_model - N_obs) / sigma_N])

   theta0 = np.clip([np.log10(kappa0), h_ml0], JOINT_BOUNDS_LO + 1e-9, JOINT_BOUNDS_HI - 1e-9)
   sol = least_squares(residuals, theta0, bounds=(JOINT_BOUNDS_LO, JOINT_BOUNDS_HI),
                       method="trf", x_scale="jac", diff_step=JOINT_FD_REL_STEP,
                       ftol=JOINT_FTOL, xtol=JOINT_XTOL, max_nfev=JOINT_MAX_NFEV)
   kappa, h_ml = 10.0 ** sol.x[0], sol.x[1]
   rmse_T = float(np.sqrt(np.mean((sol.fun[:fit_t.size] * sigma_T) ** 2)))
   rmse_N = float(np.sqrt(np.mean((sol.fun[fit_t.size:] * sigma_N) ** 2)))
   print(f"   refit at T_eq = {T_eq:.2f} K (lambda = {lam:.3f}, F_ref = {F_ref:.2f}, eps = {eps:.3f}): "
         f"kappa = {kappa:.3e} m^2/s, h_ml = {h_ml:.2f} m | success = {sol.success} "
         f"(status {sol.status}, nfev {sol.nfev}) | RMSE_T = {rmse_T:.4f} K, RMSE_N = {rmse_N:.4f} W/m^2")

   refit = row.copy()
   refit["T_eq"], refit["lambda"], refit["kappa_pde"], refit["h_ml_pde"] = T_eq, lam, kappa, h_ml
   refit["joint_rmse_T"], refit["joint_rmse_N"] = rmse_T, rmse_N
   return refit


def draw_fit(ax, t, T, ohc, T_eq, run_years, color, label, lw=2.2, zorder=3):
   """Solid over the run, thinner and dashed beyond it."""
   run = t <= run_years
   ax.plot(T[run] / T_eq, ohc[run] / T_eq, color=color, lw=lw, zorder=zorder, label=label)
   ax.plot(T[~run] / T_eq, ohc[~run] / T_eq, color=color, ls="--", lw=0.6 * lw, alpha=0.8, zorder=zorder)


def plot(li, lrm, fits, row, run_type, out, T_li, li_norm, T_model=None, model_norm=None,
         fit_name="fit", orig_row=None):
   """One OHC-vs-T panel. The Li data are divided by T_li (described in labels
   as li_norm); the LongRunMIP data and both fit curves by T_model, or by the
   fit's own T_eq when T_model is None. `fits` come from `row`'s parameters;
   orig_row, when given, is the stored fit `row` was refit from, quoted in the
   text box for comparison."""
   years, dT, N, _T_eqs = li
   T_fit = float(row["T_eq"])
   if T_model is None:
      T_model, model_norm = T_fit, "own"
   ohc_li = ohc_kelvin(N)
   (t_ramp, T_ramp, ohc_ramp), (t_step, T_step, ohc_step) = fits
   ohc_lrm = ohc_kelvin(lrm.N)

   fig, ax = plt.subplots(figsize=(9, 8.2), layout="constrained")
   ax.axvline(1.0, color="0.55", ls="--", lw=0.8)
   ax.axhline(1.0, color="0.55", ls="--", lw=0.8)

   # Data: LongRunMIP abrupt run beneath, Li et al. ramp run on top.
   ax.scatter(lrm.T / T_model, ohc_lrm / T_model, s=5, color=C_LRM, zorder=1,
              label=(rf"LongRunMIP {MODEL} abrupt 4xCO$_2$ ({lrm.t[-1]:.0f} yr), "
                     rf"/ {'fit' if model_norm == 'own' else model_norm} $T_{{eq}}$ = {T_model:.1f} K"))
   sc = ax.scatter(dT / T_li, ohc_li / T_li, c=years, s=6, cmap=plt.cm.turbo,
                   norm=mpl.colors.Normalize(vmin=0, vmax=YEAR_COLOR_MAX), zorder=2,
                   label=rf"Li et al. 2012 (1 %/yr ramp), / {li_norm} $T_{{eq}}$ = {T_li:.1f} K")

   # Fits: step response as fitted, and the same parameters under Li's ramp.
   # The two coincide once the ramp is over, so the step curve is drawn wider
   # beneath the ramp curve: it shows as a border on either side of it.
   draw_fit(ax, t_step, T_step, ohc_step, T_model, lrm.t[-1], C_STEP,
            rf"{fit_name}: abrupt 4xCO$_2$ (as fitted), / {model_norm} $T_{{eq}}$ = {T_model:.2f} K",
            lw=5.0, zorder=3)
   draw_fit(ax, t_ramp, T_ramp, ohc_ramp, T_model, years[-1], C_RAMP,
            rf"{fit_name} under Li's {RAMP_YEARS:.0f}-yr ramp, / {model_norm} $T_{{eq}}$ = {T_model:.2f} K",
            lw=2.6, zorder=4)

   end_ramp = np.flatnonzero(t_ramp <= years[-1])[-1]
   ax.plot([T_ramp[end_ramp] / T_model], [ohc_ramp[end_ramp] / T_model], marker="*", ms=13,
           color=C_RAMP, mec="black", zorder=5, label=f"ramp fit at yr {years[-1]:.0f}")
   ax.plot([dT[-1] / T_li], [ohc_li[-1] / T_li], marker="*", ms=13, color="black",
           zorder=5, label=f"Li et al. end of run (yr {years[-1]:.0f})")

   cbar = fig.colorbar(sc, ax=ax, pad=0.02)
   cbar.set_label("Year (Li et al. scatter)", fontsize=12)
   ax.set_xlim(-0.02, 1.12)
   ax.set_ylim(-0.05, 1.2)
   ax.set_xlabel(r"$T_s/T_{eq}$", fontsize=14, fontweight="bold")
   ax.set_ylabel(r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$", fontsize=14, fontweight="bold")
   ax.set_title(f"Normalized ocean heat uptake against warming, {MODEL} 4xCO$_2$\n"
                f"(Li et al. 2012 Fig. 1a/1c, LongRunMIP, and the {RUN_TYPE_DIRS[run_type]} "
                "diffusive-eps fit)",
                fontsize=15, fontweight="bold")
   ax.grid(alpha=0.3)
   ax.tick_params(labelsize=12)
   def params(r):
      return (f"$\\kappa$ = {r['kappa_pde']:.2e} m$^2$ s$^{{-1}}$, $h_{{ml}}$ = {r['h_ml_pde']:.1f} m, "
              f"$\\lambda$ = {r['lambda']:.3f}, $\\varepsilon$ = {r['epsilon']:.2f}\n"
              f"   RMSE$_T$ = {r['joint_rmse_T']:.3f} K, RMSE$_N$ = {r['joint_rmse_N']:.3f} W m$^{{-2}}$")
   box = f"{fit_name}:\n   {params(row)}"
   if orig_row is not None:
      box += f"\nstored fit ($T_{{eq}}$ = {orig_row['T_eq']:.2f} K):\n   {params(orig_row)}"
   box += f"\ndashed: fit continued past its run to {OHC_TS_EQUILIBRIUM_YEARS:,.0f} yr"
   ax.text(0.01, 0.99, box, transform=ax.transAxes, va="top", fontsize=8.5)
   ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
   out.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(str(out), dpi=160, bbox_inches="tight")
   plt.close(fig)

   return {
      f"Li, / {li_norm} T_eq": (dT[-1] / T_li, ohc_li[-1] / T_li),
      f"ramp fit, / {model_norm} T_eq": (T_ramp[end_ramp] / T_model, ohc_ramp[end_ramp] / T_model),
   }


def main():
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--run-type", type=int, default=3, choices=sorted(RUN_TYPE_DIRS),
                   help="which run type's fitted parameters to draw (default 3)")
   ap.add_argument("--out", default="figures/li2012_fig1_OHU_vs_T.png",
                   help="output PNG: Li data / transient T_eq, everything else / the fit's own")
   ap.add_argument("--out-qe", default="figures/li2012_fig1_OHU_vs_T_quasiEq.png",
                   help="output PNG: everything / Li's quasi-equilibrium T_eq")
   args = ap.parse_args()

   li = load_li2012()
   years, _dT, _N, T_eqs = li
   lrm = load_aogcm_series()[MODEL]
   table = pd.read_csv(params_csv(VARIANT, args.run_type))
   row = table.loc[table["model"] == MODEL].iloc[0]
   fits = (fit_curve(row, years[-1], ramp_years=RAMP_YEARS), fit_curve(row, lrm.t[-1]))

   T_qe = T_eqs[QUASI_EQ]
   refit = refit_fixed_teq(row, lrm, T_qe)
   fits_qe = (fit_curve(refit, years[-1], ramp_years=RAMP_YEARS), fit_curve(refit, lrm.t[-1]))
   ends = plot(li, lrm, fits, row, args.run_type, Path(args.out), T_eqs[TRANSIENT], "transient-fit")
   ends.update(plot(li, lrm, fits_qe, refit, args.run_type, Path(args.out_qe),
                    T_qe, "quasi-equilibrium-fit", T_qe, "Li quasi-equilibrium-fit",
                    fit_name=rf"refit ($T_{{eq}}$ pinned to {T_qe:.2f} K)", orig_row=row))

   print(f"Li et al. (2012): T_eq = {T_eqs[TRANSIENT]:.2f} K (transient fit), "
         f"{T_qe:.2f} K (quasi-equilibrium fit)")
   print(f"{MODEL} fit ({RUN_TYPE_DIRS[args.run_type]}): T_eq = {row['T_eq']:.2f} K; "
         f"LongRunMIP abrupt run {lrm.t[-1]:.0f} yr")
   for name, (x, y) in ends.items():
      print(f"   at yr {years[-1]:.0f}, {name:44s} T/T_eq = {x:.3f}, OHC/OHC_eq = {y:.3f}")
   print(f"-> {args.out}")
   print(f"-> {args.out_qe}")


if __name__ == "__main__":
   main()
