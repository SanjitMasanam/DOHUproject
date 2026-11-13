#!/usr/bin/env python3
"""Normalized ocean heat uptake against normalized warming for the Li et al.
(2012) ECHAM5/MPIOM 4xCO2 run, next to LongRunMIP's ECHAM5 abrupt-4xCO2 run and
the ECHAM5 fit of the diffusive-eps-joint_fixedRadiativeParams variant.

    x = T / T_eq        y = OHC / OHC_eq = A * int N dt / (C_ocean * T_eq)

Reads data/li2012_fig1_N_vs_T.csv (extract_li2012_fig1.py) and the variant's
parameter table for one run type (default 3). Two figures in figures/li2012/:

  li2012_fig1_OHU_vs_T.png          Li data / T_eq of its transient Gregory fit
                                    (140-1200 yr); LongRunMIP data and the fit
                                    / the fit's own T_eq. The fit is drawn as
                                    fitted (step forcing) and under Li's 140-yr
                                    ramp, solid over the run and dashed out to
                                    OHC_TS_EQUILIBRIUM_YEARS.
  li2012_fig1_OHU_vs_T_quasiEq.png  everything / T_eq of Li's quasi-equilibrium
                                    fit (1200-4600 yr). The fit curves come from
                                    a refit with T_eq pinned to it (F_ref and
                                    eps unchanged, lambda = F_ref/T_eq); nothing
                                    is written back to the table.

Run from the repo root:  python3 li2012_ohu_vs_T.py [--run-type 3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit, least_squares

import diffusive_common as dc
from compare_ohc_ts_across_models import load_aogcm_series
from diffusive_common import (
   JOINT_FD_REL_STEP, JOINT_FTOL, JOINT_MAX_NFEV, JOINT_XTOL,
   KAPPA_HML_BOUNDS_HI as BOUNDS_HI, KAPPA_HML_BOUNDS_LO as BOUNDS_LO, NETTOA_SCALE,
   OHC_TS_EQUILIBRIUM_YEARS, PDE_Z_MAX, Case,
)
from extract_li2012_fig1 import PERIOD_YEARS, gregory_fit
from figure_paths import RUN_TYPE_DIRS, params_csv, topic_dir
from longrun_common import EARTH_AREA, OCEAN_HEAT_CAPACITY, SECONDS_PER_YEAR
from pde_solver_1lyrEBM_diffusive import solve_model as pde_solve_model, YEAR

VARIANT = "diffusive-eps-joint_fixedRadiativeParams"
MODEL = "ECHAM5"
TRANSIENT = "transient (140-1200 yr)"
QUASI_EQ = "quasi-equilibrium (1200-4600 yr)"
RAMP_YEARS = 140.0                 # Li et al.: CO2 +1 %/yr for 140 yr (4.0x), then held

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


def fit_curve(row, run_years, ramp_years=0.0):
   """(t [yr], T [K], OHC/C [K]) of the fitted PDE under step forcing or, with
   ramp_years, forcing rising linearly to F_ref over ramp_years.

   Annual output over the run, then log-spaced out to OHC_TS_EQUILIBRIUM_YEARS.
   OHC is the time integral of N = F(t) - lambda*T - (eps-1)*H.
   """
   case = Case(float(row["F_ref"]), float(row["T_eq"]), float(row["epsilon"]),
               OHC_TS_EQUILIBRIUM_YEARS, ramp_years, PDE_Z_MAX)
   p = dc.pde_params(case, float(row["kappa_pde"]), float(row["h_ml_pde"]))
   t_yr = np.concatenate((np.arange(0.0, run_years + 1.0),
                          np.geomspace(run_years + 1.0, OHC_TS_EQUILIBRIUM_YEARS, 300)))
   sol = pde_solve_model(p, t_eval=t_yr * YEAR)
   N = (case.F_ref * dc.ramp_frac(t_yr, ramp_years) - p.lam * sol["dT"]
        - (case.eps - 1.0) * dc.surface_uptake(p, sol))
   ohc = cumulative_trapezoid(N, t_yr * SECONDS_PER_YEAR, initial=0.0)   # J m^-2
   return t_yr, sol["dT"], ohc / (p.rho_cp * PDE_Z_MAX)


def refit_fixed_teq(row, lrm, T_eq):
   """Refit (kappa, h_ml) to the LongRunMIP abrupt run with T_eq pinned: F_ref
   and eps stay at the row's values and lambda = F_ref/T_eq. Same residual as
   the fit script's Step 2, solved serially. Returns a copy of `row` carrying
   the refit parameters."""
   F_ref, eps = float(row["F_ref"]), float(row["epsilon"])
   lam = F_ref / T_eq
   fit_t, T_obs, N_obs = lrm.t, lrm.T, lrm.N * NETTOA_SCALE
   case = Case(lam * T_eq, T_eq, eps, float(fit_t.max()), 0.0, PDE_Z_MAX)

   def forward(theta):   # (T, H) at fit_t, or None
      return dc.joint_forward(case, 10.0 ** theta[0], theta[1], fit_t, fit_t, False)

   # Seed from a temperature-only fit; its RMSE and the annual Gregory RMSE
   # weight the two residuals.
   def seed_T(t, kappa, h_ml):
      out = forward([np.log10(kappa), h_ml])
      return np.full(np.shape(t), 1e6) if out is None else out[0]
   (kappa0, h_ml0), _ = curve_fit(seed_T, fit_t, T_obs, p0=[1.0e-4, 100.0], max_nfev=60,
                                  bounds=([10.0 ** BOUNDS_LO[0], BOUNDS_LO[1]],
                                          [10.0 ** BOUNDS_HI[0], BOUNDS_HI[1]]))
   sigma_T = max(1e-2, float(np.sqrt(np.mean((seed_T(fit_t, kappa0, h_ml0) - T_obs) ** 2))))
   sigma_N = max(5e-2, float(np.sqrt(np.mean((N_obs - (F_ref - lam * T_obs)) ** 2))))

   def residuals(theta):
      out = forward(theta)
      if out is None:
         return np.full(2 * fit_t.size, 1e3)
      T_model, H = out
      N_model = lam * (T_eq - T_obs) - (eps - 1.0) * H
      return np.concatenate([(T_model - T_obs) / sigma_T, (N_model - N_obs) / sigma_N])

   theta0 = np.clip([np.log10(kappa0), h_ml0], BOUNDS_LO + 1e-9, BOUNDS_HI - 1e-9)
   sol = least_squares(residuals, theta0, bounds=(BOUNDS_LO, BOUNDS_HI),
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
   """One OHC-vs-T panel. The Li data are divided by T_li (named li_norm in
   labels); the LongRunMIP data and both fit curves by T_model, or by the fit's
   own T_eq when T_model is None. `fits` come from `row`; orig_row is the
   stored fit `row` was refit from, quoted in the text box."""
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

   # The step and ramp curves coincide after the ramp, so the step curve is
   # drawn wider beneath the ramp curve.
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
   fig.savefig(out, dpi=160, bbox_inches="tight")
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
   args = ap.parse_args()
   out_dir = topic_dir("li2012")
   out, out_qe = out_dir / "li2012_fig1_OHU_vs_T.png", out_dir / "li2012_fig1_OHU_vs_T_quasiEq.png"

   li = load_li2012()
   years, _dT, _N, T_eqs = li
   lrm = load_aogcm_series()[MODEL]
   table = pd.read_csv(params_csv(VARIANT, args.run_type))
   row = table.loc[table["model"] == MODEL].iloc[0]
   fits = (fit_curve(row, years[-1], ramp_years=RAMP_YEARS), fit_curve(row, lrm.t[-1]))

   T_qe = T_eqs[QUASI_EQ]
   refit = refit_fixed_teq(row, lrm, T_qe)
   fits_qe = (fit_curve(refit, years[-1], ramp_years=RAMP_YEARS), fit_curve(refit, lrm.t[-1]))
   ends = plot(li, lrm, fits, row, args.run_type, out, T_eqs[TRANSIENT], "transient-fit")
   ends.update(plot(li, lrm, fits_qe, refit, args.run_type, out_qe,
                    T_qe, "quasi-equilibrium-fit", T_qe, "Li quasi-equilibrium-fit",
                    fit_name=rf"refit ($T_{{eq}}$ pinned to {T_qe:.2f} K)", orig_row=row))

   print(f"Li et al. (2012): T_eq = {T_eqs[TRANSIENT]:.2f} K (transient fit), "
         f"{T_qe:.2f} K (quasi-equilibrium fit)")
   print(f"{MODEL} fit ({RUN_TYPE_DIRS[args.run_type]}): T_eq = {row['T_eq']:.2f} K; "
         f"LongRunMIP abrupt run {lrm.t[-1]:.0f} yr")
   for name, (x, y) in ends.items():
      print(f"   at yr {years[-1]:.0f}, {name:44s} T/T_eq = {x:.3f}, OHC/OHC_eq = {y:.3f}")
   print(f"-> {out}")
   print(f"-> {out_qe}")


if __name__ == "__main__":
   main()
