#!/usr/bin/env python3
"""Slab + diffusive thermocline with efficacy: one joint fit of
(kappa, h_ml, eps, lambda).

Step 1 fits a Gregory line to Net TOA vs T for F_ref, lambda and T_eq. Step 2
fits [log10(kappa), h_ml, eps, lambda] by least squares against the
temperature series and the Net TOA balance N = F - lambda*T - (eps-1)*H, with
T_eq held at its Step 1 value (F = lambda*T_eq).

Models: the 8 LongRunMIP abrupt-4xCO2 runs. Run types:
   1  Step 1 on annual years 1-150; temperature fit over years 1-151
   2  Step 1 on 50-yr bins of the whole run; temperature fit over years 1-151
   3  Step 1 on 50-yr bins; temperature fit over the whole run

Flags: --test [N | M1,M2].
Output: figures/diffusive-eps-joint/<run type>/.
"""
import argparse

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit

import diffusive_common as dc
from diffusive_common import EXPT, PDE_FIT_DZ_TARGET, RHO_CP, Case
from figure_paths import BUDGET, RunLayout, figname
from longrun_common import (
   EARTH_AREA, MODELS, OCEAN_HEAT_CAPACITY, SECONDS_PER_YEAR, add_test_argument,
   announce_test_mode, load_longrun_data, perturbed_anomalies, select_models, sympy_prop_unc,
)
from plot_style import format_ax

VARIANT = "diffusive-eps-joint"
RUN_TYPES = [3, 2, 1]
TITLE_PREFIX = r"4xCO$_{2}$"

# Bounds on theta = [log10(kappa), h_ml, eps, lambda].
BOUNDS_LO = np.array([np.log10(1e-7),  100, 0.0, 0.05])
BOUNDS_HI = np.array([np.log10(1e-3), 300.0, 3.0, 5.00])

PARAM_COLS = [
   "model", "F_ref", "lambda", "T_eq", "epsilon", "kappa_pde", "h_ml_pde",
   "F_ref_unc", "lambda_unc", "T_eq_unc", "epsilon_unc",
]


# ---- Step 1: Gregory regression ----

def fit_gregory(ax, df, model, t2m, nettoa, run_type):
   """Fit N = F_ref - lambda*T and draw the panel. Run types 2 and 3 fit 50-yr
   bins (first bin dropped). Returns the regression rows Step 2 reuses."""
   n_annual = t2m.shape[0]
   if run_type != 1:
      t2m = np.array([np.mean(t2m[j:j + 50]) for j in range(50, n_annual, 50)])
      nettoa = np.array([np.mean(nettoa[j:j + 50]) for j in range(50, n_annual, 50)])

   [m, b], cov = np.polyfit(t2m, nettoa, 1, cov=True)
   m_unc = np.sqrt(cov[0, 0])
   b_unc = np.sqrt(cov[1, 1])
   sp_m, sp_b = sp.symbols("m b")
   dc.set_row(df, model, {
      "F_ref": b, "lambda": -m, "T_eq": b / (-m),
      "F_ref_unc": float(b_unc), "lambda_unc": float(m_unc),
      "T_eq_unc": sympy_prop_unc(sp_b / (-sp_m), {sp_m: m, sp_b: b},
                                 {sp_m: m_unc, sp_b: b_unc}),
   })

   xfit = np.linspace(t2m.min(), t2m.max(), 100)
   ax.scatter(t2m, nettoa, s=8, alpha=0.5, label="Data")
   ax.plot(xfit, m * xfit + b, linewidth=2, label=f"Step 1 Fit: F_ref={b:.3f}, -lambda={m:.3f}")
   resid = nettoa - (m * t2m + b)
   ss_tot = np.sum((nettoa - np.mean(nettoa)) ** 2)
   r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
   dc.annotate(ax, f"$R^2$={r2:.3f}\nRMSE={np.sqrt(np.mean(resid ** 2)):.3f}")
   format_ax(ax, text=f"{model}", xscale="linear", yscale="linear")

   return {
      "F_ref": b, "lam": -m, "T_eq": b / (-m),
      "T_reg": t2m, "N_reg": nettoa,                 # binned unless run_type 1
      "H_years": np.arange(1, 1 + n_annual), "bin_H": run_type != 1,
   }


# ---- Step 2: joint fit ----

def fit_joint(F_ref, lmbda, T_eq, t_final_years, fit_t, fit_T_abs, reg):
   """Joint fit of (kappa, h_ml, eps, lambda) to the temperature series
   (fit_t, fit_T_abs) and the Step 1 regression rows. Returns
   dc.run_joint_fit's summary."""
   T_reg, N_reg = reg["T_reg"], reg["N_reg"]

   # Seed kappa and h_ml from a temperature-only fit at eps = 1; its RMSE and
   # the Gregory RMSE set the weights of the two residuals.
   seed_case = Case(F_ref, T_eq, 1.0, t_final_years)

   def seed_dT(t, kappa, h_ml):
      return dc.surface_dT(seed_case, t, kappa, h_ml,
                           rtol=dc.SENSITIVITY_RTOL, atol=dc.SENSITIVITY_ATOL)

   (kappa0, h_ml0), _ = curve_fit(
      seed_dT, fit_t, fit_T_abs, p0=[1.0e-4, 100.0],
      bounds=([10.0**BOUNDS_LO[0], BOUNDS_LO[1]], [10.0**BOUNDS_HI[0], BOUNDS_HI[1]]),
      max_nfev=60,
   )
   sigma_T = max(1e-2, float(np.sqrt(np.mean((seed_dT(fit_t, kappa0, h_ml0) - fit_T_abs) ** 2))))
   sigma_N = max(5e-2, float(np.sqrt(np.mean((N_reg - (F_ref - lmbda * T_reg)) ** 2))))

   data = {
      "fit_t": np.asarray(fit_t, float), "T_obs": np.asarray(fit_T_abs, float),
      "T_reg": T_reg, "N_obs": N_reg,
      "H_years": np.asarray(reg["H_years"], float), "bin_H": bool(reg["bin_H"]),
      "F_frac": 1.0,
      "T_eq": float(T_eq),
      "t_final_years": float(t_final_years), "ramp_years": 0.0, "z_max": None,
      "sigma_T": sigma_T, "sigma_N": sigma_N,
      "big": np.full(int(fit_t.shape[0] + T_reg.shape[0]), 1e3),
   }
   theta0 = np.array([
      np.log10(min(max(kappa0, 10.0**BOUNDS_LO[0]), 10.0**BOUNDS_HI[0])),
      min(max(h_ml0, BOUNDS_LO[1]), BOUNDS_HI[1]), 1.0,
      max(lmbda, BOUNDS_LO[3]),
   ])
   return dc.run_joint_fit(theta0, data, BOUNDS_LO, BOUNDS_HI)


def plot_radiative_overlay(ax, ratio_ax, T_reg, N_reg, H_reg, F_ref, lmbda, eps):
   """On the Step 1 panel: the joint fit's N = F - lambda*T and the same with
   the -(eps-1)*H term; beneath it, the ratio of the two lines."""
   N_line = F_ref - lmbda * T_reg
   N_eps = N_line - (eps - 1.0) * H_reg
   rmse_eps = float(np.sqrt(np.mean((N_reg - N_eps) ** 2)))
   order = np.argsort(T_reg)
   ax.plot(T_reg[order], N_line[order], color="tab:orange", lw=2, ls="-.",
           label=f"Joint Fit: F_ref={F_ref:.3f}, -lambda={-lmbda:.3f}")
   ax.plot(T_reg[order], N_eps[order], color="green", lw=2, ls="--",
           label=rf"w/ $(\epsilon-1)H$: $\epsilon$={eps:.2f} (RMSE={rmse_eps:.3f})")
   ax.legend(loc="upper right", prop={"weight": "bold", "size": 8})

   # Mask the ratio where the denominator crosses zero (T near T_eq).
   denom = np.where(np.abs(N_line) > 0.02 * abs(F_ref), N_line, np.nan)
   ratio_ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
   ratio_ax.plot(T_reg[order], (N_eps / denom)[order], color="green", lw=1.5, marker="o", ms=3)
   ratio_ax.tick_params(labelsize=12, width=2, length=6, direction="in")
   ratio_ax.grid(alpha=0.3)


def fit_and_plot_model(figs, i, df, layout, run_type, model, reg, t2m, nettoa):
   """Step 2 for one model: run the joint fit, record it, draw every panel."""
   T_eq = reg["T_eq"]

   # Fit window by run type; the plots always cover the whole run. Temperatures
   # are carried as T/T_eq.
   fit_T = t2m[:151]/T_eq if run_type in (1, 2) else t2m/T_eq
   plot_T = t2m/T_eq
   fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
   plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
   t_final_years = max(fit_t.max(), plot_t.max())
   fit_T_abs, plot_T_abs = fit_T * T_eq, plot_T * T_eq

   joint = fit_joint(reg["F_ref"], reg["lam"], T_eq, t_final_years, fit_t, fit_T_abs, reg)
   kappa_pde, h_ml_pde = joint["kappa"], joint["h_ml"]
   eps, T_eq, lmbda, F_ref = joint["eps"], joint["T_eq"], joint["lam"], joint["F_ref"]
   fit_T, plot_T = fit_T_abs / T_eq, plot_T_abs / T_eq
   case = Case(F_ref, T_eq, eps, t_final_years)

   dc.joint_fit_check(model, joint)
   print(f"   [joint] {model}: eps={eps:.3f}+/-{joint['eps_err']:.3f} "
         f"kappa={kappa_pde:.2e} h_ml={h_ml_pde:.0f} F={F_ref:.2f} "
         f"lam={lmbda:.2f} T_eq={T_eq:.2f} (fixed at STEP-1)")
   dc.set_row(df, model, {
      "F_ref": F_ref, "lambda": lmbda, "T_eq": T_eq,
      "epsilon": eps, "epsilon_unc": joint["eps_err"],
      "kappa_pde": kappa_pde, "h_ml_pde": h_ml_pde,
      "kappa_pde_unc": joint["kappa_err"], "h_ml_pde_unc": joint["h_ml_err"],
      "T_eq_unc": joint["T_eq_err"], "lambda_unc": joint["lam_err"],
      **dc.joint_diagnostics(joint),
   })

   # Radiative diagnostics over the Step 1 regression rows.
   T_reg, N_reg = np.asarray(reg["T_reg"], float), np.asarray(reg["N_reg"], float)
   H_reg = dc.heat_uptake(case, reg["H_years"], kappa_pde, h_ml_pde)
   if reg["bin_H"]:
      H_reg = np.array([H_reg[j:j + 50].mean() for j in range(50, H_reg.shape[0], 50)])
   plot_radiative_overlay(figs.step1.axs[i], figs.step1.ratio_axs[i], T_reg, N_reg, H_reg,
                          F_ref, lmbda, eps)
   dc.plot_uptake_panels(figs, i, model, model, T_reg, N_reg, H_reg, reg["bin_H"],
                         F_ref, lmbda, eps)

   dc.plot_surface_budget_bars(
      case, model, kappa_pde, h_ml_pde,
      layout.reserve(BUDGET, figname(EXPT, model, "surface_budget_bars")))

   pde_T = dc.surface_dT(case, plot_t, kappa_pde, h_ml_pde) / T_eq
   H_plot = dc.heat_uptake(case, plot_t, kappa_pde, h_ml_pde)

   # Normalized OHC vs T/T_eq. The model's reference heat content is that of
   # the column it was solved on.
   z_max, _ = dc.pde_grid(PDE_FIT_DZ_TARGET, kappa_pde, t_final_years)
   to_joules = EARTH_AREA * SECONDS_PER_YEAR          # annual W/m^2 -> J
   ohc = (to_joules * np.cumsum(nettoa)) / OCEAN_HEAT_CAPACITY
   N_pde = F_ref - lmbda * (T_eq * pde_T) - (eps - 1.0) * H_plot
   ohc_pred = (to_joules * np.cumsum(N_pde)) / (EARTH_AREA * RHO_CP * z_max)
   dc.plot_ohc_panel(figs, i, model, t2m/T_eq, ohc/T_eq, pde_T, ohc_pred/T_eq)

   dc.plot_sensitivity_panels(figs, i, model, case, kappa_pde, h_ml_pde, lmbda,
                              plot_t, plot_T, fit_t, fit_T, pde_T, H_plot)
   dc.plot_temperature_panel(
      figs, i, model, plot_t, plot_T, pde_T, T_eq,
      note=(rf"$\kappa_{{1b+d}}$ = {kappa_pde:.1e} m^2/s, h$_{{ml}}$ = {h_ml_pde:.0f} m, "
            rf"$\epsilon$ = {eps:.2f}"))
   dc.plot_nettoa_panel(figs, i, model, case, lmbda, plot_t, plot_T, pde_T, H_plot, nettoa)


# ---- main ----

def main():
   cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
   add_test_argument(cli)
   args = cli.parse_known_args()[0]

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
      figs = dc.make_figures(MODELS, TITLE_PREFIX, r"$T_s/2\, \mathrm{ECS}$")

      reg = {}
      for i, model in enumerate(fit_models):
         t2m, nettoa = perturbed_anomalies(model, data.rx2(model), expts,
                                              n_years=150 if run_type == 1 else None)
         reg[model] = fit_gregory(figs.step1.axs[i], df, model, t2m, nettoa, run_type)
      print("Finished Step 1: Gregory params to df (Step 1 figures saved after Step 2)")

      for i, model in enumerate(fit_models):
         t2m, nettoa = perturbed_anomalies(model, data.rx2(model), expts,
                                              drop_nan_nettoa=False)
         fit_and_plot_model(figs, i, df, layout, run_type, model, reg[model], t2m, nettoa)
         progress.step(run_type, model)

      df.to_csv(layout.csv_path("fitted_model_params"), index=False)
      dc.finish_figures(figs, layout)
      print("Finished final val/result plots")

   dc.close_pool()


if __name__ == "__main__":
   main()
