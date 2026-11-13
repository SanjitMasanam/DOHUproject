#!/usr/bin/env python3
"""Shared machinery of the two joint diffusive fits (diffusive-eps-joint and
diffusive-eps-joint_fixedRadiativeParams).

Model: slab mixed layer over a diffusive thermocline with heat-uptake efficacy
eps (pde_solver_1lyrEBM_diffusive). Its TOA balance is

    N = F - lambda*T - (eps-1)*H,    H = diffusive flux out of the slab [W/m^2].

Sections: constants, PDE solves, joint least-squares fit, table and progress,
figure set, per-model panels. Design notes are in docs/DECISIONS.md.
"""
from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfcx

from figure_paths import DPI_WIDE, STEP1, figname
from pde_solver_1lyrEBM_diffusive import Params as PDEParams, solve_model as pde_solve_model, YEAR
from plot_style import (
   AXIS_LABEL_FONTSIZE,
   EXTRA_TEXT_FONTSIZE,
   format_ax,
   make_model_grid,
   make_model_grid_3d,
   make_model_grid_with_ratio,
   plot_regression_3d,
)

# =============================== Constants ===================================

RHO_CP = 4.186e6                  # volumetric heat capacity of seawater [J m^-3 K^-1]
EXPT = "4xCO2"                    # experiment token in every output filename

# Stand-in for h_ml = 0, which is singular in the solver (division by c_ml).
H_ML_PURELY_DIFFUSIVE = 1.0e-6    # m

# Main fit: saved time points per solve and target vertical spacing.
PDE_FIT_N_SAVE = 250
PDE_FIT_DZ_TARGET = 15.0          # m

# Sensitivity sweeps around each model's best fit. Diagnostic only, so they
# run at looser tolerance and with fewer saved points than the fit.
SENSITIVITY_KAPPA_FACTOR_RANGE = (0.2, 5.0)     # multiples of the fitted kappa
SENSITIVITY_HML_FACTOR_RANGE = (0.25, 50.0)     # multiples of the fitted h_ml
SENSITIVITY_KAPPA_BOUNDS = (1e-7, 1e-3)         # m^2/s
SENSITIVITY_HML_BOUNDS = (10.0, 2000.0)         # m
SENSITIVITY_N_1D = 13                           # curves per 1-D sweep
SENSITIVITY_N_2D = 9                            # points per axis of the heatmaps
SENSITIVITY_UPTAKE_HML_RANGE = (1e-3, 1000.0)   # h_ml range of the H(t) sweep [m], log-spaced
SENSITIVITY_UPTAKE_N = 25
SENSITIVITY_DZ_RANGE = (10.0, 1000.0)           # grid-spacing sweep [m]
SENSITIVITY_N_SAVE = 100
SENSITIVITY_RTOL = 1e-6
SENSITIVITY_ATOL = 1e-8

# Joint least-squares fit (scipy least_squares, TRF).
JOINT_MAX_NFEV = 200
JOINT_FTOL = 1e-8
JOINT_XTOL = 1e-8
JOINT_FD_REL_STEP = 1e-6          # relative forward-difference step of the Jacobian
JOINT_PARAM_NAMES = ["kappa", "h_ml", "eps", "lambda"]   # theta order; kappa as log10
JOINT_PARAM_UNITS = ["m^2/s", "m", "", "W/m^2/K"]
JOINT_WIDTH_THRESH = 0.5          # flag WIDE if 1-sigma/|value| exceeds this
JOINT_BOUND_RTOL = 1e-3           # flag AT-BOUND within this fraction of the bound span
JOINT_COND_THRESH = 1e8           # flag DEGENERATE if cond(J^T J) exceeds this

# Settings of the fixedRadiativeParams variant, shared with li2012_ohu_vs_T.py.
PDE_Z_MAX = 3700.0                # closed, insulated column depth [m]   (DECISIONS D1)
NETTOA_REF_Z = 2500.0             # depth the AOGCM Net TOA is referenced to [m]
NETTOA_SCALE = PDE_Z_MAX / NETTOA_REF_Z   # applied to NETTOA where it is read (DECISIONS D2)
OHC_TS_EQUILIBRIUM_YEARS = 100000.0       # horizon of the fit curve on the OHC-vs-Ts panel
KAPPA_HML_BOUNDS_LO = np.array([np.log10(1e-7), 1e-3])   # [log10(kappa), h_ml]
KAPPA_HML_BOUNDS_HI = np.array([np.log10(1e-3), 300.0])


# =============================== PDE solves ==================================

@dataclass(frozen=True)
class Case:
   """One model's PDE problem, apart from the fitted (kappa, h_ml)."""
   F_ref: float                  # forcing [W/m^2]
   T_eq: float                   # equilibrium warming [K]; lambda = F_ref/T_eq
   eps: float                    # heat-uptake efficacy
   t_final_years: float
   ramp_years: float = 0.0       # forcing rises linearly to F_ref over this long; 0 = step
   z_max: float | None = None    # column depth [m]; None = sized per solve (see pde_grid)


def ramp_frac(t_years, ramp_years):
   """Forcing as a fraction of F_ref at t_years."""
   t = np.asarray(t_years, dtype=float)
   if ramp_years <= 0:
      return np.ones_like(t)
   return np.minimum(t / ramp_years, 1.0)


def ramp_forcing(F0, ramp_years):
   """PDEParams.forcing: None for a step, else F(t) [W/m^2] with t in seconds."""
   if ramp_years <= 0:
      return None
   return lambda t: F0 * min(t / (ramp_years * YEAR), 1.0)


def log_spaced_t_eval(t_final_seconds, n_save):
   """Solver save grid: t = 0, then log-spaced from year 1 to t_final, so the
   fast early rise is resolved."""
   t_eval = np.concatenate(([0.0], np.geomspace(YEAR, t_final_seconds, n_save - 1)))
   t_eval[-1] = t_final_seconds  # keep round-off from landing outside t_span
   return t_eval


def pde_grid(dz_target, kappa, t_final_years, z_max=None):
   """(z_max [m], Nz). With z_max None the depth is 6 diffusive lengths,
   clamped to [2700, 4000] m; a given z_max is used as is."""
   if z_max is None:
      z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
      return z_max, int(z_max / dz_target + 1)
   return z_max, max(4, int(z_max / dz_target + 1))   # the flux stencil reads 3 levels


def pde_params(case, kappa, h_ml, dz_target=PDE_FIT_DZ_TARGET):
   z_max, nz = pde_grid(dz_target, kappa, case.t_final_years, case.z_max)
   return PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=case.T_eq, F0=case.F_ref, epsilon=case.eps,
      z_max=z_max, Nz=nz, t_final=case.t_final_years * YEAR,
      forcing=ramp_forcing(case.F_ref, case.ramp_years),
   )


def surface_uptake(pde_p, sol):
   """H(t) [W/m^2] = -rho_cp*kappa*dtheta/dz at z = 0 (2nd-order one-sided);
   positive when heat leaves the slab downward."""
   theta, dz = sol["theta"], sol["dz"]
   dtheta_dz0 = (-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dz)
   return -pde_p.D * dtheta_dz0


# Recent full solves. The best-fit solve is read several times per model.
_SOLVE_CACHE = {}
_SOLVE_CACHE_MAX = 8


def solve(case, kappa, h_ml, dz_target=PDE_FIT_DZ_TARGET, rtol=None, atol=None):
   """(pde_params, solution), cached. rtol/atol None = the solver's tight defaults."""
   key = (case, float(kappa), float(h_ml), float(dz_target), rtol, atol)
   hit = _SOLVE_CACHE.get(key)
   if hit is not None:
      return hit
   pde_p = pde_params(case, kappa, h_ml, dz_target)
   kw = {} if rtol is None else {"rtol": rtol, "atol": atol}
   sol = pde_solve_model(pde_p, t_eval=log_spaced_t_eval(pde_p.t_final, PDE_FIT_N_SAVE), **kw)
   if len(_SOLVE_CACHE) >= _SOLVE_CACHE_MAX:
      _SOLVE_CACHE.clear()
   _SOLVE_CACHE[key] = (pde_p, sol)
   return pde_p, sol


def surface_dT(case, t_years, kappa, h_ml, dz_target=PDE_FIT_DZ_TARGET, rtol=None, atol=None):
   """Surface warming [K] at t_years. A failed or non-finite solve returns 1e6
   everywhere, so a fit steers away from it."""
   try:
      _, sol = solve(case, kappa, h_ml, dz_target, rtol, atol)
      dT = np.interp(t_years, sol["t"] / YEAR, sol["dT"])
   except Exception:
      dT = np.full(np.shape(t_years), 1e6, dtype=float)
   if not np.all(np.isfinite(dT)):
      dT = np.full(np.shape(t_years), 1e6, dtype=float)
   return dT


def heat_uptake(case, t_years, kappa, h_ml, dz_target=PDE_FIT_DZ_TARGET):
   """H [W/m^2] at t_years; zero where the solve fails."""
   try:
      pde_p, sol = solve(case, kappa, h_ml, dz_target)
      out = np.interp(t_years, sol["t"] / YEAR, surface_uptake(pde_p, sol))
   except Exception:
      out = np.zeros(np.shape(t_years), dtype=float)
   return np.where(np.isfinite(out), out, 0.0)


def _sweep_solve(case, kappa, h_ml, dz_target):
   pde_p = pde_params(case, kappa, h_ml, dz_target)
   sol = pde_solve_model(
      pde_p, t_eval=log_spaced_t_eval(pde_p.t_final, SENSITIVITY_N_SAVE),
      rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
   )
   return pde_p, sol


def sweep_dT(case, kappa, h_ml, t_eval, dz_target=PDE_FIT_DZ_TARGET):
   """Surface warming [K] of one sweep point (uncached; runs in the pool)."""
   _, sol = _sweep_solve(case, kappa, h_ml, dz_target)
   return np.interp(t_eval, sol["t"] / YEAR, sol["dT"])


def sweep_H(case, kappa, h_ml, t_eval, dz_target=PDE_FIT_DZ_TARGET):
   """H [W/m^2] of one sweep point; zero where the solve fails."""
   try:
      pde_p, sol = _sweep_solve(case, kappa, h_ml, dz_target)
      out = np.interp(t_eval, sol["t"] / YEAR, surface_uptake(pde_p, sol))
   except Exception:
      out = np.zeros(np.shape(t_eval), dtype=float)
   return np.where(np.isfinite(out), out, 0.0)


# Fork pool for the sweep points and the Jacobian columns. Everything it
# dispatches is a module-level function of this file.
_POOL = None


def start_pool():
   global _POOL
   _POOL = multiprocessing.get_context("fork").Pool()


def close_pool():
   _POOL.close()
   _POOL.join()


# ============================ Joint least-squares fit ========================
#
# theta = [log10(kappa), h_ml] or [log10(kappa), h_ml, eps, lambda]. T_eq is
# always fixed and F = lambda*T_eq. `data` holds:
#   fit_t, T_obs        years and AOGCM temperature of the temperature residual
#   T_reg, N_obs        AOGCM T and Net TOA rows of the radiative residual
#   H_years, bin_H      years at which H is evaluated; 50-yr-bin it (first bin dropped)?
#   F_frac              forcing / F_ref at each radiative row (1 for a step)
#   T_eq, eps, lam      fixed values (eps, lam unused when they are in theta)
#   t_final_years, ramp_years, z_max
#   sigma_T, sigma_N    noise scales weighting the two residuals
#   big                 residual returned when the solve fails

def _eps_lam(theta, data):
   return (theta[2], theta[3]) if len(theta) == 4 else (data["eps"], data["lam"])


def joint_forward(case, kappa, h_ml, fit_t, H_years, bin_H):
   """(T at fit_t, H at the radiative rows), or None if the solve fails."""
   pde_p = pde_params(case, kappa, h_ml)
   try:
      sol = pde_solve_model(
         pde_p, t_eval=log_spaced_t_eval(pde_p.t_final, PDE_FIT_N_SAVE),
         rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
      )
      t_yr = sol["t"] / YEAR
      T_model = np.interp(fit_t, t_yr, sol["dT"])
      H_at = np.interp(H_years, t_yr, surface_uptake(pde_p, sol))
   except Exception:
      return None
   if bin_H:
      H_reg = np.array([H_at[i:i + 50].mean() for i in range(50, H_at.shape[0], 50)])
   else:
      H_reg = H_at
   if not (np.all(np.isfinite(T_model)) and np.all(np.isfinite(H_reg))):
      return None
   return T_model, H_reg


# Last (theta, residual) evaluated in this process: least_squares calls fun(x)
# right before jac(x), so the Jacobian can reuse it instead of re-solving.
_JOINT_LAST_R = [None, None]


def joint_residuals(theta, data):
   """Stacked residual [(T_model - T_obs)/sigma_T, (N_model - N_obs)/sigma_N]."""
   eps, lam = _eps_lam(theta, data)
   T_eq = data["T_eq"]
   case = Case(F_ref=lam * T_eq, T_eq=T_eq, eps=eps, t_final_years=data["t_final_years"],
               ramp_years=data["ramp_years"], z_max=data["z_max"])
   out = joint_forward(case, 10.0 ** theta[0], theta[1],
                       data["fit_t"], data["H_years"], data["bin_H"])
   if out is None:
      return data["big"]
   T_model, H_reg = out
   N_model = lam * (T_eq * data["F_frac"] - data["T_reg"]) - (eps - 1.0) * H_reg
   rT = (T_model - data["T_obs"]) / data["sigma_T"]
   rN = (N_model - data["N_obs"]) / data["sigma_N"]
   r = np.concatenate([rT, rN])
   _JOINT_LAST_R[0] = np.array(theta, dtype=float, copy=True)
   _JOINT_LAST_R[1] = r
   return r


def joint_jac(theta, data):
   """Forward-difference Jacobian, one pooled PDE solve per column."""
   theta = np.asarray(theta, dtype=float)
   if _JOINT_LAST_R[0] is not None and np.array_equal(_JOINT_LAST_R[0], theta):
      r0 = _JOINT_LAST_R[1]
   else:
      r0 = joint_residuals(theta, data)
   n = theta.size
   step = JOINT_FD_REL_STEP * np.maximum(np.abs(theta), 1.0)
   pert = []
   for i in range(n):
      tp = theta.copy()
      tp[i] += step[i]
      pert.append((tp, data))
   cols = _POOL.starmap(joint_residuals, pert)
   J = np.empty((r0.size, n), dtype=float)
   for i in range(n):
      J[:, i] = (cols[i] - r0) / step[i]
   return J


def run_joint_fit(theta0, data, lo, hi):
   """Bounded TRF fit of theta within [lo, hi]. Returns a summary dict: fitted
   and fixed parameters, 1-sigma errors from the Jacobian covariance, RMSEs in
   physical units, convergence status and the joint_fit_check diagnostics."""
   x0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
   sol = least_squares(
      joint_residuals, x0, jac=joint_jac, args=(data,),
      bounds=(lo, hi), method="trf", x_scale="jac",
      ftol=JOINT_FTOL, xtol=JOINT_XTOL, max_nfev=JOINT_MAX_NFEV,
   )
   theta = sol.x
   log10_kappa, h_ml = theta[0], theta[1]
   eps, lam = _eps_lam(theta, data)
   T_eq = data["T_eq"]
   kappa = 10.0 ** log10_kappa
   fit_radiative = len(theta) == 4

   # cov = s^2 (J^T J)^-1 with s^2 the reduced chi-square (residuals are 1/sigma-weighted).
   J = sol.jac
   m, n = J.shape
   dof = max(m - n, 1)
   s2 = 2.0 * sol.cost / dof
   JtJ = J.T @ J
   try:
      cov = np.linalg.inv(JtJ) * s2
      perr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
      cond = float(np.linalg.cond(JtJ))
   except np.linalg.LinAlgError:
      perr = np.full(n, np.nan)
      cond = np.inf
   kappa_err = kappa * np.log(10.0) * perr[0]     # delta method from log10(kappa)

   r = sol.fun
   nT = data["T_obs"].size
   rmse_T = float(np.sqrt(np.mean((r[:nT] * data["sigma_T"]) ** 2)))
   rmse_N = float(np.sqrt(np.mean((r[nT:] * data["sigma_N"]) ** 2)))

   span = hi - lo
   at_bound = (np.abs(theta - lo) < JOINT_BOUND_RTOL * span) | \
              (np.abs(theta - hi) < JOINT_BOUND_RTOL * span)

   return {
      "theta": theta, "perr": perr,
      "vals": np.concatenate(([kappa], theta[1:])),
      "errs": np.concatenate(([kappa_err], perr[1:])),
      "kappa": kappa, "h_ml": h_ml, "eps": eps, "T_eq": T_eq, "lam": lam,
      "F_ref": lam * T_eq,
      "kappa_err": kappa_err, "h_ml_err": perr[1],
      "eps_err": perr[2] if fit_radiative else 0.0,
      "T_eq_err": 0.0,
      "lam_err": perr[3] if fit_radiative else 0.0,
      "success": bool(sol.success), "status": int(sol.status),
      "message": str(sol.message), "nfev": int(sol.nfev), "njev": int(sol.njev),
      "cost": float(sol.cost), "optimality": float(sol.optimality),
      "reduced_chi2": s2, "cond": cond, "rmse_T": rmse_T, "rmse_N": rmse_N,
      "at_bound": at_bound,
      "relwidth": perr / np.maximum(np.abs(theta), 1e-9),   # kappa judged in log space
   }


def joint_diagnostics(s):
   """The run_joint_fit columns every variant writes to its parameter table."""
   return {
      "joint_converged": s["success"], "joint_reduced_chi2": s["reduced_chi2"],
      "joint_cond": s["cond"], "joint_rmse_T": s["rmse_T"], "joint_rmse_N": s["rmse_N"],
      "joint_any_at_bound": bool(np.any(s["at_bound"])),
   }


def joint_fit_check(model, s):
   """Print the convergence and identifiability report; return 'PASS' or 'CHECK'."""
   conv = s["success"] and s["status"] > 0
   railed = bool(np.any(s["at_bound"]))
   wide = bool(np.any(s["relwidth"] > JOINT_WIDTH_THRESH))
   degenerate = s["cond"] > JOINT_COND_THRESH
   verdict = "PASS" if (conv and not railed and not wide and not degenerate) else "CHECK"
   print(f"   [joint] {model}: {verdict}  "
         f"converged={conv} (status={s['status']}, nfev={s['nfev']}, njev={s['njev']})  "
         f"reduced_chi2={s['reduced_chi2']:.2f}  cond(JtJ)={s['cond']:.1e}")
   print(f"      RMSE_T={s['rmse_T']:.4f} K   RMSE_N={s['rmse_N']:.4f} W/m^2")
   for name, v, e, rw, ab, u in zip(JOINT_PARAM_NAMES, s["vals"], s["errs"],
                                    s["relwidth"], s["at_bound"], JOINT_PARAM_UNITS):
      flags = []
      if ab: flags.append("AT-BOUND")
      if rw > JOINT_WIDTH_THRESH: flags.append("WIDE")
      tag = ("  <- " + ", ".join(flags)) if flags else "  ok"
      vstr = f"{v:.3e}" if name == "kappa" else f"{v:.3f}"
      estr = f"{e:.2e}" if name == "kappa" else f"{e:.3f}"
      print(f"        {name:7s}= {vstr} +/- {estr} {u:8s} relwidth={rw:.2f}{tag}")
   if degenerate:
      print(f"      NOTE: cond(JtJ)={s['cond']:.1e} > {JOINT_COND_THRESH:.0e} "
            f"-> parameters not jointly identified (near-degenerate).")
   return verdict


# ============================ Table and progress ==============================

def set_row(df, model, values):
   """Write {column: value} for one model into the parameter table."""
   for col, val in values.items():
      df.loc[df["model"] == model, col] = val


class Progress:
   """Prints elapsed time and an ETA after each model."""

   def __init__(self, total):
      self.total, self.done, self.start = total, 0, time.time()

   def step(self, run_type, model):
      self.done += 1
      elapsed = time.time() - self.start
      avg = elapsed / self.done
      print(f"[Progress] run_type={run_type} model={model} "
            f"({self.done}/{self.total} model-runs done) "
            f"elapsed={elapsed/60:.1f} min, avg={avg:.1f} s/model, "
            f"ETA remaining={avg * (self.total - self.done)/60:.1f} min")


# =============================== Figure set ==================================

@dataclass
class ModelGrid:
   """One multi-model figure. Model i of the fit list draws on axs[i]."""
   fig: object
   axs: object
   ratio_axs: object = None
   artists: list = field(default_factory=list)   # (artist, value) pairs for the shared colorbar


@dataclass
class Figures:
   reg3d: ModelGrid          # (T, H, N) regression
   step1: ModelGrid          # Net TOA vs T2M, with a ratio axis per panel
   nh: ModelGrid             # Net TOA vs H
   ht: ModelGrid             # H vs T2M
   final: ModelGrid          # T2M vs time
   nettoa: ModelGrid         # Net TOA vs time
   ohc_ts: ModelGrid         # normalized OHC vs normalized T
   sens_kappa: ModelGrid
   sens_hml: ModelGrid
   sens_uptake: ModelGrid
   sens_dz: ModelGrid
   sens_t63: ModelGrid
   sens_rmse: ModelGrid
   xmax: list = field(default_factory=list)      # last plotted year, per drawn model


def make_figures(models, title_prefix, ohc_xlabel, **grid_kw):
   """Create the 13 multi-model figures of one run-type pass.

   title_prefix heads the titles that name the experiment; grid_kw goes to
   make_model_grid.
   """
   def grid(title, xlabel, ylabel, **kw):
      return ModelGrid(*make_model_grid(models, title=title, xlabel=xlabel, ylabel=ylabel,
                                        **grid_kw, **kw))

   eq_ratio = "Equilibrium Ratio ($T/T_{eq}$)"
   t2m = "2-meter Air Temperature Anomaly (K)"
   nettoa = r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)"
   uptake = r"Ocean Heat Uptake H ($W*m^{-2}$)"
   kappa, hml = r"$\kappa$ (m$^2$/s)", r"h$_{ml}$ (m)"

   reg3d = ModelGrid(*make_model_grid_3d(
      models, title=r"1-Box + Diffusion efficacy fit: N = F $-\ \lambda$T $-\ (\epsilon-1)$H"))
   step1 = ModelGrid(*make_model_grid_with_ratio(
      models, title=title_prefix + r" Net TOA vs T$_{2M}$", xlabel=t2m, ylabel=nettoa,
      ratio_ylabel=r"$\frac{F-\lambda T-(\epsilon-1)H}{F-\lambda T}$"))
   figs = Figures(
      reg3d=reg3d,
      step1=step1,
      nh=grid(title_prefix + r" Net TOA vs Ocean Heat Uptake H", uptake, nettoa),
      ht=grid(title_prefix + r" Ocean Heat Uptake H vs T$_{2M}$", t2m, uptake),
      final=grid(title_prefix + r" T$_{2M}$ vs. Time w/ Diffusive Fit", r"Time (years)",
                 r"Temperature Anomaly (K)", dpi=120, right=0.95, wspace=0.28),
      nettoa=grid(title_prefix + r" Net TOA vs. Time", "Time (years)",
                  r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)"),
      ohc_ts=grid(title_prefix + r" OHU vs. Surface Temp (Normalized)", ohc_xlabel,
                  r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$"),
      sens_kappa=grid(r"Sensitivity to $\kappa$ (h$_{ml}$ held at best fit)",
                      "Time (years)", eq_ratio),
      sens_hml=grid(r"Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)",
                    "Time (years)", eq_ratio),
      sens_uptake=grid(r"Ocean Heat Uptake Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)",
                       "Time (years)", r"$H$ (W m$^{-2}$)"),
      sens_dz=grid(r"Sensitivity to Grid Spacing dz ($\kappa$, h$_{ml}$ held at best fit)",
                   "Time (years)", eq_ratio),
      sens_t63=grid(r"Years to Reach 63% of T$_{eq}$ vs. $\kappa$/h$_{ml}$", kappa, hml),
      sens_rmse=grid(r"Fit RMSE vs. $\kappa$/h$_{ml}$ (Parameter Degeneracy)", kappa, hml),
   )
   figs.final.fig.text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center',
                       fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=-90.)
   return figs


def _shared_colorbar(grid, norm, cmap, label, recolor=True):
   """Put every artist of `grid` on one color scale and add a single colorbar."""
   for artist, value in grid.artists:
      if recolor:
         artist.set_color(cmap(0.25 + 0.65 * norm(value)))
      else:
         artist.set_norm(norm)
   sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
   sm.set_array([])
   cbar = grid.fig.colorbar(sm, ax=list(grid.axs), fraction=0.03, pad=0.02)
   cbar.set_label(label, fontsize=15, fontweight="bold")


def _set_time_axis(axs, xmax_list, scale):
   for ax, xmax in zip(axs, xmax_list):
      ax.set_xscale(scale)
      ax.set_xlim(1, xmax + 1)


def finish_figures(figs, layout):
   """Add the shared colorbars and save every figure of the pass."""
   for grid, name in ((figs.step1, "T2M_vs_NETTOA"), (figs.nh, "H_vs_NETTOA"),
                      (figs.ht, "H_vs_T2M")):
      layout.save(grid.fig, STEP1, figname(EXPT, "all_models", name), close=True)

   # 3D regression: one symmetric residual color scale across the panels.
   scatters = figs.reg3d.artists
   resid = np.concatenate([r for _, r in scatters]) if scatters else np.array([0.0])
   rmax = float(np.nanmax(np.abs(resid))) or 1.0
   norm = mpl.colors.Normalize(vmin=-rmax, vmax=rmax)
   for sc, _ in scatters:
      sc.set_norm(norm)
   sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
   sm.set_array([])
   cbar = figs.reg3d.fig.colorbar(sm, ax=list(figs.reg3d.axs), fraction=0.015, pad=0.02)
   cbar.set_label(r"N residual (obs $-$ fit) [W m$^{-2}$]", fontsize=13, fontweight="bold")
   layout.save(figs.reg3d.fig, STEP1, figname(EXPT, "all_models", "N_T_H_regression3d"),
               close=True)

   # Temperature and Net TOA time series, on linear and log time axes.
   for scale in ["linear", "log"]:
      _set_time_axis(figs.final.axs, figs.xmax, scale)
      for ax, xmax in zip(figs.final.axs, figs.xmax):
         if scale == "linear":
            # Label the reference line at x = 151 as "150", in orange.
            ticks = sorted(set(np.linspace(1, xmax + 1, 5).tolist()) | {151.0})
            ax.set_xticks(ticks)
            ax.set_xticklabels(["150" if abs(tv - 151.0) < 1e-9 else f"{tv:.0f}" for tv in ticks])
            for tick_val, tick_label in zip(ticks, ax.get_xticklabels()):
               if abs(tick_val - 151.0) < 1e-9:
                  tick_label.set_color("orange")
                  tick_label.set_fontsize(10)
         else:
            ax.xaxis.set_major_locator(mpl.ticker.LogLocator())
      layout.save_results(figs.final.fig, figname(EXPT, "all_models", "T2m_vs_t", scale))

      _set_time_axis(figs.nettoa.axs, figs.xmax, scale)
      layout.save_results(figs.nettoa.fig, figname(EXPT, "all_models", "NETTOA_timeseries", scale))
   plt.close(figs.final.fig)
   plt.close(figs.nettoa.fig)

   # Sensitivity sweeps.
   def value_range(grid):
      values = np.array([v for _, v in grid.artists])
      return values.min(), values.max()

   def grid_range(grid):
      return (np.nanmin([np.nanmin(g) for _, g in grid.artists]),
              np.nanmax([np.nanmax(g) for _, g in grid.artists]))

   _shared_colorbar(figs.sens_kappa, mpl.colors.LogNorm(*value_range(figs.sens_kappa)),
                    plt.cm.Blues, r"$\kappa$ (m$^2$/s)")
   _shared_colorbar(figs.sens_hml, mpl.colors.Normalize(*value_range(figs.sens_hml)),
                    plt.cm.Greens, r"h$_{ml}$ (m)")
   _shared_colorbar(figs.sens_uptake, mpl.colors.LogNorm(*value_range(figs.sens_uptake)),
                    plt.cm.Oranges, r"h$_{ml}$ (m)")
   _shared_colorbar(figs.sens_dz, mpl.colors.LogNorm(*value_range(figs.sens_dz)),
                    plt.cm.Purples, "dz (m)")
   t63_min, t63_max = grid_range(figs.sens_t63)
   _shared_colorbar(figs.sens_t63, mpl.colors.Normalize(vmin=t63_min, vmax=t63_max),
                    plt.get_cmap("viridis"), "Years to 63% of $T_{eq}$", recolor=False)
   rmse_min, rmse_max = grid_range(figs.sens_rmse)
   _shared_colorbar(figs.sens_rmse, mpl.colors.LogNorm(vmin=max(rmse_min, 1e-6), vmax=rmse_max),
                    plt.get_cmap("viridis"), "RMSE vs. AOGCM Data (K)", recolor=False)

   for grid, name in ((figs.sens_kappa, "sensitivity_spaghetti_kappa"),
                      (figs.sens_hml, "sensitivity_spaghetti_h_ml"),
                      (figs.sens_uptake, "sensitivity_uptake_h_ml"),
                      (figs.sens_dz, "sensitivity_spaghetti_dz")):
      for scale in ["linear", "log"]:
         _set_time_axis(grid.axs, figs.xmax, scale)
         if scale == "log":
            for ax, _ in zip(grid.axs, figs.xmax):
               ax.xaxis.set_major_locator(mpl.ticker.LogLocator())
         layout.save_results(grid.fig, figname(EXPT, "all_models", name, scale))
      plt.close(grid.fig)
   for grid, name in ((figs.sens_t63, "sensitivity_heatmap_t63"),
                      (figs.sens_rmse, "sensitivity_heatmap_rmse")):
      layout.save_results(grid.fig, figname(EXPT, "all_models", name), close=True)

   # OHC vs Ts: the year colorbar is taken from the last model's scatter.
   cbar = figs.ohc_ts.fig.colorbar(figs.ohc_ts.artists[-1], ax=figs.ohc_ts.axs.ravel().tolist(),
                                   fraction=0.025, pad=0.025)
   cbar.set_label("Year")
   layout.save_results(figs.ohc_ts.fig, figname(EXPT, "all_models", "ohc_ts"), close=True)


# ============================= Per-model panels ==============================

def annotate(ax, text, y=0.92, **kw):
   """Boxed annotation under the model tag."""
   ax.text(0.02, y, text, transform=ax.transAxes, va="top", ha="left",
           fontsize=EXTRA_TEXT_FONTSIZE,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4), **kw)


def rolling_mean(a, window=10):
   """Running mean over `window` rows; the input unchanged if it is shorter."""
   a = np.asarray(a, dtype=float)
   if window <= 1 or a.shape[0] < window:
      return a
   return np.array([a[i:i + window].mean() for i in range(a.shape[0] - window + 1)])


def style_ratio_axis(ax):
   ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
   ax.tick_params(labelsize=12, width=2, length=6, direction="in")
   ax.grid(alpha=0.3)


def plot_uptake_panels(figs, i, model, label, T_reg, N_reg, H_reg, bin_H,
                       F_ref, lmbda, eps, ramp_years=0.0):
   """N-vs-H, H-vs-T and 3D regression panels over the radiative rows
   (T_reg, N_reg, H_reg), 50-yr bins when bin_H."""
   T_reg, N_reg = np.asarray(T_reg, dtype=float), np.asarray(N_reg, dtype=float)

   # N vs H: data, the N = H line, and the full relation at each row's own T.
   ax = figs.nh.axs[i]
   ax.scatter(H_reg, N_reg, s=8, alpha=0.5, label="Data")
   ends = np.array([H_reg.min(), H_reg.max()])
   ax.plot(ends, ends, color="0.5", lw=1.0, ls="-.", label="N = H (Analytical Sol.)")
   ax.plot(H_reg, F_ref - lmbda * T_reg - (eps - 1.0) * H_reg, color="green", lw=2, ls="--",
           label=r"$N=F-\lambda T-(\epsilon-1) H$")
   format_ax(ax, text=f"{label}", xscale="linear", yscale="linear", legend_loc="lower right")

   # H vs T, ~10-yr running means: the PDE's H, and H inverted from the AOGCM N.
   ax = figs.ht.axs[i]
   window = max(1, int(round(10.0 / (50 if bin_H else 1))))
   T_roll, H_roll = rolling_mean(T_reg, window), rolling_mean(H_reg, window)
   order = np.argsort(T_roll)
   ax.plot(T_roll[order], H_roll[order], color="lightblue", lw=2, label="H (PDE fit, 10-yr mean)")
   if ramp_years <= 0:   # step-forcing solution
      ax.plot(T_roll[order], F_ref/eps - (lmbda/eps)*T_roll[order], color="0.5", lw=1.0,
              ls="-.", label="H (analytical sol.)")
   if abs(eps - 1.0) > 1e-6:
      H_inv = (F_ref - N_reg - lmbda * T_reg) / (eps - 1.0)
      ax.scatter(T_roll, rolling_mean(H_inv, window), s=8, alpha=0.5, color="green",
                 label=r"$H=(F-N-\lambda T)/(\epsilon-1)$ (AOGCM, 10-yr mean)")
   format_ax(ax, text=f"{label}\n$\\epsilon$={eps:.2f}", xscale="linear", yscale="linear",
             legend_loc="lower left")

   figs.reg3d.artists.append(
      plot_regression_3d(figs.reg3d.axs[i], T_reg, H_reg, N_reg, F_ref, lmbda, eps, model))


def plot_surface_budget_bars(case, model, kappa, h_ml, png_path):
   """Surface tendency budget of the best-fit solve, one bar per saved time.

   c_ml*dT_s/dt = F - lambda*T_s - eps*H is split into three terms [K/yr] that
   sum to dT_s/dt: F/c_ml, -(lambda*T_s + (eps-1)*H)/c_ml and -H/c_ml. Top
   panel: stacked |terms| on a log axis with |net| and T_s/T_eq; bottom panel:
   the same magnitudes as lines. H uses the solver's own first-order stencil,
   so the terms close on the integrated tendency.
   """
   pde_p, sol = solve(case, kappa, h_ml)
   t_yr = sol["t"] / YEAR
   theta, dz = sol["theta"], sol["dz"]
   T_s = theta[0]
   c_ml, lam, D = pde_p.c_ml, pde_p.lam, pde_p.D
   eps, F_ref, T_eq = case.eps, case.F_ref, case.T_eq

   forcing   = F_ref * ramp_frac(t_yr, case.ramp_years) / c_ml * YEAR
   H_flux    = D * (T_s - theta[1]) / dz             # W/m^2, >0 when the surface is warmer
   restoring = (-lam * T_s - (eps - 1.0) * H_flux) / c_ml * YEAR
   uptake    = (-H_flux) / c_ml * YEAR
   net       = forcing + restoring + uptake          # dT_s/dt [K/yr]

   comps = [
      (r"$F/c_{ml}$",                                    forcing,   "#9467bd"),
      (r"$-(\lambda T_s + (\epsilon-1)H)/c_{ml}$",       restoring, "#1f77b4"),
      (r"$-H/c_{ml}$",                                   uptake,    "#2ca02c"),
   ]
   # Bars span the midpoints to the neighbouring (log-spaced) save times.
   mid = 0.5 * (t_yr[:-1] + t_yr[1:])
   left = np.concatenate(([max(0.0, t_yr[0] - (mid[0] - t_yr[0]))], mid))
   right = np.concatenate((mid, [t_yr[-1] + (t_yr[-1] - mid[-1])]))
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
   ax.plot(t_yr, net_abs, color="black", lw=1.4, label=r"$|$net $dT_s/dt|$")
   ax.set_yscale("log")
   ax.set_ylim(floor, base.max() * 1.3)
   ax.set_ylabel(r"$|dT_s/dt$ contribution$|$ (K yr$^{-1}$)", fontsize=14, fontweight="bold")
   ax.set_title(rf"{model}: surface tendency budget "
                rf"($\kappa$={kappa:.1e}, $h_{{ml}}$={h_ml:.0f} m, $\epsilon$={eps:.2f})",
                fontsize=13, fontweight="bold")

   ax_r = ax.twinx()
   ratio_line, = ax_r.plot(t_yr, T_s / T_eq, color="red", lw=2.0, ls="--",
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

   for label, vals, color in comps:
      ax2.plot(t_yr, np.abs(vals), color=color, lw=1.6, label=label)
   ax2.set_ylabel(r"$|$contribution$|$ (K yr$^{-1}$)", fontsize=14, fontweight="bold")
   ax2.grid(True, alpha=0.3)
   ax2.set_xlim(left[0], right[-1])
   ax2.set_xlabel("Time (years)", fontsize=14, fontweight="bold")
   fig.savefig(str(png_path), dpi=DPI_WIDE, bbox_inches="tight")
   plt.close(fig)


def plot_ohc_panel(figs, i, label, T_norm, ohc_norm, T_fit_norm, ohc_fit_norm):
   """Normalized OHC against normalized surface temperature: AOGCM years
   (colored by year) and the fitted model's curve."""
   ax = figs.ohc_ts.axs[i]
   sc = ax.scatter(T_norm, ohc_norm, c=np.arange(1, 1 + ohc_norm.shape[0], 1),
                   cmap=plt.cm.turbo, norm=mpl.colors.Normalize(vmin=0, vmax=6000))
   ax.plot(T_fit_norm, ohc_fit_norm, color='green', label='1-Box + Diffusion Fit')
   ax.axvline(1.0, color="0.55", ls='--', lw=0.8)
   format_ax(ax, text=f"{label}", xscale="linear", yscale="linear", ylim=(-0.05, 1.2))
   figs.ohc_ts.artists.append(sc)


def plot_sensitivity_panels(figs, i, label, case, kappa_pde, h_ml_pde, lmbda,
                            plot_t, plot_T, fit_t, fit_T, pde_T, H_plot,
                            kappa_factor_2d=SENSITIVITY_KAPPA_FACTOR_RANGE,
                            hml_factor_2d=SENSITIVITY_HML_FACTOR_RANGE,
                            n_2d=SENSITIVITY_N_2D):
   """Sweep kappa, h_ml and dz around the best fit and draw the six
   sensitivity panels. plot_T, fit_T and pde_T are T/T_eq.

   The 1-D sweeps hold the other parameter at its best fit. The 2-D
   (kappa, h_ml) grid gives the years-to-63%-of-T_eq and fit-RMSE heatmaps.
   """
   F_ref, T_eq, eps = case.F_ref, case.T_eq, case.eps

   def clipped(value, factors, bounds):
      return max(bounds[0], value * factors[0]), min(bounds[1], value * factors[1])

   def sweep(points, t_eval):   # points: (kappa, h_ml, dz)
      return _POOL.starmap(sweep_dT, [(case, k, h, t_eval, dz) for k, h, dz in points])

   dz0 = PDE_FIT_DZ_TARGET
   kappa_grid = np.geomspace(
      *clipped(kappa_pde, SENSITIVITY_KAPPA_FACTOR_RANGE, SENSITIVITY_KAPPA_BOUNDS), SENSITIVITY_N_1D)
   hml_grid = np.linspace(
      *clipped(h_ml_pde, SENSITIVITY_HML_FACTOR_RANGE, SENSITIVITY_HML_BOUNDS), SENSITIVITY_N_1D)
   dz_grid = np.geomspace(*SENSITIVITY_DZ_RANGE, SENSITIVITY_N_1D)
   hml_uptake_grid = np.geomspace(*SENSITIVITY_UPTAKE_HML_RANGE, SENSITIVITY_UPTAKE_N)

   kappa_curves = [c / T_eq for c in sweep([(k, h_ml_pde, dz0) for k in kappa_grid], plot_t)]
   hml_curves = [c / T_eq for c in sweep([(kappa_pde, h, dz0) for h in hml_grid], plot_t)]
   uptake_curves = _POOL.starmap(
      sweep_H, [(case, kappa_pde, h, plot_t, dz0) for h in hml_uptake_grid])
   dz_curves = [c / T_eq for c in sweep([(kappa_pde, h_ml_pde, dz) for dz in dz_grid], plot_t)]

   # 2-D grid: one solve per cell on a common time grid, both metrics from it.
   kappa_grid_2d = np.geomspace(*clipped(kappa_pde, kappa_factor_2d, SENSITIVITY_KAPPA_BOUNDS), n_2d)
   hml_grid_2d = np.linspace(*clipped(h_ml_pde, hml_factor_2d, SENSITIVITY_HML_BOUNDS), n_2d)
   target_63 = (1.0 - np.exp(-1.0)) * T_eq
   t_common = np.linspace(1.0, case.t_final_years, 400)
   cells = sweep([(k, h, dz0) for h in hml_grid_2d for k in kappa_grid_2d], t_common)
   t63_grid = np.full((n_2d, n_2d), np.nan)
   rmse_grid = np.full((n_2d, n_2d), np.nan)
   for idx, dT_common in enumerate(cells):
      ih, ik = divmod(idx, n_2d)
      if dT_common[-1] >= target_63:
         t63_grid[ih, ik] = np.interp(target_63, dT_common, t_common)
      dT_at_fit = np.interp(fit_t, t_common, dT_common)
      rmse_grid[ih, ik] = np.sqrt(np.mean((dT_at_fit - fit_T * T_eq) ** 2))

   # Curves are colored by their swept value in finish_figures.
   def spaghetti(grid, values, curves, note, data=True):
      ax = grid.axs[i]
      for value, curve in zip(values, curves):
         (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
         grid.artists.append((line, value))
      if data:
         ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
         ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
         format_ax(ax, text=f"{label}", xscale="linear", yscale="linear", legend_loc="lower right")
         annotate(ax, note)
      return ax

   kappa_txt = rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s"
   eps_txt = rf"$\epsilon$ = {eps:.2f} (fixed)"
   spaghetti(figs.sens_kappa, kappa_grid, kappa_curves,
             rf"h$_{{ml}}$ = {h_ml_pde:.0f} m, {eps_txt}")
   spaghetti(figs.sens_hml, hml_grid, hml_curves, rf"{kappa_txt}, {eps_txt}")

   ax = spaghetti(figs.sens_uptake, hml_uptake_grid, uptake_curves, None, data=False)
   ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
   ax.plot(plot_t, H_plot, color="black", lw=2, label="Best Fit", zorder=5)
   if case.ramp_years <= 0:   # step-forcing solution of the diffusive-eps model
      ax.plot(plot_t, F_ref/eps - F_ref/eps*(1-erfcx(np.sqrt(plot_t*(YEAR*(lmbda/eps)**2)/(kappa_pde*RHO_CP**2)))),
              color="blue", lw=2, label="Analytical Solution")
   format_ax(ax, text=f"{label}", xscale="linear", yscale="linear", legend_loc="center right")
   annotate(ax, rf"{kappa_txt}, $\epsilon$ = {eps:.2f}, h$_{{ml}}^*$ = {h_ml_pde:.0f} m")

   spaghetti(figs.sens_dz, dz_grid, dz_curves,
             rf"{kappa_txt}, h$_{{ml}}$ = {h_ml_pde:.0f} m, {eps_txt}")

   for grid, values in ((figs.sens_t63, t63_grid), (figs.sens_rmse, rmse_grid)):
      ax = grid.axs[i]
      mesh = ax.pcolormesh(kappa_grid_2d, hml_grid_2d, values, cmap="viridis", shading="nearest")
      ax.scatter([kappa_pde], [h_ml_pde], marker="*", s=180, color="red",
                 edgecolor="black", linewidth=0.8, zorder=5, label="Best Fit")
      grid.artists.append((mesh, values))
      format_ax(ax, text=f"{label}", xscale="log", yscale="linear", legend=False, grid=False)


def plot_temperature_panel(figs, i, label, plot_t, plot_T, pde_T, T_eq, note,
                           analytic=None, pde_T_h0=None):
   """T2M against time: AOGCM, the fit, and optionally Held's analytic step
   response and the h_ml -> 0 numerical limit. Curves are passed as T/T_eq."""
   ax = figs.final.axs[i]
   ax.scatter(plot_t, T_eq*plot_T, s=4, color="red")
   ax.plot(plot_t, T_eq*plot_T, color="red", label="2-m Surface Temp.")
   if analytic is not None:
      ax.plot(plot_t, T_eq*analytic, color="blue", label="Diffusive Analytical")
   ax.plot(plot_t, T_eq*pde_T, color="purple", label="1-Box + Diffusion Fit")
   if pde_T_h0 is not None:
      ax.plot(plot_t, T_eq*pde_T_h0, color="purple", ls="-.",
              label=r"Diffusive Numerical (h$_{ml}\to0$, $\infty$ ocean)")
   annotate(ax, note, weight="bold")
   ax.axvline(151, color="orange", linestyle=":", linewidth=1, alpha=0.7)

   y_bottom, y_top = 1, 1.25 * T_eq
   ratio_ticks = np.arange(0.2, 1.21, 0.2)
   format_ax(ax, text=f"{label}", xscale="linear", yscale="linear", ylim=(y_bottom, y_top),
             yticks=ratio_ticks * T_eq, legend_loc='lower right')
   figs.xmax.append(np.max(plot_t))

   # Right-hand axis: the same curves as T/T_eq.
   ax2 = ax.twinx()
   ax2.set_ylim(y_bottom / T_eq, y_top / T_eq)
   ax2.set_yticks(ratio_ticks)
   ax2.tick_params(labelsize=14, width=2, length=8, direction="in")


def plot_nettoa_panel(figs, i, label, case, lmbda, plot_t, plot_T, pde_T, H_plot, nettoa_plot):
   """Net TOA against time (log y): AOGCM 10-yr mean, the fit's
   N = F - lambda*T - (eps-1)*H, and the same relation fed the AOGCM T."""
   def roll10(a):
      return np.array([np.mean(a[j:j + 10]) for j in range(a.shape[0] - 9)])

   T_eq, eps = case.T_eq, case.eps
   F_plot = case.F_ref * ramp_frac(plot_t, case.ramp_years)
   N_pde = F_plot - lmbda * (T_eq * pde_T) - (eps - 1.0) * H_plot
   N_obs = roll10(nettoa_plot)
   t_roll = plot_t[: N_obs.shape[0]]
   N_aogcm_eps = roll10(F_plot) - lmbda * roll10(T_eq * plot_T) - (eps - 1.0) * roll10(H_plot)

   ax = figs.nettoa.axs[i]
   ax.plot(t_roll, N_obs, color="black", label="AOGCM")
   ax.plot(plot_t, N_pde, color="purple", label="1-Box + Diffusion Fit")
   ax.plot(t_roll, N_aogcm_eps, color="darkorange", ls="--",
           label=r"$F-\lambda T_{AOGCM}-(\epsilon-1) H$ (10-yr mean)")
   format_ax(ax, text=f"{label}", xscale="linear", yscale="log", legend_loc="lower left")
