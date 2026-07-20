#!/usr/bin/env python3

import time
import multiprocessing
import rpy2.robjects as ro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
from pathlib import Path
import sympy as sp
from matplotlib.lines import Line2D
from tqdm import tqdm
from scipy.special import erfcx
from scipy.optimize import curve_fit, least_squares
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
# efficacy heat-uptake term (eps-1)*H is plotted vs time.
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

# Iterative heat-uptake-efficacy (epsilon) calibration, following Geoffroy
# 2013b's fit_radiative_epsilon: alternately (1) regress the AOGCM's Net TOA
# on T and the previous iteration's ocean heat uptake H to update F/lambda/eps,
# and (2) refit the PDE's kappa/h_ml at the new eps, repeating until the
# parameters stop moving. eps typically settles in a few iterations.
EPS_MIN_ITER = 2                 # always run at least this many iterations
EPS_MAX_ITER = 8                 # give up (use last iterate) after this many
EPS_CONVERGENCE_RTOL = 1e-1      # max relative parameter change to declare converged
EPS_FLOOR = 1e-3                 # clamp eps to >= this so a stray fit can't go <= 0
# Physical floors on the regressed forcing/feedback. The lstsq is unconstrained,
# so in degenerate corners (e.g. H nearly collinear with T in the binned
# regression) it can return F_ref <= 0 or lambda <= 0. A non-positive lambda
# flips the PDE's radiative damping (lam = F_ref/T_eq) into anti-damping and
# blows the solve up to inf/NaN, so floor both to small positive values. These
# only bind in degenerate cases -- well-posed fits sit far above them.
EPS_FREF_FLOOR = 0.1             # W/m^2
EPS_LAMBDA_FLOOR = 0.05          # W/m^2/K

# --- Reduced joint fit of (kappa, h_ml) at fixed radiative parameters ----------
# F_ref, lambda, eps and T_eq are all pinned by the deterministic STEP-1
# procedure (F_ref = intercept of the first-20-yr annual Net-TOA-vs-T fit; T_eq =
# x-intercept of the 50-yr-binned fit; lambda = F_ref/T_eq; eps = F_ref/binned-
# intercept = lambda / |binned slope|). Only the two PDE parameters remain free:
#   theta = [log10(kappa), h_ml]
# (log10 kappa for scale). They are fit by minimizing a single stacked residual --
# the AOGCM temperature timeseries AND the Net-TOA radiative balance N = F -
# lambda*T - (eps-1)*H -- each weighted by its own noise. With the radiative
# parameters fixed, the Net-TOA stream still constrains kappa/h_ml through the
# efficacy heat-uptake term (eps-1)*H (H is a property of the PDE state). The
# forward model is ONE PDE solve per residual evaluation (returns both T(t) and
# H), and the finite-difference Jacobian columns are farmed out across
# _SENS_POOL, so a full TRF step costs ~1 serial solve of wall-clock time.
RUN_JOINT_FIT = True
JOINT_MAX_NFEV = 200             # cap on residual evaluations (scipy least_squares)
JOINT_FTOL = 1e-8
JOINT_XTOL = 1e-8
JOINT_FD_REL_STEP = 1e-6         # relative forward-difference step for the parallel Jacobian
# Bounds, in theta order [log10(kappa), h_ml]. F_ref/lambda/eps/T_eq are fixed.
JOINT_BOUNDS_LO = np.array([np.log10(1e-7),  10])
JOINT_BOUNDS_HI = np.array([np.log10(1e-3), 300.0])
JOINT_PARAM_NAMES = ["kappa", "h_ml"]
# Diagnostic thresholds for the "is this parameter well optimized?" report.
JOINT_WIDTH_THRESH = 0.5         # flag a param WIDE if std/|value| exceeds this
JOINT_BOUND_RTOL = 1e-3          # flag AT-BOUND if within this frac of the bound span
JOINT_COND_THRESH = 1e8          # flag DEGENERATE if cond(J^T J) exceeds this

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


def _sensitivity_solve_dT(kappa, h_ml, F_ref, T_eq, t_final_years, t_eval, dz_target=15.0, epsilon=1.0):
   """PDE surface response for one sensitivity-sweep (kappa, h_ml, dz) point.

   Mirrors the per-point evaluation inside the main curve_fit target
   (make_pde_dT below), but at loosened tolerance and as a module-level
   function rather than a closure, so it can be dispatched to the
   multiprocessing pool -- sweep points only populate diagnostic plots,
   never the fitted kappa_pde/h_ml_pde themselves. dz_target defaults to the
   same spacing as the main fit; the dz sensitivity sweep overrides it
   directly (unclamped, so the full requested range from coarse to fine is
   actually exercised) while the kappa/h_ml sweeps leave it at the default.
   epsilon is the fitted heat-uptake efficacy (1.0 = plain diffusive model),
   passed so the sweep curves and the "Best Fit" line share the same eps.
   """
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref, epsilon=epsilon,
      z_max=z_max, Nz=nz, t_final=t_final_years * YEAR,
   )
   sol = pde_solve_model(
      pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, SENSITIVITY_N_SAVE),
      rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
   )
   return np.interp(t_eval, sol["t"] / YEAR, sol["dT"])


def _pde_grid(kappa, t_final_years, dz_target):
   """Shared z_max/Nz sizing used by make_pde_dT and pde_H_at, so the surface
   response and the heat-uptake diagnostic are read off the SAME grid."""
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   return z_max, nz


# Memoized full PDE solutions, keyed by every input that affects the solve.
# The best-fit solve gets re-read several times after fitting (final H
# regression, the reported temperature curve, the budget plot, H_plot);
# caching collapses those into ONE solve with bit-identical results. Bounded
# so curve_fit's many trial evaluations can't grow it without limit.
_PDE_SOL_CACHE = {}
_PDE_SOL_CACHE_MAX = 8


def _cached_pde_solve(kappa, h_ml, eps_, F_ref_, T_eq_, t_final_yrs_,
                      dz_target_, n_save, rtol=None, atol=None):
   """One (possibly cached) PDE solve; returns (pde_params, sol). rtol/atol
   None means the solver's tight defaults (the main-fit setting)."""
   key = (float(kappa), float(h_ml), float(eps_), float(F_ref_), float(T_eq_),
          float(t_final_yrs_), float(dz_target_), int(n_save),
          None if rtol is None else float(rtol),
          None if atol is None else float(atol))
   hit = _PDE_SOL_CACHE.get(key)
   if hit is not None:
      return hit
   z_max, nz = _pde_grid(kappa, t_final_yrs_, dz_target_)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq_, F0=F_ref_, epsilon=eps_,
      z_max=z_max, Nz=nz, t_final=t_final_yrs_ * YEAR,
   )
   kw = {} if rtol is None else {"rtol": rtol, "atol": atol}
   sol = pde_solve_model(pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, n_save), **kw)
   if len(_PDE_SOL_CACHE) >= _PDE_SOL_CACHE_MAX:
      _PDE_SOL_CACHE.clear()
   _PDE_SOL_CACHE[key] = (pde_p, sol)
   return pde_p, sol


def pde_H_at(t_years, kappa, h_ml, eps_, F_ref_, T_eq_, t_final_yrs_,
             dz_target_=PDE_FIT_DZ_TARGET):
   """Diffusive ocean heat uptake H(t) = rho_cp*kappa*dtheta/dz|_0 [W/m^2] at
   the requested years, for the efficacy calibration's radiative regression
   (N = F - lambda*T - (eps-1)*H). H is the flux the mixed layer loses DOWN
   into the thermocline; it is a property of the model state, so we evaluate
   it with the fitted eps in the surface budget. The surface gradient uses a
   2nd-order one-sided stencil (-3*theta0 + 4*theta1 - theta2)/(2*dz), and D =
   rho_cp*kappa; H > 0 when the surface is warmer than the water just below.
   """
   # If the solve fails/blows up, hand back zero uptake so the regression it
   # feeds gets finite (zero-signal) rows rather than NaNs that would poison
   # the lstsq. With F_ref/lambda floored (EPS_*_FLOOR) this should not trigger.
   try:
      pde_p, sol = _cached_pde_solve(kappa, h_ml, eps_, F_ref_, T_eq_,
                                     t_final_yrs_, dz_target_, PDE_FIT_N_SAVE)
      theta, dz = sol["theta"], sol["dz"]
      # dtheta/dz|_0 (z positive downward) via 2nd-order one-sided difference.
      dtheta_dz0 = (-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dz)
      H = pde_p.D * dtheta_dz0       # D = rho_cp*kappa; = -H_into_ocean's sign
      # theta decreases downward under surface warming (dtheta/dz0 < 0), so the
      # slab loses heat: flip sign so H > 0 means heat leaving the surface down.
      H = -H
      out = np.interp(t_years, sol["t"] / YEAR, H)
   except Exception:
      out = np.zeros(np.shape(t_years), dtype=float)
   return np.where(np.isfinite(out), out, 0.0)


def _sensitivity_solve_H(kappa, h_ml, F_ref, T_eq, t_final_years, t_eval,
                         dz_target=15.0, epsilon=1.0):
   """Diffusive ocean heat uptake H(t) = rho_cp*kappa*dtheta/dz|_0 [W/m^2] for one
   sensitivity-sweep point. Mirrors _sensitivity_solve_dT (loosened tolerance,
   module-level so it can be dispatched to _SENS_POOL) but returns the surface
   heat-uptake flux instead of the temperature, read off the SAME 2nd-order
   one-sided stencil as pde_H_at. H > 0 when the surface is warmer than the water
   just below; the caller scales it by (eps-1) for the efficacy uptake term."""
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(z_max / dz_target + 1)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref, epsilon=epsilon,
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


# =========================== Joint nonlinear fit ==============================
# These run under the fork pool during the parallel Jacobian, so they MUST be
# module-level (picklable) and defined BEFORE _SENS_POOL is created below.

def _joint_forward(theta, T_eq, eps, lam, fit_t, H_years, bin_H, t_final_years):
   """One PDE solve for the reduced joint fit; returns (T_model at fit_t, H at
   the regression years, 50-yr-binned iff bin_H). theta = [log10(kappa), h_ml];
   T_eq, eps and lam are the FIXED STEP-1 values and F0 = lam*T_eq. Loosened
   solver tolerance (SENSITIVITY_*), since this feeds a least-squares residual,
   not the final reported curve. Returns None on a failed/non-finite solve so
   the caller can substitute a large residual and steer the optimizer away."""
   log10_kappa, h_ml = theta
   kappa = 10.0 ** log10_kappa
   F0 = lam * T_eq
   z_max, nz = _pde_grid(kappa, t_final_years, PDE_FIT_DZ_TARGET)
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F0, epsilon=eps,
      z_max=z_max, Nz=nz, t_final=t_final_years * YEAR,
   )
   try:
      sol = pde_solve_model(
         pde_p, t_eval=_log_spaced_t_eval(pde_p.t_final, PDE_FIT_N_SAVE),
         rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL,
      )
      t_yr = sol["t"] / YEAR
      T_model = np.interp(fit_t, t_yr, sol["dT"])
      # Ocean heat uptake H(t) via the same 2nd-order surface-gradient stencil
      # as pde_H_at, so H matches the diagnostic used elsewhere.
      theta_grid, dz = sol["theta"], sol["dz"]
      dtheta_dz0 = (-3.0 * theta_grid[0] + 4.0 * theta_grid[1] - theta_grid[2]) / (2.0 * dz)
      H_full = -pde_p.D * dtheta_dz0
      H_at = np.interp(H_years, t_yr, H_full)
   except Exception:
      return None
   if bin_H:
      H_reg = np.array([H_at[i:i + 50].mean() for i in range(50, H_at.shape[0], 50)])
   else:
      H_reg = H_at
   if not (np.all(np.isfinite(T_model)) and np.all(np.isfinite(H_reg))):
      return None
   return T_model, H_reg


# Last (theta, residual) pair evaluated in THIS process. least_squares always
# evaluates fun(x) immediately before jac(x) at the same x, so _joint_jac can
# reuse that residual instead of re-solving the PDE at the base point (~1 solve
# saved per Jacobian). Parent-process only; pool workers never see it.
_JOINT_LAST_R = [None, None]


def _joint_residuals(theta, data):
   """Stacked, noise-weighted residual for the reduced joint fit:
      [ (T_model - T_obs)/sigma_T ,  (N_model - N_obs)/sigma_N ]
   with N_model = lambda*(T_eq - T_reg) - (eps-1)*H  (= F - lambda*T - (eps-1)H).
   theta = [log10(kappa), h_ml]; eps/lam/T_eq are fixed (data). Module-level so
   the parallel Jacobian can pool.starmap it."""
   out = _joint_forward(theta, data["T_eq"], data["eps"], data["lam"],
                        data["fit_t"], data["H_years"], data["bin_H"],
                        data["t_final_years"])
   if out is None:
      return data["big"]
   T_model, H_reg = out
   eps, lam = data["eps"], data["lam"]
   T_eq = data["T_eq"]
   N_model = lam * (T_eq - data["T_reg"]) - (eps - 1.0) * H_reg
   rT = (T_model - data["T_obs"]) / data["sigma_T"]
   rN = (N_model - data["N_obs"]) / data["sigma_N"]
   r = np.concatenate([rT, rN])
   _JOINT_LAST_R[0] = np.array(theta, dtype=float, copy=True)
   _JOINT_LAST_R[1] = r
   return r


def _joint_jac(theta, data):
   """Forward-difference Jacobian of _joint_residuals, with the n perturbed
   solves dispatched across _SENS_POOL (each column = one PDE solve, run in
   parallel), so a Jacobian costs ~1 solve of wall-clock time instead of n."""
   theta = np.asarray(theta, dtype=float)
   # Reuse the residual least_squares just computed at this same theta (fun(x)
   # always precedes jac(x) in TRF), avoiding a redundant PDE solve.
   if _JOINT_LAST_R[0] is not None and np.array_equal(_JOINT_LAST_R[0], theta):
      r0 = _JOINT_LAST_R[1]
   else:
      r0 = _joint_residuals(theta, data)
   n = theta.size
   step = JOINT_FD_REL_STEP * np.maximum(np.abs(theta), 1.0)
   pert = []
   for i in range(n):
      tp = theta.copy()
      tp[i] += step[i]
      pert.append((tp, data))
   cols = _SENS_POOL.starmap(_joint_residuals, pert)
   J = np.empty((r0.size, n), dtype=float)
   for i in range(n):
      J[:, i] = (cols[i] - r0) / step[i]
   return J


def run_joint_fit(theta0, data):
   """Bounded TRF least-squares over theta = [log10(kappa), h_ml] against the
   stacked temperature+radiative residual, with the parallel Jacobian. F_ref,
   lambda, eps and T_eq are all held FIXED at their deterministic STEP-1 values
   (data["lam"]/["eps"]/["T_eq"], F_ref = lam*T_eq). Returns a summary dict:
   fitted kappa/h_ml + 1-sigma errors (from the Jacobian covariance), the fixed
   radiative params, physical-unit RMSEs, convergence status, and the
   identifiability/bound diagnostics used by joint_fit_check."""
   lo, hi = JOINT_BOUNDS_LO, JOINT_BOUNDS_HI
   x0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
   sol = least_squares(
      _joint_residuals, x0, jac=_joint_jac, args=(data,),
      bounds=(lo, hi), method="trf", x_scale="jac",
      ftol=JOINT_FTOL, xtol=JOINT_XTOL, max_nfev=JOINT_MAX_NFEV,
   )
   theta = sol.x
   log10_kappa, h_ml = theta
   eps, lam, T_eq = data["eps"], data["lam"], data["T_eq"]   # fixed (STEP 1)
   kappa = 10.0 ** log10_kappa

   # Covariance from the final Jacobian (residuals already 1/sigma-weighted):
   # cov = s^2 (J^T J)^-1, s^2 = reduced chi-square. Reuses sol.jac -> no extra
   # PDE solves.
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
   # Delta-method: sigma_kappa = kappa * ln(10) * sigma_{log10 kappa}.
   kappa_err = kappa * np.log(10.0) * perr[0]
   errs = np.array([kappa_err, perr[1]])
   vals = np.array([kappa, h_ml])

   # Split the weighted residual back into physical-unit RMSEs.
   # least_squares already evaluated the residual at sol.x; reuse it instead of
   # paying one more PDE solve.
   r = sol.fun
   nT = data["T_obs"].size
   rmse_T = float(np.sqrt(np.mean((r[:nT] * data["sigma_T"]) ** 2)))
   rmse_N = float(np.sqrt(np.mean((r[nT:] * data["sigma_N"]) ** 2)))

   # Per-parameter diagnostics: relative width and whether it sits on a bound.
   # log10(kappa) width is judged in log space (its natural scale).
   span = hi - lo
   at_bound = (np.abs(theta - lo) < JOINT_BOUND_RTOL * span) | \
              (np.abs(theta - hi) < JOINT_BOUND_RTOL * span)
   relwidth = np.array([
      perr[0] / max(abs(log10_kappa), 1e-9),           # kappa: relative in log space
      perr[1] / max(abs(h_ml), 1e-9),
   ])

   return {
      "theta": theta, "vals": vals, "errs": errs, "perr": perr,
      "kappa": kappa, "h_ml": h_ml, "eps": eps, "T_eq": T_eq, "lam": lam,
      "F_ref": lam * T_eq,
      "kappa_err": kappa_err, "h_ml_err": perr[1], "eps_err": 0.0,
      "T_eq_err": 0.0, "lam_err": 0.0,
      "success": bool(sol.success), "status": int(sol.status),
      "message": str(sol.message), "nfev": int(sol.nfev), "njev": int(sol.njev),
      "cost": float(sol.cost), "optimality": float(sol.optimality),
      "reduced_chi2": s2, "cond": cond, "rmse_T": rmse_T, "rmse_N": rmse_N,
      "at_bound": at_bound, "relwidth": relwidth,
   }


def joint_fit_check(model, s):
   """Print the convergence + parameter-optimization report for one model and
   return an overall PASS/CHECK verdict. Flags: not-converged, any parameter
   railed to a bound, any parameter poorly constrained (wide error), or a
   degenerate (ill-conditioned) Jacobian."""
   conv = s["success"] and s["status"] > 0
   railed = bool(np.any(s["at_bound"]))
   wide = bool(np.any(s["relwidth"] > JOINT_WIDTH_THRESH))
   degenerate = s["cond"] > JOINT_COND_THRESH
   verdict = "PASS" if (conv and not railed and not wide and not degenerate) else "CHECK"
   print(f"   [joint] {model}: {verdict}  "
         f"converged={conv} (status={s['status']}, nfev={s['nfev']}, njev={s['njev']})  "
         f"reduced_chi2={s['reduced_chi2']:.2f}  cond(JtJ)={s['cond']:.1e}")
   print(f"      RMSE_T={s['rmse_T']:.4f} K   RMSE_N={s['rmse_N']:.4f} W/m^2")
   units = ["m^2/s", "m"]
   for name, v, e, rw, ab, u in zip(
         JOINT_PARAM_NAMES, s["vals"], s["errs"], s["relwidth"], s["at_bound"], units):
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


def covariance_from_lstsq(X, y, coeffs):
   """OLS coefficient covariance estimate sigma^2 (X'X)^-1 (copied from the
   2013b script) -- used for F/lambda/eps uncertainties from the radiative
   regression below."""
   n, p = X.shape
   resid = y - X @ coeffs
   dof = max(n - p, 1)
   sigma2 = float(np.sum(resid**2) / dof)
   return sigma2 * np.linalg.pinv(X.T @ X)


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
   """Scatter the AOGCM (T, H, N) points and overlay the multilinear regression
   plane N = F - lam*T - (eps-1)*H. Points are colored by signed residual
   (obs - plane), so the panel shows both the geometry of the fit and how
   tightly the data hug it. Returns (scatter, residuals) so the caller can put
   every model's panel on one shared symmetric color scale. Here H is the PDE's
   ocean heat uptake and (T, H, N) are the same rows the efficacy regression fit.
   """
   T = np.asarray(T, dtype=float); H = np.asarray(H, dtype=float); N = np.asarray(N, dtype=float)
   Nfit = F - lam * T - (eps - 1.0) * H
   resid = N - Nfit
   ss_res = float(np.sum(resid ** 2))
   ss_tot = float(np.sum((N - np.mean(N)) ** 2))
   r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
   rmse = float(np.sqrt(np.mean(resid ** 2)))

   sc = ax.scatter(T, H, N, c=resid, cmap=cmap, norm=resid_norm, s=12, depthshade=False)

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


def plot_surface_budget_bars(model, kappa, h_ml, eps, F_ref, T_eq, t_final_years,
                             png_path, dz_target=PDE_FIT_DZ_TARGET):
   """Stacked-bar decomposition of the surface tendency dT_s/dt for the best-fit
   PDE solution, one bar per saved time step. The surface node the solver
   integrates is

       c_ml * dT_s/dt = F - lambda*T_s - eps*H ,   H = D*(T_s - theta_1)/dz ,

   Regrouping the efficacy so the TOA-imbalance term N = F - lambda*T_s -
   (eps-1)*H is kept together and dividing by the mixed-layer heat capacity
   c_ml = rho_cp*h_ml gives three additive contributions [K/yr] that sum EXACTLY
   to dT_s/dt:

       +F/c_ml                          constant CO2 forcing (flat in time),
       -(lambda*T_s + (eps-1)*H)/c_ml   radiative restoring + efficacy loss,
       -H/c_ml                          plain diffusive heat loss to the thermocline.

   The top panel plots the |magnitude| of each term as stacked bars on a
   logarithmic y-axis (negative-sign terms are shown by magnitude so they fit the
   log scale), with the black line the |net dT_s/dt| and the dashed line the
   equilibration ratio T_s/T_eq on a right-hand axis. The lower panel repeats the
   same magnitudes as line plots. The surface gradient uses the SAME first-order
   forward stencil (theta_1 - T_s)/dz as the solver's surface node, so the terms
   close on the integrated tendency rather than a re-estimate. Time is drawn on a
   linear axis with each bar spanning the arithmetic midpoints to its neighbours.
   """
   pde_p, sol = _cached_pde_solve(kappa, h_ml, eps, F_ref, T_eq,
                                  t_final_years, dz_target, PDE_FIT_N_SAVE)
   t_yr = sol["t"] / YEAR
   theta, dz = sol["theta"], sol["dz"]
   T_s = theta[0]
   c_ml, lam, D = pde_p.c_ml, pde_p.lam, pde_p.D      # lam = F_ref/T_eq exactly

   # Per-second tendencies -> K/yr for readability; all three sum to dT_s/dt.
   forcing   = np.full_like(T_s, F_ref) / c_ml * YEAR
   H_flux    = D * (T_s - theta[1]) / dz             # W/m^2, >0 when surface warmer
   restoring = (-lam * T_s - (eps - 1.0) * H_flux) / c_ml * YEAR   # = (N - F)/c_ml
   uptake    = (-H_flux) / c_ml * YEAR
   net       = forcing + restoring + uptake          # = dT_s/dt [K/yr]

   comps = [
      (r"$F/c_{ml}$",                                    forcing,   "#9467bd"),
      (r"$-(\lambda T_s + (\epsilon-1)H)/c_{ml}$",       restoring, "#1f77b4"),
      (r"$-H/c_{ml}$",                                   uptake,    "#2ca02c"),
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
                rf"($\kappa$={kappa:.1e}, $h_{{ml}}$={h_ml:.0f} m, $\epsilon$={eps:.2f})",
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
outdir = Path("./figures_diffusive-eps-joint")
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
         short and sits directly beneath its main axis."""
         nmodels = len(models)
         if nmodels % ncols != 0:
            raise ValueError(f"Expected models divisible by ncols={ncols}, got {nmodels}.")
         nmodel_rows = nmodels // ncols

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

         if title:
            fig.suptitle(title, fontsize=20, fontweight="bold")

         if xlabel:
            fig.text(0.5, 0.02, xlabel, ha='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")

         if ylabel:
            fig.text(0.02, 0.5, ylabel, ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=90.)

         return fig, np.asarray(main_axs), np.asarray(ratio_axs)


      def ensure_dirs(outdir, current_dir, sections):
         for section in sections:
            (outdir / current_dir / section / "png").mkdir(parents=True, exist_ok=True)

      # Create dataframe to store model parameters
      param_cols = [
         "model",
         "F_ref",
         "lambda",
         "T_eq",
         "epsilon",
         "kappa_pde",
         "h_ml_pde",
         "F_ref_unc",
         "lambda_unc",
         "T_eq_unc",
         "epsilon_unc",
      ]
      df = pd.DataFrame(columns=param_cols)
      df["model"] = models

      # STEP 1 stashes each model's radiative-regression arrays (the same
      # 50-yr-binned-or-annual T2M/NETTOA it fit the Gregory line to) here, so
      # STEP 2's iterative efficacy fit regresses N on T and H over EXACTLY the
      # same points -- keeping eps consistent with the F_ref/lambda seed.
      # reg_arrays[model] = (t_years, t2m, nettoa) with matching rows.
      reg_arrays = {}

      # Handle to each model's STEP 1 (Net TOA vs T2M) axis, so STEP 2 can
      # overlay the efficacy-corrected prediction N = F - lambda*T - (eps-1)*H
      # once eps/kappa/h_ml/H are known. The STEP 1 figure's save is deferred
      # until after STEP 2 for the same reason.
      step1_ax_by_model = {}
      step1_ratio_ax_by_model = {}

      # 3D (T, H, N) regression figure: one panel per model, populated in STEP 2
      # (which computes H and the fitted eps), saved after STEP 2. reg3d_scatters
      # collects (scatter, residuals) so every panel shares one residual colorbar.
      reg3d_fig, reg3d_axs = make_model_grid_3d(
         models,
         title=r"1-Box + Diffusion efficacy fit: N = F $-\ \lambda$T $-\ (\epsilon-1)$H",
      )
      reg3d_idx = 0
      reg3d_scatters = []

      ensure_dirs(outdir, current_dir, ["step1", "validation", "unblinded", "budget"])

      # Prepare combined figures for each experiment
      step1_figs = {}
      step1_axs = {}
      step1_ratio_axs = {}
      step1_idx = {}
      step1_NH_figs = {}
      step1_NH_axs = {}
      step1_NH_idx = {}
      step1_NH_ax_by_model = {}
      step1_HT_figs = {}
      step1_HT_axs = {}
      step1_HT_idx = {}
      step1_HT_ax_by_model = {}
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
         step1_figs[expt], step1_axs[expt], step1_ratio_axs[expt] = make_model_grid_with_ratio(
            models, title=r"4xCO$_{2}$ Net TOA vs T$_{2M}$",
            xlabel="2-meter Air Temperature Anomaly (K)",
            ylabel=r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)",
            ratio_ylabel=r"$\frac{F-\lambda T-(\epsilon-1)H}{F-\lambda T}$",
         )
         step1_idx[expt] = 0

         step1_NH_figs[expt], step1_NH_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs Ocean Heat Uptake H", xlabel=r"Ocean Heat Uptake H ($W*m^{-2}$)", ylabel=r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)")
         step1_NH_idx[expt] = 0

         step1_HT_figs[expt], step1_HT_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Ocean Heat Uptake H vs T$_{2M}$", xlabel="2-meter Air Temperature Anomaly (K)", ylabel=r"Ocean Heat Uptake H ($W*m^{-2}$)")
         step1_HT_idx[expt] = 0

         final_figs[expt], final_axs[expt] = make_model_grid(models, dpi=120, title=r"4xCO$_{2}$ T$_{2M}$ vs. Time w/ Diffusive Fit", xlabel=r"Time (years)", ylabel=r"Temperature Anomaly (K)", right=0.95, wspace=0.28)
         final_idx[expt] = 0
         final_xmax[expt] = []
         final_figs[expt].text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=-90.)

         nettoa_figs[expt], nettoa_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs. Time", xlabel="Time (years)", ylabel=r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)")
         nettoa_idx[expt] = 0

         ohc_ts_figs[expt], ohc_ts_axs[expt] = make_model_grid(models, title=r"4xCO$_2$ OHU vs. Surface Temp (Normalized)", xlabel=r"$T_s/2\, \mathrm{ECS}$", ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$")
         ohc_ts_idx[expt] = 0

         sens_kappa_figs[expt], sens_kappa_axs[expt] = make_model_grid(models, title=r"Sensitivity to $\kappa$ (h$_{ml}$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_kappa_idx[expt] = 0

         sens_hml_figs[expt], sens_hml_axs[expt] = make_model_grid(models, title=r"Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_hml_idx[expt] = 0

         sens_hml_uptake_figs[expt], sens_hml_uptake_axs[expt] = make_model_grid(models, title=r"Ocean Heat Uptake Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)", xlabel="Time (years)", ylabel=r"$H$ (W m$^{-2}$)")
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

            # Flatten T2M and NETTOA arrays from R data. The deterministic
            # F_ref/lambda/eps/T_eq procedure below is kept IDENTICAL across run
            # types (these are physical properties of each model, not of the plot
            # window), so the full timeseries is always used -- run_type 1 is no
            # longer truncated to 150 years for this parameter estimation.
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

               # Annual anomalies, kept for (a) the first-20-yr F_ref fit, (b) the
               # 50-yr-binned T_eq/eps fit, and (c) STEP 2's efficacy regression
               # (which evaluates the PDE heat uptake H annually then 50-yr-bins
               # it the SAME way as the binned regression data below).
               t2m_annual_reg = t2m.copy()
               nettoa_annual_reg = nettoa.copy()

               # ---- F_ref: intercept of the first-20-yr ANNUAL Net-TOA-vs-T fit.
               n20 = min(20, t2m_annual_reg.shape[0])
               [m20, b20], cov20 = np.polyfit(
                  t2m_annual_reg[:n20], nettoa_annual_reg[:n20], 1, cov=True)
               m20_unc = float(np.sqrt(cov20[0, 0]))
               b20_unc = float(np.sqrt(cov20[1, 1]))

               # ---- 50-yr binned fit (FIRST bin skipped, matching the existing
               # convention) -> T_eq (x-intercept) and eps. Uses the full data for
               # every run type (run_type 1 included).
               t2m_bin = np.array([t2m_annual_reg[i:i + 50].mean()
                                   for i in range(50, t2m_annual_reg.shape[0], 50)])
               nettoa_bin = np.array([nettoa_annual_reg[i:i + 50].mean()
                                      for i in range(50, nettoa_annual_reg.shape[0], 50)])
               [mb, bb], covb = np.polyfit(t2m_bin, nettoa_bin, 1, cov=True)
               mb_unc = float(np.sqrt(covb[0, 0]))
               bb_unc = float(np.sqrt(covb[1, 1]))

               # Downstream plotting/regression uses the 50-yr bins as the
               # "regression" points (as the old run_type != 1 path did).
               t2m = t2m_bin
               nettoa = nettoa_bin

               # ---- Deterministic radiative parameters ----------------------
               #   F_ref = 20-yr intercept ;  T_eq = binned x-intercept (-bb/mb)
               #   lambda = F_ref/T_eq     ;  eps = F_ref/bb = lambda/|binned slope|
               # (binned intercept bb = F/eps and binned slope mb = -lambda/eps,
               # so eps = F_ref/bb follows and stays self-consistent with lambda.)
               F_ref = max(float(b20), EPS_FREF_FLOOR)
               T_eq_raw = float(-bb / mb)
               T_eq = T_eq_raw if T_eq_raw > 0 else 1.0     # guard degenerate x-int
               lam = max(F_ref / T_eq, EPS_LAMBDA_FLOOR)
               eps = max(F_ref / bb, EPS_FLOOR)

               # First-order uncertainty propagation through the two fits'
               # coefficients (uncorrelated, matching the script's convention).
               sp_b20, sp_mb, sp_bb = sp.symbols("b20 mb bb")
               F_ref_unc = b20_unc
               T_eq_unc = sympy_prop_unc(-sp_bb / sp_mb,
                                         {sp_bb: bb, sp_mb: mb},
                                         {sp_bb: bb_unc, sp_mb: mb_unc})
               lambda_unc = sympy_prop_unc(-sp_b20 * sp_mb / sp_bb,
                                           {sp_b20: b20, sp_mb: mb, sp_bb: bb},
                                           {sp_b20: b20_unc, sp_mb: mb_unc, sp_bb: bb_unc})
               eps_unc = sympy_prop_unc(sp_b20 / sp_bb,
                                        {sp_b20: b20, sp_bb: bb},
                                        {sp_b20: b20_unc, sp_bb: bb_unc})

               # ---- STEP 1 panel: Net TOA vs T2M with BOTH provenance fits ----
               ax = step1_axs[expt][step1_idx[expt]]
               # 50-yr bins + their fit (source of T_eq and eps); draw the line
               # out to T_eq so the x-intercept is visible on the panel.
               ax.scatter(t2m_bin, nettoa_bin, s=10, alpha=0.6, color="tab:blue",
                          label="50-yr bins (data)")
               xb = np.linspace(min(float(t2m_bin.min()), 0.0), T_eq, 100)
               ax.plot(xb, mb * xb + bb, color="tab:blue", lw=2,
                       label=rf"50-yr fit: $T_{{eq}}$={T_eq:.1f} K, $\epsilon$={eps:.2f}")
               # First-20-yr annual points + their fit (source of F_ref).
               ax.scatter(t2m_annual_reg[:n20], nettoa_annual_reg[:n20], s=8, alpha=0.5,
                          color="0.5", label="yrs 1-20 (annual)")
               x20 = np.linspace(0.0, float(t2m_annual_reg[:n20].max()), 100)
               ax.plot(x20, m20 * x20 + b20, color="0.35", lw=2,
                       label=rf"20-yr fit: $F_{{ref}}$={F_ref:.3f}")
               ax.axhline(0.0, color="0.7", lw=0.8, ls=":")

               # R^2 / RMSE of the binned fit on this panel.
               y_pred_bin = mb * t2m_bin + bb
               ss_res = np.sum((nettoa_bin - y_pred_bin) ** 2)
               ss_tot = np.sum((nettoa_bin - np.mean(nettoa_bin)) ** 2)
               r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
               rmse = np.sqrt(np.mean((nettoa_bin - y_pred_bin) ** 2))
               ax.text(
                  0.02,
                  0.80,
                  f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )

               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear")
               step1_ax_by_model[model] = ax   # STEP 2 overlays the eps*H curve here
               step1_ratio_ax_by_model[model] = step1_ratio_axs[expt][step1_idx[expt]]
               step1_idx[expt] += 1

               ax_nh = step1_NH_axs[expt][step1_NH_idx[expt]]
               step1_NH_ax_by_model[model] = ax_nh   # STEP 2 draws the N-vs-H fit here
               step1_NH_idx[expt] += 1

               step1_HT_ax_by_model[model] = step1_HT_axs[expt][step1_HT_idx[expt]]
               step1_HT_idx[expt] += 1

               # Record the deterministic 4xCO2 radiative params + uncertainties.
               df.loc[df["model"] == model, "F_ref"] = F_ref
               df.loc[df["model"] == model, "lambda"] = lam
               df.loc[df["model"] == model, "T_eq"] = T_eq
               df.loc[df["model"] == model, "epsilon"] = eps
               df.loc[df["model"] == model, "F_ref_unc"] = F_ref_unc
               df.loc[df["model"] == model, "lambda_unc"] = lambda_unc
               df.loc[df["model"] == model, "T_eq_unc"] = T_eq_unc
               df.loc[df["model"] == model, "epsilon_unc"] = eps_unc

               # Stash BOTH the 50-yr-binned (T2M, NETTOA) -- used by the STEP-2
               # regression-DIAGNOSTIC plots (N-vs-H, H-vs-T, 3D, efficacy
               # overlay), which stay binned -- AND the full annual (unbinned)
               # series, used by the joint fit (which now bins NEITHER N/T nor H).
               # H_years is the annual grid at which STEP 2 evaluates H.
               reg_arrays[model] = {
                  "t2m": t2m_bin,
                  "nettoa": nettoa_bin,
                  "t2m_annual": t2m_annual_reg,
                  "nettoa_annual": nettoa_annual_reg,
                  "H_years": np.arange(1, 1 + t2m_annual_reg.shape[0]),
                  "bin_H": True,
               }

      # NOTE: the STEP 1 (Net TOA vs T2M) figures are intentionally NOT saved
      # here. STEP 2 overlays the efficacy-corrected prediction
      # N = F - lambda*T - (eps-1)*H onto each panel once eps/kappa/h_ml are
      # fit, and the figures are saved/closed after that (see below).
      print("Finished Step 1: Gregory params to df (Step 1 figures saved after Step 2)")

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

               def make_pde_dT(F_ref_, T_eq_, t_final_yrs_, eps_=1.0, dz_target_=15.0,
                               rtol_=None, atol_=None):
                  # rtol_/atol_ None = the solver's tight defaults (final reported
                  # curves); the eps=1 SEED fit passes the loosened SENSITIVITY_*
                  # tolerances since it only sets theta0/sigma_T for the joint fit.
                  def pde_dT(t_years, kappa, h_ml):
                     # Guard the curve_fit residual: if the solver fails or
                     # returns a non-finite series at some probed (kappa, h_ml),
                     # hand back a large finite sentinel so least_squares steers
                     # away instead of crashing on a NaN Jacobian.
                     try:
                        _, sol = _cached_pde_solve(kappa, h_ml, eps_, F_ref_, T_eq_,
                                                   t_final_yrs_, dz_target_,
                                                   PDE_FIT_N_SAVE, rtol=rtol_, atol=atol_)
                        dT = np.interp(t_years, sol["t"] / YEAR, sol["dT"])
                     except Exception:
                        dT = np.full(np.shape(t_years), 1e6, dtype=float)
                     if not np.all(np.isfinite(dT)):
                        dT = np.full(np.shape(t_years), 1e6, dtype=float)
                     return dT
                  return pde_dT

               # Actual AOGCM absolute temperature anomaly over the fit/plot
               # windows [K]. These are FIXED data; the curve_fit target must
               # stay pinned to them even as T_eq = F_ref/lambda drifts across
               # efficacy iterations (fit_T/plot_T are ratios that we recompute
               # from the final T_eq afterward for the equilibrium-ratio plots).
               T_eq_seed = T_eq
               fit_T_abs = fit_T * T_eq_seed
               plot_T_abs = plot_T * T_eq_seed

               # ----- Reduced joint fit of (kappa, h_ml) at fixed radiative params
               # F_ref, lambda, eps and T_eq are all fixed by the deterministic
               # STEP-1 procedure; only the two PDE parameters are optimized here,
               # by one bounded TRF least-squares over a single stacked residual
               # (AOGCM temperature timeseries + Net-TOA radiative balance). The
               # Net-TOA stream still constrains kappa/h_ml through the efficacy
               # heat-uptake term (eps-1)*H. See run_joint_fit and JOINT_* above.
               ra = reg_arrays[model]
               # Binned arrays (unchanged): feed the STEP-2 regression-diagnostic
               # plots, which stay 50-yr-binned.
               T_reg, N_reg = np.asarray(ra["t2m"], float), np.asarray(ra["nettoa"], float)
               H_years, bin_H = ra["H_years"], ra["bin_H"]
               # Annual (unbinned) all-year arrays: the joint fit bins NEITHER
               # N/T nor H, so its Net-TOA residual runs over these instead.
               T_ann = np.asarray(ra["t2m_annual"], float)
               N_ann = np.asarray(ra["nettoa_annual"], float)

               # Efficacy eps is FIXED (deterministic STEP-1 value); read it here.
               eps = float(df.loc[df["model"] == model, "epsilon"].iloc[0])

               # Seed kappa/h_ml from a cheap fit at the FIXED eps, and set the
               # residual noise scales sigma_T (temperature-fit RMSE) and sigma_N
               # (annual Gregory RMSE) that weight the two data streams in the
               # joint objective. Only kappa/h_ml are optimized now.
               pde_dT_func = make_pde_dT(F_ref, T_eq, t_final_years, eps, dz_target_=PDE_FIT_DZ_TARGET,
                                         rtol_=SENSITIVITY_RTOL, atol_=SENSITIVITY_ATOL)
               # Seed-fit bounds track the joint-fit bounds so the two stay in
               # sync (JOINT_BOUNDS_* stores log10(kappa); convert to linear here).
               popt_seed, _ = curve_fit(
                  pde_dT_func, fit_t, fit_T_abs,
                  p0=[1.0e-4, 100.0],
                  bounds=([10.0**JOINT_BOUNDS_LO[0], JOINT_BOUNDS_LO[1]],
                          [10.0**JOINT_BOUNDS_HI[0], JOINT_BOUNDS_HI[1]]),
                  max_nfev=60,
               )
               kappa0, h_ml0 = popt_seed
               sigma_T = max(1e-2, float(np.sqrt(np.mean(
                  (pde_dT_func(fit_t, kappa0, h_ml0) - fit_T_abs) ** 2))))
               sigma_N = max(5e-2, float(np.sqrt(np.mean(
                  (N_ann - (F_ref - lmbda * T_ann)) ** 2))))

               joint_data = {
                  "fit_t": np.asarray(fit_t, float), "T_obs": np.asarray(fit_T_abs, float),
                  "T_reg": T_ann, "N_obs": N_ann,         # annual, UNBINNED
                  "T_eq": float(T_eq_seed),               # STEP-1 T_eq (fixed)
                  "eps": float(eps), "lam": float(lmbda), # STEP-1 eps/lambda (fixed)
                  "H_years": np.asarray(H_years, float), "bin_H": False,   # H not binned
                  "t_final_years": float(t_final_years),
                  "sigma_T": sigma_T, "sigma_N": sigma_N,
                  "big": np.full(int(fit_t.shape[0] + N_ann.shape[0]), 1e3),
               }
               theta0 = np.array([
                  np.log10(min(max(kappa0, 10.0**JOINT_BOUNDS_LO[0]), 10.0**JOINT_BOUNDS_HI[0])),
                  min(max(h_ml0, JOINT_BOUNDS_LO[1]), JOINT_BOUNDS_HI[1]),
               ])
               joint = run_joint_fit(theta0, joint_data)

               # Adopt the fitted kappa/h_ml. F_ref/lambda/eps/T_eq remain at their
               # deterministic STEP-1 values (run_joint_fit passes them through).
               kappa_pde, h_ml_pde = joint["kappa"], joint["h_ml"]
               pde_dT_func = make_pde_dT(F_ref, T_eq, t_final_years, eps, dz_target_=PDE_FIT_DZ_TARGET)

               # Convergence + parameter-optimization report for this model.
               joint_fit_check(model, joint)
               print(f"   [joint] {model}: kappa={kappa_pde:.2e} h_ml={h_ml_pde:.0f} | "
                     f"fixed F={F_ref:.2f} lam={lmbda:.2f} T_eq={T_eq:.2f} eps={eps:.3f}")

               # F_ref/lambda/eps/T_eq were fixed in STEP 1 (already in df); only
               # kappa/h_ml and the joint diagnostics are written here. T_eq did
               # not move, so fit_T/plot_T (the equilibrium ratios set earlier)
               # are unchanged.
               df.loc[df["model"] == model, "kappa_pde"] = kappa_pde
               df.loc[df["model"] == model, "h_ml_pde"] = h_ml_pde
               # Joint-fit uncertainties + convergence/optimization diagnostics.
               df.loc[df["model"] == model, "kappa_pde_unc"] = joint["kappa_err"]
               df.loc[df["model"] == model, "h_ml_pde_unc"] = joint["h_ml_err"]
               df.loc[df["model"] == model, "joint_converged"] = joint["success"]
               df.loc[df["model"] == model, "joint_reduced_chi2"] = joint["reduced_chi2"]
               df.loc[df["model"] == model, "joint_cond"] = joint["cond"]
               df.loc[df["model"] == model, "joint_rmse_T"] = joint["rmse_T"]
               df.loc[df["model"] == model, "joint_rmse_N"] = joint["rmse_N"]
               df.loc[df["model"] == model, "joint_any_at_bound"] = bool(np.any(joint["at_bound"]))

               # ----- Overlay the eps*H contribution on the STEP 1 panel -----
               # The STEP 1 scatter is Net TOA vs T2M with a straight Gregory
               # line N = F - lambda*T. Add the efficacy-corrected prediction
               # N = F - lambda*T - (eps-1)*H over the SAME regression points
               # (H = the fitted PDE's ocean heat uptake). The gap between this
               # curve and the straight line is exactly the efficacy's -(eps-1)*H
               # contribution; it should hug the scatter better than the line.
               # H at the annual years: keep an unbinned copy (H_annual) for the
               # N-vs-T panel's annual overlays, plus the 50-yr-binned H_reg_final
               # used by the other diagnostic panels (N-vs-H, H-vs-T, 3D, ratio).
               H_annual = pde_H_at(H_years, kappa_pde, h_ml_pde, eps,
                                    F_ref, T_eq, t_final_years)
               H_reg_final = H_annual
               if bin_H:
                  H_reg_final = np.array([H_annual[i:i + 50].mean()
                                          for i in range(50, H_annual.shape[0], 50)])
               T_reg_arr = np.asarray(T_reg, dtype=float)
               ax1 = step1_ax_by_model[model]

               # ----- N-vs-T overlays at ANNUAL (all-year) resolution -----
               # The straight radiative line N = F - lambda*T (fixed STEP-1
               # F_ref/lambda) and the efficacy-corrected N = F - lambda*T -
               # (eps-1)*H are drawn over the ENTIRE annual series instead of the
               # 50-yr bins, and the annual (T, N) AOGCM points are added as a
               # faint cloud beneath the 50-yr-bin points already on the axis.
               N_line_ann = F_ref - lmbda * T_ann
               N_eps_ann = N_line_ann - (eps - 1.0) * H_annual
               rmse_eps = float(np.sqrt(np.mean((N_ann - N_eps_ann) ** 2)))
               order_ann = np.argsort(T_ann)
               ax1.scatter(T_ann, N_ann, s=5, alpha=0.15, color="tab:red",
                           zorder=0, label="annual data")
               ax1.plot(T_ann[order_ann], N_line_ann[order_ann], color="tab:orange", lw=2, ls="-.",
                        label=rf"$F-\lambda T$ (fixed): $F$={F_ref:.3f}, $\lambda$={lmbda:.3f}")
               ax1.plot(T_ann[order_ann], N_eps_ann[order_ann], color="green", lw=2, ls="--",
                        label=rf"w/ $(\epsilon-1)H$: $\epsilon$={eps:.2f} (RMSE={rmse_eps:.3f})")
               ax1.legend(loc="upper right", prop={"weight": "bold", "size": 8})

               # ----- Ratio panel beneath the STEP 1 panel -----
               # Ratio of the green (efficacy) line to the orange (plain Gregory)
               # line, both over the ENTIRE annual series and using the SAME fixed
               # F_ref/lambda -- isolates how much the -(eps-1)*H term alone is
               # doing relative to the straight F - lambda*T line.
               ratio_ax = step1_ratio_ax_by_model[model]
               # Guard the ratio where the plain Gregory line N=F-lambda*T passes
               # through zero (T ~ T_eq): 0/0 there is undefined and, with the
               # ratio y-axis now SHARED across all model panels, one divergent
               # point would blow up the common scale. Mask that thin band to nan.
               denom = np.where(np.abs(N_line_ann) > 0.02 * abs(F_ref),
                                N_line_ann, np.nan)
               ratio_vals = N_eps_ann / denom
               ratio_ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
               ratio_ax.plot(T_ann[order_ann], ratio_vals[order_ann], color="green", lw=1.2)
               ratio_ax.tick_params(labelsize=12, width=2, length=6, direction="in")
               ratio_ax.grid(alpha=0.3)

               # ----- STEP 1 (N vs H) panel: fit of N against the ocean heat -----
               # uptake H, with the pure-efficacy term N = -(eps-1)*H and the
               # full regression plane evaluated at each point's own T,
               # N = F - lambda*T - (eps-1)*H, both plotted vs. H over the same
               # regression rows the efficacy fit used (in series order, H unsorted).
               N_eps_line = F_ref - lmbda * T_reg_arr - (eps - 1.0) * H_reg_final
               ax_nh = step1_NH_ax_by_model[model]
               ax_nh.scatter(H_reg_final, N_reg, s=8, alpha=0.5, label="Data")
               one_one = np.array([H_reg_final.min(), H_reg_final.max()])
               ax_nh.plot(one_one, one_one, color="0.5", lw=1.0, ls="-.", label="N = H (Analytical Sol.)")
               ax_nh.plot(H_reg_final, N_eps_line, color="green", lw=2, ls="--",
                          label=r"$N=F-\lambda T-(\epsilon-1) H$")
               format_ax(ax_nh, text=f"{model}", xscale="linear", yscale="linear",
                         legend_loc="lower right")

               # ----- STEP 1 (H vs T) panel: ocean heat uptake H vs surface -----
               # temperature. PDE fit = the fitted PDE's H at the regression rows
               # (light-blue line); AOGCM = H inverted from the efficacy radiative
               # relation using the AOGCM N, H = (F - N - lambda*T)/(eps-1) (green
               # dots). Both H and T are 10-yr running-mean smoothed. If the fit is
               # good the AOGCM dots track the PDE line.
               ax_ht = step1_HT_ax_by_model[model]
               N_reg_arr = np.asarray(N_reg, dtype=float)

               # 10-year running mean over the paired (T, H) regression rows.
               # Rows are annual when bin_H is False (1 yr/row) and 50-yr-binned
               # otherwise, so size the window to ~10 years and skip smoothing
               # when a single row already spans >=10 years.
               yrs_per_row = 50 if bin_H else 1
               roll_win = max(1, int(round(10.0 / yrs_per_row)))
               def roll10(a):
                  a = np.asarray(a, dtype=float)
                  if roll_win <= 1 or a.shape[0] < roll_win:
                     return a
                  return np.array([a[i:i + roll_win].mean()
                                   for i in range(a.shape[0] - roll_win + 1)])

               T_reg_roll = roll10(T_reg_arr)
               H_reg_roll = roll10(H_reg_final)
               order_T = np.argsort(T_reg_roll)
               ax_ht.plot(T_reg_roll[order_T], H_reg_roll[order_T], color="lightblue",
                          lw=2, label="H (PDE fit, 10-yr mean)")
               ax_ht.plot(T_reg_roll[order_T], F_ref/eps - (lmbda/eps)*T_reg_roll[order_T], color="0.5", lw=1.0, ls="-.", label="H (analytical sol.)")
               if abs(eps - 1.0) > 1e-6:
                  H_inv = (F_ref - N_reg_arr - lmbda * T_reg_arr) / (eps - 1.0)
                  H_inv_roll = roll10(H_inv)
                  ax_ht.scatter(T_reg_roll, H_inv_roll, s=8, alpha=0.5, color="green",
                                label=r"$H=(F-N-\lambda T)/(\epsilon-1)$ (AOGCM, 10-yr mean)")
               format_ax(ax_ht, text=f"{model}\n$\\epsilon$={eps:.2f}", xscale="linear",
                         yscale="linear", legend_loc="lower left")

               # 3D view of the same multilinear regression: (T, H, N) scatter +
               # the fitted plane N = F - lambda*T - (eps-1)*H, one panel/model.
               sc3d, resid3d = plot_regression_3d(
                  reg3d_axs[reg3d_idx], T_reg_arr, H_reg_final,
                  np.asarray(N_reg, dtype=float), F_ref, lmbda, eps, model,
               )
               reg3d_scatters.append((sc3d, resid3d))
               reg3d_idx += 1

               pde_T = pde_dT_func(plot_t, kappa_pde, h_ml_pde) / T_eq

               # Stacked-bar decomposition of the surface tendency dT_s/dt into
               # F/c_ml, -lambda*T/c_ml and -eps*H/c_ml for the best-fit solve.
               plot_surface_budget_bars(
                  model, kappa_pde, h_ml_pde, eps, F_ref, T_eq, t_final_years,
                  outdir / current_dir / "budget" / "png" / f"{expt}_{model}_surface_budget_bars{suffix}.png",
               )

               # Ocean heat uptake H(t) [W/m^2] over the plot window, for the
               # efficacy TOA decomposition N = F - lambda*T - (eps-1)*H below
               # (matches 2013b's Net TOA reconstruction; H = 0 recovers the
               # plain N = F - lambda*T when eps = 1).
               H_plot = pde_H_at(plot_t, kappa_pde, h_ml_pde, eps,
                                  F_ref, T_eq, t_final_years)

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
                  # Reference heat content = rho_cp * z_max * T_eq over the whole
                  # ocean area, where z_max is the fit's ACTUAL grid depth (the
                  # column that warms to T_eq at equilibrium). Computed from this
                  # model's fitted kappa_pde with the same _pde_grid sizing the PDE
                  # solve used, so normalized_OHC_pred/T_eq -> 1 at equilibrium
                  # instead of z_max/2700.
                  z_max, _ = _pde_grid(kappa_pde, t_final_years, PDE_FIT_DZ_TARGET)
                  ohc_ref = 5.1e14 * 4.186e6 * z_max          # OHC_eq / T_eq  [J/K]
                  normalized_OHC = (5.1e14 * 31536000 * np.cumsum(nettoa)) / (1.37e21 * 3850)

                  N_pde = F_ref - lmbda * (T_eq * pde_T) - (eps - 1.0) * H_plot
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
                     [(k, h_ml_pde, F_ref, T_eq, t_final_years, plot_t, PDE_FIT_DZ_TARGET, eps) for k in kappa_grid_1d],
                  )
               ]
               h_ml_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(kappa_pde, h, F_ref, T_eq, t_final_years, plot_t, PDE_FIT_DZ_TARGET, eps) for h in h_ml_grid_1d],
                  )
               ]

               # Ocean-heat-uptake sensitivity to h_ml: sweep h_ml over a FIXED
               # log-spaced range [1e-3, 1000] m (kappa/eps held at best fit) and
               # record the diffusive heat-uptake flux H(t) [W/m^2], with
               # H = rho_cp*kappa*dtheta/dz|_0 the diffusive surface flux.
               h_ml_uptake_grid = np.geomspace(
                  SENSITIVITY_UPTAKE_HML_RANGE[0], SENSITIVITY_UPTAKE_HML_RANGE[1],
                  SENSITIVITY_UPTAKE_N,
               )
               h_ml_uptake_curves = _SENS_POOL.starmap(
                  _sensitivity_solve_H,
                  [(kappa_pde, h, F_ref, T_eq, t_final_years, plot_t, PDE_FIT_DZ_TARGET, eps) for h in h_ml_uptake_grid],
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
                     [(kappa_pde, h_ml_pde, F_ref, T_eq, t_final_years, plot_t, dz, eps) for dz in dz_grid_1d],
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
                  (k_val, h_val, F_ref, T_eq, t_final_years, t_grid_common, PDE_FIT_DZ_TARGET, eps)
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
                  rf"h$_{{ml}}$ = {h_ml_pde:.0f} m, $\epsilon$ = {eps:.2f} (fixed)",
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
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s, $\epsilon$ = {eps:.2f} (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_hml_idx[expt] += 1

               # --- Panel 2b: ocean-heat-uptake sensitivity to h_ml ---
               # H(t) as h_ml is swept log-uniformly over [1e-3, 1000] m;
               # curves are recolored by h_ml (LogNorm) after all models are drawn.
               # The best-fit curve reuses H_plot (no extra solve).
               ax = sens_hml_uptake_axs[expt][sens_hml_uptake_idx[expt]]
               for h_val, curve in zip(h_ml_uptake_grid, h_ml_uptake_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_hml_uptake_lines[expt].append((line, h_val))
               ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
               ax.plot(plot_t, H_plot, color="black", lw=2, label="Best Fit", zorder=5)
               ax.plot(plot_t, F_ref/eps - F_ref/eps*(1-erfcx(np.sqrt(plot_t*(YEAR*(lmbda/eps)**2)/(kappa_pde*4.186e6**2)))), color="blue", lw=2, label="Analytical Solution") # Solution to H for diffusive-eps model (T_eq*lmbda/eps=F_ref/eps)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="center right")
               ax.text(
                  0.02,
                  0.92,
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s, $\epsilon$ = {eps:.2f}, h$_{{ml}}^*$ = {h_ml_pde:.0f} m",
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
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s, h$_{{ml}}$ = {h_ml_pde:.0f} m, $\epsilon$ = {eps:.2f} (fixed)",
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
                  # rf"$\tau_{{1b}}$ = {tau:.0f} yrs" + "\n" + rf"T$_d$ = {T:.0f} yrs, D$_d$ =  {D:.2e} m^2/s" + "\n" + 
                  rf"$\kappa_{{1b+d}}$ = {kappa_pde:.1e} m^2/s, h$_{{ml}}$ = {h_ml_pde:.0f} m, $\epsilon$ = {eps:.2f}",
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
               # TOA imbalance of the PDE fit, including the efficacy term
               # (matches the OHC panel's N_pde and the 3D-plane relation).
               N_pde = F_ref - lmbda * (T_eq * pde_T) - (eps - 1.0) * H_plot

               # TOA imbalance from the same radiative relation but fed the
               # ACTUAL AOGCM T2m timeseries (T_eq*plot_T) instead of the PDE-fit
               # temperature: N = F - lambda*T - (eps-1)*H, with the fitted
               # efficacy and PDE heat uptake H. T2m and H are 10-yr rolling-
               # averaged to match the (smoothed) black AOGCM curve, otherwise
               # raw year-to-year T2m noise makes this line unreadable on the log
               # axis. This should hug the observed NETTOA if the fit is good.
               T_aogcm_abs = T_eq * plot_T
               T_aogcm_roll = np.array([
                  np.mean(T_aogcm_abs[i:i + 10]) for i in range(T_aogcm_abs.shape[0] - 9)
               ])
               H_roll = np.array([
                  np.mean(H_plot[i:i + 10]) for i in range(H_plot.shape[0] - 9)
               ])
               N_aogcm_eps = F_ref - lmbda * T_aogcm_roll - (eps - 1.0) * H_roll

               ax = nettoa_axs[expt][nettoa_idx[expt]]
               ax.plot(t_rollingMu, nettoa_rollingMu, color="black", label="AOGCM")
               ax.plot(plot_t, N_pde, color="purple", label="1-Box + Diffusion Fit")
               ax.plot(t_rollingMu, N_aogcm_eps, color="darkorange", ls="--",
                       label=r"$F-\lambda T_{AOGCM}-(\epsilon-1) H$ (10-yr mean)")
               format_ax(ax, text=f"{model}", xscale="linear", yscale="log", legend_loc="lower left")
               nettoa_idx[expt] += 1

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

      # Persist the fitted parameter table (now including the efficacy epsilon
      # and the PDE kappa/h_ml) so the calibration is inspectable per model.
      df.to_csv(outdir / current_dir / f"fitted_model_params{suffix}.csv", index=False)

      # Save the STEP 1 (Net TOA vs T2M) figures now -- deferred from STEP 1 so
      # each panel carries the efficacy-corrected N = F - lambda*T - (eps-1)*H
      # overlay added during STEP 2.
      for expt in ["4xCO2"]:
         step1_figs[expt].savefig(
            outdir / current_dir / "step1" / "png" / f"{expt}_all_models_T2M_vs_NETTOA{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_figs[expt])

         step1_NH_figs[expt].savefig(
            outdir / current_dir / "step1" / "png" / f"{expt}_all_models_H_vs_NETTOA{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_NH_figs[expt])

         step1_HT_figs[expt].savefig(
            outdir / current_dir / "step1" / "png" / f"{expt}_all_models_H_vs_T2M{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_HT_figs[expt])

      # Put every 3D regression panel on one shared, symmetric residual color
      # scale, add a single colorbar, and save (into step1/ alongside the
      # Net TOA vs T2M figure it complements).
      all_resid3d = np.concatenate([r for _, r in reg3d_scatters]) if reg3d_scatters else np.array([0.0])
      rmax3d = float(np.nanmax(np.abs(all_resid3d))) or 1.0
      resid3d_norm = mpl.colors.Normalize(vmin=-rmax3d, vmax=rmax3d)
      for sc3d, _ in reg3d_scatters:
         sc3d.set_norm(resid3d_norm)
      smap3d = mpl.cm.ScalarMappable(norm=resid3d_norm, cmap="RdBu_r")
      smap3d.set_array([])
      cbar3d = reg3d_fig.colorbar(smap3d, ax=list(reg3d_axs), fraction=0.015, pad=0.02)
      cbar3d.set_label(r"N residual (obs $-$ fit) [W m$^{-2}$]", fontsize=13, fontweight="bold")
      reg3d_fig.savefig(
         outdir / current_dir / "step1" / "png" / f"4xCO2_all_models_N_T_H_regression3d{suffix}.png",
         dpi=200, bbox_inches="tight",
      )
      plt.close(reg3d_fig)

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