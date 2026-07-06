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
import rpy2.robjects as ro
import sympy as sp
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# ------------------------- User settings -------------------------

run_type = 1
results = "validation"
lin = True

# Use at least 3 iterations to match your original script, but allow convergence.
MIN_ITERATIONS = 3
MAX_ITERATIONS = 10
CONVERGENCE_RTOL = 1.0e-4

# If True, saves Step 1 and Step 2 diagnostic figures for each iteration.
SAVE_ITERATION_PLOTS = True

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


dir_list = [
    "geoffroy_replicate_results",
    "50-yr_avg_forcing_results",
    "50-yr_avg_tau_s_LR_fit_results",
]
current_dir = dir_list[run_type - 1]

print("==================================")
print(f"Current Dir: {current_dir}\nType of output: {results}\nLinear Scale: {lin}")
print("==================================")


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


def make_model_grid(models, width_per_ax=7, height_per_ax=5, dpi=None, ncols=5):
    """Create a grid large enough for all models."""
    nmodels = len(models)
    nrows = int(np.ceil(nmodels / ncols))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_ax * ncols, height_per_ax * nrows),
        dpi=dpi,
        constrained_layout=True,
    )
    axs = np.asarray(axs).ravel()
    for ax in axs[nmodels:]:
        ax.set_visible(False)
    return fig, axs


def ensure_dirs(outdir, current_dir, sections):
    for section in sections:
        (outdir / current_dir / section / "png").mkdir(parents=True, exist_ok=True)
        (outdir / current_dir / section / "pdf").mkdir(parents=True, exist_ok=True)
    (outdir / current_dir / "tables").mkdir(parents=True, exist_ok=True)


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
        else:
            start = REPLICATE_LATE_YEAR_START
        T_late_raw = select_years(T_4x_raw, start, None)
        N_late_raw = select_years(N_4x_raw, start, None)
        t_late = np.arange(start, start + len(T_late_raw), dtype=float)

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
    ax.set_xlabel("2-meter air temperature anomaly (K)")
    ax.set_ylabel(r"Net TOA radiative flux anomaly (W m$^{-2}$)")
    ax.set_title(f"{model}: radiative fit, iter {iteration + 1}")
    ax.grid(True)
    ax.legend(fontsize=8)


def plot_slow_fit(ax, slow, model, iteration):
    ax.scatter(slow["x"], slow["y"], s=8, alpha=0.5, label="AOGCM")
    ax.plot(slow["x"], slow["yfit"], linewidth=2, label=f"τs={slow['tau_s']:.3g}, as={slow['a_s']:.3g}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"log($T_{eq}$ - T) - log($T_{eq}$)")
    ax.set_title(f"{model}: slow mode, iter {iteration + 1}")
    ax.grid(True)
    ax.legend(fontsize=8)


# ------------------------- Load R data -------------------------

rdata_file = Path("./data/int_netToa_longrun.Rdata")
ro.r["load"](str(rdata_file))

data = ro.globalenv["int_nettoa_longrun_data"]
models = list(ro.globalenv["models"])
expts = list(ro.globalenv["expts"])

# Convert R strings to plain Python strings if needed.
models = [str(x) for x in models]
expts = [str(x) for x in expts]

df = pd.DataFrame(columns=PARAM_COLS)
df["model"] = models

outdir = Path("./2013b_figures")
outdir.mkdir(exist_ok=True)
ensure_dirs(outdir, current_dir, ["step1", "step2", "validation", "unblinded"])

# Cache all model series so every step uses identical baselines and slices.
series_by_model: Dict[str, SeriesBundle] = {}
for model in models:
    series_by_model[model] = make_series_bundle(data.rx2(model), expts)


# ------------------------- Iterative calibration -------------------------

converged = False
last_iteration = 0

for iteration in range(MAX_ITERATIONS):
    print(f"\n================ ITERATION {iteration + 1} ================")
    old_params = numeric_params_for_convergence(df).copy()

    # Create diagnostic figures for this iteration.
    step1_fig, step1_axs = make_model_grid(models, width_per_ax=7, height_per_ax=5)
    step2_fig, step2_axs = make_model_grid(models, width_per_ax=7, height_per_ax=5)

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
    plt.close(step1_fig)

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
            print(f"WARNING: {model} has non-positive mode amplitude(s): a_f={a_f:.4g}, a_s={a_s:.4g}")

        tau_f, tau_f_unc = fit_fast_mode(bundle.t_early, bundle.T_early, T_eq, a_f, a_s, tau_s)
        if not np.isfinite(tau_f):
            print(f"WARNING: {model} tau_f fit failed; using 4 years as fallback.")
            tau_f = 4.0
            tau_f_unc = np.nan

        thermal = thermal_params_from_modes(lmbda, a_f, a_s, tau_f, tau_s, epsilon)
        if thermal["C"] <= 0 or thermal["C0"] <= 0 or thermal["gamma"] <= 0:
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

    print("Finished Step 2: thermal parameters updated")

    # Save a table every iteration for debugging.
    df.to_csv(outdir / current_dir / "tables" / f"params_iter{iteration + 1:02d}.csv", index=False)

    # Check convergence after the minimum number of iterations.
    new_params = numeric_params_for_convergence(df)
    if iteration + 1 >= MIN_ITERATIONS:
        denom = old_params.abs().replace(0, np.nan)
        rel_change = ((new_params - old_params).abs() / denom).replace([np.inf, -np.inf], np.nan)
        max_rel_change = float(np.nanmax(rel_change.to_numpy()))
        print(f"Max relative parameter change: {max_rel_change:.3e}")
        if np.isfinite(max_rel_change) and max_rel_change < CONVERGENCE_RTOL:
            converged = True
            last_iteration = iteration
            print(f"Converged after {iteration + 1} iterations.")
            break

    last_iteration = iteration

if not converged:
    print(f"Did not meet convergence tolerance after {MAX_ITERATIONS} iterations; using final iteration.")

# Final parameter table.
df.to_csv(outdir / current_dir / "tables" / "params_final.csv", index=False)
print(df)


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
    ncols = 5
    nrows = int(np.ceil(nvars / ncols))
    fig_val, axs_val = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 6 * nrows),
        constrained_layout=True,
    )
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

        ax.set_xlabel(r"$\mu_{\rm SN}$")
        ax.set_ylabel(r"$\mu_{\rm GF}$")
        ax.set_title(f"{var}: Geoffroy vs. replication")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(handles=handles, fontsize=8)

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


# ------------------------- Final temperature-response plots -------------------------

final_fig, final_axs = make_model_grid(models, width_per_ax=10.8, height_per_ax=7.2, dpi=120)
scale = "linear" if lin else "log"

for imodel, model in enumerate(models):
    bundle = series_by_model[model]
    row = df.loc[df["model"] == model].iloc[0]
    ax = final_axs[imodel]

    T_eq = float(row["T_eq"])
    a_f = float(row["a_f"])
    a_s = float(row["a_s"])
    tau_f = float(row["tau_f"])
    tau_s = float(row["tau_s"])

    t = bundle.t_full
    T_fit = T_model(t, T_eq, a_f, a_s, tau_f, tau_s)

    ax.scatter(t, bundle.T_full, s=4, color="red")
    ax.plot(t, bundle.T_full, color="red", label="AOGCM T2M")
    ax.plot(t, T_fit, color="blue", label="EBM-ε fit")

    # Optional approximate uncertainty envelope from diagonal parameter uncertainties.
    iterations = 1000
    T_eq_unc = float(row["T_eq_unc"])
    a_s_unc = float(row["a_s_unc"])
    tau_f_unc = float(row["tau_f_unc"])
    tau_s_unc = float(row["tau_s_unc"])

    if np.all(np.isfinite([T_eq_unc, a_s_unc, tau_f_unc, tau_s_unc])):
        param_mean = np.array([T_eq, a_s, tau_f, tau_s], dtype=float)
        param_cov = np.diag(np.array([T_eq_unc**2, a_s_unc**2, tau_f_unc**2, tau_s_unc**2], dtype=float))
        draws = np.random.default_rng(12345).multivariate_normal(param_mean, param_cov, size=iterations)
        T_ensemble = np.empty((iterations, t.size), dtype=float)
        for j in range(iterations):
            T_eq_j, a_s_j, tau_f_j, tau_s_j = draws[j]
            a_f_j = 1.0 - a_s_j
            if tau_f_j <= 0 or tau_s_j <= 0:
                T_ensemble[j, :] = np.nan
            else:
                T_ensemble[j, :] = T_model(t, T_eq_j, a_f_j, a_s_j, tau_f_j, tau_s_j)
        T_mean = np.nanmean(T_ensemble, axis=0)
        T_std = np.nanstd(T_ensemble, axis=0)
        ax.fill_between(t, T_mean - 2 * T_std, T_mean + 2 * T_std, color="blue", alpha=0.08, label="±2σ")
        ax.fill_between(t, T_mean - T_std, T_mean + T_std, color="blue", alpha=0.20, label="±1σ")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Temperature anomaly (K)")
    ax.set_title(f"{model}: T2M with EBM-ε fit")
    ax.set_xscale(scale)
    if lin:
        ax.set_xticks(np.linspace(1, np.max(t), 10))
    ax.legend(fontsize=8)

final_fig.savefig(
    outdir / current_dir / results / "png" / f"4xCO2_all_models_T2M_vs_t_{scale}.png",
    dpi=200,
    bbox_inches="tight",
)
final_fig.savefig(
    outdir / current_dir / results / "pdf" / f"4xCO2_all_models_T2M_vs_t_{scale}.pdf",
    bbox_inches="tight",
)
plt.close(final_fig)

print("Done.")
print(f"Final parameter table: {outdir / current_dir / 'tables' / 'params_final.csv'}")
