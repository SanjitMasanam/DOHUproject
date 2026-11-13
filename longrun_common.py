#!/usr/bin/env python3
"""Data loading, the --test model subset, and small numeric helpers shared by
the longrun fit and comparison scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid

# Relative on purpose: the scripts are run from the repo root.
RDATA_FILE = Path("./data/int_netToa_longrun.Rdata")

# The 8 LongRunMIP models with usable piControl and abrupt-4xCO2 series.
MODELS = ['CCSM3', 'CESM1', 'CNRMCM6', 'ECHAM5', 'GISSE2R', 'IPSLCM5A', 'HadGEM2', 'MPIESM11']

EARTH_AREA = 5.1e14                    # m^2
SECONDS_PER_YEAR = 31536000
OCEAN_HEAT_CAPACITY = 1.37e21 * 3850   # full ocean [J K^-1]


# ---- data ----

def load_longrun_data(rdata_file=RDATA_FILE):
   """Load the .Rdata bundle; return (data, expts).

   `data.rx2(model).rx2(expt).rx2(var)` is an R array; `expts` is
   ['piControl', '4xCO2'].
   """
   import rpy2.robjects as ro  # deferred: needs a system R, which the helpers below do not

   ro.r["load"](str(rdata_file))
   return ro.globalenv["int_nettoa_longrun_data"], [str(x) for x in ro.globalenv["expts"]]


def as_1d_float(x):
   return np.asarray(x, dtype=float).ravel()


def get_r_array(model_data, expt, var):
   return as_1d_float(model_data.rx2(expt).rx2(var))


def finite_pair_mask(*arrays):
   """True where every array is finite."""
   mask = np.ones_like(arrays[0], dtype=bool)
   for arr in arrays:
      mask &= np.isfinite(arr)
   return mask


def perturbed_anomalies(model, model_data, expts, nettoa_scale=1.0, n_years=None,
                        drop_nan_nettoa=True):
   """(T2M, NETTOA) of the perturbed run minus the piControl means.

   n_years truncates both experiments first. With drop_nan_nettoa False only
   years with NaN T2M are dropped, and only from T2M.
   """
   means = {}
   for expt in expts:
      expt_data = model_data.rx2(expt)
      t2m = np.array(expt_data.rx2("T2M")).ravel()[:n_years]
      nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:n_years] * nettoa_scale
      good = np.isfinite(t2m) & np.isfinite(nettoa) if drop_nan_nettoa else np.isfinite(t2m)
      report_missing_years(model, expt, good)
      t2m = t2m[good]
      if drop_nan_nettoa:
         nettoa = nettoa[good]
      if expt == "piControl":
         means = {"t2m": np.mean(t2m), "nettoa": np.mean(nettoa)}
      elif expt == "4xCO2":
         return t2m - means["t2m"], nettoa - means["nettoa"]


def report_missing_years(model, expt, good):
   if np.sum(good) / len(good) != 1:
      print(f"{model} {expt} valid/all years:", np.sum(good) / len(good))


# ---- --test model subset ----

def add_test_argument(parser):
   parser.add_argument(
      "--test", nargs="?", const="2", default=None,
      help="Quick test mode: run a subset of models. Bare --test = first 2; "
           "--test N = first N; --test M1,M2 = named models.")


def select_models(models, spec):
   """The models selected by a --test value; all of them if `spec` is None."""
   if spec is None:
      return models
   if spec.isdigit():
      return models[:max(1, int(spec))]
   names = {s.strip() for s in spec.split(",") if s.strip()}
   unknown = names - set(models)
   if unknown:
      raise SystemExit(f"--test: unknown model(s) {sorted(unknown)}; "
                       f"valid: {models}")
   return [m for m in models if m in names]   # keep the original order


def announce_test_mode(fit_models, models, spec):
   if spec is not None:
      print(f"*** TEST MODE: running {len(fit_models)}/{len(models)} models: "
            f"{fit_models} ***")


# ---- numerics ----

def sympy_prop_unc(expr, values, uncertainties, skip_nonfinite=False):
   """First-order uncertainty of a SymPy expression, inputs taken as uncorrelated.

   values / uncertainties: {symbol: value} / {symbol: 1-sigma}. With
   `skip_nonfinite`, a None or non-finite uncertainty contributes nothing
   instead of propagating.
   """
   variance = 0.0
   for sym, unc in uncertainties.items():
      if skip_nonfinite and (unc is None or not np.isfinite(float(unc))):
         continue
      deriv_val = float(sp.diff(expr, sym).evalf(subs=values))
      variance += (deriv_val * unc) ** 2
   return float(np.sqrt(variance))


def covariance_from_lstsq(X, y, coeffs):
   """OLS coefficient covariance sigma^2 (X'X)^-1."""
   n, p = X.shape
   resid = y - X @ coeffs
   dof = max(n - p, 1)
   sigma2 = float(np.sum(resid**2) / dof)
   return sigma2 * np.linalg.pinv(X.T @ X)


def cumint(t, y):
   """Cumulative trapezoid integral of y(t) from the origin, aligned with `t`."""
   t0 = np.concatenate(([0.0], np.asarray(t, dtype=float)))
   y0 = np.concatenate(([0.0], np.asarray(y, dtype=float)))
   return cumulative_trapezoid(y0, t0, initial=0.0)[1:]
