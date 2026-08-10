#!/usr/bin/env python3
"""
MT2 framework: per-model fit against AMOC strength (M).

Implements the MT2 / MT2T conceptual model of ocean heat uptake from

  Gregory et al. (2024), "A new conceptual model of global ocean heat uptake",
  Clim Dyn 62:1669-1713, doi:10.1007/s00382-023-06989-z

against this project's LongRunMIP abrupt-4xCO2 runs. Equation numbers in the
comments below refer to that paper. What is computed:

  N   = N_M + N_T                                              (Eq. 5)
  N_M = p F,  p = s_0 (M - M_0)                                (Eq. 5 / 22)
  N_T = q T = c_u dT/dt + gamma (T - T_d),
        c_d dT_d/dt = gamma (T - T_d)                          (Eq. 6 / 24)
  H   = H_M + H_T,  H_M = U_0 + p int F dt,  H_T = C_u T + C_d T_d
                                                               (Eq. 7 / C41)
  kappa = N/T = kappa_M + q,  kappa_M = N_M/T                  (Eq. 10)

The model is diagnostic, as in the paper: each AOGCM's own T(t) and M are the
inputs, and N(t), H(t) and kappa(t) are the outputs.

Of this project's 8 longrun models (CCSM3, CESM1, CNRMCM6, ECHAM5, GISSE2R,
IPSLCM5A, HadGEM2, MPIESM11), only 4 appear in the paper's Table 1 AMOC-strength
(M, Sv) column -- CCSM3, CESM1, ECHAM5 and GISSE2R aren't in it, so they're
excluded from `models` entirely rather than given a placeholder M.

THREE DELIBERATE DEPARTURES FROM THE PAPER, all of them visible in the outputs:

  1. F in N_M = pF is each model's OWN Gregory-regression intercept (F_ref),
     not the paper's AOGCM-neutral F_4x = 7.5 W m^-2 (Sect. 3.1.1). N_M is
     therefore AOGCM-specific through both p and F. The paper's own diagnosed
     spread is 7.6 +- 1.0 W m^-2, so expect N_M to differ by up to ~13%.
  2. H comes from the NETTOA time-integral, because this dataset carries no 3-D
     ocean temperature. That is the paper's H_N, about 7% above its
     ocean-temperature H_O (Appendix A.3, H_N/H_O = 1.07). The two-layer fit is
     reported both ways -- the "_HO" columns of fitted_model_params.csv -- and
     the H figure draws the AOGCM H as a band spanning the two.
  3. Only abrupt4xCO2 is available, so Table 1's OHU and OHUE columns (the
     year-70 2xCO2 state of 1pctCO2) are NOT comparable to anything here. The
     right comparisons are the paper's Figs. 10b, 11a and 11c.

Every panel except the Net-TOA-vs-T2M one carries both two-layer ocean
parameterizations:
  MT2   the paper's ensemble-mean constants AC_U_ZJ / AC_D_ZJ / GAMMA, the
        same for every model (drawn faded / dash-dot). Paired with alpha_g20,
        the plain years-1-20 slope, NOT the deterministic STEP-1 alpha below --
        see two_box_analytic()'s call site.
  MT2T  gamma, C_u and C_d fit to each AOGCM individually, by the appendix
        C.4.2 regression of that model's own H_T(t) on [T, int T dt,
        int H_T dt] through the origin (drawn solid). Paired with the
        deterministic STEP-1 alpha.
The MT2T regression's term-by-term diagnostic (the appendix's Fig 17a) is
written out alongside the result figures. The Net-TOA-vs-T2M panel instead
reconstructs another script's STEP-1 diagnostic (see below) and carries
neither variant.

The one PREDICTIVE curve -- T(t) from the parameters alone, rather than from
the AOGCM's own T -- comes from the constant-forcing two-box solution of
Geoffroy et al. (2013a) Eqs. 8-15 in the EBM-epsilon form of Geoffroy et al.
(2013b), which is exactly what MT2's surface budget (1-p)F = alpha T + N_T
rearranges to. Its radiative parameters follow the deterministic STEP 1 of
longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py, with
N_T in place of N because it is N_T, not N, that vanishes at MT2's equilibrium:

  F_ref  y-intercept of the years-1-20 annual N-vs-T fit
  T_eq   x-intercept of the 50-yr-binned N_T-vs-T fit (first bin dropped)
  eps    F_eff/(that fit's y-intercept),  F_eff = (1-p)F_ref
  alpha  F_eff/T_eq, identically -(binned slope)*eps

See fit_radiative() for the derivation, for what the recipe assumes, and for
why those four are fit over the whole record on both passes. The Net-TOA-vs-T2M
figure is that fit's own provenance panel, reconstructed from
longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py's
STEP 1 (see plot_radiative_fit()) -- since it needs the whole record to mean
anything, it is produced on the unblinded pass only, like ohc_ts and tau_d.

Expected input:
  ./data/int_netToa_longrun.Rdata containing:
    - int_nettoa_longrun_data
    - expts

Outputs:
  ./figures/MT2/01_replicate/tables/fitted_model_params.csv
  ./figures/MT2/01_replicate/tables/<model>_two_layer_Nt.csv
  ./figures/MT2/01_replicate/validation/...   (default)
  ./figures/MT2/01_replicate/unblinded/...    (--unblinded)
    4xCO2_all_models_OHU_rate_vs_Year.png       (N,   Fig 10b)
    4xCO2_all_models_OHU_vs_Year.png            (H,   Fig 11a)
    4xCO2_all_models_HT_vs_T2M.png              (H_T [ZJ] against T)
    4xCO2_all_models_H_flux_vs_T2M.png          (H = gamma(T-T_d) against T)
    4xCO2_all_models_OHUE_vs_Year.png           (kappa)
    4xCO2_all_models_Temperature_vs_Year.png
    4xCO2_all_models_HT_two_layer_fit.png       (Fig 17a)
  --unblinded only:
    4xCO2_all_models_T2M_vs_NETTOA.png          (STEP-1 radiative fit)
    4xCO2_all_models_ohc_ts.png                 (the other scripts' OHU vs Ts)
    4xCO2_all_models_tau_d_vs_calibration_window.png
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from scipy.integrate import cumulative_trapezoid, solve_ivp

from figure_paths import UNBLINDED, VALIDATION, RunLayout, figname
from plot_style import (
   AXIS_LABEL_FONTSIZE, EXTRA_TEXT_FONTSIZE, format_ax, make_model_grid,
   make_model_grid_with_ratio,
)

# --- Quick test mode -----------------------------------------------------------
# Optional CLI flag to run the whole pipeline over only a subset of models, so
# plotting changes can be checked without waiting for all model fits.
_cli = argparse.ArgumentParser(description="MT2 param-fit script")
_cli.add_argument(
   "--test", nargs="?", const="2", default=None,
   help="Quick test mode: run a subset of models. Bare --test = first 2; "
        "--test N = first N; --test M1,M2 = named models.")
_cli.add_argument(
   "--unblinded", action="store_true",
   help="Unblinded pass: run every timeseries panel out to the full length of "
        "that model's run and write to unblinded/ instead of validation/.")
_cli_args = _cli.parse_known_args()[0]
_TEST_SPEC = _cli_args.test
UNBLINDED_RUN = _cli_args.unblinded

def _apply_test_subset(models):
   """Return the model subset selected by --test (full list if flag absent)."""
   if _TEST_SPEC is None:
      return models
   spec = _TEST_SPEC
   if spec.isdigit():
      return models[:max(1, int(spec))]
   names = {s.strip() for s in spec.split(",") if s.strip()}
   unknown = names - set(models)
   if unknown:
      raise SystemExit(f"--test: unknown model(s) {sorted(unknown)}; "
                       f"valid: {models}")
   return [m for m in models if m in names]   # preserve original order

# ------------------------- User settings -------------------------

VARIANT = "MT2"

# If your R data are already anomalies, set this to False.
SUBTRACT_PICONTROL_BASELINE = True

# The 8 models used across this project's other longrun_data_param_fit_script_*.py
# scripts.
ALL_LONGRUN_MODELS = [
   "CCSM3", "CESM1", "CNRMCM6", "ECHAM5", "GISSE2R", "IPSLCM5A", "HadGEM2", "MPIESM11",
]

# AMOC strength M (Sv) for whichever of ALL_LONGRUN_MODELS appear in the
# AMOC-strength table. IPSLCM5A -> IPSL-CM5A-LR (not -MR) and MPIESM11 ->
# MPI-ESM-LR (not -MR/-P) per the table's naming.
M_SV: Dict[str, float] = {
   "CNRMCM6": 17.0,    # CNRM-CM6-1
   "HadGEM2": 14.8,    # HadGEM2-ES
   "IPSLCM5A": 11.5,   # IPSL-CM5A-LR
   "MPIESM11": 23.5,   # MPI-ESM-LR
}

# Constants for the fitting procedure.
s_0 = 0.0047
M_0 = -10.2

# --- MT2 two-layer ocean model -----------------------------------------------
#   N_T = c_u dT/dt + Phi ,   Phi = gamma*(T - T_d) = c_d dT_d/dt
# c_u, c_d below are supplied area-integrated (A*c) in ZJ/K; divide by Earth's
# surface area and by seconds-per-year to land in this codebase's usual
# W yr m^-2 K^-1 units, matching gamma's W m^-2 K^-1.
EARTH_AREA_M2 = 5.1e14
SECONDS_PER_YEAR = 31536000

AC_U_ZJ = 60.0     # upper-layer area-integrated heat capacity [ZJ K^-1]
AC_D_ZJ = 454.0    # deep-layer area-integrated heat capacity [ZJ K^-1]
GAMMA = 0.470      # inter-layer coupling [W m^-2 K^-1]

# AOGCM-neutral constant term in H_M (Eq. 7). Small next to H_M's pF*t term, and
# included "for accuracy" by the paper, but it is the whole reason the early
# years need special handling: H_M carries the full U_0 from t=0, while
# physically H_M(0) = 0 -- "U_0 takes some time to build up" (after Eq. C40).
# That overstates early H_M, so H_T = H - H_M goes negative for the paper's
# first five years, which is what CALIB_SKIP_YEARS below drops.
U_0_ZJ = 84.0

# Years of the Gregory-style N-vs-T regression that yields F_ref (Sect. 2.1
# evaluates alpha and F_4x over years 1-20 of abrupt4xCO2).
GREGORY_FIT_YEARS = 20

# Bin width (years) of the long-record N_T-vs-T regression that yields T_eq,
# alpha and eps, and the number of leading bins it drops. Both match
# longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py,
# whose STEP 1 this reproduces; the first bin is skipped because the fast mode
# is still unwinding through it, which would drag the slow-mode line off the
# late-time behaviour it is meant to measure.
BIN_YEARS = 50
BIN_SKIP = 1

# Floors on the deterministic radiative parameters, as in that script. A
# degenerate regression can hand back a non-positive F, alpha or eps, and each
# one breaks the two-box solve in a different way (a negative alpha turns the
# radiative damping into anti-damping, so T_eq and both timescales go
# meaningless rather than merely wrong). Clamping keeps the run going and the
# clamped value is obvious in the table.
F_REF_FLOOR = 1e-6
ALPHA_FLOOR = 1e-3
EPS_FLOOR = 1e-3

# Early years excluded from the two-layer calibration. The paper: "There are
# small negative values for H_T for the first five years of abrupt4xCO2, which
# is physically unexpected ... we ignore those years for the purpose of
# calibrating a model for H_T" (Appendix C.4.1).
CALIB_SKIP_YEARS = 5

# The paper diagnoses H from 3-D ocean temperature (H_O); only NETTOA is
# available here, so this script's H is the paper's H_N, larger by this factor
# (Appendix A.3, regression through the origin gives H_N/H_O = 1.07). The MT2T
# fit is reported both ways -- see the "_HO" columns in two_layer_fit().
H_N_OVER_H_O = 1.07

# Multiply a per-area heat capacity [W yr m^-2 K^-1] or heat content
# [W yr m^-2] by this to get the paper's area-integrated ZJ K^-1 / ZJ.
ZJ_PER_WYRM2 = EARTH_AREA_M2 * SECONDS_PER_YEAR / 1e21      # ~16.08

C_U = AC_U_ZJ / ZJ_PER_WYRM2   # W yr m^-2 K^-1
C_D = AC_D_ZJ / ZJ_PER_WYRM2   # W yr m^-2 K^-1

# Models this script actually runs: the ALL_LONGRUN_MODELS with a resolved M.
models = [m for m in ALL_LONGRUN_MODELS if m in M_SV]
fit_models = _apply_test_subset(models)
if _TEST_SPEC is not None:
   print(f"*** TEST MODE: running {len(fit_models)}/{len(models)} models: {fit_models} ***")

PARAM_COLS = ["model", "M"]

# Extra top/bottom margin beyond make_model_grid's defaults. Those defaults are
# sized for the other scripts' 4x2 grids; with only 4 models this grid is a
# single row, so the figure is half as tall and the same fractional margins
# leave half the room -- the suptitle lands on the panels and the figure-level
# x-label lands on the bottom tick labels. Applied to every grid here.
GRID_MARGINS = dict(top=0.88, bottom=0.14)

OHU_RATE_YEARS = 150

# Years shown on the timeseries panels, and fitted over by the MT2T two-layer
# regression. The blinded (default) pass truncates every panel at
# OHU_RATE_YEARS so the panels are directly comparable and the late-run
# behaviour stays hidden; --unblinded leaves it None, which SeriesBundle
# resolves to that model's own last year, so each panel/fit runs out to the
# full length of its run. Nothing reads this directly -- every consumer goes
# through bundle.panel_years or bundle.window().
PANEL_YEARS = None if UNBLINDED_RUN else OHU_RATE_YEARS

# Window (years) of the centred running mean applied to the OHU-rate curves,
# and to the two-layer N_T before it is binned in T. Interannual variability in
# NETTOA swamps the decomposition at this scale; min_periods=1 keeps the
# smoothed series full-length, with the first and last ~5 years averaged over
# the shorter one-sided window that's available.
SMOOTH_WINDOW_YEARS = 10


# ------------------------- Data structures -------------------------

@dataclass
class SeriesBundle:
   """One model's baselined 4xCO2 series, plus the window every panel shares."""

   t_full: np.ndarray
   T_full: np.ndarray
   N_full: np.ndarray

   @property
   def panel_years(self):
      """Last year the panels show: OHU_RATE_YEARS when blinded, this model's
      own run length when not (PANEL_YEARS is None on the unblinded pass)."""
      return float(self.t_full[-1]) if PANEL_YEARS is None else PANEL_YEARS

   def window(self, n_years=None):
      """(last_year, t, T, N) truncated to `n_years`, or to the panel window.

      The one place the blinded/unblinded truncation is applied, so a panel
      and the fit behind it cannot disagree about which years they cover.
      """
      n_years = self.panel_years if n_years is None else n_years
      mask = self.t_full <= n_years
      return n_years, self.t_full[mask], self.T_full[mask], self.N_full[mask]


# ------------------------- Helper functions -------------------------

def as_1d_float(x):
   return np.asarray(x, dtype=float).ravel()


def finite_pair_mask(*arrays):
   mask = np.ones_like(arrays[0], dtype=bool)
   for arr in arrays:
      mask &= np.isfinite(arr)
   return mask


def get_r_array(model_data, expt, var):
   return as_1d_float(model_data.rx2(expt).rx2(var))


def make_series_bundle(model_data, expts):
   """Read piControl and 4xCO2 series and return a consistently baselined bundle."""
   if "piControl" not in expts or "4xCO2" not in expts:
      raise RuntimeError(f"Expected expts to include 'piControl' and '4xCO2'; got {expts}")

   T_ctrl_raw = get_r_array(model_data, "piControl", "T2M")
   N_ctrl_raw = get_r_array(model_data, "piControl", "NETTOA")
   T_4x_raw = get_r_array(model_data, "4xCO2", "T2M")
   N_4x_raw = get_r_array(model_data, "4xCO2", "NETTOA")

   if SUBTRACT_PICONTROL_BASELINE:
      T_base = float(np.nanmean(T_ctrl_raw))
      N_base = float(np.nanmean(N_ctrl_raw))
   else:
      T_base = 0.0
      N_base = 0.0

   T_full = T_4x_raw - T_base
   N_full = N_4x_raw - N_base
   t_full = np.arange(1, 1 + len(T_full), dtype=float)

   mask = finite_pair_mask(t_full, T_full, N_full)
   return SeriesBundle(t_full=t_full[mask], T_full=T_full[mask], N_full=N_full[mask])


def cumint(t, y):
   """Cumulative time-integral of y(t), anchored at the origin.

   The (0, 0) sample is prepended before integrating and dropped afterwards, so
   the result is aligned with `t` and satisfies the appendix's requirement that
   every integral -- and hence the regression -- passes through the origin.
   """
   t0 = np.concatenate(([0.0], np.asarray(t, dtype=float)))
   y0 = np.concatenate(([0.0], np.asarray(y, dtype=float)))
   return cumulative_trapezoid(y0, t0, initial=0.0)[1:]


def running_mean(y, window=SMOOTH_WINDOW_YEARS):
   """Centred running mean of a 1D series, same length as the input."""
   return pd.Series(np.asarray(y, dtype=float)).rolling(
      window, center=True, min_periods=1).mean().to_numpy()


def update_row(df, model, values):
   """Write one model's values into its row, creating any missing columns.

   Every fitted and derived quantity reaches the table through here, so a new
   key in fit_model() or in the derived-scalar block shows up in the CSV without
   also having to be declared in PARAM_COLS.
   """
   for key, val in values.items():
      if key not in df.columns:
         df[key] = np.nan
      df.loc[df["model"] == model, key] = val


# --- H_M and H_T (Eq. 7) -----------------------------------------------------
# One definition each, shared by the calibration and every figure, so the two
# cannot drift apart. Both are per unit area [W yr m^-2] unless the name says
# ZJ; multiply by ZJ_PER_WYRM2 for the paper's area-integrated ZJ.

def H_M_per_area(t, N_m):
   """H_M = U_0 + p*int F dt (Eq. 7).

   F = F_4x is constant in abrupt4xCO2, so p*int F dt collapses to N_m*t.
   """
   return U_0_ZJ / ZJ_PER_WYRM2 + N_m * np.asarray(t, dtype=float)


def aogcm_H_T(t, N, N_m, scale=1.0):
   """The AOGCM's temperature-driven heat uptake H_T = H - H_M (Appendix C.4.1),
   the quantity the two-layer calibration regresses.

   H is the NETTOA time-integral, since this dataset carries no 3-D ocean
   temperature -- i.e. the paper's H_N. `scale` is where the H_N -> H_O
   correction goes; it multiplies the AOGCM heat content only. H_M is a
   forcing-side construct (U_0 and p*F are not ocean-heat diagnostics), so it is
   left alone, which is why the scaled variant is not a uniform rescaling of the
   fitted parameters.

   H_M carries its full U_0 from t = 0 while physically H_M(0) = 0, so H_T is
   negative over the first few years. That is the artefact CALIB_SKIP_YEARS
   drops, and it is reported per model as `H_T_last_neg_year`.
   """
   t = np.asarray(t, dtype=float)
   return scale * cumint(t, N) - H_M_per_area(t, N_m)


def model_H_T_per_area(T, T_d, c_u, c_d):
   """The two-layer model's own H_T = C_u T + C_d T_d (Eq. C41).

   Exact given the solve, so the MT2/MT2T H_T needs no second integration of
   N_T -- it is the analytic integral of Eq. 6.
   """
   return c_u * np.asarray(T, dtype=float) + c_d * np.asarray(T_d, dtype=float)


# --- Constant-forcing analytic two-box solution -------------------------------
# Everything above is diagnostic: T(t) is prescribed from the AOGCM and only
# T_d is solved. This block is the one PREDICTIVE piece -- T(t) itself from the
# fitted parameters alone -- and it exists because the MT2 surface budget
#
#    F - N_M = (1-p)F = alpha T + N_T,   N_T = c_u dT/dt + gamma (T - T_d)
#                                                                 (Eq. 24 / 25)
#
# rearranges to precisely Geoffroy et al. (2013a)'s two-box system with the
# forcing reduced to (1-p)F, because N_M = pF is absorbed with no effect on T:
#
#    c_u dT /dt = (1-p)F - alpha T - gamma (T - T_d)
#    c_d dT_d/dt =                   gamma (T - T_d)
#
# so that paper's closed-form solution applies with C -> c_u, C_0 -> c_d,
# lambda -> alpha, F -> (1-p)F_ref. This is the same analytic curve
# longrun_data_param_fit_script_2013a_allModels.py plots, reached through the
# MT2 parameters instead of Geoffroy's own two-step calibration.

def effective_forcing(p, F_ref):
   """(1-p)F: the part of the ERF left to drive T after N_M = pF is absorbed."""
   return (1.0 - p) * F_ref


# --- Deterministic radiative parameters: F_ref, T_eq, alpha, eps --------------
# The STEP 1 procedure of
# longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py,
# carried over to MT2 by regressing N_T rather than N.
#
# Regressing N_T is the right move here, not a cosmetic one. MT2's surface
# budget is (1-p)F = alpha T + N_T (Eq. 25), so it is N_T that vanishes at
# equilibrium while N_M = pF persists -- the AOGCM's total N never reaches zero
# under MT2, and the x-intercept of N-vs-T would therefore not be T_eq. Since
# N_M is a constant, dropping it shifts the intercept and leaves the slope
# alone, so the two regressions differ only by that shift.
#
# Read off the two lines:
#
#   20-yr annual N-vs-T      intercept  -> F_ref              (the ERF, Sect. 2.1)
#   50-yr binned N_T-vs-T    intercept  -> bb = F_eff/eps
#                            slope      -> mb = -alpha/eps
#                            x-intercept-> T_eq = -bb/mb = F_eff/alpha
#
# with F_eff = (1-p)F_ref, so alpha = F_eff/T_eq and eps = F_eff/bb. The two
# expressions for alpha -- F_eff/T_eq and -mb*eps -- are the same number
# identically, given T_eq = -bb/mb; there is no choice to make between them.
#
# CAVEAT, inherited from that script and worth stating: bb = F/eps and
# mb = -alpha/eps is an approximation, not an identity. Working the slow mode of
# the two-box system through exactly gives a late-time slope of
# -(alpha - (eps-1) c_d phi_s/tau_s), which equals -alpha/eps only by
# coincidence. The recipe is a self-consistent estimator, not a derivation, and
# the eps it returns should be read as "the efficacy implied by the late-time
# Gregory line", not as a measurement of the efficacy in the surface budget.
#
# Fit over the WHOLE record on both passes, exactly as that script does for
# every run type: F_ref, T_eq, alpha and eps are physical properties of the
# AOGCM rather than of the plot window, and with the panels truncated at
# OHU_RATE_YEARS the binned fit would have two points left. The consequence is
# that these four scalars are NOT blind on the validation pass -- only the
# panels' data extent is. Contrast two_layer_fit(), which does respect the
# window.

def bin_by_position(y, width=BIN_YEARS, skip=BIN_SKIP):
   """Non-overlapping `width`-sample block means, dropping the first `skip`.

   By position, not by year: make_series_bundle() has already dropped any
   non-finite years, so the two coincide unless a model has gaps. A trailing
   partial block is kept, as in the diffusive script.
   """
   y = np.asarray(y, dtype=float)
   return np.array([y[i:i + width].mean()
                    for i in range(skip * width, y.shape[0], width)])


@dataclass(frozen=True)
class RadiativeFit:
   """The four deterministic radiative scalars, plus the two regressions they
   were read off, so the N_T-vs-T panel can draw its own provenance."""

   F_ref: float
   F_eff: float
   T_eq: float
   alpha: float
   eps: float
   F_ref_unc: float
   T_eq_unc: float
   alpha_unc: float
   eps_unc: float
   m20: float            # 20-yr annual N-vs-T fit
   b20: float
   mb: float             # 50-yr binned N_T-vs-T fit
   bb: float
   T_bin: np.ndarray
   NT_bin: np.ndarray

   def as_params(self):
      return {"F_ref": self.F_ref, "F_eff": self.F_eff, "T_eq_fit": self.T_eq,
              "alpha": self.alpha, "eps": self.eps,
              "F_ref_unc": self.F_ref_unc, "T_eq_fit_unc": self.T_eq_unc,
              "alpha_unc": self.alpha_unc, "eps_unc": self.eps_unc,
              "alpha_g20": -self.m20, "bin_slope": self.mb,
              "bin_intercept": self.bb, "n_bins": float(self.T_bin.size)}


_RADIATIVE_CACHE: Dict[str, RadiativeFit] = {}


def fit_radiative(model, bundle, p):
   """Read F_ref, T_eq, alpha and eps off the two regressions described above.

   `p` is the MT2 AMOC term s_0(M - M_0), needed because N_M = p*F_ref has to
   be subtracted before the binned fit and because F_eff = (1-p)F_ref sets the
   scale of alpha and eps.

   Cached per model. It depends only on the full record and on M, neither of
   which changes within a run, and fit_model() and the N_T-vs-T panel both want
   the same answer -- including the two regression lines, which only the panel
   uses.
   """
   if model in _RADIATIVE_CACHE:
      return _RADIATIVE_CACHE[model]

   T, N = bundle.T_full, bundle.N_full

   gregory = bundle.t_full <= GREGORY_FIT_YEARS
   (m20, b20), cov20 = np.polyfit(T[gregory], N[gregory], 1, cov=True)
   b20_unc = float(np.sqrt(max(cov20[1, 1], 0.0)))

   F_ref = max(float(b20), F_REF_FLOOR)
   F_eff = effective_forcing(p, F_ref)
   F_eff_unc = abs(1.0 - p) * b20_unc

   T_bin = bin_by_position(T)
   NT_bin = bin_by_position(N - p * F_ref)
   if T_bin.size < 3:
      raise RuntimeError(
         f"{model}: only {T_bin.size} bins of {BIN_YEARS} years after dropping "
         f"the first {BIN_SKIP}; too few for the N_T-vs-T fit")
   (mb, bb), covb = np.polyfit(T_bin, NT_bin, 1, cov=True)
   mb_unc = float(np.sqrt(max(covb[0, 0], 0.0)))
   bb_unc = float(np.sqrt(max(covb[1, 1], 0.0)))

   T_eq_raw = float(-bb / mb) if mb != 0.0 else np.nan
   T_eq = T_eq_raw if T_eq_raw > 0.0 else 1.0       # guard a degenerate x-int
   alpha = max(F_eff / T_eq, ALPHA_FLOOR)
   eps = max(F_eff / bb, EPS_FLOOR) if bb != 0.0 else EPS_FLOOR
   if T_eq_raw != T_eq or alpha == ALPHA_FLOOR or eps == EPS_FLOOR:
      print(f"   WARNING {model}: degenerate binned N_T-vs-T fit "
            f"(slope={mb:.4g}, intercept={bb:.4g}) -- one or more of T_eq, "
            f"alpha, eps was clamped to its floor")

   # First-order propagation through both fits, treated as independent (the
   # diffusive script's convention). T_eq = -bb/mb, eps = F_eff/bb and
   # alpha = -F_eff*mb/bb.
   T_eq_unc = float(np.hypot(bb_unc / mb, bb * mb_unc / mb ** 2)) if mb else np.nan
   eps_unc = float(np.hypot(F_eff_unc / bb, F_eff * bb_unc / bb ** 2)) if bb else np.nan
   alpha_unc = (float(np.sqrt((mb / bb * F_eff_unc) ** 2
                              + (F_eff / bb * mb_unc) ** 2
                              + (F_eff * mb / bb ** 2 * bb_unc) ** 2))
                if bb else np.nan)

   _RADIATIVE_CACHE[model] = RadiativeFit(
      F_ref=F_ref, F_eff=F_eff, T_eq=T_eq, alpha=alpha, eps=eps,
      F_ref_unc=b20_unc, T_eq_unc=T_eq_unc, alpha_unc=alpha_unc, eps_unc=eps_unc,
      m20=float(m20), b20=float(b20), mb=float(mb), bb=float(bb),
      T_bin=T_bin, NT_bin=NT_bin)
   return _RADIATIVE_CACHE[model]


@dataclass(frozen=True)
class TwoBoxAnalytic:
   """Constant-forcing solution of the two-box system above.

   T(t)   = T_eq [a_f (1 - e^-t/tau_f) + a_s (1 - e^-t/tau_s)]
   T_d(t) = T_eq [a_f phi_f (1 - e^-t/tau_f) + a_s phi_s (1 - e^-t/tau_s)]

   with a_f + a_s = 1, so T -> T_eq as t -> infinity. phi_s > 1 is normal and
   not a bug: in the slow mode the deep box sits further from equilibrium than
   the surface, because it lags.
   """

   T_eq: float
   tau_f: float
   tau_s: float
   a_f: float
   a_s: float
   phi_f: float
   phi_s: float
   lam: float        # radiative feedback (the paper's alpha)
   gamma: float      # PHYSICAL inter-layer coupling
   eps: float        # deep-ocean heat-uptake efficacy
   F_eff: float      # (1-p)F: forcing the surface box actually sees
   F_total: float    # F: the full ERF, for the total-N curve

   def T(self, t):
      t = np.asarray(t, dtype=float)
      return self.T_eq * (self.a_f * (1.0 - np.exp(-t / self.tau_f))
                          + self.a_s * (1.0 - np.exp(-t / self.tau_s)))

   def T_d(self, t):
      t = np.asarray(t, dtype=float)
      return self.T_eq * (self.a_f * self.phi_f * (1.0 - np.exp(-t / self.tau_f))
                          + self.a_s * self.phi_s * (1.0 - np.exp(-t / self.tau_s)))

   def H(self, t):
      """Physical ocean heat uptake H = gamma (T - T_d), unscaled by eps: the
      surface loses eps*H but the interior receives H."""
      return self.gamma * (self.T(t) - self.T_d(t))

   def N(self, t):
      """Total TOA imbalance N = F - lam T - (eps-1) H (Geoffroy 2013b).

      Identically the total heat storage c_u dT/dt + H, which is why the
      efficacy term appears: the surface sheds eps*H but only H leaves the
      system's surface box.
      """
      return self.F_total - self.lam * self.T(t) - (self.eps - 1.0) * self.H(t)

   def as_params(self, suffix=""):
      """The scalars worth carrying in fitted_model_params.csv."""
      return {f"{k}{suffix}": v for k, v in {
         "T_eq": self.T_eq, "tau_f": self.tau_f, "tau_s": self.tau_s,
         "a_f": self.a_f, "a_s": self.a_s, "lam": self.lam, "eps": self.eps,
         "F_eff": self.F_eff,
      }.items()}


def two_box_analytic(lam, F_eff, c_u, c_d, gamma, eps=1.0, F_total=None):
   """Solve for (tau_f, tau_s, a_f, a_s, phi_f, phi_s) -- Geoffroy 2013a Eqs. 8-15,
   extended to the EBM-epsilon system of Geoffroy 2013b.

   Efficacy enters through gamma' = eps*gamma and c_d' = eps*c_d, which turn the
   efficacy system into the standard two-box one. Two of the three groupings are
   invariant under that substitution -- gamma'/c_d' = gamma/c_d and
   4 lam gamma'/(c_u c_d') = 4 lam gamma/(c_u c_d) -- so eps only ever appears
   where gamma stands against c_u: in b, b* and the phi prefactor.

   The fitted c_u/c_d/gamma are the PHYSICAL parameters, so 2013b's "divide by
   epsilon" step does not apply here; that step exists because 2013b inverts
   modes to get C_0' and gamma', whereas this runs forward.

   Returns None if the discriminant is non-positive, i.e. if the parameters do
   not give two real relaxation timescales. That cannot happen for physical
   (positive) parameters, but a degenerate fit -- MPIESM11's full-run
   calibration returns a negative gamma -- can produce it, and a None is more
   honest than a complex tau.
   """
   if not (lam > 0.0 and gamma > 0.0 and c_u > 0.0 and c_d > 0.0 and eps > 0.0):
      return None

   gamma_p = eps * gamma                        # gamma'
   b = (lam + gamma_p) / c_u + gamma / c_d
   b_star = (lam + gamma_p) / c_u - gamma / c_d
   delta = b ** 2 - 4.0 * lam * gamma / (c_u * c_d)
   if not (delta > 0.0):
      return None
   sqrt_delta = float(np.sqrt(delta))

   scale = c_u * c_d / (2.0 * lam * gamma)      # = c_u c_d'/(2 lam gamma')
   tau_f = scale * (b - sqrt_delta)
   tau_s = scale * (b + sqrt_delta)
   phi_f = c_u / (2.0 * gamma_p) * (b_star - sqrt_delta)
   phi_s = c_u / (2.0 * gamma_p) * (b_star + sqrt_delta)
   a_f = phi_s * tau_f * lam / (c_u * (phi_s - phi_f))
   a_s = -phi_f * tau_s * lam / (c_u * (phi_s - phi_f))

   return TwoBoxAnalytic(
      T_eq=F_eff / lam, tau_f=tau_f, tau_s=tau_s, a_f=a_f, a_s=a_s,
      phi_f=phi_f, phi_s=phi_s, lam=lam, gamma=gamma, eps=eps,
      F_eff=F_eff, F_total=F_eff if F_total is None else F_total)


# ------------------------- Fitting procedure -------------------------
# fit_model() returns one model's fitted parameters; every key it returns becomes
# a column of fitted_model_params.csv via update_row().

def two_layer_fit(model, bundle, N_m, window_years=None, suffix="", scale=1.0,
                  verbose=True):
   """MT2T: this AOGCM's own two-layer parameters (appendix C.4.2), fit over
   the first `window_years` years (None = the shared panel window).

   H_T = int_0^t N_T dt' = C_u T + C_d T_d, which rearranges to
      H_T = C_u T + gamma(1 + C_u/C_d) int T dt' - (gamma/C_d) int H_T dt'
   (the appendix prints a2 = gamma + C_u/C_d, a typo -- only the form above
   inverts to the appendix's own C46). Regressing H_T on the three terms
   with no intercept, i.e. through the origin, gives a1, a2, a3 below.

   `verbose=False` silences the per-model H_T diagnostics; the calibration-window
   sweep refits this hundreds of times per model and only wants the numbers.
   """
   _, t, T, N = bundle.window(window_years)
   H_T = aogcm_H_T(t, N, N_m, scale=scale)

   # The design matrix spans the WHOLE window: its cumulative integrals have
   # to stay anchored at t = 0 for the through-origin requirement. Only the
   # ROWS of the least-squares problem are then dropped, which is what
   # excludes the paper's physically-impossible negative early H_T from the
   # objective.
   #
   # Caveat this cannot fix: H_T is both the regressand and (integrated) the
   # third predictor, so the early-year error survives inside int H_T dt as a
   # roughly constant offset at all later times, and with no intercept to
   # absorb it, it still biases a3. The paper does not address this; ramping
   # U_0 in over the first few years would, but goes beyond the paper.
   X_all = np.column_stack([T, cumint(t, T), cumint(t, H_T)])
   fit_rows = t > CALIB_SKIP_YEARS
   X = X_all[fit_rows]
   y = H_T[fit_rows]

   # Two unrelated things drive H_T negative, and conflating them would be
   # misleading:
   #   * the first year or two, where H_M carries its full U_0 although
   #     H_M(0) = 0. This is the artefact CALIB_SKIP_YEARS exists to drop.
   #     The paper's five years is empirical for its ocean-temperature H;
   #     with H from NETTOA the crossing lands elsewhere, so measure it.
   #   * late in a long run, where H_M = U_0 + pFt keeps growing linearly
   #     while the AOGCM's H saturates towards equilibrium, until H_M
   #     outgrows H entirely. That is MT2 itself breaking down -- "N_M cannot
   #     remain non-zero and unchanged as equilibrium is approached under
   #     constant forcing" (Sect. 6.3) -- and no calibration window fixes it.
   pos = np.flatnonzero(H_T >= 0.0)
   first_pos_year = float(t[pos[0]]) if pos.size else np.nan
   late_neg = (np.flatnonzero((H_T < 0.0) & (t > first_pos_year))
               if pos.size else np.array([], dtype=int))
   late_neg_year = float(t[late_neg[0]]) if late_neg.size else np.nan
   if verbose and suffix == "":
      if not (first_pos_year <= CALIB_SKIP_YEARS):
         print(f"   WARNING {model}: H_T first turns positive at year "
               f"{first_pos_year:.0f}, past CALIB_SKIP_YEARS={CALIB_SKIP_YEARS} "
               f"-- early rows are still in the fit")
      if np.isfinite(late_neg_year):
         print(f"   NOTE {model}: H_T turns negative again from year "
               f"{late_neg_year:.0f}: H_M has outgrown the AOGCM's H as it "
               f"equilibrates, so MT2 has broken down by then (Sect. 6.3)")

   a, *_ = np.linalg.lstsq(X, y, rcond=None)
   a1, a2, a3 = (float(v) for v in a)

   c_u = a1
   c_d = -a2 / a3 - a1
   gamma = a2 + a1 * a3

   # Coefficient covariance, sigma^2 (X'X)^-1, propagated to (c_u, c_d,
   # gamma) through the Jacobian of the mapping above. The full covariance
   # is used rather than the diagonal: a1, a2 and a3 are strongly correlated
   # here, so treating them as independent would badly misstate c_d and
   # gamma.
   resid = y - X @ a
   dof = max(len(y) - X.shape[1], 1)
   cov_a = float(np.sum(resid ** 2) / dof) * np.linalg.pinv(X.T @ X)
   jac = np.array([
      [1.0,          0.0,      0.0        ],   # d c_u   / d (a1, a2, a3)
      [-1.0,        -1.0 / a3, a2 / a3**2 ],   # d c_d   / d (a1, a2, a3)
      [a3,           1.0,      a1         ],   # d gamma / d (a1, a2, a3)
   ])
   c_u_unc, c_d_unc, gamma_unc = (float(v) for v in np.sqrt(
      np.clip(np.diag(jac @ cov_a @ jac.T), 0.0, None)))

   # Multiple correlation coefficient of the fit (paper quotes > 0.999).
   ss_tot = float(np.sum((y - np.mean(y)) ** 2))
   R_HT = (float(np.sqrt(max(1.0 - float(np.sum(resid ** 2)) / ss_tot, 0.0)))
           if ss_tot > 0 else np.nan)

   return {f"{k}{suffix}": v for k, v in {
      "c_u": c_u, "c_d": c_d, "gamma": gamma,
      "c_u_unc": c_u_unc, "c_d_unc": c_d_unc, "gamma_unc": gamma_unc,
      "C_u_ZJ": c_u * ZJ_PER_WYRM2, "C_d_ZJ": c_d * ZJ_PER_WYRM2,
      "C_u_ZJ_unc": c_u_unc * ZJ_PER_WYRM2,
      "C_d_ZJ_unc": c_d_unc * ZJ_PER_WYRM2,
      "R_HT": R_HT,
      "fit_start_year": float(t[fit_rows][0]),
      "fit_years": float(t[-1]),
      "H_T_first_pos_year": first_pos_year,
      "H_T_late_neg_year": late_neg_year,
   }.items()}


def fit_model(model, bundle, M):
   """Fit one model's series. Returns a dict of fitted parameters: the four
   deterministic radiative scalars from fit_radiative(), N_m, and that model's
   own (MT2T) two-layer parameters from two_layer_fit(). The MT2 ensemble-mean
   constants (C_U / C_D / GAMMA) need no fit and are used as-is for the
   comparison curves.

   NOTE (departure from the paper): the paper diagnoses F_4x = 7.6 +- 1.0 W m^-2
   from the years-1-20 regression but then adopts an AOGCM-NEUTRAL 7.5 W m^-2
   for every model (Sect. 3.1.1). This script deliberately keeps each model's
   own F_ref, so N_M = p*F_ref is AOGCM-specific through both p and F.
   """
   p = s_0 * (M - M_0)
   rad = fit_radiative(model, bundle, p)
   N_m = p * rad.F_ref

   # The plain Gregory regression over the whole panel window, recorded for
   # reference only. alpha declines with time in every AOGCM, so this is
   # systematically smaller than the years-1-20 value; the gap between the
   # alpha_g20 and alpha_full columns is how much the apparent feedback drifted
   # over the window, and `alpha` itself -- the deterministic F_eff/T_eq that
   # the analytic curves use -- should sit between them.
   _, _, T_win, N_win = bundle.window()
   slope_full, F_ref_full = np.polyfit(T_win, N_win, 1)

   params = {"p": p, "N_m": N_m,
             "F_ref_full": F_ref_full, "alpha_full": -slope_full}
   params.update(rad.as_params())

   params.update(two_layer_fit(model, bundle, N_m))
   # Same fit against H_O = H_N/1.07 (Appendix A.3), so the table shows how much
   # of C_u/C_d/gamma rides on this dataset's H coming from NETTOA rather than
   # ocean temperature. Not a uniform 1/1.07 rescaling: H_M is left unscaled.
   params.update(two_layer_fit(model, bundle, N_m, suffix="_HO",
                               scale=1.0 / H_N_OVER_H_O))
   if UNBLINDED_RUN:
      # Also carry the blinded-window fit, so the unblinded H_T figure can show
      # what the validation pass's parameters extrapolate to over the full run.
      params.update(two_layer_fit(model, bundle, N_m, OHU_RATE_YEARS, suffix="_val"))
   return params


# ------------------------- Two-layer ocean ODE -------------------------

@dataclass(frozen=True)
class TwoLayerSolve:
   """A solved two-layer ocean, driven by an AOGCM's own T(t).

   The numeric counterpart of TwoBoxAnalytic, and carried as an object for the
   same reason: the arrays and the (c_u, c_d, gamma) they were solved with
   travel together, so a panel cannot pair one variant's T_d with another
   variant's gamma.

   `t` and the arrays carry one extra LEADING sample -- the (t=0, T=0, T_d=0)
   initial condition solve() prepends -- so they are one longer than the
   bundle. Use at_data() for anything that has to line up index-for-index with
   SeriesBundle.
   """

   t: np.ndarray
   T: np.ndarray
   T_d: np.ndarray
   N_t: np.ndarray
   c_u: float
   c_d: float
   gamma: float

   def at_data(self):
      """(T, T_d) without the prepended t=0 sample, aligned to the bundle."""
      return self.T[1:], self.T_d[1:]

   def H(self, T=None, T_d=None):
      """Deep-ocean heat-uptake FLUX gamma*(T - T_d) [W m^-2], the same
      quantity TwoBoxAnalytic.H() returns -- NOT the cumulative heat CONTENT
      H_T of Eq. 7. Defaults to this solve's own arrays."""
      T = self.T if T is None else T
      T_d = self.T_d if T_d is None else T_d
      return self.gamma * (np.asarray(T, dtype=float) - np.asarray(T_d, dtype=float))

   def upto(self, n_years):
      """Boolean mask selecting the samples at or before `n_years`."""
      return self.t <= n_years

   def H_T_per_area(self, mask=slice(None)):
      """This solve's own H_T = C_u T + C_d T_d (Eq. C41), per unit area."""
      return model_H_T_per_area(self.T[mask], self.T_d[mask], self.c_u, self.c_d)


def solve_two_layer_Nt(bundle, c_u=C_U, c_d=C_D, gamma=GAMMA):
   """Integrate the deep-layer temperature T_d(t) forced by the observed
   T(t) = bundle.T_full, then return the two-layer model's implied TOA
   imbalance N_t = c_u dT/dt + gamma*(T - T_d).

   T_d(0) = 0, matching T(t=0) = 0 at the first 4xCO2 year (t is shifted so
   that year becomes t=0). T(t') between sampled years is linearly
   interpolated to drive the ODE.
   """
   t = np.concatenate(([0.0], bundle.t_full))
   T = np.concatenate(([0.0], bundle.T_full))

   def rhs(tt, y):
      T_d = y[0]
      return [(gamma / c_d) * (np.interp(tt, t, T) - T_d)]

   sol = solve_ivp(rhs, (t[0], t[-1]), y0=[0.0], t_eval=t, method="RK45")
   if not sol.success:
      raise RuntimeError(f"solve_ivp failed: {sol.message}")
   T_d = sol.y[0]

   dTdt = np.gradient(T, t)
   N_t = c_u * dTdt + gamma * (T - T_d)
   return TwoLayerSolve(t=t, T=T, T_d=T_d, N_t=N_t, c_u=c_u, c_d=c_d, gamma=gamma)


def plot_radiative_fit(ax, ratio_ax, model, bundle, params, two_layer):
   """Reconstruction, for MT2, of the Net-TOA-vs-T2M STEP-1 panel from
   longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py:
   the 50-yr-binned fit (-> T_eq, eps), the years-1-20 annual fit (-> F_ref),
   the plain F-lambda*T line, and a ratio sub-panel carrying the eps-corrected
   F-lambda*T-(eps-1)H against that plain line. (The eps-corrected curve
   appears ONLY in the ratio sub-panel -- in the main panel it sat almost on
   top of the orange line and was dropped.)

   F_ref, T_eq, alpha (lambda here) and eps all come from fit_radiative(),
   which regresses N_T rather than N but is otherwise that script's exact
   recipe (see the module docstring and fit_radiative's own docstring) -- so
   every point and every line in the main panel is numerically IDENTICAL to
   what that script draws for these models. Only the ratio panel's green curve
   differs, because H is MT2's own two-layer FLUX gamma*(T-T_d), not that
   script's PDE-solved one.

   H here means Geoffroy (2013b)'s efficacy H -- the deep-ocean heat-uptake
   FLUX, W/m^2, matching the reference script's `pde_H_at` docstring
   ("[W/m^2]") -- NOT this script's own H_T = aogcm_H_T (Eq. 7's cumulative,
   ZJ-like heat CONTENT). Using the latter here first is why an earlier pass
   of this panel had its green line diverge by orders of magnitude: content
   and flux do not share units, and unlike the bounded flux, H_T = H - H_M
   grows without bound over a multi-millennium run (Sect. 6.3). gamma*(T-T_d)
   is exactly TwoBoxAnalytic.H()'s definition, evaluated here on `two_layer`
   -- the MT2T-fitted (c_u, c_d, gamma) ODE solve of T_d, driven by this
   AOGCM's own T(t) -- so it is the empirical analogue of that analytic H.

   Plotted over the model's WHOLE record, since fit_radiative() always is;
   the caller only produces this figure on the unblinded pass.
   """
   rad = fit_radiative(model, bundle, params["p"])
   N_m = params["N_m"]
   t, T, N = bundle.t_full, bundle.T_full, bundle.N_full
   order = np.argsort(T)
   Ts = T[order]

   _, T_d = two_layer.at_data()      # drops the solve's prepended t=0 sample

   # ----- main panel: scatter + the two provenance fits -----
   ax.scatter(T, N, s=5, alpha=0.15, color="tab:red", zorder=0, label="annual data")

   N_bin = rad.NT_bin + N_m      # NT_bin = bin_by_position(N - N_m); undo the shift
   ax.scatter(rad.T_bin, N_bin, s=28, color="tab:blue", zorder=4,
              label=f"{BIN_YEARS}-yr bins (data)")
   x_bin = np.linspace(min(0.0, float(rad.T_bin.min())), rad.T_eq, 100)
   ax.plot(x_bin, rad.mb * x_bin + rad.bb + N_m, color="tab:blue", lw=2,
           label=rf"{BIN_YEARS}-yr fit: $T_{{eq}}$={rad.T_eq:.1f} K, $\epsilon$={rad.eps:.2f}")

   # R^2/RMSE of the binned fit -- invariant to the N_m shift above (it cancels
   # in both the residual and the deviation-from-mean terms), so this is the
   # same number the diffusive-eps script would report for the same bins.
   y_pred_bin = rad.mb * rad.T_bin + rad.bb
   ss_tot = np.sum((rad.NT_bin - np.mean(rad.NT_bin)) ** 2)
   r2 = 1 - np.sum((rad.NT_bin - y_pred_bin) ** 2) / ss_tot if ss_tot > 0 else np.nan
   rmse_bin = np.sqrt(np.mean((rad.NT_bin - y_pred_bin) ** 2))
   ax.text(0.02, 0.80, f"$R^2$={r2:.3f}\nRMSE={rmse_bin:.3f}", transform=ax.transAxes,
           va="top", ha="left", fontsize=EXTRA_TEXT_FONTSIZE,
           bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.4))

   gregory = t <= GREGORY_FIT_YEARS
   ax.scatter(T[gregory], N[gregory], s=8, alpha=0.5, color="0.5",
              label=f"yrs 1-{GREGORY_FIT_YEARS} (annual)")
   x20 = np.linspace(0.0, float(T[gregory].max()), 100)
   ax.plot(x20, rad.m20 * x20 + rad.b20, color="0.35", lw=2,
           label=rf"{GREGORY_FIT_YEARS}-yr fit: $F_{{ref}}$={rad.F_ref:.3f}")

   # ----- the two reference lines: plain Gregory, and eps-corrected -----
   N_line = rad.F_ref - rad.alpha * T
   ax.plot(Ts, N_line[order], color="tab:orange", lw=2, ls="-.",
           label=rf"$F-\lambda T$ (fixed): $F$={rad.F_ref:.3f}, $\lambda$={rad.alpha:.3f}")

   # The solve carries its own gamma, so this H cannot be built from a gamma
   # that a different variant's T_d was solved with.
   N_eps = N_line - (rad.eps - 1.0) * two_layer.H(T, T_d)

   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="upper right", legend_fontsize=8)

   # ----- ratio panel: eps-corrected / plain Gregory -----
   ratio_ax.axhline(1.0, color="0.5", lw=1.0, ls=":")
   denom = np.where(np.abs(N_line) > 0.02 * abs(rad.F_ref), N_line, np.nan)
   ratio_ax.plot(Ts, (N_eps / denom)[order], color="green", lw=1.2)
   ratio_ax.tick_params(labelsize=12, width=2, length=6, direction="in")
   ratio_ax.grid(alpha=0.3)


# Ensemble-mean OHU rate (W m^-2) digitised from the paper's Fig 10b, at the
# years given, plus the half-width of that figure's shaded red uncertainty
# band at each point -- also read off the figure, so approximate in the same
# way the means are. The same reference points are drawn on every model's panel.
PAPER_ENS_MEAN = np.array([
   # year  mean   unc
   (20,   2.82,  0.45),
   (40,   2.36,  0.30),
   (60,   2.18,  0.22),
   (80,   1.96,  0.20),
   (100,  1.85,  0.20),
], dtype=float)

# --- The two-parameterization visual convention ------------------------------
# Every timeseries panel draws the same two variants, and FIT_LABEL promises the
# reader they are told apart the same way everywhere: the per-AOGCM MT2T fit
# solid, the MT2 ensemble-mean constants faded dash-dot. Defined once so the
# convention cannot drift between panels.
MT2T_TAG = "MT2T"
MT2_FIXED_TAG = "MT2"
MT2T_STYLE = dict(lw=2.0)
MT2_FIXED_STYLE = dict(lw=2.0, ls="-.", alpha=0.6)

# Opacity of the extrapolated validation-fit curves on the unblinded H_T
# figure. They repeat the primary curves' colours and styles rather than adding
# five more legend entries; the figure-level note says what the fading means.
VALIDATION_LINE_ALPHA = 0.3


def variants(primary, fixed):
   """Yield (solution, tag, style) for the MT2T fit and the MT2 fixed run.

   Skips either one that is None. That is a live case only for the analytic
   solutions -- two_box_analytic() returns None for a degenerate calibration
   (MPIESM11's) -- while the numeric solves always exist.
   """
   for sol, tag, style in ((primary, MT2T_TAG, MT2T_STYLE),
                           (fixed, MT2_FIXED_TAG, MT2_FIXED_STYLE)):
      if sol is not None:
         yield sol, tag, style

# AOGCM-mean H at year 120 of abrupt4xCO2 (ZJ), and the per-model spread at the
# same time. Not digitised: the mean follows from the paper's quoted "the RMS
# error is 380 ZJ, 9% of the AOGCM-mean H" (Sect. 3.4), and the spread is the
# axis range of Fig 11c. Both are 23-model figures, so they are a scale check
# on these 4 models, not a target.
PAPER_H_YEAR = 120.0
PAPER_H_MEAN_ZJ = 380.0 / 0.09
PAPER_H_SPAN_ZJ = (3000.0, 5500.0)

# OHUE = N/T is meaningless while T is still near zero (it runs from infinity
# under constant forcing), so the ratio curves are masked below this |T|.
OHUE_T_FLOOR_K = 0.25


def panel_tag(model, params, *extra):
   """The corner label every panel carries -- the model and its AMOC strength,
   plus any extra lines that panel wants underneath."""
   return "\n".join((model, f"M={params['M']:.1f} Sv") + extra)


def plot_ohu(ax, model, bundle, params, two_layer, two_layer_fixed,
             analytic=None, analytic_fixed=None):
   """Accumulated OHU H = H_M + H_T (Eq. 7) in ZJ against the AOGCM's own H,
   in the shape of the paper's Fig 11a.

   H_M = U_0 + p*int F dt is a straight line here, F being constant in
   abrupt4xCO2; the model's H_T is C_u T + C_d T_d, the analytic integral of the
   two-layer solve (Eq. C41), so nothing is integrated twice.

   The AOGCM H is a band, not a line: the NETTOA integral is the paper's H_N,
   and H_N/1.07 is its ocean-temperature H_O (Appendix A.3). This dataset has no
   3-D ocean temperature to settle which end is right, so both are drawn.

   The first few years of H_M are wrong by construction -- U_0 is applied in
   full from t = 0 although H_M(0) must be 0 -- which is the shaded strip.

   `analytic`/`analytic_fixed` add the PREDICTIVE curves (green/purple), the H
   counterpart of plot_ohu_rate's green "N 2-Box analytic": T(t) and T_d(t) come
   from the fitted parameters alone, run forward with no AOGCM timeseries at
   all, unlike the orange/blue DIAGNOSTIC curves above, whose T_d is ODE-solved
   from the AOGCM's own observed T(t). H_M is unchanged between the two --
   it never depended on T -- so only H_T and H = H_M + H_T move.
   """
   n_years, t, _, N = bundle.window()

   H_N = cumint(t, N) * ZJ_PER_WYRM2
   ax.fill_between(t, H_N / H_N_OVER_H_O, H_N, color="black", alpha=0.15, lw=0,
                   label=r"AOGCM $H$ ($H_O$–$H_N$)")
   ax.plot(t, H_N, color="black", lw=1.5, label=r"AOGCM $H_N$")
   ax.plot(t, H_M_per_area(t, params["N_m"]) * ZJ_PER_WYRM2, color="black",
           lw=1.5, ls=":", label=r"$H_M$")
   ax.axvspan(0, CALIB_SKIP_YEARS, color="0.5", alpha=0.15, lw=0)

   # Each solve carries its own (c_u, c_d), so the fitted and fixed variants
   # cannot be handed the wrong pair. H_M is identical for both -- same t, same
   # N_m, and it never depended on T -- so it is built once.
   H_M_model = None
   for sol, sol_analytic, tag, style in (
         (two_layer, analytic, MT2T_TAG, MT2T_STYLE),
         (two_layer_fixed, analytic_fixed, MT2_FIXED_TAG, MT2_FIXED_STYLE)):
      m = sol.upto(n_years)
      t_s = sol.t[m]
      H_T_model = sol.H_T_per_area(m)
      if H_M_model is None:
         H_M_model = H_M_per_area(t_s, params["N_m"])
      ax.plot(t_s, H_T_model * ZJ_PER_WYRM2, color="tab:orange",
              label=rf"$H_T$ {tag}", **style)
      ax.plot(t_s, (H_M_model + H_T_model) * ZJ_PER_WYRM2, color="tab:blue",
              label=rf"$H_{{MT}}$ {tag}", **style)

      if sol_analytic is not None:
         H_T_pred = model_H_T_per_area(sol_analytic.T(t_s), sol_analytic.T_d(t_s),
                                       sol.c_u, sol.c_d)
         ax.plot(t_s, H_T_pred * ZJ_PER_WYRM2, color="tab:green",
                 label=rf"$H_T$ {tag} (predictive)", **style)
         ax.plot(t_s, (H_M_model + H_T_pred) * ZJ_PER_WYRM2, color="tab:purple",
                 label=rf"$H_{{MT}}$ {tag} (predictive)", **style)

   if PAPER_H_YEAR <= n_years:
      lo, hi = PAPER_H_SPAN_ZJ
      ax.plot([PAPER_H_YEAR] * 2, [lo, hi], color="red", lw=2.5, alpha=0.4,
              solid_capstyle="butt", label="Paper yr-120 spread")
      ax.plot(PAPER_H_YEAR, PAPER_H_MEAN_ZJ, "x", color="red", ms=10, mew=2.5,
              label=r"Paper $\langle H\rangle$ yr 120")

   # H rises monotonically from the origin, so the lower right is the only
   # corner free of curves -- and the upper left is where format_ax puts the
   # model text.
   # H_T/H_M comes from the params table rather than being re-derived here, so
   # the panel and fitted_model_params.csv cannot disagree -- they are read off
   # the same window at the same year by construction.
   format_ax(ax, text=panel_tag(model, params,
                                rf"$H_T/H_M$={params['H_T_over_H_M']:.2f}"),
             xscale="linear", yscale="linear", xlim=(0, n_years),
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="lower right")


# ------------------- OHU vs. Ts, the shared "ohc_ts" panel -------------------
# The same normalised OHU-against-surface-temperature diagnostic every other
# longrun script draws, so the MT2 panel can be laid next to 2013a's and the
# diffusive one's. Both axes are normalised:
#
#   x = T_s / T_eq
#   y = OHC / OHC_eq ,  OHC_eq = (whole-ocean heat capacity) * T_eq
#
# The denominator is the WHOLE ocean's heat capacity -- 1.37e21 kg of seawater
# at 3850 J kg^-1 K^-1 -- not the model's own, which is what the other scripts
# use and the only reason their curves are comparable at all. A two-layer ocean
# only ever holds (C_u + C_d) T_eq, so its curve levels off at
# (C_u + C_d)/OCEAN_HEAT_CAPACITY_ZJ, well below 1. That ceiling is a property
# of the model, not a failure of the fit, and it is printed on each panel.
OCEAN_MASS_KG = 1.37e21
OCEAN_CP = 3850.0
OCEAN_HEAT_CAPACITY_ZJ = OCEAN_MASS_KG * OCEAN_CP / 1e21     # ~5275 ZJ K^-1

# Horizon of the predictive curve, as a multiple of that model's own slow
# timescale. Per-model rather than a fixed year count because tau_s varies by
# more than 2x across these AOGCMs; 20 tau_s puts T within e^-20 of T_eq for
# all of them, and integrating N_M any further only pushes the divergent total
# further off the top of the panel.
OHC_TS_EQUILIBRIUM_TAUS = 20.0

# Colour scale (years) of the AOGCM scatter, matching the other scripts.
OHC_TS_YEAR_MAX = 6000


def normalized_ohc(H_zj, T_eq):
   """Heat content [ZJ] -> OHC/OHC_eq, the ohc_ts panels' y-axis."""
   return np.asarray(H_zj, dtype=float) / (OCEAN_HEAT_CAPACITY_ZJ * T_eq)


def plot_ohc_ts(ax, model, bundle, params, analytic):
   """One model's OHU-vs-Ts panel: the AOGCM's own trajectory against MT2's
   PREDICTIVE one.

   `analytic` may be None -- MPIESM11's full-run calibration returns a negative
   gamma, so it has no real two-box solution. The AOGCM scatter is drawn either
   way, because the T_eq both axes are normalised by is `T_eq_fit`, the STEP-1
   regression's x-intercept, not the two-box solve's `T_eq`. (The two agree
   whenever the solve exists -- alpha is defined as F_eff/T_eq_fit -- but only
   the STEP-1 one is available for every model.)

   The prediction uses no AOGCM timeseries at all. T(t) is the constant-forcing
   two-box solution from the fitted parameters, run out past equilibrium, and
   the heat content is MT2's own (Eq. 7):

      H = H_M + H_T ,   H_M = U_0 + N_M t ,   H_T = C_u T + C_d T_d

   H_T is the exact closed-form integral of N_T = c_u dT/dt + c_d dT_d/dt over
   the analytic solution (both boxes start at zero), so nothing is integrated
   numerically here and this panel cannot disagree with plot_ohu().

   TWO curves, because MT2's two halves behave completely differently once the
   run is extended to equilibrium:

     H_T   solid. Converges, and is the direct counterpart of 2013a's "Fitted
           2-Box Pred." and the diffusive script's "1-Box + Diffusion Fit".
     H     faded. N_M = pF is a constant that never decays, so H_M = U_0 + N_M t
           grows without bound and the total leaves the top of the panel while
           T is still pinned at T_eq -- the vertical streak at x = 1. That is
           not a plotting artefact but the paper's own Sect. 6.3 caveat, that
           "N_M cannot remain non-zero and unchanged as equilibrium is
           approached under constant forcing", drawn to scale.

   Returns the scatter handle, for the figure-level colorbar.
   """
   T_eq = params["T_eq_fit"]

   H_aogcm = cumint(bundle.t_full, bundle.N_full) * ZJ_PER_WYRM2
   sc = ax.scatter(bundle.T_full / T_eq, normalized_ohc(H_aogcm, T_eq),
                   c=bundle.t_full, cmap=plt.cm.turbo, s=12,
                   norm=plt.Normalize(vmin=0, vmax=OHC_TS_YEAR_MAX))
   ax.axvline(1.0, color="0.55", ls="--", lw=0.8)

   extra = (rf"$T_{{eq}}$={T_eq:.1f} K",)
   if analytic is None:
      extra += ("no real two-box solution",)
   else:
      eq_t = np.arange(1.0, OHC_TS_EQUILIBRIUM_TAUS * analytic.tau_s + 1.0)
      T_pred = analytic.T(eq_t)
      H_T = model_H_T_per_area(T_pred, analytic.T_d(eq_t),
                               params["c_u"], params["c_d"]) * ZJ_PER_WYRM2
      H_M = H_M_per_area(eq_t, params["N_m"]) * ZJ_PER_WYRM2

      ax.plot(T_pred / T_eq, normalized_ohc(H_T, T_eq), color="tab:green",
              lw=2.0, label=r"MT2 $H_T$ (predictive)")
      ax.plot(T_pred / T_eq, normalized_ohc(H_T + H_M, T_eq), color="tab:green",
              lw=2.0, ls="-.", alpha=0.45, label=r"MT2 $H=H_M+H_T$")

      ceiling = ((params["c_u"] + params["c_d"]) * ZJ_PER_WYRM2
                 / OCEAN_HEAT_CAPACITY_ZJ)
      extra += (rf"$(C_u+C_d)/C_{{ocean}}$={ceiling:.2f}",)

   format_ax(ax, text=panel_tag(model, params, *extra),
             xscale="linear", yscale="linear", ylim=(-0.05, 1.2),
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="center left",
             legend=analytic is not None)   # nothing labelled without the fit
   return sc


# ------------------- Heat uptake against T, in BOTH senses -------------------
# The MT2 counterpart of the "H vs T2M" STEP-1 panel of
# longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py.
# That script has one H; MT2 has two quantities the same letter is used for, and
# conflating them is what once made plot_radiative_fit's green line diverge by
# orders of magnitude (see its docstring), so they get one panel each:
#
#   plot_HT_vs_T      the CUMULATIVE heat content H_T = C_u T + C_d T_d [ZJ]
#                     (Eq. 7 / C41) -- plot_ohu's y-axis, against T instead of t.
#   plot_H_flux_vs_T  the deep-ocean heat-uptake FLUX H = gamma(T - T_d)
#                     [W m^-2] -- Geoffroy (2013b)'s efficacy H, which is the
#                     one the reference script's panel actually draws.
#
# Both share the same cast: the AOGCM, the two DIAGNOSTIC solves (T(t) taken
# from the AOGCM, T_d integrated) and the two PREDICTIVE two-box solutions
# (parameters only, no AOGCM timeseries), told apart by the MT2T/MT2 convention
# above and coloured as in plot_ohu -- black AOGCM, orange diagnostic, green
# predictive.

# Below this |eps - 1| the AOGCM-side inversion H = (F - N - lambda T)/(eps - 1)
# is division by ~nothing: the efficacy term is what carries H in the radiative
# relation, so an eps that close to 1 leaves H unconstrained by it. The panel
# then drops the AOGCM curve rather than drawing a near-vertical artefact.
EFFICACY_INVERSION_FLOOR = 0.02


def predictive_years(analytic):
   """Annual time grid for a predictive curve, run to equilibrium.

   Same horizon rule as plot_ohc_ts (20 slow timescales, per model rather than a
   fixed year count), which is what lets these panels show the approach to T_eq
   even on the blinded pass, where the AOGCM data stop at OHU_RATE_YEARS.
   """
   return np.arange(0.0, OHC_TS_EQUILIBRIUM_TAUS * analytic.tau_s + 1.0)


def plot_HT_vs_T(ax, model, bundle, params, two_layer, two_layer_fixed,
                 analytic=None, analytic_fixed=None, window=SMOOTH_WINDOW_YEARS):
   """Accumulated H_T [ZJ] against surface temperature.

   The AOGCM's H_T is H_N - H_M (Appendix C.4.1), the same quantity the MT2T
   calibration regresses, so its early-T dip below zero is the U_0 artefact
   CALIB_SKIP_YEARS drops -- unavoidable here, since in T-space there is no
   "first five years" strip to shade.

   The DIAGNOSTIC curves (orange) are each solve's own C_u T + C_d T_d against
   that solve's T, which is the AOGCM's own T -- so they share the black curve's
   x-values and the vertical gap between them is the model's H_T error at a
   given temperature. The PREDICTIVE curves (green) supply their own T(t) as
   well, and are run out to equilibrium, so they extend to the right of
   everything else and end at T_eq: that overshoot of the data's x-range is the
   point of the panel, not an axis-limit bug.

   Both H_T and T are `window`-year running means, matching the reference
   script's 10-yr smoothing of its H-vs-T panel. The analytic curves are already
   smooth and are left alone.
   """
   n_years, t, T_raw, N = bundle.window()
   T = running_mean(T_raw, window)

   ax.plot(T, running_mean(aogcm_H_T(t, N, params["N_m"]), window) * ZJ_PER_WYRM2,
           color="black", lw=1.5, label=r"AOGCM $H_T=H_N-H_M$")

   # Each solve carries its own (c_u, c_d) and is zipped with the analytic
   # solution built from the SAME parameters, so a predictive curve cannot be
   # drawn with the other variant's heat capacities.
   for sol, sol_analytic, tag, style in (
         (two_layer, analytic, MT2T_TAG, MT2T_STYLE),
         (two_layer_fixed, analytic_fixed, MT2_FIXED_TAG, MT2_FIXED_STYLE)):
      m = sol.upto(n_years)
      ax.plot(running_mean(sol.T[m], window),
              running_mean(sol.H_T_per_area(m), window) * ZJ_PER_WYRM2,
              color="tab:orange", label=rf"$H_T$ {tag}", **style)

      if sol_analytic is not None:
         t_pred = predictive_years(sol_analytic)
         T_pred = sol_analytic.T(t_pred)
         H_T_pred = model_H_T_per_area(T_pred, sol_analytic.T_d(t_pred),
                                       sol.c_u, sol.c_d)
         ax.plot(T_pred, H_T_pred * ZJ_PER_WYRM2, color="tab:green",
                 label=rf"$H_T$ {tag} (predictive)", **style)

   if analytic is not None:
      ax.axvline(analytic.T_eq, color="tab:green", ls=":", lw=1.2, alpha=0.7,
                 label=rf"$T_{{eq}}$={analytic.T_eq:.1f} K (MT2T)")

   # H_T climbs left-to-right, so the lower right is the free corner -- the same
   # reasoning (and the same answer) as plot_ohu's legend placement.
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="lower right",
             legend_fontsize=9)


def plot_H_flux_vs_T(ax, model, bundle, params, two_layer, two_layer_fixed,
                     analytic=None, analytic_fixed=None,
                     window=SMOOTH_WINDOW_YEARS):
   """Deep-ocean heat-uptake FLUX H = gamma(T - T_d) [W m^-2] against surface
   temperature -- the direct MT2 counterpart of the reference script's H-vs-T2M
   STEP-1 panel, drawn the same way it is there.

   The AOGCM has no H of its own in this dataset (no 3-D ocean temperature), so
   its curve is inverted out of the radiative relation exactly as the reference
   script does it,

      H = (F - N - lambda T)/(eps - 1),

   with F, lambda and eps the deterministic STEP-1 scalars from fit_radiative()
   and N the AOGCM's own NETTOA. It is therefore a radiative reconstruction, not
   an ocean diagnostic, and it inherits everything that fit is wrong about --
   which is why it is drawn as scattered points rather than as a line. It is
   dropped entirely when eps sits within EFFICACY_INVERSION_FLOOR of 1.

   The grey reference line is the reference script's, H = (F - lambda T)/eps:
   the locus where N = H, i.e. where the whole TOA imbalance is going into the
   deep ocean. It is a limit case here rather than an MT2 identity -- MT2's own
   budget is N = N_M + N_T -- so it marks the panel's upper envelope.

   Orange/green and the smoothing follow plot_HT_vs_T.
   """
   n_years, t, T_raw, N = bundle.window()
   T = running_mean(T_raw, window)
   F_ref, lam, eps = params["F_ref"], params["alpha"], params["eps"]

   if abs(eps - 1.0) > EFFICACY_INVERSION_FLOOR:
      H_inv = running_mean((F_ref - N - lam * T_raw) / (eps - 1.0), window)
      ax.scatter(T, H_inv, s=8, alpha=0.5, color="black",
                 label=r"AOGCM $H=(F-N-\lambda T)/(\epsilon-1)$")

   T_line = np.linspace(0.0, max(float(T.max()), params["T_eq_fit"]), 100)
   ax.plot(T_line, (F_ref - lam * T_line) / eps, color="0.5", lw=1.0, ls="-.",
           label=r"$H=(F-\lambda T)/\epsilon$ ($N=H$)")

   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m = sol.upto(n_years)
      ax.plot(running_mean(sol.T[m], window),
              running_mean(sol.H(sol.T[m], sol.T_d[m]), window),
              color="tab:orange", label=rf"$H=\gamma(T-T_d)$ {tag}", **style)
   for sol, tag, style in variants(analytic, analytic_fixed):
      t_pred = predictive_years(sol)
      ax.plot(sol.T(t_pred), sol.H(t_pred), color="tab:green",
              label=rf"$H$ {tag} (predictive)", **style)

   if analytic is not None:
      ax.axvline(analytic.T_eq, color="tab:green", ls=":", lw=1.2, alpha=0.7,
                 label=rf"$T_{{eq}}$={analytic.T_eq:.1f} K (MT2T)")
   ax.axhline(0.0, color="0.55", lw=0.8, ls=":")

   # The flux decays towards zero as T -> T_eq, so unlike the H_T panel the free
   # corner is the upper right.
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="upper right",
             legend_fontsize=9)


def plot_ohue(ax, model, bundle, params, two_layer, two_layer_fixed,
              window=SMOOTH_WINDOW_YEARS):
   """Ocean heat uptake efficiency kappa = N/T = kappa_M + q (Eq. 10) against
   the AOGCM's own N/T, on the same `window`-year running mean as the OHU-rate
   panels.

   kappa_M = N_M/T and q = N_T/T, the latter from the two-layer solve. Both
   diverge as T -> 0, so everything below OHUE_T_FLOOR_K is masked out.
   """
   n_years, t, T_raw, N = bundle.window()
   T = running_mean(T_raw, window)

   def over_T(num, den):
      """num/den, blanked wherever |den| is too small to mean anything."""
      num = np.broadcast_to(np.asarray(num, dtype=float), np.shape(den))
      out = np.full(np.shape(den), np.nan)
      ok = np.abs(den) >= OHUE_T_FLOOR_K
      out[ok] = num[ok] / np.asarray(den, dtype=float)[ok]
      return out

   ax.plot(t, over_T(running_mean(N, window), T), color="black",
           lw=1.5, label=r"AOGCM $\kappa=N/T$")
   ax.plot(t, over_T(params["N_m"], T), color="black", lw=1.5, ls=":",
           label=r"$\kappa_M=N_M/T$")

   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m = sol.upto(n_years)
      T_sm = running_mean(sol.T[m], window)
      q = over_T(running_mean(sol.N_t[m], window), T_sm)
      ax.plot(sol.t[m], q, color="tab:orange", label=rf"$q=N_T/T$ {tag}", **style)
      ax.plot(sol.t[m], over_T(params["N_m"], T_sm) + q, color="tab:blue",
              label=rf"$\kappa_{{MT}}$ {tag}", **style)

   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(0, n_years))


def plot_ohu_rate(ax, model, bundle, params, two_layer, two_layer_fixed,
                  analytic=None, analytic_fixed=None,
                  window=SMOOTH_WINDOW_YEARS):
   """Plot one model's 4xCO2 OHU-rate decomposition N = N_M + N_T over the
   panel window, using this model's fitted (time-independent) N_m, plus the
   two-layer ocean model's independently-derived N_T for comparison.

   Every time-varying curve is a `window`-year centred running mean; the raw
   annual AOGCM N is kept underneath in grey for reference. N_M is a fitted
   constant, so smoothing it would be a no-op.

   `two_layer_fixed` is the same two-layer N_T run on the MT2 ensemble-mean
   constants, drawn faded against the per-AOGCM (MT2T) fit.

   `analytic` adds the constant-forcing two-box prediction of the TOTAL
   N = F_ref - alpha T(t) - (eps-1)H(t) in green, against the black AOGCM N.
   Unlike every other curve here it uses no AOGCM timeseries at all, only the
   fitted parameters, so it is the one honest prediction on the panel. Its N_T
   is not drawn separately: N_T = N - N_M is the green curve minus the dotted
   line."""
   n_years, t, _, N_full = bundle.window()
   N_m = params["N_m"]

   smoothed = f"{window}-yr mean"
   ax.plot(t, N_full, color="0.6", lw=0.8, alpha=0.6, label="AOGCM N (annual)")
   ax.plot(t, running_mean(N_full, window), color="black", lw=1.5,
           label=f"AOGCM N ({smoothed})")
   # The three AOGCM-derived curves share one colour (black) and are told apart
   # by line style; the 2-layer model's independent N_T is the one coloured
   # curve, so it reads immediately against the AOGCM set.
   ax.axhline(N_m, color="black", lw=1.5, ls=":", label=r"$N_M$")
   ax.plot(t, running_mean(N_full - N_m, window), color="black", lw=1.5, ls="--",
           label=rf"$N_T$ ({smoothed})")

   # lw 1.5, not the shared styles' 2.0: these sit among the thin AOGCM curves
   # above rather than among the thick predictive ones.
   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m = sol.upto(n_years)
      ax.plot(sol.t[m], running_mean(sol.N_t[m], window), color="tab:blue",
              label=rf"$N_T$ {tag} ({smoothed})", **{**style, "lw": 1.5})
   for sol, tag, style in variants(analytic, analytic_fixed):
      ax.plot(t, sol.N(t), color="tab:green",
              label=rf"$N$ 2-Box analytic ({tag})", **style)

   ax.errorbar(PAPER_ENS_MEAN[:, 0], PAPER_ENS_MEAN[:, 1], yerr=PAPER_ENS_MEAN[:, 2],
              fmt="x", color="red", ms=10, mew=2.5, elinewidth=2.0, capsize=4,
              ls="none", label="Ens. Mean (paper)")
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(0, n_years))


def plot_temperature_timeseries(ax, model, bundle, params, two_layer, two_layer_fixed,
                                analytic=None, analytic_fixed=None):
   """Plot one model's AOGCM surface temperature against (a) the two-layer
   model's DEEP-layer temperature T_d and (b) the constant-forcing analytic
   two-box surface temperature T(t), over the panel window. The `*_fixed`
   arguments are the same curves on the MT2 ensemble-mean constants, drawn
   faded for comparison.

   The two-layer ODE's own upper-box temperature is not plotted because it is
   not a solution: T there is the prescribed AOGCM boundary condition that drives
   the ODE, so it would duplicate the black curve exactly. The green analytic
   T(t) is the genuine prediction -- it comes from the fitted parameters alone
   and never sees the AOGCM temperature -- so the black-to-green gap is the
   model's actual surface-temperature error."""
   n_years, t, T, _ = bundle.window()

   ax.plot(t, T, color="black", lw=1.5, label=r"AOGCM ($T_{2\mathrm{-meter}}$)")
   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m = sol.upto(n_years)
      ax.plot(sol.t[m], sol.T_d[m], color="tab:blue",
              label=rf"2-Box $T_{{\mathrm{{deep}}}}$ ({tag})",
              **{**style, "lw": 1.5})
   for sol, tag, style in variants(analytic, analytic_fixed):
      ax.plot(t, sol.T(t), color="tab:green",
              label=rf"2-Box $T$ analytic ({tag})", **style)
   if analytic is not None:
      ax.axhline(analytic.T_eq, color="tab:green", ls=":", lw=1.2, alpha=0.7,
                 label=rf"$T_{{eq}}$={analytic.T_eq:.1f} K (MT2T)")

   # T and T_analytic both flatten towards T_eq across the top of the panel, so
   # an upper-corner legend sits on the curves; the deep-layer curves leave the
   # lower right clear.
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(0, n_years), legend_loc="lower right")


def plot_HT_decomposition(ax, model, bundle, params, val_params=None):
   """MT2T fit diagnostic, in the shape of the appendix's Fig 17a: every term
   of

      H_T = C_u T + gamma(1 + C_u/C_d) int T dt' - (gamma/C_d) int H_T dt'

   over this model's fit window, in ZJ. The green sum of the three right-hand
   terms lying on top of H_T is the check that the regression coefficients
   were mapped back to (C_u, C_d, gamma) correctly.

   `val_params` (unblinded pass only) draws the same terms a second time,
   faded, from the validation pass's `(c_u, c_d, gamma)` -- fit over the first
   OHU_RATE_YEARS years and extrapolated across the whole run. Where that faded
   Sum peels away from H_T is where the short-window fit stops holding. H_T
   itself is parameter-free, so it is drawn once. Passing it explicitly keeps
   the decision at the call site, next to the caption that explains the fading.

   The shaded strip at the left is the first CALIB_SKIP_YEARS years, whose rows
   were excluded from the regression: H_T is negative there because H_M carries
   its full U_0 from t = 0."""
   n_years, t, T, N = bundle.window()

   H_T = aogcm_H_T(t, N, params["N_m"])
   int_T = cumint(t, T)
   int_H = cumint(t, H_T)

   def draw_terms(c_u, c_d, gamma, opacity, labelled):
      # `opacity` is the matplotlib alpha, not the feedback parameter.
      term_T = c_u * T
      term_intT = gamma * (1.0 + c_u / c_d) * int_T
      term_intH = -(gamma / c_d) * int_H
      tag = (lambda s: s) if labelled else (lambda s: "_nolegend_")
      ax.plot(t, term_T * ZJ_PER_WYRM2, color="black", lw=1.5, ls="--",
              alpha=opacity, label=tag(r"$C_u T$"))
      ax.plot(t, term_intT * ZJ_PER_WYRM2, color="black", lw=1.5, alpha=opacity,
              label=tag(r"$\gamma(1+C_u/C_d)\int T\,dt$"))
      ax.plot(t, term_intH * ZJ_PER_WYRM2, color="red", lw=1.5, alpha=opacity,
              label=tag(r"$-(\gamma/C_d)\int H_T\,dt$"))
      ax.plot(t, (term_T + term_intT + term_intH) * ZJ_PER_WYRM2,
              color="tab:green", lw=2.0, ls="--", alpha=opacity, label=tag("Sum"))
      return np.concatenate([term_T, term_intT, term_intH,
                             term_T + term_intT + term_intH]) * ZJ_PER_WYRM2

   ax.plot(t, H_T * ZJ_PER_WYRM2, color="red", lw=2.0, ls="--", label=r"$H_T$")
   drawn = draw_terms(params["c_u"], params["c_d"], params["gamma"], 1.0, True)

   text = (f"{model}\n"
           rf"$C_u$={params['C_u_ZJ']:.0f}$\pm${params['C_u_ZJ_unc']:.0f} ZJ/K" "\n"
           rf"$C_d$={params['C_d_ZJ']:.0f}$\pm${params['C_d_ZJ_unc']:.0f} ZJ/K" "\n"
           rf"$\gamma$={params['gamma']:.3f}$\pm${params['gamma_unc']:.3f} W m$^{{-2}}$K$^{{-1}}$" "\n"
           rf"$R$={params['R_HT']:.4f}")

   ylim = None
   if val_params is not None:
      draw_terms(val_params["c_u"], val_params["c_d"], val_params["gamma"],
                 VALIDATION_LINE_ALPHA, False)
      text += ("\n"
               rf"val: $C_u$={val_params['C_u_ZJ']:.0f}, "
               rf"$C_d$={val_params['C_d_ZJ']:.0f} ZJ/K, "
               rf"$\gamma$={val_params['gamma']:.3f}")
      # Scale to the primary fit and H_T only. A short-window fit extrapolated
      # over thousands of years can diverge by orders of magnitude (MPIESM11
      # does), which would otherwise flatten the curves the panel is about;
      # let the faded curves leave the panel instead.
      span = np.concatenate([drawn, H_T * ZJ_PER_WYRM2])
      lo, hi = float(np.min(span)), float(np.max(span))
      pad = 0.08 * (hi - lo) if hi > lo else 1.0
      ylim = (lo - pad, hi + pad)

   format_ax(ax, text=text, xscale="linear", yscale="linear", xlim=(0, n_years),
             ylim=ylim, text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="lower left")


# ------------- tau_d vs. calibration window (2013a/b's tau_s plot) -------------
# The MT2 analogue of the 2013a/2013b scripts' "tau_s vs. calibration window"
# panels: refit the MT2T two-layer regression over progressively longer chunks
# of the record and watch the deep-ocean timescale it implies.

# First calibration window of the sweep, and its step. Anything much shorter
# than this leaves the regression with only a handful of rows once
# CALIB_SKIP_YEARS is dropped.
TAU_D_MIN_WINDOW_YEARS = 20
TAU_D_WINDOW_STEP_YEARS = 1


def tau_d_vs_window(model, bundle, params, min_window=TAU_D_MIN_WINDOW_YEARS,
                    step=TAU_D_WINDOW_STEP_YEARS):
   """Deep-ocean timescale as a function of the MT2T calibration window.

   Only the two-layer regression is refit per window; the deterministic
   radiative parameters (F_eff, alpha, eps) stay at their whole-record STEP-1
   values, so the curves isolate how much of tau_d's drift comes from the ocean
   fit rather than from the feedback.

   Returns (windows, tau_slow, tau_deep), both timescales in years:

     tau_slow  the two-box system's SLOW eigenvalue, Geoffroy et al. (2013a)
               Eqs. 8-15 in the EBM-epsilon form of 2013b -- the same closed
               form two_box_analytic() uses for the analytic curves, evaluated
               on each window's (c_u, c_d, gamma) with that model's fixed alpha
               and eps:
                  b     = (alpha+eps*gamma)/c_u + gamma/c_d
                  delta = b^2 - 4 alpha gamma/(c_u c_d)
                  tau_s = c_u c_d / (2 alpha gamma) * (b + sqrt(delta))
     tau_deep  c_d/gamma, the deep layer's own relaxation time, which uses the
               two-layer parameters alone and knows nothing about alpha.

   Windows whose fit is degenerate -- a negative gamma or c_d, or no real pair
   of timescales -- give NaN rather than a complex or negative tau, and appear
   as a gap in the plotted curve.
   """
   last_year = bundle.panel_years
   windows = np.arange(min_window, last_year + step, step, dtype=float)
   windows = windows[windows <= last_year]

   alpha = float(params["alpha"])
   F_eff = float(params["F_eff"])
   eps = float(params["eps"])
   N_m = float(params["N_m"])

   tau_slow = np.full(windows.shape, np.nan)
   tau_deep = np.full(windows.shape, np.nan)
   for i, window in enumerate(windows):
      fit = two_layer_fit(model, bundle, N_m, window, verbose=False)
      c_u, c_d, gamma = fit["c_u"], fit["c_d"], fit["gamma"]
      if c_d > 0.0 and gamma > 0.0:
         tau_deep[i] = c_d / gamma
      sol = two_box_analytic(alpha, F_eff, c_u, c_d, gamma, eps=eps)
      if sol is not None:
         tau_slow[i] = sol.tau_s

   return windows, tau_slow, tau_deep


def plot_tau_d_vs_window(ax, model, params, windows, tau_slow, tau_deep):
   """One model's panel of the two tau_d estimates against calibration window,
   with the 1:1 line the 2013a/b panels carry -- where a curve crosses it, the
   fit is claiming a timescale as long as the record it was fit to, which is
   the point past which the estimate is extrapolation rather than measurement.

   The y-axis is clipped to the 99th percentile of the plotted values: tau_slow
   runs away by orders of magnitude wherever a window's fit is near-degenerate,
   and letting those excursions set the scale would flatten everything else.
   """
   ax.plot(windows, tau_slow, color="tab:blue", lw=2.0,
           label=r"$\tau_d$ slow mode (2013a Eqs. 8-15)")
   ax.plot(windows, tau_deep, color="tab:orange", lw=2.0,
           label=r"$\tau_d=C_d/\gamma$")
   ax.plot(windows, windows, color="black", lw=1.2, ls="--",
           label=r"$\tau_d$ = window")

   finite = np.concatenate([tau_slow, tau_deep])
   finite = finite[np.isfinite(finite)]
   ylim = None
   if finite.size:
      hi = float(np.nanpercentile(finite, 99))
      lo = min(0.0, float(np.min(finite)))
      if hi > lo:
         ylim = (lo, hi + 0.15 * (hi - lo))

   def last_finite(y):
      ok = np.flatnonzero(np.isfinite(y))
      return y[ok[-1]] if ok.size else np.nan

   format_ax(ax, text=panel_tag(
                model, params,
                rf"full-run $\tau_d$: {last_finite(tau_slow):.0f} yr (slow), "
                rf"{last_finite(tau_deep):.0f} yr ($C_d/\gamma$)"),
             xscale="linear", yscale="linear",
             xlim=(float(windows[0]), float(windows[-1])), ylim=ylim,
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="lower right")


# ------------------------- Load R data -------------------------

rdata_file = Path("./data/int_netToa_longrun.Rdata")
ro.r["load"](str(rdata_file))

data = ro.globalenv["int_nettoa_longrun_data"]
expts = list(ro.globalenv["expts"])
expts = [str(x) for x in expts]

layout = RunLayout(VARIANT, 1, results=UNBLINDED if UNBLINDED_RUN else VALIDATION)

# Title fragment describing the span the timeseries panels cover, so the figure
# says which pass it came from.
YEARS_LABEL = "full run" if UNBLINDED_RUN else f"first {OHU_RATE_YEARS} years"

# ... and which two-layer parameterization produced it.
FIT_LABEL = "MT2T per-AOGCM fit vs MT2 fixed params"

print("==================================")
print(f"Output: {layout.label}")
print("==================================")

df = pd.DataFrame(columns=PARAM_COLS)
df["model"] = fit_models
df["M"] = [M_SV[m] for m in fit_models]

# Cache all model series so the fit and the plots use identical baselines/slices.
series_by_model: Dict[str, SeriesBundle] = {
   model: make_series_bundle(data.rx2(model), expts) for model in fit_models
}


# ------------------------- Run fitting procedure -------------------------

for model in fit_models:
   bundle = series_by_model[model]
   params = fit_model(model, bundle, M_SV[model])
   update_row(df, model, params)


# ------------------------- Two-layer ODE: N_t timeseries -------------------------

# Every model is solved twice: once on its own fitted (MT2T) parameters, once
# on the MT2 ensemble-mean constants, so both appear on every panel. The
# constant-forcing analytic solution is built for the same two parameter sets.
two_layer_by_model: Dict[str, tuple] = {}
two_layer_fixed_by_model: Dict[str, tuple] = {}
analytic_by_model: Dict[str, TwoBoxAnalytic] = {}
analytic_fixed_by_model: Dict[str, TwoBoxAnalytic] = {}
for model in fit_models:
   bundle = series_by_model[model]
   row = df.loc[df["model"] == model].iloc[0]
   two_layer_kw = dict(c_u=row["c_u"], c_d=row["c_d"], gamma=row["gamma"])
   solve = solve_two_layer_Nt(bundle, **two_layer_kw)
   solve_fixed = solve_two_layer_Nt(bundle)
   two_layer_by_model[model] = solve
   two_layer_fixed_by_model[model] = solve_fixed

   # Constant-forcing analytic solution. Driven by F_eff = (1-p)F_ref, because
   # N_M = pF never reaches the surface temperature (Eq. 25), but carrying the
   # full F_ref as F_total so N() can return the TOTAL imbalance the AOGCM
   # curve is compared against. eps is the deterministic STEP-1 value for both
   # variants; alpha is NOT -- the MT2T variant (this AOGCM's own fitted
   # c_u/c_d/gamma) uses the deterministic STEP-1 alpha (F_eff/T_eq, off the
   # 50-yr-binned fit), while the MT2 fixed variant (ensemble-mean C_U/C_D/
   # GAMMA) keeps `alpha_g20`, the plain years-1-20 slope MT2 used before that
   # recipe existed -- pairing the fixed ocean constants with a feedback
   # parameter that owes nothing to any per-AOGCM fit.
   radiative_kw = dict(F_eff=row["F_eff"], eps=row["eps"], F_total=row["F_ref"])
   analytic_by_model[model] = two_box_analytic(
      lam=row["alpha"], **radiative_kw, **two_layer_kw)
   analytic_fixed_by_model[model] = two_box_analytic(
      lam=row["alpha_g20"], **radiative_kw, c_u=C_U, c_d=C_D, gamma=GAMMA)
   if analytic_by_model[model] is None:
      print(f"   WARNING {model}: no real two-box solution from the MT2T "
            f"parameters (alpha={row['alpha']:.3f}, gamma={row['gamma']:.3f}, "
            f"eps={row['eps']:.3f}) -- analytic curves omitted for this model")
   for sol, suffix in ((analytic_by_model[model], ""),
                       (analytic_fixed_by_model[model], "_MT2")):
      if sol is not None:
         update_row(df, model, sol.as_params(suffix))

   # Both solves in one CSV, with the parameters behind each, so it is
   # self-describing: *_fit columns are the MT2T solve, *_fixed the MT2 one.
   ts_df = pd.DataFrame({"t_years": solve.t, "T": solve.T,
                         "T_d_fit": solve.T_d, "N_t_fit": solve.N_t,
                         "T_d_fixed": solve_fixed.T_d, "N_t_fixed": solve_fixed.N_t})
   for key, val in {**two_layer_kw,
                    "c_u_fixed": C_U, "c_d_fixed": C_D, "gamma_fixed": GAMMA}.items():
      ts_df[key] = val
   ts_df.to_csv(layout.csv_path(f"{model}_two_layer_Nt"), index=False)

   # End-of-window H and OHUE scalars for the params table -- the same
   # quantities plot_ohu() and plot_ohue() draw, read off at the last year the
   # panels show, so the table and the figures cannot disagree. plot_ohu() reads
   # H_T_over_H_M straight back out of here rather than re-deriving it.
   n_years_end, t_b, _, N_b = bundle.window()
   m_s = solve.upto(n_years_end)
   H_N_end = float(cumint(t_b, N_b)[-1] * ZJ_PER_WYRM2)
   H_M_end = float(H_M_per_area(solve.t[m_s], row["N_m"])[-1] * ZJ_PER_WYRM2)
   H_T_end = float(solve.H_T_per_area(m_s)[-1] * ZJ_PER_WYRM2)
   T_end = float(running_mean(solve.T[m_s])[-1])
   N_t_end = float(running_mean(solve.N_t[m_s])[-1])
   N_end = float(running_mean(N_b)[-1])
   update_row(df, model, {
      "end_year": float(t_b[-1]),
      "H_aogcm_N_end": H_N_end,
      "H_aogcm_O_end": H_N_end / H_N_OVER_H_O,
      "H_M_end": H_M_end,
      "H_T_end": H_T_end,
      "H_MT_end": H_M_end + H_T_end,
      "H_T_over_H_M": H_T_end / H_M_end if H_M_end else np.nan,
      "kappa_aogcm_end": N_end / T_end if abs(T_end) >= OHUE_T_FLOOR_K else np.nan,
      "kappa_MT_end": ((row["N_m"] + N_t_end) / T_end
                       if abs(T_end) >= OHUE_T_FLOOR_K else np.nan),
   })

df.to_csv(layout.csv_path("fitted_model_params"), index=False)
print(df.to_string())


# ------------------------- Plots for all models -------------------------

# Every model's fitted row as a plain dict, looked up once instead of
# re-scanning the DataFrame in each of the figure loops below.
params_by_model: Dict[str, dict] = {
   model: df.loc[df["model"] == model].iloc[0].to_dict() for model in fit_models
}

# The validation-window two-layer fit with its "_val" suffix stripped, so the
# H_T panel can be handed it as an ordinary params dict. fit_model() only emits
# those columns on the unblinded pass, so this is empty otherwise -- which is
# what makes the faded overlay an explicit argument rather than something the
# panel infers from which keys happen to be present.
val_params_by_model: Dict[str, dict] = {
   model: stripped
   for model, params in params_by_model.items()
   if (stripped := {k[:-4]: v for k, v in params.items() if k.endswith("_val")})
   and np.isfinite(stripped.get("c_u", np.nan))
}


def model_figure(title, ylabel, draw, xlabel="Year", note=None, top=None):
   """Build one all-models grid figure and draw a panel per model.

   Saving is deliberately left to the caller. check_figure_paths.py audits
   output paths by parsing this file with `ast`, and it can only check that two
   figures do not collide if each layout.save_results(fig, figname(...)) stays
   literal at its own call site -- routing them through a `name` parameter here
   would collapse all six into one unverifiable "{name}" entry.

   `draw(ax, model, bundle, params)` fills one panel; whatever it returns is
   collected and handed back, which is how the ohc_ts figure gets the scatter
   handle for its colorbar. `note` is the italic figure-level caption, placed
   relative to the top margin rather than at a second hand-tuned constant, and
   `top` deepens that margin for the captions that need two lines.

   Returns (fig, axs, handles); the caller closes the figure after any extra
   figure-level work.
   """
   fig, axs = make_model_grid(fit_models, title=title, xlabel=xlabel,
                              ylabel=ylabel, auto_rows=True, hide_unused=True)
   margins = {**GRID_MARGINS, **({} if top is None else {"top": top})}
   fig.subplots_adjust(**margins)
   if note:
      # The caption is anchored va="top", so it grows DOWNWARD into the panels;
      # each extra line has to be cleared above the top margin. 0.035 for one
      # line and 0.065 per line after that reproduces the hand-tuned y=0.915
      # (1 line, top=0.88) and y=0.93 (2 lines, top=0.83) these figures used
      # before the two numbers were coupled.
      y = margins["top"] + 0.035 + 0.065 * note.count("\n")
      fig.text(0.5, y, note, ha="center", va="top",
               fontsize=AXIS_LABEL_FONTSIZE - 6, fontstyle="italic")
   handles = [draw(axs[i], model, series_by_model[model], params_by_model[model])
              for i, model in enumerate(fit_models)]
   return fig, axs, handles

# Net TOA vs T2M (STEP-1 radiative fit): reconstruction of
# longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py's
# Step-1 panel -- see plot_radiative_fit()'s docstring. fit_radiative() is
# always fit over the whole record, so like the ohc_ts and tau_d panels below,
# this figure is produced on the unblinded pass only.
#
# The one figure model_figure() cannot build: it needs the paired (main, ratio)
# axes of make_model_grid_with_ratio, whose gridspec sets its own margins.
if UNBLINDED_RUN:
   fig, axs, ratio_axs = make_model_grid_with_ratio(
      fit_models,
      title=r"MT2: 4xCO$_2$ Net TOA vs T$_{2M}$ (STEP-1 radiative fit)",
      xlabel="2-meter Air Temperature Anomaly (K)",
      ylabel=r"Net TOA Radiative Flux Anomaly ($W\,m^{-2}$)",
      ratio_ylabel=r"$\frac{F-\lambda T-(\epsilon-1)H}{F-\lambda T}$",
      ncols=len(fit_models),
   )
   for imodel, model in enumerate(fit_models):
      plot_radiative_fit(axs[imodel], ratio_axs[imodel], model,
                         series_by_model[model], params_by_model[model],
                         two_layer_by_model[model])

   layout.save_results(fig, figname("4xCO2", "all_models", "T2M_vs_NETTOA"))
   plt.close(fig)


# ------------------- Rate of OHU N = N_M + N_T (Fig 10b) -------------------

fig, _, _ = model_figure(
   title=(f"MT2: 4xCO2 Rate of OHU by AMOC strength (M), {YEARS_LABEL} "
          f"({SMOOTH_WINDOW_YEARS}-year running mean, {FIT_LABEL})"),
   ylabel=r"Rate of OHU ($W\,m^{-2}$)",
   draw=lambda ax, model, bundle, params: plot_ohu_rate(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model],
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "OHU_rate_vs_Year"))
plt.close(fig)


# ------------------------- Surface temperature -------------------------

fig, _, _ = model_figure(
   title=f"MT2: 4xCO2 Temperature by AMOC strength (M), {YEARS_LABEL} ({FIT_LABEL})",
   ylabel=r"T$_{2m}$ Anomaly (K)",
   draw=lambda ax, model, bundle, params: plot_temperature_timeseries(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model],
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "Temperature_vs_Year"))
plt.close(fig)


# ------------------- Accumulated OHU H = H_M + H_T (Eq. 7) -------------------

fig, _, _ = model_figure(
   title=(f"MT2: 4xCO2 accumulated OHU $H=H_M+H_T$ by AMOC strength (M), "
          f"{YEARS_LABEL} ({FIT_LABEL})"),
   ylabel="OHU (ZJ)",
   note=("Shaded strip: first "
         f"{CALIB_SKIP_YEARS} years, where $H_M$ carries its full $U_0$ although "
         r"$H_M(0)=0$. Paper reference points are 23-model abrupt4xCO2 values; "
         "Table 1's OHU column is 1pctCO2 and not comparable."),
   draw=lambda ax, model, bundle, params: plot_ohu(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model],
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "OHU_vs_Year"))
plt.close(fig)


# ------------------- Heat uptake vs. T (the reference script's panel) --------
# Two figures, because MT2 uses "H" for both a heat content and a flux; see the
# section comment above plot_HT_vs_T().

fig, _, _ = model_figure(
   title=(f"MT2: 4xCO2 accumulated $H_T$ vs T$_{{2M}}$ by AMOC strength (M), "
          f"{YEARS_LABEL} ({SMOOTH_WINDOW_YEARS}-year running mean, {FIT_LABEL})"),
   xlabel="2-meter Air Temperature Anomaly (K)",
   ylabel=r"$H_T$ (ZJ)",
   note=("Green predictive curves are run to equilibrium ("
         f"{OHC_TS_EQUILIBRIUM_TAUS:.0f}$\\tau_s$), so they extend past the "
         r"AOGCM's own temperature range and end at $T_{eq}$. The AOGCM $H_T$ "
         r"is negative at small $T$ because $H_M$ carries its full $U_0$ from "
         r"$t=0$ (the strip plot_ohu shades)."),
   draw=lambda ax, model, bundle, params: plot_HT_vs_T(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model],
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "HT_vs_T2M"))
plt.close(fig)


fig, _, _ = model_figure(
   title=(r"MT2: 4xCO2 heat-uptake flux $H=\gamma(T-T_d)$ vs T$_{2M}$ by AMOC "
          f"strength (M), {YEARS_LABEL} ({SMOOTH_WINDOW_YEARS}-year running "
          f"mean, {FIT_LABEL})"),
   xlabel="2-meter Air Temperature Anomaly (K)",
   ylabel=r"$H$ ($W\,m^{-2}$)",
   note=(r"The AOGCM points are inverted from $N=F-\lambda T-(\epsilon-1)H$ "
         r"with the STEP-1 $F$, $\lambda$, $\epsilon$ -- a radiative "
         "reconstruction, not an ocean diagnostic. Green predictive curves run "
         f"to equilibrium ({OHC_TS_EQUILIBRIUM_TAUS:.0f}$\\tau_s$)."),
   draw=lambda ax, model, bundle, params: plot_H_flux_vs_T(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model],
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "H_flux_vs_T2M"))
plt.close(fig)


# ------------------- OHUE kappa = kappa_M + q (Eq. 10) -------------------

fig, _, _ = model_figure(
   title=(f"MT2: 4xCO2 OHUE $\\kappa=N/T=\\kappa_M+q$ by AMOC strength (M), "
          f"{YEARS_LABEL} ({SMOOTH_WINDOW_YEARS}-year running mean, {FIT_LABEL})"),
   ylabel=r"OHUE ($W\,m^{-2}\,K^{-1}$)",
   note=(r"Under constant forcing $\kappa$ runs from $\infty$ to 0, so these curves are "
         "NOT comparable to Table 1's OHUE column (year-70 2xCO2 state of 1pctCO2). "
         r"$\kappa$ here is the paper's $\kappa_N$, ~3% above $\kappa_O$ (Appendix A.4); "
         rf"$|T|<{OHUE_T_FLOOR_K}$ K is masked."),
   draw=lambda ax, model, bundle, params: plot_ohue(
      ax, model, bundle, params,
      two_layer_by_model[model], two_layer_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "OHUE_vs_Year"))
plt.close(fig)


# --------------- MT2T fit diagnostic: the appendix's Fig 17a ---------------
# The faded overlay and the caption explaining it are decided together here,
# rather than the panel sniffing the params dict for `_val` keys.

fig, _, _ = model_figure(
   title=(f"MT2T: two-layer fit of $H_T(t)$ per AOGCM, {YEARS_LABEL} "
          r"($H_T = C_u T + \gamma(1+C_u/C_d)\int T\,dt - (\gamma/C_d)\int H_T\,dt$)"),
   ylabel="ZJ",
   note=(f"Faded curves: validation fit (first {OHU_RATE_YEARS} years) "
         f"extrapolated over the full run" if UNBLINDED_RUN else None),
   draw=lambda ax, model, bundle, params: plot_HT_decomposition(
      ax, model, bundle, params,
      val_params=val_params_by_model.get(model)),
)
layout.save_results(fig, figname("4xCO2", "all_models", "HT_two_layer_fit"))
plt.close(fig)


# ------------------- OHU vs. Ts (the shared "ohc_ts" panel) -------------------
# Unblinded only, like the same figure in every other script: the whole point is
# where the AOGCM's trajectory ends up after thousands of years, which the
# blinded pass must not show.
if UNBLINDED_RUN:
   fig, axs, handles = model_figure(
      title=r"MT2: 4xCO$_2$ OHU vs. Surface Temp (Normalized)",
      xlabel=r"$T_s/T_{eq}$",
      ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$",
      # Two lines of caption, so this grid needs a deeper top margin.
      top=0.83,
      note=(r"$T_{eq}=F_{eff}/\alpha$ is MT2's own equilibrium, set by $(1-p)F$ "
            r"-- not $2\times$ECS as in the other scripts' version of this figure."
            "\n"
            r"$\mathrm{OHC}_{eq}$ is the whole ocean at $T_{eq}$, which a two-layer "
            r"model cannot reach. The faded $H$ leaves the panel because "
            r"$H_M=U_0+N_M t$ never stops growing (Sect. 6.3)."),
      draw=lambda ax, model, bundle, params: plot_ohc_ts(
         ax, model, bundle, params, analytic_by_model[model]),
   )
   cbar = fig.colorbar(handles[-1], ax=axs.tolist(), fraction=0.025, pad=0.025)
   cbar.set_label("Year")
   layout.save_results(fig, figname("4xCO2", "all_models", "ohc_ts"))
   plt.close(fig)


# ------------- tau_d vs. calibration window (2013a/b's tau_s plot) -------------
# Needs the long record to say anything -- only meaningful, and only produced,
# on the unblinded pass, same as 2013a/2013b's equivalent (run_type == 3 there).
if UNBLINDED_RUN:
   def _draw_tau_d(ax, model, bundle, params):
      windows, tau_slow, tau_deep = tau_d_vs_window(model, bundle, params)
      plot_tau_d_vs_window(ax, model, params, windows, tau_slow, tau_deep)

   fig, _, _ = model_figure(
         title=(r"MT2T: $\tau_d$ vs. Calibration Window by AMOC strength (M) "
             r"($F_{eff}$, $\alpha$, $\epsilon$ fixed at their whole-record "
             "STEP-1 values)"),
      xlabel="Calibration Window (years)",
      ylabel=r"$\tau_d$ (years)",
      draw=_draw_tau_d,
   )
   layout.save_results(fig, figname("4xCO2", "all_models", "tau_d_vs_calibration_window"))
   plt.close(fig)
