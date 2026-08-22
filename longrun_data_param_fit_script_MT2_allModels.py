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
excluded from `models` rather than given a placeholder M.

A FIFTH model is fit alongside those 4: `ECHAM5_li2012`, the ECHAM5/MPIOM
abrupt-4xCO2 run of Li, von Storch and Marotzke (2012), Clim Dyn 40:1071-1086,
at a supplied M = 16 Sv. Its N and T come from that paper's Fig. 1c and 1a
rather than from the .Rdata bundle, digitized by extract_li2012_fig1.py into
data/li2012_fig1_N_vs_T.csv. It is here because it runs 6080 years and actually
reaches equilibrium, which is exactly the regime where MT2's N_M = pF is
supposed to fail (Sect. 6.3) -- and it duly does, H_T turning negative again
around year 2600. Nothing else in this project uses that record; the LongRunMIP
`ECHAM5` the other fit scripts use is a different, 1001-year run.

FOUR DELIBERATE DEPARTURES FROM THE PAPER, all of them visible in the outputs:

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
  4. ECHAM5_li2012 is not an abrupt step: its CO2 RAMPS linearly to 4x over the
     first 140 years and is constant afterwards. Every regression in this script
     assumes constant forcing, so for that model they are all windowed past the
     ramp, and no panel draws anything over it -- see SeriesBundle.ramp_years
     and .display_start. The integrals are NOT windowed: H = int N dt stays
     anchored at t = 0, because the heat the ramp put into the ocean is real and
     H_T = H - H_M would lose it. H_M correspondingly uses Eq. 7's p int F dt
     with F ramping, not the constant-F collapse N_M*t.

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
  ./data/li2012_fig1_N_vs_T.csv (optional; `python3 extract_li2012_fig1.py` to
    regenerate it from the article PDF). Absent, ECHAM5_li2012 is skipped and
    the other four models run as before.

Outputs (every name ends with the variant, `_MT2`, so these stay identifiable
next to the same plots from the other model scripts -- see figure_paths.py):
  ./figures/MT2/01_replicate/tables/fitted_model_params_MT2.csv
  ./figures/MT2/01_replicate/tables/<model>_two_layer_Nt_MT2.csv
  ./figures/MT2/01_replicate/validation/...   (default)
  ./figures/MT2/01_replicate/unblinded/...    (--unblinded)
    4xCO2_all_models_OHU_rate_vs_Year_MT2.png       (N,   Fig 10b)
    4xCO2_all_models_OHU_vs_Year_MT2.png            (H,   Fig 11a)
    4xCO2_all_models_HT_vs_T2M_MT2.png              (H_T [ZJ] against T)
    4xCO2_all_models_H_flux_vs_T2M_MT2.png          (H = gamma(T-T_d) against T)
    4xCO2_all_models_OHUE_vs_Year_MT2.png           (kappa)
    4xCO2_all_models_Temperature_vs_Year_MT2.png
    4xCO2_all_models_HT_two_layer_fit_MT2.png       (Fig 17a)
  --unblinded only:
    4xCO2_all_models_T2M_vs_NETTOA_MT2.png          (STEP-1 radiative fit)
    4xCO2_all_models_ohc_ts_MT2.png                 (the other scripts' OHU vs Ts)
    4xCO2_all_models_ohc_ts_M_sensitivity_MT2.png   (H_M on those axes, swept over M)
    4xCO2_all_models_tau_d_vs_calibration_window_MT2.png
"""
import argparse
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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

# --- The Li et al. (2012) ECHAM5/MPIOM run -----------------------------------
# Read from extract_li2012_fig1.py's digitization of that paper's Fig. 1a (dT)
# and 1c (N) rather than from the .Rdata bundle -- see the module docstring for
# why this record and not the LongRunMIP ECHAM5.
LI2012_MODEL = "ECHAM5_li2012"
LI2012_N_VS_T_CSV = (Path(__file__).resolve().parent / "data" /
                     "li2012_fig1_N_vs_T.csv")

# Length of that run's CO2 ramp. The paper's own "CO2 increase (1-140 yr)"
# adjustment period, as extract_li2012_fig1.PERIOD_YEARS names it; forcing is
# constant from year 140 to the end of the run at 6080.
LI2012_RAMP_YEARS = 140.0

# AMOC strength M (Sv) for whichever of ALL_LONGRUN_MODELS appear in the
# AMOC-strength table. IPSLCM5A -> IPSL-CM5A-LR (not -MR) and MPIESM11 ->
# MPI-ESM-LR (not -MR/-P) per the table's naming.
M_SV: Dict[str, float] = {
   "CNRMCM6": 17.0,    # CNRM-CM6-1
   "HadGEM2": 14.8,    # HadGEM2-ES
   "IPSLCM5A": 11.5,   # IPSL-CM5A-LR
   "MPIESM11": 23.5,   # MPI-ESM-LR
   # SUPPLIED, not from the paper's Table 1 -- ECHAM5/MPIOM is not in it. The
   # other four entries above are the table's own values.
   LI2012_MODEL: 16.0,
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

# Models this script actually runs: the ALL_LONGRUN_MODELS with a resolved M,
# plus the digitized Li (2012) record when its CSV is on disk. That CSV is
# derived data rather than a repo input, so a missing one is a skip with a note
# and not an error -- the other four models have nothing to do with it.
models = [m for m in ALL_LONGRUN_MODELS if m in M_SV]
if LI2012_N_VS_T_CSV.exists():
   models = models + [LI2012_MODEL]
else:
   print(f"*** NOTE: {LI2012_N_VS_T_CSV.name} not found -- skipping "
         f"{LI2012_MODEL}. Run `python3 extract_li2012_fig1.py` to build it. ***")
fit_models = _apply_test_subset(models)
if _TEST_SPEC is not None:
   print(f"*** TEST MODE: running {len(fit_models)}/{len(models)} models: {fit_models} ***")

PARAM_COLS = ["model", "M"]

# Extra top/bottom margin beyond make_model_grid's defaults. Those defaults are
# sized for the other scripts' 4x2 grids; with this script's handful of models
# the grid is a SINGLE row (model_figure pins ncols to the model count), so the
# figure is half as tall and the same fractional margins leave half the room --
# the suptitle lands on the panels and the figure-level x-label lands on the
# bottom tick labels. Applied to every grid here.
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

# End of the MT2T calibration window: the first year at which an
# EQUILIBRATION_AVG_YEARS running mean of T reaches EQUILIBRATION_FRACTION of
# T_eq. Past that point the record stops constraining the ocean parameters and
# starts actively misinforming them -- T has effectively stopped moving while
# H_M = U_0 + N_M t keeps growing linearly, which is MT2 breaking down rather
# than anything about c_u, c_d and gamma ("N_M cannot remain non-zero and
# unchanged as equilibrium is approached under constant forcing", Sect. 6.3;
# two_layer_fit()'s H_T_late_neg_year measures where that takes over).
#
# The running mean is what makes the threshold usable at all: annual T wanders
# by several tenths of a K, so the raw series would cross 99% of T_eq decades
# before the run has actually settled there. Only the CALIBRATION window moves
# -- every panel still shows the whole record, and the predictive curves still
# run to equilibrium. See equilibration_year().
EQUILIBRATION_FRACTION = 0.99
EQUILIBRATION_AVG_YEARS = 50


# ------------------------- Data structures -------------------------

@dataclass
class SeriesBundle:
   """One model's baselined 4xCO2 series, plus the window every panel shares."""

   t_full: np.ndarray
   T_full: np.ndarray
   N_full: np.ndarray

   #: Length of this run's CO2 ramp, 0 for an abrupt step. Non-zero only for
   #: ECHAM5_li2012 (LI2012_RAMP_YEARS). MT2's regressions all assume constant
   #: forcing, so this is the year every one of them starts at, and the year
   #: every panel starts at -- but NOT a truncation of the record: window()
   #: still returns the ramp years, because the cumulative integrals built on
   #: top of it have to stay anchored at t = 0.
   ramp_years: float = 0.0

   @property
   def panel_years(self):
      """Last year the panels show: this model's own run length on the
      unblinded pass (PANEL_YEARS is None there), otherwise OHU_RATE_YEARS of
      CONSTANT-FORCING record -- measured from the end of the ramp, so a ramped
      run gets as many usable years as an abrupt one rather than spending its
      whole blinded window on the ramp."""
      return (float(self.t_full[-1]) if PANEL_YEARS is None
              else self.ramp_years + PANEL_YEARS)

   @property
   def display_start(self):
      """First year any panel draws. Everything before it is the forcing ramp,
      which is not what MT2 describes: drawing H_M or H_T there would show
      curves built on a constant-F assumption over years where F was still
      rising. The values themselves are unaffected -- a curve starting here
      still carries the heat the ramp put in -- only the extent is trimmed."""
      return self.ramp_years

   def display(self, t, *arrays):
      """`t` and `arrays` masked to the years the panels draw.

      The single implementation of the "nothing over the ramp" rule, so no
      panel can forget it and the abrupt models (display_start = 0) pass
      through untouched. Compute over window(), then plot through this.
      """
      mask = np.asarray(t, dtype=float) >= self.display_start
      return (np.asarray(t, dtype=float)[mask],
              *(np.asarray(a, dtype=float)[mask] for a in arrays))

   def window(self, n_years=None):
      """(last_year, t, T, N) truncated to `n_years`, or to the panel window.

      The one place the blinded/unblinded truncation is applied, so a panel
      and the fit behind it cannot disagree about which years they cover.

      Truncated only at the RIGHT: the ramp years are still in here, and
      display() is what trims them off for plotting. Everything cumulative
      (cumint, the design matrix of two_layer_fit) depends on that.

      `n_years` is clamped to the panel window rather than trusted. The MT2T
      calibration window comes from equilibration_year(), which reads the WHOLE
      record -- so on the blinded pass an unclamped request could pull years
      past OHU_RATE_YEARS into a validation figure. Clamping here makes the
      blinding structural instead of a property of where the crossings happen
      to fall (the earliest is year 223, so nothing is clamped in practice).
      """
      n_years = (self.panel_years if n_years is None
                 else min(float(n_years), self.panel_years))
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


def make_li2012_bundle(csv_path=LI2012_N_VS_T_CSV):
   """The Li et al. (2012) ECHAM5/MPIOM series, from extract_li2012_fig1.py's
   digitization of that paper's Fig. 1a and 1c.

   Three things this record is not, each of which would otherwise be handled
   somewhere upstream:

     * it is not raw model output but a digitized figure, so it starts at the
       first year of the paper's own 11-year running mean (year 6) and ends at
       6068 rather than 6080;
     * dT and N are ALREADY anomalies against that run's control, so
       SUBTRACT_PICONTROL_BASELINE has nothing to do here -- there is no
       piControl series to subtract, and subtracting anything would double-count;
     * it is ALREADY an 11-year running mean, because that is the curve the
       paper plots in colour and the gray annual spaghetti behind it is not
       recoverable from the vector art. SMOOTH_WINDOW_YEARS smoothing on top of
       it is therefore mild double-smoothing rather than the first smoothing
       these panels apply. Harmless at this scale -- the panels exist to show
       decadal-and-longer structure -- but it is why this model's curves look
       cleaner than the other four's, and why its "AOGCM N (annual)" trace is
       not actually annual.

   The 140-year CO2 ramp is carried as SeriesBundle.ramp_years, not trimmed
   here; see that field's comment.
   """
   raw = pd.read_csv(csv_path)
   t_full = as_1d_float(raw["year"])
   T_full = as_1d_float(raw["dT_K"])
   N_full = as_1d_float(raw["N_W_m2"])

   mask = finite_pair_mask(t_full, T_full, N_full)
   return SeriesBundle(t_full=t_full[mask], T_full=T_full[mask],
                       N_full=N_full[mask], ramp_years=LI2012_RAMP_YEARS)


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


def equilibration_year(t, T, T_eq, fraction=EQUILIBRATION_FRACTION,
                       window=EQUILIBRATION_AVG_YEARS):
   """First year whose `window`-year running mean of T reaches `fraction`*T_eq.

   Returns None when the record never gets there, which is the common case
   rather than a corner: against the STEP-1 T_eq, the four models here peak at

      CNRMCM6   0.987 of T_eq over 1850 yr   -> None
      IPSLCM5A  0.948 of T_eq over 1000 yr   -> None
      HadGEM2   1.037 of T_eq over 1299 yr   -> year 844
      MPIESM11  1.312 of T_eq over 4459 yr   -> year 223

   so two of them calibrate on their whole record and the None branch is the
   one most models take. MPIESM11 stops as early as it does because its binned
   T_eq (5.28 K) under-predicts the warming the run actually reaches, not
   because the run equilibrates in two centuries -- the T_eq_frac_max column is
   there to make that visible rather than let the short window pass unexplained.

   running_mean() is the same centred smoother the OHU-rate panels use. Its
   min_periods=1 shortens the window within half a span of either end, which
   would matter for a model that only crossed in its final decades; none of the
   crossings above is anywhere near an end, so the full `window` years are
   averaged at every crossing measured here.
   """
   t = np.asarray(t, dtype=float)
   hit = np.flatnonzero(running_mean(T, window) >= fraction * T_eq)
   return float(t[hit[0]]) if hit.size else None


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

def H_M_per_area(t, N_m, ramp_years=0.0):
   """H_M = U_0 + p*int F dt (Eq. 7).

   F = F_4x is constant in abrupt4xCO2, so p*int F dt collapses to N_m*t.

   `ramp_years` covers the one run here that is not an abrupt step: with F
   rising linearly 0 -> F_ref over the first `ramp_years` years and constant
   after, p*int F dt is N_m*t^2/(2*ramp) during the ramp and N_m*(t - ramp/2)
   once it ends -- i.e. the same straight line as the abrupt case, offset back
   by half the ramp, which is the forcing the ramp did not deliver. Keeping the
   abrupt form for a ramped run would overstate H_M by N_m*ramp/2 at every
   later time (about 1000 ZJ for ECHAM5_li2012), and H_T = H - H_M would carry
   the whole error.

   Eq. 7 is written as p*int F dt precisely because F need not be constant;
   this is that integral, not an extension of the paper.
   """
   t = np.asarray(t, dtype=float)
   int_F_over_F = (t if ramp_years <= 0.0 else
                   np.where(t <= ramp_years, t ** 2 / (2.0 * ramp_years),
                            t - ramp_years / 2.0))
   return U_0_ZJ / ZJ_PER_WYRM2 + N_m * int_F_over_F


def aogcm_H_T(t, N, N_m, scale=1.0, ramp_years=0.0):
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

   `ramp_years` is passed straight through to H_M_per_area; the cumint is
   unaffected, since the AOGCM's N already knows what its own forcing did.
   """
   t = np.asarray(t, dtype=float)
   return scale * cumint(t, N) - H_M_per_area(t, N_m, ramp_years)


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
#                            ^ both windows shift past a ramped run's forcing
#                              ramp -- see the gregory mask and bin_skip_for()
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
#
# This function is also where the MT2T CALIBRATION window is decided, since
# T_eq is what defines it (equilibration_year()). The window is derived here
# and applied nowhere but fit_model()'s two_layer_fit() calls: nothing in this
# function is itself windowed, and no panel is.

def bin_by_position(y, width=BIN_YEARS, skip=BIN_SKIP):
   """Non-overlapping `width`-sample block means, dropping the first `skip`.

   By position, not by year: make_series_bundle() has already dropped any
   non-finite years, so the two coincide unless a model has gaps. A trailing
   partial block is kept, as in the diffusive script.
   """
   y = np.asarray(y, dtype=float)
   return np.array([y[i:i + width].mean()
                    for i in range(skip * width, y.shape[0], width)])


def bin_skip_for(ramp_years, width=BIN_YEARS, skip=BIN_SKIP):
   """How many leading bins the N_T-vs-T fit drops for a run with this ramp.

   BIN_SKIP alone for an abrupt step. For a ramped one, enough whole bins to
   clear the ramp FIRST -- those years are not a constant-forcing climate at
   all, so a bin straddling the ramp's end would mix the two -- and then
   BIN_SKIP more, for the same fast-mode reason as everywhere else. ECHAM5's
   140-year ramp therefore drops 3 + 1 = 4 bins, i.e. years 0-200.
   """
   return skip + int(np.ceil(max(ramp_years, 0.0) / width))


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
   eq_year: Optional[float]   # last year of the MT2T calibration window
   T_eq_frac_max: float       # highest the smoothed T ever gets, as a fraction
   bin_skip: int              # leading bins dropped (> BIN_SKIP for a ramped run)
   gregory_lo: float          # first/last year of the F_ref regression window
   gregory_hi: float

   def as_params(self):
      return {"F_ref": self.F_ref, "F_eff": self.F_eff, "T_eq_fit": self.T_eq,
              "alpha": self.alpha, "eps": self.eps,
              "F_ref_unc": self.F_ref_unc, "T_eq_fit_unc": self.T_eq_unc,
              "alpha_unc": self.alpha_unc, "eps_unc": self.eps_unc,
              "alpha_g20": -self.m20, "bin_slope": self.mb,
              "bin_intercept": self.bb, "n_bins": float(self.T_bin.size),
              "bin_skip": float(self.bin_skip),
              "gregory_lo": self.gregory_lo, "gregory_hi": self.gregory_hi,
              "eq_year": np.nan if self.eq_year is None else self.eq_year,
              "T_eq_frac_max": self.T_eq_frac_max}


_RADIATIVE_CACHE: Dict[str, RadiativeFit] = {}


def fit_radiative(model, bundle, p):
   """Read F_ref, T_eq, alpha and eps off the two regressions described above,
   for this model's OWN AMOC strength.

   `p` is the MT2 AMOC term s_0(M - M_0), needed because N_M = p*F_ref has to
   be subtracted before the binned fit and because F_eff = (1-p)F_ref sets the
   scale of alpha and eps.

   Cached per model. It depends only on the full record and on M, neither of
   which changes within a run, and fit_model() and the N_T-vs-T panel both want
   the same answer -- including the two regression lines, which only the panel
   uses. The M-sensitivity sweep is the one caller that varies p within a run,
   so it goes through refit_radiative() below and never touches this cache.
   """
   if model not in _RADIATIVE_CACHE:
      _RADIATIVE_CACHE[model] = refit_radiative(model, bundle, p)
   return _RADIATIVE_CACHE[model]


def refit_radiative(model, bundle, p, verbose=True):
   """The uncached fit. Separate from fit_radiative() so the M-sensitivity
   sweep can re-derive the whole chain at a counterfactual p without evicting
   or corrupting the per-model cache; `verbose=False` silences the per-model
   notes it would otherwise reprint once per swept M.
   """
   T, N = bundle.T_full, bundle.N_full
   ramp = bundle.ramp_years

   # The Gregory window starts where the forcing becomes constant, not at year
   # 1: on a ramped run, years 1-20 sit at a fifth of the eventual ERF and
   # their intercept is not F_ref at all. Identical to `t <= 20` when ramp = 0.
   gregory = (bundle.t_full > ramp) & (bundle.t_full <= ramp + GREGORY_FIT_YEARS)
   (m20, b20), cov20 = np.polyfit(T[gregory], N[gregory], 1, cov=True)
   b20_unc = float(np.sqrt(max(cov20[1, 1], 0.0)))

   F_ref = max(float(b20), F_REF_FLOOR)
   F_eff = effective_forcing(p, F_ref)
   F_eff_unc = abs(1.0 - p) * b20_unc

   skip = bin_skip_for(ramp)
   T_bin = bin_by_position(T, skip=skip)
   NT_bin = bin_by_position(N - p * F_ref, skip=skip)
   if T_bin.size < 3:
      raise RuntimeError(
         f"{model}: only {T_bin.size} bins of {BIN_YEARS} years after dropping "
         f"the first {skip}; too few for the N_T-vs-T fit")
   (mb, bb), covb = np.polyfit(T_bin, NT_bin, 1, cov=True)
   mb_unc = float(np.sqrt(max(covb[0, 0], 0.0)))
   bb_unc = float(np.sqrt(max(covb[1, 1], 0.0)))

   T_eq_raw = float(-bb / mb) if mb != 0.0 else np.nan
   T_eq = T_eq_raw if T_eq_raw > 0.0 else 1.0       # guard a degenerate x-int
   alpha = max(F_eff / T_eq, ALPHA_FLOOR)
   eps = max(F_eff / bb, EPS_FLOOR) if bb != 0.0 else EPS_FLOOR
   if verbose and (T_eq_raw != T_eq or alpha == ALPHA_FLOOR or eps == EPS_FLOOR):
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

   # Where the MT2T calibration stops, and the diagnostic that says why. Both
   # read the whole record, like everything else here; the window they imply is
   # applied only by fit_model(), and clamped to the panel window by
   # SeriesBundle.window() so the validation pass stays blind.
   eq_year = equilibration_year(bundle.t_full, T, T_eq)
   T_eq_frac_max = float(
      np.nanmax(running_mean(T, EQUILIBRATION_AVG_YEARS)) / T_eq)
   if verbose:
      if eq_year is None:
         print(f"   NOTE {model}: {EQUILIBRATION_AVG_YEARS}-yr mean T peaks at "
               f"{T_eq_frac_max:.3f} of T_eq={T_eq:.2f} K, short of "
               f"{EQUILIBRATION_FRACTION:.2f} -- MT2T calibrates on the whole run")
      else:
         # The year the fit actually stops at, which is not eq_year on the
         # blinded pass -- SeriesBundle.window() clamps it back to the panel
         # window there.
         applied = min(eq_year, bundle.panel_years)
         print(f"   NOTE {model}: {EQUILIBRATION_AVG_YEARS}-yr mean T reaches "
               f"{EQUILIBRATION_FRACTION:.2f} of T_eq={T_eq:.2f} K at year "
               f"{eq_year:.0f} of {bundle.t_full[-1]:.0f} (peaks at "
               f"{T_eq_frac_max:.3f}) -- MT2T calibrates to year {applied:.0f}")

   return RadiativeFit(
      F_ref=F_ref, F_eff=F_eff, T_eq=T_eq, alpha=alpha, eps=eps,
      F_ref_unc=b20_unc, T_eq_unc=T_eq_unc, alpha_unc=alpha_unc, eps_unc=eps_unc,
      m20=float(m20), b20=float(b20), mb=float(mb), bb=float(bb),
      T_bin=T_bin, NT_bin=NT_bin, eq_year=eq_year, T_eq_frac_max=T_eq_frac_max,
      bin_skip=skip, gregory_lo=ramp, gregory_hi=ramp + GREGORY_FIT_YEARS)


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

   def N_t(self, t):
      """The AMOC-free part N_T = N - N_M, i.e. N() with F_eff in place of F.

      N_M = p*F is the constant F_total - F_eff, so subtracting it just swaps
      the forcing: N_T = F_eff - lam T - (eps-1) H. This is the branch that
      actually decays to zero as T -> T_eq, which N does not under MT2, and it
      is what the two-layer solve's own `N_t` is comparable to.
      """
      return self.F_eff - self.lam * self.T(t) - (self.eps - 1.0) * self.H(t)

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

   fit_model() passes the equilibration year here, so in the ordinary case this
   window is SHORTER than the one the panels show and the fitted parameters are
   extrapolated over the rest of the record. The two other callers pass their
   own: the "_val" fit uses OHU_RATE_YEARS, and tau_d_vs_window() sweeps.

   H_T = int_0^t N_T dt' = C_u T + C_d T_d, which rearranges to
      H_T = C_u T + gamma(1 + C_u/C_d) int T dt' - (gamma/C_d) int H_T dt'
   (the appendix prints a2 = gamma + C_u/C_d, a typo -- only the form above
   inverts to the appendix's own C46). Regressing H_T on the three terms
   with no intercept, i.e. through the origin, gives a1, a2, a3 below.

   `verbose=False` silences the per-model H_T diagnostics; the calibration-window
   sweep refits this hundreds of times per model and only wants the numbers.
   """
   _, t, T, N = bundle.window(window_years)
   H_T = aogcm_H_T(t, N, N_m, scale=scale, ramp_years=bundle.ramp_years)

   # Rows before this are excluded: the paper's early-year U_0 artefact for an
   # abrupt run, and additionally the whole forcing ramp for a ramped one,
   # since MT2's H_T relation is a constant-forcing statement.
   calib_skip = max(CALIB_SKIP_YEARS, bundle.ramp_years)

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
   fit_rows = t > calib_skip
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
      if not (first_pos_year <= calib_skip):
         print(f"   WARNING {model}: H_T first turns positive at year "
               f"{first_pos_year:.0f}, past the calibration start "
               f"{calib_skip:.0f} -- early rows are still in the fit")
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

   # The plain Gregory regression over the whole PANEL window -- the display
   # extent, deliberately not rad.eq_year -- recorded for reference only. It
   # describes the span the figures show, which is what makes it comparable
   # between models; the calibration window below is per-model and would not
   # be. alpha declines with time in every AOGCM, so this is systematically
   # smaller than the years-1-20 value; the gap between the alpha_g20 and
   # alpha_full columns is how much the apparent feedback drifted over the
   # window, and `alpha` itself -- the deterministic F_eff/T_eq that the
   # predictive curves use -- should sit between them.
   _, _, T_win, N_win = bundle.window()
   slope_full, F_ref_full = np.polyfit(T_win, N_win, 1)

   params = {"p": p, "N_m": N_m, "ramp_years": bundle.ramp_years,
             "F_ref_full": F_ref_full, "alpha_full": -slope_full}
   params.update(rad.as_params())

   # Both MT2T fits stop at rad.eq_year, the year the run has effectively
   # equilibrated; None -- no crossing -- falls through to the panel window, so
   # a model that never gets there still calibrates on everything it has. The
   # resulting extent is in the table as fit_years, and on the panels as the
   # marker mark_calibration_end() draws.
   params.update(two_layer_fit(model, bundle, N_m, rad.eq_year))
   # Same fit against H_O = H_N/1.07 (Appendix A.3), so the table shows how much
   # of C_u/C_d/gamma rides on this dataset's H coming from NETTOA rather than
   # ocean temperature. Not a uniform 1/1.07 rescaling: H_M is left unscaled.
   params.update(two_layer_fit(model, bundle, N_m, rad.eq_year, suffix="_HO",
                               scale=1.0 / H_N_OVER_H_O))
   if UNBLINDED_RUN:
      # Also carry the blinded-window fit, so the unblinded H_T figure can show
      # what the validation pass's parameters extrapolate to over the full run.
      # Same window the validation pass would use, ramp offset included, so the
      # faded curves really are that pass's parameters.
      params.update(two_layer_fit(model, bundle, N_m,
                                  bundle.ramp_years + OHU_RATE_YEARS,
                                  suffix="_val"))
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

   def between(self, first_year, n_years):
      """Boolean mask for the samples a panel draws: `first_year` to `n_years`.

      The upto() the plotting functions use, with SeriesBundle.display_start
      as the lower bound so a ramped run's curves start where its forcing
      became constant. Identical to upto() when first_year is 0.
      """
      return (self.t >= first_year) & (self.t <= n_years)

   def display_mask(self, first_year, n_years):
      """(window_mask, keep) for a panel that SMOOTHS before it draws.

      `window_mask` selects everything the panel window covers; `keep` then
      selects the part actually drawn, indexed into the already-masked arrays.
      Two stages rather than one so the running mean is taken over the whole
      window and only then trimmed -- a panel starting partway in still gets a
      properly centred average at its left edge instead of a one-sided one.
      Both collapse to upto() when first_year is 0.
      """
      window_mask = self.upto(n_years)
      return window_mask, self.between(first_year, n_years)[window_mask]

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

   Plotted over the model's whole CONSTANT-FORCING record, since fit_radiative()
   always is; the caller only produces this figure on the unblinded pass. For
   an abrupt run that is the whole record.
   """
   rad = fit_radiative(model, bundle, params["p"])
   N_m = params["N_m"]

   # The ramp years are dropped from the panel as well as from the fits. During
   # a ramp N RISES with T, because the forcing is still growing, so those
   # points trace a branch running the wrong way against every line drawn here
   # -- a statement about the ramp, not about the feedback (the same argument
   # extract_li2012_fig1.PERIOD_YEARS makes for not regressing over it).
   _, T_d_full = two_layer.at_data()      # drops the solve's prepended t=0 sample
   t, T, N, T_d = bundle.display(bundle.t_full, bundle.T_full, bundle.N_full,
                                 T_d_full)
   order = np.argsort(T)
   Ts = T[order]

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

   # Named by the years it actually covers -- "yrs 1-20" for an abrupt run,
   # "yrs 140-160" for one whose forcing only became constant at year 140.
   gregory = (t > rad.gregory_lo) & (t <= rad.gregory_hi)
   ax.scatter(T[gregory], N[gregory], s=8, alpha=0.5, color="0.5",
              label=f"yrs {rad.gregory_lo + 1:.0f}-{rad.gregory_hi:.0f} (annual)")
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

   # Equilibrium under MT2 is N = N_M, not N = 0: it is N_T that vanishes while
   # N_M = p*F_ref persists. Both fitted lines above are drawn in total-N space
   # (the binned one with the N_m shift undone), so both pass through
   # (T_eq, N_m) rather than crossing zero at T_eq. Without this line the panel
   # reads as though T_eq disagreed with the fit it came from.
   ax.axhline(N_m, color="black", lw=1.5, ls=":", label=r"$N_M$")

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
   plus any extra lines that panel wants underneath.

   A ramped run says so, because otherwise its panels would silently start at a
   different year (and a different T) from the abrupt ones beside them with
   nothing on the figure to explain why.
   """
   ramp = float(params.get("ramp_years", 0.0) or 0.0)
   head = (model, f"M={params['M']:.1f} Sv")
   if ramp > 0.0:
      head += (f"ramp 0-{ramp:.0f} yr (not shown)",)
   return "\n".join(head + extra)


def mark_calibration_end(ax, params, last_year):
   """Vertical marker at the last year the MT2T calibration used.

   Everything to the RIGHT of this line is extrapolation: the panel still shows
   the whole run, but the parameters behind its model curves never saw those
   years. Only for panels whose x-axis is the year; the T-axis panels get the
   same information from plot_ohc_ts's line at T/T_eq = 1.

   `last_year` is the panel's own right-hand edge, and nothing is drawn unless
   the marker falls strictly inside it. That covers both cases where there is
   no boundary to draw and marking one would say the opposite of what is true:
   a model that never reaches EQUILIBRATION_FRACTION of T_eq (eq_year is NaN,
   fit over everything it has), and the blinded pass, where the panels stop at
   OHU_RATE_YEARS and SeriesBundle.window() has already clamped the fit to the
   same place -- so the whole-record eq_year in the params dict is NOT where
   this pass's fit ended, and drawing it would both misreport the fit and put a
   late-run year on a blinded figure.
   """
   eq_year = params.get("eq_year", np.nan)
   if not np.isfinite(eq_year) or eq_year >= last_year:
      return
   ax.axvline(eq_year, color="0.35", ls=(0, (6, 3)), lw=1.2,
              label=rf"MT2T fit ends (yr {eq_year:.0f})")


def plot_ohu(ax, model, bundle, params, two_layer, two_layer_fixed,
             analytic=None, analytic_fixed=None):
   """Accumulated OHU H = H_M + H_T (Eq. 7) in ZJ against the AOGCM's own H,
   in the shape of the paper's Fig 11a.

   H_M = U_0 + p*int F dt is a straight line here, F being constant in
   abrupt4xCO2 (for a ramped run it is that same line, offset back by half the
   ramp -- and its curved ramp section is off-panel anyway); the model's H_T is
   C_u T + C_d T_d, the analytic integral of the two-layer solve (Eq. C41), so
   nothing is integrated twice.

   The AOGCM curve is the NETTOA integral, i.e. the paper's H_N. The paper's own
   H is the ocean-temperature H_O = H_N/1.07 (Appendix A.3), and this dataset
   has no 3-D ocean temperature to settle which end is right; the H_O-to-H_N
   band that used to be shaded here is dropped, since a 7% spread on the one
   curve everything else is judged against read as an uncertainty on the data
   rather than the convention offset it is. H_O still appears in
   fitted_model_params.csv, both as `H_aogcm_O_end` and through the whole `_HO`
   fit, so the sensitivity is quantified where it can be read off as numbers.

   The first few years of H_M are wrong by construction -- U_0 is applied in
   full from t = 0 although H_M(0) must be 0 -- which is the shaded strip. A
   ramped run's panel starts past those years, so it has no strip.

   `analytic`/`analytic_fixed` add the PREDICTIVE curves (green/purple), the H
   counterpart of plot_ohu_rate's green "N_T 2-Box predictive": T(t) and T_d(t) come
   from the fitted parameters alone, run forward with no AOGCM timeseries at
   all, unlike the orange/blue DIAGNOSTIC curves above, whose T_d is ODE-solved
   from the AOGCM's own observed T(t). H_M is unchanged between the two --
   it never depended on T -- so only H_T and H = H_M + H_T move.
   """
   n_years, t, _, N = bundle.window()
   start = bundle.display_start

   # Integrated over the full window, drawn from `start`: on a ramped run these
   # curves begin partway up, carrying the heat the ramp actually delivered,
   # rather than restarting from zero at the panel's left edge.
   H_N = cumint(t, N) * ZJ_PER_WYRM2
   ax.plot(*bundle.display(t, H_N), color="black", lw=1.5, label=r"AOGCM $H_N$")
   ax.plot(*bundle.display(t, H_M_per_area(t, params["N_m"], bundle.ramp_years)
                           * ZJ_PER_WYRM2),
           color="black", lw=1.5, ls=":", label=r"$H_M$")
   if start == 0.0:
      # The strip marks the years where H_M carries its full U_0 although
      # H_M(0) = 0. A ramped run's panel starts past them, so there is nothing
      # left to shade.
      ax.axvspan(0, CALIB_SKIP_YEARS, color="0.5", alpha=0.15, lw=0)

   # Each solve carries its own (c_u, c_d), so the fitted and fixed variants
   # cannot be handed the wrong pair. H_M is identical for both -- same t, same
   # N_m, and it never depended on T -- so it is built once.
   H_M_model = None
   for sol, sol_analytic, tag, style in (
         (two_layer, analytic, MT2T_TAG, MT2T_STYLE),
         (two_layer_fixed, analytic_fixed, MT2_FIXED_TAG, MT2_FIXED_STYLE)):
      m = sol.between(start, n_years)
      t_s = sol.t[m]
      H_T_model = sol.H_T_per_area(m)
      if H_M_model is None:
         H_M_model = H_M_per_area(t_s, params["N_m"], bundle.ramp_years)
      ax.plot(t_s, H_T_model * ZJ_PER_WYRM2, color="tab:orange",
              label=rf"$H_T$ {tag}", **style)
      ax.plot(t_s, (H_M_model + H_T_model) * ZJ_PER_WYRM2, color="tab:blue",
              label=rf"$H_{{MT}}$ {tag}", **style)

      if sol_analytic is not None:
         H_T_pred = model_H_T_per_area(sol_analytic.T(t_s), sol_analytic.T_d(t_s),
                                       sol.c_u, sol.c_d)
         ax.plot(t_s, H_T_pred * ZJ_PER_WYRM2, color="tab:green",
                 label=rf"$H_T$ {tag} (analy. 2-box)", **style)
         ax.plot(t_s, (H_M_model + H_T_pred) * ZJ_PER_WYRM2, color="tab:purple",
                 label=rf"$H_{{MT}}$ {tag} (analy. 2-box)", **style)

   if start <= PAPER_H_YEAR <= n_years:
      lo, hi = PAPER_H_SPAN_ZJ
      ax.plot([PAPER_H_YEAR] * 2, [lo, hi], color="red", lw=2.5, alpha=0.4,
              solid_capstyle="butt", label="Paper yr-120 spread")
      ax.plot(PAPER_H_YEAR, PAPER_H_MEAN_ZJ, "x", color="red", ms=10, mew=2.5,
              label=r"Paper $\langle H\rangle$ yr 120")

   mark_calibration_end(ax, params, n_years)

   # H rises monotonically from the origin, so the lower right is the only
   # corner free of curves -- and the upper left is where format_ax puts the
   # model text.
   # H_T/H_M comes from the params table rather than being re-derived here, so
   # the panel and fitted_model_params.csv cannot disagree -- they are read off
   # the same window at the same year by construction.
   format_ax(ax, text=panel_tag(model, params,
                                rf"$H_T/H_M$={params['H_T_over_H_M']:.2f}"),
             xscale="linear", yscale="linear", xlim=(start, n_years),
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


def ohc_ts_scatter(ax, bundle, T_eq):
   """The AOGCM's own trajectory on the ohc_ts axes, coloured by year.

   Shared by the plain ohc_ts panel and the M-sensitivity one so the black-box
   half of the two figures is drawn by the same code and cannot drift apart.
   Returns the scatter handle, for the figure-level colorbar.

   Integrated over the whole record, scattered from display_start: a ramped run
   enters the panel partway along its own trajectory, at the T and OHC the ramp
   left it with, rather than at the origin.
   """
   H_aogcm = cumint(bundle.t_full, bundle.N_full) * ZJ_PER_WYRM2
   t, T, H_aogcm = bundle.display(bundle.t_full, bundle.T_full, H_aogcm)
   sc = ax.scatter(T / T_eq, normalized_ohc(H_aogcm, T_eq),
                   c=t, cmap=plt.cm.turbo, s=12,
                   norm=plt.Normalize(vmin=0, vmax=OHC_TS_YEAR_MAX))
   ax.axvline(1.0, color="0.55", ls="--", lw=0.8)
   return sc


#: What ohc_ts_curves() returns. A namedtuple rather than a bare tuple because
#: the M-sensitivity figure picks its series BY NAME, one row per name -- and
#: still unpacks positionally wherever that reads better.
OhcTsCurves = namedtuple("OhcTsCurves", "x H_T H_M H")


def ohc_ts_curves(params, analytic, T_eq_norm):
   """MT2's predictive (x, H_T, H_M, H_M+H_T) on the ohc_ts axes.

   All three y-series are returned whether or not a given panel draws them --
   the plain panel takes H_T and the total, the M-sensitivity panel takes H_M
   -- so the two cannot end up integrating the same curve two different ways.

   `T_eq_norm` is the T_eq BOTH axes are normalised by, passed in rather than
   read from `params` because the M-sensitivity panel normalises every swept
   curve by the AOGCM's own fitted T_eq -- a counterfactual M moves T_eq, and
   normalising each curve by its own would hide exactly the shift the figure
   exists to show. For the plain panel the two are the same number.
   """
   eq_t = np.arange(1.0, OHC_TS_EQUILIBRIUM_TAUS * analytic.tau_s + 1.0)
   T_pred = analytic.T(eq_t)
   H_T = model_H_T_per_area(T_pred, analytic.T_d(eq_t),
                            params["c_u"], params["c_d"]) * ZJ_PER_WYRM2
   # params.get: refit_at_M and fit_model both supply it, but a params dict
   # assembled anywhere else defaults to the abrupt case.
   H_M = H_M_per_area(eq_t, params["N_m"],
                      params.get("ramp_years", 0.0)) * ZJ_PER_WYRM2
   return OhcTsCurves(T_pred / T_eq_norm,
                      normalized_ohc(H_T, T_eq_norm),
                      normalized_ohc(H_M, T_eq_norm),
                      normalized_ohc(H_T + H_M, T_eq_norm))


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

   sc = ohc_ts_scatter(ax, bundle, T_eq)

   extra = (rf"$T_{{eq}}$={T_eq:.1f} K",)
   if analytic is None:
      extra += ("no real two-box solution",)
   else:
      x, y_HT, _, y_H = ohc_ts_curves(params, analytic, T_eq)

      ax.plot(x, y_HT, color="tab:green", lw=2.0,
              label=r"MT2 $H_T$ (analy. 2-box)")
      ax.plot(x, y_H, color="tab:green", lw=2.0, ls="-.", alpha=0.45,
              label=r"MT2 $H=H_M+H_T$")

      ceiling = ((params["c_u"] + params["c_d"]) * ZJ_PER_WYRM2
                 / OCEAN_HEAT_CAPACITY_ZJ)
      extra += (rf"$(C_u+C_d)/C_{{ocean}}$={ceiling:.2f}",)

      # y = (tau_s/OHC_eq) x + [(C_u+C_d)/C_ocean - tau_s/OHC_eq], i.e. the
      # straight line of that slope through the two-layer ocean's equilibrium
      # point (1, ceiling) -- the top of the solid green H_T curve. Drawn with
      # axline so it spans the panel without pulling the autoscale around.
      #
      # CAVEAT: tau_s is in years and OHC_eq in ZJ, so this slope carries units
      # of yr/ZJ while both axes are dimensionless. Numerically it lands
      # between 0.003 and 0.013 across these four models, against a ceiling of
      # 0.09-0.31, so the line is within about 1% of horizontal at the ceiling.
      slope = analytic.tau_s / (OCEAN_HEAT_CAPACITY_ZJ * T_eq)
      ax.axline((0.0, ceiling - slope), slope=slope, color="tab:purple",
                lw=1.5, ls=(0, (5, 2)),
                label=r"$\tau_s/\mathrm{OHC}_{eq}$ line")

   format_ax(ax, text=panel_tag(model, params, *extra),
             xscale="linear", yscale="linear", ylim=(-0.05, 1.2),
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="center left",
             legend=analytic is not None)   # nothing labelled without the fit
   return sc


# ---------- Sensitivity of the ohc_ts curve to the AMOC strength M ----------
# The one genuinely MT2-specific knob on that panel. Everywhere else in this
# script M is whatever the paper's Table 1 says for that AOGCM; here it is swept
# and the whole chain re-run, so the panel shows how much of the fitted curve's
# shape is actually pinned by M and how much would survive a different one.
#
# M is NOT a cosmetic scaling of one curve -- it enters at four places, and the
# swept curve is the composition of all four:
#
#   p     = s_0 (M - M_0)              rises with M
#   N_M   = p F_ref                    the constant, never-decaying uptake
#   F_eff = (1-p) F_ref                so T_eq = F_eff/alpha FALLS as M rises,
#                                      and with it alpha and eps (fit_radiative
#                                      bins N - N_M, so the binned intercept
#                                      moves while its slope does not)
#   H_T   = H - H_M                    the MT2T regressand, so c_u, c_d and
#                                      gamma are refit for every swept M
#
# Everything is therefore re-derived from the AOGCM record at each M by
# refit_at_M(); nothing is rescaled from the fitted answer. Both axes stay
# normalised by the AOGCM's OWN fitted T_eq (see ohc_ts_curves), so a swept
# curve landing at x != 1 is the point of the figure, not a bug: it is where
# that counterfactual M would have put equilibrium relative to the fitted one.
#
# What the panel DRAWS out of all that is H_M alone -- the one branch M sets
# directly, through N_M -- with H_T left to the plain ohc_ts panel. See
# plot_ohc_ts_M_sensitivity().
#
# Swept in absolute Sv rather than as a fraction of each model's own M, so the
# same set of lines appears on every panel and the panels are comparable; each
# model's own M is drawn on top in the plain panel's green.
#
# The sweep bottoms out at M_0 itself, written as the constant rather than as a
# literal -10.2 so the two cannot drift apart. That endpoint is the one value of
# M with a closed-form answer, and it anchors the whole family: p = s_0(M - M_0)
# is zero there, so N_M = 0, H_M flattens to the constant U_0, and F_eff = F_ref
# with no AMOC term taken out of it -- MT2 collapses to the plain two-box model
# every other script in this project fits, at that model's largest T_eq. Every
# other curve on the panel is that limit plus a growing N_M.
#
# M_0 is not a physical AMOC strength (no AOGCM here has one below 11.5 Sv, and
# a negative overturning is not a state any of them are in); it is the
# intercept of the paper's Eq. 22 regression, and the swept values below the
# observed range are extrapolations of that fit, not simulated climates.
M_SENSITIVITY_SV = (M_0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
M_SENSITIVITY_CMAP = plt.cm.plasma

# Shared colour scale for the swept lines, spanning the sweep AND every model's
# own M, so a model whose Table-1 M fell outside the swept range would still be
# coloured consistently with the lines around it.
M_SENSITIVITY_NORM = plt.Normalize(
   vmin=min((*M_SENSITIVITY_SV, *M_SV.values())),
   vmax=max((*M_SENSITIVITY_SV, *M_SV.values())))


def refit_at_M(model, bundle, M):
   """Re-run this model's whole M-dependent chain at a counterfactual `M`.

   The trimmed counterpart of fit_model(): the same radiative fit and the same
   MT2T two-layer fit over the same equilibration-derived window, minus the _HO
   and _val variants, which no figure asks for at a swept M. Goes through
   refit_radiative() rather than fit_radiative() so the per-model cache keeps
   holding the AOGCM's own answer.

   Returns (params, analytic); `analytic` is None on the same terms as
   two_box_analytic(), i.e. when the refit hands back parameters with no real
   two-box solution.
   """
   p = s_0 * (M - M_0)
   rad = refit_radiative(model, bundle, p, verbose=False)
   N_m = p * rad.F_ref

   params = {"M": M, "p": p, "N_m": N_m, "ramp_years": bundle.ramp_years}
   params.update(rad.as_params())
   params.update(two_layer_fit(model, bundle, N_m, rad.eq_year, verbose=False))

   analytic = two_box_analytic(
      lam=params["alpha"], F_eff=params["F_eff"], eps=params["eps"],
      F_total=params["F_ref"], c_u=params["c_u"], c_d=params["c_d"],
      gamma=params["gamma"])
   return params, analytic


def ohc_ts_M_sweep(model, bundle, params, M_values=M_SENSITIVITY_SV):
   """[(M, params, analytic), ...] for one model, one entry per swept M.

   `params` is that model's own fitted row, used only to print the fitted N_M
   beside the swept ones.

   Values of M with no real two-box solution are dropped rather than silently
   skipped -- with the whole chain refit, a swept M can land on parameters the
   AOGCM's own M does not, and a gap in the family of curves should say why.
   """
   sweep = []
   for M in M_values:
      params_M, analytic_M = refit_at_M(model, bundle, M)
      if analytic_M is None:
         print(f"   NOTE {model}: M={M:.1f} Sv gives no real two-box solution "
               f"(alpha={params_M['alpha']:.3f}, "
               f"gamma={params_M['gamma']:.3f}, "
               f"eps={params_M['eps']:.3f}) -- omitted from the M sweep")
         continue
      sweep.append((M, params_M, analytic_M))

   # N_M is the one swept quantity that does NOT need the refit to predict:
   # F_ref comes off the years-1-20 Gregory regression, which has no p in it,
   # so every refit shares the AOGCM's own F_ref and N_M = s_0(M - M_0)F_ref is
   # exactly linear in M. Printed anyway, straight off the refits rather than
   # recomputed, because it is the term the swept curves' divergence is easiest
   # to read against -- and because printing it is what would catch F_ref
   # silently acquiring an M dependence.
   swept = "  ".join(f"M={M:g}: {pm['N_m']:.4f}" for M, pm, _ in sweep)
   print(f"   {model}: N_M (W m^-2) at the fitted M={params['M']:.1f} Sv is "
         f"{params['N_m']:.4f}; swept  {swept}")
   return sweep


#: One row of the M-sensitivity figure: which of OhcTsCurves' y-series it
#: draws, how that series is labelled in the legend, and its per-panel title.
#: Adding a row here adds a row to the figure -- the grid sizes itself off this.
MSensitivityRow = namedtuple("MSensitivityRow", "field symbol title")

M_SENSITIVITY_ROWS = (
   MSensitivityRow("H_M", r"$H_M$", r"$H_M = U_0 + N_M t$"),
   MSensitivityRow("H", r"$H$", r"$H = H_M + H_T$"),
)


def plot_ohc_ts_M_sensitivity(ax, model, bundle, params, analytic, sweep, row):
   """One ohc_ts panel showing `row`'s series as a family of curves -- one per
   swept AMOC strength, plus the model's own.

   `row` is an MSensitivityRow, so the same panel code draws both rows of the
   figure and the two cannot end up differing in anything but the series:

     H_M   what M sets DIRECTLY. N_M = s_0(M - M_0)F_ref is the one quantity on
           these panels that does not come from a refit, so this row is the
           sensitivity in its rawest form.
     H     what M does to the heat content ACTUALLY compared against the AOGCM.
           H_T is in here too, and it moves with M as well -- the MT2T
           regression is refit at every swept M -- but only as a second-order
           consequence of F_eff = (1-p)F. Reading the two rows against each
           other is how much of H's spread is the N_M term and how much is the
           two-layer ocean responding to a changed forcing.

   H_T is never given a row of its own; the plain ohc_ts panel is the place to
   read it, against the same AOGCM scatter.

   Two things follow from H_M's form, apply to BOTH rows since both contain it,
   and are features of the panel rather than faults:

     * every curve with N_M > 0 leaves the TOP of the panel. N_M never decays,
       so H_M grows linearly and without bound while T is pinned at
       equilibrium -- the paper's own Sect. 6.3 caveat, and the same streak
       the plain panel's faded total shows. The panels keep the plain panel's
       y-limits rather than rescaling to fit it, so the AOGCM scatter stays
       the same size in every one of these figures and they can be read side
       by side.

       M = M_0 is the exception, and the only convergent member of either
       family: N_M = 0 there, so the top row flattens to the constant U_0 and
       the bottom row converges to U_0 + (C_u + C_d)T_eq like any two-box
       model. It is the sweep's lower endpoint for exactly that reason.
     * the streaks stand at DIFFERENT x. Each curve runs to its own
       T_eq(M)/T_eq(fitted), and T_eq falls as M rises because F_eff = (1-p)F
       does, so the family fans out horizontally as well as in slope.

   Returns the scatter handle, for the figure-level Year colorbar.
   """
   T_eq = params["T_eq_fit"]
   sc = ohc_ts_scatter(ax, bundle, T_eq)

   # Lighter than the fitted curve on purpose: these are counterfactuals, and
   # the eye should land on the AOGCM and on the model's own M first. No
   # end-of-curve marker, unlike an H_T sweep would want: both rows' series
   # carry H_M, which has no equilibrium value to mark, so the last point says
   # only where the integration was stopped.
   for i, (M, params_M, analytic_M) in enumerate(sweep):
      curves = ohc_ts_curves(params_M, analytic_M, T_eq)
      color = M_SENSITIVITY_CMAP(M_SENSITIVITY_NORM(M))
      edge = i in (0, len(sweep) - 1)      # label the two ends of the sweep only
      ax.plot(curves.x, getattr(curves, row.field), color=color, lw=1.4,
              alpha=0.65,
              label=(rf"{row.symbol} at $M$={M:g} Sv" if edge else None))

   extra = (rf"$T_{{eq}}$={T_eq:.1f} K",)
   if analytic is None:
      extra += ("no real two-box solution at this model's own $M$",)
   else:
      curves = ohc_ts_curves(params, analytic, T_eq)
      ax.plot(curves.x, getattr(curves, row.field), color="tab:green", lw=2.4,
              label=rf"MT2 {row.symbol} (fitted $M$={params['M']:.1f} Sv)")

   # How far the sweep actually moves the two quantities behind these curves:
   # N_M, which is their slope in time, and T_eq, which is where each one turns
   # vertical. Quoted as ranges so the panel carries the sensitivity as numbers,
   # not only as a spread of lines.
   if sweep:
      T_eqs = [pm["T_eq_fit"] for _, pm, _ in sweep]
      N_ms = [pm["N_m"] for _, pm, _ in sweep]
      # "to" rather than a hyphen: the sweep now starts negative, and
      # "M=-10.2-30 Sv" is unreadable.
      extra += (rf"$M$={sweep[0][0]:g} to {sweep[-1][0]:g} Sv: "
                rf"$N_M$ {min(N_ms):.2f}-{max(N_ms):.2f} $W\,m^{{-2}}$",
                rf"$T_{{eq}}$ {min(T_eqs):.1f}-{max(T_eqs):.1f} K")

   format_ax(ax, title=row.title, text=panel_tag(model, params, *extra),
             xscale="linear", yscale="linear", ylim=(-0.05, 1.2),
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend_loc="center left",
             legend=bool(sweep) or analytic is not None)
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
   "first five years" strip to shade. A ramped run has no such dip on the panel:
   its track only starts at the temperature the ramp reached.

   The DIAGNOSTIC curves (orange) are each solve's own C_u T + C_d T_d against
   that solve's T, which is the AOGCM's own T -- so they share the black curve's
   x-values and the vertical gap between them is the model's H_T error at a
   given temperature. The PREDICTIVE curves (green) supply their own T(t) as
   well, and are run out to equilibrium, so they extend to the right of
   everything else and end at T_eq: that overshoot of the data's x-range is the
   point of the panel, not an axis-limit bug.

   Both H_T and T are `window`-year running means, matching the reference
   script's 10-yr smoothing of its H-vs-T panel. The predictive curves are already
   smooth and are left alone.
   """
   n_years, t, T_raw, N = bundle.window()
   start = bundle.display_start

   # Smoothed over the full window, then trimmed: a ramped run's track enters
   # the panel at the temperature the ramp reached, not at T = 0.
   H_T_aogcm = aogcm_H_T(t, N, params["N_m"], ramp_years=bundle.ramp_years)
   _, T, H_T_sm = bundle.display(t, running_mean(T_raw, window),
                                 running_mean(H_T_aogcm, window))
   ax.plot(T, H_T_sm * ZJ_PER_WYRM2,
           color="black", lw=1.5, label=r"AOGCM $H_T=H_N-H_M$")

   # Each solve carries its own (c_u, c_d) and is zipped with the analytic
   # solution built from the SAME parameters, so a predictive curve cannot be
   # drawn with the other variant's heat capacities.
   for sol, sol_analytic, tag, style in (
         (two_layer, analytic, MT2T_TAG, MT2T_STYLE),
         (two_layer_fixed, analytic_fixed, MT2_FIXED_TAG, MT2_FIXED_STYLE)):
      m, keep = sol.display_mask(start, n_years)
      ax.plot(running_mean(sol.T[m], window)[keep],
              running_mean(sol.H_T_per_area(m), window)[keep] * ZJ_PER_WYRM2,
              color="tab:orange", label=rf"$H_T$ {tag}", **style)

      if sol_analytic is not None:
         t_pred = predictive_years(sol_analytic)
         T_pred = sol_analytic.T(t_pred)
         H_T_pred = model_H_T_per_area(T_pred, sol_analytic.T_d(t_pred),
                                       sol.c_u, sol.c_d)
         ax.plot(T_pred, H_T_pred * ZJ_PER_WYRM2, color="tab:green",
                 label=rf"$H_T$ {tag} (analy. 2-box)", **style)

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
   start = bundle.display_start
   F_ref, lam, eps = params["F_ref"], params["alpha"], params["eps"]

   # The inversion below is a constant-forcing relation, so on a ramped run it
   # is only meaningful past the ramp -- which is also where the panel starts.
   _, T = bundle.display(t, running_mean(T_raw, window))
   if abs(eps - 1.0) > EFFICACY_INVERSION_FLOOR:
      H_inv = running_mean((F_ref - N - lam * T_raw) / (eps - 1.0), window)
      ax.scatter(T, bundle.display(t, H_inv)[1], s=8, alpha=0.5, color="black",
                 label=r"AOGCM $H=(F-N-\lambda T)/(\epsilon-1)$")

   T_line = np.linspace(0.0, max(float(T.max()), params["T_eq_fit"]), 100)
   ax.plot(T_line, (F_ref - lam * T_line) / eps, color="0.5", lw=1.0, ls="-.",
           label=r"$H=(F-\lambda T)/\epsilon$ ($N=H$)")

   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m, keep = sol.display_mask(start, n_years)
      ax.plot(running_mean(sol.T[m], window)[keep],
              running_mean(sol.H(sol.T[m], sol.T_d[m]), window)[keep],
              color="tab:orange", label=rf"$H=\gamma(T-T_d)$ {tag}", **style)
   for sol, tag, style in variants(analytic, analytic_fixed):
      t_pred = predictive_years(sol)
      ax.plot(sol.T(t_pred), sol.H(t_pred), color="tab:green",
              label=rf"$H$ {tag} (analy. 2-box)", **style)

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
   start = bundle.display_start
   T = running_mean(T_raw, window)

   def over_T(num, den):
      """num/den, blanked wherever |den| is too small to mean anything."""
      num = np.broadcast_to(np.asarray(num, dtype=float), np.shape(den))
      out = np.full(np.shape(den), np.nan)
      ok = np.abs(den) >= OHUE_T_FLOOR_K
      out[ok] = num[ok] / np.asarray(den, dtype=float)[ok]
      return out

   ax.plot(*bundle.display(t, over_T(running_mean(N, window), T)), color="black",
           lw=1.5, label=r"AOGCM $\kappa=N/T$")
   ax.plot(*bundle.display(t, over_T(params["N_m"], T)), color="black", lw=1.5,
           ls=":", label=r"$\kappa_M=N_M/T$")

   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m, keep = sol.display_mask(start, n_years)
      T_sm = running_mean(sol.T[m], window)
      q = over_T(running_mean(sol.N_t[m], window), T_sm)
      t_s = sol.t[m][keep]
      ax.plot(t_s, q[keep], color="tab:orange", label=rf"$q=N_T/T$ {tag}", **style)
      ax.plot(t_s, (over_T(params["N_m"], T_sm) + q)[keep], color="tab:blue",
              label=rf"$\kappa_{{MT}}$ {tag}", **style)

   mark_calibration_end(ax, params, n_years)
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(start, n_years))


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

   `analytic` adds the constant-forcing two-box prediction of
   N_T = F_eff - alpha T(t) - (eps-1)H(t) in green. Unlike every other curve
   here it uses no AOGCM timeseries at all, only the fitted parameters, so it
   is the one honest prediction on the panel.

   It is N_T that is drawn and not the total N, so the green curve is directly
   comparable to the two curves already labelled N_T -- the AOGCM's N - N_M
   (black dashed) and the two-layer solve's own N_t (blue) -- rather than
   sitting a constant N_M above them. N_T is also the branch that decays to
   zero; the total N tends to N_M under MT2, so plotting it here invited the
   same y=0 misreading the N_M line was added to the Gregory panel to fix. The
   total is still recoverable by eye as the green curve plus the dotted line."""
   n_years, t, _, N_full = bundle.window()
   start = bundle.display_start
   N_m = params["N_m"]

   smoothed = f"{window}-yr mean"
   ax.plot(*bundle.display(t, N_full), color="0.6", lw=0.8, alpha=0.6,
           label="AOGCM N (annual)")
   ax.plot(*bundle.display(t, running_mean(N_full, window)), color="black",
           lw=1.5, label=f"AOGCM N ({smoothed})")
   # The three AOGCM-derived curves share one colour (black) and are told apart
   # by line style; the 2-layer model's independent N_T is the one coloured
   # curve, so it reads immediately against the AOGCM set.
   ax.axhline(N_m, color="black", lw=1.5, ls=":", label=r"$N_M$")
   ax.plot(*bundle.display(t, running_mean(N_full - N_m, window)), color="black",
           lw=1.5, ls="--", label=rf"$N_T$ ({smoothed})")

   # lw 1.5, not the shared styles' 2.0: these sit among the thin AOGCM curves
   # above rather than among the thick predictive ones.
   for sol, tag, style in variants(two_layer, two_layer_fixed):
      m, keep = sol.display_mask(start, n_years)
      ax.plot(sol.t[m][keep], running_mean(sol.N_t[m], window)[keep],
              color="tab:blue", label=rf"$N_T$ {tag} ({smoothed})",
              **{**style, "lw": 1.5})
   for sol, tag, style in variants(analytic, analytic_fixed):
      ax.plot(*bundle.display(t, sol.N_t(t)), color="tab:green",
              label=rf"$N_T$ analy. 2-box ({tag})", **style)

   # The paper's ensemble mean runs to year 100 only, so a run whose panel
   # starts later than that has nowhere to put it -- drawing it would stretch
   # the x-axis back over the ramp to reach points off the left of the panel.
   if start <= PAPER_ENS_MEAN[:, 0].max():
      ax.errorbar(PAPER_ENS_MEAN[:, 0], PAPER_ENS_MEAN[:, 1],
                  yerr=PAPER_ENS_MEAN[:, 2],
                  fmt="x", color="red", ms=10, mew=2.5, elinewidth=2.0, capsize=4,
                  ls="none", label="Ens. Mean (paper)")
   mark_calibration_end(ax, params, n_years)
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(start, n_years))


def plot_temperature_timeseries(ax, model, bundle, params,
                                analytic=None, analytic_fixed=None):
   """Plot one model's AOGCM surface temperature against the constant-forcing
   analytic two-box surface temperature T(t), over the panel window.
   `analytic_fixed` is the same curve on the MT2 ensemble-mean constants, drawn
   faded for comparison.

   Neither of the two-layer solve's temperatures appears here. Its upper box is
   not a solution at all -- T there is the prescribed AOGCM boundary condition
   that drives the ODE, so it would duplicate the black curve exactly -- and
   its deep-layer T_d, though a genuine solution, is a different quantity from
   everything else on the panel and was crowding the surface comparison this
   figure exists to make. T_d still drives the H and N_T panels, where it is
   compared against something commensurate.

   The green analytic T(t) is the genuine prediction -- it comes from the
   fitted parameters alone and never sees the AOGCM temperature -- so the
   black-to-green gap is the model's actual surface-temperature error.

   CAVEAT for a ramped run: the analytic curve is a CONSTANT-forcing solution
   from t = 0, i.e. the abrupt-4x response of the fitted parameters, and the
   panel shows it from the end of the ramp onwards. It has therefore had the
   full forcing for the whole 140 years the AOGCM was still ramping up, so it
   sits ABOVE the black curve at the panel's left edge by construction. That
   offset is the ramp, not a fit failure; the two converge once the AOGCM
   catches up."""
   n_years, t, T, _ = bundle.window()
   start = bundle.display_start

   ax.plot(*bundle.display(t, T), color="black", lw=1.5,
           label=r"AOGCM ($T_{2\mathrm{-meter}}$)")
   for sol, tag, style in variants(analytic, analytic_fixed):
      ax.plot(*bundle.display(t, sol.T(t)), color="tab:green",
              label=rf"analy. 2-box $T$ ({tag})", **style)
   if analytic is not None:
      ax.axhline(analytic.T_eq, color="tab:green", ls=":", lw=1.2, alpha=0.7,
                 label=rf"$T_{{eq}}$={analytic.T_eq:.1f} K (MT2T)")

   mark_calibration_end(ax, params, n_years)

   # T and T_analytic both flatten towards T_eq across the top of the panel, so
   # an upper-corner legend sits on the curves; both curves rise away from the
   # lower right, which stays clear.
   format_ax(ax, text=panel_tag(model, params), xscale="linear", yscale="linear",
             xlim=(start, n_years), legend_loc="lower right")


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
   its full U_0 from t = 0. A ramped run excludes its whole ramp instead, and
   its panel starts past it, so there is no strip to draw."""
   n_years, t, T, N = bundle.window()
   start = bundle.display_start

   H_T = aogcm_H_T(t, N, params["N_m"], ramp_years=bundle.ramp_years)
   int_T = cumint(t, T)
   int_H = cumint(t, H_T)

   # Every term is built over the full window -- the integrals have to stay
   # anchored at t = 0, exactly as in the regression these terms come from --
   # and only the drawn extent is trimmed. `drawn` (the y-range the panel is
   # scaled to) is trimmed with them, so the ramp cannot set the scale either.
   def draw_terms(c_u, c_d, gamma, opacity, labelled):
      # `opacity` is the matplotlib alpha, not the feedback parameter.
      term_T = c_u * T
      term_intT = gamma * (1.0 + c_u / c_d) * int_T
      term_intH = -(gamma / c_d) * int_H
      term_sum = term_T + term_intT + term_intH
      tag = (lambda s: s) if labelled else (lambda s: "_nolegend_")
      ax.plot(*bundle.display(t, term_T * ZJ_PER_WYRM2), color="black", lw=1.5,
              ls="--", alpha=opacity, label=tag(r"$C_u T$"))
      ax.plot(*bundle.display(t, term_intT * ZJ_PER_WYRM2), color="black",
              lw=1.5, alpha=opacity,
              label=tag(r"$\gamma(1+C_u/C_d)\int T\,dt$"))
      ax.plot(*bundle.display(t, term_intH * ZJ_PER_WYRM2), color="red", lw=1.5,
              alpha=opacity, label=tag(r"$-(\gamma/C_d)\int H_T\,dt$"))
      ax.plot(*bundle.display(t, term_sum * ZJ_PER_WYRM2), color="tab:green",
              lw=2.0, ls="--", alpha=opacity, label=tag("Sum"))
      return np.concatenate(bundle.display(
         t, term_T, term_intT, term_intH, term_sum)[1:]) * ZJ_PER_WYRM2

   ax.plot(*bundle.display(t, H_T * ZJ_PER_WYRM2), color="red", lw=2.0, ls="--",
           label=r"$H_T$")
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
      span = np.concatenate([drawn, bundle.display(t, H_T)[1] * ZJ_PER_WYRM2])
      lo, hi = float(np.min(span)), float(np.max(span))
      pad = 0.08 * (hi - lo) if hi > lo else 1.0
      ylim = (lo - pad, hi + pad)

   mark_calibration_end(ax, params, n_years)
   format_ax(ax, text=text, xscale="linear", yscale="linear",
             xlim=(start, n_years),
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
   values, so the curves isolate how much of the timescales' drift comes from
   the ocean fit rather than from the feedback.

   Returns (windows, tau_slow, tau_fast), the timescales in years, the two-box
   system's SLOW and FAST eigenvalues, Geoffroy et al. (2013a) Eqs. 8-15 in the
   EBM-epsilon form of 2013b -- the same closed form two_box_analytic() uses for
   the predictive curves, evaluated on each window's (c_u, c_d, gamma) with that
   model's fixed alpha and eps:
                  b     = (alpha+eps*gamma)/c_u + gamma/c_d
                  delta = b^2 - 4 alpha gamma/(c_u c_d)
                  tau_s = c_u c_d / (2 alpha gamma) * (b + sqrt(delta))
                  tau_f = c_u c_d / (2 alpha gamma) * (b - sqrt(delta))

   The deep layer's own relaxation time c_d/gamma used to be returned and drawn
   alongside tau_s. It knows nothing about alpha, so it is not the timescale the
   coupled system actually equilibrates on, and carrying both under one tau_d
   axis invited reading the gap between them as a spread on one estimate.
   tau_f is a distinct mode of the same solution rather than a second estimate
   of the same number, so it is drawn here without that ambiguity.

   Windows whose fit is degenerate -- a negative gamma or c_d, or no real pair
   of timescales -- give NaN rather than a complex or negative tau, and appear
   as a gap in the plotted curve.
   """
   # The sweep starts min_window years into the CONSTANT-FORCING record: on a
   # ramped run every window shorter than the ramp has no fitted rows at all
   # (two_layer_fit drops them), so those points would be a run of NaN.
   last_year = bundle.panel_years
   first_window = bundle.ramp_years + min_window
   windows = np.arange(first_window, last_year + step, step, dtype=float)
   windows = windows[windows <= last_year]

   alpha = float(params["alpha"])
   F_eff = float(params["F_eff"])
   eps = float(params["eps"])
   N_m = float(params["N_m"])

   tau_slow = np.full(windows.shape, np.nan)
   tau_fast = np.full(windows.shape, np.nan)
   for i, window in enumerate(windows):
      fit = two_layer_fit(model, bundle, N_m, window, verbose=False)
      c_u, c_d, gamma = fit["c_u"], fit["c_d"], fit["gamma"]
      sol = two_box_analytic(alpha, F_eff, c_u, c_d, gamma, eps=eps)
      if sol is not None:
         tau_slow[i] = sol.tau_s
         tau_fast[i] = sol.tau_f

   return windows, tau_slow, tau_fast


def plot_tau_d_vs_window(ax, model, params, windows, tau_slow, tau_fast):
   """One model's panel of tau_s and tau_f against calibration window, with the
   1:1 line the 2013a/b panels carry -- where a curve crosses it, the fit is
   claiming a timescale as long as the record it was fit to, which is the point
   past which the estimate is extrapolation rather than measurement.

   The y-axis is clipped to the 99th percentile of tau_slow: it runs away by
   orders of magnitude wherever a window's fit is near-degenerate, and letting
   those excursions set the scale would flatten everything else. tau_fast sits
   well below it, so the same limits hold it too.
   """
   ax.plot(windows, tau_slow, color="tab:blue", lw=2.0,
           label=r"$\tau_s$ slow mode (2013a Eqs. 8-15)")
   ax.plot(windows, tau_fast, color="tab:red", lw=2.0,
           label=r"$\tau_f$ fast mode (2013a Eqs. 8-15)")
   ax.plot(windows, windows, color="black", lw=1.2, ls="--",
           label=r"$\tau$ = window")

   # This axis IS the calibration window, so the marker says which point on the
   # blue curve fit_model() actually adopted.
   mark_calibration_end(ax, params, float(windows[-1]))

   finite = tau_slow[np.isfinite(tau_slow)]
   ylim = None
   if finite.size:
      hi = float(np.nanpercentile(finite, 99))
      lo = min(0.0, float(np.min(finite)), float(np.nanmin(tau_fast))
               if np.any(np.isfinite(tau_fast)) else 0.0)
      if hi > lo:
         ylim = (lo, hi + 0.15 * (hi - lo))

   def last_finite(y):
      ok = np.flatnonzero(np.isfinite(y))
      return y[ok[-1]] if ok.size else np.nan

   # Both ends of the curves, when they differ: the values fit_model() adopted
   # at the equilibration year, and the ones the whole record would have given.
   # Reading the two off the same curve is the point of the panel.
   extra = (rf"full-run $\tau_s$: {last_finite(tau_slow):.0f} yr, "
            rf"$\tau_f$: {last_finite(tau_fast):.1f} yr",)
   eq_year = params.get("eq_year", np.nan)
   if np.isfinite(eq_year):
      in_range = windows[0] <= eq_year <= windows[-1]
      idx = np.searchsorted(windows, eq_year)
      adopted_s = tau_slow[idx] if in_range else np.nan
      adopted_f = tau_fast[idx] if in_range else np.nan
      extra = (rf"adopted $\tau_s$: {adopted_s:.0f} yr, "
               rf"$\tau_f$: {adopted_f:.1f} yr",) + extra

   format_ax(ax, text=panel_tag(model, params, *extra),
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
# "constant-forcing years" rather than "years": the blinded window is measured
# from the end of a run's forcing ramp, so for ECHAM5_li2012 it is years
# 140-290 rather than 1-150.
YEARS_LABEL = ("full run" if UNBLINDED_RUN
               else f"first {OHU_RATE_YEARS} constant-forcing years")

# ... and which two-layer parameterization produced it.
FIT_LABEL = "MT2T per-AOGCM fit vs MT2 fixed params"

# The caption every year-axis figure carrying mark_calibration_end() shares,
# so the panels are not read as fits to everything they show. One line, like
# the other notes here -- model_figure() only budgets extra vertical room for
# captions that embed their own newlines.
#
# Unblinded only, and not merely for tidiness: the blinded panels stop at
# OHU_RATE_YEARS, which is earlier than any model's equilibration year, so
# mark_calibration_end() draws nothing there and the caption would be
# explaining an absent marker -- and its "models with no marker never get
# there" clause would be actively false, since on that pass none of them has
# one. If a model ever did cross inside the blinded window, the marker's own
# legend entry names the year.
CALIB_NOTE = ((
   f"MT2T is calibrated only up to the dashed marker -- the first year the "
   f"{EQUILIBRATION_AVG_YEARS}-yr running mean of T reaches "
   f"{EQUILIBRATION_FRACTION:.0%} of $T_{{eq}}$ -- and extrapolated beyond it; "
   "models with no marker never get there and are fit over their whole record.")
   if UNBLINDED_RUN else None)

# Short form, for the one figure that already has a caption of its own. Two
# lines will not fit: model_figure() budgets for the second by deepening the
# top margin, which moves the panels but not the suptitle, and on this
# single-row grid the two collide.
CALIB_NOTE_SHORT = ("MT2T is calibrated only up to the dashed marker and "
                    "extrapolated beyond it")

print("==================================")
print(f"Output: {layout.label}")
print("==================================")

df = pd.DataFrame(columns=PARAM_COLS)
df["model"] = fit_models
df["M"] = [M_SV[m] for m in fit_models]

# Cache all model series so the fit and the plots use identical baselines/slices.
# LI2012_MODEL is the one entry that does not come from the .Rdata bundle; see
# make_li2012_bundle for what is different about that record.
series_by_model: Dict[str, SeriesBundle] = {
   model: (make_li2012_bundle() if model == LI2012_MODEL
           else make_series_bundle(data.rx2(model), expts))
   for model in fit_models
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
            f"eps={row['eps']:.3f}) -- predictive curves omitted for this model")
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
   H_M_end = float(H_M_per_area(solve.t[m_s], row["N_m"],
                                bundle.ramp_years)[-1] * ZJ_PER_WYRM2)
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
   # ncols pinned to the model count, as the make_model_grid_with_ratio call
   # below already does: these figures are a single row, which is the shape
   # GRID_MARGINS is tuned for. Left at the default 4, a fifth model would
   # reflow the grid to two rows and those margins would be wrong for every
   # figure here.
   fig, axs = make_model_grid(fit_models, title=title, xlabel=xlabel,
                              ylabel=ylabel, ncols=len(fit_models),
                              auto_rows=True, hide_unused=True)
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
   title=(f"MT2: 4xCO2 Rate of OHU , {YEARS_LABEL} "
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
   title=f"MT2: 4xCO2 Temperature , {YEARS_LABEL} ({FIT_LABEL})",
   ylabel=r"T$_{2m}$ Anomaly (K)",
   note=CALIB_NOTE,
   draw=lambda ax, model, bundle, params: plot_temperature_timeseries(
      ax, model, bundle, params,
      analytic=analytic_by_model[model],
      analytic_fixed=analytic_fixed_by_model[model]),
)
layout.save_results(fig, figname("4xCO2", "all_models", "Temperature_vs_Year"))
plt.close(fig)


# ------------------- Accumulated OHU H = H_M + H_T (Eq. 7) -------------------

fig, _, _ = model_figure(
   title=(f"MT2: 4xCO2 accumulated OHU $H=H_M+H_T$ , "
          f"{YEARS_LABEL} ({FIT_LABEL})"),
   ylabel="OHU (ZJ)",
   note=CALIB_NOTE,
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
   title=(f"MT2: 4xCO2 accumulated $H_T$ vs T$_{{2M}}$ , "
          f"{YEARS_LABEL} ({SMOOTH_WINDOW_YEARS}-year running mean, {FIT_LABEL})"),
   xlabel="2-meter Air Temperature Anomaly (K)",
   ylabel=r"$H_T$ (ZJ)",
   note=("Green analy. 2-box curves are run to equilibrium ("
         f"{OHC_TS_EQUILIBRIUM_TAUS:.0f}$\\tau_s$), so they extend past the "
         r"AOGCM's own temperature range and end at $T_{eq}$. On the abrupt "
         r"runs the AOGCM $H_T$ is negative at small $T$ because $H_M$ carries "
         r"its full $U_0$ from $t=0$ (the strip plot_ohu shades)."),
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
         "reconstruction, not an ocean diagnostic. Green analy. 2-box curves run "
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
   title=(f"MT2: 4xCO2 OHUE $\\kappa=N/T=\\kappa_M+q$ , "
          f"{YEARS_LABEL} ({SMOOTH_WINDOW_YEARS}-year running mean, {FIT_LABEL})"),
   ylabel=r"OHUE ($W\,m^{-2}\,K^{-1}$)",
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
   note=(f"{CALIB_NOTE_SHORT}; faded curves: validation fit (first "
         f"{OHU_RATE_YEARS} years) extrapolated over the full run"
         if UNBLINDED_RUN else None),
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
      draw=lambda ax, model, bundle, params: plot_ohc_ts(
         ax, model, bundle, params, analytic_by_model[model]),
   )
   cbar = fig.colorbar(handles[-1], ax=axs.tolist(), fraction=0.025, pad=0.025)
   cbar.set_label("Year")
   layout.save_results(fig, figname("4xCO2", "all_models", "ohc_ts"))
   plt.close(fig)


# ------------- OHU vs. Ts: sensitivity to the AMOC strength M -------------
# Same axes and same AOGCM scatter as the plain ohc_ts panel, with the fitted
# curve replaced by a family of them -- one per swept M, the whole chain refit
# at each (see refit_at_M). One ROW per series in M_SENSITIVITY_ROWS: H_M on
# top, the total H = H_M + H_T beneath it, over the same sweep, so the extra
# spread the second row shows is what M does through the two-layer ocean rather
# than through N_M. Tied to the unblinded pass for the same reason as the panel
# it varies.
if UNBLINDED_RUN:
   print("--- N_M across the M sweep (F_ref is M-independent, so N_M is "
         "linear in M) ---")
   sweep_by_model = {
      model: ohc_ts_M_sweep(model, series_by_model[model],
                            params_by_model[model])
      for model in fit_models}

   # The second figure model_figure() cannot build: it is one panel per
   # (model, row) rather than one per model, so the grid is len(M_SENSITIVITY_
   # ROWS) rows deep and every model appears once in each. make_model_grid is
   # given the model list repeated once per row -- it reads only the LENGTH of
   # that list, to size the grid -- with ncols pinned to the model count so the
   # rows cannot reflow into each other under --test.
   nmodels = len(fit_models)
   fig, axs = make_model_grid(
      list(fit_models) * len(M_SENSITIVITY_ROWS),
      ncols=nmodels, auto_rows=True,
      title=(r"MT2: 4xCO$_2$ OHU vs. Surface Temp (Normalized), "
             r"sensitivity to AMOC strength $M$"),
      xlabel=r"$T_s/T_{eq}$",
      ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$",
   )
   # GRID_MARGINS is deliberately NOT applied here. It exists because the
   # other figures on this grid are a single row, so the shared fractional
   # margins leave them half the absolute room; this grid is two rows deep,
   # which is the shape those defaults were written for.
   fig.subplots_adjust(top=0.90, bottom=0.09, hspace=0.22)
   fig.text(0.5, 0.945,
            r"Faded lines: the whole chain refit at each $M$. $N_M t$ is "
            r"unbounded, so every curve with $N_M>0$ leaves the top of its "
            r"panel; only $M=M_0$, where $N_M=0$, converges",
            ha="center", va="top", fontsize=AXIS_LABEL_FONTSIZE - 6,
            fontstyle="italic")

   handles = [
      plot_ohc_ts_M_sensitivity(
         axs[irow * nmodels + icol], model, series_by_model[model],
         params_by_model[model], analytic_by_model[model],
         sweep_by_model[model], row)
      for irow, row in enumerate(M_SENSITIVITY_ROWS)
      for icol, model in enumerate(fit_models)
   ]

   cbar = fig.colorbar(handles[-1], ax=axs.tolist(), fraction=0.025, pad=0.025)
   cbar.set_label("Year")
   cbar_M = fig.colorbar(
      plt.cm.ScalarMappable(cmap=M_SENSITIVITY_CMAP, norm=M_SENSITIVITY_NORM),
      ax=axs.tolist(), fraction=0.025, pad=0.02)
   # Title rather than set_label: a vertical colorbar puts its label to the
   # RIGHT of the bar, which is exactly where the Year colorbar sits.
   cbar_M.ax.set_title(r"$M$ (Sv)", fontsize=EXTRA_TEXT_FONTSIZE,
                       fontweight="bold", pad=10)
   cbar_M.set_ticks(list(M_SENSITIVITY_SV))
   layout.save_results(fig, figname("4xCO2", "all_models", "ohc_ts_M_sensitivity"))
   plt.close(fig)


# ------------- tau_d vs. calibration window (2013a/b's tau_s plot) -------------
# Needs the long record to say anything -- only meaningful, and only produced,
# on the unblinded pass, same as 2013a/2013b's equivalent (run_type == 3 there).
if UNBLINDED_RUN:
   def _draw_tau_d(ax, model, bundle, params):
      windows, tau_slow, tau_fast = tau_d_vs_window(model, bundle, params)
      plot_tau_d_vs_window(ax, model, params, windows, tau_slow, tau_fast)

   fig, _, _ = model_figure(
         title=(r"MT2T: $\tau_s$, $\tau_f$ vs. Calibration Window "
             r"($F_{eff}$, $\alpha$, $\epsilon$ fixed at their whole-record "
             "STEP-1 values)"),
      xlabel="Calibration Window (years)",
      ylabel=r"$\tau$ (years)",
      draw=_draw_tau_d,
   )
   layout.save_results(fig, figname("4xCO2", "all_models", "tau_d_vs_calibration_window"))
   plt.close(fig)
