#!/usr/bin/env python3
"""One OHC-vs-Ts panel per AOGCM, with all three fitted models drawn together.

Every longrun script already writes its own `4xCO2_all_models_ohc_ts.png`, each
showing that script's fit alone. This one reads the *saved parameter tables* of
three of them and puts their predictions on the same axes:

    diffusive-eps-joint_fixedRadiativeParams   1-box slab + diffusive thermocline (PDE)
    2013b                                      Geoffroy et al. (2013b) EBM-epsilon (2-box)
    MT2                                        Meccia & Tsutsui two-layer + AMOC term

Nothing is refit here. The only computation is forward evaluation of each
model's own solution from its stored parameters:

  2013b     closed form. T(t) is the two-mode step response; the heat content is
            C*T + C0*T_0 exactly (T_0 recovered from H = gamma*(T - T_0), with H
            itself read off the surface budget), so no numerical integration.
  diffusive re-solves the PDE at the stored (kappa_pde, h_ml_pde) out to
            equilibrium -- ~0.1 s per model -- and reads the heat content
            straight off the temperature field, rho_cp*(h_ml*theta_0 + int theta dz).
  MT2       closed form. The stored (c_u, c_d, gamma, alpha, eps, F_eff) are fed
            back through the constant-forcing two-box solve (a copy of the MT2
            script's two_box_analytic), exactly as the MT2 script's own OHC-vs-Ts
            panel does, and drawn the same two ways: the CONVERGING H_T = C_u*T +
            C_d*T_d is the primary curve, the like-for-like counterpart of
            curve_2013b/curve_diffusive; the TOTAL H = H_M + H_T is drawn beside
            it, faded. H_M = U_0 + N_M*t never stops growing, so that second
            curve runs off the top of the panel while T is pinned at T_eq -- the
            paper's own Sect. 6.3 caveat, drawn to scale. It is kept out of the
            y-limits so it cannot flatten everything else.

On top of those three numerical curves the figure also draws each model's
CLOSED-FORM OHU(h), where h = T_s/T_eq is the fractional warming. These are the
`curve_*_analytic` functions, dashed in the same colour as the numerical curve
each approximates, and they are what makes the SHAPE of OHC(T_s) legible:

  2-box     OHU(h) = F*tau_s*(h - a_f). A straight line -- the slow-mode 2-box
            has no curvature in this plane at all, which is the finding.
  diffusive OHU = F*T*(2*sqrt(tau/pi) - h), tau = t/T, T = (rho_cp)^2*kappa/lambda^2.
            Swept in tau (exact); the write-up's inverted h/(1-h) closed form is
            available via the module switch DIFFUSIVE_ANALYTIC_APPROX. Concave,
            and unbounded -- a semi-infinite column takes up heat forever -- so
            it is flagged `divergent` like MT2's total.
  MT2       H_M(h) = U_0 + N_M*tau_s*(ln a_s - ln(1 - h)), optionally plus the
            slow-mode H_T, which is linear in h. Convex, log-divergent at h -> 1,
            and ALL of its curvature is the -N_M*tau_s*ln(1 - h) term: tau_s and
            N_M = p*F set it, nothing else does.

All three drop the fast mode, T(t) ~ T_eq(1 - a_s e^{-t/tau_s}), which is what
makes h invertible: t(h) = tau_s[ln a_s - ln(1 - h)]. So they are defined only
for h >= a_f = 1 - a_s and start at OHU = 0 there rather than at the origin.
That offset is the fast mode's heat, not an error.

A missing fit is normal, not an error: MT2 only fits the four AOGCMs with a
tabulated AMOC strength, and its two-box solve returns nothing for MPIESM11's
negative fitted gamma. Any model whose row is absent keeps its panel and its
AOGCM scatter, minus that one curve.

MT2 also only ever fit run type 1, so on a `--run-type 2`/`3` figure its curve is
read from 01_replicate while the other two come from the requested run type. The
title names that mismatch rather than letting it pass unlabelled.

Both axes are physical -- surface temperature in K, heat content in ZJ -- rather
than the individual scripts' T_s/T_eq vs OHC/OHC_eq. The three fits disagree
about T_eq by up to 3 K for the same AOGCM, so normalising each curve by its own
would put three different scales in one panel, and normalising all three by one
reference would misplace two of them.

Instead each fit's equilibrium is marked where it actually falls: a dotted
crosshair in that fit's colour, vertical at its T_eq and horizontal at the heat
content it holds once the whole model sits there. The spread between the three
crosshairs is most of why the curves separate at all, and the same two numbers
are printed in the panel tag.

Usage:
    python3 compare_ohc_ts_across_models.py                 # run type 1, all 8 models
    python3 compare_ohc_ts_across_models.py --run-type 3
    python3 compare_ohc_ts_across_models.py --run-type all  # one figure per run type
    python3 compare_ohc_ts_across_models.py --test CCSM3,HadGEM2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erfcx

from figure_paths import RUN_TYPE_DIRS, comparison_dir, figname, params_csv
from plot_style import EXTRA_TEXT_FONTSIZE, format_ax, make_model_grid
from pde_solver_1lyrEBM_diffusive import (
   Params as PDEParams, solve_model as pde_solve_model, YEAR,
)

# ------------------------------- User settings -------------------------------

# Variant directory each fit is read from, keyed by the tag used in the legend.
# Adding a fourth model to the figure means adding its variant here and a
# `curve_*` function below -- nothing else in this script is model-specific.
VARIANT_DIFFUSIVE = "diffusive-eps-joint_fixedRadiativeParams"
VARIANT_2013B = "2013b"
VARIANT_MT2 = "MT2"

# MT2 only ever fit run type 1, so its curve on a run type 2 or 3 figure is read
# from 01_replicate. The other two fits still come from the requested run type;
# the title names the mismatch.
MT2_FALLBACK_RUN_TYPE = 1

# Where this figure lands: figures/<VARIANT_COMPARISON>/comparison/. Its own
# variant directory rather than any one fit's, because it belongs to none of them.
VARIANT_COMPARISON = "model-comparison"

ALL_LONGRUN_MODELS = [
   "CCSM3", "CESM1", "CNRMCM6", "ECHAM5", "GISSE2R", "IPSLCM5A", "HadGEM2", "MPIESM11",
]

# Matches the longrun scripts: the 4xCO2 series are anomalies against that
# model's own piControl mean.
SUBTRACT_PICONTROL_BASELINE = True

# Horizon of the two CLOSED-FORM predictive curves, as a multiple of that fit's
# own slow timescale (MT2's rule, since tau_s varies by more than 2x across
# these AOGCMs). 20 tau_s puts T within e^-20 of T_eq for all of them.
EQUILIBRIUM_TAUS = 20.0

# Horizon [yr] of the PDE curve, which has no single tau to scale by: the
# thermocline's diffusive time z_max^2/kappa is ~5000 yr, so this is generous.
# Evaluated on a log-spaced grid, as in the fit script, so the early rise is
# resolved rather than spanned by one interpolation segment.
PDE_EQUILIBRIUM_YEARS = 100000.0
PDE_N_SAVE = 250
PDE_DZ_TARGET = 15.0      # m; Nz = PDE_Z_MAX/PDE_DZ_TARGET + 1
PDE_Z_MAX = 2500.0        # m; the same closed, insulated column the fits used

# Unit conversions. SECONDS_PER_YEAR is 365 d here, not the PDE solver's 365.25 d
# YEAR, because it converts the *fitted parameters* (W yr m^-2 K^-1), which every
# longrun script produced against 31536000.
EARTH_AREA_M2 = 5.1e14
SECONDS_PER_YEAR = 31536000.0
ZJ_PER_WYRM2 = EARTH_AREA_M2 * SECONDS_PER_YEAR / 1e21     # ~16.08
ZJ_PER_JM2 = EARTH_AREA_M2 / 1e21                          # PDE side, already in J

# The real ocean's heat capacity: 1.37e21 kg of seawater at 3850 J kg^-1 K^-1.
# This is the "true" OHC_eq/T_eq every other longrun script normalises its AOGCM
# scatter by -- `normalized_OHC = (5.1e14 * 31536000 * cumsum(nettoa)) / (1.37e21
# * 3850)` -- i.e. the heat content the actual ocean would hold at equilibrium,
# owing nothing to any fit's ocean geometry. Drawn as a reference line so each
# model's own ceiling can be read against it.
OCEAN_HEAT_CAPACITY_ZJ = 1.37e21 * 3850.0 / 1e21           # ~5275 ZJ K^-1

# Which fit's T_eq the true-OHC_eq line is evaluated at, in order of preference.
# It needs SOME equilibrium temperature and the AOGCMs do not come with one;
# 2013b leads because it is the only fit present for all eight models.
TRUE_OHC_T_EQ_ORDER = ("2013b", "diffusive", "mt2")

# MT2's AOGCM-neutral constant term in H_M (Eq. 7).
MT2_U_0_ZJ = 84.0

# Colour scale (years) of the AOGCM scatter, matching every other ohc_ts figure.
YEAR_COLOR_MAX = 6000

# Top of each panel's y-axis, as a multiple of that panel's true OHC_eq. Fixing
# it to the real ocean's equilibrium rather than to the tallest curve keeps every
# panel on a comparable scale and stops whichever fit has the largest heat
# capacity from squashing the data into the bottom third. See draw_panel.
Y_HEADROOM = 1.2

# ---------------------------- Analytic overlays ------------------------------
# The closed-form OHU(h) curves. Drawn by default; --no-analytic drops them.
SHOW_ANALYTIC = True

# Samples per analytic curve, and where they stop. The diffusive and MT2 forms
# both diverge as h -> 1 (as 1/(1-h) and as -ln(1-h)), so they need a cap; 0.999
# is already well past the top of every panel.
ANALYTIC_N_POINTS = 600
ANALYTIC_H_MAX = 0.999

# Which diffusive closed form to draw.
#   False  the exact parametric solution, OHU = F*T*(erfcx(sqrt(tau)) +
#          2*sqrt(tau/pi) - 1), swept in tau and converted to h. Recommended.
#   True   the write-up's inverted form, OHU = F*T*((2/pi)*h/(1-h) - h), which
#          rests on e^{x^2}erfc(x) ~ 1/(1 + sqrt(pi) x). That approximation is
#          poor at small x: 4.5x too small at h = 0.1, and actually NEGATIVE
#          below h ~ 0.37. It converges to the exact curve only as h -> 1.
DIFFUSIVE_ANALYTIC_APPROX = False

# Whether the MT2 analytic adds the slow-mode H_T = c_u*T + c_d*T_d to H_M.
# The derivation drops H_T because it is linear in h and so adds no curvature --
# true, and the reason this stays optional. Default True only so the dashed
# curve is a like-for-like partner of the drawn numerical H = H_M + H_T; set
# False to see H_M alone, exactly as derived.
MT2_ANALYTIC_INCLUDE_HT = True

# Volumetric heat capacity of seawater [J m^-3 K^-1], for the diffusive
# timescale T = (rho_cp)^2 * kappa / lambda^2. Read off the PDE solver's own
# Params where possible so the two diffusive curves cannot drift apart; this
# literal is only the fallback for a solver that computes rho_cp rather than
# storing it.
RHO_CP_FALLBACK = 4.1e6

# One entry per drawn series: the legend is built from these proxies rather than
# from whatever a given panel happens to contain, so the single legend is
# complete even though no panel draws all four.
STYLES = {
   "aogcm":     dict(color="black",     ls="-", lw=0.0, label="AOGCM (colored by year)"),
   "diffusive": dict(color="tab:green", ls="-", lw=2.0, label="1-Box + Diffusion (PDE)"),
   "2013b":     dict(color="tab:red",   ls="-", lw=2.0, label=r"Geoffroy 2013b (2-Box, $\epsilon$)"),
   "mt2":       dict(color="tab:blue",  ls="-", lw=2.0, label=r"MT2 $H_T$ (analy. 2-box)"),
   # MT2's total H = H_M + H_T: same colour as "mt2" (it is the same fit seen a
   # second way), faded so the pair reads as one model rather than two.
   "mt2_total": dict(color="tab:blue", ls="-.", lw=2.0, alpha=0.45,
                     label=r"MT2 $H=H_M+H_T$"),
   # Analytic partners: the colour of the numerical curve each approximates,
   # dashed and thinner, so a pair reads as one model seen two ways.
   "diffusive_analytic": dict(color="tab:green", ls="--", lw=1.6,
                              label=r"diffusive analytic: $FT(2\sqrt{\tau/\pi}-h)$"),
   "2box_analytic":      dict(color="tab:red",   ls="--", lw=1.6,
                              label=r"2-box analytic: $F\tau_s(h-a_f)$"),
   "mt2_analytic":       dict(color="tab:blue",  ls="--", lw=1.6,
                              label=r"MT2 analytic: $A-B\ln(1-h)$"),
}

# Draw order, which is also legend order. CURVE_KEYS stays the three numerical
# fits -- the panel tag and the `notes` bookkeeping are keyed to it. TOTAL_KEYS
# is MT2's second curve, always drawn alongside "mt2" (not gated by
# show_analytic -- it is the paper's own Sect. 6.3 caveat, not an approximation
# -- see curve_mt2_total). The analytic curves are a third, parallel tuple, so
# any of the three sets draws without the others.
CURVE_KEYS = ("diffusive", "2013b", "mt2")
TOTAL_KEYS = ("mt2_total",)
ANALYTIC_KEYS = ("diffusive_analytic", "2box_analytic", "mt2_analytic")

# The true-OHC_eq reference line. Neutral colour and a heavier dash than the
# per-fit crosshairs, because it belongs to none of them.
TRUE_OHC_STYLE = dict(color="0.35", ls=(0, (6, 3)), lw=1.6,
                      label=r"True $OHC_{eq}$ (real ocean)")

# Per-fit equilibrium crosshairs: dotted, in that fit's own colour.
EQ_LINE_KW = dict(ls=":", lw=1.4, alpha=0.8)


def legend_handles(show_analytic=SHOW_ANALYTIC):
   """Proxy artists for the one legend, in a fixed order."""
   handles = [Line2D([], [], marker="o", ls="none", color="0.4",
                     label=STYLES["aogcm"]["label"])]
   for key in CURVE_KEYS + TOTAL_KEYS + (ANALYTIC_KEYS if show_analytic else ()):
      s = STYLES[key]
      handles.append(Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"],
                            alpha=s.get("alpha", 1.0), label=s["label"]))
   handles.append(Line2D([], [], **TRUE_OHC_STYLE))
   return handles


# ------------------------------- AOGCM series --------------------------------

@dataclass(frozen=True)
class SeriesBundle:
   t: np.ndarray      # years, 1-indexed
   T: np.ndarray      # 2 m air temperature anomaly [K]
   N: np.ndarray      # net TOA flux anomaly [W m^-2]


def as_1d_float(x):
   return np.asarray(x, dtype=float).ravel()


def cumint(t, y):
   """Cumulative time-integral of y(t), anchored at the origin.

   The (0, 0) sample is prepended before integrating and dropped afterwards, so
   the result is aligned with `t` and the integral passes through the origin --
   the same helper the MT2 script uses for the AOGCM heat content.
   """
   t0 = np.concatenate(([0.0], np.asarray(t, dtype=float)))
   y0 = np.concatenate(([0.0], np.asarray(y, dtype=float)))
   return cumulative_trapezoid(y0, t0, initial=0.0)[1:]


def load_aogcm_series(rdata_file=Path("./data/int_netToa_longrun.Rdata")):
   """Every model's baselined 4xCO2 (t, T, N), keyed by model name."""
   ro.r["load"](str(rdata_file))
   data = ro.globalenv["int_nettoa_longrun_data"]
   expts = [str(x) for x in ro.globalenv["expts"]]
   if "piControl" not in expts or "4xCO2" not in expts:
      raise RuntimeError(f"Expected expts to include 'piControl' and '4xCO2'; got {expts}")

   series = {}
   for model in ALL_LONGRUN_MODELS:
      md = data.rx2(model)
      T_4x = as_1d_float(md.rx2("4xCO2").rx2("T2M"))
      N_4x = as_1d_float(md.rx2("4xCO2").rx2("NETTOA"))
      if SUBTRACT_PICONTROL_BASELINE:
         T_base = float(np.nanmean(as_1d_float(md.rx2("piControl").rx2("T2M"))))
         N_base = float(np.nanmean(as_1d_float(md.rx2("piControl").rx2("NETTOA"))))
      else:
         T_base, N_base = 0.0, 0.0

      T = T_4x - T_base
      N = N_4x - N_base
      t = np.arange(1, 1 + len(T), dtype=float)
      good = np.isfinite(t) & np.isfinite(T) & np.isfinite(N)
      series[model] = SeriesBundle(t=t[good], T=T[good], N=N[good])
   return series


# ------------------------------ Parameter tables ------------------------------

def load_params(variant, run_type, fallback_run_type=None):
   """One variant's fitted-parameter table, or None if there is none to read.

   params_csv() picks params_final.csv over fitted_model_params.csv, so 2013b's
   converged table is preferred without this having to know which variant writes
   which name.

   `fallback_run_type` covers MT2, which only ever fit run type 1: asking for
   run type 2 or 3 falls back to that one table rather than dropping the curve
   from every panel. Returns (df, run_type_actually_read) so the caller can say
   so on the figure -- a curve fit to a different run type than the ones beside
   it is exactly the sort of thing that must not go unlabelled.
   """
   for candidate in (run_type, fallback_run_type):
      if candidate is None:
         continue
      try:
         path = params_csv(variant, candidate)
      except FileNotFoundError:
         continue
      if candidate != run_type:
         print(f"   {variant}: run type {run_type} has no table; "
               f"falling back to {RUN_TYPE_DIRS[candidate]}")
      print(f"   {variant}: {path}")
      return pd.read_csv(path), candidate

   print(f"   no parameter table for {variant} run type {run_type}; "
         f"its curve is omitted from every panel")
   return None, None


def row_for(df, model):
   """That model's row as a Series, or None if the fit does not include it."""
   if df is None:
      return None
   match = df.loc[df["model"] == model]
   return None if match.empty else match.iloc[0]


# -------------------------------- The curves ---------------------------------
# One function per model. Each returns a Curve, or None when that fit cannot
# produce one. `ceiling_ZJ` is the heat content the model holds once the whole
# thing sits at T_eq: the height its curve levels off at, and the horizontal half
# of the dotted crosshair draw_panel marks its equilibrium with.


@dataclass(frozen=True)
class Curve:
   T: np.ndarray
   OHC_ZJ: np.ndarray
   T_eq: float
   ceiling_ZJ: float
   # True when OHC has no finite limit, so the curve is drawn but kept out of
   # the panel's y-limits. MT2's total H does this -- see curve_mt2_total -- as
   # do the diffusive and MT2 analytic forms.
   divergent: bool = False
   # False suppresses the dotted equilibrium crosshair. The analytic curves set
   # it: each shares its parent fit's T_eq exactly, so drawing it again would
   # only overplot the same line in the same colour.
   crosshair: bool = True


def curve_2013b(row):
   """Geoffroy 2013b EBM-epsilon, entirely closed form.

   T(t) is the two-mode step response. The heat content is the exact C*T + C0*T_0
   rather than a re-integration of N: the primed surface budget gives
   epsilon*H = F - lambda*T - C dT/dt, and H = gamma*(T - T_0) inverts to T_0, so
   both boxes are known analytically. Differentiating C*T + C0*T_0 returns
   F - lambda*T - (epsilon-1)*H = N exactly, which is what the 2013b script
   integrated numerically for the same curve.
   """
   T_eq, tau_f, tau_s = float(row["T_eq"]), float(row["tau_f"]), float(row["tau_s"])
   a_f, a_s = float(row["a_f"]), float(row["a_s"])
   C, C0, gamma = float(row["C"]), float(row["C0"]), float(row["gamma"])
   F_ref, lmbda, eps = float(row["F_ref"]), float(row["lambda"]), float(row["epsilon"])
   if not np.isfinite([T_eq, tau_f, tau_s, C, C0, gamma, eps]).all() or eps == 0 or gamma == 0:
      return None

   t = np.arange(0.0, EQUILIBRIUM_TAUS * tau_s + 1.0)
   T = T_eq * (1.0 - a_f * np.exp(-t / tau_f) - a_s * np.exp(-t / tau_s))
   dTdt = T_eq * ((a_f / tau_f) * np.exp(-t / tau_f) + (a_s / tau_s) * np.exp(-t / tau_s))

   H = (F_ref - lmbda * T - C * dTdt) / eps       # physical deep uptake [W m^-2]
   T_0 = T - H / gamma                            # deep-box temperature [K]

   return Curve(T=T, OHC_ZJ=(C * T + C0 * T_0) * ZJ_PER_WYRM2, T_eq=T_eq,
                ceiling_ZJ=(C + C0) * T_eq * ZJ_PER_WYRM2)


def curve_diffusive(row, equilibrium_years=PDE_EQUILIBRIUM_YEARS):
   """1-box slab + diffusive thermocline, re-solved from the stored PDE parameters.

   The heat content is read straight off the solved field --
   rho_cp*(h_ml*theta_0 + int theta dz) -- not from a cumulative sum of N. The
   two agree identically (the slab sheds eps*H but the thermocline receives the
   unscaled H, so int N dt is exactly the ocean's own heat content), and reading
   the field skips a numerical integration and any dependence on the save grid.
   """
   kappa, h_ml = float(row["kappa_pde"]), float(row["h_ml_pde"])
   T_eq, F_ref, eps = float(row["T_eq"]), float(row["F_ref"]), float(row["epsilon"])
   if not np.isfinite([kappa, h_ml, T_eq, F_ref, eps]).all() or kappa <= 0 or T_eq <= 0:
      return None

   p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref, epsilon=eps,
      z_max=PDE_Z_MAX, Nz=max(4, int(PDE_Z_MAX / PDE_DZ_TARGET + 1)),
      t_final=equilibrium_years * YEAR,
   )
   t_eval = np.concatenate(([0.0], np.geomspace(YEAR, p.t_final, PDE_N_SAVE - 1)))
   t_eval[-1] = p.t_final          # guard fp round-off landing outside t_span
   try:
      sol = pde_solve_model(p, t_eval=t_eval)
   except Exception as exc:
      print(f"      PDE solve failed ({exc}); diffusive curve omitted")
      return None

   theta, z = sol["theta"], sol["z"]
   ohc_jm2 = p.rho_cp * (p.h_ml * theta[0] + trapezoid(theta, z, axis=0))
   return Curve(T=sol["dT"], OHC_ZJ=ohc_jm2 * ZJ_PER_JM2, T_eq=T_eq,
                ceiling_ZJ=p.rho_cp * (p.h_ml + p.z_max) * T_eq * ZJ_PER_JM2)


def two_box_analytic(lam, F_eff, c_u, c_d, gamma, eps):
   """(tau_f, tau_s, a_f, a_s, phi_f, phi_s) for the constant-forcing two-box
   system -- Geoffroy 2013a Eqs. 8-15 extended to 2013b's efficacy system.

   A copy of the MT2 script's function of the same name, which cannot be
   imported: that script is a bare top-level loop, so importing it would run the
   whole fit. Returns None when the discriminant is non-positive, i.e. when the
   parameters give no two real relaxation timescales -- which is exactly what
   MPIESM11's negative fitted gamma produces.
   """
   if not (lam > 0.0 and gamma > 0.0 and c_u > 0.0 and c_d > 0.0 and eps > 0.0):
      return None

   gamma_p = eps * gamma
   b = (lam + gamma_p) / c_u + gamma / c_d
   b_star = (lam + gamma_p) / c_u - gamma / c_d
   delta = b ** 2 - 4.0 * lam * gamma / (c_u * c_d)
   if not (delta > 0.0):
      return None
   sqrt_delta = float(np.sqrt(delta))

   scale = c_u * c_d / (2.0 * lam * gamma)
   tau_f = scale * (b - sqrt_delta)
   tau_s = scale * (b + sqrt_delta)
   phi_f = c_u / (2.0 * gamma_p) * (b_star - sqrt_delta)
   phi_s = c_u / (2.0 * gamma_p) * (b_star + sqrt_delta)
   a_f = phi_s * tau_f * lam / (c_u * (phi_s - phi_f))
   a_s = -phi_f * tau_s * lam / (c_u * (phi_s - phi_f))
   return dict(T_eq=F_eff / lam, tau_f=tau_f, tau_s=tau_s, a_f=a_f, a_s=a_s,
               phi_f=phi_f, phi_s=phi_s)


def _mt2_two_box_solve(row):
   """The constant-forcing two-box solve behind curve_mt2 and curve_mt2_total,
   plus the (t, T, T_d) grid both curves are evaluated on.

   Factored out so the two curves cannot drift apart the way the split between
   them once did -- see curve_mt2_total's docstring. Returns None if the fit
   has no real two-box solution (MPIESM11's degenerate full-run calibration).
   """
   c_u, c_d, gamma = float(row["c_u"]), float(row["c_d"]), float(row["gamma"])
   sol = two_box_analytic(lam=float(row["alpha"]), F_eff=float(row["F_eff"]),
                          c_u=c_u, c_d=c_d, gamma=gamma, eps=float(row["eps"]))
   if sol is None:
      return None

   T_eq = sol["T_eq"]
   t = np.arange(0.0, EQUILIBRIUM_TAUS * sol["tau_s"] + 1.0)
   fast = 1.0 - np.exp(-t / sol["tau_f"])
   slow = 1.0 - np.exp(-t / sol["tau_s"])
   T = T_eq * (sol["a_f"] * fast + sol["a_s"] * slow)
   T_d = T_eq * (sol["a_f"] * sol["phi_f"] * fast + sol["a_s"] * sol["phi_s"] * slow)
   return sol, t, T, T_d


def curve_mt2(row):
   """MT2's H_T = C_u*T + C_d*T_d (Eq. C41), the temperature-driven part of the
   heat content, closed form -- exact given the two-box solve, so nothing is
   integrated here.

   This is the curve the MT2 script's own OHC-vs-Ts panel treats as primary
   ("H_T (analy. 2-box)"): it converges, so it is the like-for-like counterpart
   of curve_2013b and curve_diffusive, each a single converging predictive
   curve with its own T_eq/OHC_eq crosshair. The AMOC part H_M is drawn
   separately, as curve_mt2_total -- mixing it in here is what once made this
   curve diverge instead of converge, unlike its counterparts from the other
   two fits.
   """
   out = _mt2_two_box_solve(row)
   if out is None:
      return None
   sol, t, T, T_d = out
   c_u, c_d = float(row["c_u"]), float(row["c_d"])
   H_T = (c_u * T + c_d * T_d) * ZJ_PER_WYRM2
   return Curve(T=T, OHC_ZJ=H_T, T_eq=sol["T_eq"],
                ceiling_ZJ=(c_u + c_d) * sol["T_eq"] * ZJ_PER_WYRM2)


def curve_mt2_total(row):
   """MT2's total heat content H = H_M + H_T (Eq. 7), the like-for-like
   counterpart of the AOGCM scatter, which is the cumulative NETTOA integral --
   i.e. everything the system took up, the AMOC term included.

   Shares curve_mt2's T(t)/T_d(t) exactly -- both come off the same
   _mt2_two_box_solve -- and adds H_M = U_0 + N_M*t, the AMOC part. N_M = pF is
   a constant that never decays, so H_M grows without bound and the total
   leaves the top of the panel while T is still pinned at T_eq -- the vertical
   streak at T_eq.

   That divergence is the paper's own Sect. 6.3 caveat -- "N_M cannot remain
   non-zero and unchanged as equilibrium is approached under constant forcing" --
   drawn to scale, not a plotting artefact. It is why the returned Curve is
   flagged `divergent`: draw_panel draws it but keeps it out of the y-limits, so
   it cannot flatten every other curve into the bottom of the panel.

   `crosshair` is off because this curve's T_eq is curve_mt2's, already drawn,
   and `ceiling_ZJ` is NaN because the drawn quantity has no equilibrium value
   of its own to mark.
   """
   out = _mt2_two_box_solve(row)
   if out is None:
      return None
   sol, t, T, T_d = out
   c_u, c_d = float(row["c_u"]), float(row["c_d"])
   H_T = (c_u * T + c_d * T_d) * ZJ_PER_WYRM2
   H_M = MT2_U_0_ZJ + float(row["N_m"]) * t * ZJ_PER_WYRM2
   return Curve(T=T, OHC_ZJ=H_T + H_M, T_eq=sol["T_eq"],
                ceiling_ZJ=np.nan, divergent=True, crosshair=False)


# --------------------------- Analytic OHU(h) forms ---------------------------
# The closed forms, in two layers. The `analytic_*` functions are pure maths on
# plain floats -- no table, no dataclass -- so they can be unit-tested against a
# direct numerical integration of OHU(t) = F*t - lambda*int_0^t T dt'. The
# `curve_*_analytic` wrappers pull parameters out of a fitted row and put the
# result on the panel's physical axes (K, ZJ).
#
# All three share one approximation: the fast mode is dropped, so
#
#     T(t) ~ T_eq (1 - a_s e^{-t/tau_s})   =>   h = 1 - a_s e^{-t/tau_s},
#
# which is invertible, t(h) = tau_s [ln a_s - ln(1 - h)]. It also means h starts
# at a_f = 1 - a_s rather than at 0: these curves begin partway up the panel, at
# the temperature the discarded fast mode has already delivered by t = 0+.


def h_grid(a_f, h_max=ANALYTIC_H_MAX, n=ANALYTIC_N_POINTS):
   """h from its t=0 value a_f up to h_max, the domain of the slow-mode forms.

   Spaced in log(1 - h) rather than in h, so samples cluster where the MT2 and
   diffusive curves turn steeply upward instead of spending resolution on the
   flat early part. None when a_f is not a usable starting point -- a fit with
   a_f >= h_max has no slow mode worth drawing.
   """
   if not np.isfinite(a_f) or not 0.0 <= a_f < h_max:
      return None
   return 1.0 - np.geomspace(1.0 - a_f, 1.0 - h_max, n)


def analytic_2box_ohu(h, F, tau_s, a_f):
   """2-box: OHU(h) = F*tau_s*(h - a_f)   [W yr m^-2].

   OHU(t) = F t - lambda int_0^t T dt' with lambda*T_eq = F and the slow-only T
   gives OHU(t) = F*a_s*tau_s*(1 - e^{-t/tau_s}); substituting
   e^{-t/tau_s} = (1 - h)/a_s, and a_s - 1 = -a_f, collapses it to a straight
   line in h. The straightness is the point: the 2-box produces NO curvature in
   this plane, which is why the write-up turns to the diffusive and MT2 forms.

   Its ceiling F*tau_s*a_s is not quite the exact (C + C0)*T_eq that curve_2013b
   levels off at. Integrating the summed box budget from 0 to infinity gives the
   identity C + C0 = lambda*(a_f*tau_f + a_s*tau_s), so the analytic ceiling is
   short by exactly lambda*a_f*tau_f*T_eq -- the heat the discarded fast mode
   carries, a few percent. (With epsilon != 1 there is a second, smaller gap:
   the drawn 2013b heat content sheds (epsilon - 1)*H, for which
   OHU = F t - lambda int T dt' has no term.)
   """
   return F * tau_s * (h - a_f)


def analytic_diffusive_ohu(F, T_scale, h_max=ANALYTIC_H_MAX, n=ANALYTIC_N_POINTS):
   """Diffusive: (h, OHU) for the semi-infinite diffusive ocean [-, W yr m^-2].

   With T_s(t) = T_eq(1 - e^{tau} erfc(sqrt(tau))), tau = t/T, integrating
   OHU = F t - lambda int T dt' by parts gives

       OHU = F*T*(e^{tau} erfc(sqrt(tau)) + 2*sqrt(tau/pi) - 1)
           = F*T*(2*sqrt(tau/pi) - h),

   the second line just using e^{tau} erfc(sqrt(tau)) = 1 - h. Returned
   PARAMETRICALLY in tau rather than as OHU(h): that needs no inversion of erfc
   and so no approximation. e^{tau} overflows past tau ~ 700, so the product is
   evaluated as scipy's scaled erfcx(sqrt(tau)) -- the same function, without
   the intermediate overflow.

   Two checks worth keeping in mind. Small tau: OHU -> F*T*tau = F*t, i.e. every
   watt goes into the ocean, which is what fixes the sign as -h and not +h.
   Large tau: OHU ~ 2*F*T*sqrt(tau/pi) grows as sqrt(t) forever, because a
   semi-infinite column never fills -- hence `divergent`. The fitted PDE closes
   its column at z_max = 2500 m, so the numerical green curve does level off and
   this one does not; they part company once the warming reaches the bottom.

   Set DIFFUSIVE_ANALYTIC_APPROX for the write-up's inverted form instead.
   """
   if not (np.isfinite(F) and np.isfinite(T_scale) and T_scale > 0):
      return None

   if DIFFUSIVE_ANALYTIC_APPROX:
      # sqrt(tau(h)) = (1/sqrt(pi)) h/(1-h), from e^{x^2}erfc(x) ~ 1/(1+sqrt(pi)x),
      # so 2*sqrt(tau/pi) = (2/pi) h/(1-h). Note the MINUS h: a +h cannot be
      # right, because OHU -> F*T*tau > 0 as h -> 0 needs the two leading O(h)
      # terms to cancel rather than add.
      h = np.linspace(1e-4, h_max, n)
      return h, F * T_scale * ((2.0 / np.pi) * h / (1.0 - h) - h)

   # tau_max from the large-tau tail erfcx(sqrt(tau)) ~ 1/sqrt(pi*tau), i.e.
   # 1 - h ~ 1/sqrt(pi*tau): the sweep then reaches h_max and stops there.
   tau_max = 1.0 / (np.pi * (1.0 - h_max) ** 2)
   tau = np.concatenate(([0.0], np.geomspace(1e-6, tau_max, n - 1)))
   h = 1.0 - erfcx(np.sqrt(tau))
   return h, F * T_scale * (2.0 * np.sqrt(tau / np.pi) - h)


def analytic_mt2_hm(h, N_M, tau_s, a_s, U_0):
   """MT2's AMOC term: H_M(h) = U_0 + N_M*tau_s*(ln a_s - ln(1 - h)).

   Units follow N_M and U_0; curve_mt2_analytic passes both in ZJ.

   H_M(t) = U_0 + N_M t is linear in TIME, but t(h) = tau_s[ln a_s - ln(1 - h)]
   is logarithmic in h, so in this plane it is A - B ln(1 - h) with
   A = U_0 + N_M*tau_s*ln(a_s) and B = N_M*tau_s. B alone sets the curvature
   (d2H_M/dh2 = B/(1-h)^2), which is the write-up's punchline: the shape is
   fixed by tau_s and by N_M = p*F, and by nothing else.

   One departure from the write-up, which has A = (U_0 + pF)*tau_s*ln(a_s) and
   B = tau_s*(U_0 + pF): U_0 is a heat CONTENT (MT2 Eq. 7 fixes it at 84 ZJ) and
   pF is a FLUX, so (U_0 + pF) is not a quantity. Reading Eq. 7 literally as
   H_M = U_0 + N_M t puts U_0 in the intercept alone, which is what this does.
   It changes A and B, not the functional form or which parameters set it.

   Divergent, and logarithmically so: H_M keeps growing while T sits at T_eq.
   That is the same Sect. 6.3 caveat curve_mt2_total carries; here it is the
   -ln(1-h) that says it.
   """
   return U_0 + N_M * tau_s * (np.log(a_s) - np.log1p(-h))


def pde_rho_cp():
   """The PDE solver's own rho*c_p [J m^-3 K^-1], or the fallback if unreadable.

   Two ways in, because Params may store rho_cp as a defaulted dataclass field
   or compute it from rho and c_p: read the field default if there is one, else
   build a throwaway Params and read the attribute. Only if both fail does this
   fall back to the literal, and it says so when it does.
   """
   field = getattr(PDEParams, "__dataclass_fields__", {}).get("rho_cp")
   default = getattr(field, "default", None)
   if isinstance(default, (int, float)):
      return float(default)
   try:
      return float(PDEParams(kappa=1e-4, h_ml=50.0, dT_eq=1.0, F0=1.0).rho_cp)
   except Exception:
      print(f"      note: could not read rho_cp from the PDE solver; analytic "
            f"diffusive timescale uses {RHO_CP_FALLBACK:g} J m^-3 K^-1")
      return RHO_CP_FALLBACK


def curve_2box_analytic(row):
   """The 2-box straight line on physical axes, from the 2013b fitted row."""
   T_eq, tau_s = float(row["T_eq"]), float(row["tau_s"])
   a_f, a_s = float(row["a_f"]), float(row["a_s"])
   lmbda = float(row["lambda"])
   if not np.isfinite([T_eq, tau_s, a_f, a_s, lmbda]).all() or tau_s <= 0 or T_eq <= 0:
      return None

   # F = lambda*T_eq rather than the tabulated F_ref: the derivation uses
   # lambda*T_eq = F throughout, so taking F this way guarantees the line lands
   # on its own equilibrium even where a fit's F_ref/lambda and T_eq disagree in
   # the last digit.
   F = lmbda * T_eq
   h = h_grid(a_f)
   if h is None:
      return None
   return Curve(T=h * T_eq,
                OHC_ZJ=analytic_2box_ohu(h, F, tau_s, a_f) * ZJ_PER_WYRM2,
                T_eq=T_eq, ceiling_ZJ=F * tau_s * a_s * ZJ_PER_WYRM2,
                crosshair=False)


def curve_diffusive_analytic(row):
   """Semi-infinite diffusive OHU(h), from the row curve_diffusive already uses.

   The one parameter that is not tabulated is the diffusive timescale itself:

       T = (rho*c_p)^2 * kappa / lambda^2     [s]

   -- the write-up's kappa/nu_beta^2 with nu_beta = lambda/(rho c_p). (As
   printed, kappa/nu_beta has units of metres, not seconds.) lambda = F_ref/T_eq
   comes off the row; rho*c_p is read from the PDE solver so both diffusive
   curves rest on one value.

   This is the pure diffusive limit -- no slab (h_ml -> 0), no efficacy
   (epsilon = 1) -- so it idealises the fitted PDE rather than restating it.
   Expect it to lead the numerical curve early, where the real slab has to be
   warmed first, and to overshoot it late, where the real column has a floor.
   """
   kappa, T_eq, F_ref = float(row["kappa_pde"]), float(row["T_eq"]), float(row["F_ref"])
   if not np.isfinite([kappa, T_eq, F_ref]).all() or kappa <= 0 or T_eq <= 0:
      return None

   lmbda = F_ref / T_eq
   # SECONDS_PER_YEAR, not the solver's 365.25-day YEAR, because the result is
   # multiplied by F and converted with ZJ_PER_WYRM2, which is built on 365 d.
   T_scale = (pde_rho_cp() ** 2) * kappa / lmbda ** 2 / SECONDS_PER_YEAR   # [yr]

   out = analytic_diffusive_ohu(F=lmbda * T_eq, T_scale=T_scale)
   if out is None:
      return None
   h, ohu = out
   return Curve(T=h * T_eq, OHC_ZJ=ohu * ZJ_PER_WYRM2, T_eq=T_eq,
                ceiling_ZJ=np.nan, divergent=True, crosshair=False)


def curve_mt2_analytic(row):
   """MT2's A - B ln(1 - h), from the same row and two-box solve as curve_mt2.

   H_T is added by default (MT2_ANALYTIC_INCLUDE_HT) so this is comparable with
   the numerical H = H_M + H_T beside it, but it enters only as a straight line:
   under the slow-mode-only solution a_s(1 - e^{-t/tau_s}) = h - a_f, so
   T_d = T_eq*phi_s*(h - a_f) and H_T = [c_u*h + c_d*phi_s*(h - a_f)]*T_eq.
   Every bit of curvature in the dashed blue curve is the log term.
   """
   c_u, c_d = float(row["c_u"]), float(row["c_d"])
   sol = two_box_analytic(lam=float(row["alpha"]), F_eff=float(row["F_eff"]),
                          c_u=c_u, c_d=c_d, gamma=float(row["gamma"]),
                          eps=float(row["eps"]))
   if sol is None:
      return None

   T_eq, tau_s, a_f, a_s = sol["T_eq"], sol["tau_s"], sol["a_f"], sol["a_s"]
   N_M = float(row["N_m"])
   if not np.isfinite([T_eq, tau_s, a_f, a_s, N_M]).all() or not 0.0 < a_s <= 1.0:
      return None
   h = h_grid(a_f)
   if h is None:
      return None

   # U_0 is already in ZJ, so H_M is built in ZJ and N_M [W m^-2] converted on
   # the way in -- matching curve_mt2's U_0 + N_m*t*ZJ_PER_WYRM2.
   ohc = analytic_mt2_hm(h, N_M=N_M * ZJ_PER_WYRM2, tau_s=tau_s, a_s=a_s,
                         U_0=MT2_U_0_ZJ)
   if MT2_ANALYTIC_INCLUDE_HT:
      ohc = ohc + (c_u * h + c_d * sol["phi_s"] * (h - a_f)) * T_eq * ZJ_PER_WYRM2
   return Curve(T=h * T_eq, OHC_ZJ=ohc, T_eq=T_eq, ceiling_ZJ=np.nan,
                divergent=True, crosshair=False)


# --------------------------------- Plotting ----------------------------------

def draw_panel(ax, model, bundle, curves, notes, show_legend,
               show_analytic=SHOW_ANALYTIC):
   """One AOGCM's panel: its own trajectory plus whichever fits exist.

   `curves` maps a STYLES key to a Curve; absent keys are simply not drawn, which
   is how a model missing an MT2 fit keeps its panel. `notes` carries why each
   absent one is absent, for the panel tag. Returns the scatter handle for the
   figure-level colorbar.
   """
   ohc_aogcm = cumint(bundle.t, bundle.N) * ZJ_PER_WYRM2
   sc = ax.scatter(bundle.T, ohc_aogcm, c=bundle.t, s=12, cmap=plt.cm.turbo,
                   norm=mpl.colors.Normalize(vmin=0, vmax=YEAR_COLOR_MAX))

   # Span of everything that CONVERGES. A curve flagged `divergent` -- MT2's
   # H_M + H_T total ("mt2_total"), whose H_M term grows linearly forever, and
   # the diffusive analytic -- is drawn but deliberately left out, so it runs
   # off the top instead of flattening every other curve into the bottom of the
   # panel.
   spans_x = [bundle.T]
   spans_y = [ohc_aogcm]

   for key in CURVE_KEYS + TOTAL_KEYS + ANALYTIC_KEYS:
      curve = curves.get(key)
      if curve is None:
         continue
      s = STYLES[key]
      ax.plot(curve.T, curve.OHC_ZJ, color=s["color"], ls=s["ls"], lw=s["lw"],
              alpha=s.get("alpha", 1.0), label=s["label"])
      spans_x.append(curve.T)
      if not curve.divergent:
         spans_y.append(curve.OHC_ZJ)

      # Each fit's equilibrium point, as a dotted crosshair in its own colour:
      # vertical at T_eq, horizontal at the heat content it holds once the whole
      # model sits there. Drawn with the curve so the pair cannot get out of step
      # with the numbers in the panel tag -- both come off the same Curve. These
      # are axhline/axvline, so they span the panel without entering the limit
      # calculation below; a ceiling above the capped top is simply not visible,
      # exactly as its curve is not.
      # ... except for the analytic curves, which carry crosshair=False: their
      # T_eq is their parent fit's, already drawn, and their ceiling is either
      # infinite or (2-box) a slightly lower version of the same number, which
      # would read as a second, spurious equilibrium.
      if not curve.crosshair:
         continue
      ax.axvline(curve.T_eq, color=s["color"], **EQ_LINE_KW)
      if np.isfinite(curve.ceiling_ZJ):
         ax.axhline(curve.ceiling_ZJ, color=s["color"], **EQ_LINE_KW)

   # The true OHC_eq: the REAL ocean's heat capacity at the reference T_eq, which
   # is the denominator every other longrun script normalises its AOGCM scatter
   # by. It owes nothing to any fit's ocean geometry, so it is the line the three
   # model ceilings are actually to be judged against -- the diffusive column is
   # built to sit on it, while both box models fall well short.
   T_eq_ref = next((curves[k].T_eq for k in TRUE_OHC_T_EQ_ORDER if k in curves), None)
   true_ohc = OCEAN_HEAT_CAPACITY_ZJ * T_eq_ref if T_eq_ref is not None else None
   if true_ohc is not None:
      ax.axhline(true_ohc, **TRUE_OHC_STYLE)

   x_all = np.concatenate(spans_x)
   y_all = np.concatenate(spans_y)
   x_pad = 0.05 * np.ptp(x_all)
   ax.set_xlim(x_all.min() - x_pad, x_all.max() + x_pad)

   # Top of the y-axis is Y_HEADROOM x the true OHC_eq, always. Every panel is
   # then on its own AOGCM's equilibrium scale, so the same fraction of panel
   # height means the same fraction of that ocean's capacity everywhere and the
   # gray reference line lands at the same height in all eight. Whatever exceeds
   # it -- MT2's divergent total, a diffusive ceiling above the real ocean's --
   # simply exits the top, which is why nothing here is a max() over the curves.
   # ("mt2" itself now converges, like the other two fits, so it is no longer
   # among the curves this excludes; only "mt2_total" and the diffusive
   # analytic are.)
   # Only when there is no fit at all (hence no T_eq to evaluate it at) does the
   # panel fall back to spanning its own data.
   y_lo = min(y_all.min(), 0.0)
   y_hi = (Y_HEADROOM * true_ohc if true_ohc is not None
           else y_lo + Y_HEADROOM * ohc_aogcm.max())
   ax.set_ylim(y_lo - 0.02 * (y_hi - y_lo), y_hi)

   # One line per available fit, quoting the two numbers its dotted crosshair
   # marks. The three rarely agree, and that spread is most of why the curves
   # separate at all.
   tag = [model]
   for key, name in (("diffusive", "diff"), ("2013b", "2013b"), ("mt2", "MT2")):
      curve = curves.get(key)
      if curve is None:
         tag.append(f"{name}: {notes.get(key, 'no fit')}")
         continue
      # No OHC_eq quoted when the curve itself has no equilibrium value --
      # every one of CURVE_KEYS converges now, so this only ever bites a
      # missing fit (handled above) or a NaN slipping through some future key.
      line = rf"{name}: $T_{{eq}}$={curve.T_eq:.1f} K"
      if np.isfinite(curve.ceiling_ZJ):
         line += rf", $OHC_{{eq}}$={curve.ceiling_ZJ:.0f} ZJ"
      tag.append(line)
   if true_ohc is not None:
      tag.append(rf"true $OHC_{{eq}}$={true_ohc:.0f} ZJ")

   format_ax(ax, text="\n".join(tag), xscale="linear", yscale="linear",
             text_fontsize=EXTRA_TEXT_FONTSIZE, legend=False)
   if show_legend:
      # loc="best" rather than a fixed corner: this is the only legend on the
      # figure, so it has to dodge whatever that one panel's data happens to be.
      # framealpha=1 (matplotlib's default ~0.8 lets the turbo-coloured AOGCM
      # scatter show through and muddy the swatches) with a thin edge so the
      # solid box still reads as a legend and not an opaque patch over the data.
      ax.legend(handles=legend_handles(show_analytic), loc="best",
                prop={"weight": "bold", "size": 11},
                framealpha=0.9, edgecolor="0.6")
   return sc


def build_figure(models, series, tables, sources, run_type, legend_panel,
                 equilibrium_years, show_analytic=SHOW_ANALYTIC):
   """The whole 4x2 grid, plus its shared colorbar.

   `sources` maps each key to the run type its table was actually read from,
   which is not always `run_type` -- MT2 falls back to run type 1. Any such
   mismatch goes in the title, so the figure never silently mixes run types.
   """
   # auto_rows/hide_unused rather than the longrun scripts' fixed 4x2: --test
   # can leave an odd number of models, which the fixed layout rejects outright.
   fig, axs = make_model_grid(
      models,
      title=r"4xCO$_2$ OHC vs. Surface Temp: AOGCMs and three fitted models",
      xlabel=r"$T_s$ (K)", ylabel="Ocean Heat Content (ZJ)",
      auto_rows=True, hide_unused=True,
   )

   # Provenance line under the suptitle, as its own fig.text rather than a second
   # title line: make_model_grid puts the panels at top=0.95, so a "\n" in the
   # title would land the second line on top of the panels' model tags. Drop the
   # panels far enough to make room for it.
   borrowed = [f"{STYLES[k]['label']} from {RUN_TYPE_DIRS[rt]}"
               for k, rt in sources.items() if rt is not None and rt != run_type]
   fig.subplots_adjust(top=0.90)
   fig.text(0.5, 0.935,
            f"{RUN_TYPE_DIRS[run_type]} parameters"
            + ("; " + ", ".join(borrowed) if borrowed else ""),
            ha="center", fontsize=14, fontweight="bold", color="0.35")

   sc = None
   for imodel, model in enumerate(models):
      print(f"{model}:")
      curves = {}
      notes = {}          # key -> why its curve is absent, for the panel tag

      # A model can drop out of a fit two ways, and the panel says which: the
      # table has no row for it at all, or it has one the model cannot be solved
      # from (a degenerate fit -- MT2's two-box solve on MPIESM11).
      #
      # Each entry is (curve key, which fitted table its row comes from, how to
      # build it). An analytic curve reads the SAME row as the numerical curve
      # it approximates -- that is the whole point of drawing the two together --
      # so "2box_analytic" is handed the 2013b table, not one of its own.
      builders = [
         ("diffusive", "diffusive", lambda r: curve_diffusive(r, equilibrium_years)),
         ("2013b", "2013b", curve_2013b),
         ("mt2", "mt2", curve_mt2),
         ("mt2_total", "mt2", curve_mt2_total),
      ]
      if show_analytic:
         builders += [
            ("diffusive_analytic", "diffusive", curve_diffusive_analytic),
            ("2box_analytic", "2013b", curve_2box_analytic),
            ("mt2_analytic", "mt2", curve_mt2_analytic),
         ]

      for key, table_key, build in builders:
         row = row_for(tables[table_key], model)
         if row is None:
            if key in CURVE_KEYS:
               notes[key] = "no fit"
            continue
         try:
            curve = build(row)
         except KeyError as exc:      # an older table missing a needed column
            print(f"      {key}: table has no column {exc}; curve omitted")
            curve = None
         if curve is None:
            # Only the three numerical fits get a line in the panel tag: an
            # analytic curve is missing exactly when its parent is, and saying
            # so twice would only crowd the tag.
            if key in CURVE_KEYS:
               print(f"      {key}: parameters give no solution; curve omitted")
               notes[key] = "no 2-box solution" if key == "mt2" else "degenerate fit"
         else:
            curves[key] = curve

      n_fits = sum(k in curves for k in CURVE_KEYS)
      print(f"   drew {n_fits}/3 fits"
            + (f" + {sum(k in curves for k in ANALYTIC_KEYS)}/3 analytic"
               if show_analytic else "")
            + (f"; {', '.join(f'{k} ({v})' for k, v in notes.items())}" if notes else ""))

      sc = draw_panel(axs[imodel], model, series[model], curves, notes,
                      show_legend=(imodel == legend_panel),
                      show_analytic=show_analytic)

   cbar = fig.colorbar(sc, ax=axs.tolist(), fraction=0.025, pad=0.025)
   cbar.set_label("Year", fontsize=18, fontweight="bold")
   cbar.ax.tick_params(labelsize=14)
   return fig


# ----------------------------------- Entry -----------------------------------

def parse_args():
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--run-type", default="1",
                   choices=[str(rt) for rt in sorted(RUN_TYPE_DIRS)] + ["all"],
                   help="which run type's parameter tables to read (default 1; "
                        "MT2 only fit run type 1). 'all' produces one figure per "
                        "run type")
   ap.add_argument("--test", nargs="?", const="2", default=None,
                   help="run a subset of models: bare --test = first 2, --test N = "
                        "first N, --test M1,M2 = named models")
   ap.add_argument("--equilibrium-years", type=float, default=PDE_EQUILIBRIUM_YEARS,
                   help=f"horizon of the PDE curve (default {PDE_EQUILIBRIUM_YEARS:g})")
   ap.add_argument("--no-analytic", action="store_true",
                   help="omit the dashed closed-form OHU(h) curves, leaving only "
                        "the three numerical fits")
   ap.add_argument("--legend-panel", type=int, default=0,
                   help="index of the single panel carrying the legend (default 0)")
   return ap.parse_args()


def select_models(spec):
   """The model subset selected by --test (the full list if the flag is absent)."""
   if spec is None:
      return ALL_LONGRUN_MODELS
   if spec.isdigit():
      return ALL_LONGRUN_MODELS[: max(1, int(spec))]
   names = {s.strip() for s in spec.split(",") if s.strip()}
   unknown = names - set(ALL_LONGRUN_MODELS)
   if unknown:
      raise SystemExit(f"--test: unknown model(s) {sorted(unknown)}; "
                       f"valid: {ALL_LONGRUN_MODELS}")
   return [m for m in ALL_LONGRUN_MODELS if m in names]


def run_one(run_type, models, series, legend_panel, equilibrium_years,
            show_analytic=SHOW_ANALYTIC):
   """Read one run type's tables, build its figure, and save it."""
   print(f"Reading parameter tables from {RUN_TYPE_DIRS[run_type]}/tables/:")
   loaded = {
      "diffusive": load_params(VARIANT_DIFFUSIVE, run_type),
      "2013b": load_params(VARIANT_2013B, run_type),
      # MT2 only ever fit run type 1, so every other run type borrows that table
      # rather than losing the curve. build_figure() says so in the title.
      "mt2": load_params(VARIANT_MT2, run_type,
                         fallback_run_type=MT2_FALLBACK_RUN_TYPE),
   }
   tables = {k: df for k, (df, _) in loaded.items()}
   sources = {k: rt for k, (_, rt) in loaded.items()}
   if all(df is None for df in tables.values()):
      print(f"   no parameter tables at all for run type {run_type}; skipping")
      return

   fig = build_figure(models, series, tables, sources, run_type,
                      legend_panel, equilibrium_years, show_analytic)

   out_dir = comparison_dir(VARIANT_COMPARISON)
   stem = figname("4xCO2", "all_models", "ohc_ts", RUN_TYPE_DIRS[run_type])
   out_png = out_dir / f"{stem}.png"
   out_pdf = out_dir / f"{stem}.pdf"
   fig.savefig(out_png, dpi=200, bbox_inches="tight")
   fig.savefig(out_pdf, bbox_inches="tight")
   plt.close(fig)
   print(f"Saved {out_png}\n      {out_pdf}")


def main():
   args = parse_args()
   models = select_models(args.test)
   if args.test is not None:
      print(f"*** TEST MODE: {len(models)}/{len(ALL_LONGRUN_MODELS)} models: {models} ***")
   if not 0 <= args.legend_panel < len(models):
      raise SystemExit(f"--legend-panel must be in [0, {len(models) - 1}]")

   run_types = sorted(RUN_TYPE_DIRS) if args.run_type == "all" else [int(args.run_type)]

   # Loaded once and reused across run types: it depends on nothing but the
   # AOGCM data itself, and is the slow (rpy2) step.
   print("Loading AOGCM series...")
   series = load_aogcm_series()

   for run_type in run_types:
      if len(run_types) > 1:
         print(f"\n=== run type {run_type} ({RUN_TYPE_DIRS[run_type]}) ===")
      run_one(run_type, models, series, args.legend_panel, args.equilibrium_years,
              show_analytic=not args.no_analytic)


if __name__ == "__main__":
   main()