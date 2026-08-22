#!/usr/bin/env python3
"""Digitize the time series of Li et al. (2012), Figure 1, and plot N against T.

Fig. 1 (p. 1074 of Clim Dyn 40:1071-1086) shows five stacked time series of the
ECHAM5/MPIOM 4xCO2 run over its full 6080 years:

    (a) dT     globally averaged surface temperature change      [K]
    (b) dSST   sea surface temperature change                    [K]
    (c) N      globally averaged net TOA downward radiative flux  [W m^-2]
    (d) dQ_D   deep-ocean heat uptake below 1525 m               [W m^-2]
    (e) N-dQ_D difference between (c) and (d)                    [W m^-2]

Like Fig. 3, the figure is vector art, so the curves come straight out of the
PDF's drawing operators. Panels (a) and (c) are the pair this script exists for:
digitized together they give the paper's own Gregory plot, N against T, which is
the same relation the fit scripts draw from the LongRunMIP data as
`4xCO2_all_models_T2M_vs_NETTOA_<variant>.png`.

Note on which T: Fig. 3's shallowest panel (already digitized by
extract_li2012_fig3.py) is the 6 m *ocean* temperature, not the surface
temperature the Gregory relation is written in. Panel (a) here is the paper's
dT, and it shares an x axis with panel (c) vertex for vertex, so the pairing
needs no cross-figure assumptions. Both are the 11-year running mean the paper
plots in color; the gray annual-mean spaghetti behind them is not collected.

Run from the repo root:

    python3 extract_li2012_fig1.py [--pdf data/li2012.pdf] [--full]

It writes:
  * data/li2012_fig1_curves.csv  (panel, quantity, unit, period, year, value) --
    every digitized point of every panel, one row per point;
  * data/li2012_fig1_N_vs_T.csv  (year, dT_K, N_W_m2) -- panels (a) and (c)
    interpolated onto a common 1-year grid, the table behind the plot;
  * figures/li2012_fig1_N_vs_T.png -- N against T, colored by the paper's four
    adjustment periods, each drawn as a 50-yr running mean over a faint track of
    the 1-yr series, with an N = F - lambda*T fit over each of the two periods
    that follow the CO2 ramp (see PERIOD_YEARS on why not the ramp);
  * figures/li2012_fig1_reconstruction.png (--full) -- the five panels redrawn
    from the digitized points, the fidelity check on the calibration below.

Requires pymupdf, like extract_li2012_fig3.py, so this stays a standalone tool
outside the fit scripts' runtime dependency set.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymupdf

from extract_li2012_fig3 import PERIOD_STYLE, CURVE_COLORS, _dedupe_by_x

# 0-based page index of Fig. 1 within the article PDF (printed page 1074).
FIG1_PAGE = 3

# --- Axis calibration ---------------------------------------------------------
# All of it read off the page's own stroke operators, and re-checked at run time
# by verify_calibration() rather than trusted blindly:
#   * the x axis carries 31 minor ticks along the bottom frame, evenly spaced at
#     10.9127 pt, i.e. one per 200 yr from year 0 at the frame's left edge to
#     year 6000. The frames run ~100 yr past the last tick and the plotted
#     running mean ends at year ~6071, both consistent with the 6080-yr run;
#   * each panel carries its value ticks on the left frame edge: five per K
#     panel (0, 3, 6, 9, 12) and seven per W m^-2 panel (-2 .. 4);
#   * panels (c), (d) and (e) additionally draw a dashed zero line, whose y
#     coincides with that panel's zero tick to within 0.01 pt -- an independent
#     confirmation of the zero levels below.
PT_PER_YEAR = 10.9127 / 200.0

# The four running-mean colors are shared with Fig. 3, so PERIOD_STYLE and
# CURVE_COLORS are imported rather than restated: one place to fix if a
# different printing of the article shifts a color.
PERIODS = list(PERIOD_STYLE)


@dataclass(frozen=True)
class Panel:
   """One of Fig. 1's five stacked panels and its axis calibration.

   `top`/`bottom` are the stroked frame in PDF points (the frames abut, so a
   panel's bottom edge is the next one's top edge); `x0` is that frame's left
   edge, which is year 0 for the curve drawn inside it; `zero_y` is the y of the
   panel's zero tick and `pt_per_unit` the tick spacing, so a point's value is
   (zero_y - y) / pt_per_unit.
   """

   key: str          # the panel letter as the paper labels it
   quantity: str     # column name in the CSV
   unit: str
   label: str        # axis label, matplotlib mathtext
   top: float
   bottom: float
   x0: float
   zero_y: float
   pt_per_unit: float
   ticks: tuple      # the printed tick labels, top to bottom

   def years(self, x):
      return (np.asarray(x, float) - self.x0) / PT_PER_YEAR

   def values(self, y):
      return (self.zero_y - np.asarray(y, float)) / self.pt_per_unit


# pt_per_unit: (139.47 - 68.12) / 4 ticks / 3 K for the K panels,
#              (306.07 - 234.64) / 6 ticks / 1 W m^-2 for the flux panels.
# `ticks` is what the panel actually prints down its left edge, top to bottom;
# verify_calibration() reads each tick's y off the page and requires the
# calibration to reproduce that label, so the zero level and the scale are both
# asserted against the figure rather than assumed.
K_TICKS = (12.0, 9.0, 6.0, 3.0, 0.0)
FLUX_TICKS = (4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0)
PANELS = [
   Panel("a", "dT", "K", r"$\Delta T$ [K]",
         62.17, 145.43, 223.14, 139.47, 17.8375 / 3.0, K_TICKS),
   Panel("b", "dSST", "K", r"$\Delta$SST [K]",
         145.43, 228.68, 223.14, 222.73, 17.8375 / 3.0, K_TICKS),
   Panel("c", "N", "W m-2", r"$N$ [W m$^{-2}$]",
         228.68, 312.02, 223.01, 282.24, 11.9050, FLUX_TICKS),
   Panel("d", "dQ_D", "W m-2", r"$\Delta Q_D$ [W m$^{-2}$]",
         312.02, 395.34, 223.01, 365.58, 11.9100, FLUX_TICKS),
   Panel("e", "N_minus_dQ_D", "W m-2", r"$N-\Delta Q_D$ [W m$^{-2}$]",
         395.34, 478.68, 223.01, 448.92, 11.9033, FLUX_TICKS),
]
PANEL_BY_KEY = {p.key: p for p in PANELS}

# The two panels this script is for: the Gregory pair.
T_PANEL, N_PANEL = "a", "c"

# Frame edge colour, used to find the ticks and zero lines during verification.
AXIS_COLOR = (0.075, 0.078, 0.075)


def _stroke_segments(page, color):
   """Straight segments of `color` on the page, as an (N, 4) x0,y0,x1,y1 array."""
   segs = []
   for path in page.get_drawings():
      if path["type"] != "s" or path["color"] is None:
         continue
      if tuple(round(c, 3) for c in path["color"]) != color:
         continue
      for item in path["items"]:
         if item[0] == "l":
            segs.append((item[1].x, item[1].y, item[2].x, item[2].y))
   return np.array(segs, dtype=float).reshape(-1, 4)


def verify_calibration(page, atol_pt=0.05):
   """Re-derive the constants above from the page; raise if they moved.

   Cheap insurance against running this on a different printing of the article:
   a shifted frame would otherwise silently rescale every digitized value
   instead of failing. Checks the x tick pitch, each panel's zero tick and tick
   pitch, and (for the flux panels) that the dashed zero line sits on the zero
   tick.
   """
   segs = _stroke_segments(page, AXIS_COLOR)
   segs = segs[segs[:, 1] < 500.0]          # the figure, not the page furniture
   horiz = segs[np.abs(segs[:, 1] - segs[:, 3]) < 0.05]
   vert = segs[np.abs(segs[:, 0] - segs[:, 2]) < 0.05]
   problems = []

   # x: minor ticks hanging off the bottom frame of the last panel.
   bottom = PANELS[-1].bottom
   xt = vert[(np.abs(vert[:, 1] - bottom) < 1.0)
             & (np.abs(vert[:, 3] - vert[:, 1]) < 9.0)]
   xs = np.array(sorted({round(v, 2) for v in xt[:, 0]}))
   if len(xs) != 31:
      problems.append(f"expected 31 x ticks along y={bottom}, found {len(xs)}")
   else:
      pitch = float(np.mean(np.diff(xs)))
      if abs(pitch / 200.0 - PT_PER_YEAR) > 1e-4:
         problems.append(f"x tick pitch {pitch:.4f} pt/200yr disagrees with "
                         f"PT_PER_YEAR={PT_PER_YEAR:.6f}")

   for p in PANELS:
      yt = horiz[(np.abs(horiz[:, 0] - p.x0) < 1.5)
                 & (np.abs(horiz[:, 2] - horiz[:, 0]) < 8.0)
                 & (np.abs(horiz[:, 2] - horiz[:, 0]) > 1.0)
                 & (horiz[:, 1] > p.top) & (horiz[:, 1] < p.bottom)]
      ys = np.array(sorted({round(v, 2) for v in yt[:, 1]}))
      if len(ys) != len(p.ticks):
         problems.append(f"panel {p.key}: expected {len(p.ticks)} value ticks, "
                         f"found {len(ys)}")
         continue
      # ys is sorted by increasing y, i.e. down the page, the order p.ticks is
      # written in. Every printed label must come back out of the calibration.
      got = p.values(ys)
      err = np.abs(got - np.array(p.ticks))
      if err.max() > atol_pt / p.pt_per_unit:
         worst = int(np.argmax(err))
         problems.append(f"panel {p.key}: tick at y={ys[worst]:.2f} reads "
                         f"{got[worst]:.3f} {p.unit}, printed {p.ticks[worst]:g}")
      if p.unit != "K":
         dashes = horiz[(np.abs(horiz[:, 1] - p.zero_y) < 0.05)
                        & (horiz[:, 0] > p.x0) & (horiz[:, 2] < p.x0 + 400)]
         if len(dashes) < 50:
            problems.append(f"panel {p.key}: dashed zero line not found at "
                            f"y={p.zero_y:.2f} ({len(dashes)} segments)")
   if problems:
      raise RuntimeError("Fig. 1 calibration does not match this PDF:\n  "
                         + "\n  ".join(problems))


def panel_curves(page):
   """{panel key: {period color: (year, value) array}} for every running mean.

   A point exactly on a shared frame edge belongs to the panel ABOVE, matching
   extract_li2012_fig3's convention: on the K panels the bottom edge is below
   zero, and on the flux panels the running mean never reaches either edge.
   """
   raw = {p.key: {} for p in PANELS}
   for path in page.get_drawings():
      if path["type"] != "s" or path["color"] is None:
         continue
      key = tuple(round(c, 3) for c in path["color"])
      if key not in CURVE_COLORS:
         continue
      for item in path["items"]:
         if item[0] not in ("l", "c"):     # line segment or Bezier
            continue
         for pt in item[1:]:
            for p in PANELS:
               if p.top < pt.y <= p.bottom + 0.3:
                  raw[p.key].setdefault(key, []).append((pt.x, pt.y))
                  break

   out = {}
   for p in PANELS:
      out[p.key] = {}
      for color, plist in raw[p.key].items():
         merged = _dedupe_by_x(np.array(plist, dtype=float))
         out[p.key][color] = np.column_stack([p.years(merged[:, 0]),
                                              p.values(merged[:, 1])])
   return out


def panel_series(curves, key):
   """One merged (year, value) series for a panel, all periods end to end.

   The periods are drawn as separate colored strands that share an endpoint
   where one hands over to the next; _dedupe_by_x collapses those, on the same
   0.05 pt scale used within a strand, converted to years.
   """
   strands = [a for a in curves[key].values() if a.size]
   if not strands:
      raise RuntimeError(f"no curve points found in panel {key}")
   return _dedupe_by_x(np.vstack(strands), xtol=0.05 / PT_PER_YEAR)


def paired_N_T(curves, step_years=1.0):
   """(year, dT, N) on a common grid, from panels (a) and (c).

   The two panels are digitized independently, so their vertices do not line up
   in year; both are interpolated onto a regular grid over the years they share
   rather than one being resampled onto the other's vertices.
   """
   t = panel_series(curves, T_PANEL)
   n = panel_series(curves, N_PANEL)
   lo = max(t[0, 0], n[0, 0])
   hi = min(t[-1, 0], n[-1, 0])
   years = np.arange(np.ceil(lo), np.floor(hi) + 1, step_years)
   return years, np.interp(years, t[:, 0], t[:, 1]), np.interp(years, n[:, 0], n[:, 1])


# Year windows of the paper's four adjustment periods, as its own legends name
# them, and the two that a Gregory regression is meaningful over.
#
# NOT the CO2-increase period: this is a 140-yr ramp to 4xCO2, not an abrupt
# step, so N and T rise together while the forcing grows and a regression there
# returns a negative feedback -- a statement about the ramp, not about the
# climate. Under the constant forcing that follows, N = F - lambda*T holds, and
# fitting the transient and quasi-equilibrium stretches separately is what shows
# lambda steepening as the deep ocean adjusts.
#
# The equilibrium period is left unfitted too, for the opposite reason: N is
# flat noise about zero there, so its slope is not identifiable.
PERIOD_YEARS = {
   "CO$_2$ increase (1-140 yr)": (1.0, 140.0),
   "transient (140-1200 yr)": (140.0, 1200.0),
   "quasi-equilibrium (1200-4600 yr)": (1200.0, 4600.0),
   "equilibrium (4600-6080 yr)": (4600.0, 6080.0),
}
FIT_PERIODS = ("transient (140-1200 yr)", "quasi-equilibrium (1200-4600 yr)")

# Window of the running mean drawn on the N-vs-T plot, in years of the common
# 1-yr grid. The digitized series is already the paper's 11-yr mean, so this is
# a second, much longer smoothing: over 6000 yr the track doubles back on itself
# often enough that the 1-yr wiggle hides the shape of each period, and 50 yr is
# short against every period here (the shortest is the 140-yr ramp).
SMOOTH_YEARS = 50


def running_mean(values, window=SMOOTH_YEARS):
   """Centered running mean over `window` samples of a regular series.

   The window shrinks rather than vanishing at the two ends, so the smoothed
   curve spans the same years as the input: trimming it would cut the start of
   the CO2 ramp, which is where the track moves fastest and is most worth
   seeing.
   """
   v = np.asarray(values, float)
   n = v.size
   if n == 0:
      return v
   window = int(min(window, n))
   half = window // 2
   csum = np.concatenate([[0.0], np.cumsum(v)])
   idx = np.arange(n)
   lo = np.clip(idx - half, 0, n)
   hi = np.clip(idx - half + window, 0, n)
   return (csum[hi] - csum[lo]) / (hi - lo)


def gregory_fit(years, dT, N, lo, hi):
   """Least-squares N = F - lambda*T over years lo..hi.

   Returns (F, lambda, T_intercept) with lambda positive for a damping
   feedback, i.e. the sign convention the fit scripts use. T_intercept is where
   the line reaches N = 0: the equilibrium warming this stretch extrapolates to.
   """
   m = (years >= lo) & (years <= hi)
   if m.sum() < 3:
      return None
   slope, intercept = np.polyfit(dT[m], N[m], 1)
   if slope == 0:
      return None
   return intercept, -slope, -intercept / slope


def period_fits(years, dT, N):
   """(label, F, lambda, T_intercept, T_min) for each constant-forcing period.

   T_min is the coldest year of that period, so a fit line can be drawn over the
   range it was fitted on rather than back to T = 0: extrapolating the
   quasi-equilibrium line down to an unforced climate it says nothing about
   would triple the plot's y range to show it.
   """
   out = []
   for label in FIT_PERIODS:
      lo, hi = PERIOD_YEARS[label]
      fit = gregory_fit(years, dT, N, lo, hi)
      if fit is not None:
         out.append((label, *fit, float(dT[(years >= lo) & (years <= hi)].min())))
   return out


def write_curves_csv(curves, out):
   """Long-format CSV: one row per digitized point of every panel."""
   out.parent.mkdir(parents=True, exist_ok=True)
   with out.open("w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["panel", "quantity", "unit", "period", "year", "value"])
      for p in PANELS:
         for color in PERIODS:
            arr = curves[p.key].get(color)
            if arr is None:
               continue
            # PERIOD_STYLE's labels carry mathtext for the plot legends; the
            # CSV gets the plain-text version.
            period = PERIOD_STYLE[color].replace("$_2$", "2")
            for year, value in arr:
               w.writerow([p.key, p.quantity, p.unit, period,
                           f"{year:.2f}", f"{value:.4f}"])


def write_pair_csv(years, dT, N, out):
   """The N-vs-T table itself, on the common grid."""
   out.parent.mkdir(parents=True, exist_ok=True)
   with out.open("w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["year", "dT_K", "N_W_m2"])
      for year, t, n in zip(years, dT, N):
         w.writerow([f"{year:.0f}", f"{t:.4f}", f"{n:.4f}"])


def plot_n_vs_t(curves, png_path):
   """N against T, one colored track per adjustment period.

   Each period is drawn twice: the 1-yr series faintly, and a SMOOTH_YEARS
   running mean of it opaque on top, which is the line the eye follows. Both
   smoothed coordinates come from a single pass over the whole run rather than
   period by period, so a period's window still sees its neighbours and the
   tracks meet where the colors hand over instead of each end bending toward
   its own truncated mean.

   The coloring is the paper's own rather than a re-derived set of year cutoffs:
   each period covers the years its strand in panel (a) spans. Both coordinates
   are then read off the merged series rather than off that period's own two
   strands -- panel (a) and panel (c) change color a few years apart, and near
   year 140 N moves fast enough that honouring each panel's own handover leaves
   a visible break in the track right at its peak.
   """
   fig, ax = plt.subplots(figsize=(9, 7.4), layout="constrained")
   ax.axhline(0, color="0.4", lw=1.0, ls="--", zorder=1)

   years, dT, N = paired_N_T(curves)
   dT_smooth, N_smooth = running_mean(dT), running_mean(N)

   n_series = panel_series(curves, N_PANEL)
   for color in PERIODS:
      t_arr = curves[T_PANEL].get(color)
      if t_arr is None or not t_arr.size:
         continue
      lo = max(t_arr[0, 0], n_series[0, 0])
      hi = min(t_arr[-1, 0], n_series[-1, 0])
      idx = np.flatnonzero((years >= lo) & (years <= hi))
      if not idx.size:
         continue
      # One sample past the end, so consecutive periods butt together rather
      # than leaving a gap at the handover year.
      sl = slice(idx[0], min(idx[-1] + 2, years.size))
      ax.plot(dT[sl], N[sl], color=color, lw=0.7, alpha=0.22, zorder=2)
      ax.plot(dT_smooth[sl], N_smooth[sl], color=color, lw=1.8, zorder=4,
              label=PERIOD_STYLE[color])

   # One regression line per constant-forcing period, in that period's color:
   # the pair of them is the point of the plot, since the slope is the effective
   # feedback and it steepens between the two. Fitted on the 1-yr series, not
   # the running mean: the smoothing is there to make the track readable, and a
   # least-squares slope over 1000+ yr should not depend on it.
   color_of = {label: c for c, label in PERIOD_STYLE.items()}
   for label, F, lam, t_int, t_min in period_fits(years, dT, N):
      x = np.linspace(t_min, t_int, 100)
      ax.plot(x, F - lam * x, color=color_of[label], lw=1.2, ls=":", zorder=3,
              label=(f"fit over {label.split(' (')[0]}: "
                     f"$\\lambda$ = {lam:.2f} W m$^{{-2}}$ K$^{{-1}}$, "
                     f"$N$=0 at {t_int:.1f} K"))
      ax.plot([t_int], [0], marker="o", ms=5, mfc="none",
              color=color_of[label], zorder=5)

   ax.plot([dT[-1]], [N[-1]], marker="*", ms=13, color="black", zorder=6,
           label=f"end of run (yr {years[-1]:.0f}): "
                 f"{dT[-1]:.1f} K, {N[-1]:+.2f} W m$^{{-2}}$")
   ax.set_xlabel(PANEL_BY_KEY[T_PANEL].label, fontsize=14, fontweight="bold")
   ax.set_ylabel(PANEL_BY_KEY[N_PANEL].label, fontsize=14, fontweight="bold")
   ax.set_title("Net TOA flux against surface warming, ECHAM5/MPIOM 4xCO$_2$\n"
                "(digitized from Li et al. 2012, Fig. 1a and 1c; "
                f"{SMOOTH_YEARS}-yr running mean over the annual track)",
                fontsize=15, fontweight="bold")
   ax.grid(alpha=0.3)
   ax.tick_params(labelsize=12)
   # Below the axes rather than inside: the empty corner is the one the ramp's
   # peak passes through, and a 7-entry legend covers it wherever it is put.
   ax.legend(fontsize=9.5, ncol=2, loc="upper center",
             bbox_to_anchor=(0.5, -0.09), frameon=False)
   fig.savefig(str(png_path), dpi=160, bbox_inches="tight")
   plt.close(fig)


def plot_reconstruction(curves, png_path):
   """Redraw all five panels from the digitized points, as a fidelity check --
   if this doesn't overlay the original, the calibration constants are wrong."""
   fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 2.1 * len(PANELS) + 1),
                           sharex=True, layout="constrained", squeeze=False)
   for ax, p in zip(axs[:, 0], PANELS):
      if p.unit != "K":
         ax.axhline(0, color="black", lw=0.9, ls="--")
      for color in PERIODS:
         arr = curves[p.key].get(color)
         if arr is None or not arr.size:
            continue
         ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.4)
      ax.text(0.01, 0.93, f"({p.key})", transform=ax.transAxes,
              weight="bold", fontsize=13, va="top", ha="left")
      ax.set_ylabel(p.label, fontsize=12)
      ax.set_ylim((-1.0, 13.5) if p.unit == "K" else (-2.5, 4.5))
      ax.grid(alpha=0.3)
   axs[-1, 0].set_xlim(0, 6080)
   axs[-1, 0].set_xlabel("Time (years)", fontsize=14, fontweight="bold")
   fig.suptitle("Reconstruction of Li et al. 2012, Fig. 1", fontsize=17,
                fontweight="bold")
   fig.savefig(str(png_path), dpi=160, bbox_inches="tight")
   plt.close(fig)


def main():
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--pdf", default="data/li2012.pdf", help="path to the article PDF")
   ap.add_argument("--curves-out", default="data/li2012_fig1_curves.csv",
                   help="output CSV for every digitized point of every panel")
   ap.add_argument("--pair-out", default="data/li2012_fig1_N_vs_T.csv",
                   help="output CSV for the paired N-vs-T series")
   ap.add_argument("--plot-out", default="figures/li2012_fig1_N_vs_T.png",
                   help="output PNG for the N-vs-T plot")
   ap.add_argument("--full", action="store_true",
                   help="also redraw all five panels from the digitized points")
   ap.add_argument("--full-out", default="figures/li2012_fig1_reconstruction.png",
                   help="output PNG for --full")
   args = ap.parse_args()

   with pymupdf.open(args.pdf) as doc:
      page = doc[FIG1_PAGE]
      verify_calibration(page)
      curves = panel_curves(page)

   curves_out, pair_out = Path(args.curves_out), Path(args.pair_out)
   write_curves_csv(curves, curves_out)
   years, dT, N = paired_N_T(curves)
   write_pair_csv(years, dT, N, pair_out)

   plot_out = Path(args.plot_out)
   plot_out.parent.mkdir(parents=True, exist_ok=True)
   plot_n_vs_t(curves, plot_out)
   if args.full:
      full_out = Path(args.full_out)
      full_out.parent.mkdir(parents=True, exist_ok=True)
      plot_reconstruction(curves, full_out)

   print("Li et al. (2012) Fig. 1, digitized:")
   for p in PANELS:
      s = panel_series(curves, p.key)
      print(f"   ({p.key}) {p.quantity:14s} {s.shape[0]:5d} points, "
            f"yr {s[0, 0]:6.0f}-{s[-1, 0]:6.0f}, "
            f"{s[:, 1].min():6.2f} .. {s[:, 1].max():6.2f} {p.unit}")
   for label, F, lam, t_int, _t_min in period_fits(years, dT, N):
      print(f"   N = F - lambda*T over the {label:34s} "
            f"F = {F:5.2f} W m-2, lambda = {lam:5.3f} W m-2 K-1, "
            f"N=0 at {t_int:5.2f} K")
   print(f"   end of run (yr {years[-1]:.0f}): dT = {dT[-1]:.2f} K, "
         f"N = {N[-1]:+.2f} W m-2")
   print(f"-> {curves_out}")
   print(f"-> {pair_out}")
   print(f"-> {plot_out}")
   if args.full:
      print(f"-> {args.full_out}")


if __name__ == "__main__":
   main()
