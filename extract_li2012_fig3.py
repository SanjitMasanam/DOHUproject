#!/usr/bin/env python3
"""Digitize the ocean-temperature curves of Li et al. (2012), Figure 3.

Fig. 3 (p. 1077 of Clim Dyn 40:1071-1086) shows the annual-mean, globally
averaged ocean temperature change of the ECHAM5/MPIOM 4xCO2 run as six
vertically stacked panels, one per model level: 6, 1525, 2080, 3070, 4195 and
5170 m. The figure is embedded as vector art, not as an image, so the curves can
be read straight out of the PDF's drawing operators -- no pixel digitizing.

What the page actually contains:
  * six stroked rectangles, the panel frames, sharing x = 217.0 .. 555.8 pt and
    stacked in y (each 63.6 pt tall). The frames abut, so a panel's bottom edge
    is the next panel's top edge;
  * axis ticks as short 0.69-pt strokes: seven along the bottom frame (year 0 ..
    6000, 55.55 pt per 1000 yr) and five down each panel's left edge (0 .. 8 K,
    12.7 pt per 2 K, counting up from the frame bottom);
  * the plotted 11-yr running mean as stroked polylines in the paper's three
    period colors (navy transient, red quasi-equilibrium, firebrick
    equilibrium), plus a fourth blue used over the first years.
The gray annual-mean spaghetti behind the running mean shares its stroke width
with unrelated page art, so only the four running-mean colors are collected.

Run from the repo root:

    python3 extract_li2012_fig3.py [--pdf data/li2012.pdf] [--year 1000]

It writes two CSVs:
  * data/li2012_fig3_curves.csv (depth_m, depth_frac, year, dT_K) -- every
    digitized point of every curve, one row per point, the whole 0-6080 yr
    record per model level. This is what
    longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py
    overlays on its own ECHAM5 profile figure;
  * data/li2012_fig3_year<YEAR>.csv (depth_m, depth_frac, dT_K) -- the single
    year read off each curve, kept as the quick tabular summary the run prints.
Requires pymupdf; it is not part of the fit scripts' runtime dependency set, so
this stays a standalone tool.

Pass --full to additionally reconstruct the entire six-panel figure (every
point on every curve, not just the year-1000 reading) and save it as
figures/li2012_fig3_reconstruction.png -- a direct redraw of the paper's
Fig. 3 built from the same digitized points, as a visual check that the
digitization matches the original before it's trusted for the CSV above.
The shallowest panel additionally carries Fig. 1a (dT) and Fig. 1b (dSST),
digitized by extract_li2012_fig1.py, so the 6 m ocean level can be read against
the surface warming that drives it; --no-surface leaves them off.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from scipy.special import erfcx  # erfcx(x) = exp(x**2) erfc(x), overflow-free

from figure_paths import params_csv

# 0-based page index of Fig. 3 within the article PDF (printed page 1077).
FIG3_PAGE = 6

# Panel frames (top, bottom) in PDF points, and the model level each one plots.
# Taken from the six stroked rectangles on the page; they are listed here rather
# than re-detected so a stray rectangle elsewhere on the page cannot reorder the
# panels.
PANEL_BOUNDS = [
   (76.2, 139.8),
   (139.8, 203.4),
   (203.4, 267.0),
   (267.0, 330.7),
   (330.7, 394.3),
   (394.3, 457.9),
]
PANEL_DEPTHS_M = [6.0, 1525.0, 2080.0, 3070.0, 4195.0, 5170.0]
# Deepest level plotted; the fractions written to the CSV are relative to it, so
# they can be compared against a model column of any depth.
OCEAN_DEPTH_M = 5170.0

# (bounds, depth) pairs actually processed -- zip truncates to whichever of
# PANEL_BOUNDS/PANEL_DEPTHS_M is shorter, so trimming PANEL_DEPTHS_M to look at
# fewer levels (as above) automatically drops the trailing panels everywhere
# below (extraction, subplot count, CSV rows) instead of leaving them blank.
PANELS = list(zip(PANEL_BOUNDS, PANEL_DEPTHS_M))

# Axis calibration, all read off the tick strokes described above.
X_ORIGIN_PT = 217.0            # left frame edge = year 0
PT_PER_YEAR = 55.55 / 1000.0   # 55.55 pt between the 1000-yr ticks
K_PER_PT = 2.0 / 12.7          # 12.7 pt between the 2-K ticks

# Stroke colors of the plotted running mean, rounded to 3 decimals, mapped to
# the adjustment-period label the paper uses for that color elsewhere (Figs. 1,
# 5, 9, 11). Fig. 3 itself doesn't caption a legend, but the same four colors
# in the same left-to-right order appear there too.
PERIOD_STYLE = {
   (0.267, 0.38, 0.69): "CO$_2$ increase (1-140 yr)",
   (0.196, 0.149, 0.482): "transient (140-1200 yr)",
   (0.875, 0.192, 0.137): "quasi-equilibrium (1200-4600 yr)",
   (0.545, 0.102, 0.063): "equilibrium (4600-6080 yr)",
}
CURVE_COLORS = set(PERIOD_STYLE)

# Half-width of the x window whose points are median-ed to read one year off a
# curve. Wide enough to always catch a few vertices (the polylines step ~0.15 pt
# in x), narrow enough that the curve is locally straight: +/-0.6 pt is +/-11 yr.
SAMPLE_HALF_WIDTH_PT = 0.6

# --- Analytic semi-infinite diffusive-ocean overlay (surface response) -----
# Held's (GFDL blog #51, 2014) semi-infinite diffusive-slab solution, the same
# one used in held_exact_vs_approx.py, driven here by a forcing that ramps
# linearly 0 -> F_ref over RAMP_YEARS (the paper's "CO2 increase" period) and
# holds constant after. Duhamel's principle on the step response h(tau) gives
#     T(t) = C * (H(tau) - H(tau - RAMP_YEARS/T)),   tau = t/T
#     T = kappa*(rho_cp)**2/lambda**2,   C = F_ref*T/(lambda*RAMP_YEARS)
# using the ECHAM5 kappa_pde/lambda/F_ref fitted by the main pipeline
# (longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py).
# h(tau) is the response at the medium's zeta=0 boundary, i.e. the ocean
# surface, so this only overlays the shallowest (6 m) panel.
FITTED_PARAMS_VARIANT = "diffusive-eps-joint_fixedRadiativeParams"
FITTED_PARAMS_RUN_TYPE = 3      # 03_lrfit-50yravg
RHO_CP = 4.186e6                # vol. heat capacity of seawater [J m^-3 K^-1]
YEAR_S = 365.25 * 24 * 3600.0   # seconds per year
RAMP_YEARS = 140.0              # forcing ramp duration

pos = lambda x: np.maximum(np.asarray(x, float), 0.0)
H_exact = lambda tau: pos(tau) + 1 - 2 * np.sqrt(pos(tau) / np.pi) - erfcx(np.sqrt(pos(tau)))


def _load_diffusive_params(model="ECHAM5"):
   """(F_ref, lambda, kappa_pde) fitted for `model`, or None if unavailable."""
   try:
      csv_path = params_csv(FITTED_PARAMS_VARIANT, FITTED_PARAMS_RUN_TYPE)
   except FileNotFoundError:
      return None
   with csv_path.open(newline="") as fh:
      for row in csv.DictReader(fh):
         if row["model"] == model:
            return float(row["F_ref"]), float(row["lambda"]), float(row["kappa_pde"])
   return None


def ramp_response_years(t_years, F_ref, lam, kappa_pde, ramp_years=RAMP_YEARS, scale=1.0):
   """Surface temperature [K] of a semi-infinite diffusive ocean under a
   forcing ramping 0 -> F_ref over `ramp_years` then held constant, via the
   convolution-integral approach of held_exact_vs_approx.py's `response`.
   `scale` rescales the whole curve, e.g. to line it up with the digitized
   reconstruction at a chosen year (see `_end_of_run_dT` below)."""
   T_yr = kappa_pde * RHO_CP**2 / lam**2 / YEAR_S
   tau = np.asarray(t_years, float) / T_yr
   C = F_ref * T_yr / (lam * ramp_years)
   return scale * C * (H_exact(tau) - H_exact(tau - ramp_years / T_yr))


def _end_of_run_dT(curves, panel=0):
   """(year, dT) of the last digitized point in `panel`, taken from the final
   adjustment period (PERIOD_STYLE's last entry), used to calibrate the
   analytic overlay's amplitude to the reconstruction's end-of-run value."""
   end_key = list(PERIOD_STYLE)[-1]
   arr = curves[panel].get(end_key)
   if arr is None or arr.size == 0:
      return None
   i = np.argmax(arr[:, 0])
   return float(arr[i, 0]), float(arr[i, 1])


def _curve_points(page):
   """Points of every running-mean polyline, bucketed by panel index.

   A point sitting exactly on a shared frame edge belongs to the panel ABOVE
   (where it reads as 0 K), so the top edge is excluded from each panel's band
   and only the bottom edge is included. No curve comes near a panel top (the
   frames top out at 10 K, the warmest curve at ~8.9 K), so nothing is lost.
   """
   pts = {i: [] for i in range(len(PANELS))}
   for path in page.get_drawings():
      if path["type"] != "s" or path["color"] is None:
         continue
      if tuple(round(c, 3) for c in path["color"]) not in CURVE_COLORS:
         continue
      for item in path["items"]:
         if item[0] not in ("l", "c"):   # line segment or Bezier
            continue
         for pt in item[1:]:
            for i, ((top, bottom), _depth) in enumerate(PANELS):
               if top < pt.y <= bottom + 0.3:
                  pts[i].append((pt.x, pt.y))
   return {i: np.array(v, dtype=float) for i, v in pts.items()}


def _dedupe_by_x(points, xtol=0.05):
   """Collapse points that share (near-)identical x into one, y-averaged, point.

   Adjacent polyline segments of the same color share an endpoint, so the raw
   point cloud has near-duplicate x values at every segment boundary; sorted
   without merging they'd still plot fine as a line, but merging keeps the
   reconstructed curve to one y per x (matplotlib would otherwise draw a
   vertical hairline back and forth at every boundary when the two copies
   differ by a rounding fraction of a point).
   """
   order = np.argsort(points[:, 0])
   x, y = points[order, 0], points[order, 1]
   xs_out, ys_out = [], []
   i, n = 0, len(x)
   while i < n:
      j = i
      xsum, ysum, cnt = x[i], y[i], 1
      while j + 1 < n and x[j + 1] - x[i] <= xtol:
         j += 1
         xsum += x[j]; ysum += y[j]; cnt += 1
      xs_out.append(xsum / cnt)
      ys_out.append(ysum / cnt)
      i = j + 1
   return np.column_stack([xs_out, ys_out])


def full_curves(page):
   """Every digitized (year, dT_K) point on every curve, grouped by panel then
   by period color, each group sorted and merged into one strand by
   _dedupe_by_x. Returns {panel_idx: {color_key: Nx2 array}}."""
   pts = {i: {} for i in range(len(PANELS))}
   for path in page.get_drawings():
      if path["type"] != "s" or path["color"] is None:
         continue
      key = tuple(round(c, 3) for c in path["color"])
      if key not in CURVE_COLORS:
         continue
      for item in path["items"]:
         if item[0] not in ("l", "c"):
            continue
         for pt in item[1:]:
            for i, ((top, bottom), _depth) in enumerate(PANELS):
               if top < pt.y <= bottom + 0.3:
                  pts[i].setdefault(key, []).append((pt.x, pt.y))

   out = {}
   for i, ((_, bottom), _depth) in enumerate(PANELS):
      out[i] = {}
      for key, plist in pts[i].items():
         raw = np.array(plist, dtype=float)
         merged = _dedupe_by_x(raw)
         years = (merged[:, 0] - X_ORIGIN_PT) / PT_PER_YEAR
         dT = (bottom - merged[:, 1]) * K_PER_PT
         out[i][key] = np.column_stack([years, dT])
   return out


def curve_series(page):
   """One merged (year, dT_K) series per panel, top panel first.

   full_curves keeps each adjustment period as its own strand (they are drawn in
   different colors and dedupe within a strand only); the panel's actual record
   is all of them end to end, so they are concatenated, sorted by year and run
   through _dedupe_by_x again to collapse the shared endpoints where one period
   hands over to the next. The tolerance is the same 0.05 pt used in PDF space,
   converted to years so the two passes merge on the same scale.
   """
   curves = full_curves(page)
   out = []
   for i, (_bounds, _depth) in enumerate(PANELS):
      strands = [a for a in curves[i].values() if a.size]
      if not strands:
         raise RuntimeError(f"no curve points found in panel {i}")
      out.append(_dedupe_by_x(np.vstack(strands), xtol=0.05 / PT_PER_YEAR))
   return out


def write_curves_csv(page, out):
   """Long-format CSV of every digitized point: one row per (level, year)."""
   series = curve_series(page)
   out.parent.mkdir(parents=True, exist_ok=True)
   with out.open("w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["depth_m", "depth_frac", "year", "dT_K"])
      for (_bounds, depth), arr in zip(PANELS, series):
         for year, dT in arr:
            w.writerow([f"{depth:g}", f"{depth / OCEAN_DEPTH_M:.6f}",
                        f"{year:.2f}", f"{dT:.3f}"])
   return series


def surface_series(doc):
   """Fig. 1's surface temperature series, as [(label, Nx2 array, style), ...].

   Fig. 3 plots *ocean* temperature per model level: its shallowest panel (6 m)
   tops out near 8.8 K, well below the 10.8 K the paper quotes for the run's
   surface warming, because that 6 m curve is a global ocean mean and Fig. 1a is
   a global surface-air mean over land and ocean alike. Drawing 1a and 1b on top
   of the 6 m panel makes that offset visible instead of leaving it looking like
   a calibration error.

   Imported inside the function on purpose: extract_li2012_fig1 imports the
   color map and _dedupe_by_x from this module, so a module-level import here
   would be a cycle.
   """
   from extract_li2012_fig1 import FIG1_PAGE, panel_curves, panel_series

   page = doc[FIG1_PAGE]
   curves = panel_curves(page)
   return [
      (r"Fig. 1a  $\Delta T$ (global surface)", panel_series(curves, "a"),
       dict(color="0.15", lw=1.5, ls="-")),
      (r"Fig. 1b  $\Delta$SST (sea surface)", panel_series(curves, "b"),
       dict(color="0.45", lw=1.5, ls="-.")),
   ]


def plot_reconstruction(page, png_path, surface=None):
   """Redraw Fig. 3 from the digitized points: one stacked panel per entry in
   PANELS, one curve per adjustment period per panel, colored to match the
   paper. This is a fidelity check on the digitization, not a new figure --
   if it doesn't overlay the original almost exactly, the calibration
   constants above (panel bounds, axis origin/scale) are wrong.

   `surface`, from surface_series(), is drawn on the shallowest panel for scale:
   the ocean's 6 m level against the surface temperature driving it.
   """
   curves = full_curves(page)
   diffusive_params = _load_diffusive_params()
   overlay_scale = 1.0
   if diffusive_params is None:
      print(f"warning: no parameter table for {FITTED_PARAMS_VARIANT} run type "
            f"{FITTED_PARAMS_RUN_TYPE}, skipping analytic overlay")
   else:
      end = _end_of_run_dT(curves, panel=0)
      if end is not None:
         end_year, end_dT = end
         analytic_end = ramp_response_years(end_year, *diffusive_params)
         overlay_scale = end_dT / analytic_end

   fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 2.33 * len(PANELS) + 1),
                           sharex=True, layout="constrained", squeeze=False)
   axs = axs[:, 0]
   for i, (ax, (_bounds, depth)) in enumerate(zip(axs, PANELS)):
      for key in PERIOD_STYLE:
         arr = curves[i].get(key)
         if arr is None or arr.size == 0:
            continue
         ax.plot(arr[:, 0], arr[:, 1], color=key, lw=1.6)
      if i == 0 and surface:
         for label, arr, style in surface:
            ax.plot(arr[:, 0], arr[:, 1], label=label, zorder=4, **style)
      if i == 0 and diffusive_params is not None:
         F_ref, lam, kappa_pde = diffusive_params
         t_analytic = np.linspace(0, 6080, 2000)
         ax.plot(t_analytic,
                 ramp_response_years(t_analytic, F_ref, lam, kappa_pde),
                 color="0.5", lw=1.4, ls=":",
                 label="semi-infinite diffusive analytic\n"
                       "(unscaled)")
         ax.plot(t_analytic,
                 ramp_response_years(t_analytic, F_ref, lam, kappa_pde, scale=overlay_scale),
                 color="black", lw=1.8, ls="--",
                 label="semi-infinite diffusive analytic\n"
                       "(scaled to match at yr 6080)\n"
                       r"(ECHAM5 $\kappa,\lambda$, ramp to 140 yr)")
      ax.text(0.02, 0.95, f"{depth:g} m", transform=ax.transAxes,
              weight="bold", fontsize=14, va="top", ha="left")
      ax.set_ylim(0, 15)
      ax.grid(alpha=0.3)
      ax.tick_params(labelsize=12)
   # Below the panel: with the surface curves added, every in-axes corner
   # either holds a curve or the steep first-century rise.
   axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
                 fontsize=9, frameon=False)
   axs[-1].set_xlim(0, 6080)
   axs[-1].set_xlabel("Time (years)", fontsize=14, fontweight="bold")
   fig.suptitle("Global Mean Ocean Temperature Change [K]\n"
                "(reconstruction of Li et al. 2012, Fig. 3)",
                fontsize=18, fontweight="bold")
   fig.savefig(str(png_path), dpi=160, bbox_inches="tight")
   plt.close(fig)


def temperatures_at(page, year):
   """Digitized dT [K] at `year` for each panel in PANELS, top panel first."""
   pts = _curve_points(page)
   x_target = X_ORIGIN_PT + year * PT_PER_YEAR
   out = []
   for i, ((_, bottom), _depth) in enumerate(PANELS):
      a = pts[i]
      if a.size == 0:
         raise RuntimeError(f"no curve points found in panel {i}")
      window = a[np.abs(a[:, 0] - x_target) <= SAMPLE_HALF_WIDTH_PT]
      if window.size == 0:
         raise RuntimeError(f"year {year} is outside the plotted range of panel {i}")
      out.append((bottom - float(np.median(window[:, 1]))) * K_PER_PT)
   return out


def main():
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--pdf", default="data/li2012.pdf", help="path to the article PDF")
   ap.add_argument("--year", type=float, default=1000.0,
                   help="year of the run to read off each curve (default 1000)")
   ap.add_argument("--out", default=None,
                   help="output CSV (default data/li2012_fig3_year<YEAR>.csv)")
   ap.add_argument("--curves-out", default="data/li2012_fig3_curves.csv",
                   help="output CSV for the full year-by-year curves "
                        "(default data/li2012_fig3_curves.csv)")
   ap.add_argument("--full", action="store_true",
                   help="also reconstruct the full 6-panel figure from every "
                        "digitized point and save it to --full-out")
   ap.add_argument("--full-out", default="figures/li2012_fig3_reconstruction.png",
                   help="output PNG for --full "
                        "(default figures/li2012_fig3_reconstruction.png)")
   ap.add_argument("--no-surface", action="store_true",
                   help="leave Fig. 1a/1b off the --full reconstruction's "
                        "shallowest panel (they are drawn there for scale)")
   args = ap.parse_args()

   year = args.year
   out = Path(args.out) if args.out else Path("data") / f"li2012_fig3_year{year:g}.csv"

   with pymupdf.open(args.pdf) as doc:
      page = doc[FIG3_PAGE]
      temps = temperatures_at(page, year)
      curves_out = Path(args.curves_out)
      series = write_curves_csv(page, curves_out)
      if args.full:
         full_out = Path(args.full_out)
         full_out.parent.mkdir(parents=True, exist_ok=True)
         surface = None if args.no_surface else surface_series(doc)
         plot_reconstruction(page, full_out, surface=surface)
         print(f"-> {full_out}")

   out.parent.mkdir(parents=True, exist_ok=True)
   with out.open("w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["depth_m", "depth_frac", "dT_K"])
      for depth, dT in zip(PANEL_DEPTHS_M, temps):
         w.writerow([f"{depth:g}", f"{depth / OCEAN_DEPTH_M:.6f}", f"{dT:.3f}"])
   print(f"Li et al. (2012) Fig. 3 at year {year:g}:")
   for (depth, dT), arr in zip(zip(PANEL_DEPTHS_M, temps), series):
      print(f"   {depth:6.0f} m   dT = {dT:5.2f} K   "
            f"({arr.shape[0]} curve points, yr {arr[0, 0]:.0f}-{arr[-1, 0]:.0f})")
   print(f"-> {curves_out}")
   print(f"-> {out}")


if __name__ == "__main__":
   main()
