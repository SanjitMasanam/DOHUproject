#!/usr/bin/env python3
"""Net TOA vs T2M for every 1pctCO2 run, with the heat-uptake efficiency beneath.

Reads the bundle that `Rscript get_int_nettoa_longrun_data.R cmip_1pctCO2` writes
(data/int_netToa_1pctCO2.Rdata) and draws one (main, ratio) panel pair per model,
in the layout of the 4xCO2 `T2M_vs_NETTOA` step1 grid:

  main    N vs T: annual anomalies as a faded scatter, with their 10-yr rolling
          mean drawn on top.
  ratio   kappa = N/T of the rolling means: the transient ocean heat-uptake
          efficiency (Gregory & Mitchell 1997). It is masked where the smoothed T
          is below KAPPA_MIN_T, because N/T is noise over a near-zero denominator in the
          first decade. The dotted line is kappa over years 61-80, the window
          around CO2 doubling, computed as a ratio of means of the annual data.

Anomalies are taken against the piControl time mean, as perturbed_anomalies does
for the 4xCO2 plots. Every run is cut at N_YEARS = 100, which also keeps the
CMIP3 1pctto4x runs on their ramp (CO2 reaches 4x at yr ~139).

Usage (from the repo root):
    python3 plot_1pctCO2_nettoa_vs_T.py
    python3 plot_1pctCO2_nettoa_vs_T.py --test CM3,CESM1
    python3 plot_1pctCO2_nettoa_vs_T.py --rdata /path/to/int_netToa_1pctCO2.Rdata

Output: figures/1pctCO2/1pctCO2_all_models_T2M_vs_NETTOA_kappa.png
"""


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_paths import DPI_GRID, figname, topic_dir
from longrun_common import add_test_argument, announce_test_mode, as_1d_float, select_models
from plot_style import EXTRA_TEXT_FONTSIZE, format_ax, make_model_grid_with_ratio

# Relative on purpose: run from the repo root, like the longrun scripts.
RDATA_FILE = Path("./data/int_netToa_1pctCO2.Rdata")

EXPT = "1pctCO2"
N_YEARS = 100            # every run is cut here
WINDOW = 10              # rolling-mean length [yr]
KAPPA_MIN_T = 0.2        # K; kappa = N/T is masked below this smoothed T
KAPPA_YEARS = (61, 80)   # kappa reference window around CO2 doubling (yr 70)

T2M_LABEL = "2-meter Air Temperature Anomaly (K)"
NETTOA_LABEL = r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)"
KAPPA_LABEL = r"$\kappa = N/T$ ($W\,m^{-2}K^{-1}$)"


# ---- data ----

def load_1pct_data(rdata_file):
   """{model: {"piControl"/"1pctCO2": {"T2M", "NETTOA"}, "experiment": str}}, in bundle order."""
   if not Path(rdata_file).exists():
      raise SystemExit(f"{rdata_file} not found. Build it on the GFDL analysis machine with\n"
                       f"   Rscript get_int_nettoa_longrun_data.R cmip_1pctCO2\n"
                       f"and copy it here, or point --rdata at it.")
   import rpy2.robjects as ro  # deferred: needs a system R

   ro.r["load"](str(rdata_file))
   bundle = ro.globalenv["int_nettoa_1pctCO2_data"]
   out = {}
   for model in bundle.names:
      md = bundle.rx2(model)
      d = {ex: {v: as_1d_float(md.rx2(ex).rx2(v)) for v in ("T2M", "NETTOA")}
           for ex in ("piControl", EXPT)}
      try:
         d["experiment"] = str(md.rx2(EXPT).rx2("experiment")[0])
      except Exception:
         d["experiment"] = EXPT
      out[str(model)] = d
   return out


def anomalies(model, d, n_years=N_YEARS):
   """(T, N) of the first n_years of 1pctCO2 minus the piControl means, finite years only,
   with the matching simulation years (1-based)."""
   pi = d["piControl"]
   pi_ok = np.isfinite(pi["T2M"]) & np.isfinite(pi["NETTOA"])
   with np.errstate(invalid="ignore"):   # R's NA is a NaN payload numpy warns on; masked below
      T = d[EXPT]["T2M"][:n_years] - np.mean(pi["T2M"][pi_ok])
      N = d[EXPT]["NETTOA"][:n_years] - np.mean(pi["NETTOA"][pi_ok])
   years = np.arange(1, T.shape[0] + 1)
   ok = np.isfinite(T) & np.isfinite(N)
   if T.shape[0] < n_years:
      print(f"   {model}: only {T.shape[0]} yr of {EXPT} (wanted {n_years})")
   if not ok.all():
      print(f"   {model}: dropping {np.sum(~ok)} non-finite year(s)")
   return years[ok], T[ok], N[ok]


def rolling_mean(a, window=WINDOW):
   """Running mean over `window` points (valid part only); the input unchanged if shorter."""
   a = np.asarray(a, dtype=float)
   if window <= 1 or a.shape[0] < window:
      return a
   return np.convolve(a, np.ones(window) / window, mode="valid")


def kappa_window(years, T, N, window=KAPPA_YEARS):
   """kappa over the reference years as a ratio of means; NaN if the run is too short."""
   sel = (years >= window[0]) & (years <= window[1])
   if not sel.any():
      return np.nan
   return float(np.mean(N[sel]) / np.mean(T[sel]))


# ---- plotting ----

def plot_model(ax, ratio_ax, model, d):
   years, T, N = anomalies(model, d)
   T_roll, N_roll = rolling_mean(T), rolling_mean(N)
   kappa = np.where(T_roll >= KAPPA_MIN_T, N_roll / np.where(T_roll == 0, np.nan, T_roll), np.nan)
   k_ref = kappa_window(years, T, N)

   ax.scatter(T, N, s=8, alpha=0.35, color="tab:blue", label="Data (annual)")
   ax.plot(T_roll, N_roll, color="tab:orange", lw=2, label=f"{WINDOW}-yr rolling mean")
   format_ax(ax, text=model, legend_loc="upper right", legend_fontsize=9)
   ax.text(0.02, 0.92,
           f"{d['experiment']}, {years[-1]} yr\n"
           rf"$\kappa_{{{KAPPA_YEARS[0]}-{KAPPA_YEARS[1]}}}$={k_ref:.2f}",
           transform=ax.transAxes, va="top", ha="left", fontsize=EXTRA_TEXT_FONTSIZE,
           bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.4))

   ratio_ax.plot(T_roll, kappa, color="green", lw=1.5, marker="o", ms=3)
   if np.isfinite(k_ref):
      ratio_ax.axhline(k_ref, color="0.5", lw=1.0, ls=":")
   ratio_ax.tick_params(labelsize=12, width=2, length=6, direction="in")
   ratio_ax.grid(alpha=0.3)
   return k_ref


def main():
   ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
   ap.add_argument("--rdata", type=Path, default=RDATA_FILE,
                   help=f"1pctCO2 bundle (default {RDATA_FILE})")
   add_test_argument(ap)
   args = ap.parse_args()

   data = load_1pct_data(args.rdata)
   all_models = list(data)
   models = select_models(all_models, args.test)
   announce_test_mode(models, all_models, args.test)
   if not models:
      raise SystemExit(f"No models in {args.rdata}.")

   fig, axs, ratio_axs = make_model_grid_with_ratio(
      models, title=rf"1pctCO$_2$ Net TOA vs T$_{{2M}}$ (first {N_YEARS} yr)",
      xlabel=T2M_LABEL, ylabel=NETTOA_LABEL, ratio_ylabel=KAPPA_LABEL)

   print(f"kappa = N/T over years {KAPPA_YEARS[0]}-{KAPPA_YEARS[1]} [W m-2 K-1]:")
   for i, model in enumerate(models):
      k_ref = plot_model(axs[i], ratio_axs[i], model, data[model])
      print(f"   {model:10s} {k_ref:6.3f}")

   out = topic_dir("1pctCO2") / f"{figname(EXPT, 'all_models', 'T2M_vs_NETTOA_kappa')}.png"
   fig.savefig(out, dpi=DPI_GRID, bbox_inches="tight")
   plt.close(fig)
   print("Saved", out)


if __name__ == "__main__":
   main()
