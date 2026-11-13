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

The input can be the full bundle or the per-model checkpoints the R script writes
as it goes (data/cmip_1pctCO2_checkpoints/<label>.rds), so a partial run can be
plotted before every model has finished. Without --data, the bundle is used if it
exists, else the checkpoint directory.

Usage (from the repo root):
    python3 plot_1pctCO2_nettoa_vs_T.py
    python3 plot_1pctCO2_nettoa_vs_T.py --test CM3,CESM1
    python3 plot_1pctCO2_nettoa_vs_T.py --data /path/to/int_netToa_1pctCO2.Rdata
    python3 plot_1pctCO2_nettoa_vs_T.py --data data/cmip_1pctCO2_checkpoints       # every checkpoint
    python3 plot_1pctCO2_nettoa_vs_T.py --data data/cmip_1pctCO2_checkpoints/CM3.rds

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
CKPT_DIR = Path("./data/cmip_1pctCO2_checkpoints")

# Panel order for checkpoints (the bundle has its own): cmip_runs in get_int_nettoa_longrun_data.R.
MODEL_ORDER = ["CCSM3", "CESM1", "CNRMCM6", "ECEARTH", "ECHAM5", "GISSE2R", "IPSLCM5A",
               "MIROC3", "HadGEM2", "MPIESM11", "CM3", "ESM2M", "ESM2G"]

EXPT = "1pctCO2"
N_YEARS = 100            # every run is cut here
WINDOW = 10              # rolling-mean length [yr]
KAPPA_MIN_T = 0.2        # K; kappa = N/T is masked below this smoothed T
KAPPA_YEARS = (61, 80)   # kappa reference window around CO2 doubling (yr 70)

T2M_LABEL = "2-meter Air Temperature Anomaly (K)"
NETTOA_LABEL = r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)"
KAPPA_LABEL = r"$\kappa = N/T$ ($W\,m^{-2}K^{-1}$)"


# ---- data ----

def model_entry(md):
   """One model's R list (a bundle entry or a checkpoint's $data) as plain arrays."""
   d = {ex: {v: as_1d_float(md.rx2(ex).rx2(v)) for v in ("T2M", "NETTOA")}
        for ex in ("piControl", EXPT)}
   try:
      d["experiment"] = str(md.rx2(EXPT).rx2("experiment")[0])
   except Exception:
      d["experiment"] = EXPT
   return d


def load_1pct_data(path=None):
   """{model: {"piControl"/"1pctCO2": {"T2M", "NETTOA"}, "experiment": str}}.

   `path` is the .Rdata bundle, one checkpoint .rds, or a directory of them; None picks
   the bundle if it exists, else the checkpoint directory.
   """
   if path is None:
      path = RDATA_FILE if RDATA_FILE.exists() else CKPT_DIR
   path = Path(path)
   if not path.exists():
      raise SystemExit(f"{path} not found. Build the data on the GFDL analysis machine with\n"
                       f"   Rscript get_int_nettoa_longrun_data.R cmip_1pctCO2\n"
                       f"and copy {RDATA_FILE} or {CKPT_DIR}/ here, or point --data at it.")
   import rpy2.robjects as ro  # deferred: needs a system R

   if path.is_dir() or path.suffix.lower() == ".rds":
      files = sorted(path.glob("*.rds")) if path.is_dir() else [path]
      if not files:
         raise SystemExit(f"No checkpoint (.rds) files in {path}.")
      rank = {m: i for i, m in enumerate(MODEL_ORDER)}
      files.sort(key=lambda f: (rank.get(f.stem, len(rank)), f.stem))
      print(f"Reading {len(files)} checkpoint(s) from {path}")
      return {f.stem: model_entry(ro.r["readRDS"](str(f)).rx2("data")) for f in files}

   print(f"Reading bundle {path}")
   ro.r["load"](str(path))
   bundle = ro.globalenv["int_nettoa_1pctCO2_data"]
   return {str(m): model_entry(bundle.rx2(m)) for m in bundle.names}


def anomalies(model, d, n_years=N_YEARS):
   """(T, N) of the first n_years of 1pctCO2 minus the means of the first n_years of
   piControl, finite years only, with the matching simulation years (1-based). The bundle
   already holds n_years; checkpoints saved before max_years existed may hold more."""
   pi_T, pi_N = d["piControl"]["T2M"][:n_years], d["piControl"]["NETTOA"][:n_years]
   pi_ok = np.isfinite(pi_T) & np.isfinite(pi_N)
   with np.errstate(invalid="ignore"):   # R's NA is a NaN payload numpy warns on; masked below
      T = d[EXPT]["T2M"][:n_years] - np.mean(pi_T[pi_ok])
      N = d[EXPT]["NETTOA"][:n_years] - np.mean(pi_N[pi_ok])
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
   ap.add_argument("--data", "--rdata", dest="data", type=Path, default=None,
                   help=f"the .Rdata bundle, one checkpoint .rds, or a directory of checkpoints "
                        f"(default: {RDATA_FILE} if it exists, else {CKPT_DIR})")
   add_test_argument(ap)
   args = ap.parse_args()

   data = load_1pct_data(args.data)
   all_models = list(data)
   models = select_models(all_models, args.test)
   announce_test_mode(models, all_models, args.test)
   if not models:
      raise SystemExit("No models in the input.")

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
