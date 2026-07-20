#!/usr/bin/env python3

import rpy2.robjects as ro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sympy as sp
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator

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
   # if xspacing: ax.xaxis.set_major_locator(MultipleLocator(xspacing))
   # if yspacing: ax.yaxis.set_minor_locator(MultipleLocator(yspacing))

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
   sharex=False,
   sharey=False,
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
      sharex=sharex,
      sharey=sharey,
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


for run_type in [3, 2, 1]:
   for results in ["unblinded", "validation"]:
      dir_list = [
         "geoffroy_replicate_results",
         "50-yr_avg_forcing_results",
         "50-yr_avg_tau_s_LR_fit_results",
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
         """
         expr: SymPy expression
         values: dict of {symbol: value}
         uncertainties: dict of {symbol: uncertainty}
         """
         variance = 0.0
         for sym, unc in uncertainties.items():
            deriv = sp.diff(expr, sym)
            deriv_val = float(deriv.evalf(subs=values))
            variance += (deriv_val * unc) ** 2
         return float(np.sqrt(variance))

      def ensure_dirs(outdir, current_dir, sections):
         for section in sections:
            (outdir / current_dir / section / "png").mkdir(parents=True, exist_ok=True)

      # Create dataframe to store model parameters
      param_cols = [
         "model",
         "C",
         "C0",
         "gamma",
         "tau_f",
         "tau_s",
         "F_ref",
         "lambda",
         "T_eq",
         "a_f",
         "a_s",
         "C_unc",
         "C0_unc",
         "gamma_unc",
         "tau_f_unc",
         "tau_s_unc",
         "F_ref_unc",
         "lambda_unc",
         "T_eq_unc",
         "a_f_unc",
         "a_s_unc",
      ]
      df = pd.DataFrame(columns=param_cols)

      # Load the .Rdata file into the R global environment
      rdata_file = Path("./data/int_netToa_longrun.Rdata")
      ro.r["load"](str(rdata_file))

      # Read models and experiments into Python lists
      data = ro.globalenv["int_nettoa_longrun_data"]
      models = ['CCSM3', 'CESM1', 'CNRMCM6', 'ECHAM5', 'GISSE2R', 'IPSLCM5A', 'HadGEM2', 'MPIESM11']
      expts = list(ro.globalenv["expts"])
      df["model"] = models

      # Make output directory if needed
      outdir = Path("./figures_2013a")
      outdir.mkdir(exist_ok=True)
      ensure_dirs(outdir, current_dir, ["step1", "step2", "validation", "unblinded"])

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

      # Prepare combined figures for each experiment
      step1_figs = {}
      step1_axs = {}
      step1_idx = {}
      step2_figs = {}
      step2_axs = {}
      step2_idx = {}
      final_figs = {}
      final_axs = {}
      final_idx = {}
      final_xmax = {}
      tau_s_figs = {}
      tau_s_axs = {}
      tau_s_idx = {}
      nettoa_figs = {}
      nettoa_axs = {}
      nettoa_idx = {}
      ohc_ts_figs = {}
      ohc_ts_axs = {}
      ohc_ts_idx = {}
      assmpt_figs = {}
      assmpt_axs = {}
      assmpt_idx = {}

      # Create shared figures for the 4xCO2 experiment and a 10-panel Net TOA layout
      for expt in ['4xCO2']:
         step1_figs[expt], step1_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs T$_{2M}$", xlabel="2-meter Air Temperature Anomaly (K)", ylabel=r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)")
         step1_idx[expt] = 0

         step2_figs[expt], step2_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ log(T$_{eq}$-T) - log(T$_{eq}$) vs. Time", xlabel=r"Time (years)", ylabel=r"log($T_{eq}$-T)-log($T_{eq}$)")
         step2_idx[expt] = 0

         final_figs[expt], final_axs[expt] = make_model_grid(models, dpi=120, title=r"4xCO$_{2}$ T$_{2M}$ vs. Time w/ 2-Box Fit", xlabel=r"Time (years)", ylabel=r"Temperature Anomaly (K)", right=0.95, wspace=0.28)
         final_idx[expt] = 0
         final_xmax[expt] = []
         final_figs[expt].text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", rotation=-90.)

         nettoa_figs[expt], nettoa_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs. Time", xlabel="Time (years)", ylabel=r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)")
         nettoa_idx[expt] = 0

         tau_s_figs[expt], tau_s_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ $\tau_s$ vs. Calibration Time", xlabel=r"Calibration Time (years)", ylabel=r"$\tau_s$ (years)", sharex=True, sharey=True)
         tau_s_idx[expt] = 0

         ohc_ts_figs[expt], ohc_ts_axs[expt] = make_model_grid(models, title=r"4xCO$_2$ OHU vs. Surface Temp (Normalized)", xlabel=r"$T_s/2\, \mathrm{ECS}$", ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$")
         ohc_ts_idx[expt] = 0

         assmpt_figs[expt], assmpt_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Assumption 2 Test", xlabel="Time (years)", ylabel=r"$C_{u}\frac{dT_{u}}{dt}$")
         assmpt_idx[expt] = 0

      # ----------------- STEP 1 ---------------------

      for model in models:
         # Extract model data from the R dataset
         model_data = data.rx2(model)
         t2m_mean = 0.0
         nettoa_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Flatten T2M and NETTOA arrays from R data
            if run_type == 1:
               t2m = np.array(expt_data.rx2("T2M")).ravel()[:150]
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:150]
            else:
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

               # For non-150-year runs, average the data into 50-year bins before fitting
               if run_type != 1:
                  tmp_t2m_list = []
                  tmp_nettoa_list = []
                  for i in range(50, t2m.shape[0], 50):
                     tmp_t2m_list.append(np.mean(t2m[i : i + 50]))
                     tmp_nettoa_list.append(np.mean(nettoa[i : i + 50]))
                  t2m = np.array(tmp_t2m_list)
                  nettoa = np.array(tmp_nettoa_list)

               # Fit a linear relationship between T2M and NETTOA
               [m, b], cov = np.polyfit(t2m, nettoa, 1, cov=True)
               m_unc = np.sqrt(cov[0, 0])
               b_unc = np.sqrt(cov[1, 1])

               # Build the best-fit line for plotting
               xfit = np.linspace(t2m.min(), t2m.max(), 100)
               yfit = m * xfit + b

               # Draw the Step 1 scatter + fit on the shared figure
               ax = step1_axs[expt][step1_idx[expt]]
               ax.scatter(t2m, nettoa, s=8, alpha=0.5, label="Data")
               ax.plot(xfit, yfit, linewidth=2, label=f"Fit: F_ref={b:.3f}, -lambda={m:.3f}")

               # Compute and annotate R^2 and RMSE for the fit on this panel
               y_pred = m * t2m + b
               ss_res = np.sum((nettoa - y_pred) ** 2)
               ss_tot = np.sum((nettoa - np.mean(nettoa)) ** 2)
               r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
               rmse = np.sqrt(np.mean((nettoa - y_pred) ** 2))
               ax.text(
                  0.02,
                  0.93,
                  f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )

               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear")
               step1_idx[expt] += 1

               # Define symbolic expressions for uncertainty propagation
               sp_m, sp_b = sp.symbols("m b")
               F_expr = sp_b
               lambda_expr = -sp_m
               T_eq_expr = sp_b / (-sp_m)

               # Compute uncertainties via propagation
               F_ref_unc = sympy_prop_unc(F_expr, {sp_b: b}, {sp_b: b_unc})
               lambda_unc = sympy_prop_unc(lambda_expr, {sp_m: m}, {sp_m: m_unc})
               T_eq_unc = sympy_prop_unc(
                  T_eq_expr,
                  {sp_m: m, sp_b: b},
                  {sp_m: m_unc, sp_b: b_unc},
               )

               # Record 4xCO2 fit results and uncertainties
               if expt == "4xCO2":
                  df.loc[df["model"] == model, "F_ref"] = b
                  df.loc[df["model"] == model, "lambda"] = -m
                  df.loc[df["model"] == model, "T_eq"] = b / (-m)
                  df.loc[df["model"] == model, "F_ref_unc"] = F_ref_unc
                  df.loc[df["model"] == model, "lambda_unc"] = lambda_unc
                  df.loc[df["model"] == model, "T_eq_unc"] = T_eq_unc

      # Save the combined Step 1 and Net TOA timeseries figures
      for expt in ["4xCO2"]:
         step1_figs[expt].savefig(
            outdir / current_dir / "step1" / "png" / f"{expt}_all_models_T2M_vs_NETTOA{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step1_figs[expt])

      print("Finished Step 1: saved combined Step 1 figures & params to df")

      # ----------------- STEP 2 ---------------------

      for model in models:
         # Extract model data for Step 2
         model_data = data.rx2(model)
         t2m_first10_mean = 0.0
         t2m_mean = 0.0
         nettoa_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Prepare the first 150-year T2M and NETTOA series
            t2m_first10 = np.array(expt_data.rx2("T2M")).ravel()[0:10]
            if run_type != 3:
               t2m = np.array(expt_data.rx2("T2M")).ravel()[30:151]
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[30:151]
            else:
               t2m = np.array(expt_data.rx2("T2M")).ravel()[50:]
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[50:]

            # Filter non-nan data
            quality_filter = np.isfinite(t2m) & np.isfinite(nettoa)
            if np.sum(quality_filter) / len(quality_filter) != 1:
               print(f"{model} {expt} valid/all years:", np.sum(quality_filter) / len(quality_filter))
            t2m = t2m[quality_filter]
            nettoa = nettoa[quality_filter]

            # Remove piControl baseline from Step 2 variables
            if expt == "piControl":
               t2m_first10_mean = np.mean(t2m_first10)
               t2m_mean = np.mean(t2m)
               nettoa_mean = np.mean(nettoa)
            elif expt == "4xCO2":
               t2m_first10 = t2m_first10 - t2m_first10_mean
               t2m = t2m - t2m_mean
               nettoa = nettoa - nettoa_mean

               T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]

               # Linear regression with safe mask for log
               if run_type != 3:
                  t = np.arange(30, 30 + t2m.shape[0], 1)
               else:
                  t = np.arange(50, 50 + t2m.shape[0], 1)

               mask = ((T_eq - t2m) > 0)
               t = t[mask]
               t2m = t2m[mask]

               if run_type == 3:
                  # Create tau_s vs. run length plot
                  tau_s_lst = []
                  for i in range(151, t2m.shape[0]):
                     # Get T2M/NETTOA for desired calibration length
                     t2m_tauRun = np.array(expt_data.rx2("T2M")).ravel()[:i] - t2m_mean
                     nettoa_tauRun = nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:i] - nettoa_mean
                     t_tauRun = np.arange(0, t2m_tauRun.shape[0], 1)

                     # Fit for T_eq with calb. length (DOESN'T MATCH ACTUAL STEP 1)
                     [m_step1, b_step1] = np.polyfit(t2m_tauRun, nettoa_tauRun, 1)
                     T_eq_tauRun = b_step1/(-m_step1)

                     # Apply mask for safety, then calculate tau_s
                     mask_tauRun = ((T_eq_tauRun - t2m_tauRun) > 0)
                     t_tauRun = t_tauRun[mask_tauRun]
                     t2m_tauRun = t2m_tauRun[mask_tauRun]

                     y_tauRun = np.log(T_eq_tauRun - t2m_tauRun[30:]) - np.log(T_eq_tauRun)
                     [m, b] = np.polyfit(t_tauRun[30:], y_tauRun, 1)
                     tau_s_lst.append(-1/m)

                  ax = tau_s_axs[expt][tau_s_idx[expt]]
                  ax.plot(np.arange(151, t2m.shape[0], 1), tau_s_lst)
                  ax.plot(np.arange(151,  t2m.shape[0], 1), np.arange(151,  t2m.shape[0], 1), label='y=x', color='black')


                  fitted_params_path1 = Path('/home/Sanjit.Masanam/Documents/DeepOceanHeatUptakeProject/2013a_figures/geoffroy_replicate_results/fitted_model_params.csv')
                  fitted_params_path2 = Path('/home/Sanjit.Masanam/Documents/DeepOceanHeatUptakeProject/2013a_figures/50-yr_avg_forcing_results/fitted_model_params.csv')
                  fitted_params_path3 = Path('/home/Sanjit.Masanam/Documents/DeepOceanHeatUptakeProject/2013a_figures/50-yr_avg_tau_s_LR_fit_results/fitted_model_params.csv')

                  if fitted_params_path1.is_file() and fitted_params_path2.is_file() and fitted_params_path3.is_file():
                     df_runType1 = pd.read_csv(fitted_params_path1)
                     df_runType2 = pd.read_csv(fitted_params_path2)
                     df_runType3 = pd.read_csv(fitted_params_path3)

                     tau_s_runType1 = df_runType1.loc[df['model'] == model, "tau_s"].iloc[0]
                     tau_s_runType2 = df_runType2.loc[df['model'] == model, "tau_s"].iloc[0]
                     tau_s_runType3 = df_runType3.loc[df['model'] == model, "tau_s"].iloc[0]

                     ax.scatter(151, tau_s_runType1, s=14, color='red', label=r'Geoffroy 2013a')
                     ax.scatter(151, tau_s_runType2, s=14, color='yellow', label=r'50-yr Avg T$_{eq}$')
                     ax.scatter(t2m.shape[0], tau_s_runType3, s=14, color='green', label=r'50-yr Avg + LR Fit')

                  # How close this model actually got to equilibrium by the end
                  # of its own run (last-10-yr mean T2M anomaly / T_eq), shown as
                  # extra text under the model-name label.
                  T2M_raw_full = np.array(expt_data.rx2("T2M")).ravel()
                  eq_ratio = float((np.mean(T2M_raw_full[-10:]) - t2m_mean) / T_eq)

                  format_ax(ax, text=f"{model}\nReached {eq_ratio * 100:.0f}%\nof $T_{{eq}}$", xscale="linear", yscale="linear", legend_loc='upper right')
                  tau_s_idx[expt] += 1

               # Fit step 2
               y = np.log(T_eq - t2m) - np.log(T_eq)
               [m, b], cov = np.polyfit(t, y, 1, cov=True)
               m_unc = np.sqrt(cov[0, 0])
               b_unc = np.sqrt(cov[1, 1])
               xfit = t
               yfit = m * xfit + b

               # Compute and annotate R^2 and RMSE for the Step 2 fit
               ss_res = np.sum((y - yfit) ** 2)
               ss_tot = np.sum((y - np.mean(y)) ** 2)
               r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
               rmse = np.sqrt(np.mean((y - yfit) ** 2))

               # Draw Step 2 fit on the combined figure
               ax = step2_axs[expt][step2_idx[expt]]
               ax.scatter(t, y, s=8, alpha=0.5, label="Data")
               ax.plot(xfit, yfit, linewidth=2, label=f"Fit: log(a_s)={b:.3f}, -1/t_s={m:.3f}")
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", xspacing=False, yspacing=False, legend_loc='lower left')
               ax.text(
                  0.02,
                  0.93,
                  f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               step2_idx[expt] += 1

               # Compute 2-box parameters and save them
               lmbda = df.loc[df["model"] == model, "lambda"].iloc[0]
               tau_s = -1 / m
               a_s = np.exp(b)
               a_f = 1 - a_s
               t_first10 = np.arange(1, 11, 1)

               tau_f_values = t_first10 / (
                  np.log(a_f)
                  - np.log(1 - (t2m_first10 / T_eq) - a_s * np.exp(-t_first10 / tau_s))
               )
               tau_f = np.mean(tau_f_values) # NOTE: tau_f estimate is approximate

               C = lmbda / ((a_f / tau_f) + (a_s / tau_s))
               C0 = lmbda * (tau_f * a_f + tau_s * a_s) - C
               gamma = C0 / (tau_f * a_s + tau_s * a_f)

               # Define var expressions
               sp_lambda, sp_a_s, sp_tau_s, sp_tau_f, sp_m, sp_b = sp.symbols(
                  "lambda a_s tau_s tau_f m b"
               )
               tau_s_expr = -1 / sp_m
               a_s_expr = sp.exp(sp_b)
               sp_a_f = 1 - sp_a_s
               C_expr = sp_lambda / ((sp_a_f / sp_tau_f) + (sp_a_s / sp_tau_s))
               C0_expr = sp_lambda * (sp_tau_f * sp_a_f + sp_tau_s * sp_a_s) - C_expr
               gamma_expr = C0_expr / (sp_tau_f * sp_a_s + sp_tau_s * sp_a_f)

               # Compute uncertainties for derived parameters
               lambda_unc = df.loc[df["model"] == model, "lambda_unc"].iloc[0]
               tau_s_unc = sympy_prop_unc(tau_s_expr, {sp_m: m}, {sp_m: m_unc})
               tau_f_unc = np.std(tau_f_values)
               a_s_unc = sympy_prop_unc(a_s_expr, {sp_b: b}, {sp_b: b_unc})
               a_f_unc = sympy_prop_unc(sp_a_f, {sp_a_s: a_s}, {sp_a_s: a_s_unc})
               C_unc = sympy_prop_unc(
                  C_expr,
                  {sp_lambda: lmbda, sp_a_s: a_s, sp_tau_s: tau_s, sp_tau_f: tau_f},
                  {sp_lambda: lambda_unc, sp_a_s: a_s_unc, sp_tau_s: tau_s_unc, sp_tau_f: tau_f_unc},
               )
               C0_unc = sympy_prop_unc(
                  C0_expr,
                  {sp_lambda: lmbda, sp_a_s: a_s, sp_tau_s: tau_s, sp_tau_f: tau_f},
                  {sp_lambda: lambda_unc, sp_a_s: a_s_unc, sp_tau_s: tau_s_unc, sp_tau_f: tau_f_unc},
               )
               gamma_unc = sympy_prop_unc(
                  gamma_expr,
                  {sp_lambda: lmbda, sp_a_s: a_s, sp_tau_s: tau_s, sp_tau_f: tau_f},
                  {sp_lambda: lambda_unc, sp_a_s: a_s_unc, sp_tau_s: tau_s_unc, sp_tau_f: tau_f_unc},
               )

               # Save vars to df
               df.loc[df["model"] == model, "C"] = C
               df.loc[df["model"] == model, "C0"] = C0
               df.loc[df["model"] == model, "gamma"] = gamma
               df.loc[df["model"] == model, "tau_f"] = tau_f
               df.loc[df["model"] == model, "tau_s"] = tau_s
               df.loc[df["model"] == model, "a_f"] = a_f
               df.loc[df["model"] == model, "a_s"] = a_s

               df.loc[df["model"] == model, "C_unc"] = C_unc
               df.loc[df["model"] == model, "C0_unc"] = C0_unc
               df.loc[df["model"] == model, "gamma_unc"] = gamma_unc
               df.loc[df["model"] == model, "tau_f_unc"] = tau_f_unc
               df.loc[df["model"] == model, "tau_s_unc"] = tau_s_unc
               df.loc[df["model"] == model, "a_f_unc"] = a_f_unc
               df.loc[df["model"] == model, "a_s_unc"] = a_s_unc

      # Write the combined Step 2 figures to disk
      for expt in ["4xCO2"]:
         step2_figs[expt].savefig(
            outdir / current_dir / "step2" / "png" / f"{expt}_all_models_log_Teq_minus_T_vs_t{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(step2_figs[expt])

         # Explicitly span every panel to the full extent of all plotted data
         # (across all models) rather than relying on sharex/sharey autoscale,
         # so the longest run's data is never clipped in any panel.
         populated_axs = [ax for ax in tau_s_axs[expt] if ax.has_data()]
         if populated_axs:
            xmax = max(ax.dataLim.intervalx[1] for ax in populated_axs)
            ymax = max(ax.dataLim.intervaly[1] for ax in populated_axs)
            for ax in tau_s_axs[expt]:
               ax.set_xlim(0, xmax + 100)
               ax.set_ylim(0, ymax + 100)
               # Denser tick marks than the default locator gives.
               ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
               ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))

         tau_s_figs[expt].savefig(
            outdir / current_dir / "step2" / "png" / f"{expt}_all_models_tau_s_vs_calibration_t{suffix}.png",
            dpi=200,
            bbox_inches="tight",
         )
         plt.close(tau_s_figs[expt])

      print("Finished Step 2: saved combined Step 2 figures and all parameters to df")

      # Save fitted model parameter table to CSV
      df.to_csv(outdir / current_dir / "fitted_model_params.csv", index=False)

      # ---------------- Compare fitted parameters against published paper values ----------------
      model_paperParams = {
         "GISSE2R": {
            "F_ref": 7.3,
            "lambda": 1.70,
            "T_eq": 4.3,
            "C": 4.7,
            "C0": 126,
            "gamma": 1.16,
            "tau_f": 1.6,
            "tau_s": 184,
         },
         "HadGEM2": {
            "F_ref": 5.9,
            "lambda": 0.65,
            "T_eq": 9.1,
            "C": 6.5,
            "C0": 82,
            "gamma": 0.55,
            "tau_f": 5.3,
            "tau_s": 280,
         },
         "IPSLCM5A": {
            "F_ref": 6.4,
            "lambda": 0.79,
            "T_eq": 8.1,
            "C": 7.7,
            "C0": 95,
            "gamma": 0.59,
            "tau_f": 5.5,
            "tau_s": 286,
         },
         "MPIESM11": {
            "F_ref": 8.2,
            "lambda": 1.14,
            "T_eq": 7.3,
            "C": 7.3,
            "C0": 71,
            "gamma": 0.72,
            "tau_f": 3.9,
            "tau_s": 164,
         },
      }

      colors = [
         ("#ef9a9a", "#b71c1c"),  # red
         ("#90caf9", "#1565c0"),  # blue
         ("#a5d6a7", "#2e7d32"),  # green
         ("#ffcc80", "#ef6c00"),  # orange
      ]

      validation_vars = list(model_paperParams["GISSE2R"].keys())
      nvars = len(validation_vars)

      # Build a grid of validation plots, one panel per parameter
      fig_val, axs_val = make_model_grid(
         validation_vars,
         title=rf"{expt} Geoffroy vs. Sanjit/Nadir (w/ 95% CI)",
         ylabel="A.U.",
         ncols=5
      )

      for ivar, var in enumerate(validation_vars):
         ax = axs_val[ivar]
         handles = []
         tmp_mu_SN_list = []

         for model, (face, edge) in zip(model_paperParams.keys(), colors):
            mu_GF = model_paperParams[model][var]
            mu_SN = df.loc[df["model"] == model, f"{var}"].iloc[0]
            mu_unc = df.loc[df["model"] == model, f"{var}_unc"].iloc[0]

            ax.errorbar(
               mu_SN,
               mu_GF,
               xerr=mu_unc * 1.96,
               marker=".",
               ms=13,
               mfc=face,
               mec=edge,
               ecolor=edge,
               linewidth=1.5,
               zorder=3,
            )
            tmp_mu_SN_list.append(mu_SN)
            tmp_mu_SN_list.append(mu_GF)
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

         one_to_one = np.arange(min(tmp_mu_SN_list), max(tmp_mu_SN_list) + 0.001, 0.001)
         ax.plot(one_to_one, one_to_one, "k--", label=r"$\mu_{\rm GF}=\mu_{\rm SN}$")
         format_ax(ax, text=f"{var}", xlabel=r"$\mu_{\rm SN}$", xscale="linear", yscale="linear", legend=False)
         ax.legend(handles=handles, loc='lower left', prop={'weight': 'bold', 'size': 10})

      fig_val.savefig(
         outdir / current_dir / "validation" / "png" / f"all_validation_params_GF_vs_SN{suffix}.png",
         dpi=200,
         bbox_inches="tight",
      )
      plt.close(fig_val)

      # ----------------- Plot final results ---------------------

      for model in models:
         # Extract model data for plotting results
         model_data = data.rx2(model)
         t2m_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Flatten T2M for the chosen results window
            if results == "validation":
               t2m = np.array(expt_data.rx2("T2M")).ravel()[:151]
               nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:151]
            else:
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
               C = df.loc[df["model"] == model, "C"].iloc[0]

               T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]
               a_f = df.loc[df["model"] == model, "a_f"].iloc[0]
               a_s = df.loc[df["model"] == model, "a_s"].iloc[0]
               tau_f = df.loc[df["model"] == model, "tau_f"].iloc[0]
               tau_s = df.loc[df["model"] == model, "tau_s"].iloc[0]

               T_eq_unc = df.loc[df["model"] == model, "T_eq_unc"].iloc[0]
               a_s_unc = df.loc[df["model"] == model, "a_s_unc"].iloc[0]
               tau_f_unc = df.loc[df["model"] == model, "tau_f_unc"].iloc[0]
               tau_s_unc = df.loc[df["model"] == model, "tau_s_unc"].iloc[0]

               # Use uncertainty in the fitted parameters to sample possible model curves
               iterations = 1000
               param_mean = np.array([T_eq, a_s, tau_f, tau_s])
               param_cov = np.diag(np.array([T_eq_unc**2, a_s_unc**2, tau_f_unc**2, tau_s_unc**2]))
               params = np.random.multivariate_normal(param_mean, param_cov, size=iterations)

               # Compute the fitted temperature curve & nettoa/OHC
               t = np.arange(1, 1 + t2m.shape[0], 1)
               T = T_eq * (a_f * (1 - np.exp(-t / tau_f)) + a_s * (1 - np.exp(-t / tau_s)))
                  
               # Plot OHC vs. T_s
               if results == 'unblinded':
                  T_ohc = T_eq * (a_f * (1 - np.exp(-np.arange(1, 1 + 100000, 1) / tau_f)) + a_s * (1 - np.exp(-np.arange(1, 1 + 100000, 1) / tau_s)))
                  N_pred = F_ref - lmbda * T_ohc
                  normalized_OHC_pred = (5.1e14 * 31536000 * np.cumsum(N_pred))/(1.37e21 * 3850)
                  normalized_OHC = (5.1e14 * 31536000 * np.cumsum(nettoa))/(1.37e21 * 3850)
                     
                  [m_ohcTs, b_ohcTs] = np.polyfit(t2m[:5]/T_eq, normalized_OHC[:5]/T_eq, 1)
                  t_val = np.arange(0, 1.1, 0.1)
                     

                  cmap = plt.cm.turbo
                  norm = mpl.colors.Normalize(vmin=0, vmax=6000)
                  A = 2 * tcr[model] / T_eq
                  ax = ohc_ts_axs[expt][ohc_ts_idx[expt]]
                  sc = ax.scatter(t2m/T_eq, normalized_OHC/T_eq, c=np.arange(1, 1+normalized_OHC.shape[0], 1), cmap=cmap, norm=norm)
                  ax.plot(t_val,m_ohcTs*t_val + b_ohcTs, ls='--', color='red', label=f'Mixed Layer Depth = {(m_ohcTs*2500):.0f} m')
                  ax.plot(t_val, (t_val-A)/(1-A), ls='--', color='black', label='2-box Asymptotic Pred.')
                  ax.plot(T_ohc/T_eq,normalized_OHC_pred/T_eq, color='green', label=f'Fitted 2-Box Pred.')
                  ax.axvline(1.0, color="0.55", ls='--', lw=0.8)
                  ax.axvline(A, color="0.55", ls='--', lw=0.8)
                  format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", ylim=(-0.05, 1.2))
                  ohc_ts_idx[expt] += 1

               # Draw observed and fitted temperature curves
               ax = final_axs[expt][final_idx[expt]]
               ax.scatter(t, t2m, s=4, color="red")
               ax.plot(t, t2m, color="red", label="2-m Surface Temp.")
               ax.plot(t, T, color="blue", label="2-Box Fit")

               # Build ensemble of valid fits from parameter draws and compute mean/std
               T_ensemble = []
               for i in range(iterations):
                  T_eq_i, a_s_i, tau_f_i, tau_s_i = params[i]
                  if tau_f_i <= 0 or tau_s_i <= 0 or a_s_i <= 0 or a_s_i >= 1:
                     continue
                  T_ensemble.append(
                     T_eq_i
                     * (
                        (1 - a_s_i) * (1 - np.exp(-t / tau_f_i))
                        + a_s_i * (1 - np.exp(-t / tau_s_i))
                     )
                  )

               if len(T_ensemble) > 0:
                  T_ensemble = np.vstack(T_ensemble)
                  T_mean = T_ensemble.mean(axis=0)
                  T_std = T_ensemble.std(axis=0)

                  # Plot 2-sigma and 1-sigma shaded bands around the fit mean
                  ax.fill_between(t, T_mean - 2 * T_std, T_mean + 2 * T_std, color="blue", alpha=0.08, label="±2σ")
                  ax.fill_between(t, T_mean - T_std, T_mean + T_std, color="blue", alpha=0.2, label="±1σ")

               # Add the slow-timescale parameter and a reference line at 150 years.
               ax.text(
                  0.02,
                  0.93,
                  rf"$\tau_s$ = {tau_s:.1f} yr",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=EXTRA_TEXT_FONTSIZE,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               ax.axvline(150, color="orange", linestyle=":", linewidth=1, alpha=0.7)

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
               final_xmax[expt].append(np.max(t))

               # Secondary y-axis showing T/T_eq (Equilibrium Ratio) on the same scale;
               # ratio ticks are set to the same fractions used for the left axis so that
               # gridlines/ticks line up exactly (e.g. T_eq <-> 1, 0.8*T_eq <-> 0.8, ...)
               ax2 = ax.twinx()
               ax2.set_ylim(y_bottom / T_eq, y_top / T_eq)
               ax2.set_yticks(ratio_ticks)
               ax2.tick_params(labelsize=14, width=2, length=8, direction="in")

               final_idx[expt] += 1

               # Determine validity of assmption (2): c_u~0
               F_thresh = F_ref/100
               t2m_rollingMu_lst = []
               nettoa_rollingMu_lst = []
               for i in range(t.shape[0]-8):
                  t2m_rollingMu_lst.append(np.mean(t2m[i:i+10]))
                  nettoa_rollingMu_lst.append(np.mean(nettoa[i:i+10]))
               t2m_rollingMu = np.array(t2m_rollingMu_lst)
               nettoa_rollingMu = np.array(nettoa_rollingMu_lst)
               dT_dt_2box = np.gradient(T)
               dT_dt_true = np.gradient(t2m_rollingMu)

               ax = assmpt_axs[expt][assmpt_idx[expt]]
               ax.plot(t[:dT_dt_true.shape[0]],C*dT_dt_true, color='black', label=f'AOGCM (10 yr running mean)')
               ax.plot(t,C*dT_dt_2box, ls='--', color='blue', label='2-box')
               ax.axhline(F_thresh, color="0.55", ls='--', lw=0.8, label='Order of Mag. Threshold')
               format_ax(ax, text=f"{model}", xscale='log', yscale='linear')
               assmpt_idx[expt] += 1

               # Plot nettoa timeseries
               ax = nettoa_axs[expt][nettoa_idx[expt]]
               ax.plot(t[:nettoa_rollingMu.shape[0]], nettoa_rollingMu, color='black', label='AOGCM')
               ax.plot(t[:nettoa_rollingMu.shape[0]], F_ref - lmbda * t2m_rollingMu, label='50-yr Avg')
               if results == 'unblinded': ax.plot(t, F_ref - lmbda * T, label='2-box (50-yr Avg + LR Fit)')
               format_ax(ax, text=f"{model}", xscale="linear", yscale="log", legend_loc='lower left')
               nettoa_idx[expt] += 1

      # Save combined final figures in both linear and log x-scale variants
      for expt in ['4xCO2']:
         for scale in ["linear", "log"]:
            for ax, xmax in zip(final_axs[expt], final_xmax[expt]):
               ax.set_xscale(scale)
               ax.set_xlim(1, xmax + 1)
               if scale == "linear":
                  ax.set_xticks(np.linspace(1, xmax + 1, 5))
               else:
                  ax.xaxis.set_major_locator(mpl.ticker.LogLocator())

            for ax, xmax in zip(nettoa_axs[expt], final_xmax[expt]):
               ax.set_xscale(scale)
               ax.set_xlim(1, xmax + 1)

            final_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_T2m_vs_t_{results}_{scale}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )

            nettoa_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_NETTOA_timeseries_{scale}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )

         plt.close(final_figs[expt])
         plt.close(nettoa_figs[expt])

         if results == 'unblinded':
            assmpt_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_cdTdt_t_{results}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            plt.close(assmpt_figs[expt])

            # Saved in step 1
            cbar = ohc_ts_figs[expt].colorbar(sc, ax=ohc_ts_axs[expt].ravel().tolist(), fraction=0.025, pad=0.025)
            cbar.set_label("Year")

            ohc_ts_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_ohc_ts{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            plt.close(ohc_ts_figs[expt])

      print("Finished final val/result plots")


