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
from scipy.special import erfcx
from scipy.optimize import curve_fit

for run_type in [3, 1]:
   for results in ["validation"]:#, "unblinded"]:
      dir_list = [
         "replicate_results",
         "N/A",
         "50-yr_avg_LR_fit_results",
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


      def format_ax(ax, title="", xlabel="", ylabel="", text="",
                  xscale="linear", yscale="linear",
                  xlim=None, ylim=None,
                  xticks=None, yticks=None,
                  xspacing=True, yspacing=True,
                  legend=True, legend_loc='upper right', grid=True):

         ax.set(title=title, xlabel=xlabel, ylabel=ylabel,
               xscale=xscale, yscale=yscale,
               xlim=xlim, ylim=ylim)

         if text:
            ax.text(0.02, 0.98, text, transform=ax.transAxes, weight='bold',
                     fontsize=15, va="top", ha="left",
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4))

         if xticks is not None: ax.set_xticks(xticks)
         if yticks is not None: ax.set_yticks(yticks)

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
            constrained_layout=False,
         )

         fig.subplots_adjust(left=0.05, right=right, bottom=0.075, top=0.95, wspace=wspace, hspace=hspace)

         if title:
            fig.suptitle(title, fontsize=20, fontweight="bold")

         if xlabel:
            fig.text(0.5, 0.02, xlabel, ha='center', fontsize=18, fontweight="bold")

         if ylabel:
            fig.text(0.02, 0.5, ylabel, ha='center', va='center', fontsize=18, fontweight="bold", rotation=90.)

         return fig, np.asarray(axs).ravel()


      def ensure_dirs(outdir, current_dir, sections):
         for section in sections:
            (outdir / current_dir / section / "png").mkdir(parents=True, exist_ok=True)
            (outdir / current_dir / section / "pdf").mkdir(parents=True, exist_ok=True)

      # Create dataframe to store model parameters
      param_cols = [
         "model",
         "tau",
         "F_ref",
         "lambda",
         "T_eq",
         "a_f",
         "a_s",
         "tau_unc",
         "F_ref_unc",
         "lambda_unc",
         "T_eq_unc",
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
      outdir = Path("./diffusive")
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

         final_figs[expt], final_axs[expt] = make_model_grid(models, dpi=120, title=r"4xCO$_{2}$ T$_{2M}$ vs. Time w/ Diffusive Fit", xlabel=r"Time (years)", ylabel=r"Temperature Anomaly (K)", right=0.95, wspace=0.28)
         final_idx[expt] = 0
         final_xmax[expt] = []
         final_figs[expt].text(0.975, 0.5, "Equilibrium Ratio", ha='center', va='center', fontsize=18, fontweight="bold", rotation=-90.)

         nettoa_figs[expt], nettoa_axs[expt] = make_model_grid(models, title=r"4xCO$_{2}$ Net TOA vs. Time", xlabel="Time (years)", ylabel=r"Net TOA (10 yr rolling mean, $W\,m^{-2}$)")
         nettoa_idx[expt] = 0

         ohc_ts_figs[expt], ohc_ts_axs[expt] = make_model_grid(models, title=r"4xCO$_2$ OHU vs. Surface Temp (Normalized)", xlabel=r"$T_s/2\, \mathrm{ECS}$", ylabel=r"$\mathrm{OHC}/\mathrm{OHC}_{eq}$")
         ohc_ts_idx[expt] = 0

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
                  0.88,
                  f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=8,
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
         step1_figs[expt].savefig(
            outdir / current_dir / "step1" / "pdf" / f"{expt}_all_models_T2M_vs_NETTOA{suffix}.pdf",
            bbox_inches="tight",
         )
         plt.close(step1_figs[expt])

      print("Finished Step 1: saved combined Step 1 figures & params to df")

      # ----------------- Plot final results ---------------------

      for model in models:
         # Extract model data for plotting results
         model_data = data.rx2(model)
         t2m_mean = 0.0

         for expt in expts:
            expt_data = model_data.rx2(expt)

            # Flatten T2M for the chosen results window
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

               T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]
               T_eq_unc = df.loc[df["model"] == model, "T_eq_unc"].iloc[0]

               def fit_func_diff(x, D):
                  return 1-erfcx(x/D * (lmbda**2)/(((1025)*(3993))**2))

               def fit_func_ode(x, tau):
                  return 1-np.exp(-x/tau)

               if results == "validation" and run_type == 1: 
                  fit_T = t2m[:20]/T_eq
                  plot_T = t2m[:151]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "validation" and run_type == 3: 
                  fit_T = t2m/T_eq
                  plot_T = t2m[300:]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(300, 300 + plot_T.shape[0], 1)
               if results == "unblinded" and run_type == 1: 
                  fit_T = t2m[:20]/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "unblinded" and run_type == 3: 
                  fit_T = t2m/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)

               # Compute the fitted temperature curve & nettoa/OHC
               popt, pcov = curve_fit(fit_func_diff, fit_t, fit_T, p0=[5*10**(-5)])
               perr = np.sqrt(np.diag(pcov))
               print(popt, perr)

               D = popt[0]
               D_unc = perr[0]

               popt, pcov = curve_fit(fit_func_ode, fit_t, fit_T)
               perr = np.sqrt(np.diag(pcov))
               print(popt, perr)

               tau = popt[0]
               tau_unc = perr[0]
                  
               # # Plot OHC vs. T_s
               # if results == 'unblinded':
               #    T_ohc = T_eq * (a_f * (1 - np.exp(-np.arange(1, 1 + 100000, 1) / tau_f)) + a_s * (1 - np.exp(-np.arange(1, 1 + 100000, 1) / tau_s)))
               #    N_pred = F_ref - lmbda * T_ohc
               #    normalized_OHC_pred = (5.1e14 * 31536000 * np.cumsum(N_pred))/(1.37e21 * 3850)
               #    normalized_OHC = (5.1e14 * 31536000 * np.cumsum(nettoa))/(1.37e21 * 3850)
                     
               #    [m_ohcTs, b_ohcTs] = np.polyfit(t2m[:5]/T_eq, normalized_OHC[:5]/T_eq, 1)
               #    t_val = np.arange(0, 1.1, 0.1)
                     

               #    cmap = plt.cm.turbo
               #    norm = mpl.colors.Normalize(vmin=0, vmax=6000)
               #    A = 2 * tcr[model] / T_eq
               #    ax = ohc_ts_axs[expt][ohc_ts_idx[expt]]
               #    sc = ax.scatter(t2m/T_eq, normalized_OHC/T_eq, c=np.arange(1, 1+normalized_OHC.shape[0], 1), cmap=cmap, norm=norm)
               #    ax.plot(t_val,m_ohcTs*t_val + b_ohcTs, ls='--', color='red', label=f'Mixed Layer Depth = {(m_ohcTs*2500):.0f} m')
               #    ax.plot(t_val, (t_val-A)/(1-A), ls='--', color='black', label='2-box Asymptotic Pred.')
               #    ax.plot(T_ohc/T_eq,normalized_OHC_pred/T_eq, color='green', label=f'Fitted 2-Box Pred.')
               #    ax.set_ylim(-0.05, 1.2)
               #    ax.axvline(1.0, color="0.55", ls='--', lw=0.8)
               #    ax.axvline(A, color="0.55", ls='--', lw=0.8)
               #    ax.set_title(f"{model} {expt}:OHU vs. Surface Temp (Normalized)")
               #    ax.set_xlabel(r"$\frac{T_s}{2\, \mathrm{ECS}}$")
               #    ax.set_ylabel(r"$\frac{\mathrm{OHC}}{\mathrm{OHC}_{eq}}$")
               #    ax.legend(fontsize=8, loc='upper left')
               #    ohc_ts_idx[expt] += 1

               # Draw observed and fitted temperature curves
               ax = final_axs[expt][final_idx[expt]]
               ax.scatter(plot_t, T_eq*plot_T, s=4, color="red")
               ax.plot(plot_t, T_eq*plot_T, color="red", label="2-m Surface Temp.")
               ax.plot(plot_t, T_eq*fit_func_diff(plot_t, D), color="blue", label="Diffusive Fit")
               ax.plot(plot_t, T_eq*fit_func_ode(plot_t, tau), color="green", label="ODE Fit")

               # Add the slow-timescale parameter and a reference line at 150 years.
               ax.text(
                  0.02,
                  0.88,
                  rf"D = {D:.2e} $m^2/s$",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=8,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               ax.axvline(151, color="orange", linestyle=":", linewidth=1, alpha=0.7)

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
               final_xmax[expt].append(np.max(plot_t))

               # Secondary y-axis showing T/T_eq (Equilibrium Ratio) on the same scale;
               # ratio ticks match the left axis's fractions so gridlines line up exactly.
               ax2 = ax.twinx()
               ax2.set_ylim(y_bottom / T_eq, y_top / T_eq)
               ax2.set_yticks(ratio_ticks)
               ax2.tick_params(labelsize=14, width=2, length=8, direction="in")

               final_idx[expt] += 1

               # # Plot nettoa timeseries
               # ax = nettoa_axs[expt][nettoa_idx[expt]]
               # ax.plot(t[:nettoa_rollingMu.shape[0]], nettoa_rollingMu, color='black', label='AOGCM')
               # ax.plot(t[:nettoa_rollingMu.shape[0]], F_ref - lmbda * t2m_rollingMu, label='50-yr Avg')
               # if results == 'unblinded': ax.plot(t, F_ref - lmbda * T_eq * fit_func(plot_t, D), label='Diffusive (50-yr Avg + LR Fit)')
               # ax.set_title(f"{model} {expt} Net TOA (W m^-2)")
               # ax.set_xlabel("Time (years)")
               # ax.set_ylabel("Net TOA (10 yr rolling mean)")
               # ax.set_xscale(scale)
               # ax.set_yscale("log")
               # ax.grid(True)
               # ax.legend(fontsize=8, loc='upper right')
               # nettoa_idx[expt] += 1

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

            final_figs[expt].savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_T2m_vs_t_{results}_{scale}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            final_figs[expt].savefig(
               outdir / current_dir / results / "pdf" / f"{expt}_all_models_T2m_vs_t_{results}_{scale}{suffix}.pdf",
               bbox_inches="tight",
            )

         plt.close(final_figs[expt])

         # nettoa_figs[expt].savefig(
         #    outdir / current_dir / results / "png" / f"{expt}_all_models_NETTOA_timeseries_{scale}{suffix}.png",
         #    dpi=200,
         #    bbox_inches="tight",
         # )
         # nettoa_figs[expt].savefig(
         #    outdir / current_dir / results / "pdf" / f"{expt}_all_models_NETTOA_timeseries_{scale}{suffix}.pdf",
         #    bbox_inches="tight",
         # )
         # plt.close(nettoa_figs[expt])

         # if results == 'unblinded':
         #    cbar = ohc_ts_figs[expt].colorbar(sc, ax=ohc_ts_axs[expt].ravel().tolist(), fraction=0.025, pad=0.025)
         #    cbar.set_label("Year")

         #    ohc_ts_figs[expt].savefig(
         #       outdir / current_dir / results / "png" / f"{expt}_all_models_ohc_ts{suffix}.png",
         #       dpi=200,
         #       bbox_inches="tight",
         #    )
         #    ohc_ts_figs[expt].savefig(
         #       outdir / current_dir / results / "pdf" / f"{expt}_all_models_ohc_ts{suffix}.pdf",
         #       bbox_inches="tight",
         #    )
         #    plt.close(ohc_ts_figs[expt])

      print("Finished final val/result plots")


