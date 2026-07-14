#!/usr/bin/env python3

import time
import multiprocessing
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
from pde_solver_1lyrEBM_diffusive import Params as PDEParams, solve_model as pde_solve_model, YEAR

# "Purely diffusive" limit of the 1-box + diffusive model: h_ml -> 0 makes
# the mixed-layer heat capacity vanish, which is singular in the PDE solver
# (division by c_ml = rho_cp*h_ml), so approximate the limit with a tiny but
# numerically well-behaved mixed layer instead of exactly zero.
H_ML_PURELY_DIFFUSIVE = 1.0e-6  # m

# Sensitivity sweep of the fitted kappa/h_ml around each model's best fit.
SENSITIVITY_KAPPA_FACTOR_RANGE = (0.2, 5.0)   # multiplicative range around kappa_pde
SENSITIVITY_HML_FACTOR_RANGE = (0.25, 50.0)     # multiplicative range around h_ml_pde
SENSITIVITY_N_1D = 13                         # points in each 1-D spaghetti sweep
SENSITIVITY_N_2D = 9                          # points per axis in the joint grid (N_2D^2 solves/model)
SENSITIVITY_KAPPA_BOUNDS = (1e-7, 1e-3)       # matches the curve_fit bounds below
SENSITIVITY_HML_BOUNDS = (10.0, 2000.0)

# Diagnostic-only solver settings for the sensitivity sweep: looser tolerance
# and fewer saved points than the actual kappa_pde/h_ml_pde fit (n_save=250,
# rtol=1e-8), since these solves only feed spaghetti/heatmap plots and never
# touch the fitted parameters themselves.
SENSITIVITY_N_SAVE = 100
SENSITIVITY_RTOL = 1e-6
SENSITIVITY_ATOL = 1e-8

# Progress/ETA tracking across the whole script (all run_types x results x models).
RUN_TYPES = [1, 2, 3]
RESULT_KINDS = ["validation", "unblinded"]
_SCRIPT_START_TIME = time.time()


def _sensitivity_solve_dT(kappa, h_ml, F_ref, T_eq, t_final_years, t_eval):
   """PDE surface response for one sensitivity-sweep (kappa, h_ml) point.

   Mirrors the per-point evaluation inside the main curve_fit target
   (make_pde_dT below), but at loosened tolerance/n_save and as a
   module-level function rather than a closure, so it can be dispatched to
   the multiprocessing pool -- sweep points only populate diagnostic plots,
   never the fitted kappa_pde/h_ml_pde themselves.
   """
   z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_years * YEAR)))
   nz = int(min(401, max(121, z_max / 15.0 + 1)))
   pde_p = PDEParams(
      kappa=kappa, h_ml=h_ml, dT_eq=T_eq, F0=F_ref,
      z_max=z_max, Nz=nz, t_final=t_final_years * YEAR,
   )
   sol = pde_solve_model(pde_p, n_save=SENSITIVITY_N_SAVE, rtol=SENSITIVITY_RTOL, atol=SENSITIVITY_ATOL)
   return np.interp(t_eval, sol["t"] / YEAR, sol["dT"])


# Sensitivity-sweep points are independent (kappa, h_ml) solves, so farm them
# out across cores instead of running all ~100/model sequentially. Opened
# once and reused for the whole script; closed at the very end.
_SENS_POOL = multiprocessing.get_context("fork").Pool()

# Load the .Rdata file once and reuse it across every run_type/results
# iteration below -- the source data never changes between iterations.
rdata_file = Path("./data/int_netToa_longrun.Rdata")
ro.r["load"](str(rdata_file))

data = ro.globalenv["int_nettoa_longrun_data"]
models = ['CCSM3', 'CESM1', 'CNRMCM6', 'ECHAM5', 'GISSE2R', 'IPSLCM5A', 'HadGEM2', 'MPIESM11']
expts = list(ro.globalenv["expts"])

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

# Output directory (created once; per-run_type subdirs are made inside the loop)
outdir = Path("./figures_diffusive")
outdir.mkdir(exist_ok=True)

for run_type in RUN_TYPES:
   for results in RESULT_KINDS:
      dir_list = [
         "replicate_results",
         "50-yr_avg_forcing_results",
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
         """Propagate uncertainties through a SymPy expression (first-order, uncorrelated)."""
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
      df["model"] = models

      ensure_dirs(outdir, current_dir, ["step1", "validation", "unblinded"])

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
      sens_kappa_figs = {}
      sens_kappa_axs = {}
      sens_kappa_idx = {}
      sens_hml_figs = {}
      sens_hml_axs = {}
      sens_hml_idx = {}
      sens_t63_figs = {}
      sens_t63_axs = {}
      sens_t63_idx = {}
      sens_rmse_figs = {}
      sens_rmse_axs = {}
      sens_rmse_idx = {}
      # Artists accumulated across the model loop so each sensitivity figure can
      # be put on a single figure-wide color scale with one shared colorbar
      # (populated per subplot below, normalized/colorbarred after all models).
      sens_kappa_lines = {}   # list of (Line2D, kappa_value) per expt
      sens_hml_lines = {}     # list of (Line2D, h_ml_value) per expt
      sens_t63_meshes = {}    # list of (QuadMesh, grid) per expt
      sens_rmse_meshes = {}   # list of (QuadMesh, grid) per expt

      # Create the shared per-experiment figures (one 8-panel grid per model set)
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

         sens_kappa_figs[expt], sens_kappa_axs[expt] = make_model_grid(models, title=r"Sensitivity to $\kappa$ (h$_{ml}$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_kappa_idx[expt] = 0

         sens_hml_figs[expt], sens_hml_axs[expt] = make_model_grid(models, title=r"Sensitivity to h$_{ml}$ ($\kappa$ held at best fit)", xlabel="Time (years)", ylabel="Equilibrium Ratio ($T/T_{eq}$)")
         sens_hml_idx[expt] = 0

         sens_t63_figs[expt], sens_t63_axs[expt] = make_model_grid(models, title=r"Years to Reach 63% of T$_{eq}$ vs. $\kappa$/h$_{ml}$", xlabel=r"$\kappa$ (m$^2$/s)", ylabel=r"h$_{ml}$ (m)")
         sens_t63_idx[expt] = 0

         sens_rmse_figs[expt], sens_rmse_axs[expt] = make_model_grid(models, title=r"Fit RMSE vs. $\kappa$/h$_{ml}$ (Parameter Degeneracy)", xlabel=r"$\kappa$ (m$^2$/s)", ylabel=r"h$_{ml}$ (m)")
         sens_rmse_idx[expt] = 0

         sens_kappa_lines[expt] = []
         sens_hml_lines[expt] = []
         sens_t63_meshes[expt] = []
         sens_rmse_meshes[expt] = []

      # ----- STEP 1: fit F_ref/lambda/T_eq from the 4xCO2 T2M-vs-NETTOA regression -----

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
                  0.92,
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

      # ----- STEP 2: fit the box/diffusive/PDE models and plot final results -----

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

               def fit_func_diff(x, T):
                  return 1-erfcx(np.sqrt(x/T))

               def fit_func_ode(x, tau):
                  return 1-np.exp(-x/tau)

               if results == "validation" and run_type in (1, 2):
                  fit_T = t2m[:151]/T_eq
                  plot_T = t2m[:151]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "validation" and run_type == 3:
                  fit_T = t2m/T_eq
                  plot_T = t2m[300:]/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(300, 300 + plot_T.shape[0], 1)
               if results == "unblinded" and run_type in (1, 2):
                  fit_T = t2m[:151]/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)
               elif results == "unblinded" and run_type == 3:
                  fit_T = t2m/T_eq
                  plot_T = t2m/T_eq
                  fit_t = np.arange(1, 1 + fit_T.shape[0], 1)
                  plot_t = np.arange(1, 1 + plot_T.shape[0], 1)

               # Compute the fitted temperature curve & nettoa/OHC
               popt, pcov = curve_fit(fit_func_diff, fit_t, fit_T)
               perr = np.sqrt(np.diag(pcov))

               T = popt[0]
               T_unc = perr[0]

               # Diffusivity implied by the analytic (h_ml=0) fit's timescale
               # T, so it can be reused below as the PDE's kappa in the
               # h_ml->0 numerical limit (see pde_p_h0), keeping that curve
               # on the same diffusivity as "Diffusive Analytical" rather
               # than the joint fit's kappa_pde (which trades off against
               # a nonzero h_ml_pde and need not match D).
               D = T * 31536000 * (lmbda/((10**3)*(4.22*10**3)))**2

               popt, pcov = curve_fit(fit_func_ode, fit_t, fit_T)
               perr = np.sqrt(np.diag(pcov))

               tau = popt[0]
               tau_unc = perr[0]

               # Solve the 1-layer EBM + diffusive thermocline PDE, using
               # F_ref/lambda/T_eq already fit above and treating the
               # thermocline diffusivity kappa AND the mixed-layer depth
               # h_ml (i.e. its heat capacity, c_ml = rho_cp*h_ml) as free
               # parameters.
               t_final_years = max(fit_t.max(), plot_t.max())

               def make_pde_dT(F_ref_, T_eq_, t_final_yrs_):
                  def pde_dT(t_years, kappa, h_ml):
                     z_max = min(4000.0, max(2700.0, 6.0 * np.sqrt(kappa * t_final_yrs_ * YEAR)))
                     nz = int(min(401, max(121, z_max / 15.0 + 1)))
                     pde_p = PDEParams(
                        kappa=kappa, h_ml=h_ml, dT_eq=T_eq_, F0=F_ref_,
                        z_max=z_max, Nz=nz, t_final=t_final_yrs_ * YEAR,
                     )
                     sol = pde_solve_model(pde_p, n_save=250)
                     return np.interp(t_years, sol["t"] / YEAR, sol["dT"])
                  return pde_dT

               pde_dT_func = make_pde_dT(F_ref, T_eq, t_final_years)
               popt, pcov = curve_fit(
                  pde_dT_func, fit_t, fit_T * T_eq,
                  p0=[1.0e-4, 100.0], bounds=([1e-7, 50.0], [1e-3, 300.0]), max_nfev=60,
               )
               perr = np.sqrt(np.diag(pcov))

               kappa_pde, h_ml_pde = popt
               kappa_pde_unc, h_ml_pde_unc = perr
               pde_T = pde_dT_func(plot_t, kappa_pde, h_ml_pde) / T_eq

               # Purely diffusive counterpart (h_ml -> 0, infinite ocean): uses
               # D (the diffusivity implied by the analytic h_ml=0 fit above),
               # not kappa_pde, so this curve is directly comparable to
               # "Diffusive Analytical" -- kappa_pde comes from the joint
               # two-parameter fit where it trades off against a nonzero
               # h_ml_pde (equifinality) and so need not equal the diffusivity
               # that actually fits the data in the h_ml->0 limit. The
               # mixed-layer heat capacity is taken to ~0 so the surface
               # responds essentially instantaneously to the flux balance at
               # the top of the thermocline, and the domain is auto-deepened
               # (semi_infinite) so the deep ocean keeps absorbing heat
               # instead of hitting an insulated floor.
               pde_p_h0 = PDEParams(kappa=D, h_ml=H_ML_PURELY_DIFFUSIVE,
                                     dT_eq=T_eq, F0=F_ref, t_final=t_final_years * YEAR,
                                     semi_infinite=True)
               sol_h0 = pde_solve_model(pde_p_h0, n_save=250)
               pde_T_h0 = np.interp(plot_t, sol_h0["t"] / YEAR, sol_h0["dT"]) / T_eq

               # "Just a box" counterpart (kappa -> 0): removes the diffusive
               # thermocline entirely, leaving a bare mixed-layer slab. h_ml
               # is independently re-fit with kappa fixed at 0 (rather than
               # reusing kappa_pde/h_ml_pde), so this is its own fit to the
               # data, not the two-parameter fit merely evaluated at kappa=0.
               def pde_dT_box(t_years, h_ml):
                  return pde_dT_func(t_years, 0.0, h_ml)

               popt, pcov = curve_fit(
                  pde_dT_box, fit_t, fit_T * T_eq,
                  p0=[110.0], bounds=(5.0, 2000.0), max_nfev=60,
               )
               h_ml_box = popt[0]
               pde_T_box = pde_dT_box(plot_t, h_ml_box) / T_eq

               # ----- Sensitivity sweep: kappa & h_ml -----
               # Sweep the two free PDE parameters around this model's best fit to
               # see how sensitive/well-constrained the fit is. Every sweep point
               # is an independent PDE solve, so they're dispatched to _SENS_POOL
               # (loosened tolerance -- see SENSITIVITY_N_SAVE/RTOL/ATOL) rather
               # than run sequentially through pde_dT_func's tight-tolerance solver.
               kappa_lo = max(SENSITIVITY_KAPPA_BOUNDS[0], kappa_pde * SENSITIVITY_KAPPA_FACTOR_RANGE[0])
               kappa_hi = min(SENSITIVITY_KAPPA_BOUNDS[1], kappa_pde * SENSITIVITY_KAPPA_FACTOR_RANGE[1])
               h_ml_lo = max(SENSITIVITY_HML_BOUNDS[0], h_ml_pde * SENSITIVITY_HML_FACTOR_RANGE[0])
               h_ml_hi = min(SENSITIVITY_HML_BOUNDS[1], h_ml_pde * SENSITIVITY_HML_FACTOR_RANGE[1])

               # 1-D sweeps (other parameter held at its best-fit value) for the
               # spaghetti plots.
               kappa_grid_1d = np.geomspace(kappa_lo, kappa_hi, SENSITIVITY_N_1D)
               h_ml_grid_1d = np.linspace(h_ml_lo, h_ml_hi, SENSITIVITY_N_1D)
               kappa_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(k, h_ml_pde, F_ref, T_eq, t_final_years, plot_t) for k in kappa_grid_1d],
                  )
               ]
               h_ml_sweep_curves = [
                  c / T_eq for c in _SENS_POOL.starmap(
                     _sensitivity_solve_dT,
                     [(kappa_pde, h, F_ref, T_eq, t_final_years, plot_t) for h in h_ml_grid_1d],
                  )
               ]

               # Joint 2-D grid, shared by both heatmaps below: one PDE solve per
               # cell (on a common fine time grid), then both metrics (time to 63%
               # of T_eq, and RMSE against the AOGCM fit data) are derived from
               # that single solve by interpolation. Cells are independent, so
               # solved in parallel and reshaped back into the grid afterward.
               kappa_grid_2d = np.geomspace(kappa_lo, kappa_hi, SENSITIVITY_N_2D)
               h_ml_grid_2d = np.linspace(h_ml_lo, h_ml_hi, SENSITIVITY_N_2D)
               target_63 = (1.0 - np.exp(-1.0)) * T_eq
               t_grid_common = np.linspace(1.0, t_final_years, 400)

               grid_2d_args = [
                  (k_val, h_val, F_ref, T_eq, t_final_years, t_grid_common)
                  for h_val in h_ml_grid_2d
                  for k_val in kappa_grid_2d
               ]
               grid_2d_curves = _SENS_POOL.starmap(_sensitivity_solve_dT, grid_2d_args)

               t63_grid = np.full((SENSITIVITY_N_2D, SENSITIVITY_N_2D), np.nan)
               rmse_grid = np.full((SENSITIVITY_N_2D, SENSITIVITY_N_2D), np.nan)
               for idx, dT_common in enumerate(grid_2d_curves):
                  ih, ik = divmod(idx, SENSITIVITY_N_2D)
                  if dT_common[-1] >= target_63:
                     t63_grid[ih, ik] = np.interp(target_63, dT_common, t_grid_common)
                  dT_at_fit = np.interp(fit_t, t_grid_common, dT_common)
                  rmse_grid[ih, ik] = np.sqrt(np.mean((dT_at_fit - fit_T * T_eq) ** 2))

               # --- Panel 1: kappa spaghetti (sequential blue ramp = kappa) ---
               # Curves are recolored to the figure-wide shared norm once every
               # model is drawn (see the shared-colorbar block after the loop).
               ax = sens_kappa_axs[expt][sens_kappa_idx[expt]]
               for k_val, curve in zip(kappa_grid_1d, kappa_sweep_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_kappa_lines[expt].append((line, k_val))
               ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
               ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="lower right")
               ax.text(
                  0.02,
                  0.92,
                  rf"h$_{{ml}}$ = {h_ml_pde:.0f} m (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=8,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_kappa_idx[expt] += 1

               # --- Panel 2: h_ml spaghetti (sequential green ramp = h_ml) ---
               ax = sens_hml_axs[expt][sens_hml_idx[expt]]
               for h_val, curve in zip(h_ml_grid_1d, h_ml_sweep_curves):
                  (line,) = ax.plot(plot_t, curve, lw=1.2, alpha=0.85)
                  sens_hml_lines[expt].append((line, h_val))
               ax.scatter(plot_t, plot_T, s=4, color="red", label="AOGCM", zorder=4)
               ax.plot(plot_t, pde_T, color="black", lw=2, label="Best Fit", zorder=5)
               format_ax(ax, text=f"{model}", xscale="linear", yscale="linear", legend_loc="lower right")
               ax.text(
                  0.02,
                  0.92,
                  rf"$\kappa$ = {kappa_pde:.2e} m$^2$/s (fixed)",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=8,
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.4),
               )
               sens_hml_idx[expt] += 1

               # --- Panel 3: years-to-63%-of-T_eq heatmap over (kappa, h_ml) ---
               # Norm is applied figure-wide after the loop so every model shares
               # one color scale and one colorbar.
               ax = sens_t63_axs[expt][sens_t63_idx[expt]]
               mesh = ax.pcolormesh(kappa_grid_2d, h_ml_grid_2d, t63_grid, cmap="viridis", shading="nearest")
               ax.scatter([kappa_pde], [h_ml_pde], marker="*", s=180, color="red",
                          edgecolor="black", linewidth=0.8, zorder=5, label="Best Fit")
               sens_t63_meshes[expt].append((mesh, t63_grid))
               format_ax(ax, text=f"{model}", xscale="log", yscale="linear", legend=False, grid=False)
               sens_t63_idx[expt] += 1

               # --- Panel 4: RMSE-vs-data heatmap over (kappa, h_ml) ---
               # This is the cost surface curve_fit minimized: a sharp minimum at
               # the star means kappa/h_ml are well-constrained; a shallow ridge
               # through the star means the two parameters trade off (equifinality).
               ax = sens_rmse_axs[expt][sens_rmse_idx[expt]]
               mesh = ax.pcolormesh(kappa_grid_2d, h_ml_grid_2d, rmse_grid, cmap="viridis", shading="nearest")
               ax.scatter([kappa_pde], [h_ml_pde], marker="*", s=180, color="red",
                          edgecolor="black", linewidth=0.8, zorder=5, label="Best Fit")
               sens_rmse_meshes[expt].append((mesh, rmse_grid))
               format_ax(ax, text=f"{model}", xscale="log", yscale="linear", legend=False, grid=False)
               sens_rmse_idx[expt] += 1

               # Draw observed and fitted temperature curves
               ax = final_axs[expt][final_idx[expt]]
               ax.scatter(plot_t, T_eq*plot_T, s=4, color="red")
               ax.plot(plot_t, T_eq*plot_T, color="red", label="2-m Surface Temp.")
               ax.plot(plot_t, T_eq*fit_func_ode(plot_t, tau), color="green", label="1-Box Analytical")
               ax.plot(plot_t, T_eq*fit_func_diff(plot_t, T), color="blue", label="Diffusive Analytical")
               ax.plot(plot_t, T_eq*pde_T, color="purple", label="1-Box + Diffusion Fit")
               ax.plot(plot_t, T_eq*pde_T_h0, color="purple", ls="-.",
                       label=r"Diffusive Numerical (h$_{ml}\to0$, $\infty$ ocean)")
               ax.plot(plot_t, T_eq*pde_T_box, color="purple", ls="--",
                       label=r"1-Box Numerical ($\kappa=0$)")

               # Add the slow-timescale parameter and a reference line at 150 years.
               ax.text(
                  0.02,
                  0.92,
                  rf"$\tau_{{1b}}$ = {tau:.0f} yrs" + "\n" + rf"T$_d$ = {T:.0f} yrs, D$_d$ =  {D:.2e} m^2/s" + "\n" + rf"$\kappa_{{1b+d}}$ = {kappa_pde:.2e} m^2/s, h$_{{ml}}$ = {h_ml_pde:.0f} m",
                  transform=ax.transAxes,
                  va="top",
                  ha="left",
                  fontsize=8,
                  weight="bold",
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

               # ----- Progress / ETA -----
               # Printed once per model (this is where the sensitivity sweep's
               # PDE solves happen, so it dominates the runtime); the ETA gets
               # more accurate as more models complete.
               total_model_iters = len(RUN_TYPES) * len(RESULT_KINDS) * len(models)
               completed_model_iters = (
                  RUN_TYPES.index(run_type) * len(RESULT_KINDS) * len(models)
                  + RESULT_KINDS.index(results) * len(models)
                  + (models.index(model) + 1)
               )
               elapsed = time.time() - _SCRIPT_START_TIME
               avg_per_model = elapsed / completed_model_iters
               eta_remaining = avg_per_model * (total_model_iters - completed_model_iters)
               print(
                  f"[Progress] run_type={run_type} results={results} model={model} "
                  f"({completed_model_iters}/{total_model_iters} model-runs done) "
                  f"elapsed={elapsed/60:.1f} min, avg={avg_per_model:.1f} s/model, "
                  f"ETA remaining={eta_remaining/60:.1f} min"
               )

      # Save combined final figures in both linear and log x-scale variants
      for expt in ['4xCO2']:
         for scale in ["linear", "log"]:
            for ax, xmax in zip(final_axs[expt], final_xmax[expt]):
               ax.set_xscale(scale)
               ax.set_xlim(1, xmax + 1)
               if scale == "linear":
                  # Add a highlighted "150" tick at the orange reference line
                  # (drawn at x=151) alongside the usual evenly-spaced ticks.
                  base_ticks = np.linspace(1, xmax + 1, 5)
                  all_ticks = sorted(set(base_ticks.tolist()) | {151.0})
                  ax.set_xticks(all_ticks)
                  ax.set_xticklabels(["150" if abs(tv - 151.0) < 1e-9 else f"{tv:.0f}" for tv in all_ticks])
                  for tick_val, label in zip(all_ticks, ax.get_xticklabels()):
                     if abs(tick_val - 151.0) < 1e-9:
                        label.set_color("orange")
                        label.set_fontsize(10)
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

         # --- Shared colorbars for the 4 sensitivity figures ---
         # Put every subplot of a given figure on one common color scale, then
         # draw a single bolded colorbar spanning all panels on the right. The
         # two spaghetti figures are recolored to their figure-wide norm here.
         kappa_vals = np.array([v for _, v in sens_kappa_lines[expt]])
         kappa_norm = mpl.colors.LogNorm(vmin=kappa_vals.min(), vmax=kappa_vals.max())
         for line, v in sens_kappa_lines[expt]:
            line.set_color(plt.cm.Blues(0.25 + 0.65 * kappa_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=kappa_norm, cmap=plt.cm.Blues)
         sm.set_array([])
         cbar = sens_kappa_figs[expt].colorbar(sm, ax=list(sens_kappa_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label(r"$\kappa$ (m$^2$/s)", fontsize=15, fontweight="bold")

         hml_vals = np.array([v for _, v in sens_hml_lines[expt]])
         hml_norm = mpl.colors.Normalize(vmin=hml_vals.min(), vmax=hml_vals.max())
         for line, v in sens_hml_lines[expt]:
            line.set_color(plt.cm.Greens(0.25 + 0.65 * hml_norm(v)))
         sm = mpl.cm.ScalarMappable(norm=hml_norm, cmap=plt.cm.Greens)
         sm.set_array([])
         cbar = sens_hml_figs[expt].colorbar(sm, ax=list(sens_hml_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label(r"h$_{ml}$ (m)", fontsize=15, fontweight="bold")

         t63_min = np.nanmin([np.nanmin(g) for _, g in sens_t63_meshes[expt]])
         t63_max = np.nanmax([np.nanmax(g) for _, g in sens_t63_meshes[expt]])
         t63_norm = mpl.colors.Normalize(vmin=t63_min, vmax=t63_max)
         for mesh, _ in sens_t63_meshes[expt]:
            mesh.set_norm(t63_norm)
         sm = mpl.cm.ScalarMappable(norm=t63_norm, cmap=plt.get_cmap("viridis"))
         sm.set_array([])
         cbar = sens_t63_figs[expt].colorbar(sm, ax=list(sens_t63_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label("Years to 63% of $T_{eq}$", fontsize=15, fontweight="bold")

         rmse_min = np.nanmin([np.nanmin(g) for _, g in sens_rmse_meshes[expt]])
         rmse_max = np.nanmax([np.nanmax(g) for _, g in sens_rmse_meshes[expt]])
         rmse_norm = mpl.colors.LogNorm(vmin=max(rmse_min, 1e-6), vmax=rmse_max)
         for mesh, _ in sens_rmse_meshes[expt]:
            mesh.set_norm(rmse_norm)
         sm = mpl.cm.ScalarMappable(norm=rmse_norm, cmap=plt.get_cmap("viridis"))
         sm.set_array([])
         cbar = sens_rmse_figs[expt].colorbar(sm, ax=list(sens_rmse_axs[expt]), fraction=0.03, pad=0.02)
         cbar.set_label("RMSE vs. AOGCM Data (K)", fontsize=15, fontweight="bold")

         # Save the kappa/h_ml spaghetti figures in both linear and log x-scale
         # variants (mirroring the combined final figures above); the heatmaps
         # have no time axis so they're saved once, as-is.
         spaghetti_figs_to_save = [
            (sens_kappa_figs[expt], sens_kappa_axs[expt], "sensitivity_spaghetti_kappa"),
            (sens_hml_figs[expt], sens_hml_axs[expt], "sensitivity_spaghetti_h_ml"),
         ]
         for fig, axs, name in spaghetti_figs_to_save:
            for scale in ["linear", "log"]:
               for ax, xmax in zip(axs, final_xmax[expt]):
                  ax.set_xscale(scale)
                  ax.set_xlim(1, xmax + 1)
                  if scale == "log":
                     ax.xaxis.set_major_locator(mpl.ticker.LogLocator())

               fig.savefig(
                  outdir / current_dir / results / "png" / f"{expt}_all_models_{name}_{results}_{scale}{suffix}.png",
                  dpi=200,
                  bbox_inches="tight",
               )
               fig.savefig(
                  outdir / current_dir / results / "pdf" / f"{expt}_all_models_{name}_{results}_{scale}{suffix}.pdf",
                  bbox_inches="tight",
               )
            plt.close(fig)

         heatmap_figs_to_save = [
            (sens_t63_figs[expt], "sensitivity_heatmap_t63"),
            (sens_rmse_figs[expt], "sensitivity_heatmap_rmse"),
         ]
         for fig, name in heatmap_figs_to_save:
            fig.savefig(
               outdir / current_dir / results / "png" / f"{expt}_all_models_{name}_{results}{suffix}.png",
               dpi=200,
               bbox_inches="tight",
            )
            fig.savefig(
               outdir / current_dir / results / "pdf" / f"{expt}_all_models_{name}_{results}{suffix}.pdf",
               bbox_inches="tight",
            )
            plt.close(fig)

      print("Finished final val/result plots")

_SENS_POOL.close()
_SENS_POOL.join()