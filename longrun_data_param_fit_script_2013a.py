#!/usr/bin/env python3

import rpy2.robjects as ro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sympy as sp
from tqdm import tqdm
from matplotlib.lines import Line2D

lin = True
run_type = 3
results = 'results'

dir_list = ["geoffroy_replicate_results", "50-yr_avg_forcing_results", "50-yr_avg_tau_s_LR_fit_results"]

current_dir = dir_list[run_type-1]

print(current_dir, results, f"lin: {lin}")

def sympy_prop_unc(expr, values, uncertainties):
    """
    expr: SymPy expression
    values: dict of {symbol: value}
    uncertainties: dict of {symbol: uncertainty}
    """
    variance = 0

    for sym, unc in uncertainties.items():
        deriv = sp.diff(expr, sym)
        deriv_val = float(deriv.evalf(subs=values))
        variance += (deriv_val * unc) ** 2

    return float(np.sqrt(variance))

# Create dataframe to save model params
df = pd.DataFrame({
    "model": [], # model Name
    "C": [], # atmosphere/land/upper-ocean heat capacity
    "C0": [], # deep-ocean heat capacity
    "gamma": [], # heat exchange coefficient
    "tau_f": [], # fast relaxation time estimate
    "tau_s": [], # slow relaxation time estimate
    "F_ref": [], # radiative forcing
    "lambda": [], # radiative feedback parameter
    "T_eq": [], # equilibrium temperature
    "a_f": [],
    "a_s": [],
    "C_unc": [], # atmosphere/land/upper-ocean heat capacity
    "C0_unc": [], # deep-ocean heat capacity
    "gamma_unc": [], # heat exchange coefficient
    "tau_f_unc": [], # fast relaxation time estimate
    "tau_s_unc": [], # slow relaxation time estimate
    "F_ref_unc": [], # radiative forcing
    "lambda_unc": [], # radiative feedback parameter
    "T_eq_unc": [], # equilibrium temperature
    "a_f_unc": [],
    "a_s_unc": [],
})

rdata_file = Path("./data/int_netToa_longrun.Rdata")

# Load .Rdata file into R global environment
ro.r["load"](str(rdata_file))

# Save models/expts to lists
data = ro.globalenv["int_nettoa_longrun_data"]
models = list(ro.globalenv["models"])
expts = list(ro.globalenv["expts"])
df["model"] = models

# Define dir to save to
outdir = Path("./figures")
outdir.mkdir(exist_ok=True)

# ----------------- STEP 1 ---------------------
print("Starting Step 1")

for model in models:
    # Filter data for given model
    model_data = data.rx2(model)
    t2m_mean = 0
    nettoa_mean = 0

    for expt in expts:
        expt_data = model_data.rx2(expt)

        # Create flatten arrays of t2m, netToa
        if run_type == 1:
            t2m = np.array(expt_data.rx2("T2M")).ravel()[:150]
            nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[:150]
        else:
            t2m = np.array(expt_data.rx2("T2M")).ravel()
            nettoa = np.array(expt_data.rx2("NETTOA")).ravel()

        # Filter non-nan data
        quality_filter = np.isfinite(t2m) & np.isfinite(nettoa)
        if np.sum(quality_filter)/len(quality_filter) != 1: print(f"{model} valid/all years:", np.sum(quality_filter)/len(quality_filter))
        t2m = t2m[quality_filter]
        nettoa = nettoa[quality_filter]
        
        # Convert variables to anomalies
        if expt == 'piControl':
            t2m_mean = np.mean(t2m)
            nettoa_mean = np.mean(nettoa)
        elif expt == '4xCO2':
            t2m = t2m - t2m_mean
            nettoa = nettoa - nettoa_mean

            if run_type != 1:
                tmp_t2m_list = []
                tmp_nettoa_list = []
                for i in range(0, t2m.shape[0], 50):
                    tmp_t2m_list.append(np.mean(t2m[i:i+50]))
                    tmp_nettoa_list.append(np.mean(nettoa[i:i+50]))
                t2m = np.array(tmp_t2m_list)
                nettoa = np.array(tmp_nettoa_list)

            # Linear regression
            [m, b], cov = np.polyfit(t2m, nettoa, 1, cov=True)
            m_unc = np.sqrt(cov[0, 0])
            b_unc = np.sqrt(cov[1, 1])
            n_1 = 150

            # Fit line
            xfit = np.linspace(t2m.min(), t2m.max(), 100)
            yfit = m * xfit + b

            # Plot figure with data & fit
            plt.figure(figsize=(7, 5))
            plt.scatter(t2m, nettoa, s=8, alpha=0.5, label="Data")
            plt.plot(
                xfit,
                yfit,
                linewidth=2,
                label=f"Fit: F_ref={b:.3f},-lambda={m:.3f}"
            )
            plt.xlabel("2-meter Air Temperature Anomaly (K)")
            plt.ylabel(r"Net TOA Radiative Flux Anomaly ($W*m^{-2}$)")
            plt.title(f"{model} {expt}: Net TOA vs T2M")
            plt.grid(True)
            plt.legend()
            
            # Define var expressions
            sp_m, sp_b = sp.symbols("m b")
            F_expr = sp_b
            lambda_expr = -sp_m
            T_eq_expr = sp_b / (-sp_m)

            # Calc uncertainty
            F_ref_unc = sympy_prop_unc(
                F_expr,
                {sp_b: b},
                {sp_b: b_unc}
            )
            lambda_unc = sympy_prop_unc(
                lambda_expr,
                {sp_m: m},
                {sp_m: m_unc}
            )
            T_eq_unc = sympy_prop_unc(
                T_eq_expr,
                {sp_m: m, sp_b: b},
                {sp_m: m_unc, sp_b: b_unc}
            )
            
            # Save F_ref/lambda/T_eq and unc to df if 4xCO2 experiment
            df.loc[df["model"] == model, "F_ref"] = b
            df.loc[df["model"] == model, "lambda"] = -m
            df.loc[df["model"] == model, "T_eq"] = b/(-m)

            df.loc[df["model"] == model, "F_ref_unc"] = F_ref_unc
            df.loc[df["model"] == model, "lambda_unc"] = lambda_unc
            df.loc[df["model"] == model, "T_eq_unc"] = T_eq_unc

            # Save fig
            outfile_png = f"{outdir}/{current_dir}/step1/png/{expt}_{model}_T2M_vs_NETTOA.png"
            outfile_pdf = f"{outdir}/{current_dir}/step1/pdf/{expt}_{model}_T2M_vs_NETTOA.pdf"
            plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
            plt.savefig(outfile_pdf, dpi=200, bbox_inches="tight")
            plt.close()

print(f"Finished step 1: saved Step 1 figures & params to df")

# ----------------- STEP 2 ---------------------
print("Starting Step 2")

for model in models:
    # Filter data for given model
    model_data = data.rx2(model)
    t2m_mean = 0
    nettoa_mean = 0

    for expt in expts:
        expt_data = model_data.rx2(expt)

        # Create flatten arrays of t2m, netToa for approx. first 150 years
        t2m_first10 = np.array(expt_data.rx2("T2M")).ravel()[0:10]
        if run_type != 3:
            t2m = np.array(expt_data.rx2("T2M")).ravel()[30:151]
            nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[30:151]
        else:
            t2m = np.array(expt_data.rx2("T2M")).ravel()[50:]
            nettoa = np.array(expt_data.rx2("NETTOA")).ravel()[50:]
        
        # Filter non-nan data
        quality_filter = np.isfinite(t2m) & np.isfinite(nettoa)
        if np.sum(quality_filter)/len(quality_filter) != 1: print(f"{model} valid/all years:", np.sum(quality_filter)/len(quality_filter))
        t2m = t2m[quality_filter]
        nettoa = nettoa[quality_filter]
        
        # Convert variables to anomalies
        if expt == 'piControl':
            t2m_first10_mean = np.mean(t2m_first10)
            t2m_mean = np.mean(t2m)
            nettoa_mean = np.mean(nettoa)
        elif expt == '4xCO2':
            t2m_first10 = t2m_first10 - t2m_first10_mean
            t2m = t2m - t2m_mean
            nettoa = nettoa - nettoa_mean
            T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]
            
            # Linear regression
            if run_type != 3: t = np.arange(30, 30+t2m.shape[0], 1)
            else: t = np.arange(50, 50+t2m.shape[0], 1)
            mask = ((T_eq - t2m) > 0)

            t = t[mask]
            y = np.log(T_eq-t2m[mask])-np.log(T_eq)
            [m, b], cov = np.polyfit(t, y, 1, cov=True)
            m_unc = np.sqrt(cov[0, 0])
            b_unc = np.sqrt(cov[1, 1])
            xfit = t
            yfit = m * xfit + b

            # Plot figure with data & fit
            plt.figure(figsize=(7, 5))
            plt.scatter(t, y, s=8, alpha=0.5, label="Data")
            plt.plot(
                xfit,
                yfit,
                linewidth=2,
                label=f"Fit: log(a_s)={b:.3f},-1/t_s={m:.3f}"
            )
            plt.xlabel(r"Time (years)")
            plt.ylabel(r"log($T_{eq}$-T)-log($T_{eq}$)")
            plt.title(f"{model} {expt}: log(T_eq-T) - log(T_eq) vs. Time")
            plt.grid(True)
            plt.legend()

            # Calc important variables
            lmbda = df.loc[df["model"] == model, "lambda"].iloc[0]
            tau_s = -1/m
            a_s = np.exp(b)
            a_f = 1-a_s
            t_first10 = np.arange(1, 11, 1)
            tau_f = np.mean(t_first10/(np.log(a_f)-np.log(1-(t2m_first10/T_eq)-a_s*np.exp(-t_first10/tau_s)))) # NOT FULLY CORRECT
            C = lmbda/((a_f/tau_f) + (a_s/tau_s))
            C0 = lmbda*(tau_f*a_f + tau_s*a_s) - C
            gamma = C0/(tau_f*a_s + tau_s*a_f)

            # Define var expressions
            sp_lambda, sp_a_s, sp_tau_s, sp_tau_f, sp_m, sp_b = sp.symbols(
                "lambda a_s tau_s tau_f m b"
            )
            tau_s_expr = -1/sp_m
            a_s_expr = sp.exp(sp_b)
            sp_a_f = 1 - sp_a_s
            C_expr = sp_lambda / ((sp_a_f / sp_tau_f) + (sp_a_s / sp_tau_s))
            C0_expr = sp_lambda * (
                sp_tau_f * sp_a_f + sp_tau_s * sp_a_s
            ) - C_expr
            gamma_expr = C0_expr / (
                sp_tau_f * sp_a_s + sp_tau_s * sp_a_f
            )

            # Calc unc
            lambda_unc = df.loc[df["model"] == model, "lambda_unc"].iloc[0]
            tau_s_unc = sympy_prop_unc(tau_s_expr, {sp_m: m}, {sp_m: m_unc})
            tau_f_unc = np.std(t_first10/(np.log(a_f)-np.log(1-(t2m_first10/T_eq)-a_s*np.exp(-t_first10/tau_s))))
            a_s_unc = sympy_prop_unc(a_s_expr, {sp_b: b}, {sp_b: b_unc})
            a_f_unc = sympy_prop_unc(sp_a_f, {sp_a_s: a_s}, {sp_a_s: a_s_unc})
            C_unc = sympy_prop_unc(
                C_expr,
                {
                    sp_lambda: lmbda,
                    sp_a_s: a_s,
                    sp_tau_s: tau_s,
                    sp_tau_f: tau_f,
                },
                {
                    sp_lambda: lambda_unc,
                    sp_a_s: a_s_unc,
                    sp_tau_s: tau_s_unc,
                    sp_tau_f: tau_f_unc,
                }
            )
            C0_unc = sympy_prop_unc(
                C0_expr,
                {
                    sp_lambda: lmbda,
                    sp_a_s: a_s,
                    sp_tau_s: tau_s,
                    sp_tau_f: tau_f,
                },
                {
                    sp_lambda: lambda_unc,
                    sp_a_s: a_s_unc,
                    sp_tau_s: tau_s_unc,
                    sp_tau_f: tau_f_unc,
                }
            )
            gamma_unc = sympy_prop_unc(
                gamma_expr,
                {
                    sp_lambda: lmbda,
                    sp_a_s: a_s,
                    sp_tau_s: tau_s,
                    sp_tau_f: tau_f,
                },
                {
                    sp_lambda: lambda_unc,
                    sp_a_s: a_s_unc,
                    sp_tau_s: tau_s_unc,
                    sp_tau_f: tau_f_unc,
                }
            )

            # Save vars to df if 4xCO2 experiment
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

            # Save fig
            outfile_png = f"{outdir}/{current_dir}/step2/png/{expt}_log(Teq-T)_vs_t_{model}.png"
            outfile_pdf = f"{outdir}/{current_dir}/step2/pdf/{expt}_log(Teq-T)_vs_t_{model}.pdf"
            plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
            plt.savefig(outfile_pdf, dpi=200, bbox_inches="tight")
            plt.close()

print(df)
print("Finished Step 2: Saved all parameters to df")

# ---------------- Compare Parameters ----------------

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

for var in list(model_paperParams['GISSE2R'].keys()):
    fig, ax = plt.subplots(figsize=(6, 6))
    handles = []

    tmp_mu_SN_list = []
    for model, (face, edge) in zip(model_paperParams.keys(), colors):
        mu_GF = model_paperParams[model][var]
        mu_SN = df.loc[df["model"] == model, f"{var}"].iloc[0]
        mu_unc = df.loc[df["model"] == model, f"{var}_unc"].iloc[0]
        ax.errorbar(mu_SN, mu_GF, xerr=mu_unc*1.96, marker='.', ms=13, mfc=face,mec=edge,ecolor=edge,linewidth=1.5,zorder=3)
        tmp_mu_SN_list.append(mu_SN)
        tmp_mu_SN_list.append(mu_GF)
        handles.append(
        Line2D(
            [], [], marker='.', linestyle='None',
            markersize=13,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=1.5,
            label=model
            )
        )

    ax.plot(np.arange(min(tmp_mu_SN_list), max(tmp_mu_SN_list)+0.001, 0.001), np.arange(min(tmp_mu_SN_list), max(tmp_mu_SN_list)+0.001, 0.001), 'k--', label=r'$\mu_{\rm GF}=\mu_{\rm SN}$')
    ax.set_xlabel(r'$\mu_{\rm SN}$')
    ax.set_ylabel(r'$\mu_{\rm GF}$')
    ax.set_title(f'{var}: Geoffroy vs. Sanjit/Nadir (w/ 95% CI)')
    plt.xticks(rotation=45)
    ax.legend(handles=handles)
    plt.tight_layout()
    # Save fig
    outfile_png = f"{outdir}/{current_dir}/validation/png/{expt}_GF_vs_SN_{var}.png"
    outfile_pdf = f"{outdir}/{current_dir}/validation/pdf/{expt}_GF_vs_SN_{var}.pdf"
    plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
    plt.savefig(outfile_pdf, dpi=200, bbox_inches="tight")
    plt.close()

# ----------------- Plot Results ---------------------

for model in models:
    # Filter data for given model
    print(model)
    model_data = data.rx2(model)
    t2m_mean = 0
    nettoa_mean = 0

    for expt in expts:
        expt_data = model_data.rx2(expt)

        # Create flatten arrays of t2m
        if results == 'validation': t2m = np.array(expt_data.rx2("T2M")).ravel()[:151]
        else: t2m = np.array(expt_data.rx2("T2M")).ravel()

        # Filter non-nan data
        quality_filter = np.isfinite(t2m)
        if np.sum(quality_filter)/len(quality_filter) != 1: print(f"{model} valid/all years:", np.sum(quality_filter)/len(quality_filter))
        t2m = t2m[quality_filter]
        
        # Convert variables to anomalies
        if expt == 'piControl':
            t2m_mean = np.mean(t2m)
        elif expt == '4xCO2':
            t2m = t2m - t2m_mean
            T_eq = df.loc[df["model"] == model, "T_eq"].iloc[0]
            a_f = df.loc[df["model"] == model, "a_f"].iloc[0]
            a_s = df.loc[df["model"] == model, "a_s"].iloc[0]
            tau_f = df.loc[df["model"] == model, "tau_f"].iloc[0]
            tau_s = df.loc[df["model"] == model, "tau_s"].iloc[0]
            T_eq_unc = df.loc[df["model"] == model, "T_eq_unc"].iloc[0]
            a_f_unc = df.loc[df["model"] == model, "a_f_unc"].iloc[0]
            a_s_unc = df.loc[df["model"] == model, "a_s_unc"].iloc[0]
            tau_f_unc = df.loc[df["model"] == model, "tau_f_unc"].iloc[0]
            tau_s_unc = df.loc[df["model"] == model, "tau_s_unc"].iloc[0]
            
            iterations = 1000
            param_mean = np.array([T_eq, a_s, tau_f, tau_s])
            param_cov = np.diag(np.array([T_eq_unc**2, a_s_unc**2, tau_f_unc**2, tau_s_unc**2]))

            params = np.random.multivariate_normal(param_mean, param_cov, size=iterations)

            # Calculate
            t = np.arange(1, 1+t2m.shape[0], 1)
            T = T_eq*(a_f*(1-np.exp(-t/tau_f))+a_s*(1-np.exp(-t/tau_s)))

            # Plot figure with data & fit
            fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=120)
            ax.scatter(t, t2m, s=4, color='red')
            ax.plot(t, t2m, color='red', label="2-m Surface Temp.")
            ax.plot(t, T, color='blue', label="150-Year Fit")
            for i in range(iterations):
                T_rnd = params[i, 0]*((1-params[i, 1])*(1-np.exp(-t/params[i, 2]))+params[i, 1]*(1-np.exp(-t/params[i, 3])))
                ax.plot(t, T_rnd, color="blue", alpha=0.01)
            ax.set_xlabel(r"Time (years)")
            ax.set_ylabel(r"Temperature Anomaly (K)")
            ax.set_title(f"{model} {expt}: T2M w/ 2-box fit")
            if lin == True: scale = 'linear'
            else: scale = 'log'
            ax.set_xscale(scale)
            ax.legend()

            if lin == True: ax.set_xticks(np.linspace(1, np.max(t)+1, 10))
            ax.set_yticks(np.linspace(np.min(t2m), np.max(t2m)+1, 4))

            # Save fig
            outfile_png = f"{outdir}/{current_dir}/{results}/png/{expt}_{model}_T2m_vs_t_{scale}.png"
            outfile_pdf = f"{outdir}/{current_dir}/{results}/pdf/{expt}_{model}_T2m_vs_t_{scale}.pdf"
            plt.savefig(outfile_png, dpi=200, bbox_inches="tight")
            plt.savefig(outfile_pdf, dpi=200, bbox_inches="tight")
            plt.close()
