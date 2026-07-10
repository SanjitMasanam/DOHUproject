"""
1-layer EBM (slab mixed layer) + diffusive thermocline.
Fill in comments as you work through each section.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass, replace
from scipy.integrate import solve_ivp
from scipy.sparse import diags

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

# =================================================================
# Constants and parameters
# =================================================================
YEAR = 365.25 * 24 * 3600.0
CM2S_TO_M2S = 1.0e-4

OUTDIR = "1lyEBM_diffusive"

C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_PINK = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")


@dataclass(frozen=True)
class Params:
    # These are fixed parameters
    # that match what was used in Hansen 1984
    kappa: float = 1.0 * CM2S_TO_M2S # Diffusivity
    h_ml: float = 110.0 # height of mixed layer
    dT_eq: float = 4.2 # T equil
    F0: float = 4.3 # Forcing for 2xCO2
    rho_cp: float = 4.186e6 # vol. heat capacity of seawater
    z_max: float = 2700.0 
    Nz: int = 241 # Number of Z levels
    t_final: float = 35.0 * YEAR

    @property
    def c_ml(self):
        # Calculate heat capacity of mixed layer
        return self.rho_cp * self.h_ml

    @property
    def lam(self):
        # Calculate climate feedback parameter
        return self.F0 / self.dT_eq

    @property
    def D(self):
        # Calculate kinematic diffusivity parameter
        return self.rho_cp * self.kappa

    @property
    def tau_ml(self):
        # Calculate characterisitic response
        # time of mixed layer (1-box model)
        return self.c_ml / self.lam


# =================================================================
# Core numerics
# =================================================================
def make_rhs(dz, p):
    # Function that sets up BC and PDE of interior region
    def rhs(t, u):
        du = np.empty_like(u)
        T0 = u[0] # T_0 = theta(z=0)

        # Surface B.C. where 
        # dT/dt = F - lambda * T_0 + D * dT/dz (@ z=0) / c_ml
        du[0] = (p.F0 - p.lam * T0 + p.D * (u[1] - T0) / dz) / p.c_ml

        # dT/dt = kappa * d^2T/dz^2
        du[1:-1] = p.kappa * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dz**2

        # dT/dt = kappa * d^2T/d^2t (@ z=zmax) = 0
        du[-1] = p.kappa * 2.0 * (u[-2] - u[-1]) / dz**2
        return du

    return rhs


def solve_model(p, n_save=400, rtol=1e-8, atol=1e-10):
    # Solve PDE numerically using scipy.solve_icp
    z = np.linspace(0.0, p.z_max, p.Nz)
    dz = z[1] - z[0]
    u0 = np.zeros(p.Nz)

    sparsity = diags([1, 1, 1], [-1, 0, 1], shape=(p.Nz, p.Nz))
    sol = solve_ivp(
        make_rhs(dz, p), t_span=(0.0, p.t_final), y0=u0, method="BDF",
        t_eval=np.linspace(0.0, p.t_final, n_save),
        rtol=rtol, atol=atol, jac_sparsity=sparsity,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return {"t": sol.t, "theta": sol.y, "dT": sol.y[0], "z": z, "dz": dz,
            "sol": sol, "p": p}


def analytic_mixed_layer(t, p):
    # Analytical solution for a 1-box (ml-only) model
    return p.dT_eq * (1.0 - np.exp(-t / p.tau_ml))


def e_folding_time(t, dT, p):
    # Calculate how long it takes to reach 63% of something
    target = (1.0 - np.exp(-1.0)) * p.dT_eq
    if dT[-1] < target:
        return np.nan
    return np.interp(target, dT, t)


# =================================================================
# Small print helpers so the check output is uniform and scannable
# =================================================================
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def style_axes(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# =================================================================
# Main experiment suite
# =================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Print out model params used
    p = Params()
    section("MODEL PARAMETERS (defaults: 110 m mixed layer, k = 1 cm^2/s)")
    print(f"  F0      = {p.F0} W/m^2       dT_eq = {p.dT_eq} C")
    print(f"  lam     = F0/dT_eq = {p.lam:.4f} W m^-2 K^-1")
    print(f"  h_ml    = {p.h_ml} m  ->  c = rho_cp*h = {p.c_ml:.3e} J m^-2 K^-1")
    print(f"  kappa   = {p.kappa:.1e} m^2/s ->  D = rho_cp*kappa = {p.D:.1f} W m^-1 K^-1")
    print(f"  tau_ml  = c/lam = {p.tau_ml / YEAR:.2f} yr  (isolated-slab e-folding time)")
    print(f"  grid: Nz = {p.Nz}, dz = {p.z_max / (p.Nz - 1):.2f} m, z_max = {p.z_max} m")

    # Save solutions to each run to a dictionary
    runs = {
        "63 m mixed layer only":  solve_model(replace(p, h_ml=63.0, kappa=0.0)),
        "110 m mixed layer only": solve_model(replace(p, kappa=0.0)),
        "110 m + k = 1 cm$^2$/s": solve_model(p),
        "110 m + k = 2 cm$^2$/s": solve_model(replace(p, kappa=2.0 * CM2S_TO_M2S)),
    }

    # Check that all runs are finite
    section("CHECK 1: solver diagnostics (k = 1 run) and finiteness (all runs)")
    s = runs["110 m + k = 1 cm$^2$/s"]["sol"]
    print(f"  success={s.success}  nfev={s.nfev}  njev={s.njev}  nlu={s.nlu}")
    all_finite = all(np.all(np.isfinite(r["theta"])) for r in runs.values())
    check("all runs finite (no NaN/Inf)", all_finite)

    # Reproduce Hansen Fig. 16 using run data
    section("FIGURE: fig16_reproduction.png  +  CHECK 2: values at t = 35 yr")
    styles = {
        "63 m mixed layer only":  dict(color=C_BLUE, ls="-"),
        "110 m mixed layer only": dict(color=C_GREEN, ls="-"),
        "110 m + k = 1 cm$^2$/s": dict(color=C_VERM, ls=":"),
        "110 m + k = 2 cm$^2$/s": dict(color=C_PINK, ls=":"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, r in runs.items():
        ax.plot(r["t"] / YEAR, r["dT"], lw=2, label=name, **styles[name])
    ax.axhline(p.dT_eq, color="0.4", lw=1, ls="--")
    ax.text(19.5, p.dT_eq + 0.09, r"$\Delta T_{eq}$ = 4.2 $\degree$C", color="0.3", fontsize=9)
    ax.set(xlim=(0, 35), ylim=(0, 5), xlabel="Years",
           ylabel=r"$\Delta T$ ($\degree$C)",
           title="Response to instant CO$_2$ doubling [Hansen et al. (1984) Fig. 16]")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig16_reproduction.png")

    # Check if values at end of 35 years match estimates from Fig. 16
    fig16_eyeball = {
        "63 m mixed layer only":  4.10,
        "110 m mixed layer only": 3.85,
        "110 m + k = 1 cm$^2$/s": 2.00,
        "110 m + k = 2 cm$^2$/s": 1.60,
    }
    for name, r in runs.items():
        got = r["dT"][-1]
        exp = fig16_eyeball[name]
        check(f"dT(35 yr) [{name}] = {got:.2f} C vs ~{exp:.2f} C from Fig. 16",
              abs(got - exp) < 0.25, "eyeballed target, tol 0.25 C")

    # Check kappa = 0 numerical solution vs. analytical solution
    section("CHECK 3: kappa = 0 numeric vs analytic slab solution")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    max_err = 0.0
    for name, color in [("63 m mixed layer only", C_BLUE),
                        ("110 m mixed layer only", C_GREEN)]:
        r = runs[name]
        ana = analytic_mixed_layer(r["t"], r["p"])
        max_err = max(max_err, np.max(np.abs(r["dT"] - ana)))
        ax.plot(r["t"] / YEAR, r["dT"], color=color, lw=2, label=f"numeric, {name}")
        ax.plot(r["t"] / YEAR, ana, color="k", lw=1.2, ls="--",
                label=f"analytic, {name}")
    check(f"max |numeric - analytic| = {max_err:.2e} C", max_err < 1e-3,
          "tol 1e-3 C; errors here would implicate the slab ODE/integrator")
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)",
           title=r"$\kappa = 0$ limit: solver vs closed form "
                 r"$\Delta T_{eq}(1 - e^{-t/\tau})$")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, "analytic_vs_numeric_k0.png")

    # Check something about characteristic timescale
    section("CHECK 4: isolated-slab e-folding times vs paper text")
    for name, paper_val in [("63 m mixed layer only", 8.0),
                            ("110 m mixed layer only", 15.0)]:
        r = runs[name]
        te = e_folding_time(r["t"], r["dT"], r["p"]) / YEAR
        check(f"e-folding [{name}] = {te:.1f} yr vs paper '~{paper_val:.0f} yr'",
              abs(te - paper_val) / paper_val < 0.15, "tol 15% (paper rounds)")

    # Check stated value at 102 yrs (2.65 C) which is a long run for them
    section("CHECK 5: long-run checkpoint — 2.65 C at ~102 yr (k = 1, 110 m)")
    long_run = solve_model(replace(p, t_final=150.0 * YEAR))
    te_long = e_folding_time(long_run["t"], long_run["dT"], p) / YEAR
    target = (1.0 - np.exp(-1.0)) * p.dT_eq
    check(f"t(dT = {target:.2f} C) = {te_long:.0f} yr vs paper's 102 yr",
          abs(te_long - 102.0) / 102.0 < 0.15,
          "tol 15%: paper's exact rho_cp / grid are unknown")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(long_run["t"] / YEAR, long_run["dT"], color=C_VERM, lw=2,
            label="110 m + k = 1 cm$^2$/s")
    ax.axhline(target, color="0.4", lw=1, ls="--")
    ax.axvline(102.0, color="0.4", lw=1, ls="--")
    ax.plot([te_long], [target], "o", color="k", ms=6, zorder=5)
    ax.annotate(f"reached {target:.2f} $\\degree$C at {te_long:.0f} yr\n"
                "(paper: 102 yr)", xy=(te_long, target),
                xytext=(te_long + 8, target - 0.9), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)", xlim=(0, 150),
           title="Long run: paper's 63%-response checkpoint")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, "longrun_checkpoint.png")

    # Vary sensitivity and plot affect
    sens_runs = {}
    for dte, paper_val, shade in [(2.0, 27.0, 0.45), (3.0, 55.0, 0.65), (4.2, 102.0, 0.9)]:
        r = solve_model(replace(p, dT_eq=dte, t_final=150.0 * YEAR))
        te = e_folding_time(r["t"], r["dT"], r["p"]) / YEAR
        sens_runs[dte] = (r, te, shade)
        check(f"dT_eq = {dte} C: e-folding = {te:.0f} yr vs paper {paper_val:.0f} yr",
              abs(te - paper_val) / paper_val < 0.15, "tol 15%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.Oranges
    for dte, (r, te, shade) in sens_runs.items():
        ax.plot(r["t"] / YEAR, r["dT"], color=cmap(shade), lw=2,
                label=f"$\\Delta T_{{eq}}$ = {dte} $\\degree$C  "
                      f"(e-fold {te:.0f} yr)")
        if np.isfinite(te):
            ax.plot([te], [(1 - np.exp(-1)) * dte], "o", color=cmap(shade), ms=6)
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)", xlim=(0, 150),
           title="Higher sensitivity ⇒ disproportionately slower response\n"
                 "(dots mark the 63% e-folding point)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "sweep_sensitivity.png")

    # Vary diffusivity and plot effect
    k_values = [0.0, 0.5, 1.0, 2.0, 4.0]
    k_runs = [solve_model(replace(p, kappa=k * CM2S_TO_M2S)) for k in k_values]
    final_vals = [r["dT"][-1] for r in k_runs]
    check("dT(35 yr) strictly decreasing with k: "
          + ", ".join(f"{v:.2f}" for v in final_vals),
          np.all(np.diff(final_vals) < 0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.Blues
    for k, r, shade in zip(k_values, k_runs, np.linspace(0.35, 0.95, len(k_values))):
        ax.plot(r["t"] / YEAR, r["dT"], color=cmap(shade), lw=2,
                label=f"k = {k:g} cm$^2$/s")
    ax.axhline(p.dT_eq, color="0.4", lw=1, ls="--")
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)", xlim=(0, 35), ylim=(0, 5),
           title="Effect of thermocline diffusivity (110 m mixed layer)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    savefig(fig, "sweep_diffusivity.png")

    # Vary height of ml and plot effect
    h_values = [63.0, 110.0, 200.0]
    h_runs = [solve_model(replace(p, h_ml=h)) for h in h_values]
    final_vals = [r["dT"][-1] for r in h_runs]
    check("dT(35 yr) strictly decreasing with h: "
          + ", ".join(f"{v:.2f}" for v in final_vals),
          np.all(np.diff(final_vals) < 0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.Greens
    for h, r, shade in zip(h_values, h_runs, np.linspace(0.45, 0.9, len(h_values))):
        ax.plot(r["t"] / YEAR, r["dT"], color=cmap(shade), lw=2, label=f"h = {h:g} m")
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)", xlim=(0, 35), ylim=(0, 3),
           title="Effect of mixed-layer depth (k = 1 cm$^2$/s)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "sweep_mixed_layer_depth.png")

    # Plot the continuous depth-time evolution of the anomaly (rather than a
    # handful of discrete year snapshots). Uses a long run over the full
    # ocean depth (z_max) since the deep ocean barely responds within the
    # original 35-yr window — a longer run is needed to see it warm at all.
    deep_run = solve_model(replace(p, t_final=1000.0 * YEAR), n_save=500)
    t_yr, z, theta = deep_run["t"] / YEAR, deep_run["z"], deep_run["theta"]
    segments = [np.column_stack([theta[:, i], z]) for i in range(theta.shape[1])]
    lc = LineCollection(segments, cmap="viridis", array=t_yr, linewidths=1.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    ax.set_xlim(theta.min(), theta.max() * 1.05)
    ax.set_ylim(p.z_max, 0)
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Years")
    ax.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel="Depth z (m)",
           title="Continuous depth-time evolution of the thermocline anomaly\n"
                 "(110 m + k = 1 cm$^2$/s, 1000-yr run)")
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, "depth_profiles.png")

    # Calculate energy conservation
    r = runs["110 m + k = 1 cm$^2$/s"]
    pp, dz, th = r["p"], r["dz"], r["theta"]
    H = (pp.c_ml * th[0]
         + pp.rho_cp * dz * th[1:-1].sum(axis=0) # Many discrete but tiny slabs to approximate continuous region
         + pp.rho_cp * (dz / 2.0) * th[-1])
    F_net = pp.F0 - pp.lam * th[0] # Flux at top of ocean vs. time
    F_int = np.concatenate([[0.0], cumulative_trapezoid(F_net, r["t"])]) # Integrated flux at top of ocean
    resid = np.abs((H - H[0]) - F_int) # Difference in heat uptake and energy going in (should be same)
    rel_err = resid.max() / F_int[-1]
    check(f"max |dH - Int(F dt)| / Int(F dt) = {rel_err:.2e}", rel_err < 5e-3,
          "tol 0.5%; violation would mean flux/capacity bookkeeping is inconsistent")

    for name in ["110 m + k = 1 cm$^2$/s", "110 m + k = 2 cm$^2$/s"]:
        rr = runs[name]
        ppp, ddz, tth = rr["p"], rr["dz"], rr["theta"][:, -1]
        H_slab = ppp.c_ml * tth[0]
        H_therm = ppp.rho_cp * (ddz * tth[1:-1].sum() + (ddz / 2) * tth[-1])
        print(f"  heat stored at 35 yr [{name}]: "
              f"{100 * H_therm / (H_slab + H_therm):.0f}% is below the mixed layer")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r["t"] / YEAR, (H - H[0]) / 1e9, color=C_BLUE, lw=2,
            label="column heat content change")
    ax.plot(r["t"] / YEAR, F_int / 1e9, color="k", lw=1.2, ls="--",
            label="time-integrated surface flux")
    ax.set(xlabel="Years", ylabel="Energy (GJ m$^{-2}$)",
           title="Energy conservation: the two curves must coincide\n"
                 "(insulated bottom, k = 1 run)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "energy_budget.png")

    # Check bottom boundary condition
    uf = long_run["theta"][:, -1]
    dzl = long_run["dz"]
    grad_bot = (3 * uf[-1] - 4 * uf[-2] + uf[-3]) / (2 * dzl)
    grad_scale = np.ptp(uf) / p.z_max
    check(f"d(theta)/dz at z_max = {grad_bot:.2e} K/m", abs(grad_bot) < 1e-3 * grad_scale,
          "relative to column-mean gradient")
    from scipy.special import erfc
    frac = uf[-1] / uf[0]
    theory = erfc(p.z_max / (2.0 * np.sqrt(p.kappa * 150 * YEAR)))
    check(f"bottom anomaly after 150 yr = {uf[-1]:.2e} C = {100 * frac:.2f}% of surface",
          abs(frac) < 0.01,
          f"tol 1%; semi-infinite theory predicts ~erfc = {100 * theory:.2f}%")

    r_half = solve_model(replace(p, z_max=1350.0, Nz=121))
    d35_full = runs["110 m + k = 1 cm$^2$/s"]["dT"][-1]
    d35_half = r_half["dT"][-1]
    check(f"dT(35 yr): z_max 2700 m -> {d35_full:.4f}, 1350 m -> {d35_half:.4f}",
          abs(d35_full - d35_half) / d35_full < 1e-3, "tol 0.1%")

    # Check grid convergence (not sure what this is)
    r_fine = solve_model(replace(p, Nz=2 * p.Nz - 1))
    d35_fine = r_fine["dT"][-1]
    rel = abs(d35_fine - d35_full) / d35_fine
    check(f"dT(35 yr): Nz={p.Nz} -> {d35_full:.4f}, Nz={2*p.Nz-1} -> {d35_fine:.4f} "
          f"(rel diff {rel:.1e})", rel < 1e-2, "tol 1%")

    # Check monotone warming (again not sure what this is)
    for name, rr in list(runs.items()) + [("150-yr long run", long_run)]:
        mono = np.all(np.diff(rr["dT"]) > -1e-10)
        bounded = np.all(rr["dT"] <= rr["p"].dT_eq * (1 + 1e-9))
        check(f"{name}: monotone={mono}, bounded={bounded}", mono and bounded)

    section("DONE — figures saved in ./" + OUTDIR)


if __name__ == "__main__":
    main()
