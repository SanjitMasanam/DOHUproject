"""
1-layer EBM (slab mixed layer) + diffusive thermocline.
Fill in comments as you work through each section.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
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
    #
    #
    kappa: float = 1.0 * CM2S_TO_M2S
    h_ml: float = 110.0
    dT_eq: float = 4.2
    F0: float = 4.3
    rho_cp: float = 4.186e6
    z_max: float = 2700.0
    Nz: int = 241
    t_final: float = 35.0 * YEAR

    @property
    def c_ml(self):
        #
        return self.rho_cp * self.h_ml

    @property
    def lam(self):
        #
        return self.F0 / self.dT_eq

    @property
    def D(self):
        #
        return self.rho_cp * self.kappa

    @property
    def tau_ml(self):
        #
        #
        return self.c_ml / self.lam


# =================================================================
# Core numerics
# =================================================================
def make_rhs(dz, p):
    #
    #
    def rhs(t, u):
        du = np.empty_like(u)
        T0 = u[0]

        #
        #
        du[0] = (p.F0 - p.lam * T0 + p.D * (u[1] - T0) / dz) / p.c_ml

        #
        du[1:-1] = p.kappa * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dz**2

        #
        #
        du[-1] = p.kappa * 2.0 * (u[-2] - u[-1]) / dz**2
        return du

    return rhs


def solve_model(p, n_save=400, rtol=1e-8, atol=1e-10):
    #
    #
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
    #
    #
    return p.dT_eq * (1.0 - np.exp(-t / p.tau_ml))


def e_folding_time(t, dT, p):
    #
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

    #
    #
    p = Params()
    section("MODEL PARAMETERS (defaults: 110 m mixed layer, k = 1 cm^2/s)")
    print(f"  F0      = {p.F0} W/m^2       dT_eq = {p.dT_eq} C")
    print(f"  lam     = F0/dT_eq = {p.lam:.4f} W m^-2 K^-1")
    print(f"  h_ml    = {p.h_ml} m  ->  c = rho_cp*h = {p.c_ml:.3e} J m^-2 K^-1")
    print(f"  kappa   = {p.kappa:.1e} m^2/s ->  D = rho_cp*kappa = {p.D:.1f} W m^-1 K^-1")
    print(f"  tau_ml  = c/lam = {p.tau_ml / YEAR:.2f} yr  (isolated-slab e-folding time)")
    print(f"  grid: Nz = {p.Nz}, dz = {p.z_max / (p.Nz - 1):.2f} m, z_max = {p.z_max} m")

    #
    #
    runs = {
        "63 m mixed layer only":  solve_model(replace(p, h_ml=63.0, kappa=0.0)),
        "110 m mixed layer only": solve_model(replace(p, kappa=0.0)),
        "110 m + k = 1 cm$^2$/s": solve_model(p),
        "110 m + k = 2 cm$^2$/s": solve_model(replace(p, kappa=2.0 * CM2S_TO_M2S)),
    }

    #
    #
    section("CHECK 1: solver diagnostics (k = 1 run) and finiteness (all runs)")
    s = runs["110 m + k = 1 cm$^2$/s"]["sol"]
    print(f"  success={s.success}  nfev={s.nfev}  njev={s.njev}  nlu={s.nlu}")
    all_finite = all(np.all(np.isfinite(r["theta"])) for r in runs.values())
    check("all runs finite (no NaN/Inf)", all_finite)

    #
    #
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
           title="Response to instant CO$_2$ doubling — Hansen et al. (1984) Fig. 16")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig16_reproduction.png")

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

    #
    #
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

    section("CHECK 4: isolated-slab e-folding times vs paper text")
    for name, paper_val in [("63 m mixed layer only", 8.0),
                            ("110 m mixed layer only", 15.0)]:
        r = runs[name]
        te = e_folding_time(r["t"], r["dT"], r["p"]) / YEAR
        check(f"e-folding [{name}] = {te:.1f} yr vs paper '~{paper_val:.0f} yr'",
              abs(te - paper_val) / paper_val < 0.15, "tol 15% (paper rounds)")

    #
    #
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

    #
    #
    section("CHECK 6: e-folding time vs dT_eq (k = 1, 110 m) vs paper text")
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

    #
    #
    section("CHECK 7: diffusivity sweep (h = 110 m) — monotone slowdown with k")
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

    #
    #
    section("CHECK 8: mixed-layer depth sweep (k = 1 cm^2/s)")
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

    #
    #
    section("FIGURE: depth_profiles.png (k = 1 run)")
    r = runs["110 m + k = 1 cm$^2$/s"]
    fig, ax = plt.subplots(figsize=(5.5, 6))
    cmap = plt.cm.Blues
    profile_years = [1, 2, 5, 10, 20, 35]
    for yr, shade in zip(profile_years, np.linspace(0.35, 0.95, len(profile_years))):
        i = np.argmin(np.abs(r["t"] - yr * YEAR))
        ax.plot(r["theta"][:, i], r["z"], color=cmap(shade), lw=2,
                label=f"t = {yr} yr")
    ax.set_ylim(800, 0)
    ax.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel="Depth z (m)",
           title="Anomaly diffusing into the thermocline\n(110 m + k = 1 cm$^2$/s)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, "depth_profiles.png")

    #
    #
    section("CHECK 9: energy conservation + heat partition (k = 1 and k = 2)")
    r = runs["110 m + k = 1 cm$^2$/s"]
    pp, dz, th = r["p"], r["dz"], r["theta"]
    H = (pp.c_ml * th[0]
         + pp.rho_cp * dz * th[1:-1].sum(axis=0)
         + pp.rho_cp * (dz / 2.0) * th[-1])
    F_net = pp.F0 - pp.lam * th[0]
    F_int = np.concatenate([[0.0], cumulative_trapezoid(F_net, r["t"])])
    resid = np.abs((H - H[0]) - F_int)
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

    #
    #
    section("CHECK 10: bottom BC + semi-infinite domain")
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

    #
    #
    section("CHECK 11: grid convergence (k = 1 run)")
    r_fine = solve_model(replace(p, Nz=2 * p.Nz - 1))
    d35_fine = r_fine["dT"][-1]
    rel = abs(d35_fine - d35_full) / d35_fine
    check(f"dT(35 yr): Nz={p.Nz} -> {d35_full:.4f}, Nz={2*p.Nz-1} -> {d35_fine:.4f} "
          f"(rel diff {rel:.1e})", rel < 1e-2, "tol 1%")

    #
    #
    section("CHECK 12: monotone warming, bounded by dT_eq (all runs)")
    for name, rr in list(runs.items()) + [("150-yr long run", long_run)]:
        mono = np.all(np.diff(rr["dT"]) > -1e-10)
        bounded = np.all(rr["dT"] <= rr["p"].dT_eq * (1 + 1e-9))
        check(f"{name}: monotone={mono}, bounded={bounded}", mono and bounded)

    section("DONE — figures saved in ./" + OUTDIR)


if __name__ == "__main__":
    main()
