"""
1-layer EBM (slab mixed layer) + diffusive thermocline, following the
"Transient Response" section of Hansen et al. (1984), "Climate Sensitivity:
Analysis of Feedback Mechanisms" (Eq. 24, Appendix A, and Fig. 16).

MODEL
-----
theta(z, t) is the ocean temperature ANOMALY (deg C) relative to the
pre-forcing equilibrium; z is depth in meters, positive downward, z = 0 at
the base of the atmosphere / top of the mixed layer.

The mixed layer is a well-mixed slab of depth h whose temperature is the
surface value of the field, T_0(t) = theta(0, t). That shared node is how
the slab ODE couples to the interior PDE.

    Interior (thermocline), 0 < z < z_max:
        d(theta)/dt = kappa * d2(theta)/dz2

    Surface node (mixed-layer energy budget), z = 0:
        c * dT_0/dt = F(t) + D * d(theta)/dz |_{z=0}
        with F(t)   = F0 - lam * T_0            [Hansen Eq. 24]
             c      = rho_cp * h                [slab heat capacity, J m^-2 K^-1]
             D      = rho_cp * kappa            [so D*d(theta)/dz is the W m^-2
                                                 diffusive flux; + sign because a
                                                 warm surface over a cold interior
                                                 (d(theta)/dz < 0) must LOSE heat
                                                 downward]

    Bottom, z = z_max: insulated (zero flux). z_max is chosen large enough
    that the domain is effectively semi-infinite over the runs (checked below).
    Setting Params.semi_infinite = True makes this rigorous: solve_model then
    auto-deepens z_max (and Nz) to many diffusive penetration depths below the
    surface, so the insulated wall is never reached and the ocean behaves as an
    "effectively infinite" heat reservoir. (Equivalently, one may pass a large
    z_max by hand.)

Hansen Eq. 24 writes the surface forcing as
    F = F0 / dT_eq * (dT_eq - T_0)  =  F0 - (F0/dT_eq) * T_0,
i.e. the radiative damping coefficient is lam = F0/dT_eq. With F0 = 4.3 W m^-2
(their 3-D model's flux into the ocean at CO2 doubling) and dT_eq = 4.2 C
(their equilibrium sensitivity), lam ~ 1.02 W m^-2 K^-1.

Fig. 16 of the paper shows T_0(t) over 0-35 yr for four configurations:
    - 63 m mixed layer only (kappa = 0)          [solid]
    - 110 m mixed layer only (kappa = 0)         [solid]
    - 110 m + thermocline, k = 1 cm^2 s^-1       [dotted]
    - 110 m + thermocline, k = 2 cm^2 s^-1       [dotted]
This script reproduces that figure and validates against every number the
paper states in the text (e-folding times, the 102-yr / 2.65 C checkpoint).

WHAT WAS WRONG IN THE PREVIOUS VERSION OF THIS SCRIPT
-----------------------------------------------------
The discretization (method of lines, prognostic surface node, ghost-point
bottom Neumann, BDF) was sound and is kept. The physics parameters were not:

  1. D_coef = 2.0 W m^-1 K^-1 was ~200x too small. Dimensionally the
     diffusive heat flux is -rho_cp*kappa*d(theta)/dz, so D must equal
     rho_cp*kappa (~420 W m^-1 K^-1 for k = 1 cm^2/s). With D ~ 2 the
     thermocline was effectively decoupled from the mixed layer.
  2. The coupling term had the wrong sign (c*dT0/dt = F - lam*T0 - D*dT/dz).
     With z positive downward the correct budget is ... + D*dT/dz: when the
     surface is warmer than the water below (dT/dz < 0), the slab must lose
     heat. The old sign made it gain heat (anti-diffusive).
  3. The initial condition was an absolute temperature (280 K), but
     F0 - lam*T_0 is an anomaly equation; the paper starts from dT = 0.
  4. Parameter values: F = 7 -> F0 = 4.3 W m^-2; c = 1e8 (~24 m of water)
     -> rho_cp*110 m = 4.6e8; kappa = 5e-5 -> 1e-4 / 2e-4 m^2 s^-1;
     lam = 1 -> derived as F0/dT_eq.
  5. T_final = 5 yr -> 35 yr (Fig. 16 window) plus a 150-yr run for the
     paper's 102-yr checkpoint.

OUTPUTS
-------
All figures are saved into ./1lyEBM_diffusive/ (created if absent).
All sanity checks print PASS/FAIL blocks to stdout.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass, replace
from scipy.integrate import solve_ivp
from scipy.sparse import diags

# scipy renamed cumtrapz -> cumulative_trapezoid in 1.6; support both so the
# script runs on this machine's scipy 1.5 as well as newer stacks.
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

# =================================================================
# Constants and parameters
# =================================================================
YEAR = 365.25 * 24 * 3600.0          # seconds per year
CM2S_TO_M2S = 1.0e-4                 # 1 cm^2/s = 1e-4 m^2/s (paper quotes k in cm^2/s)

OUTDIR = "figures_diffusiveChecks"          # all figures go here (created below)

# Semi-infinite ("effectively infinite ocean") domain sizing. When
# Params.semi_infinite is True, solve_model places the insulated bottom
# SEMI_INF_PAD * (2*sqrt(kappa*t_final)) below the surface. The anomaly at
# depth z scales like erfc(z / (2*sqrt(kappa*t))), so at the padded depth the
# residual is erfc(SEMI_INF_PAD) = erfc(4) ~ 1.5e-8 of the surface value --
# i.e. the wall is genuinely never felt. SEMI_INF_MAX_NZ caps the grid so
# deep/long infinite runs stay affordable (dz coarsens if the cap is hit).
SEMI_INF_PAD = 4.0
SEMI_INF_MAX_NZ = 2001

# Fixed-order categorical palette (colorblind-safe subset of Okabe-Ito,
# validated for CVD separation). Assigned by entity, never cycled.
C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_PINK = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")


@dataclass(frozen=True)
class Params:
    """Physical + numerical parameters. Defaults reproduce the paper's
    "110 m mixed layer + k = 1 cm^2/s thermocline" configuration (the main
    dotted curve of Fig. 16). Use dataclasses.replace(p, ...) to override
    single values for experiments -- frozen so a run can't mutate them.

    Provenance of the physical values (Hansen et al. 1984):
      F0     = 4.3 W m^-2   p. 156: "our 3-D model yields F0 = 4.3 W m^-2"
                            (flux into ocean after instant CO2 doubling,
                            stratosphere equilibrated)
      dT_eq  = 4.2 C        Fig. 16 caption: "equilibrium sensitivity = 4.2 C"
      h_ml   = 110 m        p. 155: global-mean annual-maximum mixed layer
                            depth from observations (63 m is the 3-D model's
                            capped value, used for one Fig. 16 curve)
      kappa  = 1 cm^2/s     p. 156: "k ~ 1 cm^2/s provides a reasonable
                            global fit"; Fig. 16 also shows k = 2
      rho_cp = 4.186e6      J m^-3 K^-1 for seawater; gives the paper's
                            "~15 yr" isolated 110-m mixed-layer response time
                            (tau = c*dT_eq/F0 = 14.3 yr)
    """
    kappa: float = 1.0 * CM2S_TO_M2S   # thermocline diffusivity [m^2/s]
    h_ml: float = 110.0                # mixed layer depth [m]
    dT_eq: float = 4.2                 # equilibrium warming for 2xCO2 [C]
    F0: float = 4.3                    # flux into ocean at t=0 [W/m^2]
    rho_cp: float = 4.186e6            # vol. heat capacity of seawater [J m^-3 K^-1]
    z_max: float = 2700.0              # domain depth [m]; >> diffusive penetration
    Nz: int = 241                      # grid points => dz = 11.25 m
    t_final: float = 35.0 * YEAR       # Fig. 16 window; overridden for long runs
    semi_infinite: bool = False        # if True, solve_model auto-deepens z_max/Nz
                                       # so the insulated bottom is never reached
                                       # ("effectively infinite ocean" limit)

    # ---- derived quantities (comments give units) ----
    @property
    def c_ml(self):
        return self.rho_cp * self.h_ml          # slab heat capacity [J m^-2 K^-1]

    @property
    def lam(self):
        return self.F0 / self.dT_eq             # radiative damping [W m^-2 K^-1]

    @property
    def D(self):
        return self.rho_cp * self.kappa         # flux coefficient [W m^-1 K^-1]

    @property
    def tau_ml(self):
        # e-folding time of the ISOLATED mixed layer (kappa = 0), from
        # Appendix A Eqs. A11-A12: tau = c/lam = c*dT_eq/F0.
        return self.c_ml / self.lam             # [s]


# =================================================================
# Core numerics
# =================================================================
def make_rhs(dz, p):
    """RHS of the method-of-lines system for a given grid spacing.

    State vector u: u[0] = T_0 = theta(z=0) (the mixed-layer slab),
    u[1:] = theta at interior/bottom nodes. The slab is prognostic (it has
    its own heat capacity), so it stays in the state vector instead of
    being eliminated into a boundary condition.
    """
    def rhs(t, u):
        du = np.empty_like(u)
        T0 = u[0]

        # --- surface node: mixed-layer energy budget -------------------
        # F0 - lam*T0        : net radiative flux into the ocean (Eq. 24)
        # + D*(u[1]-T0)/dz   : diffusive exchange with the thermocline.
        #   Sign: if the slab is warmer than the water below, (u[1]-T0) < 0
        #   and the slab loses heat downward, as it must.
        du[0] = (p.F0 - p.lam * T0 + p.D * (u[1] - T0) / dz) / p.c_ml

        # --- interior nodes: plain diffusion ----------------------------
        du[1:-1] = p.kappa * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dz**2

        # --- bottom node: zero-flux Neumann via ghost-point mirror ------
        # (theta_ghost = u[-2] makes the centered gradient vanish at z_max)
        du[-1] = p.kappa * 2.0 * (u[-2] - u[-1]) / dz**2
        return du

    return rhs


def solve_model(p, n_save=400, rtol=1e-8, atol=1e-10):
    """Integrate the model from a zero-anomaly initial state.

    Returns a dict with the saved times [s], the full field theta(z, t),
    the surface series dT(t) = theta(0, t), the grid, and the raw solver
    object (for diagnostics).

    Numerics notes:
      - IC is theta = 0 everywhere: the paper's experiment is an instant
        CO2 doubling applied to an equilibrated ocean, so anomalies start
        at zero and F(t=0) = F0.
      - BDF because the diffusion operator is stiff (explicit RK would need
        dt < dz^2/(2*kappa) ~ 2 weeks; BDF takes ~100 steps for 35 yr).
      - jac_sparsity: the Jacobian is tridiagonal; telling the solver saves
        it from finite-differencing a dense Nz x Nz matrix.
      - n_save = 400 output times => smooth dT(t) curves and accurate
        trapezoid time-integrals in the energy check.
      - semi_infinite: the domain is auto-deepened to SEMI_INF_PAD diffusive
        penetration depths so the insulated bottom is never felt (see the
        SEMI_INF_* constants). We rebind p via replace() so the returned p,
        grid, and all energy/BC bookkeeping stay consistent with the grid
        actually solved. kappa == 0 (pure slab) is a no-op: with no diffusion
        the depth is irrelevant.
    """
    if p.semi_infinite and p.kappa > 0.0:
        dz0 = p.z_max / (p.Nz - 1)                     # keep original resolution
        z_max = max(p.z_max, SEMI_INF_PAD * 2.0 * np.sqrt(p.kappa * p.t_final))
        Nz = min(int(np.ceil(z_max / dz0)) + 1, SEMI_INF_MAX_NZ)  # cap cost; dz coarsens if hit
        p = replace(p, z_max=z_max, Nz=Nz, semi_infinite=False)

    z = np.linspace(0.0, p.z_max, p.Nz)
    dz = z[1] - z[0]
    u0 = np.zeros(p.Nz)                       # anomaly starts at zero

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
    """Closed-form solution for the ISOLATED mixed layer (kappa = 0).

    c*dT/dt = F0 - lam*T  =>  T(t) = dT_eq * (1 - exp(-t/tau)),
    tau = c/lam (Hansen Appendix A, Eqs. A11-A12). Used as an exact target
    the numerical solver must reproduce.
    """
    return p.dT_eq * (1.0 - np.exp(-t / p.tau_ml))


def e_folding_time(t, dT, p):
    """First time at which dT reaches (1 - 1/e) of dT_eq, by linear
    interpolation on the (monotone) surface series. NaN if never reached
    within the run."""
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
    """Recessive grid/axes so the data, not the furniture, dominates."""
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):          # (list-indexing of spines needs mpl>=3.4)
        ax.spines[side].set_visible(False)


# =================================================================
# Main experiment suite
# =================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # -----------------------------------------------------------------
    # Parameter echo: derived quantities, so unit errors are visible at
    # a glance (this is exactly where the old version went wrong).
    # -----------------------------------------------------------------
    p = Params()
    section("MODEL PARAMETERS (defaults: 110 m mixed layer, k = 1 cm^2/s)")
    print(f"  F0      = {p.F0} W/m^2       dT_eq = {p.dT_eq} C")
    print(f"  lam     = F0/dT_eq = {p.lam:.4f} W m^-2 K^-1")
    print(f"  h_ml    = {p.h_ml} m  ->  c = rho_cp*h = {p.c_ml:.3e} J m^-2 K^-1")
    print(f"  kappa   = {p.kappa:.1e} m^2/s ->  D = rho_cp*kappa = {p.D:.1f} W m^-1 K^-1")
    print(f"  tau_ml  = c/lam = {p.tau_ml / YEAR:.2f} yr  (isolated-slab e-folding time)")
    print(f"  grid: Nz = {p.Nz}, dz = {p.z_max / (p.Nz - 1):.2f} m, z_max = {p.z_max} m")

    # -----------------------------------------------------------------
    # The four Fig. 16 configurations. kappa = 0 turns the model into the
    # pure slab EBM (solid curves); k = 1, 2 cm^2/s are the dotted curves.
    # -----------------------------------------------------------------
    runs = {
        "63 m mixed layer only":  solve_model(replace(p, h_ml=63.0, kappa=0.0)),
        "110 m mixed layer only": solve_model(replace(p, kappa=0.0)),
        "110 m + k = 1 cm$^2$/s": solve_model(p),
        "110 m + k = 2 cm$^2$/s": solve_model(replace(p, kappa=2.0 * CM2S_TO_M2S)),
    }

    # -----------------------------------------------------------------
    # CHECK 1: solver diagnostics + finiteness. A stiff solver that
    # struggles (huge nfev, tiny steps) usually signals a bad Jacobian or
    # an unstable (e.g. anti-diffusive) formulation, so surface it.
    # -----------------------------------------------------------------
    section("CHECK 1: solver diagnostics (k = 1 run) and finiteness (all runs)")
    s = runs["110 m + k = 1 cm$^2$/s"]["sol"]
    print(f"  success={s.success}  nfev={s.nfev}  njev={s.njev}  nlu={s.nlu}")
    all_finite = all(np.all(np.isfinite(r["theta"])) for r in runs.values())
    check("all runs finite (no NaN/Inf)", all_finite)

    # -----------------------------------------------------------------
    # FIGURE 1: the Fig. 16 reproduction (the main deliverable).
    # Solid = slab only, dotted = slab + diffusive thermocline, matching
    # the paper's line styles; axes limits match the paper's (0-35 yr,
    # 0-5 C) for a direct visual comparison.
    # -----------------------------------------------------------------
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

    # CHECK 2: endpoint values against the paper's curves (read off the
    # printed figure by eye, so tolerance is generous: +/- 0.25 C).
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

    # -----------------------------------------------------------------
    # CHECK 3 + FIGURE: kappa = 0 limit vs the exact analytic solution.
    # This isolates the ODE part (surface budget + time integrator) from
    # the PDE part: if this fails, the bug is in the slab equation.
    # -----------------------------------------------------------------
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

    # CHECK 4: slab e-folding times the paper states in its text:
    # "~8 years" for the 63 m model (p. 155, discussing Fig. 3) and
    # "~15 years" for the isolated 110 m layer (p. 155).
    section("CHECK 4: isolated-slab e-folding times vs paper text")
    for name, paper_val in [("63 m mixed layer only", 8.0),
                            ("110 m mixed layer only", 15.0)]:
        r = runs[name]
        te = e_folding_time(r["t"], r["dT"], r["p"]) / YEAR
        check(f"e-folding [{name}] = {te:.1f} yr vs paper '~{paper_val:.0f} yr'",
              abs(te - paper_val) / paper_val < 0.15, "tol 15% (paper rounds)")

    # -----------------------------------------------------------------
    # CHECK 5 + FIGURE: the strongest quantitative anchor in the paper.
    # p. 156: with h = 110 m, k = 1 cm^2/s, dT_eq = 4.2 C, "the time
    # required to reach a response of 2.65 C [= (1-1/e)*4.2] is 102 years".
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # CHECK 6 + FIGURE: response time vs climate sensitivity.
    # Paper p. 156: with k = 1 cm^2/s and h = 110 m, the e-folding time is
    # 27 yr for dT_eq = 2 C and 55 yr for dT_eq = 3 C (102 yr for 4.2 C).
    # This is the paper's central point: ocean response time grows FASTER
    # than linearly with sensitivity, because higher f both weakens the
    # restoring flux and lets diffusion engage more of the deep ocean.
    # -----------------------------------------------------------------
    section("CHECK 6: e-folding time vs dT_eq (k = 1, 110 m) vs paper text")
    sens_runs = {}
    for dte, paper_val, shade in [(2.0, 27.0, 0.45), (3.0, 55.0, 0.65), (4.2, 102.0, 0.9)]:
        r = solve_model(replace(p, dT_eq=dte, t_final=150.0 * YEAR))
        te = e_folding_time(r["t"], r["dT"], r["p"]) / YEAR
        sens_runs[dte] = (r, te, shade)
        check(f"dT_eq = {dte} C: e-folding = {te:.0f} yr vs paper {paper_val:.0f} yr",
              abs(te - paper_val) / paper_val < 0.15, "tol 15%")

    # Sequential single-hue ramp: dT_eq is a magnitude, so encode it as
    # light -> dark of one hue rather than unrelated categorical colors.
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

    # -----------------------------------------------------------------
    # CHECK 7 + FIGURE: diffusivity sweep. More diffusion = more heat
    # drained into the thermocline = slower surface warming, so dT(t)
    # must decrease MONOTONICALLY with k at any fixed time.
    # -----------------------------------------------------------------
    section("CHECK 7: diffusivity sweep (h = 110 m) — monotone slowdown with k")
    k_values = [0.0, 0.5, 1.0, 2.0, 4.0]                    # cm^2/s
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
    # center-right sits in the gap between the k=0 and k=0.5 curves, keeping
    # the legend clear of the dT_eq reference line at the top
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    savefig(fig, "sweep_diffusivity.png")

    # -----------------------------------------------------------------
    # CHECK 8 + FIGURE: mixed-layer depth sweep. A deeper slab has more
    # heat capacity, so it warms more slowly -- same monotonicity logic.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # FIGURE: continuous depth-time evolution of the anomaly (k = 1 run).
    # Rather than a handful of discrete year snapshots, show every saved
    # profile theta(z) at once, colored continuously by time. A long run
    # over the full ocean depth (z_max) is needed since the deep ocean
    # barely responds within the original 35-yr window.
    # -----------------------------------------------------------------
    section("FIGURE: depth_profiles.png (k = 1 run, 1000-yr, full depth)")
    deep_run = solve_model(replace(p, t_final=1000.0 * YEAR), n_save=500)
    t_yr, z, theta = deep_run["t"] / YEAR, deep_run["z"], deep_run["theta"]
    segments = [np.column_stack([theta[:, i], z]) for i in range(theta.shape[1])]
    lc = LineCollection(segments, cmap="viridis", array=t_yr, linewidths=1.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    ax.set_xlim(theta.min(), theta.max() * 1.05)
    ax.set_ylim(p.z_max, 0)                              # depth increases downward
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Years")
    ax.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel="Depth z (m)",
           title="Continuous depth-time evolution of the thermocline anomaly\n"
                 "(110 m + k = 1 cm$^2$/s, 1000-yr run)")
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, "depth_profiles.png")

    # -----------------------------------------------------------------
    # FIGURE + CHECK: finite vs effectively-infinite ocean (k = 1, 110 m).
    # The default z_max = 2700 m has an INSULATED bottom, so once the
    # diffusive front reaches it (~577 yr here) heat can no longer drain
    # downward and the capped column saturates toward dT_eq FASTER than a
    # truly deep ocean would. semi_infinite=True auto-deepens the domain so
    # the bottom is never felt: the deep ocean keeps absorbing heat and the
    # surface lags. The two must be indistinguishable early and diverge late.
    # -----------------------------------------------------------------
    section("FIGURE: finite_vs_infinite.png  +  CHECK: finite/infinite ocean (3000-yr)")
    T_LONG = 3000.0 * YEAR
    finite = solve_model(replace(p, t_final=T_LONG))                     # z_max=2700, insulated
    infinite = solve_model(replace(p, t_final=T_LONG, semi_infinite=True))  # auto-deep
    print(f"  finite: z_max = {finite['p'].z_max:.0f} m, Nz = {finite['p'].Nz}")
    print(f"  infinite: z_max = {infinite['p'].z_max:.0f} m, Nz = {infinite['p'].Nz}")

    # Early-time agreement: before the front reaches 2700 m the insulated wall
    # is irrelevant, so the two surface series must coincide. (t_eval grids are
    # identical since only z_max/Nz differ, so compare pointwise.)
    t_yr_fi = finite["t"] / YEAR
    early = t_yr_fi < 400.0
    early_gap = np.max(np.abs(finite["dT"][early] - infinite["dT"][early]))
    check(f"max|finite - infinite| for t < 400 yr = {early_gap:.2e} C",
          early_gap < 1e-3 * p.dT_eq, "tol 1e-3*dT_eq; wall not yet reached")

    # Late-time divergence: the capped column warms toward equilibrium faster.
    dfin, dinf = finite["dT"][-1], infinite["dT"][-1]
    check(f"dT(3000 yr): finite = {dfin:.3f} C > infinite = {dinf:.3f} C",
          dfin > dinf and dfin <= p.dT_eq * (1 + 1e-9)
          and dinf <= p.dT_eq * (1 + 1e-9),
          "finite (insulated) saturates faster; both bounded by dT_eq")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_yr_fi, finite["dT"], color=C_VERM, lw=2,
            label=f"finite ($z_{{max}}$ = {p.z_max:.0f} m, insulated)")
    ax.plot(infinite["t"] / YEAR, infinite["dT"], color=C_BLUE, lw=2,
            label=r"effectively infinite ocean")
    ax.axhline(p.dT_eq, color="0.4", lw=1, ls="--")
    ax.text(2350, p.dT_eq + 0.05, r"$\Delta T_{eq}$", color="0.3", fontsize=9)
    ax.set(xlabel="Years", ylabel=r"$\Delta T$ ($\degree$C)", xlim=(0, 3000),
           ylim=(0, p.dT_eq * 1.05),
           title="Finite (insulated) vs effectively infinite ocean\n"
                 "(110 m + k = 1 cm$^2$/s): identical early, diverge late")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, "finite_vs_infinite.png")

    # -----------------------------------------------------------------
    # CHECK 9 + FIGURE: energy conservation. The bottom is insulated, so
    # the change in total heat content must equal the time-integrated
    # surface flux:  c*T0 + rho_cp*Int(theta dz) - 0  ==  Int(F0-lam*T0)dt.
    # We use the discretely-conserved heat functional (slab + dz-weighted
    # interior + half-weighted bottom node, matching the finite-volume
    # reading of the stencil) so this check isolates TIME-integration
    # error rather than O(dz) bookkeeping noise.
    # -----------------------------------------------------------------
    section("CHECK 9: energy conservation + heat partition (k = 1 and k = 2)")
    r = runs["110 m + k = 1 cm$^2$/s"]
    pp, dz, th = r["p"], r["dz"], r["theta"]
    H = (pp.c_ml * th[0]                                  # slab
         + pp.rho_cp * dz * th[1:-1].sum(axis=0)          # interior cells
         + pp.rho_cp * (dz / 2.0) * th[-1])               # half-width bottom cell
    F_net = pp.F0 - pp.lam * th[0]
    F_int = np.concatenate([[0.0], cumulative_trapezoid(F_net, r["t"])])
    resid = np.abs((H - H[0]) - F_int)
    rel_err = resid.max() / F_int[-1]
    check(f"max |dH - Int(F dt)| / Int(F dt) = {rel_err:.2e}", rel_err < 5e-3,
          "tol 0.5%; violation would mean flux/capacity bookkeeping is inconsistent")

    # Physical probe: where did the heat go? The fraction stored below the
    # mixed layer is exactly why the dotted Fig. 16 curves are slower.
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

    # -----------------------------------------------------------------
    # CHECK 10: bottom boundary condition. On the most demanding run
    # (150 yr), the one-sided gradient at z_max should be ~0 (Neumann)
    # and the bottom anomaly itself ~0 (domain effectively semi-infinite).
    # -----------------------------------------------------------------
    section("CHECK 10: bottom BC + semi-infinite domain")
    uf = long_run["theta"][:, -1]
    dzl = long_run["dz"]
    grad_bot = (3 * uf[-1] - 4 * uf[-2] + uf[-3]) / (2 * dzl)   # 2nd-order one-sided
    grad_scale = np.ptp(uf) / p.z_max                            # typical gradient
    check(f"d(theta)/dz at z_max = {grad_bot:.2e} K/m", abs(grad_bot) < 1e-3 * grad_scale,
          "relative to column-mean gradient")
    # Semi-infinite diffusion theory predicts the anomaly reaching depth z
    # scales like erfc(z / (2*sqrt(kappa*t))) of the surface value -- about
    # 0.5% of the surface anomaly at z_max after 150 yr. So a tiny nonzero
    # bottom value is EXPECTED; the physical requirement is only that it be
    # a negligible fraction of the surface anomaly (the domain-halving test
    # below is the decisive proof the wall doesn't affect the answer).
    from scipy.special import erfc
    frac = uf[-1] / uf[0]
    theory = erfc(p.z_max / (2.0 * np.sqrt(p.kappa * 150 * YEAR)))
    check(f"bottom anomaly after 150 yr = {uf[-1]:.2e} C = {100 * frac:.2f}% of surface",
          abs(frac) < 0.01,
          f"tol 1%; semi-infinite theory predicts ~erfc = {100 * theory:.2f}%")

    # Halve the domain: if z_max is truly 'effectively infinite', the
    # 35-yr surface answer must not care. (Nz chosen to keep dz identical
    # so this isolates the domain-size effect from resolution.)
    r_half = solve_model(replace(p, z_max=1350.0, Nz=121))
    d35_full = runs["110 m + k = 1 cm$^2$/s"]["dT"][-1]
    d35_half = r_half["dT"][-1]
    check(f"dT(35 yr): z_max 2700 m -> {d35_full:.4f}, 1350 m -> {d35_half:.4f}",
          abs(d35_full - d35_half) / d35_full < 1e-3, "tol 0.1%")

    # -----------------------------------------------------------------
    # CHECK 11: grid-refinement convergence. Doubling resolution should
    # barely move the surface answer; if it moves a lot, the first-order
    # surface-flux discretization is under-resolved.
    # -----------------------------------------------------------------
    section("CHECK 11: grid convergence (k = 1 run)")
    r_fine = solve_model(replace(p, Nz=2 * p.Nz - 1))
    d35_fine = r_fine["dT"][-1]
    rel = abs(d35_fine - d35_full) / d35_fine
    check(f"dT(35 yr): Nz={p.Nz} -> {d35_full:.4f}, Nz={2*p.Nz-1} -> {d35_fine:.4f} "
          f"(rel diff {rel:.1e})", rel < 1e-2, "tol 1%")

    # -----------------------------------------------------------------
    # CHECK 12: qualitative physical behavior on every run: warming is
    # monotone (constant forcing, no oscillatory physics in this model)
    # and can never exceed the equilibrium sensitivity.
    # -----------------------------------------------------------------
    section("CHECK 12: monotone warming, bounded by dT_eq (all runs)")
    for name, rr in list(runs.items()) + [("150-yr long run", long_run)]:
        mono = np.all(np.diff(rr["dT"]) > -1e-10)
        bounded = np.all(rr["dT"] <= rr["p"].dT_eq * (1 + 1e-9))
        check(f"{name}: monotone={mono}, bounded={bounded}", mono and bounded)

    section("DONE — figures saved in ./" + OUTDIR)


if __name__ == "__main__":
    main()
