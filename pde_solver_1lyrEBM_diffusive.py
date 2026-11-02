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

    Bottom, z = z_max: insulated (zero flux). By DEFAULT the domain is CLOSED
    at z_max = 2500 m: that floor is a real physical edge, not an approximation
    to a semi-infinite ocean. The ocean is then a finite heat reservoir, so on
    long enough runs the diffusive front reaches the floor, the whole column
    warms, and the surface equilibrates to dT_eq.

    Setting Params.semi_infinite = True is an OPT-IN escape hatch that recovers
    the old "effectively infinite ocean" behavior: solve_model auto-deepens
    z_max (and Nz) to many diffusive penetration depths below the surface, so
    the insulated wall is never reached. It is off by default and used only for
    the finite-vs-infinite comparison below. (Equivalently, one may pass a
    large z_max by hand.)

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

from figure_paths import CHECKS_DIR

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
F2X = 5.35 * np.log(2)               # 2xCO2 forcing [W m^-2], Myhre et al. (1998)
T_2X = 70.0 * YEAR                   # linear ramp reaches F2X here (1%/yr CO2 doubling time)

# All figures from this module's verification suite go here (created below).
# figure_paths owns the location so it moves with the rest of the tree; this
# module keeps its own single-panel plot style, which is deliberately different
# from the longrun scripts' multi-panel house style.
OUTDIR = CHECKS_DIR

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
    epsilon: float = 1.0               # deep-ocean heat-uptake efficacy
                                       # (Geoffroy 2013b): the surface budget
                                       # loses eps*H while the interior receives
                                       # the unscaled diffusive flux H. 1.0 = off
                                       # (recovers the plain diffusive model).
    rho_cp: float = 4.186e6            # vol. heat capacity of seawater [J m^-3 K^-1]
    z_max: float = 2500.0              # ocean depth [m]; a CLOSED, physical floor
                                       # with zero-flux BC (finite heat reservoir)
    Nz: int = 241                      # grid points => dz = 11.25 m
    t_final: float = 35.0 * YEAR       # Fig. 16 window; overridden for long runs
    semi_infinite: bool = False        # OPT-IN escape hatch (default off): if True,
                                       # solve_model auto-deepens z_max/Nz so the
                                       # insulated bottom is never reached
                                       # ("effectively infinite ocean" limit)
    forcing: object = None             # optional F(t) [W/m^2], t in s; replaces
                                       # the constant F0 in the surface budget

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
        # F0 - lam*T0            : net radiative flux into the ocean (Eq. 24)
        # + eps*D*(u[1]-T0)/dz   : diffusive exchange with the thermocline,
        #   scaled by the heat-uptake efficacy eps (Geoffroy 2013b). eps=1
        #   is the plain diffusive model; eps>1 makes the deep uptake more
        #   effective at cooling the surface than a 1:1 balance would give.
        #   Sign: if the slab is warmer than the water below, (u[1]-T0) < 0
        #   and the slab loses heat downward, as it must. Note the INTERIOR
        #   (du[1:-1]) is deliberately left unscaled, so the surface loses
        #   eps*H while the thermocline receives H -- the (eps-1)*H imbalance
        #   is the extra top-of-atmosphere response.
        F = p.F0 if p.forcing is None else p.forcing(t)
        du[0] = (F - p.lam * T0 + p.epsilon * p.D * (u[1] - T0) / dz) / p.c_ml

        # --- interior nodes: plain diffusion ----------------------------
        du[1:-1] = p.kappa * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dz**2

        # --- bottom node: zero-flux Neumann via ghost-point mirror ------
        # (theta_ghost = u[-2] makes the centered gradient vanish at z_max)
        du[-1] = p.kappa * 2.0 * (u[-2] - u[-1]) / dz**2
        return du

    return rhs


def solve_model(p, n_save=400, rtol=1e-8, atol=1e-10, t_eval=None):
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
        trapezoid time-integrals in the energy check. Only used to build the
        default linear t_eval grid; ignored if t_eval is passed explicitly
        (e.g. a log-spaced grid to resolve early times on a log-t plot).
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

    if t_eval is None:
        t_eval = np.linspace(0.0, p.t_final, n_save)

    sparsity = diags([1, 1, 1], [-1, 0, 1], shape=(p.Nz, p.Nz))
    sol = solve_ivp(
        make_rhs(dz, p), t_span=(0.0, p.t_final), y0=u0, method="BDF",
        t_eval=t_eval,
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


def analytic_diffusive_profile(z, t, p):
    """Held's exact step-response profile for the purely diffusive ocean (no
    slab, semi-infinite), Carslaw & Jaeger (1959) sec. 2.7:

        h(zeta, tau) = erfc(zeta/(2 sqrt(tau))) - e^(zeta+tau) erfc(zeta/(2 sqrt(tau)) + sqrt(tau))

    with zeta = z/H, tau = t/T, H = D/lam and T = H^2/kappa. The second term
    uses erfcx(x) = e^(x^2) erfc(x) to avoid overflow. Returns theta [C].
    """
    from scipy.special import erfc, erfcx
    H = p.D / p.lam
    zeta, tau = z / H, t / (H**2 / p.kappa)
    x = zeta / (2 * np.sqrt(tau))
    return p.dT_eq * (erfc(x) - erfcx(x + np.sqrt(tau)) * np.exp(-x**2))


def analytic_diffusive_surface(t, t_knots, F_knots, p):
    """Surface theta [C] of the purely diffusive ocean (no slab, semi-infinite)
    for piecewise-linear forcing through (t_knots, F_knots), t_knots[0] = 0 and
    F held constant after the last knot. By Duhamel on the step response
    h(tau) = 1 - erfcx(sqrt(tau)): the jump F_knots[0] at t = 0 gives
    F_knots[0]*h, and a ramp of slope s starting at t_k gives
    s*T*H((t - t_k)/T), with H(tau) = int_0^tau h = tau + 1 - 2 sqrt(tau/pi)
    - erfcx(sqrt(tau)). A ramp on [t_k, t_k+1] is a ramp at t_k minus one at
    t_k+1.
    """
    from scipy.special import erfcx
    T = (p.D / p.lam)**2 / p.kappa
    h = lambda x: 1 - erfcx(np.sqrt(x))
    H = lambda x: x + 1 - 2 * np.sqrt(x / np.pi) - erfcx(np.sqrt(x))
    tau = lambda dt: np.maximum(dt, 0.0) / T
    t = np.asarray(t, float)[:, None]
    s = np.diff(F_knots) / np.diff(t_knots)
    ramps = s * T * (H(tau(t - t_knots[:-1])) - H(tau(t - t_knots[1:])))
    return (F_knots[0] * h(tau(t[:, 0])) + ramps.sum(axis=1)) / p.lam


# -----------------------------------------------------------------
# Beck & Woodbury (2013) X32B10T0: slab 0 < x < 1 heated by convection
# from a fluid at T = 1 through x = 0 (Biot number Bi), insulated at x = 1,
# T = 0 initially. All arguments and results are dimensionless. Equation
# numbers are the report's (X32B1-n).
# -----------------------------------------------------------------
def beck_U(x, t, Bi):
    """Eq. 10, U = e^(Bi x + Bi^2 t) erfc(x/sqrt(4t) + Bi sqrt(t)), in the
    erfcx form that does not overflow at large Bi."""
    from scipy.special import erfcx
    return np.exp(-x**2 / (4 * t)) * erfcx(x / np.sqrt(4 * t) + Bi * np.sqrt(t))


def beck_short(x, t, Bi):
    """Eq. 7a: semi-infinite solution, the insulated wall not yet felt."""
    from scipy.special import erfc
    return erfc(x / np.sqrt(4 * t)) - beck_U(x, t, Bi)


def beck_intermediate(x, t, Bi):
    """Eq. 9: Eq. 7a plus its mirror image in the insulated wall at x = 1."""
    from scipy.special import erfc
    return beck_short(x, t, Bi) + erfc((2 - x) / np.sqrt(4 * t)) - beck_U(2 - x, t, Bi)


def beck_eigenvalues(Bi, m):
    """First m roots of tan(beta) = Bi/beta (Eq. 13), solved as
    beta sin(beta) - Bi cos(beta) = 0 so there is no tan pole; the m-th root
    lies in [(m-1) pi, (m-1) pi + pi/2], where that function changes sign."""
    from scipy.optimize import brentq
    f = lambda b: b * np.sin(b) - Bi * np.cos(b)
    return np.array([brentq(f, k * np.pi, k * np.pi + np.pi / 2) for k in range(m)])


def beck_long(x, t, Bi, m_max):
    """Eq. 12a: eigenfunction series truncated at m_max terms."""
    b = beck_eigenvalues(Bi, m_max)[:, None]
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    shape = x.shape
    x, t = x.ravel()[None, :], t.ravel()[None, :]
    terms = ((b**2 + Bi**2) * np.cos(b * (1 - x)) * np.cos(b)
             / (b**2 * (b**2 + Bi**2 + Bi)) * np.exp(-b**2 * t))
    return (1 - 2 * Bi * terms.sum(axis=0)).reshape(shape)


def beck_windows(x, A):
    """Times at which the short (Eq. 6) and intermediate (Eq. 8) forms stop
    being accurate to 10^-A; the long-time series takes over after the second
    (Eq. 11). At x = 0 the two coincide and there is no intermediate window."""
    return 0.12 * (2 - x)**2 / A, 0.12 * (2 + x)**2 / A


def beck_m_max(t, A):
    """Eq. 14: series terms needed for 10^-A accuracy at times >= t, rounded
    rather than floored as in the report's own fdX32B10T0.m -- the floor of
    Eq. 14 keeps only 2 terms at x = 1 and misses 10^-6 by an order of
    magnitude against the report's tables."""
    return max(1, int(0.5 + np.sqrt(A * np.log(10) / t) / np.pi))


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
    path = OUTDIR / name
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
    OUTDIR.mkdir(parents=True, exist_ok=True)

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
    # FIGURE + CHECK: the model's closed ocean vs the old effectively-infinite
    # limit (k = 1, 110 m). The DEFAULT model is the closed column: z_max =
    # 2500 m with an insulated floor, so once the diffusive front reaches it
    # (~577 yr here) heat can no longer drain downward, the whole column warms
    # and the surface equilibrates to dT_eq. semi_infinite=True is the opt-in
    # escape hatch that auto-deepens the domain so the floor is never felt:
    # the deep ocean keeps absorbing heat and the surface lags indefinitely.
    # The two must be indistinguishable early and diverge late -- this figure
    # is what quantifies how much the 2500 m floor actually matters.
    # -----------------------------------------------------------------
    section("FIGURE: finite_vs_infinite.png  +  CHECK: closed vs infinite ocean (3000-yr)")
    T_LONG = 3000.0 * YEAR
    finite = solve_model(replace(p, t_final=T_LONG))                     # default: closed at 2500 m
    infinite = solve_model(replace(p, t_final=T_LONG, semi_infinite=True))  # opt-in auto-deep
    print(f"  finite: z_max = {finite['p'].z_max:.0f} m, Nz = {finite['p'].Nz}")
    print(f"  infinite: z_max = {infinite['p'].z_max:.0f} m, Nz = {infinite['p'].Nz}")

    # Early-time agreement: before the front reaches 2500 m the insulated wall
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

    # The defining consequence of a closed domain: run it long enough and the
    # 2500 m ocean must reach a UNIFORM anomaly equal to dT_eq -- no residual
    # gradient, since there is nowhere left for heat to drain to. A
    # semi-infinite ocean never does this. The horizon is set by the time to
    # diffusively homogenize the column, z_max^2/kappa ~ 2300 yr (NOT the
    # column's radiative time rho_cp*z_max/lam ~ 350 yr), so 3000 yr is still
    # visibly stratified (~0.4 C top-to-bottom) and this needs its own run.
    T_EQUIL = 20000.0 * YEAR
    prof_end = solve_model(replace(p, t_final=T_EQUIL), n_save=50)["theta"][:, -1]
    spread = np.ptp(prof_end)
    check(f"{T_EQUIL / YEAR:.0f}-yr profile is uniform at dT_eq: spread = {spread:.2e} C, "
          f"mean = {prof_end.mean():.4f} C vs dT_eq = {p.dT_eq} C",
          spread < 1e-3 and abs(prof_end.mean() - p.dT_eq) < 1e-3,
          "closed column equilibrates through its full depth")

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
    # CHECK 9 + FIGURE: energy conservation for four diffusive timescales,
    # plotted as the surface step response and depth-time evolution
    # (see energy_budget_figure).
    # -----------------------------------------------------------------
    energy_budget_figure(p)

    # Physical probe: where did the heat go? The fraction stored below the
    # mixed layer is exactly why the dotted Fig. 16 curves are slower.
    for name in ["110 m + k = 1 cm$^2$/s", "110 m + k = 2 cm$^2$/s"]:
        rr = runs[name]
        ppp, ddz, tth = rr["p"], rr["dz"], rr["theta"][:, -1]
        H_slab = ppp.c_ml * tth[0]
        H_therm = ppp.rho_cp * (ddz * tth[1:-1].sum() + (ddz / 2) * tth[-1])
        print(f"  heat stored at 35 yr [{name}]: "
              f"{100 * H_therm / (H_slab + H_therm):.0f}% is below the mixed layer")

    # -----------------------------------------------------------------
    # CHECK 10: bottom boundary condition. The floor at z_max is a CLOSED,
    # zero-flux (Neumann) edge, so the one-sided gradient there must be ~0
    # on the most demanding run (150 yr). That is the only requirement --
    # the bottom anomaly itself is NOT required to vanish, since the domain
    # is a finite reservoir rather than an approximation to a semi-infinite
    # ocean; how far the front has penetrated by 150 yr is reported below as
    # a diagnostic only.
    # -----------------------------------------------------------------
    section("CHECK 10: bottom BC (closed, zero-flux floor at z_max)")
    uf = long_run["theta"][:, -1]
    dzl = long_run["dz"]
    grad_bot = (3 * uf[-1] - 4 * uf[-2] + uf[-3]) / (2 * dzl)   # 2nd-order one-sided
    grad_scale = np.ptp(uf) / p.z_max                            # typical gradient
    check(f"d(theta)/dz at z_max = {grad_bot:.2e} K/m", abs(grad_bot) < 1e-3 * grad_scale,
          "relative to column-mean gradient")
    # Diagnostic (not a pass/fail): how deep has the front reached at 150 yr?
    # Free diffusion into a semi-infinite half-space would leave
    # erfc(z / (2*sqrt(kappa*t))) of the surface anomaly at depth z, which is
    # what the closed column still tracks closely while the floor is far from
    # being felt. Once the two part company, the finite depth is doing real work.
    from scipy.special import erfc
    frac = uf[-1] / uf[0]
    theory = erfc(p.z_max / (2.0 * np.sqrt(p.kappa * 150 * YEAR)))
    print(f"  bottom anomaly after 150 yr = {uf[-1]:.2e} C = {100 * frac:.2f}% of "
          f"surface (free-diffusion erfc reference: {100 * theory:.2f}%)")

    # The floor is physical, so halving the ocean depth is a DIFFERENT model,
    # not a convergence test -- a 1350 m ocean holds half the heat and must
    # warm the surface at least as fast. Early on (35 yr) neither floor has
    # been felt, so the two still agree; the point of the check is the
    # inequality, which holds at all times.
    d35_full = runs["110 m + k = 1 cm$^2$/s"]["dT"][-1]
    r_half = solve_model(replace(p, z_max=1350.0, Nz=121))
    d35_half = r_half["dT"][-1]
    check(f"dT(35 yr): z_max 2500 m -> {d35_full:.4f}, 1350 m -> {d35_half:.4f}",
          d35_half >= d35_full * (1 - 1e-9),
          "shallower closed ocean warms at least as fast (equal before the floor is felt)")

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

    # -----------------------------------------------------------------
    # CHECK 13 + FIGURE: depth profile vs Held's exact h(zeta, tau) for the
    # purely diffusive ocean. The analytic case has no slab, so the slab is
    # shrunk to h_ml = dz/2 -- exactly the half-cell the surface node owns in
    # the finite-volume reading of the stencil -- and the domain is made
    # semi-infinite so the floor is never felt.
    # -----------------------------------------------------------------
    section("CHECK 13: depth profile vs analytic h(zeta, tau) (no slab, semi-infinite)")
    pd = replace(p, h_ml=p.z_max / (p.Nz - 1) / 2, semi_infinite=True)
    T_diff = (pd.D / pd.lam)**2 / pd.kappa                 # diffusive timescale [s]
    taus = [0.25, 1.0, 4.0]
    rd = solve_model(replace(pd, t_final=taus[-1] * T_diff), t_eval=[tau * T_diff for tau in taus])
    ana = analytic_diffusive_profile(rd["z"][:, None], rd["t"][None, :], rd["p"])
    err = np.max(np.abs(rd["theta"] - ana))
    check(f"max |numeric - analytic| over z and tau = {taus} is {err:.2e} C "
          f"(T = {T_diff / YEAR:.0f} yr)", err < 1e-2 * p.dT_eq, "tol 1% of dT_eq")

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (tau, c) in enumerate(zip(taus, [C_BLUE, C_VERM, C_GREEN])):
        ax.plot(rd["theta"][:, i], rd["z"], color=c, lw=2, label=rf"numeric, $\tau$ = {tau:g}")
        ax.plot(ana[:, i], rd["z"], color="k", lw=1.2, ls="--")
    # Held's length scale H = D/V_beta with V_beta = beta/(rho c); in this
    # module's units (D = rho_cp*kappa) that is D/lam. kappa and lam are the
    # same for all three taus, so it is one line.
    H_held = rd["p"].D / rd["p"].lam
    ax.axhline(H_held, color="k", lw=1.2, ls=":",
               label=rf"$\mathcal{{H}} = \mathcal{{D}}/\mathcal{{V}}_\beta$ = {H_held:.0f} m")
    ax.set_ylim(p.z_max, 0)
    ax.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel="Depth z (m)",
           title=r"Depth profile vs analytic $h(\zeta, \tau)$ (dashed)" + "\n"
                 r"purely diffusive, k = 1 cm$^2$/s")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, "depth_profile_vs_analytic.png")

    # -----------------------------------------------------------------
    # CHECK 14 + FIGURE: at tau = 1, how far does each model mode sit from
    # the no-slab analytic profile? Only the no-slab semi-infinite run should
    # match; the 110 m runs are diagnostics of what the slab and floor change.
    # -----------------------------------------------------------------
    section("CHECK 14: tau = 1 profile, analytic vs slab/floor modes")
    t1 = T_diff                                            # tau = 1
    modes = {
        "no slab, semi-infinite":    (pd, C_BLUE),
        "110 m slab, semi-infinite": (replace(p, semi_infinite=True), C_VERM),
        "110 m slab, 2500 m floor":  (p, C_GREEN),
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (pm, c) in modes.items():
        rm = solve_model(replace(pm, t_final=t1), t_eval=[t1])
        ana1 = analytic_diffusive_profile(rm["z"], t1, rm["p"])
        print(f"  {name}: max |numeric - analytic| = {np.max(np.abs(rm['theta'][:, 0] - ana1)):.2e} C, "
              f"surface {rm['theta'][0, 0]:.3f} vs {ana1[0]:.3f} C")
        ax.plot(rm["theta"][:, 0], rm["z"], color=c, lw=2, label=name)
    ax.plot(ana1, rm["z"], color="k", lw=1.2, ls="--", label=r"analytic $h(\zeta, 1)$")
    ax.set_ylim(p.z_max, 0)
    ax.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel="Depth z (m)",
           title=rf"Depth profile at $\tau$ = 1 ({t1 / YEAR:.0f} yr), k = 1 cm$^2$/s")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, "depth_profile_tau1_modes.png")

    beck_slab_figure(p)
    held_wmghg_figure(p)
    held_linear_figure(p)

    section(f"DONE — figures saved in {OUTDIR}")


BECK_A = 6                       # beck_slab_figure: windows/terms for 10^-A accuracy
BECK_BIS = [0.1, 1.0, 10.0]      # Biot numbers tabulated in the X32B10T0 report
# Dash for the semi-infinite numeric band: in units of its 4-pt line width, so
# short enough that the gaps also show in the legend sample.
BECK_SEMI_DASH = (0, (2, 1.5))
BECK_DEPTHS = [(0.0, C_BLUE), (0.5, C_GREEN), (1.0, C_PINK)]   # subset of the table's x = 0, .25, ..., 1
BECK_TABLE_X = [0.0, 0.25, 0.5, 0.75, 1.0]                       # data/X32B10T0_tables.txt columns 2..6


def beck_slab_figure(p):
    """Closed column vs the exact X32B10T0 slab solution (Beck & Woodbury 2013,
    Exact Analytical Conduction Toolbox): a slab 0 < x < L heated by convection
    from a fluid at T_f through x = 0 and insulated at x = L. With the slab
    shrunk to h_ml = dz/2 that is exactly this model -- k = D, h_1 = lam,
    T_f = F0/lam = dT_eq, L = z_max -- so Bi = lam*z_max/D = z_max/H,
    x = z/z_max, t = kappa*t/z_max^2 and T = theta/dT_eq. Bi is set by kappa at
    fixed lam and z_max, so all three runs share the grid, and x = 0, .25, ...
    fall on nodes.

    Each of the report's short-, intermediate- and long-time forms is drawn
    over the whole t range, with ticks where its 10^-A window ends; the
    report's tabulated values (data/X32B10T0_tables.txt) are overlaid as a
    check on the formulas.

    The same runs with the floor removed (semi_infinite=True, so solve_model
    deepens the column until it is never felt) are drawn dashed. They must
    follow the closed column until the wall is felt, then fall below it, and
    track the short-time form -- the exact semi-infinite solution -- at every t.
    Dimensionless x and t keep L = z_max = 2500 m for both.
    """
    section("FIGURE: beck_X32B10T0_comparison.png (closed column vs exact slab solution)")
    table = np.loadtxt("data/X32B10T0_tables.txt", skiprows=1)
    dz = p.z_max / (p.Nz - 1)
    t = np.linspace(0.0, 1.0, 401)[1:]                     # dimensionless; forms are singular at 0
    fig, axes = plt.subplots(1, len(BECK_BIS), figsize=(15, 5.5))
    for ax, Bi in zip(axes, BECK_BIS):
        kappa = p.lam * p.z_max / (p.rho_cp * Bi)
        T_slab = p.z_max**2 / kappa                        # t = 1 [s]
        rb = solve_model(replace(p, kappa=kappa, h_ml=dz / 2, t_final=T_slab), t_eval=t * T_slab)
        rs = solve_model(replace(p, kappa=kappa, h_ml=dz / 2, t_final=T_slab, semi_infinite=True),
                         t_eval=t * T_slab)
        tab = table[np.isclose(table[:, 0], Bi)]
        print(f"  Bi = {Bi:g}: kappa = {kappa / CM2S_TO_M2S:.3g} cm^2/s, t = 1 is {T_slab / YEAR:.0f} yr; "
              f"semi-infinite column z_max = {rs['p'].z_max:.0f} m")
        err_tab, err_num, err_semi = {}, {}, 0.0
        for x, c in BECK_DEPTHS:
            j = BECK_TABLE_X.index(x)
            num = rb["theta"][round(x * (p.Nz - 1))] / p.dT_eq
            t_short, t_int = beck_windows(x, BECK_A)
            m_max = beck_m_max(t_int, BECK_A)
            forms = {
                "short":        (lambda tt: beck_short(x, tt, Bi), "--", (0, t_short)),
                "intermediate": (lambda tt: beck_intermediate(x, tt, Bi), ":", (t_short, t_int)),
                "long":         (lambda tt: beck_long(x, tt, Bi, m_max), "-.", (t_int, np.inf)),
            }
            # The deepened grid keeps dz, but index by depth rather than assume it.
            num_semi = rs["theta"][np.argmin(np.abs(rs["z"] - x * p.z_max))] / p.dT_eq
            err_semi = max(err_semi, np.max(np.abs(num_semi - beck_short(x, t, Bi))))
            ax.plot(t, num, color=c, lw=4, alpha=0.35)     # band, so the forms on top stay visible
            ax.plot(t, num_semi, color=c, lw=4, alpha=0.35, ls=BECK_SEMI_DASH)
            for name, (form, ls, (lo, hi)) in forms.items():
                ax.plot(t, form(t), color=c, lw=1, ls=ls)
                # Diagnostics inside the window only (empty at x = 0 for "intermediate")
                inn, inn_tab = (t >= lo) & (t < hi), (tab[:, 1] >= lo) & (tab[:, 1] < hi)
                if inn.any():
                    err_num[name] = max(err_num.get(name, 0), np.max(np.abs(num - form(t))[inn]))
                if inn_tab.any():
                    err_tab[name] = max(err_tab.get(name, 0), np.max(np.abs(
                        form(tab[inn_tab, 1]) - tab[inn_tab, 2 + j])))
            ends = [te for te in (t_short, t_int) if te <= 1]
            ax.plot(ends, np.interp(ends, t, num), ls="none", marker="|", ms=12, mew=2, color=c)
            ax.plot(tab[:, 1], tab[:, 2 + j], ls="none", marker="o", ms=4, mfc="none", mec=c)
        for name in forms:
            print(f"    {name:<13} in window: max|analytic - table| = {err_tab.get(name, np.nan):.1e}, "
                  f"max|numeric - analytic| = {err_num[name]:.1e}")
        print(f"    semi-infinite numeric vs short-time form, all t: max diff = {err_semi:.1e}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1 * rb["theta"].max() / p.dT_eq)
        ax.set(xlabel=r"$\tilde t = \kappa t / z_{max}^2$",
               title=rf"Bi = {Bi:g}  ($\kappa$ = {kappa / CM2S_TO_M2S:.3g} cm$^2$/s, "
                     rf"$\tilde t$ = 1 is {T_slab / YEAR:.0f} yr)")
        style_axes(ax)
    axes[0].set_ylabel(r"$\tilde T = \theta / \Delta T_{eq}$")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, label=rf"$\tilde x$ = {x:g}") for x, c in BECK_DEPTHS]
    handles += [
        Line2D([], [], color="k", lw=4, alpha=0.35, label=f"numeric, closed {p.z_max:.0f} m floor"),
        Line2D([], [], color="k", lw=4, alpha=0.35, ls=BECK_SEMI_DASH, label="numeric, semi-infinite"),
        Line2D([], [], color="k", lw=1, ls="--", label="short-time (Eq. 7a)"),
        Line2D([], [], color="k", lw=1, ls=":", label="intermediate (Eq. 9)"),
        Line2D([], [], color="k", lw=1, ls="-.", label="long-time series (Eq. 12a)"),
        Line2D([], [], color="k", ls="none", marker="|", ms=12, mew=2,
               label=rf"window ends ($10^{{-{BECK_A}}}$)"),
        Line2D([], [], color="k", ls="none", marker="o", ms=4, mfc="none", label="report table"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Numerical vs exact slab solution (Beck & Woodbury 2013, X32B10T0)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    savefig(fig, "beck_X32B10T0_comparison.png")


HELD_TAU_MAX = 100.0   # held_wmghg_figure: last fixed-tau profile slice
HELD_SCALES = [(5, C_VERM), (80, C_GREEN), (200, C_BLUE), (1200, C_PINK), (10000, C_ORANGE)]
LINEAR_PROFILE_TAUS = [5.0, 10.0, 20.0]       # held_linear_figure (c); within the T = 5 yr run's range


def held_params(p, scale, t_final, z_floor=None):
    """Purely diffusive ocean with diffusive timescale T = scale [yr].

    T = rho_cp^2 kappa / lam^2 is set by kappa at fixed lam, so H = D/lam and
    the penetration depth sqrt(kappa t) both grow with T. The slab is shrunk
    to h_ml = dz/2 (the surface node's half-cell), and dz resolves the smaller
    of H and sqrt(kappa t_final). z_floor=None places the insulated bottom deep
    enough to be semi-infinite; otherwise it is a closed floor at z_floor [m].
    """
    kappa = scale * YEAR * p.lam**2 / p.rho_cp**2
    L = np.sqrt(kappa * t_final)                         # diffusion length at t_final
    z_max = SEMI_INF_PAD * 2 * L if z_floor is None else z_floor
    Nz = int(np.ceil(z_max / (min(p.rho_cp * kappa / p.lam, L, z_max) / 40))) + 1
    return replace(p, kappa=kappa, h_ml=z_max / (Nz - 1) / 2, z_max=z_max, Nz=Nz,
                   t_final=t_final)


def energy_budget_figure(p):
    """CHECK 9 + FIGURE relaxation_time_check.png: energy conservation, the
    surface step response, and the depth-time evolution of theta.

    Purely diffusive ocean (slab h_ml = dz/2, deep enough to be semi-infinite)
    under a step forcing F = F2X switched on at t = 0, for four diffusive
    timescales T = rho_cp^2 kappa / lam^2: kappa varies, while lam = F0/dT_eq
    and the step are the same for every run. Each run gets its own grid, sized as in
    held_wmghg_figure.

    The bottom is insulated, so the change in column heat content must equal
    the time-integrated surface flux:  rho_cp*Int(theta dz) == Int(F - lam*T0)dt.
    The heat functional is the discretely-conserved one (half-width surface
    and bottom cells, dz-weighted interior), so the check isolates
    TIME-integration error rather than O(dz) bookkeeping noise.

    The checks also grade depth profiles at fixed tau = t/T (0.25, 1, 4),
    from separate runs long enough to reach tau for each T: with no slab and
    a semi-infinite ocean theta depends only on (z/H, t/T), so they must
    match Held's h(zeta, tau).

    Figure: (a) surface theta vs tau on a linear axis, a single curve with
    each run covering its own tau range, against the analytic h(0, tau);
    (b) one panel per run of every profile theta(zeta = z/H), coloured by tau,
    which must collapse across runs; (c) isotherms: the depth zeta reached by
    fixed theta levels vs tau, numeric (from (b)'s run) against Held's
    h(zeta, tau) (dashed).
    """
    section("CHECK 9: energy conservation vs depth (no slab, semi-infinite, step forcing F2X, 4 timescales)")
    forcing = lambda t: np.full_like(t, F2X, dtype=float)
    t_final = 500.0 * YEAR
    taus = [0.25, 1.0, 4.0]
    # Log spacing resolves the sqrt(t) surface jump at early times (it sets
    # the trapezoid error in Int(F dt)); linear spacing covers late times.
    t_eval = np.unique(np.concatenate([
        [0.0], np.geomspace(1e-3, t_final / YEAR, 300), np.linspace(0.0, t_final / YEAR, 300)])) * YEAR

    runs = []
    for scale, c in HELD_SCALES[:4]:
        r = solve_model(replace(held_params(p, scale, t_final), forcing=forcing), t_eval=t_eval)
        th, dz, kappa, Nz = r["theta"], r["dz"], r["p"].kappa, r["p"].Nz
        H = p.rho_cp * dz * (th[0] / 2 + th[1:-1].sum(axis=0) + th[-1] / 2)
        F_int = np.concatenate([[0.0], cumulative_trapezoid(forcing(r["t"]) - p.lam * th[0], r["t"])])
        rel_err = np.abs(H - F_int).max() / F_int[-1]
        check(f"T = {scale:4d} yr (kappa = {kappa / CM2S_TO_M2S:6.2f} cm^2/s, Nz = {Nz}): "
              f"max |dH - Int(F dt)| / Int(F dt) = {rel_err:.2e}", rel_err < 5e-3,
              "tol 0.5%; violation would mean flux/capacity bookkeeping is inconsistent")
        # Profiles at fixed tau: t = tau*T runs past t_final for the slow runs.
        T_s = scale * YEAR
        rs = solve_model(replace(held_params(p, scale, taus[-1] * T_s), forcing=forcing),
                         t_eval=[tau * T_s for tau in taus])
        ana = F2X / p.F0 * analytic_diffusive_profile(rs["z"][:, None], rs["t"][None, :], rs["p"])
        err = np.max(np.abs(rs["theta"] - ana))
        check(f"  profiles at tau = {taus}: max |numeric - analytic| = {err:.2e} C",
              err < 1e-2 * F2X / p.lam, "tol 1% of F2X/lam")
        runs.append((r, H, F_int, kappa, c, scale, rs))

    # (a) surface theta vs tau on a linear axis, every run over its own tau
    # range up to TAU_MAX. The runs overlap in tau, so earlier ones are drawn
    # wider to stay visible. Analytic: the surface (zeta = 0) step response,
    # (F/lam) h(0, tau) = (F/lam)(1 - e^tau erfc(sqrt(tau))).
    # (b) every profile theta(zeta) up to TAU_MAX, coloured by tau. With no
    # slab and a semi-infinite ocean theta depends only on (zeta, tau), so one
    # run stands for all of them: the fastest, rerun to exactly TAU_MAX with
    # output at each integer tau. Integer tau is marked in red in both panels
    # and on the colorbar.
    TAU_MAX = 10.0
    tau_int = np.arange(1.0, TAU_MAX + 1)
    th_eq = F2X / p.lam
    pr = runs[0][-1]["p"]
    H_ref = pr.D / pr.lam
    tau_line = np.linspace(1e-6, TAU_MAX, 400)          # tau = 0 is 0/0 at zeta = 0
    analytic = lambda tau: F2X / p.F0 * analytic_diffusive_profile(0.0, tau * H_ref**2 / pr.kappa, pr)

    scale_b = runs[0][5]
    T_b = scale_b * YEAR
    tau_b = np.unique(np.concatenate([np.linspace(0.0, TAU_MAX, 401), tau_int]))
    rb = solve_model(replace(held_params(p, scale_b, TAU_MAX * T_b), forcing=forcing), t_eval=tau_b * T_b)
    zeta_b = rb["z"] / (rb["p"].D / rb["p"].lam)

    fig = plt.figure(figsize=(12, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.0], width_ratios=[1.0, 0.025])
    ax = fig.add_subplot(gs[0, 0])
    for (r, H, F_int, kappa, c, scale, rs), lw in zip(runs, [7, 5, 3.5, 2]):
        zH = r["p"].D / r["p"].lam
        tau_r = r["t"] / (scale * YEAR)
        keep = tau_r <= TAU_MAX
        ax.plot(tau_r[keep], r["dT"][keep], color=c, lw=lw, solid_capstyle="butt",
                label=rf"$\mathcal{{T}}$ = {scale} yr, $\mathcal{{H}}$ = {zH:.0f} m")
    ax.plot(tau_line, analytic(tau_line), color="k", lw=1, ls="--", label=r"analytic, $h(0, \tau)$")
    for tau in tau_int:
        ax.axvline(tau, color="r", lw=0.8, alpha=0.7, zorder=0)
    ax.plot([], [], color="r", lw=0.8, label=r"integer $\tau$")
    ax.axhline(th_eq, color="0.6", lw=0.8, ls=":")
    ax.set(xlim=(0, TAU_MAX), ylim=(0, 1.05 * th_eq), xlabel=r"$\tau = t/\mathcal{T}$",
           ylabel=r"surface $\theta$ anomaly ($\degree$C)",
           title=rf"(a) surface step response, F = F$_{{2x}}$, "
                 rf"$F/\lambda$ = {th_eq:.2f} $\degree$C (no slab, semi-infinite)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(ax)

    ax_p = fig.add_subplot(gs[1, 0])
    segments = [np.column_stack([rb["theta"][:, j], zeta_b]) for j in range(tau_b.size)]
    lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0.0, TAU_MAX),
                        array=tau_b, linewidths=1.2)
    ax_p.add_collection(lc)
    j_int = np.searchsorted(tau_b, tau_int)
    for j in j_int:
        ax_p.plot(rb["theta"][:, j], zeta_b, color="r", lw=1.2)
    ax_p.set_xlim(-0.02 * th_eq, 1.05 * th_eq)
    ax_p.set_ylim(4 * np.sqrt(TAU_MAX), 0)              # depth increases downward
    ax_p.set(xlabel=r"$\theta$ anomaly ($\degree$C)", ylabel=r"$\zeta = z/\mathcal{H}$")
    ax_p.set_title(rf"(b) depth-time evolution, the same for every $\mathcal{{T}}$ "
                   rf"(computed with $\mathcal{{T}}$ = {scale_b} yr); red: integer $\tau$")
    style_axes(ax_p)
    cbar = fig.colorbar(lc, cax=fig.add_subplot(gs[1, 1]))
    cbar.ax.hlines(tau_int, 0, 1, color="r", lw=1.2)
    cbar.set_label(r"$\tau = t/\mathcal{T}$")

    # (c) isotherms zeta(tau) at fixed theta: theta decreases monotonically with
    # depth, so each level is a single curve that deepens with tau.
    ax_c = fig.add_subplot(gs[2, 0])
    th_levels = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    th_colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(th_levels)))
    zeta_max = 4 * np.sqrt(TAU_MAX)
    ax_c.contour(tau_b, zeta_b, rb["theta"], levels=th_levels, colors=th_colors, linewidths=2.5)
    zz = np.linspace(0.0, zeta_max, 300)
    tt = np.linspace(1e-6, TAU_MAX, 401)
    th_ana = F2X / p.F0 * analytic_diffusive_profile(zz[:, None] * H_ref, tt[None, :] * H_ref**2 / pr.kappa, pr)
    ax_c.contour(tt, zz, th_ana, levels=th_levels, colors="k", linewidths=1, linestyles="--")
    for th, c in zip(th_levels, th_colors):
        ax_c.plot([], [], color=c, lw=2.5, label=rf"$\theta$ = {th:g} $\degree$C")
    ax_c.plot([], [], color="k", lw=1, ls="--", label=r"analytic, $h(\zeta, \tau)$")
    # z = V_beta t with V_beta = lam/rho_cp: H = kappa/V_beta and T = kappa/V_beta^2,
    # so in (zeta, tau) it is zeta = tau.
    ax_c.plot([0, TAU_MAX], [0, TAU_MAX], color="0.45", lw=1.5, ls=":",
              label=r"$z = \mathcal{V}_\beta t$, i.e. $\zeta = \tau$")
    ax_c.set(xlim=(0, TAU_MAX), ylim=(zeta_max, 0), xlabel=r"$\tau = t/\mathcal{T}$",
             ylabel=r"$\zeta = z/\mathcal{H}$",
             title=rf"(c) isotherms: depth reached by fixed $\theta$ (computed with $\mathcal{{T}}$ = {scale_b} yr)")
    ax_c.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    style_axes(ax_c)
    fig.tight_layout(h_pad=3)
    savefig(fig, "relaxation_time_check.png")


def held_wmghg_figure(p):
    """CHECK 15 + FIGURE: numerical recreation of figures/held_wmghg_response.png.

    Purely diffusive ocean (slab h_ml = dz/2, deep enough to be semi-infinite)
    driven by the WMGHG forcing of data/forcing_1850_2100.txt from 1850.
    The timescale T = rho_cp^2 kappa / lam^2 is set by kappa at fixed lam, so
    H = D/lam and the penetration depth both grow with T; each run gets its
    own grid (dz resolves the smaller of H and sqrt(kappa t)). The check is
    graded in units of the TCR, the response at yr 70 of a linear ramp
    reaching F2X there, as in held_exact_vs_approx.py; the panels plot
    temperature divided by T.

    held_wmghg_response_numeric_tcr.png repeats the semi-infinite runs the
    original way: temperature/TCR on linear axes, and (c) profiles of theta/TCR at fixed
    calendar years.

    The depth profiles (c) are at fixed tau = t/T up to HELD_TAU_MAX, each
    from a separate run to tau_max*T on a grid refined for the smallest tau.
    Those runs outlast the data, so their forcing is linearly extrapolated
    past 2100 (least-squares slope of the last 20 yr, continued from the 2100
    value). (a) and (b) show only 1850-2100, with no extrapolation, and dot
    only the slices that fall in it.
    """
    section("CHECK 15: WMGHG historical response, numeric vs analytic (no slab, semi-infinite)")
    yr, F_ghg = np.loadtxt("data/forcing_1850_2100.txt", skiprows=1, usecols=(0, 3), unpack=True)
    t_knots = (yr - yr[0]) * YEAR
    ramp_knots = np.array([0.0, T_2X]), np.array([0.0, F2X])
    plot_yrs = yr
    t_final = (plot_yrs[-1] - yr[0]) * YEAR

    # Profile runs only: np.interp holds the 2100 value, so adding the slope
    # past t_knots[-1] continues the forcing linearly from there.
    slope = np.polyfit(yr[-20:], F_ghg[-20:], 1)[0] / YEAR
    forcing_ext = lambda t: np.interp(t, t_knots, F_ghg) + slope * max(t - t_knots[-1], 0.0)

    scales = [(20, C_VERM), (80, C_GREEN), (200, C_BLUE)]
    profile_taus = np.geomspace(0.05, HELD_TAU_MAX, 5)
    print(f"  profile runs: WMGHG forcing extrapolated past {yr[-1]:.0f} at {slope * YEAR:.4f} W/m^2/yr, "
          f"to {yr[0] + HELD_TAU_MAX * scales[-1][0]:.0f} "
          f"(F = {forcing_ext(HELD_TAU_MAX * scales[-1][0] * YEAR):.1f} W/m^2)")
    # _tcr figure: more timescales, so 20 yr takes the orange freed by 10000 yr.
    tcr_scales = [(5, C_VERM), (20, C_ORANGE), (80, C_GREEN), (200, C_BLUE), (1200, C_PINK)]
    profile_yrs = [1900, 1950, 2000, 2050, 2100]

    def solve_runs(run_scales, z_floor=None, profiles=True):
        """One run per timescale. z_floor=None: deep enough to be semi-infinite,
        graded against the analytic solution. Otherwise a closed, insulated
        floor at z_floor [m]; the semi-infinite analytic curve is then only a
        reference, so the difference is printed, not graded. profiles=False
        skips the fixed-tau profile runs (rs = None)."""
        runs = []
        for scale, c in run_scales:
            ph = replace(held_params(p, scale, t_final, z_floor),
                         forcing=lambda t: np.interp(t, t_knots, F_ghg))
            kappa, Nz = ph.kappa, ph.Nz

            r = solve_model(ph, t_eval=(plot_yrs - yr[0]) * YEAR)
            tcr = solve_model(replace(ph, forcing=lambda t: np.interp(t, *ramp_knots),
                                      t_final=70.0 * YEAR), t_eval=[70.0 * YEAR])["dT"][0]
            ana = analytic_diffusive_surface(r["t"], t_knots, F_ghg, ph)
            tcr_ana = analytic_diffusive_surface([70.0 * YEAR], *ramp_knots, ph)[0]
            err = np.max(np.abs(r["dT"] / tcr - ana / tcr_ana))
            label = (f"T = {scale:5d} yr (kappa = {kappa:.2e} m^2/s, Nz = {Nz}): "
                     f"max |numeric - analytic| = {err:.2e} TCR, T/TCR in {plot_yrs[-1]:.0f} = {r['dT'][-1] / tcr:.3f}")
            if z_floor is None:
                check(label, err < 1e-2, "tol 0.01 TCR")
            else:
                print(f"  {label}  (vs semi-infinite analytic)")

            if not profiles:
                runs.append((r, ana, tcr, tcr_ana, c, scale, None))
                continue

            # Fixed-tau profiles. Refine dz so the earliest slice's penetration
            # depth sqrt(tau_min) H is resolved by ~40 cells. A floor deeper
            # than the semi-infinite padding is never felt by tau_max, so the
            # shallower padded domain stands in for it.
            T_s = scale * YEAR
            t_prof = profile_taus[-1] * T_s
            felt = z_floor is not None and z_floor < SEMI_INF_PAD * 2 * np.sqrt(kappa * t_prof)
            ps = held_params(p, scale, t_prof, z_floor if felt else None)
            Nz_s = max(ps.Nz, int(np.ceil(ps.z_max / (np.sqrt(profile_taus[0] * kappa * T_s) / 40))) + 1)
            ps = replace(ps, Nz=Nz_s, h_ml=ps.z_max / (Nz_s - 1) / 2, forcing=forcing_ext)
            rs = solve_model(ps, t_eval=[tau * T_s for tau in profile_taus])

            runs.append((r, ana, tcr, tcr_ana, c, scale, rs))
        return runs

    semi_inf = solve_runs(scales)
    extra = solve_runs([sc for sc in tcr_scales if sc[0] not in dict(scales)], profiles=False)
    print("  --- 2500 m insulated floor (diagnostic) ---")
    floored = solve_runs(scales, z_floor=2500.0)
    by_scale = {run[5]: run for run in semi_inf + extra}
    tcr_runs = [by_scale[scale][:4] + (c, scale, None) for scale, c in tcr_scales]

    # Panel (c) depth axes, one range per tau panel:
    #   zeta = z/sqrt(kappa T) = z/H (Held's nondimensional depth). Every run
    #     penetrates to zeta ~ sqrt(tau), so the range is the same for all.
    #   z [m]. Penetration is ~sqrt(kappa tau T), so the slowest run sets the
    #     range (or the floor, when there is one).
    zeta = lambda r, scale: r["z"] / np.sqrt(r["p"].kappa * scale * YEAR)
    z = lambda r, scale: r["z"]
    z_pen = lambda tau: 4 * np.sqrt(semi_inf[-1][0]["p"].kappa * tau * scales[-1][0] * YEAR)
    figures = [
        (semi_inf, "semi-infinite", zeta, lambda tau: 4 * np.sqrt(tau),
         r"$\zeta = z/\sqrt{\kappa \mathcal{T}}$", "held_wmghg_response_numeric.png"),
        (semi_inf, "semi-infinite", z, z_pen,
         "Depth z (m)", "held_wmghg_response_numeric_z.png"),
        (floored, "2500 m floor", z, lambda tau: min(z_pen(tau), 2500.0),
         "Depth z (m)", "held_wmghg_response_numeric_2500m.png"),
    ]
    # (a), (b): temperature divided by the run's diffusive timescale T [yr].
    # T spans 20-200 yr, so the values span ~10x; symlog keeps the negative
    # early response (WMGHG (Sm) forcing < 0 before ~1860) and the (c) slices,
    # dotted at t = tau T, most of which fall before 1880.
    # (c): theta/T [C/yr], linear: the early WMGHG (Sm) forcing is negative.
    for runs, domain, depth, depth_max, ylabel, name in figures:
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, len(profile_taus))
        ax_a = fig.add_subplot(gs[0, :2])
        ax_b = fig.add_subplot(gs[0, 2:])
        for r, ana, _, _, c, scale, rs in runs:
            ax_a.plot(plot_yrs, r["dT"] / scale, color=c, lw=2, label=f"{scale} yrs")
            ax_b.plot(plot_yrs, r["dT"] / scale, color=c, lw=2, label=f"numeric, {scale} yrs")
            ax_b.plot(plot_yrs, ana / scale, color="k", lw=1, ls="--")
            t_dot = yr[0] + profile_taus * scale
            shown = t_dot <= yr[-1]
            for ax in (ax_a, ax_b):
                ax.plot(t_dot[shown], rs["dT"][shown] / scale, "o",
                        color=c, mec="k", mew=0.6, ms=6, zorder=3)
        for ax in (ax_a, ax_b):
            ax.plot([], [], "o", color="0.7", mec="k", mew=0.6, ms=6, label=r"(c) $\tau$ slices")

        for i, tau in enumerate(profile_taus):
            ax = fig.add_subplot(gs[1, i])
            for *_, c, scale, rs in runs:
                ax.plot(rs["theta"][:, i] / scale, depth(rs, scale), color=c, lw=1.5,
                        label=f"{scale} yrs (t = {tau * scale:.3g} yr)")
            ax.set_ylim(depth_max(tau), 0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            ax.set(xlabel=r"$\theta / \mathcal{T}$ ($\degree$C yr$^{-1}$)",
                   title=("(c) depth profiles, " if i == 0 else "") + rf"$\tau$ = {tau:.3g}")
            if i == 0:
                ax.set_ylabel(ylabel)
                ax.legend(frameon=False, fontsize=7, loc="lower right")
            style_axes(ax)

        ab_ylabel = r"temperature/$\mathcal{T}$ ($\degree$C yr$^{-1}$)"
        ax_a.set(xlabel="year", ylabel=ab_ylabel, xlim=(1845, yr[-1] + 5),
                 title=f"(a) WMGHG response, numerical (no slab, {domain})")
        ax_a.set_yscale("symlog", linthresh=1e-5)
        ax_a.legend(frameon=False, fontsize=9, loc="center right")
        ax_b.set(xlabel="year", ylabel=ab_ylabel, xlim=(1845, yr[-1] + 5),
                 title="(b) numeric (solid) vs semi-infinite analytic (black dashed)")
        ax_b.set_yscale("symlog", linthresh=1e-5)
        ax_b.legend(frameon=False, fontsize=9, loc="center right")
        for ax in (ax_a, ax_b):
            style_axes(ax)
        fig.tight_layout(h_pad=3)
        savefig(fig, name)

    # Semi-infinite runs for tcr_scales, normalized by the TCR on linear axes, with
    # profiles at fixed calendar years. A profile penetrates to zeta ~
    # sqrt(t/T), so the fastest run sets each panel's depth range.
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, len(profile_yrs))
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    prof_idx = np.searchsorted(plot_yrs, profile_yrs)
    for r, ana, tcr, tcr_ana, c, scale, _ in tcr_runs:
        ax_a.plot(plot_yrs, r["dT"] / tcr, color=c, lw=2, label=f"{scale} yrs")
        ax_b.plot(plot_yrs, r["dT"] / tcr, color=c, lw=2, label=f"numeric, {scale} yrs")
        ax_b.plot(plot_yrs, ana / tcr_ana, color="k", lw=1, ls="--")
        for ax in (ax_a, ax_b):
            ax.plot(plot_yrs[prof_idx], r["dT"][prof_idx] / tcr, "o",
                    color=c, mec="k", mew=0.6, ms=6, zorder=3)
    ax_a.plot(plot_yrs, np.interp(plot_yrs, yr, F_ghg) / F2X, "k", lw=1.5, label="forcing/F2X")
    for ax in (ax_a, ax_b):
        ax.plot([], [], "o", color="0.7", mec="k", mew=0.6, ms=6, label="(c) time slices")

    theta_max = max(r["theta"][:, prof_idx].max() / tcr for r, _, tcr, *_ in tcr_runs)
    for i, (y, j) in enumerate(zip(profile_yrs, prof_idx)):
        ax = fig.add_subplot(gs[1, i])
        for r, _, tcr, _, c, scale, _ in tcr_runs:
            ax.plot(r["theta"][:, j] / tcr, zeta(r, scale), color=c, lw=1.5, label=f"{scale} yrs")
        ax.set_ylim(4 * np.sqrt((y - yr[0]) / tcr_scales[0][0]), 0)
        ax.set_xlim(-0.02 * theta_max, 1.05 * theta_max)
        ax.set(xlabel=r"$\theta$/TCR", title=("(c) depth profiles, " if i == 0 else "") + str(y))
        if i == 0:
            ax.set_ylabel(r"$\zeta = z/\sqrt{\kappa \mathcal{T}}$")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        style_axes(ax)

    ax_a.set(xlabel="year", ylabel="temperature/TCR", xlim=(1845, yr[-1] + 5),
             title="(a) WMGHG response, numerical (no slab, semi-infinite)")
    ax_b.set(xlabel="year", ylabel="temperature/TCR", xlim=(1845, yr[-1] + 5),
             title="(b) numeric (solid) vs semi-infinite analytic (black dashed)")
    for ax in (ax_a, ax_b):
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        style_axes(ax)
    fig.tight_layout(h_pad=3)
    savefig(fig, "held_wmghg_response_numeric_tcr.png")



def held_linear_figure(p):
    """CHECK 16 + FIGURE: held_wmghg_response_numeric.png under a pure linear ramp.

    Same purely diffusive, semi-infinite ocean and five timescales as
    held_wmghg_figure, driven by F(t) = F2X * t/T_2X from t = 0: it reaches
    F2X at yr 70 (the ramp that defines the TCR) and keeps rising at that
    rate. Runs go to 100 yr.

    Temperatures are plotted as theta/T [C/yr]. Since theta scales with the
    ramp amplitude A_T = F2X*T/(lam*T_2X), theta/T = F2X/(lam*T_2X) * g(zeta,
    tau), with g(zeta, tau) = Int_0^tau h(zeta, u) du the time-integral of
    Held's step response h; every run must collapse onto it, each over its own
    range of tau = t/T. The checks are done on theta/A_T = g directly.

    (a) surface theta/T vs tau, numeric vs analytic;
    (b) theta/T as a heatmap over tau and zeta, each run filling its own tau
        range (later runs, i.e. smaller T and larger tau, drawn first), with
        the analytic g(zeta, tau) as dashed contours. The check compares each
        run's whole (zeta, tau) field with g.
    (c) depth profiles at LINEAR_PROFILE_TAUS (dotted lines on (b)) vs g.
    """
    section("CHECK 16: linear-ramp response, numeric vs analytic (no slab, semi-infinite)")
    t_final = 100.0 * YEAR
    plot_yrs = np.arange(0.0, t_final / YEAR + 1)
    ramp_knots = np.array([0.0, t_final]), np.array([0.0, F2X * t_final / T_2X])
    forcing = lambda t: F2X * t / T_2X
    j70 = np.searchsorted(plot_yrs, T_2X / YEAR)

    def ramp_field(zeta, taus, n=4000):
        """g(zeta, tau) = Int_0^tau h(zeta, u) du on the grid zeta x taus, by a
        cumulative trapezoid on one log-graded u grid (h ~ sqrt(u) near the
        surface) up to max(taus), interpolated to each tau. h is
        analytic_diffusive_profile with dT_eq divided out."""
        H, T = p.D / p.lam, (p.D / p.lam)**2 / p.kappa
        taus = np.atleast_1d(taus)
        u = np.concatenate([[0.0], np.geomspace(1e-8 * taus[taus > 0].min(), taus.max(), n)])
        h = analytic_diffusive_profile(np.atleast_1d(zeta)[:, None] * H,
                                       np.maximum(u, 1e-300)[None, :] * T, p) / p.dT_eq
        g = cumulative_trapezoid(h, u, axis=1, initial=0.0)
        return np.array([np.interp(taus, u, row) for row in g])

    rate = F2X / (p.lam * T_2X / YEAR)            # theta/T = rate * g  [C/yr]

    runs = []
    for scale, c in HELD_SCALES:
        T_s = scale * YEAR
        A_T = F2X * T_s / (p.lam * T_2X)
        ph = replace(held_params(p, scale, t_final), forcing=forcing)
        r = solve_model(ph, t_eval=plot_yrs * YEAR)
        ana = analytic_diffusive_surface(r["t"], *ramp_knots, ph)
        tcr = r["dT"][j70]
        err = np.max(np.abs(r["dT"] - ana)) / ana[j70]
        check(f"T = {scale:5d} yr (kappa = {ph.kappa:.2e} m^2/s, Nz = {ph.Nz}): "
              f"max |numeric - analytic| = {err:.2e} TCR, T/TCR at yr {t_final / YEAR:g} = {r['dT'][-1] / tcr:.3f}",
              err < 1e-2, "tol 0.01 TCR")

        # Whole (zeta, tau) field vs g, below the surface down to 6 sqrt(tau_max)
        # (g there is < 1e-8 of its surface value), in units of the yr-70 surface value.
        tau = plot_yrs / scale
        zeta = r["z"] / (ph.D / ph.lam)
        k = zeta <= 6 * np.sqrt(tau[-1])
        g_err = np.abs(r["theta"][k] / A_T - ramp_field(zeta[k], tau))
        check(f"  field over zeta <= {zeta[k][-1]:.2f}, tau <= {tau[-1]:g}: "
              f"max |theta/A_T - g| = {g_err.max() * A_T / tcr:.2e} TCR", g_err.max() * A_T / tcr < 1e-2,
              "tol 0.01 TCR")
        runs.append((r, ana, c, scale, zeta, tau))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, len(HELD_SCALES))
    ax_a = fig.add_subplot(gs[0, :])
    # The runs overlap in tau on one curve, so earlier ones are drawn wider.
    for (r, ana, c, scale, *_), lw in zip(runs, [8, 6, 4.5, 3, 1.5]):
        tau = plot_yrs / scale
        ax_a.plot(tau, r["dT"] / scale, color=c, lw=lw, solid_capstyle="butt", label=f"numeric, {scale} yrs")
        ax_a.plot(tau, ana / scale, color="k", lw=1, ls="--")
        ax_a.plot(T_2X / YEAR / scale, r["dT"][j70] / scale, "o", color=c, mec="k", mew=0.6, ms=6, zorder=3)
    ax_a.plot([], [], color="k", lw=1, ls="--", label="semi-infinite analytic")
    ax_a.plot([], [], "o", color="0.7", mec="k", mew=0.6, ms=6, label="yr 70: F = F$_{2x}$")
    ax_a.set(xlim=(0, plot_yrs[-1] / HELD_SCALES[0][0]), ylim=(0, None),
             xlabel=r"years since forcing onset / $\mathcal{T}$", ylabel=r"$\theta_0 / \mathcal{T}$ ($\degree$C yr$^{-1}$)",
             title=f"(a) surface response to a linear ramp, F$_{{2x}}$ at yr 70 "
                   f"({t_final / YEAR:g} yr runs; no slab, semi-infinite)")
    ax_a.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    style_axes(ax_a)

    # zeta = z/sqrt(kappa T) = z/H; theta/T = rate*g(zeta, tau) for every run,
    # so the runs tile one field. Largest tau range first: the shorter-T runs'
    # wider ranges are overdrawn at small tau by the longer-T runs.
    ax_b = fig.add_subplot(gs[1, :-1])
    tau_max = plot_yrs[-1] / HELD_SCALES[0][0]
    zeta_max = 2 * np.sqrt(tau_max)
    vmax = max(r["dT"].max() / scale for r, _, _, scale, *_ in runs)
    for r, _, _, scale, zeta, tau in runs:
        k = zeta <= 1.05 * zeta_max
        pc = ax_b.pcolormesh(tau, zeta[k], r["theta"][k] / scale, shading="nearest",
                             cmap="viridis", vmin=0.0, vmax=vmax)
    tt = np.linspace(0.0, tau_max, 201)
    zz = np.linspace(0.0, zeta_max, 121)
    cs = ax_b.contour(tt, zz, rate * ramp_field(zz, tt), levels=np.linspace(0, vmax, 9)[1:-1],
                      colors="k", linewidths=0.8, linestyles="--")
    ax_b.clabel(cs, fmt="%.2f", fontsize=7)
    ax_b.set(xlim=(0, tau_max), ylim=(zeta_max, 0), xlabel=r"$\tau = t/\mathcal{T}$",
             ylabel=r"$\zeta = z/\sqrt{\kappa \mathcal{T}}$",
             title=r"(b) $\theta/\mathcal{T}$: numeric (color, all runs) vs analytic (dashed contours)")
    fig.colorbar(pc, ax=ax_b, pad=0.01, fraction=0.03,
                 label=r"$\theta / \mathcal{T}$ ($\degree$C yr$^{-1}$)")
    style_axes(ax_b)

    # (c) depth profiles at the tau marked on (b), numeric from the T = 5 yr run
    # (the only one reaching tau = 20) vs the analytic g (dashed).
    ax_c = fig.add_subplot(gs[1, -1], sharey=ax_b)
    r, _, _, scale, zeta, tau = runs[0]
    for tau_c, c in zip(LINEAR_PROFILE_TAUS, ["#d62728", "#ff7f0e", "#e377c2"]):
        j = np.argmin(np.abs(tau - tau_c))
        ax_c.plot(r["theta"][:, j] / scale, zeta, color=c, lw=2.5, label=rf"$\tau$ = {tau_c:g}")
        ax_c.plot(rate * ramp_field(zz, tau_c)[:, 0], zz, color="k", lw=1, ls="--")
        ax_b.axvline(tau_c, color=c, lw=1.5, ls=":")
    ax_c.plot([], [], color="k", lw=1, ls="--", label="analytic")
    ax_c.set(xlim=(0, None), xlabel=r"$\theta / \mathcal{T}$ ($\degree$C yr$^{-1}$)",
             title="(c) depth profiles")
    ax_c.tick_params(labelleft=False)
    ax_c.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax_c)
    fig.text(0.5, 0.012, rf"$\theta/\mathcal{{T}} = [F_{{2x}}/(\lambda\,T_{{2x}})]\;g(\zeta,\tau)$, "
             rf"$F_{{2x}}/(\lambda T_{{2x}})$ = {rate:.3f} $\degree$C yr$^{{-1}}$ ($T_{{2x}}$ = 70 yr), "
             r"$g(\zeta,\tau) = \int_0^\tau h(\zeta,u)\,du$: all runs collapse onto it",
             ha="center", fontsize=9)
    fig.tight_layout(h_pad=3, rect=(0, 0.04, 1, 1))
    savefig(fig, "held_linear_response_numeric.png")


if __name__ == "__main__":
    main()
