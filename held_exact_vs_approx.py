"""
Exact vs approximate step response for the semi-infinite diffusive ocean.
Held, GFDL blog #51 (2014); analytic solution Carslaw & Jaeger (1959) sec. 2.7.

Nondimensional (tau = t/T):   dh/dtau = d2h/dzeta2,  -dh/dzeta|_0 = 1 - h

    h_exact  = 1 - exp(tau) erfc(sqrt(tau))
    h_approx = 1 / (1 + (pi tau)^-1/2)

Linear ramp forcing, F(t) = F_max*t/70 for all t (no plateau at yr 70).
Duhamel + parts gives
    T(t) ~ H(tau),   H(tau) = int_0^tau h du
    H_exact  = tau + 1 - 2 sqrt(tau/pi) - exp(tau) erfc(sqrt(tau))
    H_approx = tau     - 2 sqrt(tau/pi) + (2/pi) ln(1 + sqrt(pi tau))

The closed form H_exact is validated here against a brute-force numerical
convolution of the impulse response g = dh/dtau,
    g(tau) = 1/sqrt(pi tau) - exp(tau) erfc(sqrt(tau))
integrated directly against the linear ramp forcing (Held's eq. for T(tau)).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erfcx  # erfcx(x) = exp(x^2) erfc(x), overflow-free

RAMP, TMAX = 70.0, 200.0
SCALES = [(5, "#c0392b"), (41, "#2c5aa0"), (80, "#3fa34d")]

pos = lambda x: np.maximum(np.asarray(x, float), 0.0)

h_exact = lambda tau: 1.0 - erfcx(np.sqrt(pos(tau)))
h_approx = lambda tau: np.sqrt(np.pi * pos(tau)) / (1 + np.sqrt(np.pi * pos(tau)))

H_exact = lambda tau: pos(tau) + 1 - 2 * np.sqrt(pos(tau) / np.pi) - erfcx(np.sqrt(pos(tau)))
H_approx = lambda tau: pos(tau) - 2 * np.sqrt(pos(tau) / np.pi) + (2 / np.pi) * np.log1p(np.sqrt(np.pi * pos(tau)))

g_exact = lambda tau: 1 / np.sqrt(np.pi * tau) - erfcx(np.sqrt(tau))


def response(t, scale, H):
    """Unnormalized linear-ramp response (unit forcing/sensitivity, F/lambda=1):
    T(t) = (T/RAMP) * H(tau), the T/RAMP prefactor being the ramp's slope
    1/tau70 in tau-space (tau70 = RAMP/T)."""
    t = np.asarray(t, float)
    return (scale / RAMP) * (H(t / scale))


def response_numeric(t, scale):
    """Same quantity by direct quadrature of int F(xi) g(tau-xi) dxi.

    No closed form used, so agreement with response(..., H_exact) is an
    independent check of the analytic antiderivative.
    """
    tau70 = RAMP / scale

    def conv(tau):
        # F(xi) = xi/tau70; g has an integrable xi^-1/2 endpoint singularity
        f = lambda xi: (xi / tau70) * g_exact(tau - xi)
        return quad(f, 0, tau, points=[tau], limit=400)[0]

    return np.array([conv(x / scale) for x in np.atleast_1d(t)])


def dH_dT(T, h):
    """Analytic d/dT[H(RAMP/T)] at fixed t=RAMP, via chain rule with tau=RAMP/T:
    dH/dT = dH/dtau * dtau/dT = h(tau) * (-RAMP/T**2).
    """
    T = np.asarray(T, float)
    tau = RAMP / T
    return -RAMP / T**2 * h(tau)


def dH_dT_numeric(T, H, step=1e-3):
    """Same derivative by central finite difference of H itself -- independent
    of the chain-rule algebra in dH_dT, so agreement checks that algebra."""
    T = np.asarray(T, float)
    return (H(RAMP / (T + step)) - H(RAMP / (T - step))) / (2 * step)


# Physical parameters behind T: T = kappa*(rho c)^2/lambda^2 (Held's semi-infinite
# diffusive slab), so a change in the diffusivity kappa or feedback lambda acts on
# the response only through T -- chain rule below. Nominal kappa/lambda are the
# ECHAM5 4xCO2 fit (longrun_data_param_fit_script_diffusive-eps-joint_fixedRadiativeParams.py),
# same values used in extract_li2012_fig3.py's analytic overlay.
RHO_CP = 4.186e6                # vol. heat capacity of seawater [J m^-3 K^-1]
YEAR_S = 365.25 * 24 * 3600.0   # seconds per year
KAPPA0 = 4.063496069778497e-05  # ECHAM5 fitted kappa_pde [m^2/s]
LAMBDA0 = 0.5751284118979058    # ECHAM5 fitted lambda [W m^-2 K^-1]


def dResponse_dkappa(T, h, lam=LAMBDA0):
    """dH/dkappa at fixed lambda: dH/dT * dT/dkappa, dT/dkappa = (rho c)^2/(lambda^2*YEAR_S)."""
    dT_dkappa = RHO_CP**2 / (lam**2 * YEAR_S)
    return dH_dT(T, h) * dT_dkappa


def dResponse_dlambda(T, h, kappa=KAPPA0):
    """dH/dlambda at fixed kappa: dH/dT * dT/dlambda, with lambda implied by T at
    that kappa (lambda = sqrt(kappa*(rho c)^2/(T*YEAR_S))), so dT/dlambda = -2T/lambda."""
    T = np.asarray(T, float)
    lam = np.sqrt(kappa * RHO_CP**2 / (T * YEAR_S))
    return dH_dT(T, h) * (-2 * T / lam)


fig = plt.figure(figsize=(12, 8.5))
_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
# axes[0..1] are the top row (a) step response and (b) linear response;
# axes[2..3] the bottom row (c) N and (d) kappa_N of the same experiment.
axes = [fig.add_subplot(_gs[r, i]) for r in range(2) for i in range(2)]
tau, t = np.linspace(1e-9, 10, 2000), np.linspace(1e-6, TMAX, 2000)

axes[0].plot(tau, h_exact(tau), "#2c5aa0", lw=2, label=r"exact")
axes[0].plot(tau, h_approx(tau), "#c0392b", lw=2, label=r"approx")
axes[0].set(xlabel=r"$\tau=t/\mathcal{T}$", ylabel="equilibration fraction $h$",
            title="(a) step response", xlim=(0, 10), ylim=(0, 1))

for scale, c in SCALES:
    exact, approx = response(t, scale, H_exact), response(t, scale, H_approx)
    axes[1].plot(t, exact, c, lw=2, label=f"{scale} yrs")
    axes[1].plot(t, approx, c, lw=1.5, ls="--")

    # independent check: numerical convolution should land on the analytic curve
    t_chk = np.array([5, 20, 40, 60, 70, 90, 120, 160, 200], float)
    num = response_numeric(t_chk, scale)
    print(f"T={scale:3d} yr   max |analytic - numerical| = "
          f"{np.abs(num - response(t_chk, scale, H_exact)).max():.2e}")
    print(f"T={scale:3d} yr   t=200: exact {exact[-1]:.3f}  approx {approx[-1]:.3f}"
          f"  ({100*(approx[-1]-exact[-1])/exact[-1]:+.1f}%)")

axes[1].set(xlabel="time (yrs)",
            # T_eq = F_max/lambda, the equilibrium warming for the yr-70 forcing
            ylabel=r"$T/T_{eq}=(\mathcal{T}/70)\,H(\tau)$",
            title="(b) linear response\nsolid=exact; dashed=approx",
            xlim=(0, TMAX))

T_chk = np.array([5, 10, 20, 40, 60, 80, 100], float)

# The dH/dT, dH/dkappa and dH/dlambda sensitivities are no longer plotted; the
# analytic-vs-finite-difference check on dH_dT is kept as console output.
print("d/dT[H(70/T)] analytic vs numerical central difference:")
for T0 in T_chk:
    a_e, n_e = dH_dT(T0, h_exact), dH_dT_numeric(T0, H_exact)
    a_a, n_a = dH_dT(T0, h_approx), dH_dT_numeric(T0, H_approx)
    print(f"   T={T0:5.1f} yr   exact {a_e:+.6f} vs {n_e:+.6f}"
          f"   approx {a_a:+.6f} vs {n_a:+.6f}")

# --- bottom row: N, T and kappa_N for the linear-ramp experiment --------------
# All three panels inherit the F/lambda normalization of response(): with the
# linear ramp F(t) = F_max*f(t), f = t/RAMP (the `pos` curve of panel b), the
# surface temperature is T(t) = (F_max/lambda)*r(t) with r = response(...), and
# the net TOA imbalance is N(t) = F - lambda*T = F_max*(f - r). F_max therefore
# drops out of both -- writing it as F_max = lambda*T_eq, the equilibrium warming
# of panel (b)'s axis, so that the two panels share one normalization,
#     N/(lambda*T_eq) = f(t) - r(t)      (panel c)
# leaving one curve per diffusive timescale T, exact (solid) and approximate
# (dashed) -- the same three experiments as panel (b). The companion
# T/(F_max/lambda) = r(t) is already panel (b) itself, so it is not repeated.
#
# NOTE the kappa of panel (d) is NOT the diffusivity kappa [m^2/s] of KAPPA0
# above: it is the Gregory-style ocean heat-uptake efficiency, the heat the ocean
# absorbs per degree of surface warming,
#     kappa_N(t) = N(t)/T(t) = (F(t) - lambda*T(t)) / T(t) = lambda*(f - r)/r,
# i.e. lambda times the panel-(b) gap between the no-lag forcing and the lagged
# response, per unit response. F_max cancels here too (the dimensional
# cross-check below confirms the cancellation) but lambda does not, so (d) is the
# one bottom panel carrying units [W m^-2 K^-1].
def imbalance(t, scale, H):
    """N/(lambda*T_eq) = N/F_max = f - r: the fraction of the instantaneous
    forcing that is still
    going into the ocean rather than being radiated by the warmed surface."""
    return response(t, scale, pos) - response(t, scale, H)


def uptake_efficiency(t, scale, H, lam=LAMBDA0):
    """kappa_N = lambda*(f - r)/r [W m^-2 K^-1] at time t for timescale(s) `scale`."""
    r = response(t, scale, H)
    return lam * (response(t, scale, pos) - r) / r


def uptake_efficiency_dimensional(t, scale, H, lam=LAMBDA0, F_max=7.4):
    """Same quantity built dimensionally from N and T with an explicit F_max --
    no algebraic cancellation, so agreement checks the (f - r)/r shortcut."""
    T_surf = (F_max / lam) * response(t, scale, H)          # K
    F_t = F_max * np.asarray(t, float) / RAMP               # W/m^2
    return (F_t - lam * T_surf) / T_surf                    # W m^-2 K^-1


# kappa_N diverges as t -> 0 (r ~ t^3/2 while f ~ t, so kappa_N ~ t^-1/2: at the
# first instant the ocean absorbs everything and T is still zero), so panel (d)
# starts its time axis at 1 yr rather than the 1e-6 of the panel-(b) grid.
t_d = np.linspace(1.0, TMAX, 2000)

for scale, c in SCALES:
    lbl = rf"$\mathcal{{T}}$={scale} yrs"
    axes[2].plot(t, imbalance(t, scale, H_exact), c, lw=2, label=lbl)
    axes[2].plot(t, imbalance(t, scale, H_approx), c, lw=1.5, ls="--")
    axes[3].plot(t_d, uptake_efficiency(t_d, scale, H_exact), c, lw=2, label=lbl)
    axes[3].plot(t_d, uptake_efficiency(t_d, scale, H_approx), c, lw=1.5, ls="--")

for ax in axes[2:]:
    ax.axvline(RAMP, color="0.6", lw=0.8, ls=":")

_dashed = "\nsolid=exact; dashed=approx"
axes[2].set(xlabel="time (yrs)",
            title="(c) net TOA imbalance" + _dashed, xlim=(0, TMAX))
# Written out unnormalized as N = F*f - lambda*T, with F the yr-70 forcing,
# F(t) = F*(t/70) and T = (F/lambda)*(T/70)*H(tau) from panel (b). The plotted
# values are that divided by F -- imbalance() -- since F cancels throughout.
axes[2].set_ylabel(r"$N=F(t/70)-\lambda(F/\lambda)(\mathcal{T}/70)\,H(\tau)$",
                   fontsize=9)
axes[2].legend(fontsize=8, loc="upper left")

axes[3].set(xlabel="time (yrs)",
            ylabel=r"$\kappa_N=\dfrac{F(t)-\lambda T}{T}$ [W m$^{-2}$ K$^{-1}$]",
            title=rf"(d) heat-uptake efficiency ($\lambda$={LAMBDA0:.3f} W m$^{{-2}}$ K$^{{-1}}$)"
                  + _dashed,
            xlim=(t_d[0], TMAX))
axes[3].legend(fontsize=8, loc="upper right")

print("kappa_N = (F - lambda*T)/T  [W m^-2 K^-1]:  (f-r)/r form vs dimensional N/T")
for scale, _ in SCALES:
    for t0 in (10.0, RAMP, TMAX):
        a = uptake_efficiency(t0, scale, H_exact)
        n = uptake_efficiency_dimensional(t0, scale, H_exact)
        print(f"   T={scale:3d} yr  t={t0:5.1f} yr:  {a:+.6f} vs {n:+.6f}")

axes[1].axvline(RAMP, color="0.6", lw=0.8, ls=":")
for ax in axes:
    ax.grid(alpha=0.25, lw=0.6)
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=8, loc="lower right")

fig.tight_layout(h_pad=2.5)
fig.savefig("figures/held_exact_vs_approx.png", dpi=160)
