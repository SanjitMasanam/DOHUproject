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
integrated directly against the linear ramp forcing (Held's eq. for T(tau)),
    T(tau) = int_0^tau F(xi) g(tau - xi) dxi.
A second check builds g straight from the exact step response, g = dh_exact/dtau
by finite difference, so it also tests the hand-derived g_exact formula.

Matched 1-box, c dT/dt = -lambda T + F with c = lambda*T (the heat capacity of a
layer V_beta*T = H deep, so its relaxation time is the diffusive T):
    h_box = 1 - exp(-tau),  g_box = exp(-tau),  H_box = tau - 1 + exp(-tau)
compared with the diffusive model in figures/held_box_vs_diffusive.png.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erfcx  # erfcx(x) = exp(x^2) erfc(x), overflow-free

RAMP, TMAX = 70.0, 200.0
SCALES = [(5, "#c0392b"), (41, "#2c5aa0"), (80, "#3fa34d"), (200, "#257ec2"), (1200, "#d11fb6")]

pos = lambda x: np.maximum(np.asarray(x, float), 0.0)

h_exact = lambda tau: 1.0 - erfcx(np.sqrt(pos(tau)))
h_approx = lambda tau: np.sqrt(np.pi * pos(tau)) / (1 + np.sqrt(np.pi * pos(tau)))

H_exact = lambda tau: pos(tau) + 1 - 2 * np.sqrt(pos(tau) / np.pi) - erfcx(np.sqrt(pos(tau)))
H_approx = lambda tau: pos(tau) - 2 * np.sqrt(pos(tau) / np.pi) + (2 / np.pi) * np.log1p(np.sqrt(np.pi * pos(tau)))

g_exact = lambda tau: 1 / np.sqrt(np.pi * tau) - erfcx(np.sqrt(tau))

# 1-box with c = lambda*T = rho c_p V_beta T, the heat capacity of a layer of
# Held's depth scale H = V_beta*T, so c/lambda = T: same lambda, T and H as the
# diffusive model, all the heat stored in one well-mixed layer.
h_box = lambda tau: -np.expm1(-pos(tau))
g_box = lambda tau: np.exp(-tau)
H_box = lambda tau: pos(tau) + np.expm1(-pos(tau))


def g_from_h(tau, h=h_exact, eps=1e-5):
    """g = dh/dtau by central difference of the step response itself. The step
    is relative to tau so it never crosses tau=0, where g ~ tau^-1/2."""
    return (h(tau * (1 + eps)) - h(tau * (1 - eps))) / (2 * eps * tau)


def response(t, scale, H):
    """Unnormalized linear-ramp response (unit forcing/sensitivity, F/lambda=1):
    T(t) = (T/RAMP) * H(tau), the T/RAMP prefactor being the ramp's slope
    1/tau70 in tau-space (tau70 = RAMP/T)."""
    t = np.asarray(t, float)
    return (scale / RAMP) * (H(t / scale))


def response_numeric(t, scale, g=g_exact, F=None, knots=()):
    """Same quantity by direct quadrature of int F(xi) g(tau-xi) dxi.

    No closed form used, so agreement with response(..., H_exact) is an
    independent check of the analytic antiderivative. With g=g_from_h the
    kernel comes directly from h_exact, checking g_exact as well.

    F is the forcing as a function of time in years (in units of F_max, so the
    result is lambda*T/F_max); it defaults to the linear ramp t/RAMP. `knots`
    are times [yr] where F has kinks (e.g. the years of a tabulated series),
    passed to quad as breakpoints.
    """
    if F is None:
        F = lambda tt: tt / RAMP
    knots = np.asarray(knots, float) / scale

    def conv(tau):
        # g has an integrable xi^-1/2 endpoint singularity at xi = tau
        f = lambda xi: F(xi * scale) * g(tau - xi)
        pts = [k for k in knots if 0 < k < tau] + [tau]
        return quad(f, 0, tau, points=pts, limit=max(400, 2 * len(pts)))[0]

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
    num_h = response_numeric(t_chk, scale, g_from_h)
    print(f"T={scale:3d} yr   max |analytic - numerical| (g = dh_exact/dtau) = "
          f"{np.abs(num_h - response(t_chk, scale, H_exact)).max():.2e}")
    print(f"T={scale:3d} yr   t=200: exact {exact[-1]:.3f}  approx {approx[-1]:.3f}"
          f"  ({100*(approx[-1]-exact[-1])/exact[-1]:+.1f}%)")
    num_box = response_numeric(t_chk, scale, g_box)
    print(f"T={scale:3d} yr   1-box max |analytic - numerical| = "
          f"{np.abs(num_box - response(t_chk, scale, H_box)).max():.2e}")

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


# --- Held's historical-forcing figure: WMGHG forcing through the convolution ---
# T(t) = int_0^t F(xi) g(t - xi) dxi with F the well-mixed GHG forcing of
# data/forcing_1850_2100.txt, integrated from 1850 (forcing taken as switching
# on there) and plotted 1880-2011. Forcing is normalized by F2X and temperature
# by the TCR, i.e. the response at yr 70 of a linear ramp reaching F2X there --
# response_numeric(RAMP, scale) in the same units -- so lambda drops out.
F2X = 5.35 * np.log(2)          # forcing for doubled CO2 [W m^-2], Myhre et al. (1998)
yr, F_ghg = np.loadtxt("data/forcing_1850_2100.txt", skiprows=1, usecols=(0, 3), unpack=True)
F_of_t = lambda tt: np.interp(tt, yr - yr[0], F_ghg / F2X)
plot_yrs = yr[(yr >= 1880) & (yr <= 2011)]

fig2, ax = plt.subplots(figsize=(6.5, 5))
WMGHG_SCALES = [(1, "#8c564b"), (5, "#c0392b"), (80, "#3fa34d"), (200, "#257ec2"), (1200, "#d11fb6"), (10000, "#faee0a")]
for scale, c in WMGHG_SCALES:
    tcr = response_numeric(RAMP, scale)[0]
    T_hist = response_numeric(plot_yrs - yr[0], scale, F=F_of_t, knots=yr - yr[0])
    ax.plot(plot_yrs, T_hist / tcr, c, lw=2, label=f"{scale} yrs")
    print(f"T={scale:3d} yr   T/TCR in {plot_yrs[-1]:.0f} = {T_hist[-1] / tcr:.3f}")
ax.plot(plot_yrs, F_of_t(plot_yrs - yr[0]), "k", lw=1.5, label="forcing/F2X")
ax.set(xlabel="year", ylabel="temperature/TCR", xlim=(1870, 2020))
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=9, loc="upper left")
fig2.tight_layout()
fig2.savefig("figures/held_wmghg_response.png", dpi=160)


# --- Held's ramp-then-hold figure: linear ramp to yr 70, constant after --------
# F(t) = F2X*min(t, 70)/70, through the same convolution. Temperature is divided
# by the TCR, the response at yr 70 (where the ramp ends, so it is the same TCR
# as above), so F2X and lambda drop out.
F_hold = lambda tt: np.minimum(tt, RAMP) / RAMP
t_hold = np.linspace(1e-6, TMAX, 401)

fig3, ax = plt.subplots(figsize=(6.5, 5))
HOLD_SCALES = [(1, "#8c564b"), (5, "#c0392b"), (20, "#2c5aa0"), (80, "#3fa34d"),
               (200, "#e69f00"), (1200, "#d11fb6"), (10000, "k")]
for scale, c in HOLD_SCALES:
    tcr = response_numeric(RAMP, scale)[0]
    T_hold = response_numeric(t_hold, scale, F=F_hold, knots=(RAMP,))
    ax.plot(t_hold, T_hold / tcr, c, lw=2, label=f"{scale} yrs")
    print(f"T={scale:3d} yr   T/TCR at t={TMAX:.0f} = {T_hold[-1] / tcr:.3f}")

# T -> inf limit (tau -> 0). With the small-tau series
#     e^tau erfc(sqrt(tau)) = 1 - 2 tau^1/2/sqrt(pi) + tau - 4 tau^3/2/(3 sqrt(pi)) + O(tau^2),
# the constant, tau^1/2 and tau terms of H_exact cancel, leaving
#     H(tau) ~ 4 tau^3/2 / (3 sqrt(pi)).
# The ramp-then-hold response is a ramp minus the same ramp delayed by RAMP, so
#     T/TCR -> [t^3/2 - (t - 70)_+^3/2] / 70^3/2,
# independent of T: the prefactor (T/RAMP)*T^-3/2 cancels against the TCR, so
# any scale passed to response() gives the same curve.
H_lead = lambda tau: 4 / (3 * np.sqrt(np.pi)) * pos(tau)**1.5
hold_lead = lambda tt: response(tt, 1.0, H_lead) - response(tt - RAMP, 1.0, H_lead)
T_lim = hold_lead(t_hold) / hold_lead(RAMP)
ax.plot(t_hold, T_lim, color="0.45", lw=1.5, ls="--",
        label=r"$\mathcal{T}\to\infty$: $[t^{3/2}-(t-70)_+^{3/2}]/70^{3/2}$")
print(f"T->inf       T/TCR at t={TMAX:.0f} = {T_lim[-1]:.3f}")

# T -> 0 limit during the ramp: the ocean equilibrates instantly, T = F/lambda,
# and TCR -> F_max/lambda, so T/TCR = t/70 for t <= 70.
ax.plot([0, RAMP], [0, 1], color="k", lw=1.2, ls=":", label=r"$\mathcal{T}\to 0$: $T = F/\lambda$, i.e. $t/70$")
ax.set(xlabel="time (yrs)", ylabel="temperature/TCR", xlim=(0, TMAX), ylim=(0, None))
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=9, loc="upper left")

# Inset: TCR/TEQ = response(RAMP, T, H_exact), the fraction of equilibrium
# realized at yr 70, against T, with a dot per timescale of the main panel.
axin = ax.inset_axes([0.6, 0.1, 0.37, 0.27])
T_line = np.geomspace(1.0, 1e4, 400)
axin.plot(T_line, response(RAMP, T_line, H_exact), "0.3", lw=1.2)
for scale, c in HOLD_SCALES:
    tcr_teq = response(RAMP, scale, H_exact)
    axin.plot(scale, tcr_teq, "o", color=c, mec="k", mew=0.5, ms=5)
    print(f"T={scale:5d} yr   TCR/TEQ = {tcr_teq:.3f}")
axin.set_xscale("log")
axin.set(xlim=(0.6, 1.7e4), ylim=(0, 1))
axin.set_xlabel(r"$\mathcal{T}$ (yrs)", fontsize=8, labelpad=1)
axin.set_ylabel("TCR/TEQ", fontsize=8, labelpad=1)
axin.tick_params(labelsize=7)
axin.grid(alpha=0.25, lw=0.6)
fig3.tight_layout()
fig3.savefig("figures/held_ramp_hold_response.png", dpi=160)


# --- Same figure with the ramp continued to TMAX instead of held at yr 70 -------
# F(t) = F2X*t/70 for all t, so T is the closed-form linear response of panel (b)
# (checked against response_numeric above) and T/TCR = H(t/T)/H(70/T).
fig3b, ax = plt.subplots(figsize=(6.5, 5))
for scale, c in HOLD_SCALES:
    T_ramp = response(t_hold, scale, H_exact) / response(RAMP, scale, H_exact)
    ax.plot(t_hold, T_ramp, c, lw=2, label=f"{scale} yrs")
    print(f"T={scale:5d} yr   continued ramp: T/TCR at t={TMAX:.0f} = {T_ramp[-1]:.3f}")

# T -> inf: H ~ tau^3/2, so T/TCR -> (t/70)^3/2. T -> 0: T = F/lambda throughout.
T_lim = response(t_hold, 1.0, H_lead) / response(RAMP, 1.0, H_lead)
ax.plot(t_hold, T_lim, color="0.45", lw=1.5, ls="--",
        label=r"$\mathcal{T}\to\infty$: $(t/70)^{3/2}$")
ax.plot(t_hold, t_hold / RAMP, color="k", lw=1.2, ls=":",
        label=r"$\mathcal{T}\to 0$: $T = F/\lambda$, i.e. $t/70$")
print(f"T->inf       continued ramp: T/TCR at t={TMAX:.0f} = {T_lim[-1]:.3f}")
ax.axvline(RAMP, color="0.6", lw=0.8, ls=":")
ax.set(xlabel="time (yrs)", ylabel="temperature/TCR", xlim=(0, TMAX), ylim=(0, None))
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=9, loc="upper left")

axin = ax.inset_axes([0.13, 0.35, 0.32, 0.19])
axin.plot(T_line, response(RAMP, T_line, H_exact), "0.3", lw=1.2)
for scale, c in HOLD_SCALES:
    axin.plot(scale, response(RAMP, scale, H_exact), "o", color=c, mec="k", mew=0.5, ms=5)
axin.set_xscale("log")
axin.set(xlim=(0.6, 1.7e4), ylim=(0, 1))
axin.set_xlabel(r"$\mathcal{T}$ (yrs)", fontsize=8, labelpad=1)
axin.set_ylabel("TCR/TEQ", fontsize=8, labelpad=1)
axin.tick_params(labelsize=7)
axin.grid(alpha=0.25, lw=0.6)
fig3b.tight_layout()
fig3b.savefig("figures/held_ramp_continued_response.png", dpi=160)


# --- Diffusive ocean vs the matched 1-box, timescale by timescale -------------
# Each T gets a 1-box with c = lambda*T (relaxation time T, depth H): solid =
# diffusive, dashed = 1-box, same colour per T. (c) and (d) divide each model by
# its own TCR, response_numeric(RAMP, scale, g) -- the TCR/TEQ of panel (b).
models = [("diffusive", g_exact, H_exact, "-"), ("1-box", g_box, H_box, "--")]
BOX_SCALES = [(T0, c) for T0, c in WMGHG_SCALES if T0 in (5, 200, 10000)]
fig4, ((ax_h, ax_r), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(12, 9))

tau = np.linspace(0.0, 10.0, 1001)
ax_h.plot(tau, h_exact(tau), "k", lw=2, label=r"diffusive, $1 - e^{\tau}\,\mathrm{erfc}\sqrt{\tau}$")
ax_h.plot(tau, h_box(tau), "0.45", lw=2, ls="--", label=r"1-box, $1 - e^{-\tau}$")
ax_h.set(xlabel=r"$\tau=t/\mathcal{T}$", ylabel="equilibration fraction $h$",
         title="(a) step response (same for every $\\mathcal{T}$)", xlim=(0, 10), ylim=(0, 1.02))
ax_h.legend(fontsize=9, loc="lower right")

T_line = np.geomspace(1.0, 1e4, 400)
for name, g, H, ls in models:
    ax_r.plot(T_line, [response(RAMP, T0, H) for T0 in T_line], "k", lw=1.5, ls=ls, label=name)
for scale, c in BOX_SCALES:
    for name, g, H, ls in models:
        ax_r.plot(scale, response(RAMP, scale, H), "o", color=c, mec="k", mew=0.6, ms=6)
ax_r.set_xscale("log")
ax_r.set(xlabel=r"diffusive / relaxation time scale $\mathcal{T}$ (yrs)", ylabel="TCR/TEQ",
         title="(b) fraction of equilibrium realized at yr 70 of the ramp", xlim=(1, 1e4), ylim=(0, 1))
ax_r.legend(fontsize=9, loc="upper right")

print("diffusive vs matched 1-box (each / own TCR):")
for scale, c in BOX_SCALES:
    out = []
    for name, g, H, ls in models:
        tcr = response_numeric(RAMP, scale, g)[0]
        T_hold = response_numeric(t_hold, scale, g, F=F_hold, knots=(RAMP,)) / tcr
        T_hist = response_numeric(plot_yrs - yr[0], scale, g, F=F_of_t, knots=yr - yr[0]) / tcr
        label = f"{scale} yrs" if name == "diffusive" else None
        ax_c.plot(t_hold, T_hold, c, lw=2, ls=ls, label=label)
        ax_d.plot(plot_yrs, T_hist, c, lw=2, ls=ls, label=label)
        out.append(f"{name}: TCR/TEQ {tcr:.3f}, T/TCR yr {TMAX:.0f} {T_hold[-1]:.3f}, "
                   f"{plot_yrs[-1]:.0f} {T_hist[-1]:.3f}")
    print(f"   T={scale:5d} yr   " + "   |   ".join(out))
ax_d.plot(plot_yrs, F_of_t(plot_yrs - yr[0]), "k", lw=1.5, label="forcing/F2X")
for ax in (ax_c, ax_d):
    ax.plot([], [], "0.3", lw=2, ls="-", label="diffusive")
    ax.plot([], [], "0.3", lw=2, ls="--", label=r"1-box, $c = \lambda\mathcal{T}$")
ax_c.set(xlabel="time (yrs)", ylabel="temperature/TCR", xlim=(0, TMAX), ylim=(0, None),
         title="(c) ramp to yr 70, then hold")
ax_c.legend(fontsize=8, loc="upper left", ncol=2)
ax_d.set(xlabel="year", ylabel="temperature/TCR", xlim=(1870, 2020),
         title="(d) WMGHG forcing")
ax_d.legend(fontsize=8, loc="upper left", ncol=2)
for ax in (ax_h, ax_r, ax_c, ax_d):
    ax.grid(alpha=0.25, lw=0.6)
fig4.tight_layout(h_pad=2)
fig4.savefig("figures/held_box_vs_diffusive.png", dpi=160)
