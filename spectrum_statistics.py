import numpy as np
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit
from qca import multipage
from figure3 import select

# plotting defaults
import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
font = {"size": 12, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc("font", **font)


def brody_fit(x, n):
    def brody_func(x, eta, A):
        b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
        return A*b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))

    popt, pcov = curve_fit(brody_func, x, n, p0=[0.0, 1.0], bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov

L = 18
IC = "c1_f0"
Skey = [6, 1, 13, 14]
cs = ["darkturquoise", "darkorange", "chartreuse", "crimson"]

fig, axs = plt.subplots(1, 1, figsize=(4, 3), sharex=False)
fig2, axs2 = plt.subplots(1, 1, figsize=(4, 3), sharex=True)
fig3s = []

for j, (c, S) in enumerate(zip(cs, Skey)):
    sim = select(L, S, IC, V="H", BC="0")
    h5file = sim["h5file"]
    d = h5file["cut_half"][:]
    etas = []
    detas = []
    svns = []
    ii = 0
    t0 = 10

    fig3, axs3 = plt.subplots(3,3,figsize=(4,4), sharex=True, sharey=True)
    for ti, rho in enumerate(d[t0: 100]):
        e = eigvalsh(rho)
        es = e[e>1e-13]
        NN = len(es)
        es = np.sort(es)
        es = es[NN//3: 2*NN//3]
        ns = range(len(es))
        s = es[1:] - es[:-1]
        s /= np.mean(s)

        n, bin, _ = axs.hist(
            s, density=True, alpha=1, histtype="step", bins=15, log=False
        )

        x = (bin[1:] + bin[:-1]) / 2.0
        xs = np.linspace(x[0], x[-1], 300)
        xs = xs[xs > 0]

        func, popt, pcov = brody_fit(x, n)
        detas.append(np.sqrt(np.diag(pcov)[0]))
        etas.append(popt[0])

        if ti%10 == 0:
            row, col = ii// 3, ii%3
            ax3 = axs3[row, col]
            dn = n[1] - n[0]
            x = np.insert(x, 0, 0)
            x = np.insert(x, -1, 0)
            n = [n[0] - dn/2] + n + [n[-1] + dn/2]
            ax3.step(x, n, where="mid")
            ax3.plot(xs, func(xs))
            ax3.set_title(f"t={t0+ti}",pad=-13)
            fig3.suptitle("$R = %d$" % S)
            ii += 1
            if col == 1 and row == 2:
                ax3.set_xlabel("$\delta E/\overline{\delta E}$")
            if col == 0 and row == 1:
                ax3.set_ylabel("density")
            ax3.tick_params(direction='inout')

    fig3.subplots_adjust(hspace=0, wspace=0)
    fig3s.append(fig3)
    etas = np.array(etas)
    detas = np.array(detas)
    ts = np.arange(2, len(etas) + 2)
    mask = detas < 2
    etas = etas[mask]
    detas = detas[mask]
    ts = ts[mask]

    if S == Skey[0]:
        pass
    else:
        if S == Skey[1]:
            label = r"$R = %s$" % S
        else:

            label = str(S)
        axs2.plot(ts, etas, markersize=2, color=c, label=label)
        axs2.errorbar(ts, etas, yerr=detas, fmt="o", elinewidth=1, markersize=2, c=c)

    #axs2.errorbar(ts, etas, color=c)

axs2.legend(loc="lower left")
axs.set_xlabel("$\delta E / \overline{\delta E}$")
axs2.set_xlabel("$t$")
axs2.set_ylabel("$\eta$")
fig.tight_layout()
fig2.tight_layout()

multipage("figures/spectrum_statistics_fixed-15bins.pdf", figs=[fig2]+fig3s)
