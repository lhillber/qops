import matrix as mx
from matrix import ops
import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import gamma
from scipy.optimize import curve_fit
from copy import copy
import measures as ms
from states import make_state
from qca import multipage
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    def brody_func(x, eta):
        b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
        return b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))

    popt, pcov = curve_fit(brody_func, x, n, bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov


IC = "c1_f0"
Skey = [6, 1, 13, 14]
bin_ubs = [7, 14, 35, 18]
L = 18
fig, axs = plt.subplots(len(Skey), 1, figsize=(4, 6), sharex=False)
fig2, axs2 = plt.subplots(len(Skey), 1, figsize=(4, 6), sharex=True)

for j, (bub, S) in enumerate(zip(bin_ubs, Skey)):
    sim = select(L, S, IC, V="H", BC="0")
    h5file = sim["h5file"]
    d = h5file["cut_half"][:]
    etas = []
    detas = []
    svns = []
    i = 0

    for rho in d[2:]:
        bins = np.linspace(0, bub, 20)
        s = eigvalsh(rho)
        N = len(s)
        s = s[s > 1e-3]
        N = len(s)/N
        s /= np.mean(s)
        n, bin, _ = axs[j].hist(
                s, density=True, alpha=1, histtype="step", bins=bins, log=False
        )
        axs[j].set_title("$R = %d$" % S)
        x = (bin[1:] + bin[:-1]) / 2.0
        xs = np.linspace(x[0], x[-1], 300)
        xs = xs[xs > 0]
        func, popt, pcov = brody_fit(x, n)
        detas.append(np.sqrt(np.diag(pcov)[0]))
        axs[j].plot(xs, func(xs))
        etas.append(popt[0])
        i += 1

    etas = np.array(etas)
    detas = np.array(detas)
    ts = np.arange(2, len(etas)+2)
    mask = detas < 1
    etas = etas[mask]
    detas = detas[mask]
    ts = ts[mask]
    axs2[j].errorbar(ts, etas, yerr=detas, fmt="o", elinewidth=1, markersize=2, c="k")
    axs2[j].set_title("$R = %d$" % S)
    axs2[j].set_ylabel("$\eta$")
axs[3].set_xlabel("$s_i / \overline{s_i}$")
axs2[3].set_xlabel("$t$")
fig.tight_layout()
fig2.tight_layout()

multipage("figures/spectrum_statistics_fixed-15bins.pdf", figs=[fig2])
