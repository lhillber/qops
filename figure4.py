import qca
import matplotlib.gridspec as gridspec
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import cycle
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import glob, os
from figure2 import select

from matplotlib import rc
rc("text", usetex=True)
font = {"size": 9, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}",
    r"\usepackage{sansmath}",  # sanserif math
    r"\sansmath",
]



def fit_page(sba):
    L = len(sba) + 1
    ells = np.arange(L - 1)

    def page_func(ell, a, logK):
        return (ell + 1) * np.log(a) - np.log(1 + a ** (-L + 2 * (ell + 1))) + logK

    popt, pcov = curve_fit(page_func, ells, sba)

    def func(ell):
        return page_func(ell, *popt)

    return (func, popt, pcov)


def simple_plot(plot_fname):
    figa, axa = plt.subplots(1, 1, figsize=(1.5, 1.3))
    for o, (color, S) in enumerate(zip(cs, Skey)):
        for IC in ICkey:
            sb_avg = []
            sb_std = []
            sb_res = []
            ns = []
            for L in Lkey:
                ells = np.arange(L - 1)
                dells = np.linspace(0, L - 2, 100)
                sim = select(L, S, IC, "H", "1-00")
                if sim is None:
                    print("No sim!")
                    continue
                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]
                sb = h5file["sbond-2"][:]
                sba = np.mean(sb[100:], axis=0)
                sbd = np.std(sb[100:], axis=0)
                sb_avg += [sba[L // 2]]
                sb_std += [sbd[L // 2]]
                axa.set_xlabel("$\\ell$")
                axa.set_ylabel("Iteration")
                # axa.label_outer()
                func, popt, pcov = fit_page(sba)
                sb_res += [(func(ells) - sba) ** 2]
                if L in (18,):
                    ls = lss[L]
                    axa.fill_between(
                        ells,
                        sba + sbd,
                        sba - sbd,
                        facecolor=color,
                        alpha=0.4,
                        zorder=o
                    )
                    axa.scatter(
                        ells,
                        sba,
                        color=color,
                        marker="o",
                        s=2,
                        zorder=o
                    )
                    axa.plot(
                        dells,
                        func(dells),
                        color=color,
                        ls=ls,
                        lw=1,
                        label="$R={}$".format(S),
                        zorder=o
                    )
                    axa.set_xlabel("$\\ell$")
                    axa.set_ylabel("$\\overline{S}_{\\ell}$")
                    axa.xaxis.set_major_locator(MaxNLocator(integer=True))
                    axa.set_xticks([0, 8, 16])
                    axa.set_xticklabels(["$1$", "$9$", "$17$"])
                    axa.set_yticks([0, 4, 8])
    figa.subplots_adjust(left=0.35, bottom=0.32, top=1, right=0.95) 
    #axa.legend(
    #    loc=2,
    #    fontsize=8,
    #    handlelength=1,
    #    labelspacing=0.2,
    #    handletextpad=0.2,
    #    frameon=False,
    #)
    qca.multipage(plot_fname, clip=False)


if __name__ == "__main__":


    def ket(x):
        return "$\\vert " + x + "\\rangle$"

    lines = {
        "d3": {"name": ket("d3"), "ls": "-", "c": "C1", "m": "s"},
        "d4": {"name": ket("d4"), "ls": "-", "c": "C2", "m": "^"},
        "d5": {"name": ket("d5"), "ls": "-", "c": "C3", "m": "d"},
        "d6": {"name": ket("d6"), "ls": "-", "c": "C4", "m": "*"},
        "d7": {"name": ket("d7"), "ls": "-", "c": "C5", "m": "v"},
    }


    Skey = [13, 14, 6, 1]
    cs = ["limegreen", "darkorange", "crimson", "darkturquoise"]

    Lkey = [10, 12, 14, 16, 18]
    lss = {18: "-", 14: "--", 10: ":"}
    ICkey = ("d4",)
    plot_fname = "figures/figure4/pagecurves_V1.pdf"

    fig = plt.figure(figsize=(3.4, 2))
    gs = gridspec.GridSpec(2, 3)
    ax = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    axins1 = ax.inset_axes((1 - 0.335, 1 - 0.335, 0.3, 0.3))

    for color, S in zip(cs, Skey):
        for IC in ICkey:
            sb_avg = []
            sb_std = []
            sb_res = []
            ns = []
            for L in Lkey:
                ells = np.arange(L - 1)
                dells = np.linspace(0, L - 2, 100)
                sim = select(L, S, IC, "H", "1-00")
                if sim is None:
                    print("No sim!")
                    continue
                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]
                sb = h5file["sbond-2"][:]
                sba = np.mean(sb[100:], axis=0)
                sbd = np.std(sb[100:], axis=0)
                sb_avg += [sba[L // 2]]
                sb_std += [sbd[L // 2]]
                ax.set_xlabel("$\\ell$")
                ax.set_ylabel("Iteration")
                # ax.label_outer()
                func, popt, pcov = fit_page(sba)
                sb_res += [(func(ells) - sba) ** 2]
                if L in (18,):
                    ls = lss[L]
                    ax.fill_between(
                        ells,
                        sba + sbd,
                        sba - sbd,
                        color=color,
                        alpha=0.3
                    )
                    ax.scatter(
                        ells,
                        sba,
                        color=color,
                        marker="o",
                        s=2
                    )

                    ax.plot(
                        dells,
                        func(dells),
                        color=color,
                        ls=ls,
                        lw=1,
                        label="$R={}$".format(S),
                    )
                    ax.set_xlabel("$\\ell$")
                    ax.set_ylabel("$\\overline{S}_{\\ell}$")
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    if L == 18:
                        axins1.semilogx(
                            sb[:, int(L / 2)],
                            color=color,
                            lw=1,
                            label=lines[IC]["name"],
                        )
            sb_avg = np.array(sb_avg)
            sb_std = np.array(sb_std)
            ax2.errorbar(
                Lkey,
                sb_avg,
                yerr=sb_std,
                color=color,
                fmt="-o",
                capsize=1,
                elinewidth=1,
                markersize=2,
                lw=1,
            )
            sb_rss = [np.sum(res) for res in sb_res]
            ax3.plot(Lkey, sb_rss, "-o", color=color, lw=1, markersize=2)

    plt.subplots_adjust(wspace=0.7, hspace=0.1, bottom=0.22, right=0.97, top=0.97)

    ax2.set_ylabel("$\\overline{S}_{L/2}$", fontsize=9, labelpad=-2)
    ax2.set_yticks([2.0, 8.0])
    ax2.set_xticks([])

    ax3.set_ylabel("RSS", fontsize=9, labelpad=-11)
    ax3.set_xlabel("$L$")
    ax3.set_yticks([0, 0.4])
    ax3.set_xticks([10, 14, 18])

    ax.set_yticks([0, 4, 8])
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlim(right=17)
    ax.set_ylim(top=11)
    ax.legend(
        loc=2,
        fontsize=9,
        handlelength=1,
        labelspacing=0.2,
        handletextpad=0.2,
        frameon=False,
    )

    axins1.set_xticks([1, 1000])
    axins1.set_xticklabels([0, 3])
    axins1.set_yticks([0, 8])
    axins1.set_xlabel(r" $\log(t)$", fontsize=9, labelpad=-6)
    axins1.set_ylabel("$S_{L/2}$", fontsize=9, labelpad=-2)
    axins1.tick_params(axis="both", labelsize=9)
    axins1.axvline(100, color="k", lw=0.5)
    axins1.patch.set_alpha(1)
    qca.multipage(plot_fname, clip=False)
    print("plot saved to ", plot_fname)
    plt.close("all")
    simple_plot("figures/figure4/pagecurves-simple_V1.pdf")
