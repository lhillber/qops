import qca
import h5py
import numpy as np
import measures as ms
from states import make_state
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import glob
import os
from measures import renyi_entropy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.cm as cm


mpl.rcParams["text.latex.preamble"] = ["\\usepackage{amsmath}"]
font = {"size": 9, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc(*("font",), **font)

def shade_color(color, amount=0.5):
    """
    Lightens/darkens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def ket(x):
    return "$\\vert " + x + "\\rangle$"


def exp(x):
    return "$\\langle " + x + "\\rangle$"


def select(L, S, IC, V, BC):
    name = "L{}_T*_V{}_r1_S{}_M2_IC{}_BC{}.hdf5".format(L, V, S, IC, BC)
    data_dir_glob = "/home/lhillber/documents/research/cellular_automata/qeca/qops/qca_output/master_bak/data/{}".format(
        name
    )
    print(name)
    sims = [
        dict(
            L=int(os.path.basename(f).split("L")[1].split("T")[0][:-1]),
            T=int(os.path.basename(f).split("T")[1].split("V")[0][:-1]),
            V=os.path.basename(f).split("V")[1].split("r")[0][:-1],
            r=int(os.path.basename(f).split("r")[1].split("S")[0][:-1]),
            S=int(os.path.basename(f).split("S")[1].split("M")[0][:-1]),
            M=int(os.path.basename(f).split("M")[1].split("IC")[0][:-1]),
            IC=os.path.basename(f).split("IC")[1].split("BC")[0][:-1],
            BC=os.path.basename(f).split("BC")[1].split(".")[0],
            h5file=h5py.File(f),
        )
        for f in glob.glob(data_dir_glob)
    ]
    if len(sims) == 0:
        return None
    sim = sims[np.argmax(np.array([s["T"] for s in sims]))]
    return sim


def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
            for i in range(N + 1)
        ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def cmap_fromcolors(colors):
    cs = []
    for color in colors:
        try:
            c = mcolors.cnames[color]
        except:
            c = color
        cs.append(mcolors.to_rgba(c))
    N = len(cs)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (indices[i], cs[i - 1, ki], cs[i, ki])
            for i in range(N + 1)
        ]
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def colorbar_index(colorbar_label, ncolors, cmap, val_min, val_max):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("right", size="7%", pad="8%")
    colorbar = plt.colorbar(mappable, cax=cax)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(
        list(map(int, np.linspace(int(val_min), int(val_max), ncolors)))
    )
    colorbar.set_label(colorbar_label, rotation=0, y=0.55, labelpad=1.8)
    return cax


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def network_measures_scatter():
    fig, axs = plt.subplots(1, 2, figsize=(3, 1.8))
    BC = "0"
    Skey = [13, 1, 6]
    meas_axs = [["scenter-2", "C"], ["Y", "C"]]
    L = 10
    ICkey = ["f0", "f0-1", "f0-2", "f0-1-2", "d3", "d4", "d5", "d6", "d7"]
    V = "H"
    for col, measures in enumerate(meas_axs):
        d = {meas: {"avg": [], "std": [], "c": []} for meas in measures}
        for IC in ICkey:
            for S in Skey:
                ax = axs[col]
                for meas in measures:
                    sim = select(L, S, IC, V, BC)
                    if sim is None:
                        print("No sim!")
                        continue
                    S = sim["S"]
                    L = sim["L"]
                    IC = sim["IC"]
                    h5file = sim["h5file"]
                    d[meas]["c"] += [S]
                    if meas in ("scenter", "scenter-2"):
                        try:
                            d[meas]["avg"] += [np.mean(h5file[meas][100:])]
                            d[meas]["std"] += [np.std(h5file[meas][100:])]
                        except:
                            d[meas]["avg"] += [
                                np.mean(h5file["sbond"][100:, int(L / 2)])
                            ]
                            d[meas]["std"] += [
                                np.std(h5file["sbond"][100:, int(L / 2)])
                            ]
                    else:
                        d[meas]["avg"] += [np.mean(h5file[meas][100:])]
                        d[meas]["std"] += [np.std(h5file[meas][100:])]
                # ax.errorbar(
                #    d[measures[0]]["avg"],
                #    d[measures[1]]["avg"],
                #    xerr=d[measures[0]]["std"],
                #    yerr=d[measures[1]]["std"],
                # )
                #

        ax.scatter(
            d[measures[0]]["avg"],
            d[measures[1]]["avg"],
            c=[Skey.index(i) for i in d["C"]["c"]],
            cmap=cmap,
            linewidths=0.1,
            edgecolor="k",
            alpha=0.8,
            vmin=0,
            vmax=len(Skey) - 1,
        )

        print("enter")
        state = make_state(18, "R123")
        dMI = ms.get_MI(state, order=2)
        dC = ms.network_clustering(dMI)
        dY = ms.network_disparity(dMI)
        dsc = ms.get_center_entropy(state, order=2)
        Rdata = {"C": dC, "Y": dY, "scenter-2": dsc}

        # ax.axvline(Rdata[measures[0]], c="k", lw=1)
        # ax.axhline(Rdata[measures[1]], c="k", lw=1)
        ax.scatter(Rdata[measures[0]], Rdata[measures[1]], c="k", marker="x", s=60)
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.set_yticks([0.0, 0.1, 0.2])
        # if col==1:
        #    ax.set_xticks([0.1, 0.3, 0.5])
        # else:
        #    ax.set_xticks([4, 6, 8])
        ax.set_ylabel(lines[measures[1]]["name"])
        ax.set_xlabel(lines[measures[0]]["name"])
    cax = colorbar_index(
        r"$R$", ncolors=len(Skey), cmap=cmap, val_min=min(Skey), val_max=max(Skey)
    )

    cax.set_yticks(np.arange(min(Skey), max(Skey) + 1))
    cax.set_yticklabels(Skey)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    ax.label_outer()


if __name__ == "__main__":


    cs = ['olivedrab','indigo',  'crimson']
    cmap = mcolors.ListedColormap(cs)
    lines = {
        "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
        "exp-z": {"name": exp("\sigma^z"), "ls": "-", "c": "C5", "m": "v"},
        "s-2": {"name": " $s$", "ls": "-", "c": "C5", "m": "v"},
        "scenter": {"name": " $\Delta S^{(2)}_{L/2}$", "ls": "-", "c": "C5", "m": "v"},
        "scenter-2": {
            "name": " $\overline{S}^{(2)}_{L/2}$",
            "ls": "-",
            "c": "C5",
            "m": "v",
        },
        "C": {"name": " $\overline{\mathcal{C}}$", "ls": "-", "c": "C5", "m": "v"},
        "Y": {"name": " $\overline{\mathcal{Y}}$", "ls": "-", "c": "C5", "m": "v"},
    }

    plot_fname = "figures/figure3-L10_V2.pdf"

    network_measures_scatter()

    fig, axs = plt.subplots(1, 1, figsize=(3, 1.8), sharey=True, sharex=True)

    BC = "0"
    Vkey = ["H", "HP_45", "HP_90"]
    Skey = [13, 1, 6]
    L = 10
    ICkey = ["f0", "R123"]
    Rstack = []
    count = 0
    lss = ['-', '-.', '--']
    for V in Vkey:
        for IC in ICkey:
            for S in Skey:
                if S == 14:
                    ax = axs[1]
                    Vs = Vkey
                    cmap = plt.get_cmap("cool" , len(Vkey))
                    c = cs1[Vs.index(V)]
                    ls = lss[Vs.index(V)]
                else:
                    ax = axs
                    Vs = ["H"]
                    c = cmap(Skey.index(S))
                    #if S == 1:
                    #    c = shade_color(c, 1.3)
                    ls = '-'
                sim = select(L, S, IC, V, BC)
                if sim is None:
                    print("No sim!")
                    continue
                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]
                sc = h5file["scenter-2"]
                d = np.abs(np.diff(sc))
                d = moving_average(d, n=L)
                state = make_state(18, "R123")
                Rsc = ms.get_center_entropy(state, order=2)
                if IC[0] == "R":
                    Rstack += [d]
                    count += 1
                else:
                    if V in Vs:
                        ax.semilogy(d, c="k", lw=1.2, ls=ls)
                        ax.semilogy(d, c=c, lw=1, ls=ls)
                ax.set_ylabel(lines["scenter"]["name"])
                ax.set_xlabel("$t$")
                ax.label_outer()
    Rstack = np.array(Rstack)
    Ravg = np.mean(Rstack)
    Rstd = np.std(Rstack)
    print(Ravg, Rstd)
    ax.fill_between(
        [0, 1001-L], [Ravg + Rstd] * 2, [Ravg - Rstd] * 2, facecolor="k", alpha=0.4,
        zorder=10
    )
    ax.text(0.045,0.055, "Random initializations",transform=ax.transAxes)

    fig.subplots_adjust(left=0.2, bottom=0.2)
    qca.multipage(plot_fname, clip=True)
    print("plot saved to ", plot_fname)
