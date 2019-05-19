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
import matplotlib.patheffects as pe

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


def select(L, S, IC, V, BC, T=None):

    maxoverT = False
    if T is None:
        T = "*"
        maxoverT = True

    name = "L{}_T{}_V{}_r1_S{}_M2_IC{}_BC{}.hdf5".format(L, T, V, S, IC, BC)
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
    if maxoverT:
        sim = sims[np.argmax(np.array([s["T"] for s in sims]))]
    else:
        sim = sims[0]

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
        cdict[key] = [(indices[i], cs[i - 1, ki], cs[i, ki]) for i in range(N + 1)]
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def colorbar(label, ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("right", size="7%", pad="8%")
    colorbar = plt.colorbar(mappable, cax=cax)
    colorbar.set_label(label, rotation=0, y=0.55, labelpad=1.8)
    return cax, colorbar


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def network_measures_scatter(Skey):
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.8))
    BC = "0"
    meas_axs = [["scenter-2", "C"], ["Y", "C"]]
    L = 18
    ICkey = ["f0", "f0-1", "f0-2", "f0-1-2", "d3", "d4", "d5", "d6", "d7"]
    V = "H"
    for col, measures in enumerate(meas_axs):
        d = {meas: {"avg": [], "std": [], "c": []} for meas in measures}
        ax = axs[col]
        ax.set_yscale("log")
        # ax.set_xscale("log")
        for IC in ICkey:
            for S in Skey:
                sim = select(L, S, IC, V, BC)
                if sim is None:
                    print("No sim!")
                    continue
                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]
                for meas in measures:

                    d[meas]["c"] += [S]
                    if meas in ("scenter", "scenter-2"):
                        try:
                            d[meas]["avg"] += [np.mean(h5file[meas][500:])]
                            d[meas]["std"] += [np.std(h5file[meas][500:])]
                        except:
                            d[meas]["avg"] += [
                                np.mean(h5file["sbond"][500:, int(L / 2)])
                            ]
                            d[meas]["std"] += [
                                np.std(h5file["sbond"][500:, int(L / 2)])
                            ]
                    else:
                        d[meas]["avg"] += [np.mean(h5file[meas][500:])]
                        d[meas]["std"] += [np.std(h5file[meas][500:])]
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
        statestrs = ["R123", "W", "GHZ"]
        ss = [40, 40, 40]
        Cs = [f"C6-3_{ph}" for ph in np.linspace(30, 180-30, 100)]
        ss += [3]*len(Cs)
        statestrs += Cs
        markers = ["x", "d", "*"]
        markers += ["."]*len(Cs)
        for m, s, statestr in zip(markers, ss, statestrs):
            state = make_state(L, statestr)
            dMI = ms.get_MI(state, order=1)
            dsc = ms.get_center_entropy(state, order=2)

            dC = ms.network_clustering(dMI)
            dY = ms.network_disparity(dMI)
            Rdata = {"C": dC, "Y": dY, "scenter-2": dsc}
            if m in ("x", "."):
                facecolors = "k"
            else:
                facecolors = "none"
            if m == "*":
                print(Rdata[measures[0]])
                print(Rdata[measures[1]])
            ax.scatter(
                Rdata[measures[0]],
                Rdata[measures[1]],
                facecolors=facecolors,
                edgecolors="k",
                marker=m,
                s=s,
            )
        ax.set_yticks([1e-5, 1e-3, 1e-1])
        if col==1:
            ax.set_xticks([0.1, 0.4])
        else:
            ax.set_xticks([1.0, 8.0])
        ax.set_ylabel(lines[measures[1] + "avg"]["name"])
        ax.set_xlabel(lines[measures[0] + "avg"]["name"], labelpad=-5)

    NS = len(Skey)
    cax, cb = colorbar(r"$R$", ncolors=NS, cmap=cmap)
    #cb.solids.set_rasterized(False)

    l, u = cax.get_ylim()
    tick_locs = (np.arange(NS)+0.5) * (u - l) / NS
    # tick_locs = [(2 * j + 1) / 8.0 for j in range(Skey)]
    cb.set_ticks(tick_locs)
    cb.set_ticklabels(Skey)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    ax.label_outer()


def plot_deltaS_bond(Skey):
    for meas in ["scenter-2"]:
        fig, axs = plt.subplots(1, 1, figsize=(3, 1.5), sharey=True, sharex=True)
        BC = "0"
        Vkey = ["H", "HP_45"]
        L = 18
        ICkey = ["f0", "R123"]
        Rstack = []
        count = 0
        lss = ["-", "--"]
        for V in Vkey:
            for IC in ICkey:
                for S in Skey:
                    if S == 14:
                        ax = axs
                        Vs = Vkey
                        c = cs[Skey.index(S)]
                        ls = lss[Vs.index(V)]
                    else:
                        ax = axs
                        Vs = ["H"]
                        # c = cmap(Skey.index(S))
                        c = cs[Skey.index(S)]
                        # if S == 1:
                        #    c = shade_color(c, 1.3)
                        ls = "-"
                    sim = select(L, S, IC, V, BC)
                    if sim is None:
                        print("No sim!")
                        continue
                    S = sim["S"]
                    L = sim["L"]
                    IC = sim["IC"]
                    h5file = sim["h5file"]

                    d = h5file[meas][10:]
                    if meas == "scenter-2":
                        d = np.abs(np.diff(d))
                    d = moving_average(d, n=L)
                    Rstate = make_state(L, "R123")
                    Rsc = ms.get_center_entropy(Rstate, order=2)
                    if IC[0] == "R":
                        Rstack += [d]
                        count += 1
                    else:
                        if V in Vs:
                            line, = ax.plot(
                                d,
                                c=c,
                                lw=1,
                                ls="-",
                                path_effects=[
                                    pe.Stroke(linewidth=0.8, foreground="k"),
                                    pe.Normal(),
                                ],
                            )
                    if ls == "--":
                        line.set_dashes([2, 2, 15, 2])
                    ax.set_yscale("log")
                    ax.minorticks_off()
                    ax.set_ylabel(lines[meas]["name"])
                    ax.set_xlabel("$t$")
                    ax.label_outer()
        Rstack = np.array(Rstack)
        Ravg = np.mean(Rstack)
        Rstd = np.std(Rstack)
        if meas == "scenter-2":
            ax.fill_between(
                [0, 1001 - L],
                [Ravg + Rstd],
                [Ravg - Rstd],
                facecolor="k",
                alpha=0.3,
                zorder=10,
            )
            ax.text(0.045, 2 * 0.055, "Random initializations", transform=ax.transAxes)

        fig.subplots_adjust(left=0.2, bottom=0.2)


if __name__ == "__main__":

    # cs = ["olivedrab", "indigo", "crimson", "gold"]
    cs = ["darkturquoise", "darkorange", "chartreuse", "crimson"]
    Skey = [13, 14, 1, 6]
    cmap = mcolors.ListedColormap(cs)

    lines = {
        "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
        "exp-z": {"name": exp("\sigma^z"), "ls": "-", "c": "C5", "m": "v"},
        "s-2": {"name": " $s$", "ls": "-", "c": "C5", "m": "v"},
        "scenter": {"name": " $\Delta S^{(2)}_{L/2}$", "ls": "-", "c": "C5", "m": "v"},
        "scenter-2": {
            "name": " $\Delta S^{(2)}_{L/2}$",
            "ls": "-",
            "c": "C5",
            "m": "v",
        },
        "scenter-2avg": {
            "name": " $\overline{S}^{(2)}_{L/2}$",
            "ls": "-",
            "c": "C5",
            "m": "v",
        },
        "Cavg": {"name": " $\overline{\mathcal{C}}$", "ls": "-", "c": "C5", "m": "v"},
        "Davg": {"name": " $\overline{\mathcal{D}}$", "ls": "-", "c": "C5", "m": "v"},
        "Yavg": {"name": " $\overline{\mathcal{Y}}$", "ls": "-", "c": "C5", "m": "v"},
        "C": {"name": " $\mathcal{C}$", "ls": "-", "c": "C5", "m": "v"},
        "D": {"name": " $\mathcal{D}$", "ls": "-", "c": "C5", "m": "v"},
        "Y": {"name": " $\mathcal{Y}$", "ls": "-", "c": "C5", "m": "v"},
    }

    plot_fname = "figures/figure3/figure3_L18_V3.pdf"

    network_measures_scatter(Skey)
    plot_deltaS_bond(Skey)

    fig, axs = plt.subplots(2, 1, figsize=(3, 2.3), sharex=True)
    for col, meas in enumerate(["C", "Y"]):
        ax = axs[col]
        BC = "0"
        Vkey = ["H", "HP_45"]
        L = 18
        ICkey = ["f0", "R123"]
        Rstack = []
        count = 0
        lss = ["-", "--", ":"]
        for V in Vkey:
            for IC in ICkey:
                for S in Skey:
                    if S == 14:
                        Vs = Vkey
                        c = cs[Skey.index(S)]
                        ls = lss[Vs.index(V)]
                    else:
                        Vs = ["H"]
                        c = cs[Skey.index(S)]
                        ls = "-"
                    sim = select(L, S, IC, V, BC)
                    if sim is None:
                        print("No sim!")
                        continue
                    S = sim["S"]
                    L = sim["L"]
                    IC = sim["IC"]
                    h5file = sim["h5file"]
                    d = h5file[meas][10:]
                    d = moving_average(d, n=L)
                    Rstate = make_state(L, "R123")
                    Rsc = ms.get_center_entropy(Rstate, order=2)
                    if IC[0] == "R":
                        Rstack += [d]
                        count += 1
                    else:
                        zorder = None
                        if S == 1:
                            zorder = 10
                        if V in Vs:
                            line, = ax.plot(
                                d,
                                c=c,
                                lw=1,
                                ls="-",
                                zorder=zorder,
                                path_effects=[
                                    pe.Stroke(linewidth=0.8, foreground="k"),
                                    pe.Normal(),
                                ],
                            )

                            if ls == "--":
                                line.set_dashes([2, 2, 10, 2])

                    if meas == "C":
                        ax.set_yscale("log")
                        ax.set_yticks([1e-5, 1e-3, 1e-1])
                    elif meas == "Y":
                        ax.set_yticks([0.07, 0.14, 0.21])
                    ax.set_ylabel(lines[meas]["name"])
                    ax.set_xlabel("$t$")
                    ax.label_outer()
        fig.subplots_adjust(left=0.2, bottom=0.2, hspace=0.1)
    qca.multipage(plot_fname, clip=True, dpi=10 * fig.dpi)
    print("plot saved to ", plot_fname)
