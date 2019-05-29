import qca
import h5py
import numpy as np
import measures as ms
from states import make_state
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.patheffects as pe

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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


def network_measures_scatter(
    Skey,
    L=19,
    BC="1-00",
    ICkey=["c1_f0", "R123"],
    V="H",
    axs=None,
    meas_axs=[["scenter-2", "C"], ["scenter-2", "Y"]],
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(1.5, 2.3))
    # ICkey = ["f0", "f0-1", "f0-2", "f0-1-2", "d3", "d4", "d5", "d6", "d7"]
    for col, measures in enumerate(meas_axs):
        d = {meas: {"avg": [], "std": [], "c": []} for meas in measures}
        ax = axs[col]
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
        #   d[measures[0]]["avg"],
        #   d[measures[1]]["avg"],
        #   xerr=d[measures[0]]["std"],
        #   yerr=d[measures[1]]["std"],
        #   fmt = ".k",
        # )

        ax.scatter(
            d[measures[0]]["avg"],
            d[measures[1]]["avg"],
            c=[Skey.index(i) for i in d["scenter-2"]["c"]],
            cmap=cmap,
            linewidths=0.1,
            edgecolor="k",
            alpha=0.8,
            vmin=0,
            vmax=len(Skey) - 1,
        )

        print("enter")
        statestrs = ["R123", "W", "GHZ"]
        markers = ["x", "*", "s", "^"]
        ss = [40, 40, 40, 40]

        for m, s, statestr in zip(markers, ss, statestrs):
            state = make_state(L, statestr)
            dMI = ms.get_MI(state, order=1)
            dsc = ms.get_center_entropy(state, order=2)
            dC = ms.network_clustering(dMI)
            dY = ms.network_disparity(dMI)
            statedata = {"C": dC, "Y": dY, "scenter-2": dsc}
            if m in ("x", "."):
                facecolors = "k"
            else:
                facecolors = "none"
            ax.scatter(
                statedata[measures[0]],
                statedata[measures[1]],
                facecolors=facecolors,
                edgecolors="k",
                marker=m,
                s=s,
            )

        ax.minorticks_off()
        ax.set_xticks([1, 4, 7])
        if meas == "C":
            ax.set_yscale("log")
            ax.set_yticks([1e-5, 1e-3, 1e-1])
            ax.set_yticklabels([])
            ax.set_ylim([5e-6, 1.1])
            ax.set_xticklabels([])
        elif meas == "Y":
            ax.set_yticks([0.05, 0.25, 0.45])
            ax.set_yticklabels([])
            ax.set_xticklabels(["$1$", "$4$", "$7$"])
            ax.set_xlabel(lines[measures[0] + "avg"]["name"], labelpad=-1)


def network_measures_timeseries(
    Skey,
    L=19,
    BC="1-00",
    ICkey=["c1_f0", "R123"],
    Vkey=["H"],
    axs=None,
    measures=["C", "Y"],
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(3, 2.3), sharex=True)
    for col, meas in enumerate(measures):
        ax = axs[col]
        Rstack = []
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
                    d = h5file[meas][3:]
                    if IC[0] == "R":
                        Rstack += [d]
                    else:
                        zorder = None
                        if S == 1:
                            zorder = 10
                        if V in Vs:

                            md = moving_average(d, n=3)
                            line, = ax.plot(
                                3 + np.arange(len(md)),
                                md,
                                c=c,
                                lw=1,
                                ls="-",
                                zorder=zorder,
                            )

                            if ls == "--":
                                line.set_dashes([2, 2, 10, 2])

                    if meas == "C":
                        ax.set_yscale("log")
                        ax.set_yticks([1e-5, 1e-3, 1e-1])
                        ax.set_ylim([5e-6, 1.0])
                    elif meas == "Y":
                        ax.set_yticks([0.05, 0.15, 0.25])
                        # ax.set_ylim([0.05, 0.245])
                    ax.set_ylabel(lines[meas]["name"])
                    ax.set_xlabel("$t$")
                    ax.label_outer()


def cluster_angle_scaling():
    fig, axs = plt.subplots(3, 1, figsize=(3, 7), sharex=True)
    phs = np.linspace(10, 360, 100)
    L = 18
    bases = ["6-3", "3-6", "9-2", "2-9"]
    for base in bases:
        statedata = {"C": [], "Y": [], "sc": []}
        statestrs = [f"C{base}_{ph}" for ph in phs]
        for statestr in statestrs:
            state = make_state(L, statestr)
            dMI = ms.get_MI(state, order=1)
            dsc = ms.get_center_entropy(state, order=2)
            dC = ms.network_clustering(dMI)
            dY = ms.network_disparity(dMI)
            statedata["C"].append(dC)
            statedata["Y"].append(dY)
            statedata["sc"].append(dsc)
        axs[0].plot(phs, statedata["C"])
        axs[1].plot(phs, statedata["Y"])
        axs[2].plot(phs, statedata["sc"], label=base)
    axs[2].legend()
    axs[0].set_ylabel(lines["C"]["name"])
    axs[1].set_ylabel(lines["Y"]["name"])
    axs[2].set_ylabel(lines["scenter-2"]["name"])
    axs[2].set_xlabel("$\phi$ [deg]")
    axs[2].label_outer()


def deltaS_bond_timeseries(Skey):
    for meas in ["scenter-2"]:
        fig, axs = plt.subplots(1, 1, figsize=(3, 1.5), sharey=True, sharex=True)
        BC = "0"
        Vkey = ["H", "HP_45"]
        L = 18
        ICkey = ["f0", "R123"]
        Rstack = []
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


def QI_measures(Lkey, measures):
    statestrs = ["R123", "W", "GHZ"]
    markers = ["x", "d", "*", "^"]
    QI = np.zeros((len(statestrs), len(Lkey), len(measures)))
    for j, statestr in enumerate(statestrs):
        for k, L in enumerate(Lkey):
            state = make_state(L, statestr)
            MI = ms.get_MI(state, order=1)
            sc = ms.get_center_entropy(state, order=2)
            C = ms.network_clustering(MI)
            Y = ms.network_disparity(MI)
            QI[j, k, 0] = sc
            QI[j, k, 1] = C
            QI[j, k, 2] = Y
    for m, measure in enumerate(["scenter", "C", "Y"]):
        fig, ax = plt.subplots(1, 1, figsize=(1.3, 1.0))
        for j, (mk, statestr) in enumerate(zip(markers, statestrs)):
            if mk in ("x", "."):
                facecolors = "k"
            else:
                facecolors = "none"
            for k, L in enumerate(Lkey):
                ax.scatter(
                    Lkey,
                    QI[j, :, m],
                    facecolors=facecolors,
                    edgecolors="k",
                    marker=mk,
                    s=40,
                )
        ax.set_ylabel(lines[measures[1] + "avg"]["name"])
        ax.set_xlabel(lines[measures[0] + "avg"]["name"])
        # ax.set_yscale("log")
        fig.subplots_adjust(left=0.22, bottom=0.22, top=0.9)


# Lscaling of states, long time averages
def measure_Lscaling(
    Skey,
    Lkey=[7, 9, 11, 13, 15, 17, 19],
    L=19,
    BC="1-00",
    ICkey=["c1_f0", "R123"],
    V="H",
    axs=None,
    measures=["C", "Y"],
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(1.3, 1.0))

    lta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    dlta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    for i, S in enumerate(Skey):
        for j, IC in enumerate(ICkey):
            for k, L in enumerate(Lkey):
                sim = select(L, S, IC, V, BC)
                if sim is None:
                    print("No sim!")
                    continue

                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]

                for m, measure in enumerate(measures):
                    if measure == "Dscenter-2":
                        d = h5file["scenter-2"]
                        d = np.abs(np.diff(d))

                    else:
                        d = h5file[measure]
                    d = d[500:]
                    # d = moving_average(d, n=L)
                    lta[i, j, k, m] = np.mean(d)
                    dlta[i, j, k, m] = np.std(d)

    assert IC[0] == "R"
    initrand = np.mean(lta[:, 1, :, :], axis=0)
    dinitrand = np.mean(dlta[:, 1, :, :], axis=0)

    for m, measure in enumerate(measures):
        ax = axs[m]
        for i, S in enumerate(Skey):
            c = cs[i]
            for j, IC in enumerate(ICkey):
                r = initrand[:, m]
                dr = dinitrand[:, m]
                y = lta[i, j, :, m]
                dy = dlta[i, j, :, m]
                x = Lkey
                if j != 1:
                    ax.scatter(x, y, marker="o", s=3, c=c)
                    ax.fill_between(x, y + dy, y - dy, facecolor=c, alpha=0.2)
                    if measure in ("C", "Dscenter-2", "Y"):
                        mm, bb = np.polyfit(x, np.log(y), 1)
                        xs = np.linspace(x[0], x[-1], 100)
                        ys = np.e ** (bb + mm * xs)
                        ax.plot(xs, ys, c=c, label=f"$\lambda = {round(mm, 2)}$")

        ax.scatter(x, r, marker="o", s=3, c="k")
        ax.fill_between(x, r + dr, r - dr, facecolor="k", alpha=0.2)
        if measure in ("C", "Dscenter-2", "Y"):
            mm, bb = np.polyfit(x, np.log(r), 1)
            xs = np.linspace(x[0], x[-1], 100)
            rs = np.e ** (bb + mm * xs)
            ax.plot(xs, rs, c="k", label=f"$\lambda = {round(mm, 2)}$")

            # ax.legend(bbox_to_anchor=(1,1))

        ax.set_ylabel(lines[measure + "avg"]["name"], labelpad=-1)
        if measure == "C":
            ax.set_yscale("log")
            ax.set_yticks([1e-5, 1e-3, 1e-1])
            ax.set_ylim([5e-6, 1.0])
            ax.set_xticks([6, 12, 18])
            ax.set_xticklabels([])
        elif measure == "Y":
            ax.set_yticks([0.05, 0.25, 0.45])
            ax.set_xticks([6, 10, 14, 18])
            ax.set_xlabel("$L$")


if __name__ == "__main__":

    Skey = [13, 14, 6, 1]
    cs = ["limegreen", "darkorange", "crimson", "darkturquoise"]
    cmap = mcolors.ListedColormap(cs)

    lines = {
        "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
        "exp-z": {"name": exp("\sigma^z"), "ls": "-", "c": "C5", "m": "v"},
        "s-2": {"name": " $s$", "ls": "-", "c": "C5", "m": "v"},
        "scenter-2": {
            "name": r"$\Delta S^{(2)}_{L/2}$",
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
        "Dscenter-2": {
            "name": r"$\Delta S^{(2)}_{L/2}$",
            "ls": "-",
            "c": "C5",
            "m": "v",
        },
        "Dscenter-2avg": {
            "name": r"$\overline{\Delta S}^{(2)}_{L/2}$",
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

    plot_fname = "figures/figure3/figure3_fixedBC_source.pdf"

    fig = plt.figure(figsize=(4.75, 2.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    Cax1 = fig.add_subplot(gs[0, :-2])
    Cax2 = fig.add_subplot(gs[0, -2:-1])
    Cax3 = fig.add_subplot(gs[0, -1])

    Yax1 = fig.add_subplot(gs[1, :-2])
    Yax2 = fig.add_subplot(gs[1, -2:-1])
    Yax3 = fig.add_subplot(gs[1, -1])

    # network_measures_timeseries(Skey, axs=[Cax1, Yax1])
    # measure_Lscaling(Skey, axs=[Cax2, Yax2])
    network_measures_scatter(Skey, axs=[Cax3, Yax3])
    # fig.subplots_adjust(left=0.175, bottom=0.16, top=0.98, right=0.97, hspace=0.1)

    # deltaS_bond_timmeseries(Skey)
    # cluster_angle_scaling()
    qca.multipage(plot_fname, clip=False, dpi=300)
    print("plot saved to ", plot_fname)
