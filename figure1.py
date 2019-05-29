import numpy as np
from qca import main
import matplotlib.pyplot as plt
from figure3 import select
import matplotlib as mpl
from classical import iterate, lut
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

from matplotlib import rc
rc("text", usetex=True)
font = {"size": 11, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}",
    r"\usepackage{sansmath}",  # sanserif math
    r"\sansmath",
]


def ket(x):
    return "$\\vert " + x + "\\rangle$"


def exp(x):
    return "$\\langle " + x + "\\rangle$"


names = {
    "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
    "exp-z": {"name": exp("\hat{\sigma}_j^z"), "ls": "-", "c": "C5", "m": "v"},
    "s-2": {"name": " $s^{(2)}_j$", "ls": "-", "c": "C5", "m": "v"},
}

defaults = {
    "Ls": [],
    "Ts": [],
    "Vs": [],
    "rs": [],
    "Ss": [],
    "Ms": [],
    "ICs": [],
    "BCs": [],
    "tasks": ["s-2", "exp-z"],
    "sub_dir": "default",
    "thread_as": "zip",
}

L = 18
T = 100
r = 1
# Rkey = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Rkey = [1, 6, 13, 14]
V = "H"
M = 2
BC = "0"
IC = "c1_f0"

for R in Rkey:
    sim = select(L=L, S=R, IC=IC, V=V, BC=BC)
    defaults["Ls"].append(L)
    defaults["Ts"].append(T)
    defaults["Vs"].append(V)
    defaults["rs"].append(r)
    defaults["Ss"].append(R)
    defaults["Ms"].append(M)
    defaults["ICs"].append(IC)
    defaults["BCs"].append(BC)

# main(defaults=defaults)

for R in Rkey:
    plt.close("all")
    # classical simulation
    Rc = lut[R]
    ICc = np.zeros(L, dtype=np.int32)
    ICc[L // 2-1] = 1
    C = iterate(L, T, Rc, ICc)

    sim = select(L=L, S=R, IC=IC, V=V, BC=BC)
    S = sim["S"]
    L = sim["L"]
    IC = sim["IC"]
    d = sim["h5file"]
    z = d["exp-z"]
    s = d["s-2"]

    T = (L-1) * 3 + 1  # plot ylim

    # spin
    w, h = mpl.figure.figaspect(z[0:T])
    fig, axs = plt.subplots(1, 3, figsize=(2.25, 2), sharey=True)

    ax = axs[1]
    im1 = ax.imshow(
        z[0:T],
        interpolation=None,
        origin="lower",
        cmap="inferno_r",
        vmin=-1,
        vmax=1,
    )
    ticks = [-1, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    #cax.text(
    #    1.3,
    #    0.5,
    #    names["exp-z"]["name"],
    #    rotation=0,
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="center",
    #)
    cax.axis("off")
    #cbar = fig.colorbar(im1, cax=cax, ticks=ticks)
    #cbar.set_ticks(ticks)
    #
    #ax.set_xlabel("$j$", labelpad=0)
    ax.set_xticks([0, 8, 17])
    ax.set_xticklabels([])

    # ax.set_yticks([i*(L-1) for i in range(4)])
    ax.set_yticklabels([])
    # ax.set_ylabel("$t$", labelpad=0)
    # ax.set_title("$R = %d $" % R)
    ax.text(
        1.1, 1.15, "$T_{%d} $" % R, transform=ax.transAxes, ha="center", va="center"
    )

    # entropy
    ax = axs[2]
    im2 = ax.imshow(
        s[0:T],
        interpolation=None,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ticks = [0, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    #cax.text(
    #    1.3,
    #    0.5,
    #    names["s-2"]["name"],
    #    rotation=0,
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="center",
    #)

    cbar = fig.colorbar(im2, cax=cax, ticks=ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["↑" + ", " + str(ticks[0]), "↓"+", " + str(ticks[1])])

    ax.set_yticklabels([])
    ax.set_xticks([0, 8, 17])
    ax.set_xticklabels([])
    #ax.set_xlabel("$j$", labelpad=0)
    # ax.set_ylabel("$t$", labelpad=0)
    # ax.set_title("$R = %d $" % R)

    # classical
    ax = axs[0]
    im3 = ax.imshow(
        C[0:T],
        interpolation=None,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    cax.axis("off")

    legend_elements = [
        Patch(facecolor="k", edgecolor="k", label="1"),
        Patch(facecolor="w", edgecolor="k", label="0"),
    ]
    ylim = ax.get_ylim()
    ax.plot(
        [0], [1000], linestyle="none", marker="s", markeredgecolor="k", c="k", label="1"
    )
    ax.plot(
        [0], [1000], linestyle="none", marker="s", markeredgecolor="k", c="w", label="0"
    )
    #ax.legend(
    #    # handles=legend_elements,
    #    bbox_to_anchor=(2.2, 0.76),
    #    handlelength=0.7,
    #    markerscale=1.2,
    #    frameon=False,
    #)
    ax.set_ylim(ylim)
    ax.set_xticks([0, 8, 17])

    ax.set_yticks([i * (L - 1) for i in range(4)])

    ax.set_yticklabels(["$"+str(i * (L - 1))+"$" for i in range(4)])

    ax.set_xlabel("$j$", labelpad=0)
    ax.set_ylabel("$t$", labelpad=0)
    ax.text(
        0.5,
        1.15,
        "$C_{%d} $" % Rc,
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    plt.subplots_adjust(left=0.155, right=0.86, wspace=0.06)
    #zoom=0.5
    #w, h = fig.get_size_inches()
    #fig.set_size_inches(w * zoom, h * zoom)

    axs[0].text(0.5, 46, "$\mathrm{A}$", color="w", weight="bold")
    axs[1].text(0.5, 46, "$\mathrm{B}$", color="w", weight="bold")
    axs[2].text(0.5, 46, "$\mathrm{C}$", color="w", weight="bold")
    plt.savefig("figures/figure1/R{}_viz.pdf".format(R), dpi=300)
