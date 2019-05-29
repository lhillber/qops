import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from figure3 import select, ket, exp
from matrix import ops
from measures import local_entropies_from_rhos, local_exp_vals_from_rhos
from mpl_toolkits.axes_grid1 import ImageGrid

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

if __name__ == "__main__":

    names = {
        "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
        "exp-z": {"name": exp("\hat{\sigma_j}^z"), "ls": "-", "c": "C5", "m": "v"},
        "exp-x": {"name": exp("\hat{\sigma_j}^x"), "ls": "-", "c": "C5", "m": "v"},
        "s-2": {"name": " $s^{(2)}_j$", "ls": "-", "c": "C5", "m": "v"},
    }
    cmaps = ["inferno_r", "inferno"]

    plot_fname = "figures/figure2/figure2_V5.pdf"
    fig = plt.figure(figsize=(4.75, 3.7))
    Skey = ["3.6", "3.13", "3.14", "5.4", "5.2"]
    measures = ["exp-z", "s-2"]
    IC = "c1_f0"
    L = 18
    T = (L - 1) * 3 + 1  # plot ylim
    letts1 = [
        r"$\mathrm{A}$",
        r"$\mathrm{C}$",
        r"$\mathrm{E}$",
        r"$\mathrm{G}$",
        r"$\mathrm{I}$",
    ]
    letts2 = [
        r"$\mathrm{B}$",
        r"$\mathrm{D}$",
        r"$\mathrm{F}$",
        r"$\mathrm{H}$",
        r"$\mathrm{J}$",
    ]
    clett1 = ["w", "w", "w", "w", "w"]
    clett2 = ["k", "k", "k", "w", "k"]
    letts = [letts1, letts2]
    cletts = [clett1, clett2]
    for row, (meas, letti, cli) in enumerate(zip(measures, letts, cletts)):
        grid = ImageGrid(
            fig,
            int("21" + str(1 + row)),
            nrows_ncols=(1, 5),
            direction="row",
            axes_pad=0.1,
            add_all=True,
            cbar_mode="single",
            cbar_location="right",
            cbar_size="20%",
            cbar_pad=0.05,
        )
        for col, (S, lett, cl) in enumerate(zip(Skey, letti, cli)):
            N, S = map(int, S.split("."))
            ax = grid[col]
            if N == 3:
                sim = select(L=L, S=S, IC=IC, V="H", BC="0")
                if sim is None:
                    print("No sim!")
                    continue
                S = sim["S"]
                L = sim["L"]
                IC = sim["IC"]
                h5file = sim["h5file"]
                if meas[0] == "e":
                    ticks = [-1, 1]
                    ticklabels = ["↑", "↓"]
                else:
                    ticks = [0, 1]
                    ticklabels = ["$0$","$1$"]
                vmin, vmax = ticks
                d = h5file[meas]

            elif N == 5:
                der = "/home/lhillber/documents/research/cellular_automata/qeca/qops"
                der = os.path.join(der, f"qca_output/hamiltonian/rule{S}/rho_i.npy")
                one_site = np.load(der)
                one_site = one_site.reshape(2000, 22, 2, 2)
                one_site = one_site[::, 2:-2, :, :]
                T5, L5, *_ = one_site.shape
                d = np.zeros((T5, L5))
                ti = 0
                for t, rhoi in enumerate(one_site):
                    if t % 10 == 0:
                        if meas == "exp-z":
                            d[ti, :] = local_exp_vals_from_rhos(rhoi, ops["Z"])
                        elif meas == "s-2":
                            d[ti, :] = local_entropies_from_rhos(rhoi, order=2)
                        ti += 1

            I = ax.imshow(
                d[0:T],
                origin="lower",
                interpolation=None,
                cmap=cmaps[row],
                vmin=vmin,
                vmax=vmax,
            )
            ax.cax.colorbar(I)
            ax.cax.set_yticks(ticks)
            ax.cax.set_yticklabels(ticklabels)

            ax.set_xticks([0, 8, 17])
            ax.set_yticks([i * (L - 1) for i in range(4)])

            ax.set_yticklabels([])
            ax.set_xticklabels([])

            ax.text(0.5, 46, lett, color=cl, family="sans-serif", weight="bold")

            if col == len(Skey) - 1:
                ax.cax.text(
                    1.6,
                    0.5,
                    names[meas]["name"],
                    rotation=0,
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                )

            if row == 0 and col < 3:
                ax.set_title(r"$T_{%d}$" % S)
            elif row == 0 and col > 2:
                ax.set_title(r"${F_{%d}}$" % S)

            ax.tick_params(direction="out")

    grid[0].set_yticklabels(["$"+str(i * (L - 1))+"$" for i in range(4)])
    grid[0].set_xticklabels(["$0$", "$8$", "$17$"])
    grid[0].set_xlabel("$j$", labelpad=0)
    grid[0].set_ylabel("$t$", labelpad=0)
    fig.subplots_adjust(hspace=0.1, left=0.05, top=0.93)
    plt.savefig(plot_fname, dpi=300)
    print("plot saved to ", plot_fname)
