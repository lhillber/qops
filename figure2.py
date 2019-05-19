import qca
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import glob
import os
from figure3 import select
import matplotlib.colors
from matrix import ops
from measures import local_entropies_from_rhos, local_exp_vals_from_rhos
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid


# For a given measure, plot a set of rules

mpl.rcParams["text.latex.preamble"] = ["\\usepackage{amsmath}"]
font = {"size": 9, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc(*("font",), **font)

def ket(x):
    return "$\\vert " + x + "\\rangle$"

def exp(x):
    return "$\\langle " + x + "\\rangle$"


if __name__ == "__main__":

    names = {
        "c1_f0": {"name": ket("010"), "ls": "-", "c": "C5", "m": "v"},
        "exp-z": {"name": exp("\hat{\sigma_j}^z"), "ls": "-", "c": "C5", "m": "v"},
        "exp-x": {"name": exp("\hat{\sigma_j}^x"), "ls": "-", "c": "C5", "m": "v"},
        "s-2": {"name": " $s^{(2)}_j$", "ls": "-", "c": "C5", "m": "v"},
    }
    cmaps = ["inferno_r", "inferno"]

    plot_fname = "figures/figure2/figure2_V5.pdf"
    fig = plt.figure(figsize=(2.25, 3.7))
    Skey = ["3.6", "3.13", "3.14", "5.4"]
    measures = ['exp-z', 's-2']
    L = 18
    T = (L-1) * 3 + 1  # plot ylim
    ICkey = ["c1_f0"]
    for IC in ICkey:
        for row, meas in enumerate(measures):
            grid = ImageGrid(fig, int("31"+str(1+row)),
                                  nrows_ncols=(1, 4),
                                  direction="row",
                                  axes_pad=0.1,
                                  add_all=True,
                                  label_mode="L",
                                  #share_all=True,
                                  cbar_mode="single",
                                  cbar_location="right",
                                  cbar_size="20%",
                                  cbar_pad=0.05,
                                  )
            for col, S in enumerate(Skey):
                N, S = map(int, S.split("."))
                ax = grid[col]
                if N == 3:
                    sim = select(L=L, S=S, IC=IC, V="H", BC="0")
                    if sim is None:
                        print('No sim!')
                        continue
                    S = sim["S"]
                    L = sim["L"]
                    IC = sim["IC"]
                    h5file = sim["h5file"]
                    if meas[0] == 'e':
                        ticks = [-1, 1]
                    else:
                        ticks = [0, 1]
                    vmin, vmax = ticks
                    d = h5file[meas]

                elif N == 5:
                    der = "/home/lhillber/documents/research/cellular_automata/qeca/qops"
                    der = os.path.join(der, f"qca_output/hamiltonian/rule{S}/rho_i.npy")
                    one_site = np.load(der)
                    one_site = one_site.reshape(2000, 21, 2, 2)
                    print(one_site.shape)
                    T5, L5, *_ = one_site.shape
                    d = np.zeros((T5, L5))
                    ti=0
                    for t, rhoi in enumerate(one_site):
                        if t % 10 == 0:
                            if meas == "exp-z":
                                d[ti,:] = local_exp_vals_from_rhos(rhoi, ops["Z"])
                            elif meas == "s-2":
                                d[ti,:] = local_entropies_from_rhos(rhoi, order=2)
                            ti+=1

                I = ax.imshow(
                    d[0:T],
                    origin="lower",
                    interpolation=None,
                    cmap=cmaps[row],
                    vmin=vmin,
                    vmax=vmax
                )
                ax.cax.colorbar(I)
                #ax.cax.toggle_label(True)

                ax.cax.set_yticks(ticks)

                if N == 3:
                    ax.set_xticks([0,8,17])
                elif N== 5:
                    ax.set_xticks([0,10,20])

                ax.set_yticks([i * (L - 1) for i in range(4)])
                ax.set_yticklabels([i * (L - 1) for i in range(4)])

                if row != len(measures) - 1:
                    ax.set_xticklabels([])

                if col == len(Skey)-1:
                    ax.cax.text(1.6, 0.5, names[meas]['name'], rotation=0, transform=ax.transAxes,
                                ha='left', va='center')

                if col == 0 and row == 0:
                    ax.set_title(r"$R = {}$".format(S), fontsize=9)
                elif col > 0 and row == 0:
                    ax.set_title(r"${}$".format(S), fontsize=9)

                #ax.set_xlabel("$j$", labelpad=0)
                ax.set_ylabel("$t$", labelpad=0)
                ax.tick_params(direction='inout')
    print(len(fig.get_axes()))
    fig.subplots_adjust(left=0.15,right=0.81, hspace=0.15)
    plt.savefig(plot_fname, bbox_inches="tight", dpi=10*fig.dpi)
    print("plot saved to ", plot_fname)
