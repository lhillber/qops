import os
import glob
import h5py
import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ket(x):
    return "$\\vert " + x + "\\rangle$"


def exp(x):
    return "$\\langle " + x + "\\rangle$"


names = {
    "c1_f0": ket("010"),
    "c3_f1": ket("010"),
    "exp-z": exp("\sigma^{(z)}_j"),
    "s-2": r"$s_j$",
    "scenter-2avg": r" $\overline{s}^{(2)}_{:L/2}$",
    "Dscenter-2": r"$\Delta s^{(2)}_{:L/2}$",
    "Dscenter-2avg": r"$\overline{\Delta s}^{(2)}_{:L/2}$",
    "Cavg": r"$\overline{\mathcal{C}}$",
    "Davg": r"$\overline{\mathcal{D}}$",
    "Yavg": r"$\overline{\mathcal{Y}}$",
    "C": "$\mathcal{C}$",
    "D": "$\mathcal{D}$",
    "Y": "$\mathcal{Y}$",
}

colors = {6: "crimson", 1: "darkturquoise", 14: "darkorange", 13: "olivedrab", "R": "k"}
markers = {"G": "s", "W": "*", "R": "x", "C": "d"}
lines = {"H": "", "HP_45": "--"}


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


def select(T, L, S, IC, V, BC):
    maxoverT = False
    if T is None:
        T = "*"
        maxoverT = True

    name = f"L{L}_T{T}_V{V}_r1_S{S}_M2_IC{IC}_BC{BC}.hdf5"
    data_dir_glob = "/home/lhillber/documents/research/cellular_automata/qeca/qops/qca_output/master/data/{}".format(
        name
    )
    # print(name)
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
        print("No sim:", name)
    if maxoverT:
        sim = sims[np.argmax(np.array([s["T"] for s in sims]))]
    else:
        sim = sims[0]

    namefound = "L{}_T{}_V{}_r1_S{}_M2_IC{}_BC{}.hdf5".format(
        *[sim[k] for k in ["L", "T", "V", "S", "IC", "BC"]]
    )
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


def brody_fit(x, n):
    def brody_func(x, eta):
        b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
        return b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))

    popt, pcov = curve_fit(brody_func, x, n, p0=[0.1], bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,))/n, mode='valid')

coeffs = [
    [1.0 / 2],
    [2.0 / 3, -1.0 / 12],
    [3.0 / 4, -3.0 / 20, 1.0 / 60],
    [4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
]


def firstdiff(d, acc, dx):
    assert acc in [2, 4, 6, 8]
    dd = np.sum(
        np.array(
            [
                (
                    coeffs[acc // 2 - 1][k - 1] * d[k * 2 :]
                    - coeffs[acc // 2 - 1][k - 1] * d[: -k * 2]
                )[acc // 2 - k : len(d) - (acc // 2 + k)]
                / dx
                for k in range(1, acc // 2 + 1)
            ]
        ),
        axis=0,
    )
    return dd
