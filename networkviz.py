import numpy as np
import matplotlib.pyplot as plt
from figure3 import select, ket
import matplotlib as mpl
from qca import multipage
import networkx as nx
from states import make_state
import measures as ms

font = {"size": 10, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}"
    r"\usepackage{helvet}",  # set the normal font here
    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!I
]
from matplotlib import rc
rc("text", usetex=True)








Rkey = [1, 6, 13, 14]
layout = "spring"


def draw_MI(M, ax, layout="spring"):
    M[M < 1e-15] = 0.0
    M[M < np.median(M)] = 0
    G = nx.from_numpy_matrix(M)

    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "bipartite":
        pos = nx.bipartite_layout(G, nodes=range(len(M) // 2))
    elif layout == "circular":
        pos = nx.circular_layout(G)

    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    ws = np.array([w for w in weights])
    mx = max(ws)
    mn = min(ws)
    if mx != mn:
        ws = (ws - mn) / (mx - mn)
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color="k",
        node_size=6,
        alphs=0.5,
        edgelist=edges,
        edge_color="k",
        width=ws,
    )

for t in [10, 50, 100, 500, 1000]:
    fig, axs = plt.subplots(1, 8, figsize=(4.75, 0.8))
    for col, R in enumerate(Rkey):
        sim = select(L=18, S=R, IC="c1_f0", V="H", BC="0")
        if sim is None:
            print("No sim!")
        L = sim["L"]
        h5file = sim["h5file"]
        Ms = h5file["MI"]
        M = Ms[t]
        draw_MI(M, axs[col], layout=layout)
        axs[col].set_title(r"$T_{%d}$" % R)

    for col, statestr in enumerate(["C6-3", "GHZ", "W", "R"]):
        state = make_state(L, statestr)
        M = ms.get_MI(state, order=1)
        draw_MI(M, axs[4 + col], layout=layout)
        if statestr[0] == "C":
            axs[4 + col].set_title(ket("C"))
        else:
            axs[4 + col].set_title(ket(statestr))
        C = ms.network_clustering(M)
        Y = ms.network_disparity(M)
        print(statestr, " C:", C, " Y:", Y)
    fig.subplots_adjust(wspace=0.4, top=0.7, left=0.05, right=1, bottom=0)

plot_fname = f"figures/figure3/{layout}_layout_medclip.pdf"
print(plot_fname)
multipage(plot_fname, clip=False)
