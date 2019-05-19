import numpy as np
import matplotlib.pyplot as plt
from figure3 import select, ket
import matplotlib as mpl
from qca import main, multipage
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from states import make_state, bvecs
import measures as ms
from matrix import op_on_state, listkron
mpl.rcParams["text.latex.preamble"] = ["\\usepackage{amsmath}"]
font = {"size": 10, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc(*("font",), **font)
import h5py
from matplotlib.patches import Patch


Rkey = [1, 6, 13, 14]
layout="spring"


def draw_MI(M, ax, layout="spring"):
    M[M<1e-15] = 0.0
    M[M<np.median(M)]=0
    G = nx.from_numpy_matrix(M)

    if layout=="spring":
        pos = nx.spring_layout(G)
    elif layout == "bipartite":
        pos=nx.bipartite_layout(G, nodes=range(len(M)//2))
    elif layout == "circular":
        pos = nx.circular_layout(G)

    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    ws = np.array([w for w in weights])
    mx = max(ws)
    mn = min(ws)
    if mx != mn:
        ws = (ws - mn) / (mx - mn)
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color='k',
        node_size=5,
        alphs=0.5,
        edgelist=edges,
        edge_color="k",
        width=ws)

for t in [100]:
    plt.clf()
    fig, axs = plt.subplots(2, 4, figsize=(4.5, 2.5))
    for col, R in enumerate(Rkey):
        sim = select(L=18, S=R, IC="c1_f0", V="H", BC="0")
        if sim is None:
           print("No sim!")
        L = sim["L"]
        h5file = sim["h5file"]
        Ms = h5file["MI"]
        M = Ms[t]
        draw_MI(M, axs[0, col], layout=layout)
        if col == 0:
            axs[0, col].set_title(r"$R = %d$" % R)
        else:
            axs[0, col].set_title(r"$%d$" % R)

    for col, statestr in enumerate(["C6-3", "GHZ", "W", "R"]):
        state = make_state(L, statestr)
        M = ms.get_MI(state, order=1)
        draw_MI(M, axs[1, col], layout=layout)
        if statestr[0] == "C":
            axs[1, col].set_title(ket("C"))
        else:
            axs[1, col].set_title(ket(statestr))
        C = ms.network_clustering(M)
        Y = ms.network_disparity(M)
        print(statestr, " C:", C, " Y:", Y)

    plot_fname = f"figures/figure3/{layout}_layout_t{t}_medclip.pdf"
    print(plot_fname)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_fname)


