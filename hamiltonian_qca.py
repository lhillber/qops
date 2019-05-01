import matrix as mx
from matrix import ops
import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import gamma
from scipy.optimize import curve_fit
from copy import copy
import measures as ms
from states import make_state
from qca import multipage
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# plotting defaults
import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
font = {"size": 12, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc("font", **font)

def brody_fit(x, n):
    def brody_func(x, eta):
        b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
        return b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))

    popt, pcov = curve_fit(brody_func, x, n, bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov

def make_H(L, R, V):
    H = np.zeros((2 ** L, 2 ** L), dtype=np.complex128)
    if type(V) is not str:
        ops['V'] = V
        V = 'V'
    bulks = []
    lbs = []
    rbs = []
    for s, b in enumerate(mx.dec_to_bin(R, 4)[::-1]):
        if b:
            r = mx.dec_to_bin(s, 2)
            if r[1] == 0:
                rbs += [str(r[0]) + V ]
            if r[0] == 0:
                lbs += [V + str(r[1])]
            bulks += [str(r[0])+V+str(r[1])]

    #print('rule', R, mx.dec_to_bin(R, 4), lbs, bulks, rbs)
    for i in range(1, L - 1):
        l = i - 1
        r = L - 3 - l
        left = np.eye(2 ** l)
        right = np.eye(2 ** r)
        for bulk in bulks:
            bulk = mx.listkron([ops[o] for o in bulk])
            H += mx.listkron([left, bulk, right])

    # boundaries
    end = np.eye(2**(L-2))
    for rb in rbs:
        rb = mx.listkron([ops[o] for o in rb])
        #H += mx.listkron([end, rb])

    for lb in lbs:
        lb = mx.listkron([ops[o] for o in lb])
        #H += mx.listkron([lb, end])

    assert mx.isherm(H)
    return H

# plot U and H
def UHviz(U, H, R):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
    Hviz = copy(H)
    Uviz = copy(U)
    #Hviz[np.abs(Hviz) <= 1e-10] = np.nan
    #Uviz[np.abs(Uviz) <= 1e-10] = np.nan
    #Hviz[Hviz != 0.0] = 1.0
    #Uviz[Uviz != 0.0] = 1.0

    axs[0].imshow(np.abs(Hviz), cmap='gray_r')
    axs[0].set_title(r"$H$")
    axs[1].imshow(np.abs(Uviz), cmap='gray_r')
    axs[1].set_title(r"$U = e ^{-i H}$")
    fig.suptitle(r"$R = {}, L = {}$".format(R, L))
    fig.subplots_adjust(top=0.8)

def spectrum_viz(eigs):
    s = eigs[1:] - eigs[:-1]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    axs[0].plot(eigs, '.k')

    idx = np.arange(len(eigs))
    cs = ["C0"]
    for n, c in enumerate(cs):
        f  = 1.0 / 2.0**(n+4)
        r = np.ptp(eigs)
        b = r*f
        mask = np.logical_and(eigs < b, eigs > -b)
        e = eigs[mask]
        s = e[1:] - e[:-1]
        s /= np.mean(s)
        # s = s/np.std(s)
        s = s[s<r/10]
        axs[0].plot(idx[mask], e)
        n, bin, _ = axs[1].hist(
            s, density=True, alpha=1, histtype="step", bins = 15, log=False, color=c
        )
        x = (bin[1:] + bin[:-1]) / 2.0
        xs = np.linspace(x[0], x[-1], 300)
        xs = xs[xs > 0]
        func, popt, pcov = brody_fit(x, n)
        #print("eta:", popt[0])
        axs[1].text(0.76, 0.93,r'$\eta = %.2f \pm %.2f$' % (popt[0], np.sqrt(np.diag(pcov)[0])),
             horizontalalignment='center',
             verticalalignment='center',
             transform = axs[1].transAxes)
        axs[1].plot(xs, func(xs), color=c)
        #axs[2].plot([len(s)], [popt[0]], 'x', color=c)
        fig.suptitle(r"$R = {}, L = {}$".format(R, L))

    axs[0].set_xlabel(r"$i$")
    axs[0].set_ylabel(r"$E_i$")
    axs[1].set_xlabel(r"$s/\bar{s}$")
    axs[1].set_ylabel(r"$P(s/\bar{s})$")
    #axs[2].set_xlabel('sample size')
    #axs[2].set_ylabel('Brody parameter')
    fig.subplots_adjust(wspace=0.4)

def time_evolve(U):
    psi = make_state(L, "c1_f0")
    ts = np.arange(0, 60, dt)
    zs = np.zeros((len(ts), L))
    ss = np.zeros((len(ts), L))
    fig, axs = plt.subplots(1, 2, figsize=(6, 8))
    for ti, t in enumerate(ts):
        rhos = ms.get_local_rhos(psi)
        zs[ti, :] = ms.local_exp_vals_from_rhos(rhos, ops["Z"])
        ss[ti, :] = ms.local_entropies_from_rhos(rhos, order=2)
        psi = U.dot(psi)

    axs[0].imshow(zs, origin="lower", interpolation=None)
    axs[0].set_title(r"$\langle \sigma_z \rangle$")
    axs[0].set_ylabel("$t$")
    axs[0].set_xlabel("$j$")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].imshow(ss, origin="lower", interpolation=None)
    axs[1].set_yticks([])
    axs[1].set_title(r"$s_2$")
    axs[1].set_xlabel("$j$")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.subplots_adjust(top=0.8)
    fig.tight_layout()

def deg_subspaces(eigs, vecs):
    deg = {}
    for i, (eig, vec) in enumerate(zip(eigs, vecs.T)):
        try:
            deg[eig] += [vec]
        except KeyError:
            deg[eig] = [vec]
    return deg

if __name__ == '__main__':

    dt = 1.0
    Rs = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    Ls = (9,)
    V = 'H'
    figaxs = {}
    for L in Ls:
        for R in Rs:
            make=False
            H = make_H(L, R, V)
            for R2 in Rs:
                for B in ('1', '0'):
                    V2 = mx.make_U2(B)
                    A = make_H(L, R2, V2)
                    if np.allclose(mx.commute(H, A), np.zeros_like(H, dtype=np.complex128)):
                        make = True
                        print(R, (R2, B))
                        fig, axs = plt.subplots(1,3, figsize=(6, 2.6))
                        eigsH, vecsH = eigh(H)
                        eigsA, vecsA = eigh(A)
                        print(eigsA, vecsA)
                        degA = deg_subspaces(eigsA, vecsA)
                        eigsH2 = []
                        for eig, vecs in degA.items():
                            if eig != 0 :
                                for i, vecH in enumerate(vecsH.T):
                                    rep = np.zeros_like(vecH)
                                    for j, vec in enumerate(vecs):
                                        rep[j] = np.conjugate(vec).dot(vecH)
                                    if np.all(rep == vecH):
                                        eigsH2 += [eigsH[i]]

                        print(degA)

                        axs[0].imshow(np.abs(H), cmap="gray_r")
                        axs[1].imshow(np.abs(A), cmap="gray_r")
                        axs[0].grid(True)
                        axs[1].grid(True)
                        axs[0].set_xticks([0, 5, 10, 15])
                        axs[1].set_xticks([0, 5, 10, 15])
                        axs[2].plot(eigsH2, 'o', ms=9)
                        axs[2].plot(eigsA, 'o')
                        axs[0].set_title('hamiltonian:\n R={}, V={}'.format(R, V), color='C0')
                        axs[1].set_title('symmetry:\n R={}, V={}'.format(R2, B), color='C1')
                        axs[2].set_title('spectra')
            if make:
                multipage("figures/hamiltonian/symmetries/rule{}_symm.pdf".format(R))
                #plt.show()
                plt.close("all")

                #U = expm(-1j * dt * H)
                #eigs = eigvalsh(H)
                #eigs = np.sort(eigs)
                #np.save("eigs/R{}_V{}_L{}.npy".format(R, V, L), eigs)

                #UHviz(U, H, R)
                #spectrum_viz(eigs)
                ##time_evolve(U)
                #multipage("figures/hamiltonian/rule{}_continuous.pdf".format(R))
                #plt.close('all')
                #print()
