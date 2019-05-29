import numpy as np
from numpy.linalg import eigvalsh, eigh, matrix_power
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit
from qca import multipage
from figure3 import select, moving_average

# plotting defaults
import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
font = {"size": 9, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc("font", **font)


def brody_fit(x, n):
    def brody_func(x, eta, A):
        b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
        return A * b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))

    popt, pcov = curve_fit(brody_func, x, n, p0=[0.0, 1.0], bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov



from measures import renyi_entropy
L = 18
IC = "c1_f0"
Skey = [13, 14, 1, 6]
cs = ["darkturquoise", "darkorange", "limegreen", "crimson"]
for j, (c, S) in enumerate(zip(cs, Skey)):
    sim = select(L, S, IC, V="H", BC="0")
    h5file = sim["h5file"]
    d = h5file["cut_half"][:]
    for ti, rho in enumerate(d[100:101]):
        spec, vecs = eigh(rho)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        print(renyi_entropy(rho))
        axs[0].set_title("spectrum")
        axs[0].semilogy(spec, color=c, marker="o")

        axs[1].set_title("column vector magnitude")
        axs[1].imshow(np.abs(vecs), cmap="gist_gray_r")
        fig.suptitle("$T_{%d}$"%S)
multipage("figures/figure4/eigenRDM.pdf")
plt.close("all")
print("done")
if __name__ == "__main__":
    L = 18
    IC = "c1_f0"
    Skey = [13, 14, 1, 6]
    cs = ["darkturquoise", "darkorange", "limegreen", "crimson"]
    fig, axs = plt.subplots(1, 1, figsize=(4, 3), sharex=False)
    fig2, axs2 = plt.subplots(1, 1, figsize=(3, 2), sharex=True)
    fig2.subplots_adjust(left=0.2, bottom=0.2, hspace=0.1)

    fig3s = []

    for j, (c, S) in enumerate(zip(cs, Skey)):
        sim = select(L, S, IC, V="H", BC="0")
        h5file = sim["h5file"]
        try:
            espec = h5file["espec"]
        except:
            d = h5file["cut_half"]
            espec = np.zeros((d.shape[0], d.shape[1]))
            for ti, rho in enumerate(d):
                espec[ti, :] = eigvalsh(rho)
            h5file["espec"] = espec

        etas = []
        detas = []
        svns = []
        ii = 0
        t0 = 10

        fig3, axs3 = plt.subplots(3, 3, figsize=(4, 4), sharex=True, sharey=True)
        for ti, es in enumerate(espec[t0:1000]):
            es = es[es > 1e-6]
            NN = len(es)
            es = np.sort(es)
            es = es[NN // 3 : 2 * NN // 3]
            ns = range(len(es))
            s = es[1:] - es[:-1]
            s /= np.mean(s)
            n, bin, _ = axs.hist(
                s, density=True, alpha=1, histtype="step", bins=10, log=False
            )

            x = (bin[1:] + bin[:-1]) / 2.0
            xs = np.linspace(x[0], x[-1], 100)
            xs = xs[xs > 0]

            func, popt, pcov = brody_fit(x, n)
            detas.append(np.sqrt(np.diag(pcov)[0]))
            etas.append(popt[0])

            if (ti+t0) % 100 == 0:
                row, col = ii // 3, ii % 3
                ax3 = axs3[row, col]
                dx = x[1] - x[0]
                n = np.insert(n, 0, 0)
                n = np.insert(n, len(n), 0)
                x = np.insert(x, 0, x[0] - dx / 2)
                x = np.insert(x, len(x), x[-1] + dx / 2)
                ax3.step(x, n, where="mid")
                ax3.plot(xs, func(xs))
                ax3.set_title(f"t={t0+ti}", pad=-13)
                fig3.suptitle("$R = %d$" % S)
                ii += 1
                if col == 1 and row == 2:
                    ax3.set_xlabel("$\delta E/\overline{\delta E}$")
                if col == 0 and row == 1:
                    ax3.set_ylabel("density")
                ax3.tick_params(direction="inout")

        fig3.subplots_adjust(hspace=0, wspace=0)
        fig3s.append(fig3)
        etas = np.array(etas)
        detas = np.array(detas)
        ts = np.arange(2, len(etas) + 2)
        mask = detas < 1
        etas = etas[mask]
        detas = detas[mask]
        ts = ts[mask]

        if S == 6:
            pass
        else:
            if S == 13:
                label = r"$R = %s$" % S
            else:

                label = str(S)
            aetas = moving_average(etas, n=L)
            axs2.plot(aetas, marker=None, color=c, label=label, lw=1)
            avgerr = np.mean(detas)
            axs2.plot(ts, etas, c=c, alpha=0.3)
            #axs2.errorbar(ts, etas, yerr=detas, color=c)
        axs2.set_xticks([0, 250, 500, 750, 1000])

    axs2.legend(loc="lower right")
    axs.set_xlabel("$\delta E / \overline{\delta E}$")
    axs2.set_xlabel("$t$")
    axs2.set_ylabel("$\eta$")
    fig.tight_layout()
    fig2.tight_layout()

    multipage(
        "figures/figure4/spectrum_statistics_fixed-10bins.pdf",
        figs=[fig2] + fig3s,
        clip=True,
        dpi=10 * fig.dpi,
    )
