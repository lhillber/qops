from math import sqrt, e, pi
import numpy as np, scipy.odr.odrpack as odrpack, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import interpolate
from os import environ

def flin(B, x):
    return B[0] * x + B[1]


def fpow(B, x):
    return B[1] * x ** B[0] + B[2]


def gaussian(B, x):
    return 1.0 / (B[0] * sqrt(2 * pi)) * e ** (-(x - B[1]) ** 2 / (2 * B[0] ** 2))


def ftwo_gaussian(B, x):
    return 1.0 / (2 * B[0] * sqrt(2 * pi)) * e ** (-(x - B[1]) ** 2 / (2 * B[0] ** 2)) + 1.0 / (2 * B[2] * sqrt(2 * pi)) * e ** (-(x - B[3]) ** 2 / (2 * B[2] ** 2))


def fpoly(B, x):
    return np.sum([b * x ** m for m, b in enumerate(B)])


def fexp(B, x, base=e):
    return B[1] * base ** (B[0] * x) + B[2]


def fnexp(B, x, base=e):
    return B[0] + np.sum([B[i] * base ** (x * B[i + 1]) for i in range(1, len(B) - 1)])


def fsexp(B, x):
    return B[0] * B[1] ** (x ** B[2]) + B[3]


def do_odr(f, x, xe, y, ye, estimates):
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y, sx=xe, sy=ye)
    odr = odrpack.ODR(data, model, estimates)
    output = odr.run()
    return output


def chi2_calc(f, betas, x, y):
    chi2 = 0
    for xval, yval in zip(x, y):
        chi2 += (yval - f(betas, xval)) ** 2

    return chi2


def do_fit(func, beta_est, x_list, y_list, xerr=None, yerr=None):
    xerr = [1] * len(x_list) if xerr is None else xerr
    yerr = [1] * len(y_list) if yerr is None else yerr
    fit = do_odr(func, x_list, xerr, y_list, yerr, beta_est)
    chi2 = chi2_calc(func, fit.beta, x_list, y_list)
    return fit.beta, fit.sd_beta, chi2


def plot_f_fits(func, beta_est, x_list, y_list, ax, label, color, xerr=None, yerr=None, kwargs={}):
    xerr = [1e-07] * len(x_list) if xerr is None else xerr
    yerr = [1e-07] * len(y_list) if yerr is None else yerr
    fit_beta, chi2 = do_fit(func, beta_est, x_list, y_list, xerr=xerr, yerr=yerr)
    xs = np.linspace(min(x_list), max(x_list), 100)
    ax.plot(x_list, y_list, 'o', label='tmax = ' + str(label), color=color)
    ax.legend(loc='lower right')
    ax.set_yscale('log', basey=2)
