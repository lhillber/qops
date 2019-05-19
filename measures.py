import numpy as np
from numpy import log2, sqrt
from numpy.linalg import matrix_power
import scipy as sp, scipy.linalg, matrix as mx
from math import pi


def spectrum(rho):
    spec = sp.linalg.eigvalsh(rho)
    return spec


def vn_entropy(rho, tol=1e-14):
    spec = spectrum(rho)
    s = -np.sum(((el * log2(el) if el >= tol else 0.0) for el in spec))
    return s


def renyi_entropy(rho, order=2, tol=1e-14):
    if order == 1:
        return vn_entropy(rho, tol=tol)
    else:
        denom = 1.0 - order
        # spec = spectrum(rho)
        # s = np.real(log2(np.sum(spec**order)) / denom)
        s = np.real(log2(np.trace(matrix_power(rho, order))) / denom)
    return s


def exp_val(state, A):
    if len(state.shape) == 2:
        exp_val = np.real(np.trace(state.dot(A)))
    else:
        if len(state.shape) == 1:
            exp_val = np.real(np.conjugate(state).dot(A.dot(state)))
        else:
            raise ValueError("Input state not understood")
    return exp_val


def network_density(mat):
    l = len(mat)
    lsq = l * (l - 1)
    return sum(sum(mat)) / lsq


def network_clustering(mat):
    l = len(mat)
    matsq = matrix_power(mat, 2)
    matcube = matrix_power(mat, 3)
    for i in range(len(matsq)):
        matsq[i][i] = 0

    denominator = sum(sum(matsq))
    numerator = np.trace(matcube)
    if numerator == 0.0:
        return 0.0
    return numerator / denominator


def network_disparity(mat, eps=1e-17j):
    numerator = np.sum(mat ** 2, axis=1)
    denominator = (np.sum(mat, axis=1)) ** 2
    return (1 / len(mat) * sum(numerator / (denominator + eps))).real


def get_local_rhos(state):
    L = int(log2(len(state)))
    local_rhos = np.asarray([mx.rdms(state, [j]) for j in range(L)])
    return local_rhos


def get_twosite_rhos(state):
    L = int(log2(len(state)))
    twosite_rhos = np.asarray(
        [mx.rdms(state, [j, k]) for j in range(L) for k in iter(range(j))]
    )
    return twosite_rhos


def select_twosite(twosite_rhos, j, k):
    if j == k:
        raise ValueError(
            "[{}, {}] not valid two site indicies (cannot be the same)".format(j, k)
        )
    row = max(j, k)
    col = min(j, k)
    ind = sum(range(row)) + col
    return twosite_rhos[ind]


def symm_mat_from_vec(vec):
    N = len(vec)
    L = int((1 + sqrt(1 + 8 * N)) / 2)
    mat = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j != k:
                mat[(j, k)] = select_twosite(vec, j, k)

    return mat


def get_bipartition_rhos(state):
    N = int(log2(len(state))) - 1
    c = int(N / 2)
    left_rdms = [0] * c
    left_rdm = mx.rdms(state, range(c))
    left_rdms[-1] = left_rdm
    right_rdms = [0] * (N - c)
    right_rdm = mx.rdms(state, range(c + 1, N + 1))
    right_rdms[0] = right_rdm
    for j in range(c - 1):
        left_rdm = mx.traceout_last(left_rdm)
        left_rdms[c - j - 2] = left_rdm
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[j + 1] = right_rdm

    if N % 2 != 0:
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[-1] = right_rdm
    return left_rdms + right_rdms


def get_local_exp_vals(state, A):
    local_rhos = get_local_rhos(state)
    local_exp_vals = local_exp_vals_from_rhos(local_rhos, A)
    return local_exp_vals


def local_exp_vals_from_rhos(rhos, A):
    local_exp_vals = np.asarray([exp_val(rho, A) for rho in rhos])
    return local_exp_vals


def get_exp2_vals(state, A, B):
    twosite_rhos = get_twosite_rhos(state)
    exp2_mat = exp2_vals_from_rhos(twosite_rhos, A, B)
    return exp2_mat


def exp2_vals_from_rhos(rhos, A, B):
    exp2_vals = np.asarray([exp_val(rho, np.kron(A, B)) for rho in rhos])
    exp2_mat = symm_mat_from_vec(exp2_vals)
    return exp2_mat


def get_local_entropies(state, order):
    local_rhos = get_local_rhos(state)
    local_s = local_entropies_from_rhos(local_rhos, order)
    return local_s


def local_entropies_from_rhos(rhos, order):
    s = np.asarray([renyi_entropy(rho, order) for rho in rhos])
    return s


def twosite_entropies_from_rhos(rhos, order):
    twosite_s = np.asarray([renyi_entropy(rho, order) for rho in rhos])
    twosite_s_mat = symm_mat_from_vec(twosite_s)
    return twosite_s_mat


def get_twosite_entropies(state, order):
    twosite_rhos = get_twosite_rhos(state)
    twosite_s_mat = twosite_entropies_from_rhos(twosite_rhos, order)
    return twosite_s_mat


def get_bipartition_entropies(state, order):
    bipartition_rhos = get_bipartition_rhos(state)
    bipart_s = np.asarray([renyi_entropy(rho, order) for rho in bipartition_rhos])
    return bipart_s


def get_center_rho(state):
    L = int(log2(len(state)))
    center_rho = mx.rdms(state, list(range(int(L / 2))))
    return center_rho


def get_center_entropy(state, order):
    center_rho = get_center_rho(state)
    center_s = renyi_entropy(center_rho, order)
    return center_s


def get_MI(state, order=1, eps=1e-14):
    s = get_local_entropies(state, order=order)
    s2 = get_twosite_entropies(state, order=order)
    return MI_from_entropies(s, s2, eps=eps)


def MI_from_rhos(onesite, twosite, order=1, eps=1e-14):
    s = local_entropies_from_rhos(onesite, order)
    s2 = twosite_entropies_from_rhos(twosite, order)
    return MI_from_entropies(s, s2, eps=eps)


def MI_from_entropies(s, s2, eps=1e-14):
    L = len(s)
    MI = np.zeros((L, L))
    for j in range(L):
        for k in range(j):
            if j != k:
                MI[(j, k)] = (s[j] + s[k] - s2[(j, k)]) / 2.0
                MI[(k, j)] = MI[(j, k)]
            elif j == k:
                MI[(j, k)] = 0

    return MI


def get_g2(state, A, B):
    exp2 = get_exp2_vals(state, A, B)
    exp1a = get_local_exp_vals(state, A)
    exp1b = get_local_exp_vals(state, B)
    return g2_from_exps(exp2, exp1a, exp1b)


def g2_from_rhos(onesite, twosite, A, B):
    exp2 = exp2_vals_from_rhos(twosite, A, B)
    exp1a = local_exp_vals_from_rhos(onesite, A)
    exp1b = local_exp_vals_from_rhos(onesite, B)
    return g2_from_exps(exp2, exp1a, exp1b)


def g2_from_exps(exp2, exp1a, exp1b):
    L = len(exp1a)
    g2 = np.zeros((L, L))
    for j in range(L):
        for k in range(j):
            g2[(j, k)] = exp2[(j, k)] - exp1a[j] * exp1b[k]
            g2[(k, j)] = g2[(j, k)]

    return g2


def autocorr(x, h=1):
    N = len(x)
    mu = np.mean(x)
    acorr = sum(((x[j] - mu) * (x[j + h] - mu) for j in range(N - h)))
    denom = sum(((x[j] - mu) ** 2 for j in range(N)))
    if denom > 1e-14:
        acorr = acorr / denom
    else:
        print("auto correlation denom less than", 1e-14)
    return acorr


def fourier(sig, dt=1, h=1):
    sig = np.nan_to_num(sig)
    # remove transient. TODO: check heuristic
    if len(sig) > 300:
        sig = sig[300:]
    sig = sig - np.mean(sig)
    n = sig.size
    ps = np.absolute(np.fft.rfft(sig) / n) ** 2
    fs = np.fft.rfftfreq(n, dt)
    a1 = autocorr(sig, h=1)
    a = a1
    rn = 1 - a ** 2
    rn = rn / (1 - 2 * a * np.cos(2 * pi * fs / dt) + a ** 2)
    rn = rn * sum(ps) / sum(rn)
    return np.asarray([fs, ps[: n // 2 + 1], rn])


def fourier2D(vec, dt=1, dx=1):
    vec = np.nan_to_num(vec)
    T, L = vec.shape
    if T > 300:
        vec = vec[300:, :]
        T = T - 300
    vec = vec - np.mean(vec)
    ps = np.absolute(np.fft.fft2(vec) / (L * T)) ** 2
    ps = ps[: T // 2 + 1, : L // 2 + 1]
    ws = np.fft.rfftfreq(T, d=dt)
    ks = np.fft.rfftfreq(L, d=dx)
    ret = np.zeros((len(ws) + 1, len(ks) + 1))
    ret[0, 1:] = ks
    ret[1:, 0] = ws
    ret[1:, 1:] = ps
    return ret
