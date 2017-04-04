#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Basic quantum measures
#
# by Logan Hillberry
# ===============================================================

from math import log, sqrt
import numpy as np
import numpy as np
import scipy as sp
import scipy.linalg
import matrix as mx


# von Neumann entropy of reduced density matrix rho
# -------------------------------------------------
def vn_entropy(rho, tol=1e-14, get_snum=False):
    # define 0 log 0 = 0 where 0 = tol
    evals = sp.linalg.eigvalsh(rho)
    s = -sum(el*log(el, 2) if el >= tol else 0.0  for el in evals)
    # option for schmidt rank (number of non zero evals of rho)
    if get_snum:
        snum = sum(el > tol for el in evals)
        return s, snum
    else:
        return s

# expectation value of operator A w.r.t state
# -------------------------------------------
def exp_val(state, A):
    if len(state.shape) == 2:
        # input is a density matrix
        exp_val = np.real(np.trace(state.dot(A)))
    elif len(state.shape) == 1:
        # input is a state vector
        exp_val = np.real(np.conj(state).dot(A.dot(state)))
    else:
        raise ValueError('Input state not understood')
    return exp_val

# Thanks to David Vargas for the next three network measures

def network_density(mat):
    #calculates density, also termed connectance in some
    #literature. defined on page 134 of mark newman's book
    #on networks.
    l=len(mat)
    lsq=l*(l-1)
    return sum(sum(mat))/lsq

def network_clustering(mat):
    #calculates the clustering coefficient
    #as it is defined in equation 7.39 of
    #mark newman's book on networks. page 199.
    l=len(mat)
    matsq=np.linalg.matrix_power(mat, 2)
    matcube=np.linalg.matrix_power(mat, 3)
    #zero out diagonal entries. so we do not count edges as
    #connected triples.
    for i in range(len(matsq)):
        matsq[i][i]=0
    denominator=sum(sum(matsq))
    numerator=np.trace(matcube)
    #if there are no closed paths of length
    #three the clustering is automatically
    #set to zero.
    if numerator==0.:
        return 0.
    else:
        return numerator/denominator

def network_disparity(mat, eps=1j*1e-17):
    numerator=np.sum(mat**2,axis=1)
    denominator=np.sum(mat, axis=1)**2
    return (1/len(mat) * sum(numerator/(denominator + eps))).real


# get list of local density matrices
# ----------------------------------
def get_local_rhos(state):
    L = int(log(len(state), 2))
    local_rhos = np.asarray([mx.rdms(state, [j]) for j in range(L)])
    return local_rhos

# get list of two-site density matrices
# -------------------------------------
def get_twosite_rhos(state):
    L = int(log(len(state), 2))
    twosite_rhos = np.asarray([mx.rdms(state, [j, k]) for j in range(L) for k in
        range(j)])
    return twosite_rhos

# use two indicies to select rho from the list produces by get_twosite_rhos
# -------------------------------------------------------------------------
def select_twosite(twosite_rhos, j, k):
    if j == k:
        raise ValueError('[{}, {}] not valid two site indicies (cannot be the\
                same)'.format(j, k))
    row = max(j, k)
    col = min(j, k)
    ind = sum(range(row)) + col
    return twosite_rhos[ind]

# form a symmetric matrix from a vector of the triangular elements
# ----------------------------------------------------------------
def symm_mat_from_vec(vec):
    N = len(vec)
    L = int((1 + sqrt(1 + 8*N))/2)
    mat = np.zeros((L,L))
    for j in range(L):
        for k in range(L):
            if j != k:
                mat[j ,k] = select_twosite(vec, j, k)
    return mat

# get list of bi-partition density matrices
# -----------------------------------------
def get_bipartition_rhos(state):
    N = int(log(len(state), 2)) - 1
    c = int(N/2)
    left_rdms  = [0] * c
    left_rdm = mx.rdms(state, range(c))
    left_rdms[-1] = left_rdm
    right_rdms = [0] * (N - c)
    right_rdm = mx.rdms(state, range(c+1, N+1))
    right_rdms[0] = right_rdm
    for j in range(c-1):
        left_rdm = mx.traceout_last(left_rdm)
        left_rdms[c-j-2] = left_rdm
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[j+1] = right_rdm
    if N % 2 != 0:
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[-1] = right_rdm
    return left_rdms + right_rdms

# get list of local expectation values
# ------------------------------------
def get_local_exp_vals(state, A):
    local_rhos = get_local_rhos(state)
    local_exp_vals = local_exp_vals_from_rhos(local_rhos, A)
    return local_exp_vals

def local_exp_vals_from_rhos(rhos, A):
    local_exp_vals = np.asarray(
            [exp_val(rho, A) for rho in rhos])
    return local_exp_vals

def get_exp2_vals(state, A, B):
    twostie_rhos = get_twosite_rhos(state)
    exp2_mat = exp2_vals_from_rhos(twosite_rhos, A, B)
    return exp2_mat

def exp2_vals_from_rhos(rhos, A, B):
    exp2_vals = np.asarray(
            [exp_val(rho, np.kron(A,B)) for rho in rhos])
    exp2_mat = symm_mat_from_vec(exp2_vals)
    return exp2_mat

# get list of local von Neumann entropies
# ---------------------------------------
def get_local_entropies(state):
    local_rhos = get_local_rhos(state)
    local_s = local_entropies_from_rhos(local_rhos)
    return local_s

def local_entropies_from_rhos(rhos):
    s = np.asarray([vn_entropy(rho) for rho in rhos])
    return s

# von Neumann entropies of bipartitions
# -------------------------------------
def get_twosite_entropies(state):
    twosite_rhos = get_twosite_rhos(state)
    twosite_s_mat = local2
    return twosite_s_mat

def twosite_enropies_from_rhos(rhos):
    twosite_s = np.asarray([vn_entropy(rho) for rho in rhos])
    twosite_s_mat = symm_mat_from_vec(twosite_s)
    return twosite_s_mat

# get list of bi-partition von Neumann entropies
# ----------------------------------------------
def get_bipartition_entropies(state):
    bipartition_rhos = get_bipartition_rhos(state)
    bipart_s = np.asarray([vn_entropy(rho) for rho in bipartition_rhos])
    return bipart_s

def get_center_entropy(state):
    L = int(log(len(state)))
    center_rho = mx.rdms(state, list(range(int(L/2))))
    center_s = vn_entropy(center_rho)
    return center_s

# mutual information
# ------------------

def get_MI(state, eps=1e-14):
    s = get_local_entropies(state)
    s2 = get_twosite_entropies(state)
    return MI_from_entropies(s, s2, eps=eps)

def MI_from_rhos(onesite, twosite, eps=1e-14):
    s  = local_entropies_from_rhos(onesite)
    s2 = twosite_entropies_from_rhos(twosite)
    return MI_from_entropies(s, s2, eps=eps)

def MI_from_entropies(s, s2, eps=1e-14):
    L = len(s)
    MI = np.zeros((L, L))
    for j in range(L):
        for k in range(j):
            if j != k:
                MI[j,k] = s[j] + s[k] - s2[j, k]
                MI[k,j] = MI[j,k]
            elif j == k:
                MI[j,k] = eps
    return MI

# g2_{j,k}(A,B) = <AB>jk - <A>j<B>k correlator
# ----------------------------------------
def get_g2(state, A, B):
    exp2 = get_exp2_vals(state, A, B)
    exp1a = get_local_exp(state, A)
    exp1b = get_local_exp(state, B)
    return g2_from_exps(exp2, exp1a, exp1b)

def g2_from_rhos(onesite,_twosite, A, B):
    exp2 = exp2_vals_from_rhos(twosite, A, B)
    exp1a = local_exp_vals_from_rhos(onesite, A)
    exp1b = local_exp_vals_from_rhos(onesite, B)
    return g2_from_exps(exp2, exp1a, exp1b)

def g2_from_exps(exp2, exp1a, exp1b):
    L = len(exp1a)
    g2 = np.zeros((L, L))
    for j in range(L):
        for k in range(j):
            g2[j,k] = exp2[j,k] - exp1a[j] * exp1b[k]
            g2[k,j] = g2[j,k]
    return g2


