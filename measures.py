#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Basic quantum measures
#
# by Logan Hillberry
# ===============================================================

from math import log
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
        print('input state not understood')
        raise
    return exp_val


# get list of local density matrices
# ----------------------------------
def get_local_rhos(state):
    L = int(log(len(state), 2))
    local_rhos = np.asarray([mx.rdms(state, [j]) for j in range(L)])
    return local_rhos


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
    local_exp_vals = np.asarray(
            [exp_val(rho, A) for rho in local_rhos]
        )
    return local_exp_vals


# get list of local von Neumann entropies
# ---------------------------------------
def get_local_entropies(state):
    local_rhos = get_local_rhos(state)
    local_s = np.asarray([vn_entropy(rho) for rho in local_rhos])
    return local_s


# get list of bi-partition von Neumann entropies
# ----------------------------------------------
def get_bipartition_entropies(state):
    bipartition_rhos = get_bipartition_rhos(state)
    bipart_s = np.asarray([vn_entropy(rho) for rho in bipartition_rhos])
    return bipart_s
