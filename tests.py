#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Testing states.py and matrix.py
#
# by Logan Hillberry
# ===============================================================

import numpy as np
import scipy as sp
import scipy.linalg
import matrix as mx
import states as ss
import copy
from math import log

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

def exp_val(rho, A):
    exp_val = np.real(np.trace(rho.dot(A)))
    return exp_val

def get_local_rhos(state):
    L = int(log(len(state), 2))
    local_rhos = np.asarray([mx.rdms(state, j) for j in range(L)])
    return local_rhos

def get_bipartition_rhos(state):
    L = int(log(len(state), 2)) - 1
    print('L',L+1)
    c = int(L/2)
    left_inds  = np.array(range(c+1))
    right_inds = np.array(range(c+1, L+1))
    print('l', left_inds)
    print('r', right_inds)
    left_rdms  = [0] * c
    right_rdms = [0] * (L - c)
    left_rdm = mx.rdms(state, left_inds)
    if L % 2 == 0:
        print('enter')
        left_rdms[-1] = left_rdm
        right_rdm = copy.copy(left_rdm)
    else:
        right_rdm = mx.rdms(state, right_inds)
        left_rdms[-1] = left_rdm
        right_rdms[0] = right_rdm

    right_inds = left_inds.copy() + 1
    left_inds = np.delete(left_inds, -1)
    right_inds = np.delete(right_inds, -1)
    m = 1
    print(left_rdm.shape, right_rdm.shape)
    while len(left_inds) > 0:
        print('l', left_inds)
        print('r', c-1 + np.array(right_inds), right_inds)
        left_rdm = mx.rdmr(left_rdm, left_inds)
        right_rdm = mx.rdmr(right_rdm, right_inds)
        #left_rdm = mx.rdms(state, left_inds)
        #right_rdm = mx.rdms(state, c-1 + np.array(right_inds))
        print(left_rdm.shape, right_rdm.shape)
        left_rdms[c - m - 1] = left_rdm
        right_rdms[m-1] = right_rdm
        left_inds = np.delete(left_inds, -1)
        right_inds = np.delete(right_inds, -1)
        m += 1
    # for odd system size L, the right half gets the extra cut
    if len(right_inds) > 0:
        print('RR', c-1 + np.array(right_inds))
        right_rdm = mx.rdmr(right_rdm, [1])
        right_rdms[-1] = right_rdm
        right_inds = np.delete(right_inds, -1)
    return np.asarray(left_rdms + right_rdms)


def get_local_exp_vals(state, A):
    local_rhos = get_local_rhos(state)
    local_exp_vals = np.asarray(
            [exp_val(rho) for rho in local_rhos]
        )
    return local_exp_vals

def get_local_entopies(state):
    local_rhos = get_local_rhos(state)
    local_s = np.asarray([vn_entropy(rho) for rho in local_rhos])
    return local_s

def get_bipartition_entropies(state):
    bipartition_rhos = get_bipartition_rhos(state)
    bipart_s = np.asarray([vn_entropy(rho) for rho in bipartition_rhos])
    return bipart_s

if __name__ == '__main__':
    L = 3
    R = ss.make_state(L, 'R298')
    R = np.array([i for i in range(2**(2*L))]).reshape(2**L,2**L)
    print(R)
    #print(get_bipartition_rhos(G))
    #rint(get_bipartition_entropies(G))
    #r1 = mx.rdms(R, [0])
    r2 = mx.rdmr(R,[0])
    #print(r1)
    #print(sum(np.diagonal(r1)))
    #print(' ')
    print(r2)
    print(sum(np.diagonal(r2)))
    #print(np.allclose(r1,r2))
