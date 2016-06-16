#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# =============================================================================
#
# Simple implementation of quantum elementary cellular automata
#
# By Logan Hillberry
# =============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import states as ss
import matrix as mx
import measures as ms
from math import log, pi
from cmath import exp
from itertools import cycle

L = 13
T = 26

V = 'X'
r = 2
S = 40
mode = 'alt'

IC = 'c1_f0'
BC = '1_0000'

spiner = cycle(('-', '\\','|', '/'))

def get_V(V, s):
    V_conf = V.split('-')
    if len(V_conf) == 2:
        V_string, ph = V_conf
        ph = eval(ph)*pi/180.0
        Pmat = np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*ph)]], dtype=complex )
        ss.ops['P'] = Pmat
    else:
        V_string = V_conf[0]
    Vmat= mx.listdot([ss.ops[k] for k in V_string])
    return s*Vmat + (1-s)*ss.ops['I']

def get_Us(V, r, S, BC_type, BC_conf):
    U = make_U(V, r, S)
    if BC_type == '1':
        bUs = make_boundary_Us(V, r, S, BC_conf)
        lUs = bUs[0:r]
        rUs = bUs[r:2*r + 1]
        return lUs, U, rUs
    elif BC_type == '0':
        return U

def get_mode(mode_name, L):
    if mode_name == 'swp':
        d = 1
    elif mode_name == 'alt':
        d = 2
    elif mode_name == 'blk':
        d = 3
    return make_mode_list(d, L)

def make_mode_list(d, L):
    mode_list = []
    for delta in range(d):
        for j in range(L):
            if j % d == delta:
                mode_list.append(j)
    return mode_list

def neighborhood(j, L, r, BC_type):
    if BC_type == '1':
        if j < r:
            js = range(0, j+r+1)
        elif L - j - 1 < r:
            js = range(j-r, L)
        else:
            js = range(j-r, j+r+1)
        return list(js)
    elif BC_type == '0':
        return

def make_U(V, r, S):
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2**N)[::-1]
    U = np.zeros((2**(N+1), 2**(N+1)), dtype=complex)
    for sig, s in enumerate(Sb):
        sigb = mx.dec_to_bin(sig, N)
        Vmat = get_V(V, s)
        ops = [ss.ops[str(op)] for op in sigb[0:r]] + [Vmat] +\
                [ss.ops[str(op)] for op in sigb[r:2*r+1]]
        U += mx.listkron(ops)
    return U

def make_boundary_Us(V, r, S, BC_conf):
    BC_conf = list(map(int, BC_conf))
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2**N)[::-1]
    bUs = []
    for w in [0, 1]: # left (0) and right (1) side
        for c in range(r):
            U = np.zeros((2**(r+1+c), 2**(r+1+c)), dtype=complex)
            fixed  = list(BC_conf[w*r:r+w*r])
            for i in range(2**(r+c)):
                var = mx.dec_to_bin(i, r+c)
                var1 = var[0:c]
                var2 = var[c::]
                if w == 0:
                    n = fixed + var1 + var2
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = [ss.ops[str(op)] for op in var1] + [Vmat] +\
                            [ss.ops[str(op)] for op in var2]
                elif w == 1:
                    n = fixed + var1 + var2
                    #n =  var2 + var1 + fixed
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = [ss.ops[str(op)] for op in var2] + [Vmat] +\
                            [ss.ops[str(op)] for op in var1]
                U += mx.listkron(ops)
            bUs.append(U)
    return bUs



def iterate(state, U, lUs, rUs, L, r, BC_type):
    for j in get_mode(mode, L):
            js = neighborhood(j, L, r, BC_type)
            if j < r:
                u = lUs[j]
            elif L - j - 1 < r:
                u = rUs[L-j-1]
            else:
                u = U
            state = mx.op_on_state(u, js, state)
    return state


def time_evolve(state, U, lUs, rUs, L, r, BC_type):
    yield state
    for t in range(T):
        state = iterate(state, U, lUs, rUs, L, r, BC_type)
        yield state


def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(pp, format='pdf', bbox_inches='tight',
                        bbox_extra_artist=extra_artist)
        else:
            fig.savefig(pp, format='pdf', bbox_extra_artist=extra_artist)
        if clf==True:
            fig.clf()
    pp.close()
    return

def track_progress(spiner, message, progress):
    sys.stdout.write("{0} {1}: {2:.1f} %\r".format(
                    next(spiner), message, progress))
    sys.stdout.flush()

if __name__ == '__main__':
    G = 2**2**(2*r)
    print(G)
    for S in range(G):
        state = ss.make_state(L, IC)
        BC_type, BC_conf = BC.split('_')
        lUs, U, rUs = get_Us(V, r, S, BC_type, BC_conf)
        exp = np.zeros((T+1,L))
        for t, next_state in enumerate(time_evolve(state, U, lUs, rUs, L, r, BC_type)):
            #print(t/T * 100)
            exp[t,::] = ms.get_local_exp_vals(next_state, ss.ops['Z'])
        fig = plt.figure(S%100)
        ax = fig.add_subplot(111)
        ax.imshow(exp, interpolation='None', origin='lower')
        ax.set_title('S = {}'.format(S))

        progress = 100 * S/G
        message ='evolving r=2 rules'
        track_progress(spiner, message, progress)
        if (S+1)%100 == 0:
            multipage('r2/r2_classical_S{}-{}.pdf'.format(S-99, S))
            plt.clf()

    multipage('r2_classical_S{}.pdf'.format('remainder'))
