#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
#
# =============================================================================
#
# Simple implementation of quantum cellular automata in 1D generalized to
# arbitrary neighborhood radius r. Quantum elementary cellular automata (QECA)
# have r = 1. For neighborhood radius r, there are 2^2^(2*r) rules enumerated
# with an integer S between 0 and 2^2^(2*r)-1. Below I provide a list of
# possible neighborhood configurations and a corresponding rule number for r = 1
# and r = 2. Rule numbers are additive. For example, a QECA which is active with
# neighbors 1_0 and 0_1, where the underscore represents the site to be update, 
# has a rule number S = 4 + 2 = 6.
#
# Parameters:
# -----------
# 1)  L : int, length of 1D lattice (system size)
# 2)  T : int, number of iterations to simulate
# 3)  V : str, description of single-site unitary
# 4)  r : int, neighborhood radius
# 5)  S : int, rule number
# 6)  M : int or str, update mode for iterations
# 7)  BC : str, boundary conditions
# 8)  IC : str or tuple, initial condition. See states.py for more info
#
# More info:
# ----------
# 1)  L should not exceed 27. Hilbert space dimension and thus required
#     computational resources increases as 2^L
#
# 2)  T increases required computational resources only linearly
#
# 3)  V must be a string of keys found in the global dictionary 'ops' in
#     states.py.  Selecting V = 'X' (Pauli-x operator) corresponds to classical
#     evolution when provided with a Fock state initial condition. Selecting V =
#     'H' (Hadamard) or V = 'HP-45' (Hadamard times 45 degree phase gate) causes
#     decisively quantum behavior. NOTE: as of now, the sequence 'P-<deg>' can
#     only be placed at the END of the V string.
#
# 4)  Increasing r will increase required simulation time because update
#     operators are square matrices of dimension 2^(2*r + 1).
#
# 5)  S can no be negative or exceed 2^2^(2*r)
#
# 6)  M defines the update ordering by providing a 'skip size', denoted d. In an
#     iteration, sweeping the update from site 0 to L-1 has d = 1, provided by
#     setting M = 1 or M = 'swp'.  Alternating update (evens then odds) has d =
#     2, provided by M = 2 or M = 'alt'. In general, update is ordered by first
#     sites with index j such that j mod d = 0, then sites with index j mod d =
#     1, and so on until finally sites with j mod d = d-1 are updated. The
#     choice of d = 3 may be provided by M = 3 or M = 'blk'.
#
# 7)  BC first specifies ring or box boundaries with '0' or '1' respectively. If
#     the first character of BC is '0' all subsequent characters are ignored and
#     simulations are performed with ring boundary conditions. If the first
#     character of BC is '1', then BC is split at a dash '-' and everything
#     after the dash is used to configure the box boundaries. To be properly
#     configured, BC must have 2*r characters after the dash where each
#     character is either a '0' or a '1'. The first r characters set the left-r
#     boundary qubits to be set to |0> or |1> depending on the provided
#     character in the BC string.  Similarly, the last r characters set the
#     right-r boundary qubits. For example, BC = '1-0000' is a valid
#     configuration fo boundary conditions for QCA with r=2. It means that every
#     state in the evolution will be of the form |00>|Psi>|00>. Setting BC =
#     '1-0110' means every state in the evolution will be of the form
#     |01>|Psi>|10>. Finally, setting BC = '0' will use ring boundary conditions
#     for evolution.
#
# 8)  See states.py for more info on making states with stings and tuples.
#
#
# Single configuration rule numbers for r = 1 and r = 2 in the form 
# leftNeighbors_rightNeighbors : S
#
#  r = 1        r = 2
#  -----        ----- 
# 1_1 : 8    11_11 : 32786
# 1_0 : 4    11_10 : 16384
# 0_1 : 2    11_01 : 8192
# 0_0 : 1    11_00 : 4096
#            10_11 : 2048
#            10_10 : 1024
#            10_01 : 512
#            10_00 : 256
#            01_11 : 128
#            01_10 : 64
#            01_01 : 32
#            01_00 : 16
#            00_11 : 8
#            00_10 : 4
#            00_01 : 2
#            00_00 : 1
#
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
spiner = cycle(('-', '\\','|', '/'))


# r=2 totalistic rule numbers
N0 = 1
N1 = 278
N2 = 5736
N3 = 26752
N4 = 32786

# default behavior
# ----------------
def main():
    # lists of simulation parameters
    Ls  = [17]
    Ts  = [60]
    Vs  = ['X', 'HP-0']
    rs  = [2]
    Ss  = [4+16, N2+N3]
    Ms  = [14]
    ICs = ['c2_f0-1']
    BCs = ['1-0000', '0']

    # how to compose sim params into simulations (power set or zipped lists)
    thread_as = 'power' # use 'power' or 'zip'


    # execution block
    params_list = make_params_list(thread_as, Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    n_sims = len(params_list)
    for sim_id, params in enumerate(params_list):
        # initialize
        time_step_gen = get_time_step_gen(**params)
        L, T, S = [params[key] for key in ['L', 'T', 'S']]
        exp = np.zeros((T+1,L))

        # measures
        for t, next_state in enumerate(time_step_gen):
            exp[t,::] = ms.get_local_exp_vals(next_state, ss.ops['Z'])

        # plotting
        fig = plt.figure(sim_id)
        ax = fig.add_subplot(111)
        ax.imshow(exp, interpolation='None', origin='lower', vmin=-1, vmax=1)
        ax.set_title('S = {}'.format(S))

        # percent complete
        progress = 100 * (sim_id+1)/n_sims
        message ='finished simulation {}'.format(sim_id)
        track_progress(spiner, message, progress)

    # save all figures
    multipage('test.pdf')


# use method thread_as to make list of params_dict
# ------------------------------------------------
def make_params_list(thread_as, Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    if thread_as in ('zip', 'cycle', 'zipcycle', 'zip_cycle', 'zip cycle'):
            params_list = zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    elif thread_as in ('power', 'set', 'powerset', 'power_set', 'power set'):
            params_list = power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    else:
        print('Argument thread_as was given {} and is neither zip nor power'.format(
            thread_as))
        raise
    return params_list


# make list of params_dict by zipping input lists (shorter inputs are cycled)
# ---------------------------------------------------------------------------
def zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    args = (Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    lens = [l for l in map(len, args) ]
    ind = np.argmax(lens)
    to_zip = [el for el in map(cycle, args)]
    to_zip[ind] = args[ind]
    # number of sims given by longest input list
    return [make_params_dict(*params) for params in zip(*to_zip)]


# make list of parms_dict by forming the power set of input lists
# ---------------------------------------------------------------
def power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    # number of sims given by product of lenghts of input lists
    return [make_params_dict(L, T, V, r, S, M, IC, BC)
            for L in Ls
            for T in Ts
            for V in Vs
            for r in rs
            for S in Ss
            for M in Ms
            for IC in ICs
            for BC in BCs]


# dictionary data structure for simulation parameters
# ---------------------------------------------------
def make_params_dict(L, T, V, r, S, M, IC, BC):
    return {
            'L' : L,
            'T' : T,
            'V' : V,
            'r' : r,
            'S' : S,
            'M' : M,
            'IC' : IC,
            'BC' : BC
            }


# NOTE get_V only works if P-<th> appears at the end of the input V.
# ex) HP-45 is ok, P-45H will will not work.
# TODO fix this limitation.

# convert input V (string) to local unitary (2d numpy array)
# ----------------------------------------------------------
def get_V(V, s):
    V_conf = V.split('-')
    if len(V_conf) == 2:
        V_string, ph = V_conf
        ph = eval(ph)*pi/180.0
        Pmat = np.array([ [1.0,  0.0 ], [0.0 , exp(1.0j*ph)] ], dtype=complex)
        try:
            del ss.ops['P']
        except:
            pass
        ss.ops['P'] = Pmat
    else:
        V_string = V_conf[0]
    Vmat= mx.listdot([ss.ops[k] for k in V_string])
    return s*Vmat + (1-s)*ss.ops['I']


# gather update operators for boundaries and bulk
# -----------------------------------------------
def get_Us(V, r, S, BC_conf):
    U = make_U(V, r, S)
    bUs = make_boundary_Us(V, r, S, BC_conf)
    lUs = bUs[0:r]
    rUs = bUs[r:2*r + 1]
    return lUs, U, rUs


# convert input mode (string) to spatial update ordering
# ------------------------------------------------------
def get_mode(mode_name, L):
    if mode_name in ('sweep', 'swp'):
        d = 1
    elif mode_name in ('alternate', 'alt'):
        d = 2
    elif mode_name in ('block', 'blk'):
        d = 3
    else:
        try:
            d = int(mode_name)
        except:
            print('mode is not swp (d=1), alt (d=2), blk (d=3), or convertible\
            to an integer step size.')
            raise
    return make_mode_list(d, L)


# create spatial update ordering with skip size z for lattice length L
# --------------------------------------------------------------------
def make_mode_list(d, L):
    mode_list = []
    for delta in range(d):
        for j in range(L):
            if j % d == delta:
                mode_list.append(j)
    return mode_list


# make bulk update operator
# -------------------------
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


# make boundary update operators
# -----------------------------
def make_boundary_Us(V, r, S, BC_conf):
    BC_conf = list(map(int, BC_conf))
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2**N)[::-1]
    bUs = []
    for w in [0, 1]: # left (0) and right (1) side
        fixed_o  = list(BC_conf[w*r:r+w*r])
        for c in range(r):
            U = np.zeros((2**(r+1+c), 2**(r+1+c)), dtype=complex)
            for i in range(2**(r+c)):
                if w == 0:
                    var = mx.dec_to_bin(i, r+c)
                    var1 = var[0:c]
                    var2 = var[c::]
                    fixed = fixed_o[c:r]
                    n = fixed + var1 + var2
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = [ss.ops[str(op)] for op in var1] + [Vmat] +\
                            [ss.ops[str(op)] for op in var2]
                elif w == 1:
                    var = mx.dec_to_bin(i, r+c)[::-1]
                    var1 = var[c::][::-1]
                    var2 = var[0:c]
                    fixed = fixed_o[0:r-c:]
                    n =  var1 + var2 + fixed
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = [ss.ops[str(op)] for op in var1] + [Vmat] +\
                            [ss.ops[str(op)] for op in var2]
                U += mx.listkron(ops)
            bUs.append(U)
    return bUs


# single QCA iteration
# --------------------
def iterate(state, U, lUs, rUs, mode_list, L, r, BC_type):
    if BC_type == '1':
        for j in mode_list:
            if j < r:
                Nj = range(0, j+r+1)
                u = lUs[j]
            elif L - j - 1 < r:
                Nj = range(j-r, L)
                u = rUs[L-j-1]
            else:
                Nj = range(j-r, j+r+1)
                u = U
            Nj = list(Nj)
            state = mx.op_on_state(u, Nj, state)

    elif BC_type == '0':
        for j in mode_list:
            Nj = [k%L for k in range(j-r, j+r+1)]
            state = mx.op_on_state(U, Nj, state)
    return state


# update state with T iterations, yield each new state
# ----------------------------------------------------
def time_evolve(state, U, lUs, rUs, mode_list, L, T, r, BC_type):
    yield state
    for t in range(T):
        state = iterate(state, U, lUs, rUs, mode_list, L, r, BC_type)
        yield state


# get a generator of time evolved states
# --------------------------------------
def get_time_step_gen(L, T, V, r, S, M, IC, BC):
    # check data type of simulation parameters
    arg_names = ['L', 'T', 'V', 'r', 'S', 'M', 'IC', 'BC']
    args = [L, T, V, r, S, M, IC, BC]
    given_types = [t for t in map(type, args)]
    proper_types = [(int,), (int,), (str,), (int,), (int,), (str, int),
        (str, tuple), (str,)]
    mask = np.array([not t in pt for t, pt in zip(given_types, proper_types)])
    if np.any(mask == True):
        def grab_bad(lst):
            return np.array(lst)[mask]

        bad_inputs = [bad for bad in map(grab_bad,
            (arg_names, args, given_types, proper_types))]

        for arg_name, bad_arg, bad_type, proper_type in zip(*bad_inputs):
            print('Argument {} is {} (type {}) should be one of types {}'.format(
                arg_name, bad_arg, bad_type, proper_type))
        raise
    
    # check validity of rule number
    G = 2**2**(2*r)
    if S >= G:
        print('rule {} is invalid (need S < 2^2^(2*r) = {})'.format(S, G))
        raise

    # check validity of boundary conditions
    if BC[0] == '0':
        BC_type, BC_conf = '0', '00'
    elif BC[0] == '1':
        try:
            BC_type, BC_conf = BC.split('-')
        except:
            print('BC configuration is not formatted properly')
            raise
        if len(BC_conf) != 2*r:
            print('BC configuration {} is not valid'.format(BC_conf))
            raise
    else:
        print('BC type {} is not understood'.format(BC[0]))
        raise

    # execute valid time evolution
    state = ss.make_state(L, IC)
    mode_list = get_mode(M, L)
    lUs, U, rUs = get_Us(V, r, S, BC_conf)
    time_step_gen = time_evolve(state, U, lUs, rUs, mode_list, L, T, r, BC_type)
    return time_step_gen


# save multipage pdfs
# -------------------
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


# track progress in line of std out
# ---------------------------------
def track_progress(spiner, message, progress):
    sys.stdout.write("{0} {1}: {2:.1f} %\r".format(
                    next(spiner), message, progress))
    sys.stdout.flush()



# execute default behavior
# ----------------
if __name__ == '__main__':
    main()
