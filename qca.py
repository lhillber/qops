#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
#
# =============================================================================
#
# Description:
# -----------
# Simple implementation of quantum cellular automata in 1D generalized to
# arbitrary neighborhood radius r. Quantum elementary cellular automata (QECA)
# have r = 1. For neighborhood radius r, there are 2^2^(2*r) rules enumerated
# with an integer S between 0 and 2^2^(2*r)-1. Below I provide a list of
# possible neighborhood configurations and a corresponding rule number for r = 1
# and r = 2. Rule numbers are additive. For example, a QECA which is active with
# neighbors 1_0 and 0_1, where the underscore represents the site to be update, 
# has a rule number S = 4 + 2 = 6.
#
# Usage:
# -----
# To run this file with default behavior, execute the following command in the
# terminal while in the directory containing this script
#
#                                 python3 qca.py
#
# This will create a file called test.pdf which plots a space time grid of
# expectation values of the Pauli-z operator with respect to each qubit at each
# time step of the qca evolution. The default parameters can be set in the globally defined
# dictionary called defaults below.
#
# To run the file with lists of parameters supplied from the command line, use
#
#    python3 qca.py "<Ls>" "<Ts>" "<Vs>" "<rs>" "<Ss>" "<Ms>" "<BCs>" "<ICs>"
#
# where "<PARAMs>" represents a comma separated list of PARAM supplied in
# quotes. For example, the following will recreate the default behivor
#
#        python3 qca.py "16" "60" "H" "2" "20, 32488" "4" "1-0000" "c2_f0-1" 
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
# 8)  IC : str or list, initial condition. See states.py for more info
#
# More info:
# ----------
# 1)  L should not exceed 27. Hilbert space dimension and thus required
#     computational resources increases as 2^L.
#
# 2)  T increases required computational resources only linearly.
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
# 5)  S can not be negative or exceed 2^2^(2*r)
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
#     boundary qubits to be |0> or |1> depending on the provided character in
#     the BC string.  Similarly, the last r characters set the right-r boundary
#     qubits. For example, BC = '1-0000' is a valid configuration for boundary
#     conditions of QCA with r = 2. It means that every state in the evolution
#     will be of the form |00>|Psi>|00>. Setting BC = '1-0110' means every state
#     in the evolution will be of the form |01>|Psi>|10>. Finally, setting BC =
#     '0' will use ring boundary conditions
#     for evolution.
#
# 8)  See states.py for more info on making states with stings or lists.
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
# r=2 totalistic rule numbers, defined here for convenience
N0 = 1
N1 = 278
N2 = 5736
N3 = 26752
N4 = 32786
#
# By Logan Hillberry
# =============================================================================


# custom modules
import states as ss
import matrix as mx
import measures as ms

# external modules
import h5py
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# built in modules
import os
import sys
import time
import errno
from math import log, pi
from itertools import cycle
from cmath import exp, sin, cos


# default behavior (used if no arguments are supplied form the command line)
# --------------------------------------------------------------------------
defaults = {
        'Ls' : [13],
        'Ts' : [60],
        'Vs' : ['RP_90-180'],
        'rs' : [1],
        'Ss' : [1, 6, 7, 9, 14],
        'Ms' : [2],
        'ICs': ['c1_f0'],
        'BCs': ['1-00'],
        'thread_as' : 'power',
        'sub_dir'   : 'link_test',
        'tasks'     : ['g2-ZZ', 'g2-XY', 'g2-XZ', 'MI']
        }

# typ is 'data' or 'plots',
# make sure project_dir points to your local clone of the qops repo
def dir_params(typ, sub_dir):
    return {                              # location of this file
            'project_dir' : os.path.dirname(os.path.realpath(__file__)),
            'output_dir'  : 'qca_output', # appends to project_dir
            'sub_dir'     : sub_dir,      # appends to output_dir
            'typ'         : typ           # appends to sub_dir (plots or data)
            }

# Unique name of simulation parameters
def make_name(L,T, V, r, S, M, IC, BC):
    name = 'L{}_T{}_V{}_r{}_S{}_M{}_IC{}_BC{}'.format(L, T, V, r, S, M, IC, BC)
    return name

# create and return a path, makes it if it doesn't exist
def make_path(project_dir, output_dir, sub_dir, typ):
    path = os.path.join(project_dir, output_dir, sub_dir, typ)
    os.makedirs(path, exist_ok=True)
    return path

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def main():
    # initialize parallel communication. rank is the name for each parallel
    # process from 0 to nprocs - 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    args = get_args()
    sub_dir = defaults['sub_dir']
    thread_as = defaults['thread_as']
    params_list = make_params_list(thread_as, *args)
    n_sims = len(params_list)
    tasks = defaults['tasks']
    for sim_id, params in enumerate(params_list):
        if sim_id % nprocs == rank:
            t0 = time.time()
            name = make_name(**params)
            # file names
            master_data_path = make_path(**dir_params('data', 'master'))
            master_data_fname = os.path.join(master_data_path, name) + '.hdf5'
            master_plot_path = make_path(**dir_params('plots', 'master'))
            master_plot_fname = os.path.join(master_plot_path, name) + '.pdf'
            # sym link names
            data_path = make_path(**dir_params('data', defaults['sub_dir']))
            data_fname = os.path.join(data_path, name) + '.hdf5'
            plot_path = make_path(**dir_params('plots', defaults['sub_dir']))
            plot_fname = os.path.join(plot_path, name) + '.pdf'

            params['name'] = name
            params['data_fname'] = master_data_fname
            params['plot_fname'] = master_plot_fname

            h5file = h5py.File(master_data_fname)
            h5file, tasks_added = run_required(params, tasks, h5file)
            h5file.close()
            symlink_force(master_data_fname, data_fname)
            t_elapsed = time.time() - t0
            print(print_string(params, rank, tasks_added, t_elapsed))

            # plotting
            '''
            fig = plt.figure(0)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(exp, interpolation='None', origin='lower', vmin=-1, vmax=1)
            ax.set_title('S = {}'.format(S))
            '''
            # TODO: move plot saving to the plotting function
            ##multipage(plot_fname)

            # tracking progress only possible for serial simulations

def print_string(params, rank, tasks_added, t_elapsed):
    print_string = '\n' + ('='*80) +'\n'
    print_string += 'Rank: {}\n'.format(rank)
    if len(tasks_added) == 0:
        print_string += 'Nothing to add to {}\n'.format(params['name'])
    else:
        print_string += 'Updated: {}\n'.format(params['name'])
        print_string += 'with {}\n'.format(tasks_added)
    print_string += 'total file size: {:.2f} MB\n'.format(
            os.path.getsize(params['data_fname'])/1e6)
    print_string +='took: {:.2f} s\n'.format(t_elapsed)
    print_string += 'data at:\n'
    print_string += params['data_fname'] + '\n'
    print_string += 'plots at:\n'
    print_string += params['plot_fname'] + '\n'
    print_string += '='*80
    return print_string

def run_required(params, tasks, h5file):
    avail_time_tasks, avail_meas_tasks = get_avail_tasks(h5file)
    needed_time_tasks, needed_meas_tasks = check_deps(
            tasks, avail_time_tasks, avail_meas_tasks)

    needed_time_tasks = [task for task in needed_time_tasks if
            task not in avail_time_tasks]
    needed_meas_tasks = [task for task in needed_meas_tasks if
            task not in avail_meas_tasks]
    if len(needed_time_tasks) > 0:
        time_evolve(params, needed_time_tasks, h5file)
    if len(needed_meas_tasks) > 0:
        measure(params, needed_meas_tasks, h5file)
    tasks_added = needed_time_tasks + needed_meas_tasks
    return h5file, tasks_added

def time_evolve(params, time_tasks, h5file):
    # initialize
    time_step_gen = get_time_step_gen(params)
    L, T, S = [params[key] for key in ['L', 'T', 'S']]
    init_args = [(L, T)]*len(time_tasks)
    res = init_tasks(time_tasks, init_args, res={})
    for t, next_state in enumerate(time_step_gen):
        for task in time_tasks:
            task_map[task]['set'](next_state, res[task], t)
    added_time_tasks = res.pop('tasks_tmp')
    try:
        updated_time_tasks = np.append(
                h5file['time_tasks'][::], added_time_tasks)
        del h5file['time_tasks']
    except:
        updated_time_tasks = added_time_tasks
    h5file['time_tasks'] = updated_time_tasks
    save_dict_hdf5(res, h5file)
    del res

def measure(params, meas_tasks, h5file):
    L, T, S = [params[key] for key in ['L', 'T', 'S']]
    init_args = [(L, T)]*len(meas_tasks)
    res = init_tasks(meas_tasks, init_args, res={})
    for task in meas_tasks:
        task_slim = task.split('-')[0]
        task_map[task_slim]['set'](h5file, res, task)
    added_meas_tasks = res.pop('tasks_tmp')
    try:
        updated_meas_tasks = np.append(
                h5file['meas_tasks'][::], added_meas_tasks)
        del h5file['meas_tasks']
    except:
        updated_meas_tasks = added_meas_tasks
    h5file['meas_tasks'] = updated_meas_tasks
    save_dict_hdf5(res, h5file)

def get_avail_tasks(h5file):
    try:
        avail_time_tasks = h5file['time_tasks'][::].astype('<U9')
    except:
        avail_time_tasks = []
    try:
        avail_meas_tasks = h5file['meas_tasks'][::].astype('<U9')
    except:
        avail_meas_tasks = []
    return avail_time_tasks, avail_meas_tasks

def check_deps(tasks, avail_time_tasks, avail_meas_tasks):
    time_tasks = [task for task in tasks if task in implemented_time_tasks]
    meas_tasks = [task for task in tasks if task.split('-')[0] in implemented_meas_tasks]
    required_time_tasks = [task for task in time_tasks if
            task not in avail_time_tasks]
    required_meas_tasks = [task for task in meas_tasks if
            task not in avail_meas_tasks]
    needed_time_tasks, needed_meas_tasks = recurs_check_deps(
            required_time_tasks, required_meas_tasks,
            avail_time_tasks, avail_meas_tasks, requested_time_tasks=[],
            requested_meas_tasks=[])
    return needed_time_tasks, needed_meas_tasks[::-1]

def recurs_check_deps(time_tasks, meas_tasks, avail_time_tasks, avail_meas_tasks,
        requested_time_tasks=[], requested_meas_tasks=[]):
    for meas_task in meas_tasks:
        meas_task_slim = meas_task.split('-')[0]
        deps = task_map[meas_task_slim]['deps'](meas_task)
        for dep in deps:
            dep_slim = dep.split('-')[0]
            if dep_slim in implemented_time_tasks:
                if not dep in requested_time_tasks:
                    if not dep in avail_time_tasks:
                        requested_time_tasks += [dep]
            elif dep_slim in implemented_meas_tasks:
                if not dep in requested_meas_tasks:
                    if not dep in avail_meas_tasks:
                        requested_meas_tasks += [dep]
                        required_time_tasks, required_meas_tasks =\
                                recurs_check_deps([], [dep],
                                avail_time_tasks, avail_meas_tasks,
                                requested_meas_tasks=requested_meas_tasks,
                                requested_time_tasks=requested_time_tasks)
            else:
                raise ValueError('requested task {} is not implemented'.format(dep))
    required_time_tasks = unique(
        time_tasks + requested_time_tasks + list(avail_time_tasks))
    required_meas_tasks = unique(
        meas_tasks + requested_meas_tasks + list(avail_meas_tasks))

    return required_time_tasks, required_meas_tasks

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_args():
    args = sys.argv[1:]
    nargs = len(args)
    keys = ('Ls', 'Ts', 'Vs', 'rs', 'Ss', 'Ms', 'ICs', 'BCs')
    if nargs not in (0, 10):
        raise ValueError('incorrect number of arguments')
    else:
        # supplied lists of simulation parameters
        if nargs == 10:
            args_dict = make_args_dict(args)
            defaults.update(args_dict)
    return [defaults[key] for key in keys]


# use method supplied by thread_as to make list of params_dict
# ------------------------------------------------------------
def make_params_list(thread_as, Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    if thread_as in ('zip', 'cycle', 'zipcycle', 'zip_cycle', 'zip cycle'):
            params_list = zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    elif thread_as in ('power', 'set', 'powerset', 'power_set', 'power set'):
            params_list = power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    else:
        raise ValueError(
            'Argument thread_as was given {} and is neither zip nor power'.format(
            thread_as))
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


def make_args_dict(args):
    arg_names = ['Ls', 'Ts', 'Vs', 'rs', 'Ss', 'Ms', 'ICs', 'BCs',
                 'sub_dir', 'thread_as']
    args_dict = dict(zip(arg_names, [0]*len(arg_names)))
    for arg_name, arg in zip(arg_names, args):
        if not arg_name in ('sub_dir', 'thread_as'):
            arg = arg2list(arg)
            if arg_name in ('Ls', 'Ts', 'rs', 'Ss', 'Ms'):
                arg = list(map(int, arg))
        args_dict[arg_name] = arg
    return args_dict


def arg2list(arg):
    try :
        a = arg.split(',')
    except:
        a = [arg]
    return a


# dictionary data structure for simulation parameters
# ---------------------------------------------------
def make_params_dict(L, T, V, r, S, M, IC, BC):
    return {
            'L'  : L,
            'T'  : T,
            'V'  : V,
            'r'  : r,
            'S'  : S,
            'M'  : M,
            'IC' : IC,
            'BC' : BC
            }


# NOTE get_V only works if P-<th> appears at the end of the input V.
# ex) HP-45 is ok, P-45H will will not work.
# TODO fix this limitation.

# convert input V (string) to local unitary (2d numpy array)
# ----------------------------------------------------------
def get_V(V, s):
    Vs, angs = V.split('_')
    angs = angs.split('-')
    ang_inds = [i for i, v in enumerate(Vs) if v in ('P', 'R', 'p')]

    if len(angs) != len(ang_inds):
        raise ValueError('impropper V configuration {}:\
                need one phase per every phase gate'.format(V))
    ang_id = 0
    Vmat = np.eye(2)
    for v in Vs:
        if v == 'P':
            ph = eval(angs[ang_id])*pi/180.0
            Pmat = np.array([ [1.0,  0.0 ], 
                              [0.0 , exp(1.0j*ph)] ],
                              dtype=complex)
            ss.ops['P'] = Pmat
            ang_id += 1
            Vmat = Vmat.dot(ss.ops[v])

        elif v == 'R':
            th = eval(angs[ang_id])*pi/180.0
            Rmat = np.array([ [cos(th/2),  -sin(th/2) ], 
                              [sin(th/2) , cos(th/2)] ],
                              dtype=complex)
            ss.ops['R'] = Rmat
            ang_id += 1
            Vmat = Vmat.dot(ss.ops[v])

        elif v == 'p':
            global_phase = eval(angs[ang_id])*pi/180
            global_phase = exp(1.0j*global_phase)
            ang_id += 1
            Vmat = global_phase * Vmat

        else:
            try:
                Vmat = Vmat.dot(ss.ops[v])
            except:
                raise ValueError('string op {} not found in states.ops'.format(v))
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
            raise ValueError('mode is not swp (d=1), alt (d=2), blk (d=3), or convertible\
            to an integer step size.')
    return make_mode_list(d, L)


# create spatial update ordering with skip size d for lattice length L
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
            if j < r:           # left boundary
                Nj = range(0, j+r+1)
                u = lUs[j]
            elif L - j - 1 < r: # right boundary
                Nj = range(j-r, L)
                u = rUs[L-j-1]
            else:               # bulk
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
def gen_time_step(state, U, lUs, rUs, mode_list, L, T, r, BC_type):
    yield state
    for t in range(T):
        state = iterate(state, U, lUs, rUs, mode_list, L, r, BC_type)
        yield state


# get a generator of time evolved states
# --------------------------------------
def get_time_step_gen(params):
    # check data type of simulation parameters
    arg_names = ['L', 'T', 'V', 'r', 'S', 'M', 'IC', 'BC']
    L, T, V, r, S, M, IC, BC = [params[key] for key in arg_names]
    args = [L, T, V, r, S, M, IC, BC]
    given_types = [t for t in map(type, args)]
    proper_types = [(int,), (int,), (str,), (int,), (int,), (str, int),
        (str, list), (str,)]
    mask = np.array([not t in pt for t, pt in zip(given_types, proper_types)])
    if np.any(mask == True):
        def grab_bad(lst):
            return np.array(lst)[mask]

        bad_inputs = [bad for bad in map(grab_bad,
            (arg_names, args, given_types, proper_types))]

        for arg_name, bad_arg, bad_type, proper_type in zip(*bad_inputs):
            raise ValueError(
                'Argument {} is {} (type {}) should be one of types {}'.format(
                    arg_name, bad_arg, bad_type, proper_type))

    # check validity of rule number
    G = 2**2**(2*r)
    if S >= G:
        raise ValueError('rule S = {} is invalid (need S < 2^2^(2*r) = {})'.format(S, G))

    # check validity of boundary conditions
    if BC[0] == '0':
        BC_type, BC_conf = '0', '00' # arbitrary
    elif BC[0] == '1':
        try:
            BC_type, BC_conf = BC.split('-')
        except:
            raise ValueError('BC configuration is not formatted properly')
        if len(BC_conf) != 2*r:
            raise ValueError('BC configuration {} is not valid'.format(BC_conf))
    else:
        raise ValueError('BC type {} is not understood'.format(BC[0]))

    # execute valid time evolution
    state = ss.make_state(L, IC)
    mode_list = get_mode(M, L)
    lUs, U, rUs = get_Us(V, r, S, BC_conf)
    time_step_gen = gen_time_step(state, U, lUs, rUs, mode_list, L, T, r, BC_type)
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
    sys.stdout.write("{0} {1} {2:.1f} %\r".format(
                    next(spiner), message, progress))
    sys.stdout.flush()

# recursive save dict to hdf5 file
# --------------------------------
# http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
def save_dict_hdf5(dic, h5file):
    recurs_save_dict_hdf5(h5file, '/', dic)


def recurs_save_dict_hdf5(h5file, path, dic_):
    for key, item in dic_.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recurs_save_dict_hdf5(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%item)

def load_dict_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recurs_load_dict_hdf5(h5file, '/')

def recurs_load_dict_hdf5(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recurs_load_dict_hdf5(h5file, path + key + '/')
    return ans


# Measures
# ========
implemented_time_tasks = ['one_site', 'two_site', 'sbond', 'scenter']
implemented_meas_tasks = ['exp', 'exp2', 's', 's2', 'MI', 'g2']

# functions for dependencies of tasks
def dep_g2(task):
    _, ops = task.split('-')
    deps = ['exp2-'+ops, 'exp-'+ops[0], 'exp-'+ops[1]]
    return deps

def dep_MI(task):
    return ['s', 's2']

def dep_one_site(task):
    return ['one_site']

def dep_two_site(task):
    return ['two_site']

# functions allocating memory for requested results
def init_one_site(L, T):
    return np.zeros((T+1, L, 2, 2), dtype=complex)
#
def init_two_site(L, T):
    return np.zeros((T+1, int(0.5*L*(L-1)), 4, 4), dtype=complex)
#
def init_sbond(L, T):
    return np.zeros((T+1, L-1))
#
def init_scalar(L, T):
    return np.zeros(T+1)
#
def init_vec(L, T):
    return np.zeros((T+1,L))
#
def init_mat(L, T):
    return np.zeros((T+1, L, L))


# functions for setting time tasks
def set_one_site(state, one_site, t):
    one_site[t, ::, ::, ::] = ms.get_local_rhos(state)
#
def set_two_site(state, two_site, t):
    two_site[t, ::, ::, ::] = ms.get_twosite_rhos(state)
#
def set_sbond(state, sbond, t):
    sbond[t, ::] = ms.get_bipartition_entropies(state)
#
def set_scenter(state, scenter, t):
    scenter[t] = ms.get_center_entropy(state)
#
# functions for setting meas tasks
def set_exp(h5file, res, task):
    exp = res[task]
    _, op = task.split('-')
    deps = task_map['exp']['deps'](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        exp[t, ::] = ms.local_exp_vals_from_rhos(rhos, ss.ops[op])

def set_exp2(h5file, res, task):
    exp2 = res[task]
    _, ops = task.split('-')
    deps = task_map['exp2']['deps'](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        exp2[t, ::, ::] = ms.exp2_vals_from_rhos(
                rhos, ss.ops[ops[0]], ss.ops[ops[1]])

def set_s(h5file, res, task):
    s = res[task]
    deps = task_map['s']['deps'](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        s[t, ::] = ms.local_entropies_from_rhos(rhos)

def set_s2(h5file, res, task):
    s2 = res[task]
    deps = task_map['s2']['deps'](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        s2[t, ::, ::] = ms.twosite_enropies_from_rhos(rhos)

def set_MI(h5file, res, task):
    MI = res[task]
    deps = task_map['MI']['deps'](task)
    ss, s2s = gather_deps(h5file, res, deps)
    for t, (s, s2) in enumerate(zip(ss, s2s)):
        MI[t,::, ::] = ms.MI_from_entropies(s, s2)

def set_g2(h5file, res, task):
    g2 = res[task]
    deps = task_map['g2']['deps'](task)
    exp2s, exp1as, exp1bs = gather_deps(h5file, res, deps)
    for t, (exp2, exp1a, exp1b) in enumerate(zip(exp2s, exp1as, exp1bs)):
        g2[t,::, ::] = ms.g2_from_exps(exp2, exp1a, exp1b)

# Init a set of tasks as a dictionary
def init_tasks(tasks, argss, res={}):
    for task, args in zip(tasks, argss):
        task_slim = task.split('-')[0]
        res[task] = task_map[task_slim]['init'](*args)
    res['tasks_tmp'] = np.array(tasks).astype('|S9')
    return res

# gather a lsit of required dependencies, same order as given to deps
def gather_deps(h5file, res, deps):
    new_avail = [k for k in res.keys()]
    old_avail = [k for k in h5file.keys()]
    out = []
    for dep in deps:
        if dep in new_avail:
            out += [res[dep]]
        elif dep in old_avail:
            out += [h5file[dep][::]]
    return out

task_map = {
    'one_site' : {
        'init' : init_one_site,
        'set'  : set_one_site
            },

    'two_site' : {
        'init' : init_two_site,
        'set'  : set_two_site
            },

    'sbond'    : {
        'init' : init_sbond,
        'set'  : set_sbond
            },

    'scenter'  : {
        'init' : init_scalar,
        'set'  : set_scenter
            },

    's'    : {
        'deps' : dep_one_site,
        'init' : init_vec,
        'set'  : set_s,
            },

    's2'   : {
        'deps' : dep_two_site,
        'init' : init_mat,
        'set'  : set_s2
            },

    'g2'   : {
        'deps' : dep_g2,
        'init' : init_mat,
        'set'  : set_g2
            },

    'MI'   : {
        'deps' : dep_MI,
        'init' : init_mat,
        'set'  : set_MI
            },

    'exp'  : {
        'deps' : dep_one_site,
        'init' : init_vec,
        'set'  : set_exp
            },

    'exp2' : {
        'deps' : dep_two_site,
        'init' : init_mat,
        'set'  : set_exp2
            }
        }


# execute default behavior
# ------------------------
if __name__ == '__main__':
    main()
