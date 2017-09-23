#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#
# =============================================================================
#
# Description:
# -----------
# Full implementation of quantum cellular automata in 1D generalized to
# arbitrary neighborhood radius r. Quantum elementary cellular automata (QECA)
# have r = 1. For neighborhood radius r, there are 2^2^(2*r) rules enumerated
# with an integer S between 0 and 2^2^(2*r)-1. Below I provide a list of
# possible neighborhood configurations and a corresponding rule number for r = 1
# and r = 2. Rule numbers are additive. For example, a QECA which is active with
# neighbors 1_0 and 0_1, where the underscore represents the site to be updated,
# has a rule number S = 4 + 2 = 6, since 1_0 is 4 in binary and 0_1 is 2.
#
# Usage:
# -----
# To run this file with default behavior, execute the following command in the
# terminal while in the directory containing this script
#
#                                 python3 qca.py
#
# The default parameters can be set in the globally defined dictionary called
# defaults below.
#
# To run the file with lists of parameters supplied from the command line, use
#
#    python3 qca.py "<Ls>" "<Ts>" "<Vs>" "<rs>" "<Ss>" "<Ms>" "<ICs>" "<BCs>" "<tasks>" "<dir_name>" "<thread_as>"
#
# where each "<PARAMs>" represents a space separated list of PARAM supplied in
# quotes. For example, the following will recreate the default behivor
#
# python3 qca.py "15" "60" "H" "1" "1 6 9 14" "2" "c1_f0" "1-00" "g2-XX g2-YY g2-ZZ D C Y scenter" "default" "power"
#
# To run independent simulations in parallel, execute
#
#                  mpiexec -np <nprocs> python3 pca.py
#
# where <nprocs> is the number of parallel processes to be used (between 4 and 8
# is a reasonable number for most modern laptops). You can still supply
# command line arguments detailing a list of simulations to be executed as described
# above.
#
#
# Simulation Parameters:
# ----------------------
# 1)  L  : int, length of 1D lattice (system size)
# 2)  T  : int, number of iterations to simulate
# 3)  V  : str, description of single-site unitary
# 4)  r  : int, neighborhood radius
# 5)  S  : int, rule number
# 6)  M  : int or str, update mode for iterations
# 7)  IC : str or list, initial condition. See states.py for more info
# 8)  BC : str, boundary conditions
#
# Control Parameters:
# -------------------
# 9)  tasks     : str, measures to be applied to each simulation
# 10) dir_name  : str, name of directory to which all results will be linked
# 11) thread_as : str, method for combining lists of simulation parameters
#
# More info:
# ----------
# 1)  L should not exceed 27. Hilbert space dimension and thus required
#     computational resources increases as 2^L.
#
# 2)  T increases required computational resources linearly.
#
# 3)  V must be a string of keys found in the global dictionary 'ops' in
#     states.py.  Selecting V = 'X' (Pauli-x operator) corresponds to classical
#     evolution when provided with a Fock state initial condition. Selecting V =
#     'H' (Hadamard) or V = 'HP_45' (Hadamard times 45 degree phase gate) causes
#     decisively quantum behavior.
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
# 7)  See states.py for more info on making states with stings or lists.
#
# 8)  BC first specifies ring or box boundaries with '0' or '1' respectively. If
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
# 9)  tasks may be requested as high-level calculations and dependencies are
#     automatically handled. For example if 'g2-XY' is a requested task, then
#     first single-site and two-site reduced density matrices are
#     automatically calculated as well as the expectation values <X>, <Y>, <XY>
#     and returns <XY> - <X><Y>.
#     All intermediate calculations are also saved. Under the hood, tasks are
#     separated into 'time_tasks' and 'meas_tasks'. The time_tasks are those
#     which must be computed from the full state vector, e.g., single and
#     two-site reduced density matrices. The meas_tasks are those that can be
#     computed from time tasks or other existing meas_tasks. Note that bond
#     entropy 'sbond' and center bond entropy 'scenter' are considered to be
#     time tasks so that we don't have to save bi-partition reduced density
#     matrices to disk (as these can easily become very large). See the globally
#     defined dictionary task_map for a list of implemented tasks and their
#     associated methods.
#
# 10) dir_name is the name of a directory to be created and filled with the
#     requested simulations. The actual data is always saved with
#     dir_name = 'master' and the requested simulations are soft linked to the
#     directory with dir_name as supplied. This way, if the user requests a
#     simulation in a new dir_name, but has already run the same simulation
#     before, it won't rerun the simulation.
#
# 11) thread_as tells the program how to handle input list of params that are
#     different sizes. Three methods are currently implemented. In the following
#     examples I use "ai", "bj", and "ck" in place of actual simulation
#     parameters.
#         a) thread_as = 'power':
#            takes the power set of all parameters. e.g.,
#            "a1 a2" " b1" "c1 c2 c3" -> ([a1, b1, c1],
#                                         [a1, b1, c2],
#                                         [a1, b1, c3],
#                                         [a2, b1, c1],
#                                         [a2, b1, c2],
#                                         [a2, b1, c3])
#         b) thread_as = 'cycle':
#            zip-cycle  shorter lists e.g.,
#            "a1 a2" " b1" "c1 c2 c3" -> ([a1, b1, c1],
#                                         [a2, b1, c2],
#                                         [a1, b1, c3])
#
#         c) thread_as = 'last':
#            repeat the last element of shorter lists e.g.,
#            "a1 a2" " b1" "c1 c2 c3" -> ([a1, b1, c1],
#                                         [a2, b1, c2],
#                                         [a2, b1, c3])
#
# Here are single configuration rule numbers for r = 1 and r = 2 in the form
# leftNeighbors_rightNeighbors : S. These tables are useful in constructing rule
# numbers which are active for specific configurations. Just add up the ones you
# want!
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
# r = 2 totalistic rule numbers, defined here for convenience
N0 = 1
N1 = 278
N2 = 5736
N3 = 26752
N4 = 32786
#
#
# How to add a new measure:
# ------------------------
# This file implements a general way of adding measures. Measures are split into
# two types of tasks `time tasks` and `measure tasks`. Time tasks are those
# which require the entire quantum state to compute and are thus quite
# computationally intensive (include things like one and two site reduced
# density matrices and bond entropy calculations). Measure tasks are those that
# can be computed from any of the time tasks or other measure tasks (e.g., local
# and two site expectation values can be calculated with one and two site
# reduced density matrices. The g_2 correlator can be calculated from the one
# and two site expectation values). To add a new measure, the following steps
# can be followed (for each step I'll show the steps for adding the network
# measure `network_density`):
#
# 1) Write a function into measures.py that can calculate your measure
#
# def network_density(mat):
#     l=len(mat)
#     lsq=l*(l-1)
#     return sum(sum(mat))/lsq
#
#
# 2) Add a key for the function name to the list of inmplemented_meas_tasks
# (search for it; its a globally defined list. I'll use `D` in the example).
#
# implemented_meas_tasks = ['exp', 'exp2', 's', 's2', 'MI', 'g2', 'C', 'Y', 'D']
#
#
# 3) Write or choose a dependencies function (network density is a network
# measure on the mutual information, so it's only dependency is `MI` which is
# already implemented with its dependencies.)
#
# def dep_MI(task):
#     return ['MI']
#
#
# 4) Write or choose an appropriate init_function for your measure (is it a
# scalar at each time step? a vector of lenght L?)
#
# def init_scalar(L, T):
#    return np.zeros(T+1)
#
#
# 5) write a set function which calls the function from step 1) at each step of
# the time evolution. Follow this general pattern. Note
#
# def set_D(h5file, res, task):
#     D = res[task]
#     deps = task_map['D']['deps'](task)
#     MI, = gather_deps(h5file, res, deps) # important comma on LHS!
#     for t, mat in enumerate(MI):
#         D[t] = ms.network_density(mat)
#
#
# 5) Put it all together as a dictionary in the task_map dictionary (another
# globbaly defined variable; search for it.)
#
# task_map = {
# ...
#     'D'   : {
#         'deps' : dep_MI,
#         'init' : init_scalar,
#         'set'  : set_D
#             },
# ...
# }
#
# And that's it! Just add your new key (e.g. `D`) to the list of tasks in the
# default params dictionary below or request it from the command line.
#
# By Logan Hillberry
# Last updated 21 August 2016
# =============================================================================


# custom modules
import states as ss
import matrix as mx
import measures as ms

# external modules
import h5py
import numpy as np
from mpi4py import MPI
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# built in modules
import os
import sys
import time
import errno
from math import log, pi
from cmath import exp, sin, cos
from itertools import cycle, zip_longest

# plotting defaults
#mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)

# default arguments (used if no arguments are supplied form the command line)
# --------------------------------------------------------------------------
defaults = {
        # simulation parameter lists
        'Ls' : [10],
        'Ts' : [60],
        'Vs' : ['H'],
        'rs' : [1],
        'Ss' : [6],
        'Ms' : [2],
        'ICs': ['d'],
        'BCs': ['1-00'],
        # control parameters
        'tasks'     : ['g2-xx', 'g2-yy', 'g2-zz', 'FT-D', 'FT-Y', 'FT-C',
            'scenter', 'sbond', 'bipartitions', 'center_partition'],
        'sub_dir'   : 'doublon',
        'thread_as' : 'power'
        }

# The main loop, executes all simulations using requested number of processes
def main():
    # initialize parallel communication. rank is the name for each parallel
    # process from 0 to nprocs - 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    # parse input arguments and update defaults dictionary as needed
    args = get_args()
    sub_dir = defaults['sub_dir']
    thread_as = defaults['thread_as']
    tasks = defaults['tasks']
    # make list of params dictionaries
    params_list = make_params_list(thread_as, *args)
    # loop through all requested simulations
    for sim_id, params in enumerate(params_list):
        # assign independent simulations to available processors
        if sim_id % nprocs == rank:
            # start timer
            t0 = time.time()
            # file names. All simulations are saved under the sub dir 'master'
            name = make_name(**params)
            master_data_path  = make_path(**dir_params('data', 'master'))
            master_data_fname = os.path.join(master_data_path, name) + '.hdf5'
            master_plot_path  = make_path(**dir_params('plots', 'master'))
            master_plot_fname = os.path.join(master_plot_path, name) + '.pdf'
            # sym link names. Simulation results are soft linked to the
            # user-requested sub_dir
            data_path  = make_path(**dir_params('data', defaults['sub_dir']))
            data_fname = os.path.join(data_path, name) + '.hdf5'
            plot_path  = make_path(**dir_params('plots', defaults['sub_dir']))
            plot_fname = os.path.join(plot_path, name) + '.pdf'

            # store names in params dict
            params['name'] = name
            params['data_fname'] = data_fname
            params['plot_fname'] = plot_fname

            # open hdf5 file, update its contents, and save
            h5file = h5py.File(master_data_fname)
            h5file, tasks_added = run_required(params, tasks, h5file)

            # run the plotting routine
            plot(params, h5file, plot_fname)

            # close the hdf5 file
            h5file.close()

            # link the results to the user-requested sub_dir
            symlink_force(master_data_fname, data_fname)

            # get simulation time
            t_elapsed = time.time() - t0

            # print a meta data summary of the simulation
            print(print_string(params, rank, tasks_added, t_elapsed))


# QCA implementation
# ==================

# top level simulation call. Simulation described by params is updated with
# tasks and saved to the open hdf5 file called `h5file`
def run_required(params, tasks, h5file, rewrite_meas=False, rewrite_time=False):
    # check for dependencies of user-requested tasks
    time_tasks = [task for task in tasks if task in implemented_time_tasks]
    meas_tasks = [task for task in tasks if task.split('-')[0] in
        implemented_meas_tasks]
    needed_time_tasks, needed_meas_tasks = check_deps(tasks, h5file)
    # run the necessary time tasks
    if rewrite_time:
        time_evolve(params, time_tasks, h5file)
    elif len(needed_time_tasks) > 0:
        time_evolve(params, needed_time_tasks, h5file)

    # run the necessary measurement tasks
    if rewrite_meas:
        measure(params, meas_tasks, h5file)
    elif len(needed_meas_tasks) > 0:
        measure(params, needed_meas_tasks, h5file)

    # collect the names of all new tasks
    tasks_added = needed_time_tasks + needed_meas_tasks
    # return the updated hdf5 file and the names of all new tasks
    return h5file, tasks_added

# run time tasks
def time_evolve(params, time_tasks, h5file):
    # initialize:
    # Python generator of time evolution
    time_step_gen = get_time_step_gen(params)
    # useful simulation parameters
    L, T, S = [params[key] for key in ['L', 'T', 'S']]
    # a list of args for initializing the numpy arrays for each task
    init_args = [(L, T)]*len(time_tasks)
    # a dictionary of numpy arrays of zeros, allocating memory for the results
    res = init_tasks(time_tasks, init_args, res={})

    # iterate through time evolution
    for t, next_state in enumerate(time_step_gen):
        # at each time step, run all the time tasks
        for task in time_tasks:
            # replace the zeros with the actual data (`set` the values)
            task_map[task]['set'](next_state, res[task], t)
    # get a list of newly added simulations (pop deletes a key from the dictionary
    # and returns the corresponding value)
    added_time_tasks = res.pop('tasks_tmp')

    # if the open hdf5 file already has some time tasks available, append the new
    # ones
    try:
        updated_time_tasks = np.append(
                h5file['time_tasks'][::], added_time_tasks)
        del h5file['time_tasks']
    # if there are no existing time tasks in the hdf5 file (i.e. this is a brand
    # new simulation), then the above `try` block will throw an error. We catch
    # the error and instead we dont ask the hdf5 file for its list of time tasks
    except:
        updated_time_tasks = added_time_tasks

    # set the new list of time tasks, whether they are appended to old ones, or
    # a brand new list
    h5file['time_tasks'] = updated_time_tasks

    # save the results dictionary to the hdf5 file
    save_dict_hdf5(res, h5file)
    # delete the results dictionary to free up some memory
    del res

# run measurement tasks
def measure(params, meas_tasks, h5file):
    # same comments as in time tasks above
    L, T, S = [params[key] for key in ['L', 'T', 'S']]
    init_args = [(L, T)]*len(meas_tasks)
    res = init_tasks(meas_tasks, init_args, res={})

    # notice we don't need to loop through the time evolution again. That is why
    # we split tasks into `time` and `meas` and always execute time tasks first.
    # We only do the heavy lifting once.

    # for each measurement task...
    for task in meas_tasks:
        # get the `slim task name` e.g. If the user requests exp-Z, we only need to know
        # its of type 'exp' and I dont need to know the 'Z' until we go to calculate
        # expectation values. This minimizes repeated code in the task_map.
        task_slim = task.split('-')[0]
        # same comments as in time tasks above
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
    del res

# get a generator of time evolved states
def get_time_step_gen(params):
    # make sure supplied parameters are valid
    L, T, V, r, S, M, IC, BC_type, BC_conf = validate_params(params)
    # make a Python generator of the valid time evolution:
    # initial condition
    state = ss.make_state(L, IC)
    # update mode
    mode_list = get_mode(M, L)
    # update unitaries
    lUs, U, rUs = get_Us(V, r, S, BC_conf)
    # make and return the Python generator
    time_step_gen = gen_time_step(
            state, U, lUs, rUs, mode_list, L, T, r, BC_type)
    return time_step_gen

# convert input mode (string) to spatial update ordering
def get_mode(mode_name, L):
    if mode_name in ('sweep', 'swp'):
        m = 1
    elif mode_name in ('alternate', 'alt'):
        m = 2
    elif mode_name in ('block', 'blk'):
        m = 3
    else:
        try:
            m = int(mode_name)
        except:
            raise ValueError('mode is not swp (d=1), alt (d=2), blk (d=3), or\
            convertible to an integer step size.')
    return make_mode_list(m, L)

# create spatial update ordering with skip size d for lattice length L
def make_mode_list(d, L):
    mode_list = []
    for delta in range(d):
        for j in range(L):
            if j % d == delta:
                mode_list.append(j)
    return mode_list

# gather update operators for boundaries and bulk
def get_Us(V, r, S, BC_conf):
    # Update unitary for bulk of the lattice
    U = make_U(V, r, S)
    # list of update unitaries for boundary sites
    bUs = make_boundary_Us(V, r, S, BC_conf)
    # left boundary unitaries
    lUs = bUs[0:r]
    # right boundary unitaries
    rUs = bUs[r:2*r + 1]
    # return all the update unitaries
    return lUs, U, rUs

# make bulk update operator
def make_U(V, r, S):
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2**N)[::-1]
    U = np.zeros((2**(N+1), 2**(N+1)), dtype=complex)
    for sig, s in enumerate(Sb): # 'sig' is for bit significance
        sigb = mx.dec_to_bin(sig, N)
        Vmat = get_V(V, s)
        ops = [mx.ops[str(op)] for op in sigb[0:r]] + [Vmat] +\
                [mx.ops[str(op)] for op in sigb[r:2*r+1]]
        U += mx.listkron(ops)
    return U

# make boundary update operators
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
                    ops = [mx.ops[str(op)] for op in var1] + [Vmat] +\
                            [mx.ops[str(op)] for op in var2]
                elif w == 1:
                    var = mx.dec_to_bin(i, r+c)[::-1]
                    var1 = var[c::][::-1]
                    var2 = var[0:c]
                    fixed = fixed_o[0:r-c:]
                    n =  var1 + var2 + fixed
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = [mx.ops[str(op)] for op in var1] + [Vmat] +\
                            [mx.ops[str(op)] for op in var2]
                U += mx.listkron(ops)
            bUs.append(U)
    return bUs

# make unitary, single-site op for use in QCA update scheme
def get_V(V, s):
    # make the 2x2 unitary matrix (an element of the group U(2)) requested by
    # the user with the `V` param (see matrix.py for more details)
    Vmat = mx.make_U2(V)
    # turn it on or off based on s (little s is a rule element of rule number S)
    return s*Vmat + (1-s)*mx.ops['I']

# single QCA iteration (one time step)
def iterate(state, U, lUs, rUs, mode_list, L, r, BC_type):
    # if box boundar conditioins are requested
    if BC_type == '1':
        # Loop through all sites in the order given by mode_list
        for j in mode_list:
            # make the neighborhood `Nj` and the appropriate unitary `u` for
            # left boundary sites, bulk sites, or right boundary sites:
            if j < r:           # left boundary
                Nj = range(0, j+r+1)
                u = lUs[j]
            elif L - j - 1 < r: # right boundary
                Nj = range(j-r, L)
                u = rUs[L-j-1]
            else:               # bulk
                Nj = range(j-r, j+r+1)
                u = U
            # In python 3, `range` returns a Python generator. Make it a list.
            Nj = list(Nj)

            # update the state. Lots of magic here! See matrix.py for more details.
            state = mx.op_on_state(u, Nj, state)

    # if ring boundar conditions are requested
    elif BC_type == '0':
        for j in mode_list:
            # use a % b := a mod b to make the neighborhoods live on a ring
            Nj = [k%L for k in range(j-r, j+r+1)]
            # update the state. Lots of magic here! See matrix.py for more details.
            state = mx.op_on_state(U, Nj, state)

    # return the updated state
    return state

# Update state with T iterations, yield each new state. Makes a Python generator
def gen_time_step(state, U, lUs, rUs, mode_list, L, T, r, BC_type):
    # first yield the initial state
    yield state
    # update the state `T` times. Yield each new state
    for t in range(T):
        state = iterate(state, U, lUs, rUs, mode_list, L, r, BC_type)
        yield state

# check validity of simulation parameters
def validate_params(params):
    arg_names = ['L', 'T', 'V', 'r', 'S', 'M', 'IC', 'BC']
    proper_types = [(int,), (int,), (str,), (int,), (int,), (str, int),
        (str, list), (str,)]
    L, T, V, r, S, M, IC, BC = [params[key] for key in arg_names]
    args = [L, T, V, r, S, M, IC, BC]
    given_types = list(map(type, args))
    mask = np.array([not t in pt for t, pt in zip(given_types, proper_types)])

    # check data type of each param
    if np.any(mask == True):
        def grab_bad(lst):
            return np.array(lst)[mask]

        bad_inputs = [bad for bad in map(grab_bad,
            (arg_names, args, given_types, proper_types))]

        for arg_name, bad_arg, bad_type, proper_type in zip(*bad_inputs):
            raise ValueError(
                'Argument {} is {} (type {}) should be of type {}'.format(
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
            raise ValueError('BC {} is not formatted properly'.format(BC))
        if len(BC_conf) != 2*r:
            raise ValueError('BC configuration {} is not valid'.format(BC_conf))
    else:
        raise ValueError('BC type {} is not understood'.format(BC[0]))
    return L, T, V, r, S, M, IC, BC_type, BC_conf

# summary of simulation ('\n' prints a new line)
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


# Control
# =======

# dictionary data structure for simulation parameters
# ---------------------------------------------------

# Get a list of simulation parameters. Ordering is defined by sim_param_keys
def get_args():
    # take in supplied command line arguments
    args = sys.argv[1:] # cut off the first element, which is the file name
    nargs = len(args)
    sim_param_keys = ('Ls', 'Ts', 'Vs', 'rs', 'Ss', 'Ms', 'ICs', 'BCs')
    all_keys = defaults.keys()
    # make sure user supplies either ZERO or ALL parameters
    if nargs not in (0, len(list(all_keys))):
        raise ValueError('incorrect number of arguments')
    else:
        # update the params dict as needed
        args_dict = make_args_dict(args)
        defaults.update(args_dict)
    # return a list of simulation parameters
    return [defaults[key] for key in sim_param_keys]


def make_args_dict(args):
    # list of all simulation parameters in a well defined order
    arg_names = ['Ls', 'Ts', 'Vs', 'rs', 'Ss', 'Ms', 'ICs', 'BCs',
                 'tasks', 'sub_dir', 'thread_as']
    args_dict = defaults
    for arg_name, arg in zip(arg_names, args):
        if not arg_name in ('sub_dir', 'thread_as'):
            arg = arg.split(' ')
            if arg_name in ('Ls', 'Ts', 'rs', 'Ss', 'Ms'):
                arg = list(map(int, arg))
        args_dict[arg_name] = arg
    return args_dict


# use method supplied by thread_as to make list of params_dict
# ------------------------------------------------------------

# top level wraper for_making params lists
def make_params_list(thread_as, Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    if thread_as in ('zip', 'cycle', 'zipcycle', 'zip_cycle', 'zip cycle'):
            params_list = zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    elif thread_as in ('power', 'set', 'powerset', 'power_set', 'power set'):
            params_list = power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)

    elif thread_as in ('repeat', 'last', 'repeatlast'
            'repeat_last' 'repeat last'):
            params_list = repeat_last_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    else:
        raise ValueError(
            'Argument thread_as was given {} and is neither zip nor power'.format(
            thread_as))
    return params_list

# make list of parms_dict by forming the power set of input lists
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

# make list of params_dict by zipping input lists (shorter inputs are cycled)
def zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    args = (Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    lens = [l for l in map(len, args) ]
    ind = np.argmax(lens)
    to_zip = [el for el in map(cycle, args)]
    to_zip[ind] = args[ind]
    # number of sims given by longest input list
    return [make_params_dict(*params) for params in zip(*to_zip)]

# make list of params_dict by zipping input lists (shorter inputs are cycled)
def repeat_last_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    args = (Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    lens = np.array([l for l in map(len, args) ])
    ind = np.argmax(lens)
    longest = lens[ind]
    pads = longest - lens
    to_zip = [arg + [arg[-1]] * pad for arg, pad in zip(args, pads)]
    # number of sims given by longest input list
    return [make_params_dict(*params) for params in zip_longest(*to_zip)]

# simulation parameter dictionary
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


# File I/O
# ========

# typ is 'data' or 'plots',
# make sure project_dir points to your local clone of the qops repo
def dir_params(typ, sub_dir):
    return {                              # location of this file
            'project_dir' : os.path.dirname(os.path.realpath(__file__)),
            'output_dir'  : 'qca_output', # appends to project_dir
            'sub_dir'     : sub_dir,      # appends to output_dir
            'typ'         : typ           # appends to sub_dir (plots or data)
            }

# unique name of simulation parameters
def make_name(L, T, V, r, S, M, IC, BC):
    name = 'L{}_T{}_V{}_r{}_S{}_M{}_IC{}_BC{}'.format(L, T, V, r, S, M, IC, BC)
    return name

# create and return a file path, makes it if it doesn't exist
def make_path(project_dir, output_dir, sub_dir, typ):
    path = os.path.join(project_dir, output_dir, sub_dir, typ)
    os.makedirs(path, exist_ok=True)
    return path

# symlink target to link_name, forces if it already exists
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

# save multipage pdfs
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


# recursive save dict to hdf5 file
# http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
def save_dict_hdf5(dic, h5file):
    recurs_save_dict_hdf5(h5file, '/', dic)

def recurs_save_dict_hdf5(h5file, path, dic_):
    for key, item in dic_.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            if path+key in h5file.keys():
                h5file[path + key][::] = item
            else:
                h5file[path + key] = item
        elif isinstance(item, dict):
            recurs_save_dict_hdf5(h5file, path + key + '/', item)

        # special case for bipartition density matricies only
        elif isinstance(item, list):
            item_T = [[item[j][i] for j in range(len(item))] for i in
                range(len(item[0]))]
            for k, el in enumerate(item_T):
                if path + key + '/c' + str(k) in h5file.keys():
                    h5file[path + key + '/c' + str(k)][::] = el
                else:
                    h5file[path + key + '/c'+str(k)] = el
        else:
            raise ValueError('Cannot save %s type'%item)

# recursive load dict from hdf5 file
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

# functions for dependencies of tasks
def dep_g2(task):
    _, ops = task.split('-')
    deps = ['exp2-'+ops, 'exp-'+ops[0], 'exp-'+ops[1]]
    return deps
#
def dep_FT(task):
    _, meas = task.split('-')
    if meas in ('D', 'C', 'Y'):
        deps = [meas]
    elif meas in ('x', 'y', 'z'):
        deps = ['exp-' + meas]
    else:
        raise ValueError('unknown Fourier transform method for measure \
            {}'.format())
    return deps
#
def dep_MI(task):
    return ['MI']
#
def dep_s_s2(task):
    return ['s', 's2']
#
def dep_one_site(task):
    return ['one_site']
#
def dep_two_site(task):
    return ['two_site']

# functions allocating memory for requested results
def init_one_site(L, T):
    return np.zeros((T+1, L, 2, 2), dtype=complex)
#
def init_two_site(L, T):
    return np.zeros((T+1, int(0.5*L*(L-1)), 4, 4), dtype=complex)
#
def init_bipartitions(L, T):
    N = L - 1
    c = int(N/2)
    left_dims = [2**(l+1) for l in range(c)]
    right_dims = left_dims
    if N % 2 != 0:
        right_dims = np.append(right_dims, 2**(c+1))
    dims = np.append(left_dims, right_dims[::-1])
    init_shape = [[np.zeros((d,d), dtype=complex) for d in dims]]*(T+1)
    return init_shape
#
def init_center_partition(L, T):
    return np.zeros((T+1, 2**int(L/2), 2**int(L/2)), dtype=complex)
#
def init_sbond(L, T):
    return np.zeros((T+1, L-1))
#
def init_scalar(L, T):
    return np.zeros(T+1)
#
def init_FT(L, T):
    offset = 0
    if T > 300:
        offset = 300
    return np.zeros(((T - offset + 1) // 2 + 1, 3))
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
def set_bipartitions(state, bipartitions, t):
    bipartitions[t][::][::][::] = ms.get_bipartition_rhos(state)
#
def set_center_partition(state, center_partition, t):
    center_partition[t, ::, ::] = ms.get_center_rho(state)
#
def set_sbond(state, sbond, t):
    L = int(log(len(state), 2))
    N = L - 1
    R = int(N/2)
    norm = [min(c + 1, L - c - 1) for c in range(N)]
    sbond[t, ::] = ms.get_bipartition_entropies(state)/norm
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
        exp[t, ::] = ms.local_exp_vals_from_rhos(rhos, mx.ops[op.upper()])

def set_FT(h5file, res, task):
    FT = res[task]
    _, meas = task.split('-')
    deps = task_map['FT']['deps'](task)
    sig, = gather_deps(h5file, res, deps)
    # NOTE: Can't FT <Y> because name is degenerate with disparity
    if meas in ('x', 'y', 'z'):
        sig = np.mean(sig, axis = 1)
    FT[::,::] = ms.fourier(sig).T

def set_exp2(h5file, res, task):
    exp2 = res[task]
    _, ops = task.split('-')
    deps = task_map['exp2']['deps'](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        exp2[t, ::, ::] = ms.exp2_vals_from_rhos(
                rhos, mx.ops[ops[0].upper()], mx.ops[ops[1].upper()])

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

def set_D(h5file, res, task):
    D = res[task]
    deps = task_map['D']['deps'](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        D[t] = ms.network_density(mat)

def set_C(h5file, res, task):
    C = res[task]
    deps = task_map['C']['deps'](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        C[t] = ms.network_clustering(mat)

def set_Y(h5file, res, task):
    Y = res[task]
    deps = task_map['Y']['deps'](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        Y[t] = ms.network_disparity(mat)

def set_g2(h5file, res, task):
    g2 = res[task]
    deps = task_map['g2']['deps'](task)
    exp2s, exp1as, exp1bs = gather_deps(h5file, res, deps)
    for t, (exp2, exp1a, exp1b) in enumerate(zip(exp2s, exp1as, exp1bs)):
        g2[t,::, ::] = ms.g2_from_exps(exp2, exp1a, exp1b)

def name_exp(task):
    op = task.split('-')[1]
    title = r'$\langle \sigma^{%s} \rangle$' % op.lower()
    xlabel = 'Site'
    ylabel = 'Iteration'
    return title, xlabel, ylabel

def name_FT(task):
    typ, meas = task.split('-')
    title = ''
    xlabel = 'Frequency'
    ylabel = r'$\mathcal{F}$(%s)' % meas
    # TODO: fix slim name degeneracy of disparity Y and expectation <Y>
    if meas in ('x', 'y', 'z'):
       ylabel = r'$\mathcal{F}\left(\overline{\langle \sigma^{%s}\rangle}\right)$' % meas.lower()
    return title, xlabel, ylabel

def name_s(task):
    title = r'$s^{\mathrm{vN}}$'
    xlabel = 'Site'
    ylabel = 'Iteration'
    return title, xlabel, ylabel

def name_sbond(task):
    title = r'$s^{\mathrm{bond}}$'
    xlabel = 'Cut'
    ylabel = 'Iteration'
    return title, xlabel, ylabel

# maping task names to their dep, init, and set functions
implemented_time_tasks = ['one_site', 'two_site', 'bipartitions',
    'center_partition', 'sbond', 'scenter']
implemented_meas_tasks = ['exp', 'exp2', 's', 's2', 'MI', 'g2', 'D', 'C', 'Y',
    'FT']

task_map = {
    # time tasks have no dependencies
    'one_site' : {
        'init' : init_one_site,
        'set'  : set_one_site
            },

    'two_site' : {
        'init' : init_two_site,
        'set'  : set_two_site
            },

    'bipartitions' : {
        'init' : init_bipartitions,
        'set'  : set_bipartitions
            },

    'center_partition' : {
        'init' : init_center_partition,
        'set'  : set_center_partition
            },

    'sbond'    : {
        'init' : init_sbond,
        'set'  : set_sbond,
        'name' : name_sbond
            },

    'scenter'  : {
        'init' : init_scalar,
        'set'  : set_scenter
            },

    # meas tasks may depend on time or other meas tasks
    's'    : {
        'deps' : dep_one_site,
        'init' : init_vec,
        'set'  : set_s,
        'name' : name_s
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
        'deps' : dep_s_s2,
        'init' : init_mat,
        'set'  : set_MI
            },

    'D'   : {
        'deps' : dep_MI,
        'init' : init_scalar,
        'set'  : set_D
            },

    'C'   : {
        'deps' : dep_MI,
        'init' : init_scalar,
        'set'  : set_C
            },

    'Y'   : {
        'deps' : dep_MI,
        'init' : init_scalar,
        'set'  : set_Y
            },

    'exp'  : {
        'deps' : dep_one_site,
        'init' : init_vec,
        'set'  : set_exp,
        'name' : name_exp
            },

    'exp2' : {
        'deps' : dep_two_site,
        'init' : init_mat,
        'set'  : set_exp2
            },

    'FT'  : {
        'deps' : dep_FT,
        'init' : init_FT,
        'set'  : set_FT,
        'name' : name_FT

            },
        }

# Init a set of tasks as a dictionary
def init_tasks(tasks, argss, res={}):
    for task, args in zip(tasks, argss):
        task_slim = task.split('-')[0]
        res[task] = task_map[task_slim]['init'](*args)
    res['tasks_tmp'] = np.array(tasks).astype('|S9')
    return res

# gather a list of required dependencies, same order as given by deps
def gather_deps(h5file, res, deps):
    new_avail = [k for k in res.keys()]
    old_avail = [k for k in h5file.keys()]
    out = []
    for dep in deps:
        if dep in new_avail:
            out += [res[dep]]
        elif dep in old_avail:
            out += [h5file[dep][::]]
        else:
            raise ValueError('Dependency error, \'{}\' not available'.format(dep))
    return out

def check_deps(tasks, h5file):
    # separate tasks into time and meas
    time_tasks = [task for task in tasks if task in implemented_time_tasks]
    meas_tasks = [task for task in tasks if task.split('-')[0] in implemented_meas_tasks]
    avail_time_tasks, avail_meas_tasks = get_avail_tasks(h5file)
    # tasks not yet preformed plus their dependencies
    required_time_tasks, required_meas_tasks = recurs_check_deps(
            time_tasks, meas_tasks,
            avail_time_tasks, avail_meas_tasks)

    # remove any required taskes that are already available
    needed_time_tasks = [task for task in required_time_tasks if not task in
            avail_time_tasks]
    needed_meas_tasks = [task for task in required_meas_tasks if not task in
            avail_meas_tasks]
    return needed_time_tasks, needed_meas_tasks

# recursivly check dependencies
def recurs_check_deps(time_tasks, meas_tasks, avail_time_tasks, avail_meas_tasks,
        requested_meas_tasks=[], requested_time_tasks=[],
        required_meas_tasks=[]):
    for meas_task in meas_tasks:
        meas_task_slim = meas_task.split('-')[0]
        meas_deps = ['']
        requested_meas_tasks=[]
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
                                required_meas_tasks=required_meas_tasks)
            else:
                raise ValueError('requested task {} is not implemented'.format(dep))
        required_meas_tasks += requested_meas_tasks[::-1] + [meas_task]
    required_time_tasks = time_tasks + requested_time_tasks
    #required_meas_tasks = meas_tasks + requested_meas_tasks
    return required_time_tasks, unique(required_meas_tasks)


# get a list of existing time and meas tasks from an hdf5 file
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

# get unique elements of seq, preserving ordering (like set, but set reorders)
def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# plotting (currently developing)
# ========

def plot_name(fignum, params):
    name = params['name']
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1,1,1)
    ax.text(0.05, 0.5, name)
    ax.axis([0,1,0,1])
    ax.axis('off')

def plot(params, h5file, plot_fname):
    tasks = [k for k in h5file.keys()]
    L, T = params['L'], params['T']
    exp_tasks = [task for task in tasks if task.split('-')[0] == 'exp']
    s_tasks = [task for task in tasks if task in ('s', 'sbond')]
    nm_tasks = [task for task in tasks if task in ('D', 'C', 'Y')]
    FT_nm_tasks = [task for task in tasks if task in ('FT-D', 'FT-C', 'FT-Y',
    'FT-x', 'FT-y','FT-z')]
    plot_name(1, params)
    if len(exp_tasks) > 0:
        plot_vecs(2, exp_tasks, h5file)
    if len(s_tasks) > 0:
        plot_vecs(3, s_tasks, h5file, zspan=[0, 1])
    if 'scenter' in h5file.keys():
        plot_scenter(4, h5file)
    if len(nm_tasks) > 0:
        plot_network_measures(5, nm_tasks, h5file)
    if len(FT_nm_tasks) > 0:
        plot_FT_network_measures(6, T, FT_nm_tasks, h5file)
    plt.tight_layout()
    multipage(plot_fname)

def plot_scenter(fignum, h5file):
        fig = plt.figure(fignum)
        ax = fig.add_subplot(1,1,1)
        scenter = h5file['scenter'][::]
        plot_scalar(ax, scenter, '', 'Iteration', r'$s^{\mathrm{center}}$')

def plot_vecs(fignum, vec_tasks, h5file, xspan=None, yspan=None, zspan=None):
    fig = plt.figure(fignum)
    nax = len(vec_tasks)
    for task_id, task in enumerate(vec_tasks):
        ax = fig.add_subplot(1, nax, task_id + 1)
        vec = h5file[task][::]
        T, L = vec.shape
        if xspan is None: xspan = [0, L]
        if yspan is None: yspan = [0, min(61, T)]
        if zspan is None: zspan = [-1, 1]
        title, xlabel, ylabel = task_map[task.split('-')[0]]['name'](task)
        ax = plot_vec(ax, vec, '', xlabel, ylabel, title, xspan, yspan,
                zspan)

def plot_vec(ax, vec, title, xlabel, ylabel, zlabel, xspan, yspan, zspan):
    cax = ax.imshow(vec[yspan[0]:yspan[1], xspan[0]:xspan[1]],
            interpolation='None',
            origin='lower')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ticks = [zspan[0], sum(zspan)/2, zspan[1]]
    ticks=None
    cbar = plt.colorbar(cax, ticks=ticks,
            shrink=0.75, pad=0.08)
    cbar.ax.set_title(zlabel, y=1.01)
    return ax

def plot_scalar(ax, scalar, title, xlabel, ylabel,
        xs=None, xspan=None, ymin=None, label='', logy=False, kwargs={}):
    T = len(scalar)
    if xspan is None: xspan = [0, min(61, T)]
    if xs is None:
        xs = np.linspace(xspan[0], xspan[1], int(xspan[1] - xspan[0]))
        scalar = scalar[xspan[0]:xspan[1]]
    if logy:
        ax.semilogy(xs[1::], scalar[1::], label=label, **kwargs)
    else:
        ax.plot(xs, scalar, label=label, **kwargs)
    if not ymin is None:
        ax.set_ylim(bottom=ymin)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_network_measures(fignum, nm_tasks, h5file, xspan=None):
    fig = plt.figure(fignum)
    nax = len(nm_tasks)
    for task_id, task in enumerate(nm_tasks):
        ax = fig.add_subplot(nax, 1, task_id + 1)
        nm = h5file[task][::]
        name = r'$\mathcal{%s}$' % task
        plot_scalar(ax, nm, '', 'Iteration', name)

def peak_find(XX, YY, YYlima, YYlimb):
    Y = YY[YY>YYlima]
    X = XX[YY>YYlima]
    Ylimb = YYlimb[YY>YYlima]
    dx = XX[1] - XX[0]
    I = np.arange(len(X)-1)
    inds = I[abs(X[:-1] - X[1:]) > dx]
    Xs = np.split(X, inds)
    Ys = np.split(Y, inds)
    Ylimbs = np.split(Ylimb, inds)
    xpks = []
    ypks = []
    for x, y, ylimb in zip(Xs, Ys, Ylimbs):
        if len(x) > 0:
            ymx = np.max(y)
            xmx = x[np.argmax(y)]
            ylb = ylimb[np.argmax(y)]
            if ymx > ylb:
                ypks.append(ymx)
                xpks.append(xmx)
    return xpks, ypks

def plot_FT_network_measures(fignum, T, FT_nm_tasks, h5file, xspan=None):
    fig = plt.figure(fignum)
    nax = len(FT_nm_tasks)
    for task_id, task in enumerate(FT_nm_tasks):
        ax = fig.add_subplot(nax, 1, task_id + 1)
        fs = h5file[task][::, 0]
        FT_nm = h5file[task][::, 1]
        RN_nm = h5file[task][::, 2]
        chi22_RN_nm = 5.991 * RN_nm
        fpks, ppks = peak_find(fs, FT_nm, RN_nm, chi22_RN_nm)
        title, xlabel, ylabel = task_map[task.split('-')[0]]['name'](task)
        if task_id != nax - 1:
            xlabel = ''
            ax.set_xticks([])

        ymin = min([RN_nm[0], RN_nm[-1]])
        plot_scalar(ax, FT_nm, title, xlabel, ylabel, xs=fs, logy=True,
            ymin=ymin, kwargs={})
        plot_scalar(ax, RN_nm, title, xlabel, ylabel, xs=fs, logy=True,
            ymin=ymin, kwargs={'ls':'--'})
        plot_scalar(ax, chi22_RN_nm, title, xlabel, ylabel, xs=fs, logy=True,
            ymin=ymin, kwargs={'ls':':'})
        ax.scatter(fpks, ppks, c='r', marker='*', linewidth=0,
                s=35, zorder=3)

# execute default behavior
# ------------------------
if __name__ == '__main__':
    main()
