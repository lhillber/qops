#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Functions for matrix manipulations of quantum states/operators
#
# by Logan Hillberry
# ===============================================================


from math import log
from cmath import sqrt, sin, cos, exp, pi
from functools import reduce
import numpy as np
import scipy.sparse as sps

# global dictionary of common 2x2 operators
ops = {
        'H' : 1.0 / sqrt(2.0) * \
              np.array( [[1.0,  1.0 ],[1.0,  -1.0]], dtype=complex),
        'I' : np.array( [[1.0,  0.0 ],[0.0,   1.0]], dtype=complex ),
        'X' : np.array( [[0.0,  1.0 ],[1.0,   0.0]], dtype=complex ),
        'Y' : np.array( [[0.0, -1.0j],[1.0j,  0.0]], dtype=complex ),
        'Z' : np.array( [[1.0,  0.0 ],[0.0 , -1.0]], dtype=complex ),
        'S' : np.array( [[1.0,  0.0 ],[0.0 , 1.0j]], dtype=complex ),
        'T' : np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*pi/4.0)]], dtype=complex),
        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
      }


# apply k-site-wide op to a list of k sites (js) of to the state-vector state.
# ds is a list of local dimensions for each site of state, and a
# lattice of qubits (d=2) is assumed if ds is not provided.
# -------------------------------------------------------------------------------
def op_on_state(op, js, state, ds=None):
    if ds is None:
        L = int( log(len(state), 2) )
        ds = [2]*L
    else:
        L = len(ds)
    dn = np.prod(np.array(ds).take(js))
    dL = np.prod(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = list(rest) + list(js)
    state = state.reshape(ds).transpose(ordering)\
            .reshape(int(dL/dn), dn).dot(op).reshape(ds)\
            .transpose(np.argsort(ordering)).reshape(dL)
    return state


# mememory intensive method, builds full rank local operator
# ----------------------------------------------------------
def op_on_state_2(local_op_list, js, state):
    L = int( log(len(state), 2) )
    I_list = [np.eye(2.0, dtype=complex)]*L
    for j, local_op in zip(js, local_op_list):
        I_list[j] = local_op
    big_op = listkron(I_list)
    return big_op.dot(state)


# partial trace of a state vector, js are the site indicies kept
# --------------------------------------------------------------
def rdms(state, js, ds=None):
    js = np.array(js)
    if ds is None:
        L = int( log(len(state), 2) )
        ds = [2]*L
    else:
        L = len(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = np.concatenate((js, rest))
    dL = np.prod(ds)
    djs = np.prod(np.array(ds).take(js))
    drest = np.prod(np.array(ds).take(rest))
    block = state.reshape(ds).transpose(ordering).reshape(djs, drest)
    RDM = np.zeros((djs, djs), dtype=complex)
    for i in range(djs):
        for j in range(i, djs):
            Rij = np.inner(block[i,:], np.conj(block[j,:]))
            RDM[i, j] = Rij
            RDM[j, i] = np.conj(Rij)
    return RDM


# TODO: BROKEN! fix it.
# partial trace of density matrix rho, keeping sites with indices js
# ------------------------------------------------------------------
def rdmr(rho, js):
    print('WARNING: this function is NOT verified')
    L = int(log(len(rho), 2))
    d = 2*L

    n = len(js)
    js = list(js)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = js+list(rest)
    orderingout = [o + L for o in ordering]

    ordering = ordering + orderingout
    block = rho.reshape(([2]*d)).transpose(ordering).reshape(2**(d-2*n), 2**n, 2**n)
    print(block)
    RDM = np.trace(block)
    return RDM/np.trace(RDM)


# Thanks to Daniel Jaschke for the following two methods
# partial trace of density marix rho, removing the first site
# -----------------------------------------------------------
def traceout_first(rho, ld=2):
    L = int(log(len(rho), 2))
    dim = ld**(L - 1)
    rho2L = np.zeros((dim, dim), dtype=rho.dtype)
    rho = np.reshape(np.transpose(np.reshape(rho, (ld, dim, ld, dim)),
                                  (1, 3, 0, 2)), (dim, dim, ld**2))
    mask = np.cumsum((ld + 1) * np.ones(ld, dtype=int)) - (ld + 1)
    for kk in range(dim):
        for jj in range(dim):
            rho2L[kk, jj] = np.sum(rho[kk, jj, mask])
    return rho2L


# partial trace of density marix rho, removing the last site
# ----------------------------------------------------------
def traceout_last(rho, ld=2):
    L = int(log(len(rho), 2))
    dim = ld**(L - 1)
    rho1n = np.zeros((dim, dim), dtype=rho.dtype)
    rho = np.reshape(np.transpose(np.reshape(rho, (dim, ld, dim, ld)),
                                  (0, 2, 1, 3)), (dim, dim, ld**2))
    mask = np.cumsum((ld + 1) * np.ones(ld, dtype=int)) - (ld + 1)
    for kk in range(dim):
        for jj in range(dim):
            rho1n[kk, jj] = np.sum(rho[kk, jj, mask])
    return rho1n

# convert input V (string) to local unitary (2d numpy array)
def make_U2(V):
    # split input string (V) into op string (Vs) and angle string (angs) at underscore
    Vs_angs = V.split('_')
    if len(Vs_angs) == 2:
        Vs, angs = Vs_angs
        # further split angle string at dashes into individual angles
        angs = angs.split('-')
    elif len(Vs_angs) == 1:
        Vs, angs = Vs_angs[0], []
    # get the indices of the string Vs which require an angle parameter. Right
    # now this supports 'P' for phase gates and 'R' for orthogonal matrices, and
    # 'p' for global phase
    ang_inds = [i for i, v in enumerate(Vs) if v in ('P', 'R', 'p')]

    # make sure the user supplies enough angles for the requested ops
    if len(angs) != len(ang_inds):
        raise ValueError('improper V configuration {}:\
                need one angle for every P, R, and p'.format(V))

    # initialize a counter to track which angle-needing op we are currently
    # constructing in the for loop
    ang_id = 0
    # initialize the 2x2 unitary as the identity (becomes the final result)
    Vmat = np.eye(2, dtype=complex)
    # for each requested op, do...
    for v in Vs:
        # if a phase gate is requested make it with the current angle and update
        # the result
        if v == 'P':
            # get the ang_idth angle of the list angs
            ang = angs[ang_id]
            # convert string angle in degrees to float angle in radians
            ang_in_rad = string_deg2float_rad(ang)
            # get a phase gate with the current angle
            Pmat = make_Pmat(ang_in_rad)
            # update the result by RIGHT multiplying by the current phase gate
            Vmat  = Vmat.dot(Pmat)
            # increment the angle counter
            ang_id += 1

        # if orthogonal gate is requested make it with the current angle
        elif v == 'R':
            # make orthoganal matrix with the current angle
            ang = angs[ang_id]
            ang_in_rad = string_deg2float_rad(ang)
            Rmat = make_Rmat(ang_in_rad)
            # update result and angle counter
            Vmat = Vmat.dot(Rmat)
            ang_id += 1

        # if a global phase is requested, make it
        elif v == 'p':
            ang = angs[ang_id]
            ang_in_rad = string_deg2float_rad(ang)
            global_phase = make_global_phase(ang_in_rad)
            Vmat = global_phase * Vmat
            ang_id += 1

        # if the requested op does NOT take angle parameter...
        else:
            # try to pull it from the global ops dictionary
            try:
                Vmat = Vmat.dot(ops[v])
            # if that fails, raise an error
            except:
                raise ValueError('string op {} not understood'.format(v))
    # return the final result
    return Vmat



# make a phase gate
def make_Pmat(ang_in_rad):
    return np.array([ [1.0,  0.0 ], 
                      [0.0 , exp(1.0j*ang_in_rad)] ],
                      dtype=complex)

# make an orthogonal matrix
def make_Rmat(ang_in_rad):
    return np.array([ [cos(ang_in_rad/2),  -sin(ang_in_rad/2) ],
                      [sin(ang_in_rad/2) , cos(ang_in_rad/2)] ],
                      dtype=complex)

# make a global phase
def make_global_phase(ang_in_rad):
    return  exp(1.0j*ang_in_rad)

# convert a string of an angle in degrees to a float in radians
def string_deg2float_rad(string_deg):
    float_rad = eval(string_deg)*pi/180.0
    return float_rad

# Hermitian conjugate
# -------------------
def dagger(rho):
    return np.transpose(np.conj(rho))


# check if a matrix U is unitary
# ------------------------------
def isU(U):
    m,n = U.shape
    Ud = np.conjugate(np.transpose(U))
    UdU = np.dot(Ud, U)
    UUd = np.dot(U, Ud)
    I = np.eye(n, dtype=complex)
    if np.allclose(UdU, I):
        if np.allclose(UUd, I):
            return True
    else:
        return False

# check density matrix is Hermitian
# ---------------------------------
def isherm(rho):
    if np.allclose(rho, dagger(rho)):
        return True
    else:
        return False

def issymm(mat):
    matT = mat.T
    if np.allclose(mat, matT):
        return True
    else:
        return False

# check density matrix has trace one
# ----------------------------------
def istrace_one(rho):
    tr = np.trace(rho)
    if tr == 1.0:
        return True
    else:
        return False

# check density matrix is positive
# --------------------------------
def ispos(rho):
    evals = sp.linalg.eigvalsh(rho)
    if np.all(evals >= 0.0):
        return True
    else:
        return False

# Kroeneker product list of matrices
# ----------------------------------
def listkron(matlist):
    return reduce(lambda A,B: np.kron(A, B), matlist)


# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron(matlist):
    return sps.csc_matrix(reduce(lambda A, B: sps.kron(A,B,'csc'),matlist)) 


# dot product list of matrices
# ----------------------------
def listdot(matlist):
    return reduce(lambda A, B: np.dot(A, B), matlist)


# replace small elements in an array
# ----------------------------------
def edit_small_vals(mat, tol=1e-14, replacement=0.0):
    if not type(mat) is np.ndarray:
        mat = np.asarray(mat)
    mat[mat<=tol] = replacement
    return mat


# concatinate two dictionaries (second arg replaces first if keys in common)
# --------------------------------------------------------------------------
def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


# concatinate a list of dictionaries
# ----------------------------------
def listdicts(dictlist):
    return reduce(lambda d1, d2: concat_dicts(d1, d2), dictlist)


# convert n in base-10 to base-2 with count digits
# ------------------------------------------------
def dec_to_bin(n, count):
     return [(n >> y) & 1 for y in range(count-1, -1, -1)]

# convert n in base-2 to base-10
# ------------------------------
def bin_to_dec(n):
    return int(''.join(list(map(str, n))), 2)

# sparse matrix tensor product (custom, just for fun)
# ---------------------------------------------------
def tensor(A, B):
    a_nrows, a_ncols = A.shape
    b_nrows, b_ncols = B.shape
    m_nrows, m_ncols = a_nrows*b_nrows, a_ncols*b_ncols
    b = list(zip(B.row, B.col, B.data))
    a = list(zip(A.row, A.col, A.data))
    M = np.zeros((m_nrows, m_ncols))
    for a_row, a_col, a_val in a:
        for b_row, b_col, b_val in b:
            row = a_row * b_nrows + b_row
            col = a_col * a_ncols + b_col
            M[row, col] = a_val * b_val
    return M


if __name__ == '__main__':

    L = 7
    IC = 'f0-3-4_t90-p90'

    js = [0,2,3]
    op = listkron( [ops['X']]*(len(js)-1) + [ops['H']] ) 

    print()
    print('op = XXH,', 'js = ', str(js)+', ', 'IC = ', IC)
    print()

    init_state3 = ss.make_state(L, IC)
    init_rj = [rdms(init_state3, [j]) for j in range(L)]
    init_Z_exp = [np.trace(r.dot(ops['Z']).real) for r in init_rj]
    init_Y_exp = [np.trace(r.dot(ops['Y']).real) for r in init_rj]
    print('initial Z exp vals:', init_Z_exp)
    print('initial Y exp vals:', init_Y_exp)

    final_state = op_on_state(op, js, init_state3)

    final_rj = [rdms(final_state, [j]) for j in range(L)]
    final_Z_exp = [np.trace(r.dot(ops['Z'])).real for r in final_rj]
    final_Y_exp = [np.trace(r.dot(ops['Y'])).real for r in final_rj]
    print('final Z exp vals:', final_Z_exp)
    print('final Y exp vals:', final_Y_exp)

