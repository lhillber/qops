#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# ==============================================================================
#
# Description:
# ============
#
# Enable the creation of 2x2 unitary matrices i.e., elements of the group U(2),
# from simple string specifications. Written in Python 3. We make use of the
# decomposition of 2x2 unitaries (https://en.wikipedia.org/wiki/Unitary_matrix)
#
#                           p(th0) P(th1) R(th2) P(th3)
#
# where p is a global phase, P is a phase gate, and R is an orthogonal matrix.
# Also supported is the use of constant matrices (no free parameters like an
# angle) such as the Pauli matrices and Hadamard gate.
#
# See the global dictionary called ops and the functions make_global_phase,
# make_Rmat, and make_Pmat, to change the definitions of the constant matrices,
# global phase, orthogonal matrices, and phase gates.
#
#
# Usage:
# ======
#
# Change the input string in this file, called V, to whatever you like. Then
# from the command line, while in the directory containing this file, execute,
#
#                            python3 U2_string_op.py
#
#
# The input string:
# =================
#
# The parameter V, the input string, must be organized as follows:
#
#                   <ops_string>_<ang0>-<ang1>-...-<angN-1>
#
# where ops_string is a string of characters which represent 2x2 matrices which
# will be multiplied together to from the unitary matrix. The op_string is read
# left to right and each new op is RIGHT multiplied onto the previous result.
# The dash-separated sequence of angles (assumed to be given in degrees) is also
# read left to right and each angle is applied once to an op in op_string which
# needs an angle parameter, e.g. p, P, or R.
#
# For example, V = 'pHRZP_90-45-10' returns p(90) H R(45) Z P(10).
#
#
# By Logan Hillberry
#
# ==============================================================================

import numpy as np
from cmath import exp
from math import sin, cos, sqrt, pi

# input string
# ------------
V = 'pPRP_45-90-45-180' # random example
#V = 'HP_45'        # backwards compatible with qeca style
#V = 'RP_90-225'    # same as 'HP_45', since H = RP_45-180

# default behavior
def main(V):
    # make the matrix
    Vmat = get_V(V)
    # check that it is unitary
    U_result = isU(Vmat)
    # print the results
    print('input string \'{}\' gives'.format(V))
    print(Vmat)
    print('which is unitary: {}'.format(U_result))
    return Vmat

# Main algorithm
# convert input V (string) to local unitary (2d numpy array)
# TODO: More clever algorithm without the if statements?
def get_V(V):
    # split input string (V) into op string (Vs) and angle string (angs) at underscore
    Vs, angs = V.split('_')
    # further split angle string at dashes in individual angles
    angs = angs.split('-')
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

# Helper functions
# ----------------
# global dictionary of useful ops with no free parameters
ops = {
        'H' : 1.0 / sqrt(2.0) * \
              np.array( [[1.0,  1.0 ],[1.0,  -1.0]], dtype=complex),

        'I' : np.array( [[1.0,  0.0 ],[0.0,   1.0]], dtype=complex ),
        'X' : np.array( [[0.0,  1.0 ],[1.0,   0.0]], dtype=complex ),
        'Y' : np.array( [[0.0, -1.0j],[1.0j,  0.0]], dtype=complex ),
        'Z' : np.array( [[1.0,  0.0 ],[0.0 , -1.0]], dtype=complex ),

        'S' : np.array( [[1.0,  0.0 ],[0.0 , 1.0j]], dtype=complex ),
        'T' : np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*pi/4.0)]], dtype=complex ),

        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
      }


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

# check if a matrix U is unitary
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


# Bonus function! This is useful to "turn on/off" a single qubit unitary for
# quantum cellular automata applications. s = 1 means 'on' s = 0 means 'off'
def V_for_qca(Vmat, s):
    return s*Vmat + (1-s)*ops['I']


# run default behavior when this file is executed
if __name__ == '__main__':
    main(V)

