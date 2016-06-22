#!/usr/bin/python3
#
# =============================================================================
# This file is used to store useful matrices as a global constants in a
# dictionary. It also enables the creation of quantum states. The make_state
# function takes a lattice size L and a state specification IC, which is either
# a string or a list of tuples. The List of tuples is for global superpositions:
# each tuple contains a coefficient and a state specification string.
#
# A state specification string starts with a single letter corresponding to a
# function in this file (it's a key in the dictionary called smap below).
# Lowercase keys are for separable states while capital keys are for entangled
# states. Everything after that first letter is a configuration. Underscores
# separate different config sections and dashes separate params within a config
# section:
#
#   function  | key |             config               | spec. string example
# ------------+-----+----------------------------------+------------------------
#             |     |                                  | 'f0-3_t90-p0_t45_p180'
#     fock    |  f  |<i-j-k...>_t<th>-p<ph>_t<th>-p<ph>| 'f2_t90-p90'
#             |     |                                  | 'f0-2-4-6'
# ------------+-----+----------------------------------+------------------------
#             |     |                                  | 'st90-P1'
#  spin_wave  |  s  |    T<n> OR t<th>-P<m> OR p<ph>   | 'sT2-p30'
#             |     |                                  | 'sT2-P1'
# ------------+-----+----------------------------------+------------------------
#  rand_state |  r  | <p>-<s>_t<th>-p<ph>_t<th>-p<ph>  | 'r75_t45-p90'
#             |     |                                  | 'r5-234_t45-p90'
# ------------+-----+----------------------------------+------------------------
#  rand_throw |  R  |               <s>                | 'R234'
# ------------+-----+----------------------------------+------------------------
#     Bell    |  B  |            <j-k>_<b>             | 'B0-1_3'
# ------------+-----+----------------------------------+------------------------
#     GHZ     |  G  |                NA                | 'G'
# ------------+-----+----------------------------------+------------------------
#      W      |  W  |                NA                | 'W'
# ------------+-----+----------------------------------+------------------------
#             |     |                                  | 'c1_f0'
#   center    |  c  |           <Lc>_<IC>              | 'c4_W'
#             |     |                                  | 'c2_r'
#
# Convention:  |0> = (1, 0) is at the top of the Bloch sphere.
#              |1> = (0, 1) is at the bottom of the Bloch sphere.
#
# Description of config sections:
#   + fock: a fock state of qubits
#       + section 1, <i-j-k...>: site indices of excitation
#       + section 2, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                excitation qubits (default t180_p0 if not given)
#       + section 3, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                background qubits (default t0_p0 if not given)
#
#   + spin_wave: fock states with twists in theta and/or phi across the lattice
#       + section 1, t<th> (p<ph>) holds theta (phi) constant at th (ph)
#                    T<n> (P<m>) winds theta (phi) n (m) times
#
#   + rand_state: a random fock state:
#       + section 1, <p>: probability of excitation at each site expressed as an
#                         int. That is, p=75 means prop of 3/4 for an excitation
#                    <s>: OPTIONAL - seed for random number generator
#       + sections 2 and 3, same as sections 2 and 3 in fock above
#
#   + rand_throw: random qubits:
#       + section 1, <s>: seed for random number generator
#
#   + Bell: a member of the Bell basis embedded in the lattice
#       + section 1 <j-k>: two site indices to share the bell state
#       + section 2 <b>: specify which Bell state according to b=0, 1, 2, or 3.
#                       0 : 1/sqrt 2 (|00>+|11>)
#                       1 : 1/sqrt 2 (|00>-|11>)
#                       2 : 1/sqrt 2 (|01>+|10>)
#                       3 : 1/sqrt 2 (|01>-|10>)
#
#   + center: embed any IC into the center of the lattice
#       + section 1 <LC>, the length of the central region. <IC> some other
#         IC spec
#
#
# By Logan Hillberry
# =============================================================================


from cmath import sqrt, sin, cos, exp, pi
import numpy as np
import matrix as mx

# Global constants
# ================

# dictionary of local operators and local basis (b for basis)
# -----------------------------------------------------------

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



brhos = {
        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
        }

bvecs = {
        '0'  : np.array( [1.0, 0.0], dtype=complex ),
        '1'  : np.array( [0.0, 1.0], dtype=complex ),
        'es' : np.array( [1./sqrt(2), 1./sqrt(2)], dtype=complex ),
        }



# State Creation
# ==============

# qubit on the block sphere. th is from vertical and ph is from x around z.
# th and ph are expected in degrees
# -------------------------------------------------------------------------
def qubit(t, p):
    t = pi/180.0 * t
    p = pi/180.0 * p
    return cos(t/2.0) * bvecs['0'] + exp(1j*p) * sin(t/2.0) * bvecs['1']

# helper for defining excitation and background qubits
# ----------------------------------------------------
def make_config_dict(config):
    config_list = config.split('_')
    n = len(config_list)
    if n == 1:
        conf_dict = {'ex_list' : [int(ex) for ex in config.split('-')],
                'ex_config' : {'t':180, 'p':0},
                'bg_config' : {'t':0, 'p':0} }

    elif n == 2:
        ex_list = [int(ex) for ex in config_list[0].split('-')]
        ex_config = {ang[0] : eval(ang[1:]) for ang in config_list[1].split('-')}
        conf_dict = {'ex_list': ex_list,
                'ex_config' : ex_config,
                'bg_config' : {'t':0, 'p':0} }

    elif n == 3:
        ex_list = [int(ex) for ex in config_list[0].split('-')]
        ex_config = {ang[0] : eval(ang[1:]) for ang in config_list[1].split('-')}
        bg_config = {ang[0] : eval(ang[1:]) for ang in config_list[2].split('-')}
        conf_dict = {'ex_list': ex_list,
                'ex_config' : ex_config,
                'bg_config' : bg_config}
    return conf_dict


# create Fock state
# -----------------
def fock(L, config):
    config_dict = make_config_dict(config)
    ex_list = np.array(config_dict['ex_list'])
    qubits = np.array([qubit(**config_dict['bg_config'])]*L)
    for ex in ex_list:
        qubits[ex, ::] = qubit(**config_dict['ex_config'])
    state = mx.listkron(qubits)
    return state


# create GHZ state
# ----------------
def GHZ (L, congif):
    s1=['1']*(L)
    s2=['0']*(L)
    state = (1.0/sqrt(2.0)) \
            * (mx.listkron([bvecs[key] for key in s1]) \
            +   mx.listkron([bvecs[key] for key in s2]))

    return state


# create W state
# --------------
def W(L, config):
    return 1.0/sqrt(L) * sum(fock(L, str(j)) for j in range(L))

# create one of the four bell states
# ----------------------------------
def Bell(L, config):
    jk, typ = config.split('_')
    j, k = jk.split('-')
    coeff = 1.0
    if typ in ('1', '3'):
        coeff = -1.0
    if typ in ('2', '3'):
        state = 1/sqrt(2)*(fock(L, j) + coeff*fock(L, k))

    elif typ in ('0', '1'):
        state = 1/sqrt(2)*(mx.listkron([qubit(0.0, 0.0)]*L) + coeff*fock(L, jk))
    return state

# create qubits with uniform (lowercase) or winding (capital) theta and phi
# -------------------------------------------------------------------------
def spin_wave(L, config):
    Tt, Pp = config.split('-')
    ang_dict = {'T' : np.linspace(0.0,  pi*float(Tt[1:]), L),
                't' : [float(Tt[1:])]*L,
                'P' : np.linspace(0.0, 2*pi*float(Pp[1:]), L),
                'p' : [float(Pp[1:])]*L,
                    }
    th_list = ang_dict[Tt[0]]
    ph_list = ang_dict[Pp[0]]
    qubit_list = [0.0]*L
    for j, (th, ph) in enumerate(zip(th_list, ph_list)):
        qubit_list[j] = qubit(th, ph)
    return mx.listkron(qubit_list)

# create a random state of excitation and bacground qubits
# --------------------------------------------------------
def rand_state(L, config):
    ps_qex_qbg_conf = config.split('_')
    ps = ps_qex_qbg_conf[0].split('-')
    p = float('.'+ps[0])
    s = None
    if len(ps) == 2:
        s = ps[1]

    if len(ps_qex_qbg_conf)==1:
        state_dict = {'ex':bvecs['1'], 'bg':bvecs['0']}

    if len(ps_qex_qbg_conf)==2:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph), 'bg':bvecs['0']}

    if len(ps_qex_qbg_conf)==3:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        bg_th, bg_ph = ps_qex_qbg_conf[2].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph), 'bg':qubit(bg_th, bg_ph)}

    prob = [p, 1.0 - p]
    if s is not None:
        np.random.seed(int(s))
    distrib = np.random.choice(['ex','bg'], size=L, p=prob)
    return mx.listkron([state_dict[i] for i in distrib])


# random throw in Hilbert space
# -----------------------------
def random_throw(L, config):
    np.random.seed(int(config))
    state = np.random.rand(2**L, 2) - 0.5
    state = state.view(dtype=np.complex128)[...,0]
    state = state/np.sqrt(np.conj(state).dot(state))
    return state

# create a state with any of the above states
# embeded in the center of the lattice
# -------------------------------------------
def center(L, config):
    Lcent = config.split('_')[0]
    cent_IC = config.split('_')[1::]
    cent_IC = '_'.join(cent_IC)
    len_cent = int(Lcent)
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    if cent_IC[0] == 'f':
        config_dict = make_config_dict(cent_IC[1::])
    else:
        config_dict = make_config_dict('0')
    bg_qubit = qubit(**config_dict['bg_config'])
    left = mx.listkron([bg_qubit for _ in range(len_L)])
    cent = make_state(len_cent, cent_IC)
    right = mx.listkron([bg_qubit for _ in range(len_R)])
    if len_back == 0:
        return cent
    elif len_back == 1:
        return mx.listkron([cent, right])
    else:
        return mx.listkron([left, cent, right])


# make the specified state
# ------------------------
smap = { 'f' : fock,
         'c' : center,
         's' : spin_wave,
         'r' : rand_state,
         'R' : random_throw,
         'G' : GHZ,
         'W' : W,
         'B' : Bell }

# make quantum state from state specification string or list of tuples
# --------------------------------------------------------------------
def make_state (L, IC):
    if type(IC) == str:
        name = IC[0]
        config = IC[1:]
        state = smap[name](L, config)
    elif type(IC) == list:
        state = np.zeros(2**L, dtype = complex)
        for s in IC:
            name = s[0][0]
            config = s[0][1:]
            coeff = s[1]
            state = state + coeff * smap[name](L, config)
    return state


if __name__ == '__main__':
    import simulation.measures as ms
    import simulation.states as ss
    import simulation.time_evolve as te
    import matplotlib.pyplot as plt

    state = make_state(4,'B0-1_3')
    rho12 = mx.rdms(state, [0,1])
    print(ms.vn_entropy(rho12))

    '''
    L_list = [4]
    # spin down (or |1>) at sites 0 and 2 spin up (or |0>) at 1 and 3
    IC_list = ['f0-2']
    for L, IC in zip(L_list, IC_list):
        print ("L = ", str(L), " IC = ", str(IC) )
        print('Expect spin down (or |1>) at sites 0 and 2. Spin up (or |0>) at 1 and 3')

        print()

        print('state vector:')
        state = make_state(L, IC)
        print(state)

        print()

        # reduced density matrix for each site calculated from the state
        rho_list = [mx.rdms(state, [j]) for j in range(L)]

        # measure z projection at each site. Take real part because measurements
        # of Hermitian ops always give a real result
        meas_list = [np.trace(rho.dot(ss.ops['Z'])).real for rho in rho_list ]

        print('expectation value along Z axis:')
        print(meas_list)
       '''
