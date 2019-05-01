from cmath import sqrt, sin, cos, exp, pi
import numpy as np, random
from matrix import listkron

bvecs = {'0':np.array([1.0, 0.0], dtype=complex), 
 '1':np.array([0.0, 1.0], dtype=complex), 
 'es':np.array([1.0 / sqrt(2), 1.0 / sqrt(2)], dtype=complex)}

def edit_small_vals(mat, tol=1e-14, replacement=0.0):
    if type(mat) is not np.ndarray:
        mat = np.asarray(mat)
    mat[np.abs(mat) <= tol] = replacement
    return mat


def qubit(t, p):
    t = pi / 180.0 * t
    p = pi / 180.0 * p
    return cos(t / 2.0) * bvecs['0'] + exp(1j * p) * sin(t / 2.0) * bvecs['1']


def make_config_dict(config):
    config_list = config.split('_')
    n = len(config_list)
    if n == 1:
        config_dict = {'ex_list':[int(ex) for ex in config.split('-')], 
         'ex_config':{'t':180, 
          'p':0}, 
         'bg_config':{'t':0, 
          'p':0}}
    else:
        if n == 2:
            ex_list = [int(ex) for ex in config_list[0].split('-')]
            ex_config = {ang[0]:eval(ang[1:]) for ang in config_list[1].split('-')}
            config_dict = {'ex_list':ex_list,  'ex_config':ex_config, 
             'bg_config':{'t':0, 
              'p':0}}
        else:
            if n == 3:
                ex_list = [int(ex) for ex in config_list[0].split('-')]
                ex_config = {ang[0]:eval(ang[1:]) for ang in config_list[1].split('-')}
                bg_config = {ang[0]:eval(ang[1:]) for ang in config_list[2].split('-')}
                config_dict = {'ex_list':ex_list,  'ex_config':ex_config, 
                 'bg_config':bg_config}
            else:
                print(config_list)
    return config_dict


def fock(L, config):
    config_dict = make_config_dict(config)
    ex_list = np.array(config_dict['ex_list'])
    qubits = np.array([qubit(**config_dict['bg_config'])] * L)
    for ex in ex_list:
        qubits[ex, :] = qubit(**config_dict['ex_config'])

    state = listkron(qubits)
    return state


def doublon(L, config):
    try:
        D = int(config.split('_')[0])
        config = config[1:]
    except IndexError:
        D = 2

    if D == 'd':
        D = 2
    else:
        D = int(D)
    fock_config = '-'.join([str(i) for i in range(L) if i % D == 0])
    fock_config = ''.join([fock_config, config])
    return fock(L, fock_config)


def rand_n(L, config):
    csplit = config.split('_')
    ns = csplit[0]
    nssplit = ns.split('-')
    n = int(nssplit[0])
    if len(nssplit) > 1:
        s = int(nssplit[1]) * L * n
    else:
        s = None
    random.seed(s)
    fock_config = '-'.join([str(i) for i in random.sample(range(L), n)])
    if len(csplit) > 1:
        config = csplit[1:]
        fock_config = ''.join([fock_config, config])
    return fock(L, fock_config)


def GHZ(L, config):
    s1 = [
     '1'] * L
    s2 = ['0'] * L
    state = 1.0 / sqrt(2.0) * (listkron([bvecs[key] for key in s1]) + listkron([bvecs[key] for key in s2]))
    return state


def W(L, config):
    return 1.0 / sqrt(L) * sum((fock(L, str(j)) for j in range(L)))


def Bell(L, config):
    jk, typ = config.split('_')
    j, k = jk.split('-')
    coeff = 1.0
    if typ in ('1', '3'):
        coeff = -1.0
    if typ in ('2', '3'):
        state = 1 / sqrt(2) * (fock(L, j) + coeff * fock(L, k))
    else:
        if typ in ('0', '1'):
            state = 1 / sqrt(2) * (listkron([qubit(0.0, 0.0)] * L) + coeff * fock(L, jk))
    return state

def Bell_array(L, config):
    try:
        bell_type = config[0]
    except:
        bell_type = '0'
    ic = 'B0-1_{}'.format(bell_type)
    singlet = make_state(2, ic)
    if L%2 == 0:
        state = listkron([singlet]*int(L/2))
    else:
        state = listkron([singlet]*int((L-1)/2) + [bvecs['0']])
    return state

def spin_wave(L, config):
    Tt, Pp = config.split('-')
    ang_dict = {'T':np.linspace(0.0, pi * float(Tt[1:]), L),  't':[
      float(Tt[1:])] * L, 
     'P':np.linspace(0.0, 2 * pi * float(Pp[1:]), L), 
     'p':[
      float(Pp[1:])] * L}
    th_list = ang_dict[Tt[0]]
    ph_list = ang_dict[Pp[0]]
    qubit_list = [0.0] * L
    for j, (th, ph) in enumerate(zip(th_list, ph_list)):
        qubit_list[j] = qubit(th, ph)

    return listkron(qubit_list)


def rand_state(L, config):
    ps_qex_qbg_conf = config.split('_')
    ps = ps_qex_qbg_conf[0].split('-')
    p = float('.' + ps[0])
    s = None
    if len(ps) == 2:
        s = ps[1]
    if len(ps_qex_qbg_conf) == 1:
        state_dict = {'ex':bvecs['1'], 
         'bg':bvecs['0']}
    if len(ps_qex_qbg_conf) == 2:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph),  'bg':bvecs['0']}
    if len(ps_qex_qbg_conf) == 3:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        bg_th, bg_ph = ps_qex_qbg_conf[2].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph),  'bg':qubit(bg_th, bg_ph)}
    prob = [p, 1.0 - p]
    if s is not None:
        np.random.seed(int(s))
    distrib = np.random.choice(['ex', 'bg'], size=L, p=prob)
    return listkron([state_dict[i] for i in distrib])


def rand_plus(L, config):
    exs_ps_qex_qbg_conf = config.split('_')
    exs = exs_ps_qex_qbg_conf[0].split('-')
    exs = np.array([int(ex) for ex in exs])
    ps = exs_ps_qex_qbg_conf[1].split('-')
    p = float('.' + ps[0])
    prob = [p, 1.0 - p]
    s = None
    if len(ps) == 2:
        s = ps[1]
    if s is not None:
        np.random.seed(int(s))
    if len(exs_ps_qex_qbg_conf) == 2:
        state_dict = {'ex':bvecs['1'], 
         'bg':bvecs['0']}
    if len(exs_ps_qex_qbg_conf) == 3:
        ex_th, ex_ph = exs_ps_qex_qbg_conf[2].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph),  'bg':bvecs['0']}
    if len(exs_ps_qex_qbg_conf) == 4:
        ex_th, ex_ph = exs_ps_qex_qbg_conf[2].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        bg_th, bg_ph = exs_ps_qex_qbg_conf[3].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])
        state_dict = {'ex':qubit(ex_th, ex_ph),  'bg':qubit(bg_th, bg_ph)}
    distrib = np.random.choice(['ex', 'bg'], size=L, p=prob)
    distrib[exs] = 'ex'
    state = listkron([state_dict[q] for q in distrib])
    return state


def random_throw(L, config):
    np.random.seed(None)
    if len(config) > 0:
        np.random.seed(int(config))
    state = np.random.rand(2 ** L, 2) - 0.5
    state = (state.view(dtype=np.complex128))[(Ellipsis, 0)]
    state = state / np.sqrt(np.conj(state).dot(state))
    return state


def center(L, config):
    Lcent = config.split('_')[0]
    cent_IC = config.split('_')[1:]
    cent_IC = '_'.join(cent_IC)
    len_cent = int(Lcent)
    len_back = L - len_cent
    len_L = int(len_back / 2)
    len_R = len_back - len_L
    if cent_IC[0] == 'f':
        config_dict = make_config_dict(cent_IC[1:])
    else:
        config_dict = make_config_dict('0')
    bg_qubit = qubit(**config_dict['bg_config'])
    left = listkron([bg_qubit for _ in range(len_L)])
    cent = make_state(len_cent, cent_IC)
    right = listkron([bg_qubit for _ in range(len_R)])
    if len_back == 0:
        return cent
        if len_back == 1:
            return listkron([cent, right])


    #if len_back == 0:
    #    return cent
    #    if len_back == 1:
    #        return listkron([cent, right])


    if len_back == 0:
        return cent
    elif len_back == 1:
        return listkron([cent, right])
    return listkron([left, cent, right])


smap = {'f':fock, 
 'd':doublon, 
 'n':rand_n, 
 'c':center, 
 's':spin_wave, 
 'r':rand_state, 
 'p':rand_plus, 
 'R':random_throw, 
 'G':GHZ, 
 'W':W, 
 'B':Bell,
 'S':Bell_array,
 }

def make_state(L, IC):
    if type(IC) == str:
        name = IC[0]
        config = IC[1:]
        state = smap[name](L, config)
    else:
        if type(IC) == list:
            state = np.zeros(2 ** L, dtype=complex)
            for s in IC:
                coeff = s[0]
                name = s[1][0]
                config = s[1][1:]
                state += coeff * smap[name](L, config)

    state = edit_small_vals(state.real) + 1j * edit_small_vals(state.imag)
    return state


if __name__ == '__main__':
    import measures as ms
    from matrix import ops
    L = 8
    ICs = ['f0-3_t90-p0_t45-p180', 'f2_t90-p90', 'f0-2-4-6', 'st90-P1',
     'sT2-p30', 'sT2-P1', 'r75_t45-p90', 'r5-234_t45-p90',
     'p0_5-12_t90-p90', 'd', 'd_t0-p0_t180-p0', 'R234', 'B0-1_3', 'G',
     'W', 'c1_f0', 'c4_W', 'c2_r5', [(1 / sqrt(2), 'f0'), (1 / sqrt(2), 'f1')]]
    print('Testing: create all example states for a system of ' + str(L) + ' sites and measure the expectation value of spin in the X, Y, and Z directions at each site, the von Neumann entropy of each site, and the von Neumann entropy of all bipartitions of the sites.\n')
    for IC_id, IC in enumerate(ICs):
        print(str(IC_id + 1) + ')', IC)
        state = make_state(L, IC)
        xs = ms.get_local_exp_vals(state, ops['X'])
        ys = ms.get_local_exp_vals(state, ops['Y'])
        zs = ms.get_local_exp_vals(state, ops['Z'])
        ss = ms.get_local_entropies(state)
        sb = ms.get_bipartition_entropies(state)
        print('<X>: ', xs)
        print('<Y>: ', ys)
        print('<Z>: ', zs)
        print('S_vN: ', ss)
        print('S_b: ', sb, '\n')
