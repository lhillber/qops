#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Testing states.py, matrix.py, and measures.py
#
# by Logan Hillberry
# ===============================================================

import states as ss
import copy
from math import log




if __name__ == '__main__':
    import time
    L = 17
    R = ss.make_state(L, 'R298')
    G = ss.make_state(L, 'G')

    state = R

    t0 = time.time()
    s1 = get_bipartition_entropies(state)
    t1 = time.time()
    s2 = np.asarray([vn_entropy(rho) for rho in
        bi_partite_rdm(state)])
    t2 = time.time()

    print(s1)
    print(s2)
    print('new', t1-t0, 'old', t2-t1)
    print(np.allclose(s1,s2))
