#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# ==============================================================
# Testing states.py, matrix.py, and measures.py
#
# by Logan Hillberry
# ===============================================================

import states as ss
import matrix as mx
import measures as ms


if __name__ == '__main__':
    L = 5
    R     = ss.make_state(L, 'R298')
    W     = ss.make_state(L, 'W')
    B0    = ss.make_state(L, 'c2_B0-1_0')
    B1    = ss.make_state(L, 'c2_B0-1_1')
    B2    = ss.make_state(L, 'c2_B0-1_2')
    B3    = ss.make_state(L, 'c2_B0-1_3')
    GHZ   = ss.make_state(L, 'G')
    twist = ss.make_state(L, 'sT2-P1')

    for state in [R, W, B0, B1, B2, B3, GHZ, twist]:
        s  = ms.get_local_entropies(state)
        sc = ms.get_bipartition_entropies(state)
        x  = ms.get_local_exp_vals(state, ss.ops['X'])
        y  = ms.get_local_exp_vals(state, ss.ops['Y'])
        z  = ms.get_local_exp_vals(state, ss.ops['Z'])
        print('==================================================')
        print(s)
        print(sc)
        print(x)
        print(y)
        print(z)
