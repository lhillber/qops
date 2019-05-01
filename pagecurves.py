import qca, h5py, numpy as np, matplotlib as mpl
from matplotlib.pyplot import pyplot as plt
from itertools import cycle
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import glob, os

def fit_page(sba):
    L = len(sba) + 1
    ells = np.arange(L - 1)

    def page_func(ell, a, logK):
        return (ell + 1) * np.log(a) - np.log(1 + a ** (-L + 2 * (ell + 1))) + logK

    popt, pcov = curve_fit(page_func, ells, sba)

    def func(ell):
        return page_func(ell, *popt)

    return (
     func, popt, pcov)


if '__name__' == '__maine__':
    mpl.rcParams['text.latex.preamble'] = [
     '\\usepackage{amsmath}']
    font = {'size':12,  'weight':'normal'}
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rc(*('font', ), **font)

    def ket(x):
        return '$\\vert ' + x + '\\rangle$'


    lines = {'f7-11':{'name':ket('7, 11'), 
      'ls':'-',  'c':'C0',  'm':'s'}, 
     'f5-9-13':{'name':ket('N3'), 
      'ls':'-',  'c':'C1',  'm':'s'}, 
     'f3-7-11-15':{'name':ket('N4'), 
      'ls':'-',  'c':'C2',  'm':'d'}, 
     'f0':{'name':ket('l'), 
      'ls':'-',  'c':'C3',  'm':'^'}, 
     'f0-18':{'name':ket('0, 18'), 
      'ls':'-',  'c':'C4',  'm':'*'}, 
     'n2-123':{'name':ket('n2'), 
      'ls':'-',  'c':'C1',  'm':'o'}, 
     'n3-123':{'name':ket('n3'), 
      'ls':'-',  'c':'C2',  'm':'s'}, 
     'n4-123':{'name':ket('n4'), 
      'ls':'-',  'c':'C3',  'm':'d'}, 
     'n5-123':{'name':ket('n5'), 
      'ls':'-',  'c':'C4',  'm':'^'}, 
     'n9-123':{'name':ket('n9'), 
      'ls':'-',  'c':'C5',  'm':'v'}, 
     'n6':{'name':ket('n6'), 
      'ls':'-',  'c':'C5',  'm':'d'}, 
     'n7':{'name':ket('n7'), 
      'ls':'-',  'c':'C6',  'm':'d'}, 
     'n8':{'name':ket('n8'), 
      'ls':'-',  'c':'C7',  'm':'d'}, 
     'n9':{'name':ket('n9'), 
      'ls':'-',  'c':'C8',  'm':'d'}, 
     'n10':{'name':ket('n10'), 
      'ls':'-',  'c':'C9',  'm':'d'}, 
     'n4-1':{'name':ket('n4-1'), 
      'ls':'-',  'c':'C8',  'm':'^'}, 
     'n4-2':{'name':ket('n4-2'), 
      'ls':'-',  'c':'C9',  'm':'*'}, 
     'n4-3':{'name':ket('n4-3'), 
      'ls':'-',  'c':'C4',  'm':'v'}, 
     'd3':{'name':ket('d3'), 
      'ls':'-',  'c':'C1',  'm':'s'}, 
     'd4':{'name':ket('d4'), 
      'ls':'-',  'c':'C2',  'm':'^'}, 
     'd5':{'name':ket('d5'), 
      'ls':'-',  'c':'C3',  'm':'d'}, 
     'd6':{'name':ket('d6'), 
      'ls':'-',  'c':'C4',  'm':'*'}, 
     'd7':{'name':ket('d7'), 
      'ls':'-',  'c':'C5',  'm':'v'}}
    Skey = [
     14]
    L = 18
    ells = np.arange(L - 1)
    dells = np.linspace(0, L - 1, 100)
    ICkey = ('d3', 'd4', 'd5', 'd6', 'd7')
    m, n = (1, 5)
    for S in Skey:
        plt.close('all')
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
        fig4, ax4 = plt.subplots(1, 1, figsize=(6, 3))
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3))
        fig, axs = plt.subplots(m, n, figsize=(8, 6), sharex=True, sharey=True)
        fig0, axs0 = plt.subplots(m, n, figsize=(8, 6), sharex=True, sharey=True)
        plot_fname = 'figures/pagecurves/dstates/rule{}_pagecurves-dstates_V3.pdf'.format(S)
        sb_avg = []
        sb_std = []
        ns = []
        for IC in ICkey:
            name = 'L{}_T*_VH_r1_S{}_M2_IC{}_BC1-00.hdf5'.format(L, S, IC)
            data_dir_glob = '/home/lhillber/documents/research/cellular_automata/qeca/qops/qca_output/master_bak/data/{}'.format(name)
            print(name)
            sims = [dict(L=int(os.path.basename(f).split('L')[1].split('T')[0][:-1]), T=int(os.path.basename(f).split('T')[1].split('V')[0][:-1]), V=os.path.basename(f).split('V')[1].split('r')[0][:-1], r=int(os.path.basename(f).split('r')[1].split('S')[0][:-1]), S=int(os.path.basename(f).split('S')[1].split('M')[0][:-1]), M=int(os.path.basename(f).split('M')[1].split('IC')[0][:-1]), IC=os.path.basename(f).split('IC')[1].split('BC')[0][:-1], BC=os.path.basename(f).split('BC')[1].split('.')[0], h5file=h5py.File(f)) for f in glob.glob(data_dir_glob)]
            if len(sims) == 0:
                continue
            sim = sims[np.argmax(np.array([s['T'] for s in sims]))]
            S = sim['S']
            L = sim['L']
            IC = sim['IC']
            i = ICkey.index(IC)
            r = i // m % m
            c = i % n
            add_n = False
            if IC[0] == 'n':
                ns += [int(IC.split('-')[0][1:])]
                add_n = True
            h5file = sim['h5file']
            sb = h5file['sbond-2'][:]
            z = h5file['exp-z'][:]
            sba = np.mean(sb[250:], axis=0)
            sbd = np.std(sb[250:], axis=0)
            if add_n:
                sb_avg += [sba]
                sb_std += [sbd]
            ax = axs[c]
            ax.imshow(sb[0:60], origin='lower', interpolation=None)
            ax.set_xlabel('$\\ell$')
            ax.set_title(lines[IC]['name'])
            ax.set_ylabel('Iteration')
            ax.label_outer()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax0 = axs0[c]
            ax0.imshow(z[0:60], origin='lower', interpolation=None)
            ax0.set_xlabel('$j$')
            ax0.set_title(lines[IC]['name'])
            ax0.set_ylabel('Iteration')
            ax0.label_outer()
            ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.errorbar(ells,
              sba, yerr=sbd, ls=None,
              color=lines[IC]['c'],
              fmt=lines[IC]['m'],
              label=lines[IC]['name'])
            func, popt, pcov = fit_page(sba)
            ax2.plot(dells, func(dells), color=lines[IC]['c'])
            ax2.set_xlabel('$\\ell$')
            ax2.set_ylabel('$\\overline{S}_{\\ell}$')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax4.plot(ells,
              sbd, color=lines[IC]['c'],
              marker=lines[IC]['m'],
              label=lines[IC]['name'])
            ax4.set_xlabel('$\\ell$')
            ax4.set_ylabel('$\\Delta S_{\\ell}$')
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax3.semilogx(sb[:, int(L / 2)], color=lines[IC]['c'],
              lw=1,
              label=lines[IC]['name'])
            ax3.set_ylabel('$S_{L/2}$')
            ax3.set_xlabel('Iteration')

        sb_avg = np.array(sb_avg)
        sb_std = np.array(sb_std)
        fig0.suptitle('$\\langle \\sigma_z \\rangle$')
        fig.suptitle('$S_{\\ell}$')
        ax4.legend(framealpha=0.5)
        ax3.legend(framealpha=0.5)
        ax2.legend(framealpha=0.5)
        fig0.tight_layout()
        fig0.subplots_adjust(top=0.99)
        fig.tight_layout()
        fig.subplots_adjust(top=0.99)
        qca.multipage(plot_fname)
        print('plot saved to ', plot_fname)
