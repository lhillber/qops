N0 = 1
N1 = 278
N2 = 5736
N3 = 26752
N4 = 32786
import states as ss
import matrix as mx
import measures as ms
import h5py
import numpy as np
from mpi4py import MPI
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import os, re, sys, time, errno
from math import log, pi
from cmath import exp, sin, cos
from itertools import cycle, zip_longest

mpl.rcParams["text.latex.preamble"] = ["\\usepackage{amsmath}"]
font = {"size": 12, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rc(*("font",), **font)
defaults = {
    "Ls": [12],
    "Ts": [60],
    "Vs": ["H"],
    "rs": [1],
    "Ss": [6],
    "Ms": [2],
    "ICs": ["c1_f0"],
    "BCs": ["0"],
    "tasks": [
        "g2-xx",
        "g2-yy",
        "g2-zz",
        "FT-D",
        "FT-Y",
        "FT-C",
        "FT-x",
        "FT-z",
        "FT2D-x",
        "FT2D-z",
        "FT2D-s",
        "scenter",
        "sbond",
        "cut_twos",
        "cut_half",
    ],
    "sub_dir": "default",
    "thread_as": "power",
}


def main(
    master="master_bak", defaults=defaults, rewrite_meas=False, rewrite_time=False
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    args = get_args(defaults=defaults)
    sub_dir = defaults["sub_dir"]
    thread_as = defaults["thread_as"]
    tasks = defaults["tasks"]
    params_list = make_params_list(thread_as, *args)
    if rank == 0:

        print(
            "\n Hello from rank 0! You are running {} jobs on {} cores...".format(
                len(params_list), nprocs
            )
        )
    for sim_id, params in enumerate(params_list):
        if sim_id % nprocs == rank:
            t0 = time.time()
            name = make_name(**params)
            print("rank: {},\n{}".format(rank, name))
            master_data_path = make_path(**dir_params("data", master))
            master_data_fname = os.path.join(master_data_path, name) + ".hdf5"
            data_path = make_path(**dir_params("data", defaults["sub_dir"]))
            name, T = name_check(master_data_fname, data_path)
            params["T"] = T
            master_data_path = make_path(**dir_params("data", master))
            master_data_fname = os.path.join(master_data_path, name) + ".hdf5"
            master_plot_path = make_path(**dir_params("plots", master))
            master_plot_fname = os.path.join(master_plot_path, name) + ".pdf"
            data_path = make_path(**dir_params("data", defaults["sub_dir"]))
            data_fname = os.path.join(data_path, name) + ".hdf5"
            plot_path = make_path(**dir_params("plots", defaults["sub_dir"]))
            plot_fname = os.path.join(plot_path, name) + ".pdf"
            params["name"] = name
            params["data_fname"] = data_fname
            params["plot_fname"] = plot_fname
            try:
                h5file = h5py.File(master_data_fname)
            except:
                print("deleting {}".format(master_data_fname))
                os.remove(master_data_fname)
                h5file = h5py.File(master_data_fname)

            h5file, tasks_added = run_required(
                params,
                tasks,
                h5file,
                rank,
                rewrite_time=rewrite_time,
                rewrite_meas=rewrite_meas,
            )
            plot(params, h5file, plot_fname)
            h5file.close()
            symlink_force(master_data_fname, data_fname)
            t_elapsed = time.time() - t0
            print(print_string(params, rank, tasks_added, t_elapsed))


def run_required(params, tasks, h5file, rank, rewrite_meas=False, rewrite_time=False):
    time_tasks = [
        task for task in tasks if task.split("-")[0] in implemented_time_tasks
    ]
    meas_tasks = [
        task for task in tasks if task.split("-")[0] in implemented_meas_tasks
    ]
    required_time_tasks, required_meas_tasks = check_deps(tasks, h5file)
    avail_time_tasks, avail_meas_tasks = get_avail_tasks(h5file)
    needed_time_tasks = [
        task for task in required_time_tasks if task not in avail_time_tasks
    ]
    needed_meas_tasks = [
        task for task in required_meas_tasks if task not in avail_meas_tasks
    ]
    tasks_added = []
    if rewrite_time:
        time_evolve(params, required_time_tasks, h5file, rank)
        tasks_added += required_time_tasks
    else:
        if len(needed_time_tasks) > 0:
            tasks_added += needed_time_tasks
            time_evolve(params, needed_time_tasks, h5file, rank)
    if rewrite_meas:
        tasks_added += required_meas_tasks
        measure(params, required_meas_tasks, h5file)
    else:
        if len(needed_meas_tasks) > 0:
            tasks_added += needed_meas_tasks
            measure(params, needed_meas_tasks, h5file)
    return (h5file, tasks_added)


def time_evolve(params, time_tasks, h5file, rank):
    time_step_gen = get_time_step_gen(params)
    L, T, S = [params[key] for key in ("L", "T", "S")]
    init_args = [(L, T)] * len(time_tasks)
    res = init_tasks(time_tasks, init_args, res={})
    for t, next_state in enumerate(time_step_gen):
        for task in time_tasks:
            task_slim = task.split("-")[0]
            if task[0] == "s":
                try:
                    order = int(task.split("-")[1])
                except IndexError:
                    order = 2

                task_map[task_slim]["set"](next_state, res[task], t, order)
            else:
                task_map[task_slim]["set"](next_state, res[task], t)

    added_time_tasks = res.pop("tasks_tmp")
    try:
        updated_time_tasks = np.append(h5file["time_tasks"][:], added_time_tasks)
        del h5file["time_tasks"]
    except:
        updated_time_tasks = added_time_tasks

    h5file["time_tasks"] = updated_time_tasks
    save_dict_hdf5(res, h5file)
    del res


def measure(params, meas_tasks, h5file):
    L, T, S = [params[key] for key in ("L", "T", "S")]
    init_args = [(L, T)] * len(meas_tasks)
    res = init_tasks(meas_tasks, init_args, res={})
    for task in meas_tasks:
        task_slim = task.split("-")[0]
        if task[0] == "s":
            try:
                order = int(task.split("-")[1])
            except IndexError:
                order = 2

            task_map[task_slim]["set"](h5file, res, task, order)
        else:
            task_map[task_slim]["set"](h5file, res, task)

    added_meas_tasks = res.pop("tasks_tmp")
    try:
        updated_meas_tasks = np.append(h5file["meas_tasks"][:], added_meas_tasks)
        del h5file["meas_tasks"]
    except:
        updated_meas_tasks = added_meas_tasks

    h5file["meas_tasks"] = updated_meas_tasks
    save_dict_hdf5(res, h5file)
    del res


def get_time_step_gen(params):
    L, T, V, r, S, M, IC, BC_type, BC_conf = validate_params(params)
    state = ss.make_state(L, IC)
    mode_list = get_mode(M, L)
    lUs, U, rUs = get_Us(V, r, S, BC_conf)
    time_step_gen = gen_time_step(state, U, lUs, rUs, mode_list, L, T, r, BC_type)
    return time_step_gen


def get_mode(mode_name, L):
    if mode_name in ("sweep", "swp"):
        m = 1
    else:
        if mode_name in ("alternate", "alt"):
            m = 2
        else:
            if mode_name in ("block", "blk"):
                m = 3
            else:
                try:
                    m = int(mode_name)
                except:
                    raise ValueError(
                        "mode is not swp (d=1), alt (d=2), blk (d=3), or            convertible to an integer step size."
                    )

    return make_mode_list(m, L)


def make_mode_list(d, L):
    mode_list = []
    for delta in range(d):
        for j in range(L):
            if j % d == delta:
                mode_list.append(j)

    return mode_list


def get_Us(V, r, S, BC_conf):
    U = make_U(V, r, S)
    bUs = make_boundary_Us(V, r, S, BC_conf)
    lUs = bUs[0:r]
    rUs = bUs[r : 2 * r + 1]
    return (lUs, U, rUs)


def make_U(V, r, S):
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2 ** N)[::-1]
    U = np.zeros((2 ** (N + 1), 2 ** (N + 1)), dtype=complex)
    for sig, s in enumerate(Sb):
        sigb = mx.dec_to_bin(sig, N)
        Vmat = get_V(V, s)
        ops = (
            [mx.ops[str(op)] for op in sigb[0:r]]
            + [Vmat]
            + [mx.ops[str(op)] for op in sigb[r : 2 * r + 1]]
        )
        U += mx.listkron(ops)

    if not mx.isU(U):
        raise AssertionError
    return U


def make_boundary_Us(V, r, S, BC_conf):
    BC_conf = list(map(int, BC_conf))
    N = 2 * r
    Sb = mx.dec_to_bin(S, 2 ** N)[::-1]
    bUs = []
    for w in (0, 1):
        fixed_o = list(BC_conf[w * r : r + w * r])
        for c in range(r):
            U = np.zeros((2 ** (r + 1 + c), 2 ** (r + 1 + c)), dtype=complex)
            for i in range(2 ** (r + c)):
                if w == 0:
                    var = mx.dec_to_bin(i, r + c)
                    var1 = var[0:c]
                    var2 = var[c:]
                    fixed = fixed_o[c:r]
                    n = fixed + var1 + var2
                    s = Sb[mx.bin_to_dec(n)]
                    Vmat = get_V(V, s)
                    ops = (
                        [mx.ops[str(op)] for op in var1]
                        + [Vmat]
                        + [mx.ops[str(op)] for op in var2]
                    )
                else:
                    if w == 1:
                        var = mx.dec_to_bin(i, r + c)[::-1]
                        var1 = var[c:][::-1]
                        var2 = var[0:c]
                        fixed = fixed_o[0 : r - c]
                        n = var1 + var2 + fixed
                        s = Sb[mx.bin_to_dec(n)]
                        Vmat = get_V(V, s)
                        ops = (
                            [mx.ops[str(op)] for op in var1]
                            + [Vmat]
                            + [mx.ops[str(op)] for op in var2]
                        )
                U += mx.listkron(ops)

            bUs.append(U)

    return bUs


def get_V(V, s):
    Vmat = mx.make_U2(V)
    return s * Vmat + (1 - s) * mx.ops["I"]


def iterate(state, U, lUs, rUs, mode_list, L, r, BC_type):
    if BC_type == "1":
        for j in mode_list:
            if j < r:
                Nj = range(0, j + r + 1)
                u = lUs[j]
            else:
                if L - j - 1 < r:
                    Nj = range(j - r, L)
                    u = rUs[L - j - 1]
                else:
                    Nj = range(j - r, j + r + 1)
                    u = U
                Nj = list(Nj)
                state = mx.op_on_state(u, Nj, state)

    else:
        if BC_type == "0":
            for j in mode_list:
                Nj = [k % L for k in range(j - r, j + r + 1)]
                state = mx.op_on_state(U, Nj, state)

    return state


def gen_time_step(state, U, lUs, rUs, mode_list, L, T, r, BC_type):
    yield state
    for t in range(T):
        state = iterate(state, U, lUs, rUs, mode_list, L, r, BC_type)
        yield state


def validate_params(params):
    arg_names = ["L", "T", "V", "r", "S", "M", "IC", "BC"]
    proper_types = [
        (int,),
        (int,),
        (str,),
        (int,),
        (int,),
        (str, int),
        (str, list),
        (str,),
    ]
    L, T, V, r, S, M, IC, BC = [params[key] for key in arg_names]
    args = [L, T, V, r, S, M, IC, BC]
    given_types = list(map(type, args))
    mask = np.array([t not in pt for t, pt in zip(given_types, proper_types)])
    if np.any(mask == True):

        def grab_bad(lst):
            return np.array(lst)[mask]

        bad_inputs = [
            bad for bad in map(grab_bad, (arg_names, args, given_types, proper_types))
        ]
        for arg_name, bad_arg, bad_type, proper_type in zip(*bad_inputs):
            raise ValueError(
                "Argument {} is {} (type {}) should be of type {}".format(
                    arg_name, bad_arg, bad_type, proper_type
                )
            )

    G = 2 ** (2 ** (2 * r))
    if S >= G:
        raise ValueError(
            "rule S = {} is invalid (need S < 2^2^(2*r) = {})".format(S, G)
        )
    if BC[0] == "0":
        BC_type, BC_conf = ("0", "00")
    else:
        if BC[0] == "1":
            try:
                BC_type, BC_conf = BC.split("-")
            except:
                raise ValueError("BC {} is not formatted properly".format(BC))

            if len(BC_conf) != 2 * r:
                raise ValueError("BC configuration {} is not valid".format(BC_conf))
        else:
            raise ValueError("BC type {} is not understood".format(BC[0]))
    return (L, T, V, r, S, M, IC, BC_type, BC_conf)


def print_string(params, rank, tasks_added, t_elapsed):
    print_string = "\n================================================================================\n"
    print_string += "Rank: {}\n".format(rank)
    if len(tasks_added) == 0:
        print_string += "Nothing to add to {}\n".format(params["name"])
    else:
        print_string += "Updated: {}\n".format(params["name"])
        print_string += "with {}\n".format(tasks_added)
    print_string += "total file size: {:.2f} MB\n".format(
        os.path.getsize(params["data_fname"]) / 1000000.0
    )
    print_string += "took: {:.2f} s\n".format(t_elapsed)
    print_string += "data at:\n"
    print_string += params["data_fname"] + "\n"
    print_string += "plots at:\n"
    print_string += params["plot_fname"] + "\n"
    print_string += "================================================================================"
    return print_string


def get_args(defaults=defaults):
    args = sys.argv[1:]
    nargs = len(args)
    sim_param_keys = ("Ls", "Ts", "Vs", "rs", "Ss", "Ms", "ICs", "BCs")
    all_keys = defaults.keys()
    if nargs not in (0, len(list(all_keys))):
        raise ValueError("incorrect number of arguments")
    else:
        args_dict = make_args_dict(args, defaults=defaults)
        defaults.update(args_dict)
    return [defaults[key] for key in sim_param_keys]


def make_args_dict(args, defaults=defaults):
    arg_names = [
        "Ls",
        "Ts",
        "Vs",
        "rs",
        "Ss",
        "Ms",
        "ICs",
        "BCs",
        "tasks",
        "sub_dir",
        "thread_as",
    ]
    args_dict = defaults
    for arg_name, arg in zip(arg_names, args):
        if arg_name not in ("sub_dir", "thread_as"):
            arg = arg.split(" ")
            if arg_name in ("Ls", "Ts", "rs", "Ss", "Ms"):
                arg = list(map(int, arg))
        args_dict[arg_name] = arg

    return args_dict


def make_params_list(thread_as, Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    if thread_as in ("zip", "cycle", "zipcycle", "zip_cycle", "zip cycle"):
        params_list = zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    else:
        if thread_as in ("power", "set", "powerset", "power_set", "power set"):
            params_list = power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
        else:
            if thread_as in ("repeat", "last", "repeatlastrepeat_lastrepeat last"):
                params_list = repeat_last_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
            else:
                raise ValueError(
                    "Argument thread_as was given {} and is neither zip, power, or repeat".format(
                        thread_as
                    )
                )
    return params_list


def power_set_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    return [
        make_params_dict(L, T, V, r, S, M, IC, BC)
        for L in Ls
        for T in iter(Ts)
        for V in iter(Vs)
        for r in iter(rs)
        for S in iter(Ss)
        for M in iter(Ms)
        for IC in iter(ICs)
        for BC in iter(BCs)
    ]


def zip_cycle_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    args = (Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    lens = [l for l in map(len, args)]
    ind = np.argmax(lens)
    to_zip = [el for el in map(cycle, args)]
    to_zip[ind] = args[ind]
    return [make_params_dict(*params) for params in zip(*to_zip)]


def repeat_last_params_list(Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs):
    args = (Ls, Ts, Vs, rs, Ss, Ms, ICs, BCs)
    lens = np.array([l for l in map(len, args)])
    ind = np.argmax(lens)
    longest = lens[ind]
    pads = longest - lens
    to_zip = [arg + [arg[-1]] * pad for arg, pad in zip(args, pads)]
    return [make_params_dict(*params) for params in zip_longest(*to_zip)]


def make_params_dict(L, T, V, r, S, M, IC, BC):
    return {"L": L, "T": T, "V": V, "r": r, "S": S, "M": M, "IC": IC, "BC": BC}


def dir_params(typ, sub_dir):
    return {
        "project_dir": os.path.dirname(os.path.realpath(__file__)),
        "output_dir": "qca_output",
        "sub_dir": sub_dir,
        "typ": typ,
    }


def make_name(L, T, V, r, S, M, IC, BC):
    name = "L{}_T{}_V{}_r{}_S{}_M{}_IC{}_BC{}".format(L, T, V, r, S, M, IC, BC)
    return name


def name_check(master_data_fname, data_path):
    master_data_path = os.path.dirname(master_data_fname)
    name = os.path.basename(master_data_fname).split(".hdf5")[0]
    headT = name.split("_T")[0]
    tailT = name.split("_V")[1]
    T = int(name.split("_T")[1].split("_V")[0])
    anynumber = "[-/+]?\\d*\\.?\\d+"
    re_str = headT + "_T" + anynumber + "_V" + tailT + ".hdf5"
    avail_fnames = [f for f in os.listdir(master_data_path) if re.match(re_str, f)]
    if len(avail_fnames) == 0:
        pass
    else:
        for fname in avail_fnames:
            Tavail = int(os.path.basename(fname).split("_T")[1].split("_V")[0])
            if T > Tavail:
                os.remove(os.path.join(master_data_path, fname))
                try:
                    os.remove(os.path.join(data_path, fname))
                except:
                    print("Deleted target but could not remove link")

            elif T <= Tavail:
                headT = name.split("_T")[0]
                tailT = name.split("_V")[1]
                name = headT + "_T" + str(Tavail) + "_V" + tailT
                T = Tavail

    return (name, T)


def make_path(project_dir, output_dir, sub_dir, typ):
    path = os.path.join(project_dir, output_dir, sub_dir, typ)
    os.makedirs(path, exist_ok=True)
    return path


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        try:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(target, link_name)
            else:
                raise e
        finally:
            e = None
            del e


def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(
                pp, format="pdf", bbox_inches="tight", bbox_extra_artist=extra_artist
            )
        else:
            fig.savefig(pp, format="pdf", bbox_extra_artist=extra_artist)
        if clf == True:
            fig.clf()

    pp.close()


def save_dict_hdf5(dic, h5file):
    recurs_save_dict_hdf5(h5file, "/", dic)


def recurs_save_dict_hdf5(h5file, path, dic_):
    for key, item in dic_.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            if path + key in h5file.keys():
                h5file[path + key][:] = item
            else:
                h5file[path + key] = item
        elif isinstance(item, dict):
            recurs_save_dict_hdf5(h5file, path + key + "/", item)
        elif isinstance(item, list):
            item_T = [
                [item[j][i] for j in range(len(item))] for i in range(len(item[0]))
            ]
            for k, el in enumerate(item_T):
                if path + key + "/c" + str(k) in h5file.keys():
                    h5file[path + key + "/c" + str(k)][:] = el
                else:
                    h5file[path + key + "/c" + str(k)] = el

        else:
            raise ValueError("Cannot save %s type" % item)


def load_dict_hdf5(filename):
    with h5py.File(filename, "r") as (h5file):
        return recurs_load_dict_hdf5(h5file, "/")


def recurs_load_dict_hdf5(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recurs_load_dict_hdf5(h5file, path + key + "/")

    return ans


def dep_g2(task):
    _, ops = task.split("-")
    deps = ["exp2-" + ops, "exp-" + ops[0], "exp-" + ops[1]]
    return deps


def dep_FT(task):
    FT_meas = task.split("-")
    meas = FT_meas[1]
    if meas in ("D", "C", "Y", "s", "s-2", "scenter", "scenter-2"):
        deps = [meas]
    else:
        if meas in ("x", "y", "z"):
            deps = ["exp-" + meas]
        else:
            raise ValueError(
                "unknown Fourier transform method for measure {}".format(meas)
            )
    return deps


def dep_FT2D(task):
    _, meas = task.split("-")
    if meas in "s":
        deps = [meas]
    else:
        if meas in ("x", "y", "z"):
            deps = ["exp-" + meas]
        else:
            raise ValueError(
                "unknown 2D Fourier transform method for measure        {}".format(meas)
            )
    return deps


def dep_MI(task):
    return ["MI"]


def dep_s_s2(task):
    return ["s-1", "s2-1"]


def dep_one_site(task):
    return ["one_site"]


def dep_two_site(task):
    return ["two_site"]


def init_one_site(L, T):
    return np.zeros((T + 1, L, 2, 2), dtype=complex)


def init_two_site(L, T):
    return np.zeros((T + 1, int(0.5 * L * (L - 1)), 4, 4), dtype=complex)


def init_cut_twos(L, T):
    N = L - 1
    c = int(N / 2)
    left_dims = [2 ** (l + 1) for l in range(c)]
    right_dims = left_dims
    if N % 2 != 0:
        right_dims = np.append(right_dims, 2 ** (c + 1))
    dims = np.append(left_dims, right_dims[::-1])
    init_shape = [[np.zeros((d, d), dtype=complex) for d in dims]] * (T + 1)
    return init_shape


def init_cut_half(L, T):
    return np.zeros((T + 1, 2 ** int(L / 2), 2 ** int(L / 2)), dtype=complex)


def init_sbond(L, T):
    return np.zeros((T + 1, L - 1))


def init_scalar(L, T):
    return np.zeros(T + 1)


def init_FT(L, T):
    offset = 0
    if T > 300:
        offset = 300
    return np.zeros(((T - offset + 1) // 2 + 1, 3))


def init_FT2D(L, T):
    offset = 0
    if T > 300:
        offset = 300
    nws = (T - offset) // 2 + 1 + T % 2
    nks = L // 2 + 1
    return np.zeros((1 + nws, 1 + nks))


def init_vec(L, T):
    return np.zeros((T + 1, L))


def init_mat(L, T):
    return np.zeros((T + 1, L, L))


def set_one_site(state, one_site, t):
    one_site[t, :, :, :] = ms.get_local_rhos(state)


def set_two_site(state, two_site, t):
    two_site[t, :, :, :] = ms.get_twosite_rhos(state)


def set_cut_twos(state, cut_twos, t):
    cut_twos[t][:][:][:] = ms.get_bipartition_rhos(state)


def set_cut_half(state, cut_half, t):
    cut_half[t, :, :] = ms.get_center_rho(state)


def set_sbond(state, sbond, t, order):
    L = int(log(len(state), 2))
    N = L - 1
    R = int(N / 2)
    sbond[t, :] = ms.get_bipartition_entropies(state, order)


def set_scenter(state, scenter, t, order):
    scenter[t] = ms.get_center_entropy(state, order)


def set_exp(h5file, res, task):
    exp = res[task]
    _, op = task.split("-")
    deps = task_map["exp"]["deps"](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        exp[t, :] = ms.local_exp_vals_from_rhos(rhos, mx.ops[op.upper()])


def set_FT(h5file, res, task):
    FT = res[task]
    FT_meas = task.split("-")
    meas = FT_meas[1]
    deps = task_map["FT"]["deps"](task)
    sig, = gather_deps(h5file, res, deps)
    if meas in ("x", "y", "z", "s", "s-2", "sbond", "sbond-2"):
        sig = np.mean(sig, axis=1)
    FT[:, :] = ms.fourier(sig).T


def set_FT2D(h5file, res, task):
    FT2D = res[task]
    _, meas = task.split("-")
    deps = task_map["FT2D"]["deps"](task)
    sig, = gather_deps(h5file, res, deps)
    FT2D[:, :] = ms.fourier2D(sig)
    FT2D[(0, 0)] = None


def set_exp2(h5file, res, task):
    exp2 = res[task]
    _, ops = task.split("-")
    deps = task_map["exp2"]["deps"](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        exp2[t, :, :] = ms.exp2_vals_from_rhos(
            rhos, mx.ops[ops[0].upper()], mx.ops[ops[1].upper()]
        )


def set_s(h5file, res, task, order):
    s = res[task]
    deps = task_map["s"]["deps"](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        s[t, :] = ms.local_entropies_from_rhos(rhos, order)


def set_s2(h5file, res, task, order):
    s2 = res[task]
    deps = task_map["s2"]["deps"](task)
    rhoss, = gather_deps(h5file, res, deps)
    for t, rhos in enumerate(rhoss):
        s2[t, :, :] = ms.twosite_entropies_from_rhos(rhos, order)


def set_MI(h5file, res, task):
    MI = res[task]
    deps = task_map["MI"]["deps"](task)
    ss, s2s = gather_deps(h5file, res, deps)
    for t, (s, s2) in enumerate(zip(ss, s2s)):
        MI[t, :, :] = ms.MI_from_entropies(s, s2)


def set_D(h5file, res, task):
    D = res[task]
    deps = task_map["D"]["deps"](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        D[t] = ms.network_density(mat)


def set_C(h5file, res, task):
    C = res[task]
    deps = task_map["C"]["deps"](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        C[t] = ms.network_clustering(mat)


def set_Y(h5file, res, task):
    Y = res[task]
    deps = task_map["Y"]["deps"](task)
    MI, = gather_deps(h5file, res, deps)
    for t, mat in enumerate(MI):
        Y[t] = ms.network_disparity(mat)


def set_g2(h5file, res, task):
    g2 = res[task]
    deps = task_map["g2"]["deps"](task)
    exp2s, exp1as, exp1bs = gather_deps(h5file, res, deps)
    for t, (exp2, exp1a, exp1b) in enumerate(zip(exp2s, exp1as, exp1bs)):
        g2[t, :, :] = ms.g2_from_exps(exp2, exp1a, exp1b)


def name_exp(task):
    op = task.split("-")[1]
    title = "$\\langle \\sigma^{%s}_j \\rangle$" % op.lower()
    xlabel = "Site"
    ylabel = "Iteration"
    return (title, xlabel, ylabel)


def name_FT(task):
    typ, meas = task.split("-")
    title = ""
    xlabel = "Frequency"
    ylabel = "$\\mathcal{P}$(%s)" % meas
    if meas in ("x", "y", "z"):
        ylabel = (
            "$\\mathcal{P}\\left(\\overline{\\langle \\sigma^{%s}\\rangle}\\right)$"
            % meas.lower()
        )
    return (title, xlabel, ylabel)


def name_FT2D(task):
    typ, meas = task.split("-")
    xlabel = "Wavenumber k"
    ylable = "Frequency $\\omega$"
    title = "$\\mathcal{P}$(%s)" % meas
    if meas in ("x", "y", "z"):
        ylabel = (
            "$\\mathcal{F}\\left(\\overline{\\langle \\sigma^{%s}\\rangle}\\right)$"
            % meas.lower()
        )
    return (title, xlabel, ylabel)


def name_s(task):
    try:
        order = task.split("-")[1]
    except:
        order = 1
    title = r"$s^{(%s)}_j$" % order
    xlabel = "Site"
    ylabel = "Iteration"
    return (title, xlabel, ylabel)


def name_sbond(task):
    try:
        order = task.split("-")[1]
    except:
        order = 1
    title = r"$s^{(%d)}_{L/2}$" % order
    xlabel = "Cut"
    ylabel = "Iteration"
    return (title, xlabel, ylabel)


implemented_time_tasks = [
    "one_site",
    "two_site",
    "cut_twos",
    "cut_half",
    "sbond",
    "scenter",
]
implemented_meas_tasks = [
    "exp",
    "exp2",
    "s",
    "s2",
    "MI",
    "g2",
    "D",
    "C",
    "Y",
    "FT",
    "FT2D",
]

task_map = {
    "one_site": {"init": init_one_site, "set": set_one_site},
    "two_site": {"init": init_two_site, "set": set_two_site},
    "cut_twos": {"init": init_cut_twos, "set": set_cut_twos},
    "cut_half": {"init": init_cut_half, "set": set_cut_half},
    "sbond": {"init": init_sbond, "set": set_sbond, "name": name_sbond},
    "scenter": {"init": init_scalar, "set": set_scenter},
    "s": {"deps": dep_one_site, "init": init_vec, "set": set_s, "name": name_s},
    "s2": {"deps": dep_two_site, "init": init_mat, "set": set_s2},
    "g2": {"deps": dep_g2, "init": init_mat, "set": set_g2},
    "MI": {"deps": dep_s_s2, "init": init_mat, "set": set_MI},
    "D": {"deps": dep_MI, "init": init_scalar, "set": set_D},
    "C": {"deps": dep_MI, "init": init_scalar, "set": set_C},
    "Y": {"deps": dep_MI, "init": init_scalar, "set": set_Y},
    "exp": {"deps": dep_one_site, "init": init_vec, "set": set_exp, "name": name_exp},
    "exp2": {"deps": dep_two_site, "init": init_mat, "set": set_exp2},
    "FT": {"deps": dep_FT, "init": init_FT, "set": set_FT, "name": name_FT},
    "FT2D": {"deps": dep_FT2D, "init": init_FT2D, "set": set_FT2D},
}


def init_tasks(tasks, argss, res={}):
    for task, args in zip(tasks, argss):
        task_slim = task.split("-")[0]
        res[task] = task_map[task_slim]["init"](*args)

    res["tasks_tmp"] = np.array(tasks).astype("|S9")
    return res


def gather_deps(h5file, res, deps):
    new_avail = [k for k in res.keys()]
    old_avail = [k for k in h5file.keys()]
    out = []
    for dep in deps:
        if dep in new_avail:
            out += [res[dep]]
        elif dep in old_avail:
            out += [h5file[dep][:]]
        else:
            raise ValueError("Dependency error, '{}' not available".format(dep))

    return out


def check_deps(tasks, h5file):
    time_tasks = [
        task for task in tasks if task.split("-")[0] in implemented_time_tasks
    ]
    meas_tasks = [
        task for task in tasks if task.split("-")[0] in implemented_meas_tasks
    ]
    avail_time_tasks, avail_meas_tasks = get_avail_tasks(h5file)
    required_time_tasks, required_meas_tasks = recurs_check_deps(
        time_tasks, meas_tasks, avail_time_tasks, avail_meas_tasks
    )
    return required_time_tasks, required_meas_tasks


def recurs_check_deps(
    time_tasks,
    meas_tasks,
    avail_time_tasks,
    avail_meas_tasks,
    requested_meas_tasks=[],
    requested_time_tasks=[],
    required_meas_tasks=[],
):
    for meas_task in meas_tasks:
        meas_task_slim = meas_task.split("-")[0]
        requested_meas_tasks = []
        deps = task_map[meas_task_slim]["deps"](meas_task)
        for dep in deps:
            dep_slim = dep.split("-")[0]
            if dep_slim in implemented_time_tasks:
                if dep not in requested_time_tasks:
                    requested_time_tasks += [dep]
            else:
                if dep_slim in implemented_meas_tasks:
                    if dep not in requested_meas_tasks:
                        requested_meas_tasks += [dep]
                        required_time_tasks, required_meas_tasks = recurs_check_deps(
                            [],
                            [dep],
                            avail_time_tasks,
                            avail_meas_tasks,
                            requested_meas_tasks=requested_meas_tasks,
                            required_meas_tasks=required_meas_tasks,
                        )
                    else:
                        raise ValueError(
                            "requested task {} is not implemented".format(dep)
                        )

        required_meas_tasks += requested_meas_tasks[::-1] + [meas_task]
    required_time_tasks = time_tasks + requested_time_tasks
    return unique(required_time_tasks), unique(required_meas_tasks)


def get_avail_tasks(h5file):
    try:
        avail_time_tasks = h5file["time_tasks"][:].astype("<U9")
    except:
        avail_time_tasks = []

    try:
        avail_meas_tasks = h5file["meas_tasks"][:].astype("<U9")
    except:
        avail_meas_tasks = []

    return (avail_time_tasks, avail_meas_tasks)


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not x in seen if not seen_add(x)]


def plot_name(fignum, params):
    name = params["name"]
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1, 1, 1)
    ax.text(0.05, 0.5, name)
    ax.axis([0, 1, 0, 1])
    ax.axis("off")


def plot(params, h5file, plot_fname):
    tasks = [k for k in h5file.keys()]
    L, T = params["L"], params["T"]
    exp_tasks = [task for task in tasks if task.split("-")[0] == "exp"]
    s_tasks = [task for task in tasks if task in ("s", "s-2", "sbond", "sbond-2")]
    nm_tasks = [task for task in tasks if task in ("D", "C", "Y")]
    FT_nm_tasks = [
        task
        for task in tasks
        if task in ("FT-D", "FT-C", "FT-Y", "FT-x", "FT-y", "FT-z", "FT-s")
    ]
    FT2D_tasks = [
        task for task in tasks if task in ("FT2D-x", "FT2D-y", "FT2D-z", "FT2D-s")
    ]
    plot_name(1, params)
    if len(exp_tasks) > 0:
        plot_vecs(2, exp_tasks, h5file)
        plot_exp_avg(3, exp_tasks, h5file)
    if len(s_tasks) > 0:
        plot_vecs(4, s_tasks, h5file)
    if "s" in h5file.keys():
        s = h5file["s"][:]
        title = "$\\overline{s^{(1)_j}$"

    elif "s-2" in h5file.keys():
        s = h5file["s-2"][:]
        title = "$\\overline{s^{(2)_j}$"

        plot_scalar(
            plt.figure(5).add_subplot(1, 1, 1),
            np.mean(s, axis=1),
            "",
            "Iteration",
        )
    if "scenter" in h5file.keys():
        fig = plt.figure(6)
        ax = fig.add_subplot(1, 1, 1)
        scenter = h5file["scenter"][:]
        title = r"$S^(1)_{L/2}$"
        plot_scalar(ax, scenter, "", "Iteration", title)

    elif "scenter-2" in h5file.keys():
        fig = plt.figure(7)
        ax = fig.add_subplot(1, 1, 1)
        scenter = h5file["scenter-2"][:]
        title = r"$S^(2)_{L/2}$"
        plot_scalar(ax, scenter, "", "Iteration", title)

    if len(nm_tasks) > 0:
        plot_network_measures(8, nm_tasks, h5file)
    if len(FT_nm_tasks) > 0:
        plot_FT_network_measures(9, FT_nm_tasks, h5file)
    if len(FT2D_tasks) > 0:
        plot_FT2D(10, FT2D_tasks, h5file)
    plt.tight_layout()
    multipage(plot_fname)


def plot_scenter(fignum, h5file):
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1, 1, 1)
    try:
        scenter = h5file["scenter"][:]
        title = r"$S^(1)_{L/2}$"
        plot_scalar(ax, scenter, "", "Iteration", title)
    except:
        title = r"$S^(2)_{L/2}$"
        scenter = h5file["scenter-2"][:]
        plot_scalar(ax, scenter, "", "Iteration", title)



def plot_vecs(fignum, vec_tasks, h5file, xspan=None, yspan=None, zspan=None):
    fig = plt.figure(fignum)
    nax = len(vec_tasks)
    for task_id, task in enumerate(vec_tasks):
        ax = fig.add_subplot(1, nax, task_id + 1)
        vec = h5file[task][:]
        T, L = vec.shape
        if xspan is None:
            xspan = [0, L]
        if yspan is None:
            yspan = [0, min(100, T)]
        if zspan is None:
            zspan = [-1, 1]
        title, xlabel, ylabel = task_map[task.split("-")[0]]["name"](task)
        ax = plot_vec(ax, vec, "", xlabel, ylabel, title, xspan, yspan, zspan)


def plot_exp_avg(fignum, exp_tasks, h5file):
    fig = plt.figure(fignum)
    ax = fig.add_subplot(1, 1, 1)
    for task_id, task in enumerate(exp_tasks):
        scalar = np.mean(h5file[task][:], axis=1)
        title, xlabel, ylabel = task_map[task.split("-")[0]]["name"](task)
        ax = plot_scalar(
            ax,
            scalar,
            "spatial spin averages",
            xlabel="Iteration",
            ylabel="",
            label=title,
        )

    ax.legend()


def plot_vec(ax, vec, title, xlabel, ylabel, zlabel, xspan, yspan, zspan):
    cax = ax.imshow(
        vec[yspan[0] : yspan[1], xspan[0] : xspan[1]],
        interpolation="None",
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ticks = [zspan[0], sum(zspan) / 2, zspan[1]]
    ticks = None
    cbar = plt.colorbar(cax, ticks=ticks, shrink=0.75, pad=0.08)
    cbar.ax.set_title(zlabel, y=1.01)
    return ax


def plot_scalar(
    ax,
    scalar,
    title,
    xlabel,
    ylabel,
    xs=None,
    xspan=None,
    ymin=None,
    label="",
    logy=False,
    kwargs={},
):
    T = len(scalar)
    if xspan is None:
        xspan = [0, min(61, T)]
    if xs is None:
        xs = np.linspace(xspan[0], xspan[1], int(xspan[1] - xspan[0]))
        scalar = scalar[xspan[0] : xspan[1]]
    if logy:
        ax.semilogy(xs, scalar, label=label, **kwargs)
    else:
        ax.plot(xs, scalar, label=label, **kwargs)
    if ymin is not None:
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
        nm = h5file[task][:]
        name = "$\\mathcal{%s}$" % task
        plot_scalar(ax, nm, "", "Iteration", name)


def peak_find(XX, YY, YYlima, YYlimb):
    Y = YY[YY > YYlima]
    X = XX[YY > YYlima]
    Ylimb = YYlimb[YY > YYlima]
    dx = XX[1] - XX[0]
    I = np.arange(len(X) - 1)
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

    return (xpks, ypks)


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


def plot_FT2D(fignum, FT2D_tasks, h5file):
    fig = plt.figure(fignum)
    nax = len(FT2D_tasks)
    for task_id, task in enumerate(FT2D_tasks):
        ax = fig.add_subplot(1, nax, task_id + 1)
        FT2D = h5file[task][:]
        ks = FT2D[0, 1:]
        ws = FT2D[1:, 0]
        ps = FT2D[1:, 1:]
        extent = extents(ks) + extents(ws)
        title, xlabel, ylabel = ("FT2D", "k", "w")
        ps[(0, 0)] = np.nan
        if np.mean(ps[1:, 1:]) < 1e-05:
            cax = ax.imshow(
                ps,
                interpolation="none",
                origin="lower",
                aspect="auto",
                extent=extent,
                norm=LogNorm(),
            )
            cbar = fig.colorbar(cax)
        else:
            cax = ax.imshow(
                ps, interpolation="none", origin="lower", aspect="auto", extent=extent
            )
            cbar = fig.colorbar(cax)


def plot_FT_network_measures(fignum, FT_nm_tasks, h5file, xspan=None):
    fig = plt.figure(fignum)
    nax = len(FT_nm_tasks)
    for task_id, task in enumerate(FT_nm_tasks):
        ax = fig.add_subplot(nax, 1, task_id + 1)
        fs = h5file[task][:, 0]
        FT_nm = h5file[task][:, 1]
        RN_nm = h5file[task][:, 2]
        chi22_RN_nm = 5.991 * RN_nm
        fpks, ppks = peak_find(fs, FT_nm, RN_nm, chi22_RN_nm)
        title, xlabel, ylabel = task_map[task.split("-")[0]]["name"](task)
        if task_id != nax - 1:
            xlabel = ""
            ax.set_xticks([])
        ymin = min([RN_nm[0], RN_nm[-1]])
        plot_scalar(
            ax, FT_nm, title, xlabel, ylabel, xs=fs, logy=True, ymin=ymin, kwargs={}
        )
        plot_scalar(
            ax,
            RN_nm,
            title,
            xlabel,
            ylabel,
            xs=fs,
            logy=True,
            ymin=ymin,
            kwargs={"ls": "--"},
        )
        plot_scalar(
            ax,
            chi22_RN_nm,
            title,
            xlabel,
            ylabel,
            xs=fs,
            logy=True,
            ymin=ymin,
            kwargs={"ls": ":"},
        )
        ax.scatter(fpks, ppks, c="r", marker="*", linewidth=0, s=35, zorder=3)


if __name__ == "__main__":
    main()
