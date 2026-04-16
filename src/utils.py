
from multiprocessing import Pool
from sys import stdout
from tqdm import tqdm
import atexit

_pool = None
_n_cpu = None

def _get_pool(n_cpu):
    """
    Lazily initialize (or reuse) a module-level Pool of size n_cpu.
    """
    global _pool, _n_cpu
    if _pool is None:
        _pool = Pool(n_cpu)
        _n_cpu = n_cpu
        atexit.register(cleanup_pool)
    elif _n_cpu != n_cpu:
        raise RuntimeError(f"Tried to get pool with {n_cpu} cpus "
                           f"but we only have one with {_n_cpu}!")
    return _pool

def cleanup_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None


def do_parallel(
    func,
    args,
    n_cpu,
    verbose: bool = False,
    **kwargs
):
    """
    Execute func over args in parallel using n_cpu processes.
    Parameters
    ----------
    func : callable
        A function to be executed in parallel.
    args : iterable
        An iterable of arguments to pass to func.
    n_cpu : int
        Number of CPUs to use.
    verbose : bool
        Whether to show a progress bar.
    **kwargs
        Additional arguments passed to tqdm.
    Returns
    -------
    list
        Results from all function calls.
    """
    kwargs.setdefault("total", len(args))
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if n_cpu == 1:
        return list(tqdm(map(func, args), **kwargs))
    pool = _get_pool(n_cpu)
    return list(tqdm(pool.imap_unordered(func, args), **kwargs))


def _unpack_args(packed):
    func, args, kwargs = packed
    return func(*args, **kwargs)


def do_parallel_star(
    func,
    args_list,
    n_cpu,
    verbose: bool = False,
    **kwargs
):
    """
    Wrapper around do_parallel that supports multi-argument functions.

    Parameters
    ----------
    func : callable
        A function that takes multiple arguments.
    args_list : list of tuples
        Each tuple contains (args, kwargs) or just args for each call.
        - If element is a tuple/list of (args, kwargs): func(*args, **kwargs)
        - If element is a tuple/list of args: func(*args)
        - If element is a dict: func(**element)
    n_cpu : int
        Number of CPUs to use.
    verbose : bool
        Whether to show progress bar.
    **kwargs
        Additional arguments passed to do_parallel.

    Returns
    -------
    list
        Results from all function calls.
    """
    packed_args = []
    for item in args_list:
        if isinstance(item, dict):
            packed_args.append((func, (), item))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
            packed_args.append((func, item[0], item[1]))
        else:
            packed_args.append((func, item, {}))

    return do_parallel(_unpack_args, packed_args, n_cpu, verbose=verbose, **kwargs)
