from multiprocessing import Pool
from itertools import count, repeat
from sys import stdout
from typing import Optional, Callable, Iterable
from tqdm import tqdm
import atexit

def _pack_args(args_list, func):
    """Normalize mixed positional and keyword call specifications."""
    packed_args = []
    for item in args_list:
        if isinstance(item, dict):
            packed_args.append((func, (), item))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], dict):
            packed_args.append((func, item[0], item[1]))
        else:
            packed_args.append((func, item, {}))
    return packed_args

def _unpack_args(packed):
    """Execute one normalized ``(func, args, kwargs)`` call tuple."""
    func, args, kwargs = packed
    return func(*args, **kwargs)

def do_parallel(
    func: Callable,
    args: Iterable,
    n_cpu: int,
    initializer: Optional[Callable] = None,
    initargs: Optional[Iterable] = None,
    verbose: bool = False,
    chunksize: int = 1,
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
    initializer : callable, optional
        A function to initialize each worker process. It will be called with the arguments in initargs
    initargs : iterable, optional
        An iterable of arguments to pass to the initializer function for each worker process.
    verbose : bool
        Whether to show a progress bar.
    **kwargs
        Additional arguments passed to tqdm.
    Returns
    -------
    list
        Results from all function calls.
    """
    try:
        kwargs.setdefault("total", len(args))
    except TypeError:
        ...
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if initargs is None:
        initargs = ()

    if n_cpu == 1:
        if initializer is not None:
            initializer(*initargs)
        return list(tqdm(map(func, args), **kwargs))

    with Pool(n_cpu, initializer=initializer, initargs=initargs) as pool:
        return list(tqdm(pool.imap_unordered(func, args, chunksize=chunksize), **kwargs))


def do_parallel_star(
    func: Callable,
    args_list: Iterable[tuple],
    n_cpu: int,
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
    packed_args = _pack_args(args_list, func)
    return do_parallel(_unpack_args, packed_args, n_cpu, verbose=verbose, **kwargs)


def do_parallel_star_pool(
    pool: Optional[Pool],
    func: Callable,
    args_list: Iterable[tuple],
    verbose: bool = False,
    chunksize: int = 1,
    **kwargs
):
    """
    Like do_parallel_star but dispatches onto a pre-created persistent Pool.

    If pool is None (i.e. n_cpu == 1), the tasks are executed serially in the
    calling process instead, which keeps the single-process debug path working.

    Parameters
    ----------
    pool : Pool or None
        A multiprocessing.Pool created externally. Pass None to run serially.
    func : callable
        A function that takes multiple arguments.
    args_list : list of tuples
        Same semantics as do_parallel_star.
    verbose : bool
        Whether to show a tqdm progress bar.
    chunksize : int
        imap_unordered chunksize (ignored for serial path).
    **kwargs
        Additional arguments forwarded to tqdm.
    """
    packed_args = _pack_args(args_list, func)
    try:
        kwargs.setdefault("total", len(packed_args))
    except TypeError:
        pass
    kwargs.setdefault("disable", not verbose)
    kwargs.setdefault("ncols", 0)
    kwargs.setdefault("file", stdout)

    if pool is None:
        return list(tqdm(map(_unpack_args, packed_args), **kwargs))

    return list(tqdm(pool.imap_unordered(_unpack_args, packed_args, chunksize=chunksize), **kwargs))
