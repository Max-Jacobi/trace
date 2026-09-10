"""
File Module

  This modules provides the file handling functionality for the fluid tracer particles.
  It implements methods to parse available files, load data into shared memory and interpolate from file data to particle positions.
  Data loading is performed in chunks to limit memory usage while allowing for parallel file loading.
"""

import os
import sys
import atexit
from abc import ABC, abstractmethod
from typing import TextIO, Any
from multiprocessing.shared_memory import SharedMemory
from functools import reduce

import numpy as np
from tqdm import tqdm

from .utils import do_parallel_star
from .interpolators.base import InterpolatorBase


class FileHandler(ABC):
    files: np.ndarray  # array of dictionaries linking file paths with metadata for loading
    times: np.ndarray  # array of times corresponding to the files
    keys: list[str]  # list of required data keys
    memory_size: int  # maximum memory size required to load data per key
    shared_memory: tuple[dict[str, str], ...]  # tuple of dictionaries linking data keys with shared memory names
    cur_times: np.ndarray  # array of currently loaded times in shared memory
    extra_data: Any = None # opaque object for storing extra data if needed in parsing/loading/interpolation
    interpolators: list[InterpolatorBase]  # list of interpolator callables for currently loaded data

    def __init__(
        self,
        interpolator: type[InterpolatorBase],
        directory: str,
        keys: list[str],
        n_cpu: int = 1,
        max_tot_memory: int | None = None,
        files_per_step: int | None = None,
        verbose: bool = False,
        out_file: TextIO = sys.stdout,
        interpolator_kwargs: dict[str, Any] = {},
        positive_keys: list[str] | None = None,
        ) -> None:
        self.keys = keys
        # Fields that cannot physically be negative, floored at 0 after every
        # load. Entries not among `keys` are ignored rather than rejected, so
        # a caller can pass one standard list whatever the format supplies.
        self.positive_keys = [key for key in (positive_keys or []) if key in keys]
        self._clamp_warned: set[str] = set()

        self.interpolator_cls = interpolator

        self.extra_data = {
            "interpolator": interpolator,
            "interpolator_kwargs": interpolator_kwargs,
            }

        self.parallel_kwargs = {
            "n_cpu": n_cpu,
            "verbose": verbose,
            "file": out_file,
        }

        self.parse_files(directory)
        if self.memory_size == 0:
            raise ValueError(
                "No files with valid data were found in the given directory, "
                "or all files were missing the requested keys."
            )
        # One shared memory segment of memory_size is allocated per key per
        # snapshot slot, so a slot costs memory_size * len(self.keys).
        step_memory = self.memory_size * len(self.keys)
        if files_per_step is not None:
            self.n_files_per_step = max(1, files_per_step)
        elif max_tot_memory is not None:
            self.n_files_per_step = int(max(1, max_tot_memory // step_memory))
        else:
            raise ValueError("Either files_per_step or max_tot_memory must be specified.")
        self.tot_memory = self.n_files_per_step * step_memory

        self.allocate_memory()

        self.cur_times = np.array([])

    @abstractmethod
    def list_files(self, directory: str) -> list[str]:
        """
        List all relevant files in the given directory.
        Arguments:
        directory : str
            Path to the directory to search for files.
        Returns:
        list[str]
            A list of file paths.
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_file(
        file_path: str,
        keys: list[str],
        extra_data: Any = None,
        ) -> tuple[float, str, Any, int]:
        """
        Parse a single file and return its time and data as a dictionary.
        Arguments:
        file_path : str
            Path to the file to parse.
        keys : list[str]
            List of required data keys.
        extra_data : Any
            Opaque object for passing extra data if needed.
        Returns:
        tuple[float, str, list[str], int]
            float : Time associated with the file.
            str : File path. If no relevant data is found, return an empty string.
            Any : Opaque object containing metadata for loading.
            int : memory size required to load the data.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_step_to_memory(
        metadata_dict: dict[str, Any],
        shared_memory: dict[str, str],
        extra_data: Any = None,
        ) -> None:
        """
        Load data from all files belonging to one time step into shared memory.

        Arguments:
        metadata_dict : dict[str, Any]
            Dictionary linking file paths with metadata for loading.
        shared_memory : dict[str, str]
            Dictionary linking data keys with shared memory names.
        extra_data : Any
            Opaque object for passing extra data if needed.
        """
        pass

    @staticmethod
    @abstractmethod
    def setup_interpolator(
        shared_memory: dict[str, str],
        extra_data: Any = None,
        ) -> InterpolatorBase:
        """
        Setup interpolator from data stored in shared memory for one timestep.

        Arguments:
        shared_memory : dict[str, str]
            Dictionary linking data keys with shared memory names.
        extra_data : Any
            Opaque object for passing extra data if needed.

        Returns:
        list[InterpolatorBase]
            List of interpolator callables for the loaded data.
        """
        pass

    def parse_files(self, directory: str) -> None:
        """
        Parse available files in the given directory
        Arguments:
        directory : str
            Path to the directory containing files.
        """
        file_list = np.array(self.list_files(directory))
        files = []
        times = []
        self.memory_size = 0

        for time, file_path, metadata, memory_size in do_parallel_star(
            type(self).parse_file,
            [(file_path, self.keys, self.extra_data) for file_path in file_list],
            desc="Parsing files",
            unit="files",
            **self.parallel_kwargs
        ):
            if not file_path:
                continue
            data_dict = {file_path: metadata}
            files.append(data_dict)
            times.append(time)
            if memory_size > self.memory_size:
                self.memory_size = memory_size

        times = np.array(times)
        files = np.array(files)

        self.times, rev_idx = np.unique(times, return_inverse=True)
        self.files = np.array([reduce(lambda a, b: {**a, **b}, files[rev_idx == i], {})
                      for i, _ in enumerate(self.times)])

        # Every key must be present at every time.  A key missing from one
        # snapshot is never loaded into that snapshot's buffer, which then
        # still holds whatever the previous chunk left there -- silently
        # integrating tracers through a stale field.  This bites formats
        # that write one variable per file (a variable dumped at a different
        # cadence than the rest), so it is checked rather than assumed.
        found = {key for group in self.files for keys in group.values() for key in keys}
        gaps = {}
        for time, group in zip(self.times, self.files):
            have = {key for keys in group.values() for key in keys}
            missing = [key for key in self.keys if key not in have]
            if missing:
                gaps[float(time)] = missing
        if gaps:
            shown = sorted(gaps)[:5]
            detail = "; ".join(f"t={t:g} missing {gaps[t]}" for t in shown)
            more = f" (and {len(gaps) - len(shown)} more times)" if len(gaps) > len(shown) else ""
            raise KeyError(
                f"Not every requested key is available at every snapshot time in "
                f"{directory}: {detail}{more}. Keys found anywhere: {sorted(found)}. "
                "Trim --keys, or restrict the directory to times that carry all of them."
            )

    def get_available_psm(self) -> int:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize

    def allocate_memory(self) -> None:
        """
        Allocate shared memory for data storage.
        First check if the available shared memory is sufficient for the required memory size,
          then create shared memory blocks for each key and time step.
        """

        available_psm = self.get_available_psm()
        if self.tot_memory > 0.9*available_psm:
            raise MemoryError(
                f"Required total memory for loading data ({self.tot_memory / 1e9:.2f} GB) "
                f"exceeds 90% of available shared memory ({available_psm / 1e9:.2f} GB)."
            )

        shared_mem = tuple(
            {key: SharedMemory(create=True, size=self.memory_size) for key in self.keys}
            for _ in range(self.n_files_per_step)
            )
        self.shared_memory = tuple({key: sh[key].name for key in sh} for sh in shared_mem)

        for sh in shared_mem:
            for key in sh:
                sh[key].close()

        atexit.register(self.free_shared_memory)

    def free_shared_memory(self) -> None:
        """
        Cleanup shared memory.
        """
        # Nothing to free if __init__ raised before allocate_memory().
        for sh in getattr(self, "shared_memory", ()):
            for key in sh:
                try:
                    sm = SharedMemory(name=sh[key])
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

                try:
                    sm.close()
                    sm.unlink()
                except FileNotFoundError:
                    continue

    def __del__(self):
        self.free_shared_memory()

    def load_chunk(
        self,
        start_index: int,
        forward: bool,
        ) -> None:
        """
        Load files into shared memory starting from start_index.
        Arguments:
        start_index : int
            Index of the file to start loading from.
        forward : bool
            Direction of loading files.
        """
        if forward:
            indices = np.arange(start_index, min(start_index + self.n_files_per_step, len(self.files)))
        else:
            indices = np.arange(start_index, max(start_index - self.n_files_per_step, -1), -1)

        tasks = [(self.files[i], self.shared_memory[j], self.extra_data)
                 for j, i in enumerate(indices)]

        msg = f"Loading t={self.times[indices[0]]:.0f}-{self.times[indices[-1]]:.0f}"

        do_parallel_star(
            type(self).load_step_to_memory,
            tasks,
            desc=msg,
            unit="file",
            **self.parallel_kwargs
        )
        self._clamp_positive(len(indices))
        self.cur_times = self.times[indices]

    def _clamp_positive(self, n_slots: int) -> None:
        """
        Floor every `positive_keys` field at 0 in the freshly loaded slots.

        Some writers emit small negative values for quantities that cannot be
        negative -- a non-monotone interpolation onto an output surface
        overshoots at a shock front, and undershoots to just below zero on the
        cold side of it.  The interpolators here are bound-preserving, so they
        do not create such values, but they do faithfully reproduce them and
        let one poisoned sample drag down a whole stencil's worth of queries.
        Clamping the samples rather than the interpolated result therefore
        fixes both, and fixes the seed masses too, which integrate the density
        field directly rather than through an interpolator.
        """
        if not self.positive_keys:
            return

        n_elem = self.memory_size // 8
        for slot in range(n_slots):
            for key in self.positive_keys:
                shm = SharedMemory(name=self.shared_memory[slot][key])
                try:
                    buf = np.ndarray(n_elem, dtype=np.float64, buffer=shm.buf)
                    n_bad = int(np.count_nonzero(buf < 0))
                    if not n_bad:
                        continue
                    worst = float(buf.min())
                    np.maximum(buf, 0.0, out=buf)
                    if key not in self._clamp_warned:
                        self._clamp_warned.add(key)
                        print(
                            f"WARNING: field '{key}' is declared non-negative but "
                            f"{n_bad} of {n_elem} samples in a loaded snapshot were "
                            f"negative (most negative {worst:.3e}); floored at 0. "
                            f"Further occurrences of '{key}' are clamped silently.",
                            file=self.parallel_kwargs["file"],
                            flush=True,
                        )
                finally:
                    shm.close()



    @classmethod
    def setup_interpolators(
        cls,
        keys: list[str],
        shared_memory: tuple[dict[str, str], ...],
        extra_data: Any = None,
        ) -> list[InterpolatorBase]:
        """
        Setup interpolators for currently loaded data in shared memory.
          Should be called after load_chunk on each process.
        """
        return [cls.setup_interpolator({key: shm_names[key] for key in keys}, extra_data)
                for shm_names in shared_memory]

    def get_chunk_indices(
        self,
        start_t: float,
        end_t: float,
        overlap: bool = True,
        n_snap: int = 2,
    ) -> tuple[np.ndarray, bool, float, float]:
        file_times = self.times
        n_files_per_step = self.n_files_per_step
        forward = end_t > start_t
        # Each chunk of n_files_per_step snapshots yields
        # n_files_per_step - n_snap + 1 integration steps, so consecutive
        # chunks must overlap by n_snap - 1 snapshots.
        stride = n_files_per_step - (n_snap - 1)
        if stride < 1:
            raise ValueError(
                f"n_files_per_step ({n_files_per_step}) must be at least n_snap "
                f"({n_snap}) to perform any integration steps per chunk."
            )

        if forward:
            t_start = np.min(file_times[file_times >= start_t])
            t_end = np.max(file_times[file_times <= end_t])
            i_start = file_times.searchsorted(t_start, side='left')
            i_end = file_times.searchsorted(t_end, side='left')
            if overlap:
                chunk_indices = np.arange(i_start, i_end+1, stride)
            else:
                chunk_indices = np.arange(i_start, i_end+1, n_files_per_step)
        else:
            t_start = np.max(file_times[file_times <= start_t])
            t_end = np.min(file_times[file_times >= end_t])
            i_start = file_times.searchsorted(t_start, side='left')
            i_end = file_times.searchsorted(t_end, side='left')
            if overlap:
                chunk_indices = np.arange(i_start, i_end-1, -stride)
            else:
                chunk_indices = np.arange(i_start, i_end-1, -n_files_per_step)
        return chunk_indices, forward, t_start, t_end
