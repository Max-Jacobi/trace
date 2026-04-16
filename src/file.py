"""
File Module

  This modules provides the file handling functionality for the fluid tracer particles.
  It implements methods to parse available files, load data into shared memory and interpolate from file data to particle positions.
  Data loading is performed in chunks to limit memory usage while allowing for parallel file loading.
"""

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
    files: list[dict[str, list[str]]] # list of dictionaries linking file paths with contained data keys
    times: np.ndarray  # array of times corresponding to the files
    keys: list[str]  # list of required data keys
    memory_size: int  # maximum memory size required to load data per key
    shared_memory: tuple[dict[str, str], ...]  # tuple of dictionaries linking data keys with shared memory names
    cur_times: np.ndarray  # array of currently loaded times in shared memory
    extra_data: Any = None # opaque object for storing extra data if needed in parsing/loading/interpolation
    interpolators: list[InterpolatorBase]  # list of interpolator callables for currently loaded data

    def __init__(
        self,
        directory: str,
        keys: list[str],
        n_cpu: int = 1,
        max_tot_memory: int | None = None,
        files_per_step: int | None = None,
        verbose: bool = False,
        out_file: TextIO = sys.stdout,
        ) -> None:
        self.keys = keys

        self.parallel_kwargs = {
            "n_cpu": n_cpu,
            "verbose": verbose,
            "file": out_file,
        }

        self.parse_files(directory)
        if files_per_step is not None:
            self.n_files_per_step = files_per_step
            self.tot_memory = self.n_files_per_step * self.memory_size
        elif max_tot_memory is not None:
            self.n_files_per_step = max_tot_memory // self.memory_size
            self.tot_memory = self.n_files_per_step * self.memory_size
        else:
            raise ValueError("Either files_per_step or max_tot_memory must be specified.")

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

    @abstractmethod
    def setup_interpolator(
        self,
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

    def allocate_memory(self) -> None:
        """
        Allocate shared memory for data storage.
        """
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
        for sh in self.shared_memory:
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
            **self.parallel_kwargs
        )
        self.cur_times = self.times[indices]



    def setup_interpolators(self, keys: list[str]) -> list[InterpolatorBase]:
        """
        Setup interpolators for currently loaded data in shared memory.
          Should be called after load_chunk on each process.
        """
        return [self.setup_interpolator({key: shm_names[key] for key in keys}, self.extra_data)
                for shm_names in self.shared_memory[:len(self.cur_times)]]
