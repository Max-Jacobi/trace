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

from .utils import do_parallel_star, cell_centres, cell_measure
from .mass import MassDensity
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
    # Coordinates this format's own cells are described in, i.e. how the axes
    # of native_cell_weights' bounds are to be read. A seeder that reads those
    # bounds compares it against its own before trusting them. None means the
    # format does not say, which no seeder will accept for its native cells.
    grid_geometry: str | None = None

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
        density_key: str = 'rho',
        vel_keys: tuple[str, ...] = (),
        adm_mass: float | None = None,
        ut_key: str = 'u_t',
        ) -> None:
        self.keys = keys
        # What this format's fields mean as a mass density. Owned here rather
        # than by the seeders, because the answer depends on the data's
        # spacetime and coordinates, not on how tracers are placed.
        self._mass_density = self.build_mass_density(
            density_key=density_key, vel_keys=tuple(vel_keys),
            adm_mass=adm_mass, ut_key=ut_key)
        # Checked on first use rather than here: a handler opened just to read
        # fields does not need a density, and should not have to name one.
        # Seeding touches this before any integration starts, so the error
        # still arrives long before a run has spent anything.
        self._mass_keys_missing = [
            k for k in self._mass_density.flux_keys if k not in keys]
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

    @property
    def mass_density(self) -> MassDensity:
        """
        What this format's fields mean as a mass density -- see :mod:`src.mass`.

        Every seeder and the output writer go through this, so the rho-vs-D
        decision is made once, here, by the side that knows the data.
        """
        if self._mass_keys_missing:
            raise ValueError(
                f"{type(self).__name__}: the mass density needs "
                f"{self._mass_keys_missing}, which are not among the loaded "
                f"keys {self.keys}. Add them to --keys, or drop whatever asked "
                "for them (--adm-mass needs --ut-key)."
            )
        return self._mass_density

    def build_mass_density(
        self,
        density_key: str,
        vel_keys: tuple[str, ...],
        adm_mass: float | None,
        ut_key: str,
        ) -> MassDensity:
        """
        How this format's fields become a mass density.

        The default is the conserved ``D = rho*W*sqrt(gamma)`` when an ADM mass
        is given and plain ``rho`` otherwise, which is right for every format
        shipped here: all three are GR runs on a spherical grid outside the
        remnant. Override it in a format that needs something else -- a
        Newtonian run (return a ``MassDensity`` with ``adm_mass=None`` whatever
        was asked for), a different spacetime, or one that dumps its own
        ``sqrt(gamma)`` and ``W`` and should use those instead of the analytic
        stand-in. See :mod:`src.mass`.
        """
        return MassDensity(density_key, vel_keys, adm_mass=adm_mass, ut_key=ut_key)

    def cell_weights_from_values(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        values: np.ndarray,
        r_sample: float | np.ndarray | None = None,
        ) -> np.ndarray:
        """
        The sampling weight of every cell, from the fields on them.

        Shared by every format implementing :meth:`native_cell_weights`, which
        is why it lives here: a format supplies its cells and the values on
        them, and this turns them into the mass each cell holds, or -- on a
        surface -- the mass crossing it per unit time.

        `values` follows ``self.mass_density.density_keys`` for volume cells,
        or ``flux_keys`` for surface cells.

        `r_sample` is None for volume cells. For surface cells it is the radius
        each value was actually sampled at -- one number for a format storing
        global shells, one per cell under AMR. The flux is evaluated *there*,
        ``r_sample**2 * rho * v_r`` per unit solid angle, and handed to the
        requested sphere unchanged. That is a nearest-neighbour carry of
        ``r**2 rho v_r``, the conserved flux of a steady radial outflow, rather
        than of ``rho v_r``: carrying the latter and multiplying by the
        requested radius squared would be off by ``(R/r_sample)**2``, the same
        for every cell of a global grid and so not averaging out.
        """
        pos = cell_centres(lo, hi, r_sample)
        measure = cell_measure(lo, hi, r_sample)
        if r_sample is None:
            return self.mass_density.density(values, pos) * measure
        return self.mass_density.radial_flux(values, pos) * measure

    def native_cell_weights(
        self,
        slot: int,
        surface_radius: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The format's **own** cells, as coordinate bounds plus a weight each.

        Optional. Implement it if the format can say where its cells are and
        how much mass is in them, so mass-weighted seeding samples the data as
        written instead of interpolating it onto a helper grid first. That
        removes the interpolation error and the midpoint rule, and is
        *cheaper* -- the helper route's cost is dominated by building an
        interpolator per sampled snapshot, not by evaluating it.

        Cells are described one at a time, by a lower and an upper bound along
        each axis, and nothing about how they are arranged comes back. One
        global grid, a union of meshblocks and an AMR hierarchy all flatten to
        the same pair of arrays, so the seeder needs to know nothing about the
        format's mesh. :func:`src.utils.tensor_cell_bounds` builds the pair
        from a separable grid's edges, which is the whole of the work for a
        format that has one.

        The only requirements are that the cells **tile the region without
        overlapping** -- otherwise the sampled mass is double counted where
        they do -- and that ghost cells are excluded.

        What comes back is a *mass*, not a field value. Read
        ``self.mass_density.density_keys`` (or ``flux_keys`` for a surface) out
        of shared memory and hand them to :meth:`cell_weights_from_values`,
        which does the rest. Nothing outside this class then has to know
        whether the right density here is ``rho``, the conserved ``D``, or
        something only this format can work out.

        Parameters
        ----------
        slot : int
            Shared-memory slot holding the snapshot to read, i.e. an index into
            ``self.shared_memory`` alongside ``cur_times``.
        surface_radius : float or None
            When None, return volume cells. Otherwise return the cells the
            sphere at exactly that radius passes through -- the one radial cell
            per angular patch whose extent contains it, half-open
            ``r_lo <= R < r_hi`` so a sphere on a face takes exactly one layer --
            with two axes instead of three. Nothing is snapped: the sphere is
            the one asked for, so it coincides exactly with the outer boundary
            of a volume seeded out to the same radius.

        The axes are those of :attr:`grid_geometry`, which a format
        implementing this must set. Every seeder in :mod:`src.seeds` works in
        ``'spherical'`` and refuses native cells declared in anything else.

        Returns
        -------
        lo, hi : ndarray, shape (D, n_cells)
            Lower and upper bound of every cell along every axis. For a
            ``'spherical'`` geometry: ``(r, cos(theta), phi)`` for a volume,
            ``(cos(theta), phi)`` on a surface. ``lo <= hi`` elementwise, whichever way the format's own
            axes run -- AthenaK's equal-solid-angle ``cos(theta)`` descends
            where GR-Athena++'s ascends, and sorting the pair here spares every
            caller the special case.
        weights : ndarray, shape (n_cells,)
            The mass each cell holds, or -- when `surface_radius` is given --
            the mass crossing the sphere through that cell's patch per unit
            time, signed so that an inflowing patch is negative. Pass the
            radius each surface value was sampled at to
            :meth:`cell_weights_from_values` as ``r_sample``; see there for why.

        Raises
        ------
        NotImplementedError
            If this format cannot enumerate its cells.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot enumerate its own cells; "
            "use --weight-grid helper."
        )

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
