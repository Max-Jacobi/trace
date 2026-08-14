"""
Tests for the AthenaK spherical-grid VTK reader and file handler.

The synthetic-file tests build small AthenaK-layout ``.vtk`` files from
scratch, so they run anywhere.  ``TestRealFiles`` additionally checks the
reader against a real AthenaK dump if one is present at
``ATHENAK_SAMPLE_DIR`` (set below), and is skipped otherwise.
"""

import os

import numpy as np
import pytest

from src.athenak import (
    FIELD_MAP,
    AthenaKFileHandler,
    polar_axis,
    read_grid,
    scan_vtk,
)
from src.utils import fill_spherical_ghosts
from src.interpolators import PchipInterpolator3D

ATHENAK_SAMPLE_DIR = os.environ.get(
    "ATHENAK_SAMPLE_DIR",
    os.path.expanduser(
        "~/Documents/Projects/athenaK_tracers/sph_example_files/"
        "Comp_DD2NQTtab_n128_b4097_BH8_NS8_MBH4.3_MNS1.6_d40_3e15G_intdip_M1"
    ),
)


# The two polar grid conventions the reader must handle: what AthenaK
# writes today, and the cell-centred uniform-theta grid we are asking for.
CONVENTIONS = ["mu_node", "theta_cell"]


def athenak_grid(n_r=12, n_th=10, n_ph=16, r_min=10.0, r_max=50.0,
                 convention="mu_node"):
    """Return the (r, theta, phi) axes of an AthenaK-style spherical grid."""
    r = np.linspace(r_min, r_max, n_r)
    if convention == "mu_node":
        # uniform in cos(theta), poles included, theta descending
        th = np.arccos(np.linspace(-1.0, 1.0, n_th))
        ph = np.arange(n_ph) * (2 * np.pi / n_ph)
    elif convention == "theta_cell":
        # uniform in theta, cell-centred, theta ascending, phi cell-centred
        th = (np.arange(n_th) + 0.5) * (np.pi / n_th)
        ph = (np.arange(n_ph) + 0.5) * (2 * np.pi / n_ph)
    else:
        raise ValueError(convention)
    return r, th, ph


def write_athenak_vtk(path, r, th, ph, time, fields):
    """
    Write one AthenaK-layout legacy binary VTK file.

    ``fields`` maps scalar name -> array of shape ``(n_r, n_th, n_ph)``.
    """
    n_r, n_th, n_ph = len(r), len(th), len(ph)
    # File order is phi slowest, r fastest.
    pts = np.empty((n_ph, n_th, n_r, 3), dtype=">f4")
    pts[..., 0] = r[None, None, :]
    pts[..., 1] = th[None, :, None]
    pts[..., 2] = ph[:, None, None]

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"# AthenaK data at time={time} cycle=0 nradii={n_r}\n".encode())
        f.write(b"BINARY\nDATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {n_r} {n_th} {n_ph}\n".encode())
        f.write(f"POINTS {n_r * n_th * n_ph} float\n".encode())
        f.write(pts.tobytes())
        f.write(b"\nFIELD FieldData 2\n")
        f.write(b"TIME 1 1 float\n")
        f.write(np.array([time], dtype=">f4").tobytes())
        f.write(b"\nRADII 1 %d float\n" % n_r)
        f.write(np.asarray(r, dtype=">f4").tobytes())
        f.write(f"\nPOINT_DATA {n_r * n_th * n_ph}\n".encode())
        for name, values in fields.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            # (n_r, n_th, n_ph) -> file order (n_ph, n_th, n_r)
            f.write(np.asarray(values, dtype=">f4").transpose(2, 1, 0).tobytes())
            f.write(b"\n")


def sample_field(r, th, ph, seed=0):
    """A smooth, non-separable field on the (r, theta, phi) grid."""
    R, TH, PH = np.meshgrid(r, th, ph, indexing="ij")
    x = R * np.sin(TH) * np.cos(PH)
    y = R * np.sin(TH) * np.sin(PH)
    z = R * np.cos(TH)
    return 1.0 + seed + 0.01 * x + 0.02 * y + 0.03 * z + 1e-4 * x * y


def make_dataset(tmp_path, times, keys=("dens",), n_r=12, n_th=10, n_ph=16,
                 convention="mu_node"):
    """Write one file per (time, key), the way AthenaK splits its output."""
    r, th, ph = athenak_grid(n_r, n_th, n_ph, convention=convention)
    for i_t, time in enumerate(times):
        for key in keys:
            write_athenak_vtk(
                tmp_path / f"bhns.{key}.{i_t:05d}.vtk", r, th, ph, time,
                {key: sample_field(r, th, ph, seed=hash(key) % 7) * (1 + 0.1 * i_t)},
            )
    return r, th, ph


class TestScanVtk:
    def test_reads_header_and_grid(self, tmp_path):
        r, th, ph = athenak_grid()
        path = str(tmp_path / "a.vtk")
        write_athenak_vtk(path, r, th, ph, 42.0, {"dens": sample_field(r, th, ph)})

        info = scan_vtk(path)
        assert info["dims"] == (len(r), len(th), len(ph))
        assert info["time"] == 42.0
        assert set(info["scalars"]) == {"dens"}

        r_out, th_out, ph_out = read_grid(path, info)
        np.testing.assert_allclose(r_out, r, rtol=1e-6)
        np.testing.assert_allclose(th_out, th, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ph_out, ph, rtol=1e-5, atol=1e-6)

    def test_multiple_scalars_keep_their_offsets(self, tmp_path):
        """A scalar written after another must still be addressable."""
        r, th, ph = athenak_grid()
        path = str(tmp_path / "b.vtk")
        f0 = sample_field(r, th, ph, seed=0)
        f1 = sample_field(r, th, ph, seed=3)
        write_athenak_vtk(path, r, th, ph, 1.0, {"weights": f0, "dens": f1})

        info = scan_vtk(path)
        assert list(info["scalars"]) == ["weights", "dens"]
        from src.athenak import read_block
        offset, dtype = info["scalars"]["dens"]
        got = read_block(path, offset, f1.size, dtype).reshape(f1.shape[::-1]).transpose(2, 1, 0)
        np.testing.assert_allclose(got, f1, rtol=1e-6)

    def test_rejects_non_vtk(self, tmp_path):
        path = tmp_path / "junk.vtk"
        path.write_bytes(b"# vtk DataFile Version 3.0\n# nope\nBINARY\n")
        with pytest.raises(ValueError):
            scan_vtk(str(path))


class TestFillSphericalGhosts:
    @pytest.mark.parametrize("ng", [1, 3])
    @pytest.mark.parametrize("node_centred", [True, False])
    def test_interior_and_phi_periodicity(self, ng, node_centred):
        n_r, n_th, n_ph = 3, 8, 12
        ar = np.arange(n_r * n_th * n_ph, dtype=float).reshape(n_r, n_th, n_ph)
        buf = np.zeros((n_r, n_th + 2 * ng, n_ph + 2 * ng))
        fill_spherical_ghosts(buf, ar, ng, node_centred=node_centred)

        np.testing.assert_array_equal(buf[:, ng:-ng, ng:-ng], ar)
        # phi ghosts wrap around the full 2*pi range, in order
        np.testing.assert_array_equal(buf[:, ng:-ng, :ng], ar[:, :, -ng:])
        np.testing.assert_array_equal(buf[:, ng:-ng, -ng:], ar[:, :, :ng])

    @pytest.mark.parametrize("ng", [1, 3])
    def test_cell_centred_pole_ghosts_are_the_exact_continuation(self, ng):
        """
        For a cell-centred polar grid the mirror is the exact analytic
        continuation of a field smooth on the sphere, so the ghost rows must
        match it to machine precision -- for *every* ghost row, not just the
        middle one (which is what a reversed fill order would still get right).
        """
        n_th, n_ph = 16, 32
        d_th = np.pi / n_th
        th = (np.arange(n_th) + 0.5) * d_th
        ph = (np.arange(n_ph) + 0.5) * (2 * np.pi / n_ph)

        def f(t, p):
            return np.sin(t) * np.cos(p) + 0.5 * np.cos(t) + 0.25 * np.sin(t) ** 2 * np.sin(2 * p)

        TH, PH = np.meshgrid(th, ph, indexing="ij")
        buf = np.zeros((n_th + 2 * ng, n_ph + 2 * ng))
        fill_spherical_ghosts(buf, f(TH, PH), ng, node_centred=False)

        for k in range(1, ng + 1):
            th_lo = -(k - 0.5) * d_th                       # ghost below theta = 0
            th_hi = np.pi + (k - 0.5) * d_th                # ghost beyond theta = pi
            np.testing.assert_allclose(buf[ng - k, ng:-ng], f(-th_lo, ph + np.pi), atol=1e-12)
            np.testing.assert_allclose(buf[-ng + k - 1, ng:-ng],
                                       f(2 * np.pi - th_hi, ph + np.pi), atol=1e-12)

    @pytest.mark.parametrize("ng", [1, 3])
    def test_node_centred_pole_ghosts_mirror_the_row_inside(self, ng):
        """
        With a node on the pole the mirror source shifts by one row, since
        the pole row is its own mirror image.
        """
        n_r, n_th, n_ph = 2, 9, 10
        rng = np.random.default_rng(0)
        ar = rng.normal(size=(n_r, n_th, n_ph))
        ar[:, 0, :] = ar[:, 0, 0][:, None]    # poles are phi-independent
        ar[:, -1, :] = ar[:, -1, 0][:, None]
        buf = np.zeros((n_r, n_th + 2 * ng, n_ph + 2 * ng))
        fill_spherical_ghosts(buf, ar, ng, node_centred=True)

        for k in range(1, ng + 1):
            np.testing.assert_allclose(
                buf[:, ng - k, ng:-ng], np.roll(ar[:, k, :], n_ph // 2, axis=-1)
            )
            np.testing.assert_allclose(
                buf[:, -ng + k - 1, ng:-ng], np.roll(ar[:, -1 - k, :], n_ph // 2, axis=-1)
            )

    def test_axisymmetric_field_is_continuous_across_the_pole(self, ng=3):
        """
        For a field that only depends on mu, the pole ghosts must reproduce
        the field's own reflection -- no phi-roll artefact should survive.
        """
        n_r, n_th, n_ph = 1, 11, 8
        mu = np.linspace(-1, 1, n_th)
        ar = np.broadcast_to((1 - mu**2)[None, :, None], (n_r, n_th, n_ph)).copy()
        buf = np.zeros((n_r, n_th + 2 * ng, n_ph + 2 * ng))
        fill_spherical_ghosts(buf, ar, ng, node_centred=True)
        for k in range(1, ng + 1):
            np.testing.assert_allclose(buf[:, ng - k, ng:-ng], 1 - mu[k] ** 2)


class TestPolarAxis:
    def test_detects_mu_node_centred(self):
        th = np.arccos(np.linspace(-1, 1, 32))          # descending
        name, nodes, flip, node_centred = polar_axis(th)
        assert (name, flip, node_centred) == ("mu", False, True)
        np.testing.assert_allclose(nodes, np.linspace(-1, 1, 32), atol=1e-12)

    def test_detects_theta_cell_centred(self):
        th = (np.arange(32) + 0.5) * (np.pi / 32)        # ascending
        name, nodes, flip, node_centred = polar_axis(th)
        assert (name, flip, node_centred) == ("theta", False, False)
        np.testing.assert_allclose(nodes, th, atol=1e-12)

    def test_flags_reversed_storage_order(self):
        th = ((np.arange(32) + 0.5) * (np.pi / 32))[::-1]
        name, nodes, flip, node_centred = polar_axis(th)
        assert (name, flip) == ("theta", True)
        assert nodes[0] < nodes[-1]

    def test_rejects_a_grid_uniform_in_neither(self):
        th = np.sort(np.arccos(np.linspace(-1, 1, 32)) ** 1.3)
        with pytest.raises(ValueError, match="uniform in neither"):
            polar_axis(th)


class TestFileHandler:
    def _handler(self, directory, keys, **kwargs):
        return AthenaKFileHandler(
            interpolator=PchipInterpolator3D,
            directory=str(directory),
            keys=list(keys),
            n_cpu=1,
            files_per_step=2,
            interpolator_kwargs={"max_cache_size_GB": 0.05},
            **kwargs,
        )

    def test_groups_one_file_per_variable_into_one_timestep(self, tmp_path):
        make_dataset(tmp_path, times=[0.0, 10.0], keys=("dens", "temperature"))
        fh = self._handler(tmp_path, ["rho", "T"])
        try:
            np.testing.assert_allclose(fh.times, [0.0, 10.0])
            # one entry per time, each holding both single-variable files
            assert all(len(files) == 2 for files in fh.files)
        finally:
            fh.free_shared_memory()

    def test_missing_key_raises(self, tmp_path):
        make_dataset(tmp_path, times=[0.0, 10.0], keys=("dens",))
        with pytest.raises(KeyError, match="V_u_x"):
            self._handler(tmp_path, ["rho", "V_u_x"])

    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_interpolates_grid_points_exactly(self, tmp_path, convention):
        """
        Round-trip check: reading, reshaping, ghost-filling and interpolating
        must return the stored value at a grid point.  This is what catches
        an axis-order or transpose mistake, and it has to hold for either
        polar grid convention.
        """
        r, th, ph = make_dataset(tmp_path, times=[0.0, 10.0], keys=("dens",),
                                 convention=convention)
        expected = sample_field(r, th, ph, seed=hash("dens") % 7)

        fh = self._handler(tmp_path, ["rho"])
        try:
            fh.load_chunk(0, forward=True)
            interp = AthenaKFileHandler.setup_interpolator(
                {"rho": fh.shared_memory[0]["rho"]}, fh.extra_data
            )
            interp.load()

            # Interior grid points only: the radial PCHIP does not extrapolate.
            i_r, i_th, i_ph = np.meshgrid(
                np.arange(1, len(r) - 1), np.arange(1, len(th) - 1), np.arange(len(ph)),
                indexing="ij",
            )
            i_r, i_th, i_ph = i_r.ravel(), i_th.ravel(), i_ph.ravel()
            rr, tt, pp = r[i_r], th[i_th], ph[i_ph]
            coords = np.array([
                rr * np.sin(tt) * np.cos(pp),
                rr * np.sin(tt) * np.sin(pp),
                rr * np.cos(tt),
            ])
            got = interp(coords)[0]
            np.testing.assert_allclose(got, expected[i_r, i_th, i_ph], rtol=1e-4)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_rejects_a_grid_uniform_in_neither_theta_nor_mu(self, tmp_path):
        """The interpolators index axis 1 uniformly, so this must not pass."""
        n_r, n_th, n_ph = 6, 8, 8
        r = np.linspace(10.0, 50.0, n_r)
        th = np.sort(np.arccos(np.linspace(-1, 1, n_th)) ** 1.3)
        ph = np.arange(n_ph) * (2 * np.pi / n_ph)
        write_athenak_vtk(
            str(tmp_path / "x.vtk"), r, th, ph, 0.0,
            {"dens": sample_field(r, th, ph)},
        )
        with pytest.raises(ValueError, match="uniform in neither"):
            self._handler(tmp_path, ["rho"])

    def test_cell_centred_theta_grid_is_accurate_at_the_pole(self, tmp_path):
        """
        The whole point of asking AthenaK for a cell-centred uniform-theta
        grid: near the pole a uniform-mu grid loses ~all its accuracy on a
        non-axisymmetric field, and a uniform-theta one does not.
        """
        n_r, n_th, n_ph = 6, 64, 64
        errors = {}
        for convention in CONVENTIONS:
            d = tmp_path / convention
            d.mkdir()
            r, th, ph = athenak_grid(n_r, n_th, n_ph, convention=convention)
            R, TH, PH = np.meshgrid(r, th, ph, indexing="ij")
            field = np.sin(TH) * np.cos(PH)           # m=1: the hard case
            for i_t, t in enumerate([0.0, 10.0]):
                write_athenak_vtk(str(d / f"a.{i_t}.vtk"), r, th, ph, t, {"dens": field})

            fh = self._handler(d, ["rho"])
            try:
                fh.load_chunk(0, forward=True)
                interp = AthenaKFileHandler.setup_interpolator(
                    {"rho": fh.shared_memory[0]["rho"]}, fh.extra_data
                )
                interp.load()
                # midway through the first cell off the pole, for each grid
                th_asc = np.sort(th)
                tq = 0.5 * th_asc[0] if th_asc[0] > 1e-12 else 0.5 * th_asc[1]
                pq, rq = 0.3, r[n_r // 2]
                coords = np.array([[rq * np.sin(tq) * np.cos(pq)],
                                   [rq * np.sin(tq) * np.sin(pq)],
                                   [rq * np.cos(tq)]])
                errors[convention] = abs(interp(coords)[0, 0] - np.sin(tq) * np.cos(pq))
                interp.unload()
            finally:
                fh.free_shared_memory()

        assert errors["theta_cell"] < 1e-4
        assert errors["theta_cell"] < 0.01 * errors["mu_node"]


@pytest.mark.skipif(
    not os.path.isdir(ATHENAK_SAMPLE_DIR),
    reason=f"no AthenaK sample data at {ATHENAK_SAMPLE_DIR}",
)
class TestRealFiles:
    """
    Asserts what must hold for *any* AthenaK dump rather than the geometry
    of whichever grid is on disk.  That has already changed once (linear
    radius with an equal-solid-angle polar axis, then geometric radius
    with a cell-centred uniform-theta one), and absorbing exactly that
    without being edited is the point of the reader.
    """

    def _path(self):
        import glob
        return sorted(glob.glob(f"{ATHENAK_SAMPLE_DIR}/*mhd_w_d*.vtk"))[0]

    def test_reads_a_real_dump(self):
        info = scan_vtk(self._path())
        n_r, n_th, n_ph = info["dims"]
        assert n_r > 1 and n_th > 1 and n_ph > 1
        assert info["n_points"] == n_r * n_th * n_ph
        assert info["time"] is not None
        assert FIELD_MAP["rho"] in info["scalars"]

    def test_grid_is_separable_and_supported(self):
        path = self._path()
        info = scan_vtk(path)
        _, _, n_ph = info["dims"]
        # read_grid raises unless POINTS is a separable (r, theta, phi) grid
        r, th, ph = read_grid(path, info)

        assert np.all(np.diff(r) > 0), "radius must be ascending"
        np.testing.assert_allclose(np.diff(ph), 2 * np.pi / n_ph, atol=1e-4)

        # polar_axis must recognise the convention, and the exactly uniform
        # nodes it returns must reproduce the file's own coordinates.
        name, nodes, flip, _ = polar_axis(th)
        assert name in ("theta", "mu")
        from_file = np.cos(th) if name == "mu" else th
        np.testing.assert_allclose(nodes, from_file[::-1] if flip else from_file, atol=1e-5)

    def test_every_field_is_present_at_every_time(self):
        """
        One variable per file makes it easy to end up with a variable
        dumped at a different cadence than the rest, which the pipeline
        rejects.  Fail here, with a readable message, instead of at the
        start of a production run.
        """
        import glob
        from collections import defaultdict
        per_time = defaultdict(set)
        for path in sorted(glob.glob(f"{ATHENAK_SAMPLE_DIR}/*.vtk")):
            info = scan_vtk(path)
            per_time[info["time"]] |= set(info["scalars"]) - {"weights"}
        assert per_time, "no AthenaK dumps found"
        everything = set().union(*per_time.values())
        gaps = {t: sorted(everything - have) for t, have in per_time.items()
                if have != everything}
        assert not gaps, f"fields missing at some times: {gaps}"


class TestUnusableSamples:
    """
    AthenaK's mean neutrino energies are J/n, so wherever the number
    density vanishes they carry no information -- arriving either as inf,
    or as a finite number up to the float32 ceiling depending on how far
    the denominator underflowed.  Both have to go, and substituting is
    right rather than masking, because the flux they multiply downstream is
    vanishing there anyway.
    """

    def _dataset(self, tmp_path, junk):
        r, th, ph = athenak_grid(12, 10, 16)
        field = sample_field(r, th, ph)
        broken = field.copy() * 1e-6          # a plausible mean-energy scale
        broken[5, 4, ::3] = junk
        for i_t, time in enumerate([0.0, 10.0]):
            write_athenak_vtk(str(tmp_path / f"a.{i_t}.vtk"), r, th, ph, time,
                              {"dens": field, "e:0": broken})
        return r, th, ph, field, broken

    def _load(self, tmp_path):
        fh = AthenaKFileHandler(
            interpolator=PchipInterpolator3D, directory=str(tmp_path),
            keys=["rho", "eps_nue"], n_cpu=1, files_per_step=2,
            interpolator_kwargs={"max_cache_size_GB": 0.05},
        )
        fh.load_chunk(0, forward=True)
        return fh

    @pytest.mark.parametrize(
        "junk", [np.inf, -np.inf, np.nan, 3.4e38, -3.4e38, 1e-2],
        ids=["inf", "-inf", "nan", "float32_max", "-float32_max", "merely_absurd"],
    )
    def test_unusable_samples_are_replaced(self, tmp_path, capsys, junk):
        """
        A finite but absurd value is just as unusable as an inf, and is the
        case a plain isfinite() check misses.
        """
        from src.athenak import SANE_FILL, _warned_nonfinite
        _warned_nonfinite.clear()
        r, th, ph, field, broken = self._dataset(tmp_path, junk)

        fh = self._load(tmp_path)
        try:
            assert "replaced" in capsys.readouterr().out
            ng = PchipInterpolator3D.n_ghosts
            from multiprocessing.shared_memory import SharedMemory
            shm = SharedMemory(name=fh.shared_memory[0]["eps_nue"])
            try:
                buf = np.ndarray(fh.extra_data["shape"], dtype=np.float64,
                                 buffer=shm.buf).copy()
            finally:
                shm.close()
        finally:
            fh.free_shared_memory()

        interior = buf[:, ng:-ng, ng:-ng]
        bad = ~np.isfinite(broken) | (np.abs(broken) > 1e-3)
        assert bad.any(), "the test data should contain unusable samples"
        np.testing.assert_array_equal(interior[bad], SANE_FILL)
        np.testing.assert_allclose(interior[~bad], broken[~bad], rtol=1e-5)

    def test_plausible_values_are_left_alone(self, tmp_path):
        """The clamp must not touch data inside the physical range."""
        from src.athenak import _warned_nonfinite
        _warned_nonfinite.clear()
        r, th, ph, field, broken = self._dataset(tmp_path, 1e-7)   # small, valid
        fh = self._load(tmp_path)
        try:
            ng = PchipInterpolator3D.n_ghosts
            from multiprocessing.shared_memory import SharedMemory
            shm = SharedMemory(name=fh.shared_memory[0]["eps_nue"])
            try:
                buf = np.ndarray(fh.extra_data["shape"], dtype=np.float64,
                                 buffer=shm.buf).copy()
            finally:
                shm.close()
        finally:
            fh.free_shared_memory()
        np.testing.assert_allclose(buf[:, ng:-ng, ng:-ng], broken, rtol=1e-5)

    def test_a_field_without_a_bound_keeps_large_values(self, tmp_path):
        """
        Only keys listed in FIELD_MAX_ABS are range-checked; a large density
        is a physical statement, not a broken sample.
        """
        from src.athenak import FIELD_MAX_ABS
        assert "rho" not in FIELD_MAX_ABS
        r, th, ph = athenak_grid(12, 10, 16)
        field = sample_field(r, th, ph)
        field[3, 3, 3] = 1e6
        for i_t, time in enumerate([0.0, 10.0]):
            write_athenak_vtk(str(tmp_path / f"a.{i_t}.vtk"), r, th, ph, time,
                              {"dens": field})
        fh = AthenaKFileHandler(
            interpolator=PchipInterpolator3D, directory=str(tmp_path),
            keys=["rho"], n_cpu=1, files_per_step=2,
            interpolator_kwargs={"max_cache_size_GB": 0.05},
        )
        try:
            fh.load_chunk(0, forward=True)
            ng = PchipInterpolator3D.n_ghosts
            from multiprocessing.shared_memory import SharedMemory
            shm = SharedMemory(name=fh.shared_memory[0]["rho"])
            try:
                buf = np.ndarray(fh.extra_data["shape"], dtype=np.float64,
                                 buffer=shm.buf).copy()
            finally:
                shm.close()
        finally:
            fh.free_shared_memory()
        assert buf[3, ng + 3, ng + 3] == pytest.approx(1e6, rel=1e-5)
