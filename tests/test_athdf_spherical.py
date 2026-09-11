"""
Tests for the Athena++ athdf meshblock reader and interpolator.

The synthetic writer emulates what Athena++ writes with ghost_zones = true,
including the polar boundary conventions verified on real dumps: pole ghost
x2v coordinates mirrored back into the domain (non-monotonic rows), ghost
data taken from across the pole with vel2 AND vel3 sign-flipped.
"""

import numpy as np
from src.utils import cell_centres, densitization_factor
import h5py
import pytest

from src.athdf_spherical import (
    SphericalAthdfFileHandler,
    _fix_polar_ghost_nodes,
    _root_faces,
    _transport_velocity,
)
from src.interpolators import MeshblockPchipInterpolator

NG = 2  # ghost cells the synthetic files carry


# ---------------------------------------------------------------------------
# synthetic athdf writer

def _to_physical(th, ph):
    """Map extended-chart angles to physical ones + polar vector signs."""
    s = np.ones_like(th)
    thp, php = th.copy(), ph.copy()
    for mask, refl in ((th < 0, -th), (th > np.pi, 2 * np.pi - th)):
        thp = np.where(mask, refl, thp)
        php = np.where(mask, ph + np.pi, php)
        s = np.where(mask, -1.0, s)
    return thp, php % (2 * np.pi), s


def _block_axes(spec, n_root, loc, level, n_int, ng):
    lo, hi = spec[0], spec[1]
    d = (hi - lo) / (n_root * 2**level)
    faces = lo + d * (loc * n_int + np.arange(-ng, n_int + ng + 1))
    return faces, 0.5 * (faces[:-1] + faces[1:])


def write_athdf(
    path, time, fields_by_dataset, *,
    root_size=(8, 8, 8),
    r_lim=(10.0, 50.0),
    levels=None,
    locations=None,
    interior=(4, 4, 4),
    ng=NG,
    max_level_attr=None,
    ):
    """
    Write one synthetic athdf file.

    ``fields_by_dataset``: dict dataset name -> dict raw variable name ->
    ``f(r, theta, phi)`` evaluated on physical coordinates.  ``vel2``/``vel3``
    automatically receive the polar-boundary sign flip in mirrored ghosts.
    """
    root_size = np.asarray(root_size)
    interior = np.asarray(interior)
    n_rb = root_size // interior
    if levels is None:
        levels = np.zeros(int(np.prod(n_rb)), dtype=int)
        locations = np.array(
            [[i, j, k]
             for k in range(n_rb[2]) for j in range(n_rb[1]) for i in range(n_rb[0])]
        )
    levels = np.asarray(levels)
    locations = np.asarray(locations)
    n_blocks = len(levels)
    mb = interior + 2 * ng
    specs = [np.array([r_lim[0], r_lim[1], 1.0]),
             np.array([0.0, np.pi, 1.0]),
             np.array([0.0, 2 * np.pi, 1.0])]

    faces = np.empty((3, n_blocks, mb.max() + 1))
    centers = np.empty((3, n_blocks, mb.max()))
    for b in range(n_blocks):
        for a in range(3):
            f_a, c_a = _block_axes(specs[a], root_size[a], locations[b, a],
                                   levels[b], interior[a], ng)
            faces[a, b, :mb[a] + 1] = f_a
            centers[a, b, :mb[a]] = c_a

    # data on the (block, i, j, k) chart grid, then to file order (block, k, j, i)
    r = centers[0][:, :mb[0], None, None]
    th = centers[1][:, None, :mb[1], None]
    ph = centers[2][:, None, None, :mb[2]]
    thp, php, sign = _to_physical(np.broadcast_to(th, (n_blocks, *mb)),
                                  np.broadcast_to(ph, (n_blocks, *mb)))
    rb = np.broadcast_to(r, (n_blocks, *mb))

    with h5py.File(path, 'w') as f:
        names, n_vars, dsets = [], [], []
        for dset, fields in fields_by_dataset.items():
            data = np.empty((len(fields), n_blocks, mb[2], mb[1], mb[0]), dtype='f4')
            for v, (name, fn) in enumerate(fields.items()):
                values = fn(rb, thp, php)
                if name in ('vel2', 'vel3'):
                    values = sign * values
                data[v] = values.transpose(0, 3, 2, 1)
                names.append(name)
            n_vars.append(len(fields))
            dsets.append(dset)
            f.create_dataset(dset, data=data)

        # file x2v: pole ghost centres mirrored back into the domain
        x2v = centers[1][:, :mb[1]].copy()
        x2v[x2v < 0] *= -1
        x2v[x2v > np.pi] = 2 * np.pi - x2v[x2v > np.pi]

        f.create_dataset('Levels', data=levels.astype('i4'))
        f.create_dataset('LogicalLocations', data=locations.astype('i8'))
        f.create_dataset('x1f', data=faces[0][:, :mb[0] + 1].astype('f4'))
        f.create_dataset('x2f', data=faces[1][:, :mb[1] + 1].astype('f4'))
        f.create_dataset('x3f', data=faces[2][:, :mb[2] + 1].astype('f4'))
        f.create_dataset('x1v', data=centers[0][:, :mb[0]].astype('f4'))
        f.create_dataset('x2v', data=x2v.astype('f4'))
        f.create_dataset('x3v', data=centers[2][:, :mb[2]].astype('f4'))

        f.attrs['Coordinates'] = np.bytes_('schwarzschild')
        f.attrs['Time'] = np.float32(time)
        f.attrs['MaxLevel'] = np.int32(
            levels.max() if max_level_attr is None else max_level_attr)
        f.attrs['NumMeshBlocks'] = np.int32(n_blocks)
        f.attrs['MeshBlockSize'] = mb.astype('i4')
        f.attrs['RootGridSize'] = root_size.astype('i4')
        for a in range(3):
            f.attrs[f'RootGridX{a + 1}'] = specs[a].astype('f8')
        f.attrs['DatasetNames'] = np.array(dsets, dtype='S21')
        f.attrs['NumVariables'] = np.array(n_vars, dtype='i4')
        f.attrs['VariableNames'] = np.array(names, dtype='S21')


def smooth_field(r, th, ph):
    """Cartesian-smooth scalar, automatically consistent across the pole."""
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return 1.0 + 0.01 * x + 0.02 * y - 0.015 * z


def uniform_velocity_fields(vx, vy, vz):
    """
    utilde^i fields (coordinate basis, flat space) for a constant Cartesian
    velocity -- _transport_velocity(bh_mass=0) must return the constants.
    """
    W = 1.0 / np.sqrt(1.0 - (vx**2 + vy**2 + vz**2))

    def vel1(r, th, ph):
        return W * (np.sin(th) * np.cos(ph) * vx
                    + np.sin(th) * np.sin(ph) * vy + np.cos(th) * vz)

    def vel2(r, th, ph):
        return W * (np.cos(th) * np.cos(ph) * vx
                    + np.cos(th) * np.sin(ph) * vy - np.sin(th) * vz) / r

    def vel3(r, th, ph):
        return W * (-np.sin(ph) * vx + np.cos(ph) * vy) / (r * np.sin(th))

    return {'vel1': vel1, 'vel2': vel2, 'vel3': vel3}


def sph_to_cart(r, th, ph):
    return np.array([r * np.sin(th) * np.cos(ph),
                     r * np.sin(th) * np.sin(ph),
                     r * np.cos(th)])


# ---------------------------------------------------------------------------
# unit tests

class TestRootFaces:
    def test_uniform(self):
        np.testing.assert_allclose(_root_faces((1.0, 3.0, 1.0), 4),
                                   [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_geometric_matches_cumulative_construction(self):
        lo, hi, ratio, n = 300.0, 2e5, 1.035, 256
        faces = _root_faces((lo, hi, ratio), n)
        dx = np.diff(faces)
        np.testing.assert_allclose(dx[1:] / dx[:-1], ratio, rtol=1e-12)
        assert faces[0] == lo
        np.testing.assert_allclose(faces[-1], hi, rtol=1e-12)


class TestPolarGhostFix:
    def test_reflects_only_non_monotone_rows(self):
        n_int, ng = 4, 2
        d = np.pi / 8
        faces = -ng * d + d * np.arange(n_int + 2 * ng + 1)  # pole block at 0
        centres = 0.5 * (faces[:-1] + faces[1:])
        file_row = centres.copy()
        file_row[:ng] = -file_row[:ng]  # mirrored, non-monotone
        interior_row = centres + 10 * d  # a block away from the pole
        x2v = np.stack([file_row, interior_row])
        x2f = np.stack([faces, faces + 10 * d])
        fixed, pole_lo, pole_hi = _fix_polar_ghost_nodes(x2v, x2f, ng)
        np.testing.assert_allclose(fixed[0], centres, atol=1e-14)
        np.testing.assert_allclose(fixed[1], interior_row)
        assert list(pole_lo) == [True, False]
        assert not pole_hi.any()


class TestTransportVelocity:
    def test_pure_radial_schwarzschild(self):
        r = np.array([4.0])
        th = np.array([np.pi / 3])
        ph = np.array([1.0])
        u0, M = 0.7, 1.0
        alpha2 = 1 - 2 * M / r
        W = np.sqrt(1 + u0**2 / alpha2)
        vx, vy, vz = _transport_velocity(
            np.array([u0]), np.zeros(1), np.zeros(1), r, th, ph, M)
        v_expected = np.sqrt(alpha2) / W * u0
        np.testing.assert_allclose(
            [vx[0], vy[0], vz[0]],
            v_expected * np.array([np.sin(th) * np.cos(ph),
                                   np.sin(th) * np.sin(ph),
                                   np.cos(th)]).ravel())

    def test_uniform_cartesian_velocity_flat_space(self):
        rng = np.random.default_rng(0)
        r = rng.uniform(10, 50, 40)
        th = rng.uniform(0.05, np.pi - 0.05, 40)
        ph = rng.uniform(0, 2 * np.pi, 40)
        v0 = (0.1, -0.2, 0.15)
        u = uniform_velocity_fields(*v0)
        vx, vy, vz = _transport_velocity(
            u['vel1'](r, th, ph), u['vel2'](r, th, ph), u['vel3'](r, th, ph),
            r, th, ph, bh_mass=0.0)
        np.testing.assert_allclose(vx, v0[0], rtol=1e-12)
        np.testing.assert_allclose(vy, v0[1], rtol=1e-12)
        np.testing.assert_allclose(vz, v0[2], rtol=1e-12)


# ---------------------------------------------------------------------------
# handler + interpolator tests

class TestFileHandler:
    def _handler(self, directory, keys, **kwargs):
        return SphericalAthdfFileHandler(
            interpolator=MeshblockPchipInterpolator,
            directory=str(directory),
            keys=list(keys),
            n_cpu=1,
            files_per_step=2,
            bh_mass=0.0,
            **kwargs,
        )

    def _loaded(self, fh, step=0):
        fh.load_chunk(step, True)
        interp = fh.setup_interpolators(fh.keys, fh.shared_memory, fh.extra_data)[0]
        interp.load()
        return interp

    def test_multi_file_step_merging_and_series_autodetection(self, tmp_path):
        # per time: prim file, user_out_var file, and a cons file that
        # carries no requested key and must be skipped
        for i, t in enumerate([0.0, 10.0]):
            write_athdf(tmp_path / f'sim.out1.{i:05d}.athdf', t,
                        {'prim': {'rho': smooth_field}})
            write_athdf(tmp_path / f'sim.out3.{i:05d}.athdf', t,
                        {'user_out_var': {'Temperature': smooth_field}})
            write_athdf(tmp_path / f'sim.out2.{i:05d}.athdf', t,
                        {'cons': {'dens': smooth_field}})
        fh = self._handler(tmp_path, ['rho', 'T'])
        try:
            np.testing.assert_allclose(fh.times, [0.0, 10.0])
            assert all(len(files) == 2 for files in fh.files)
        finally:
            fh.free_shared_memory()

    def test_duplicated_time_prefers_later_file(self, tmp_path):
        # a restarted segment writing the same snapshot time must supersede
        # the original: the lexically later path wins
        write_athdf(tmp_path / 'a.00000.athdf', 0.0,
                    {'prim': {'rho': lambda r, th, ph: np.full_like(r, 1.0)}})
        write_athdf(tmp_path / 'b.00000.athdf', 0.0,
                    {'prim': {'rho': lambda r, th, ph: np.full_like(r, 2.0)}})
        fh = self._handler(tmp_path, ['rho'])
        try:
            interp = self._loaded(fh)
            np.testing.assert_array_equal(interp.interpolator.data['rho'], 2.0)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_missing_key_raises(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        with pytest.raises(KeyError, match='T'):
            self._handler(tmp_path, ['rho', 'T'])

    def test_rad_transform_raises(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        with pytest.raises(ValueError, match='rad-transform'):
            self._handler(tmp_path, ['rho'], rad_transform='log')

    def test_interpolates_cell_centres_exactly(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        fh = self._handler(tmp_path, ['rho'])
        try:
            interp = self._loaded(fh)
            x1v, x2v, x3v = (fh.extra_data[k] for k in ('x1v', 'x2v', 'x3v'))
            rng = np.random.default_rng(2)
            n_blocks, nb = x1v.shape
            b = rng.integers(0, n_blocks, 50)
            i, j, k = (rng.integers(NG, nb - NG, 50) for _ in range(3))
            r, th, ph = x1v[b, i], x2v[b, j], x3v[b, k]
            vals = interp(sph_to_cart(r, th, ph))
            # data and coordinates are stored float32, so exactness means
            # float32 precision here
            expected = np.float32(smooth_field(r, th, ph)).astype(np.float64)
            np.testing.assert_allclose(vals[0], expected, rtol=1e-6)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_smooth_field_accurate_everywhere(self, tmp_path):
        # off-node queries incl. exactly on block faces, near both poles,
        # and just either side of the phi seam
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        fh = self._handler(tmp_path, ['rho'])
        try:
            interp = self._loaded(fh)
            rng = np.random.default_rng(3)
            r = rng.uniform(10.0, 50.0, 200)
            th = rng.uniform(0.0, np.pi, 200)
            ph = rng.uniform(0.0, 2 * np.pi, 200)
            r[0], th[1], th[2] = 30.0, 1e-6, np.pi - 1e-6  # face / poles
            ph[3], ph[4] = 1e-9, 2 * np.pi - 1e-9          # phi seam
            vals = interp(sph_to_cart(r, th, ph))
            expected = smooth_field(r, th, ph)
            assert np.isfinite(vals).all()
            # the field is trigonometric in (theta, phi); on this deliberately
            # coarse 8-cell grid PCHIP truncation error is ~1e-2 relative
            np.testing.assert_allclose(vals[0], expected, rtol=2e-2, atol=2e-2)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_out_of_domain_is_nan(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        fh = self._handler(tmp_path, ['rho'])
        try:
            interp = self._loaded(fh)
            coords = np.array([[5.0, 0.0, 0.0], [60.0, 0.0, 0.0],
                               [np.nan, 0.0, 0.0]]).T
            assert np.isnan(interp(coords)).all()
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_uniform_velocity_is_uniform_across_poles_and_blocks(self, tmp_path):
        # exercises the whole velocity chain: W factors, rotation, and the
        # undoing of the polar boundary's vel3 sign flip in ghost cells
        v0 = (0.1, -0.2, 0.15)
        write_athdf(tmp_path / 's.00000.athdf', 0.0,
                    {'prim': dict(rho=smooth_field, **uniform_velocity_fields(*v0))})
        fh = self._handler(tmp_path, ['rho', 'V_u_x', 'V_u_y', 'V_u_z'])
        try:
            interp = self._loaded(fh)
            # every stored sample, ghosts included, must be the constant
            mi = interp.interpolator
            for key, expect in zip(('V_u_x', 'V_u_y', 'V_u_z'), v0):
                np.testing.assert_allclose(mi.data[key], expect, rtol=2e-5,
                                           err_msg=key)
            # and so must interpolated values near the poles
            rng = np.random.default_rng(4)
            r = rng.uniform(15, 45, 50)
            th = np.concatenate([rng.uniform(0, 0.05, 25),
                                 rng.uniform(np.pi - 0.05, np.pi, 25)])
            ph = rng.uniform(0, 2 * np.pi, 50)
            vals = interp(sph_to_cart(r, th, ph))
            for key, expect in zip(('V_u_x', 'V_u_y', 'V_u_z'), v0):
                np.testing.assert_allclose(vals[interp.keys.index(key)], expect,
                                           rtol=2e-4, atol=1e-5, err_msg=key)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_regridding_between_steps_raises(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}})
        levels = np.zeros(8 + 7, dtype=int)
        levels[8:] = 1  # bogus layout change at the second time
        locs = [[i, j, k] for k in range(2) for j in range(2) for i in range(2)]
        child = [[i, j, k] for k in range(2) for j in range(2) for i in range(2)]
        locs = np.array(locs[:-1] + [[c[0] + 2, c[1] + 2, c[2] + 2] for c in child])
        write_athdf(tmp_path / 's.00001.athdf', 10.0, {'prim': {'rho': smooth_field}},
                    levels=levels, locations=locs)
        fh = self._handler(tmp_path, ['rho'])
        try:
            with pytest.raises(Exception, match='layout'):
                fh.load_chunk(0, True)
        finally:
            fh.free_shared_memory()


class TestOctree:
    def _amr_layout(self):
        # 2x2x2 root blocks; root block (0,0,0) refined into 8 children
        levels = [0] * 7 + [1] * 8
        locs = [[i, j, k]
                for k in range(2) for j in range(2) for i in range(2)][1:]  # skip (0,0,0)
        locs += [[i, j, k] for k in range(2) for j in range(2) for i in range(2)]
        return np.array(levels), np.array(locs)

    def test_two_level_block_map_and_interpolation(self, tmp_path):
        levels, locs = self._amr_layout()
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}},
                    levels=levels, locations=locs)
        fh = SphericalAthdfFileHandler(
            interpolator=MeshblockPchipInterpolator, directory=str(tmp_path),
            keys=['rho'], n_cpu=1, files_per_step=1, bh_mass=0.0)
        try:
            bm = fh.extra_data['block_map']
            assert bm.shape == (4, 4, 4)
            # coarse block (1,0,0) at level 0 fills a 2x2x2 span
            assert (bm[2:4, 0:2, 0:2] == 0).all()
            # refined children each fill exactly one fine slot
            assert sorted(bm[0:2, 0:2, 0:2].ravel()) == list(range(7, 15))

            fh.load_chunk(0, True)
            interp = fh.setup_interpolators(fh.keys, fh.shared_memory,
                                            fh.extra_data)[0]
            interp.load()
            # block search lands in the right block for fine + coarse points
            mi = interp.interpolator
            x1v = fh.extra_data['x1v']
            probes = [(7, NG + 1), (0, NG + 1)]  # (block id, radial cell)
            for b, i in probes:
                rq = np.array([x1v[b, i]])
                thq = np.array([fh.extra_data['x2v'][b, NG + 1]])
                phq = np.array([fh.extra_data['x3v'][b, NG + 1]])
                bid, valid = mi._find_blocks(rq, thq, phq)
                assert valid[0] and bid[0] == b
            # smooth field stays accurate across the coarse-fine boundary
            rng = np.random.default_rng(5)
            r = rng.uniform(10.0, 30.0, 100)  # around the refined octant
            th = rng.uniform(0.1, np.pi / 2, 100)
            ph = rng.uniform(0.0, np.pi / 2, 100)
            vals = interp(sph_to_cart(r, th, ph))
            np.testing.assert_allclose(vals[0], smooth_field(r, th, ph),
                                       rtol=2e-2, atol=2e-2)
            interp.unload()
        finally:
            fh.free_shared_memory()

    def test_indivisible_interior_raises(self, tmp_path):
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': {'rho': smooth_field}},
                    max_level_attr=3)  # 2^3 does not divide interior 4
        with pytest.raises(ValueError, match='divisible'):
            SphericalAthdfFileHandler(
                interpolator=MeshblockPchipInterpolator, directory=str(tmp_path),
                keys=['rho'], n_cpu=1, files_per_step=1, bh_mass=0.0)


class TestNativeCellWeights:
    """
    The native weight grid for the mass-weighted seeding modes.

    Meshblocks arrive flattened into one list of cells, so the tests that
    matter are about those cells covering the region exactly once and about the
    weight being the cell's mass rather than a raw field. Getting bounds or
    ghosts wrong scales every sampled mass by the same factor, and no
    downstream check would notice.
    """

    R_LIM = (10.0, 50.0)
    INTERIOR = (4, 4, 4)
    VEL = ('V_u_x', 'V_u_y', 'V_u_z')

    def _loaded(self, tmp_path, levels=None, locations=None, with_velocity=False,
                **kwargs):
        fields = {'rho': smooth_field}
        keys = ['rho']
        if with_velocity:
            fields.update({f'vel{i}': smooth_field for i in (1, 2, 3)})
            keys += list(self.VEL)
            kwargs.setdefault('vel_keys', self.VEL)
        write_athdf(tmp_path / 's.00000.athdf', 0.0, {'prim': fields},
                    levels=levels, locations=locations)
        fh = SphericalAthdfFileHandler(
            interpolator=MeshblockPchipInterpolator, directory=str(tmp_path),
            keys=keys, n_cpu=1, files_per_step=1, bh_mass=0.0, **kwargs)
        fh.load_chunk(0, True)
        return fh

    @staticmethod
    def _volume(lo, hi):
        return float((((hi[0] ** 3 - lo[0] ** 3) / 3)
                      * (hi[1] - lo[1]) * (hi[2] - lo[2])).sum())

    @staticmethod
    def _at_nodes(fh, keys, i_r=None):
        """`keys` at every interior node, block by block, via the interpolator."""
        ng = fh.extra_data['file_ng']
        i1, i2, i3 = fh.extra_data['interior']
        r = fh.extra_data['x1v'][:, ng:ng + i1]
        th = fh.extra_data['x2v'][:, ng:ng + i2]
        ph = fh.extra_data['x3v'][:, ng:ng + i3]
        blocks = range(fh.extra_data['shape'][0]) if i_r is None else i_r.keys()
        pts = []
        for b in blocks:
            r_b = r[b] if i_r is None else r[b, i_r[b]:i_r[b] + 1]
            rg, tg, pg = np.meshgrid(r_b, th[b], ph[b], indexing='ij')
            pts.append(np.array(sph_to_cart(rg.ravel(), tg.ravel(), pg.ravel())))
        pos = np.concatenate(pts, axis=1)
        interps = fh.setup_interpolators(list(keys), fh.shared_memory,
                                         fh.extra_data)
        out = []
        for interp in interps:
            interp.load()
            try:
                out.append(interp(pos))
            finally:
                interp.unload()
        return np.concatenate(out, axis=0), pos

    def test_cells_tile_the_domain_volume(self, tmp_path):
        """
        The cells together must fill the shell exactly once. Tolerance is
        1e-6, not machine epsilon: the writer stores faces as float32, as real
        Athena++ dumps do, so faces shared between blocks telescope only to
        float32 precision.
        """
        fh = self._loaded(tmp_path)
        try:
            lo, hi, w = fh.native_cell_weights(0)
            assert lo.shape == hi.shape == (3, w.size)
            assert np.all(hi >= lo)
            r_lo, r_hi = self.R_LIM
            exact = 4 * np.pi * (r_hi ** 3 - r_lo ** 3) / 3
            assert self._volume(lo, hi) == pytest.approx(exact, rel=1e-6)
        finally:
            fh.free_shared_memory()

    def test_multilevel_blocks_tile_exactly(self, tmp_path):
        """
        The one test that catches a coarse/fine double count or gap. A parent
        block left in alongside its children would inflate this sum by its own
        volume, and nothing else in the pipeline would flag it.
        """
        levels, locs = TestOctree()._amr_layout()
        fh = self._loaded(tmp_path, levels=levels, locations=locs)
        try:
            lo, hi, w = fh.native_cell_weights(0)
            # 7 coarse blocks plus 8 refined children, all of interior size
            assert w.size == 15 * int(np.prod(self.INTERIOR))
            r_lo, r_hi = self.R_LIM
            exact = 4 * np.pi * (r_hi ** 3 - r_lo ** 3) / 3
            assert self._volume(lo, hi) == pytest.approx(exact, rel=1e-6)
        finally:
            fh.free_shared_memory()

    def test_ghost_cells_are_excluded(self, tmp_path):
        """
        A leaked ghost layer pushes cells outside the domain, so the global
        min/max radius is the sharpest single signal for it, and the cell
        count is the other half of the same check.
        """
        fh = self._loaded(tmp_path)
        try:
            lo, hi, w = fh.native_cell_weights(0)
            n_blocks = fh.extra_data['shape'][0]
            assert w.shape == (n_blocks * int(np.prod(self.INTERIOR)),)
            assert lo[0].min() == pytest.approx(self.R_LIM[0], rel=1e-6)
            assert hi[0].max() == pytest.approx(self.R_LIM[1], rel=1e-6)
        finally:
            fh.free_shared_memory()

    @pytest.mark.parametrize("r_surf", [30.0, 25.0])
    def test_surface_cells_cover_the_sphere(self, tmp_path, r_surf):
        """
        30.0 is the headline case: it sits exactly on an interior block face,
        where a left/right mixup in the radial search takes two radial layers
        instead of one and doubles the solid angle.
        """
        fh = self._loaded(tmp_path, with_velocity=True)
        try:
            lo, hi, w = fh.native_cell_weights(0, surface_radius=r_surf)
            assert lo.shape == hi.shape == (2, w.size)
            solid = float(((hi[0] - lo[0]) * (hi[1] - lo[1])).sum())
            assert solid == pytest.approx(4 * np.pi, rel=1e-6)
        finally:
            fh.free_shared_memory()

    def test_weight_is_the_cell_mass(self, tmp_path):
        """
        The point of going native is skipping the interpolator, so the weight
        must equal density-at-the-node times cell volume: PCHIP reproduces its
        own nodes, and the cell volume comes from the block's own faces.
        """
        fh = self._loaded(tmp_path)
        try:
            lo, hi, w = fh.native_cell_weights(0)
            rho, _ = self._at_nodes(fh, ('rho',))
            dV = (hi[0] ** 3 - lo[0] ** 3) / 3 * (hi[1] - lo[1]) * (hi[2] - lo[2])
            np.testing.assert_allclose(w, rho[0] * dV, rtol=1e-6)
        finally:
            fh.free_shared_memory()

    def test_an_adm_mass_switches_the_weight_to_the_conserved_density(self, tmp_path):
        """
        The rho-vs-D decision is the handler's, and the hook is the only place
        it shows: same call, same cells, a weight scaled by W*sqrt(gamma).
        """
        write_athdf(tmp_path / 's.00000.athdf', 0.0,
                    {'prim': {'rho': smooth_field,
                              'u_t': lambda r, th, ph: np.full_like(r, -1.05)}})
        common = dict(interpolator=MeshblockPchipInterpolator,
                      directory=str(tmp_path), keys=['rho', 'u_t'],
                      n_cpu=1, files_per_step=1, bh_mass=0.0)
        plain = SphericalAthdfFileHandler(**common)
        dens = SphericalAthdfFileHandler(adm_mass=2.7, **common)
        try:
            plain.load_chunk(0, True)
            dens.load_chunk(0, True)
            lo, hi, w_plain = plain.native_cell_weights(0)
            _, _, w_dens = dens.native_cell_weights(0)
            u_t, _ = self._at_nodes(plain, ('rho', 'u_t'))
            want = w_plain * densitization_factor((lo[0] + hi[0]) / 2, u_t[1], 2.7)
            np.testing.assert_allclose(w_dens, want, rtol=1e-6)
            assert np.all(np.abs(w_dens) > np.abs(w_plain))
        finally:
            plain.free_shared_memory()
            dens.free_shared_memory()

    @pytest.mark.parametrize("r_surf", [30.0, 25.0, 27.5])
    def test_surface_weight_is_the_flux_through_the_containing_cell(self, tmp_path, r_surf):
        """
        The solid-angle test alone cannot see a wrong *radial* layer: it still
        returns one cell per angular patch either way. Pin it by value against
        the flux at the containing cell's own node, which fixes both the layer
        and the density the weight was built from.
        """
        fh = self._loaded(tmp_path, with_velocity=True)
        try:
            lo, hi, w = fh.native_cell_weights(0, surface_radius=r_surf)
            ng = fh.extra_data['file_ng']
            x1f = fh.extra_data['x1f']
            i_r = {}
            for b in range(fh.extra_data['shape'][0]):
                r_e = x1f[b, ng:ng + self.INTERIOR[0] + 1]
                if not (r_e[0] <= r_surf < r_e[-1]):
                    continue
                j = int(np.searchsorted(r_e, r_surf, 'right')) - 1
                assert r_e[j] <= r_surf < r_e[j + 1]   # straddles the sphere
                i_r[b] = j
            assert len(i_r) * self.INTERIOR[1] * self.INTERIOR[2] == w.size

            vals, _ = self._at_nodes(fh, ('rho', *self.VEL), i_r=i_r)
            # The flux is taken where each value was sampled -- r**2 rho v_r at
            # the block's own x1v -- and carried to the sphere unchanged. The
            # projection direction is the cell's centre in (cos theta, phi).
            x1v = fh.extra_data['x1v']
            r_s = np.repeat([x1v[b, ng + j] for b, j in i_r.items()],
                            self.INTERIOR[1] * self.INTERIOR[2])
            pos = cell_centres(lo, hi, r_s)
            v_r = sum(pos[i] * vals[1 + i] for i in range(3)) / r_s
            dA = r_s ** 2 * (hi[0] - lo[0]) * (hi[1] - lo[1])
            np.testing.assert_allclose(w, vals[0] * v_r * dA, rtol=1e-6)
        finally:
            fh.free_shared_memory()

    @pytest.mark.parametrize("amr", [False, True])
    @pytest.mark.parametrize("r_edge", [27.5, 30.0])
    def test_a_clipped_volume_ends_exactly_on_the_sphere(self, tmp_path, amr, r_edge):
        """
        The volume a seeder takes out to R must be bounded by exactly the
        sphere a surface seeder uses at R -- no half cell either way, at any
        refinement level. 27.5 sits inside a cell, 30.0 on a block face.
        """
        from src.seeds import _native_cells
        levels, locs = TestOctree()._amr_layout() if amr else (None, None)
        fh = self._loaded(tmp_path, levels=levels, locations=locs)
        try:
            r_lo = self.R_LIM[0]
            cells, _ = _native_cells(fh, 0, ranges=((r_lo, r_edge), (-1.0, 1.0),
                                                    (0.0, 2 * np.pi)))
            assert cells.hi[0].max() == pytest.approx(r_edge, rel=1e-12)
            exact = 4 * np.pi * (r_edge ** 3 - r_lo ** 3) / 3
            assert cells.measure().sum() == pytest.approx(exact, rel=1e-6)
        finally:
            fh.free_shared_memory()
