"""
Minimal reader + trilinear interpolator for PyCompOSE HDF5 EOS tables, so
post-processing can evaluate EOS-dependent quantities (entropy, enthalpy,
pressure, composition ...) along a tracer's recorded (rho, T, Ye) history
without depending on the external ``tabulatedEOS`` package.

Table layout (PyCompOSE): 1-D axes ``nb`` (baryon number density, fm^-3,
log-spaced), ``yq`` (charge fraction, linear) and ``t`` (temperature, MeV,
log-spaced); every 3-D dataset is shaped ``(nb, yq, t)``. Quantities
``Q1``..``Q7`` are the CompOSE thermodynamic columns, in nuclear units.

Inputs are given in the tracer's units: ``rho`` in geometric solar-mass
units (G = c = M_sun = 1), ``T`` in MeV and ``Ye`` dimensionless. Outputs
are in the table's nuclear units (MeV, fm, k_B), which for the usual
quantities means entropy in k_B/baryon, enthalpy and eps dimensionless and
pressure in MeV/fm^3.

Example
-------
    eos = PyCompOSEEOS("SFHo.h5")
    s = eos("entr", traj.data["rho"], traj.data["T"], traj.data["r_0"])
"""

from functools import cached_property

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# CODATA 2014, identical to tabulatedEOS.unit_system so both agree to rounding.
_C = 2.99792458e10          # cm/s
_G = 6.67408e-8             # cm^3 g^-1 s^-2
_MSUN = 1.98848e33          # g
_MEV = 1.6021766208e-6      # erg
GEOM_TO_CGS_DENSITY = _C**6 / (_G**3 * _MSUN**2)   # g/cm^3 per geometric-solar density unit
MEV_TO_G = _MEV / _C**2                             # g per MeV/c^2

DERIVED = {
    'entr': lambda f: f['Q2'][:],                                        # k_B / baryon
    'eps': lambda f: f['Q7'][:],                                         # e/(m_B n_B) - 1
    'pres': lambda f: f['Q1'][:] * f['nb'][:][:, None, None],            # MeV / fm^3
    'enth': lambda f: 1 + f['Q7'][:] + f['Q1'][:] / float(f['mn'][()]),  # (e + P)/(m_B n_B)
}


class PyCompOSEEOS:
    """Trilinear interpolation in (log10 nb, yq, log10 T) of a PyCompOSE table."""

    def __init__(self, path: str):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.mn = float(f['mn'][()])             # neutron mass, MeV
            self.axes = (np.log10(f['nb'][:]), f['yq'][:], np.log10(f['t'][:]))
            self._keys = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
        self._interp: dict[str, tuple[RegularGridInterpolator, bool]] = {}

    def keys(self) -> list[str]:
        return list(DERIVED) + self._keys

    @cached_property
    def h_inf(self) -> float:
        """Global table minimum of the specific enthalpy: the Bernoulli reference h at infinity."""
        with h5py.File(self.path, 'r') as f:
            return float(np.min(DERIVED['enth'](f)))

    def rho_to_nb(self, rho) -> np.ndarray:
        """Geometric-solar mass density -> baryon number density in fm^-3."""
        return np.asarray(rho, dtype=float) * GEOM_TO_CGS_DENSITY / (self.mn * MEV_TO_G) * 1e-39

    def _get(self, key: str):
        if key not in self._interp:
            with h5py.File(self.path, 'r') as f:
                data = DERIVED[key](f) if key in DERIVED else f[key][:]
            log = bool(np.all(data > 0))   # interpolate strictly positive quantities in log space
            rgi = RegularGridInterpolator(self.axes, np.log10(data) if log else data,
                                          bounds_error=False, fill_value=None)
            self._interp[key] = (rgi, log)
        return self._interp[key]

    def __call__(self, key: str, rho, T, Ye) -> np.ndarray:
        """Interpolate `key` at (rho [geometric], T [MeV], Ye); inputs are clipped to the table range."""
        rgi, log = self._get(key)
        pts = np.stack(np.broadcast_arrays(
            np.log10(self.rho_to_nb(rho)), np.asarray(Ye, dtype=float), np.log10(np.asarray(T, dtype=float))
        ), axis=-1)
        pts = np.clip(pts, [a[0] for a in self.axes], [a[-1] for a in self.axes])
        out = np.full(pts.shape[:-1], np.nan)
        ok = np.all(np.isfinite(pts), axis=-1)
        out[ok] = rgi(pts[ok])
        return 10**out if log else out
