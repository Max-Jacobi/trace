"""
What a format's fields mean as a mass density.

Seeding weights every cell by the mass it holds, or by the mass crossing it
per unit time. Which field carries that, and what has to be done to it first,
is a property of the data and not of the seeding strategy: a GR run in
spherical coordinates needs the conserved ``D = rho*W*sqrt(gamma)``, a
Newtonian one needs ``rho`` and nothing else, and a run that dumps its own
metric should use that rather than an analytic stand-in.

So a :class:`~src.file.FileHandler` hands one of these out (see
``FileHandler.build_mass_density``) and nothing upstream -- no seeder, no
output writer -- has to know which case it is in. Instances are plain data and
pickle, so a worker pool can carry one.
"""

import numpy as np

from .utils import densitization_factor


class MassDensity:
    """
    Rest-mass density, densitized to the conserved ``D`` when ``adm_mass`` is
    given, from an analytic isotropic-Schwarzschild metric and the fluid's own
    ``u_t`` (see :func:`src.utils.densitization_factor`).

    This is the default every shipped format uses, all three being GR runs on a
    spherical grid. A format whose spacetime, coordinates or dumped fields need
    something else overrides ``FileHandler.build_mass_density`` and returns a
    subclass; the two methods below are the whole interface.

    ``density_keys`` and ``flux_keys`` say which fields to load, and the
    ``values`` passed back in must follow that order. They are separate so that
    volume seeding does not pay for reading velocities it never uses.
    """

    def __init__(
        self,
        density_key: str = 'rho',
        vel_keys: tuple[str, ...] = (),
        adm_mass: float | None = None,
        ut_key: str = 'u_t',
        ) -> None:
        self.density_key = density_key
        self.vel_keys = tuple(vel_keys)
        self.adm_mass = adm_mass
        self.ut_key = ut_key

        self.density_keys: tuple[str, ...] = (
            (density_key,) if adm_mass is None else (density_key, ut_key))
        self.flux_keys: tuple[str, ...] = (
            *self.density_keys,
            *(k for k in self.vel_keys if k not in self.density_keys))

    def density(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Mass per unit volume at `positions`, from `values` for `density_keys`.

        `values` is ``(len(density_keys), n)`` and `positions` is ``(3, n)``.
        """
        rho = values[self.density_keys.index(self.density_key)]
        if self.adm_mass is None:
            return rho
        r = np.linalg.norm(positions, axis=0)
        u_t = values[self.density_keys.index(self.ut_key)]
        return rho * densitization_factor(r, u_t, self.adm_mass)

    def radial_flux(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Radial mass flux per unit area and time, from `values` for `flux_keys`.

        Signed: an inflowing cell gives a negative flux, which the seeders rely
        on to subtract from the net crossing mass.
        """
        if len(self.vel_keys) != 3:
            raise ValueError(
                "A radial mass flux needs three velocity keys; this MassDensity "
                f"was built with {self.vel_keys!r}. Pass vel_keys to the "
                "FileHandler."
            )
        r = np.linalg.norm(positions, axis=0)
        v_r = sum(positions[i] * values[self.flux_keys.index(vk)]
                  for i, vk in enumerate(self.vel_keys)) / r
        return self.density(values[:len(self.density_keys)], positions) * v_r

    def describe(self) -> str:
        if self.adm_mass is None:
            return f"{self.density_key} (rest-mass density, not densitized)"
        return (f"D = {self.density_key}*W*sqrt(gamma) from {self.ut_key} and "
                f"M_ADM = {self.adm_mass:g}")
