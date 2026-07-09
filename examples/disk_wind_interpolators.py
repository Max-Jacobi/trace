"""
examples/disk_wind_interpolators.py

Comparison (d): same disk-wind ejecta scenario as examples/disk_wind.py,
but holding the time integrator fixed (RK4) and varying only the spatial
interpolator (Linear vs PCHIP).  This isolates the spatial interpolation
order's contribution to the error, decoupled from the choice of time
integrator -- see examples/disk_wind.py for that complementary comparison
(c): interpolator held fixed at PCHIP, integrator varied.

Run directly:

    PYTHONPATH=. python examples/disk_wind_interpolators.py

All physics, grid, tracer setup, and plotting machinery are reused from
examples/disk_wind.py; only the scheme registry differs.
"""

from src.integrators.rk4 import RK4
from src.interpolators.regular import RegularInterpolator3D
from src.interpolators.pchip import PchipInterpolator3D

from examples.disk_wind import run_comparison

SCHEMES = [
    ("Linear + RK4", RegularInterpolator3D, RK4(monotone=True)),
    ("PCHIP + RK4",  PchipInterpolator3D,   RK4(monotone=True)),
]


if __name__ == "__main__":
    run_comparison(SCHEMES, "disk_wind_interpolators")
