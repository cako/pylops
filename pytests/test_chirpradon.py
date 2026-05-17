import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"

import pytest

from pylops.optimization.sparsity import fista
from pylops.signalprocessing import ChirpRadon2D, ChirpRadon3D
from pylops.utils import dottest
from pylops.utils.seismicevents import linear2d, linear3d, makeaxis
from pylops.utils.wavelets import ricker

par1 = {
    "nt": 11,
    "nhx": 21,
    "nhy": 13,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "engine": "numpy",
}  # odd, numpy
par2 = {
    "nt": 11,
    "nhx": 20,
    "nhy": 10,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "engine": "numpy",
}  # even, numpy
par1f = {
    "nt": 11,
    "nhx": 21,
    "nhy": 13,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "engine": "fftw",
}  # odd, fftw
par2f = {
    "nt": 11,
    "nhx": 20,
    "nhy": 10,
    "pymax": 1e-2,
    "pxmax": 2e-2,
    "engine": "fftw",
}  # even, fftw


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ChirpRadon2D(par, dtype):
    """Dot-test, forward, analytical inverse and sparse inverse
    for ChirpRadon2D operator
    """
    parmod = {
        "ot": 0,
        "dt": 0.004,
        "nt": par["nt"],
        "ox": par["nhx"] * 10 / 2,
        "dx": 10,
        "nx": par["nhx"],
        "f0": 40,
    }
    theta = [
        20,
    ]
    t0 = [
        0.1,
    ]
    amp = [
        1.0,
    ]

    # Create axis
    t, _, hx, _ = makeaxis(parmod)

    # Create wavelet
    wav, _, _ = ricker(t[:41], f0=parmod["f0"])

    # Generate model
    x = np.asarray(linear2d(hx, t, 1500.0, t0, theta, amp, wav)[1].astype(dtype))
    Rop = ChirpRadon2D(
        np.asarray(t.astype(dtype)),
        np.asarray(hx.astype(dtype)),
        par["pxmax"],
        dtype=dtype,
    )
    assert dottest(
        Rop,
        par["nhx"] * par["nt"],
        par["nhx"] * par["nt"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward, adjoint inverse dtype check
    y = Rop * x.ravel()
    xadj = Rop.H * y
    xinvana = Rop.inverse(y)
    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert xinvana.dtype == dtype
    assert_array_almost_equal(x.ravel(), xinvana, decimal=3)

    # Sparse inverse
    xinv, _, _ = fista(Rop, y, niter=30, eps=1e0)
    assert_array_almost_equal(x.ravel(), xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par1f), (par2f)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ChirpRadon3D(par, dtype):
    """Dot-test, forward, analytical inverse and sparse inverse
    for ChirpRadon3D operator
    """
    if par["engine"] == "fftw" and backend == "cupy":
        pytest.skip("fftw does not work with CuPy arrays")

    parmod = {
        "ot": 0,
        "dt": 0.004,
        "nt": par["nt"],
        "ox": par["nhx"] * 10 / 2,
        "dx": 10,
        "nx": par["nhx"],
        "oy": par["nhy"] * 10 / 2,
        "dy": 10,
        "ny": par["nhy"],
        "f0": 40,
    }
    theta = [
        20,
    ]
    phi = [
        0,
    ]
    t0 = [
        0.1,
    ]
    amp = [
        1.0,
    ]

    # Create axis
    t, _, hx, hy = makeaxis(parmod)

    # Create wavelet
    wav, _, _ = ricker(t[:41], f0=parmod["f0"])

    # Generate model
    x = np.asarray(
        linear3d(hy, hx, t, 1500.0, t0, theta, phi, amp, wav)[1].astype(dtype)
    )

    Rop = ChirpRadon3D(
        np.asarray(t.astype(dtype)),
        np.asarray(hy.astype(dtype)),
        np.asarray(hx.astype(dtype)),
        (par["pymax"], par["pxmax"]),
        engine=par["engine"],
        dtype=dtype,
        **dict(flags=("FFTW_ESTIMATE",), threads=2),
    )
    assert dottest(
        Rop,
        par["nhy"] * par["nhx"] * par["nt"],
        par["nhy"] * par["nhx"] * par["nt"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward, adjoint inverse dtype check
    y = Rop * x.ravel()
    xadj = Rop.H * y
    xinvana = Rop.inverse(y)
    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert xinvana.dtype == dtype
    assert_array_almost_equal(x.ravel(), xinvana, decimal=3)

    # Sparse inverse
    xinv, _, _ = fista(Rop, y, niter=30, eps=1e0)
    assert_array_almost_equal(x.ravel(), xinv, decimal=3)
