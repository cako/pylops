import os

import numpy as np
import pytest

from pylops.signalprocessing import PWSmoother2D, PWSprayer2D
from pylops.utils import dottest

par1 = {
    "nx": 16,
    "nt": 30,
}  # even
par2 = {
    "nx": 17,
    "nt": 31,
}  # odd

np.random.seed(10)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PWSprayer2D(par, dtype):
    """Dot-test and forward/adjoint  for PWSprayer2D"""
    sigma = np.zeros((par["nx"], par["nt"]), dtype=dtype)

    Sop = PWSprayer2D(
        dims=(par["nx"], par["nt"]),
        sigma=sigma,
        dtype=dtype,
    )
    dottest(
        Sop,
        Sop.shape[0],
        par["nx"] * par["nt"],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    x = np.ones(par["nx"] * par["nt"], dtype=dtype)
    y = Sop * x
    xadj = Sop.H * x
    assert y.dtype == dtype
    assert xadj.dtype == dtype


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1, reason="Not CuPy enabled"
)
@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_PWSmoother2D(par, dtype):
    """Dot-test and forward/adjoint for PWSmoother2D"""
    sigma = np.zeros((par["nx"], par["nt"]), dtype=dtype)

    Sop = PWSmoother2D(
        dims=(par["nx"], par["nt"]),
        sigma=sigma,
        dtype=dtype,
    )
    dottest(
        Sop,
        Sop.shape[0],
        par["nx"] * par["nt"],
        rtol=1e-3 if dtype == np.float32 else 1e-6,
    )

    x = np.ones(par["nx"] * par["nt"], dtype=dtype)
    y = Sop * x
    xadj = Sop.H * x
    assert y.dtype == dtype
    assert xadj.dtype == dtype
