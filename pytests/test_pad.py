import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_equal

    backend = "numpy"
import pytest

from pylops.basicoperators import Pad
from pylops.utils import dottest

par1 = {"ny": 11, "nx": 11, "pad": ((0, 2), (4, 5))}  # square
par2 = {"ny": 21, "nx": 11, "pad": ((3, 1), (0, 3))}  # rectangular

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_Pad_1d_negative(par):
    """Check error is raised when pad has negative number"""
    with pytest.raises(ValueError, match="Padding must be positive"):
        _ = Pad(dims=par["ny"], pad=(-10, 0))


@pytest.mark.parametrize("par", [(par1)])
def test_Pad_2d_negative(par):
    """Check error is raised when pad has negative number for 2d"""
    with pytest.raises(ValueError, match="Padding must be positive"):
        _ = Pad(dims=(par["ny"], par["nx"]), pad=((-10, 0), (3, -5)))


@pytest.mark.parametrize("par", [(par1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Pad1d(par, dtype):
    """Dot-test and forward/adjoint for Pad operator on 1d signal"""
    Pop = Pad(dims=par["ny"], pad=par["pad"][0], dtype=dtype)
    assert dottest(
        Pop,
        Pop.shape[0],
        Pop.shape[1],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.arange(par["ny"], dtype=dtype) + 1.0
    y = Pop * x
    xadj = Pop.H * y

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_equal(x, xadj)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Pad2d(par, dtype):
    """Dot-test and adjoint for Pad operator on 2d signal"""
    Pop = Pad(dims=(par["ny"], par["nx"]), pad=par["pad"], dtype=dtype)
    assert dottest(
        Pop,
        Pop.shape[0],
        Pop.shape[1],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = (np.arange(par["ny"] * par["nx"], dtype=dtype) + 1.0).reshape(
        par["ny"], par["nx"]
    )
    y = Pop * x.ravel()
    xadj = Pop.H * y

    assert y.dtype == dtype
    assert xadj.dtype == dtype
    assert_array_equal(x.ravel(), xadj)
