import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal

    backend = "numpy"
import numpy as npp
import pytest

from pylops.optimization.basic import lsqr
from pylops.signalprocessing import Shift
from pylops.utils import dottest
from pylops.utils.wavelets import gaussian

par1 = {
    "nt": 41,
    "nx": 41,
    "ny": 11,
    "imag": 0,
    "dtype": "float64",
}  # square real  (fp64)
par2 = {
    "nt": 41,
    "nx": 21,
    "ny": 11,
    "imag": 0,
    "dtype": "float64",
}  # overdetermined real (fp64)
par1s = {
    "nt": 41,
    "nx": 41,
    "ny": 11,
    "imag": 0,
    "dtype": "float32",
}  # square real (fp32)
par2s = {
    "nt": 41,
    "nx": 21,
    "ny": 11,
    "imag": 0,
    "dtype": "float32",
}  # overdetermined real (fp32)
par1j = {
    "nt": 41,
    "nx": 41,
    "ny": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # square complex
par2j = {
    "nt": 41,
    "nx": 21,
    "ny": 11,
    "imag": 1j,
    "dtype": "complex128",
}  # overdetermined complex


@pytest.mark.parametrize("par", [(par1)])
def test_unknown_engine(par):
    """Check error is raised if unknown engine is passed"""
    with pytest.raises(ValueError, match="`engine` must be numpy"):
        _ = Shift(
            par["nt"],
            1.0,
            engine="foo",
        )


@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
def test_Shift1D(par):
    """Dot-test and forward/adjoint/inversion for Shift operator on 1d data"""
    np.random.seed(0)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    shift = 5.5
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )

    Sop = Shift(
        par["nt"], shift, real=True if par["imag"] == 0 else False, dtype=par["dtype"]
    )
    assert dottest(
        Sop,
        par["nt"],
        par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)


@pytest.mark.skipif(
    int(os.environ.get("TEST_CUPY_PYLOPS", 0)) == 1,
    reason="SciPy engine not compatible with CuPy",
)
@pytest.mark.parametrize("par", [(par1), (par1s), (par1j)])
def test_Shift1D_scipy(par):
    """Dot-test and forward/adjoint/inversion for Shift operator on 1d data
    with scipy engine and workers"""
    np.random.seed(0)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    shift = 5.5
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )

    Sop = Shift(
        par["nt"],
        shift,
        real=True if par["imag"] == 0 else False,
        engine="scipy",
        dtype=par["dtype"],
        **dict(workers=4),
    )
    assert dottest(
        Sop,
        par["nt"],
        par["nt"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_Shift2D(par):
    """Dot-test and forward/adjoint/inversion for Shift operator on 2d data"""
    np.random.seed(0)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    shift = 5.5

    # 1st axis
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )
    x = np.outer(x, np.ones(par["nx"], dtype=dtype))
    Sop = Shift(
        (par["nt"], par["nx"]),
        shift,
        axis=0,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x.ravel()
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)

    # 2nd axis
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )
    x = np.outer(x, np.ones(par["nx"], dtype=dtype)).T
    Sop = Shift(
        (par["nx"], par["nt"]),
        shift,
        axis=1,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x.ravel()
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)


@pytest.mark.parametrize("par", [(par1), (par2), (par1s), (par2s), (par1j), (par2j)])
def test_Shift2Dvariable(par):
    """Dot-test and forward/adjoint/inversion for Shift operator on 2d data with variable shift"""
    np.random.seed(0)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    shift = npp.arange(par["nx"])

    # 1st axis
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )
    x = np.outer(x, np.ones(par["nx"], dtype=dtype))
    Sop = Shift(
        (par["nt"], par["nx"]),
        shift,
        axis=0,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x.ravel()
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        y,
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)

    # 2nd axis
    x = np.asarray(
        gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
        + par["imag"] * gaussian(np.arange(par["nt"] // 2 + 1), 2.0)[0].astype(dtype)
    )
    x = np.outer(x, np.ones(par["nx"], dtype=dtype)).T
    Sop = Shift(
        (par["nx"], par["nt"]),
        shift,
        axis=1,
        real=True if par["imag"] == 0 else False,
        dtype=par["dtype"],
    )
    assert dottest(
        Sop,
        par["nt"] * par["nx"],
        par["nt"] * par["nx"],
        complexflag=0 if par["imag"] == 0 else 3,
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    y = Sop * x.ravel()
    xadj = Sop.H * y
    xlsqr = lsqr(
        Sop,
        Sop * x.ravel(),
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]

    assert y.dtype == par["dtype"]
    assert xadj.dtype == par["dtype"]
    assert_array_almost_equal(x, xlsqr, decimal=2 if dtype == np.float32 else 4)
