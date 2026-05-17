import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np
    from cupy.testing import assert_array_almost_equal
    from cupyx.scipy.signal.windows import triang

    backend = "cupy"
else:
    import numpy as np
    from numpy.testing import assert_array_almost_equal
    from scipy.signal.windows import triang

    backend = "numpy"

import pytest

from pylops.optimization.basic import lsqr
from pylops.signalprocessing import Convolve1D, Convolve2D, ConvolveND
from pylops.utils import dottest

# filters
nfilt = (5, 6, 5)
h1 = triang(nfilt[0], sym=True)
h2 = np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True))
h3 = np.outer(
    np.outer(triang(nfilt[0], sym=True), triang(nfilt[1], sym=True)),
    triang(nfilt[2], sym=True),
).reshape(nfilt)

par1_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 0,
}  # zero phase, first direction
par2_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": 0,
    "axis": 0,
}  # non-zero phase, first direction
par3_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 1,
}  # zero phase, second direction
par4_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": nfilt[0] // 2 - 1,
    "axis": 1,
}  # non-zero phase, second direction
par5_1d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": nfilt[0] // 2,
    "axis": 2,
}  # zero phase, third direction
par6_1d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": nfilt[0] // 2 - 1,
    "axis": 2,
}  # non-zero phase, third direction

par1_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 0,
}  # zero phase, first direction
par2_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 0,
}  # non-zero phase, first direction
par3_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 1,
}  # zero phase, second direction
par4_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 1,
}  # non-zero phase, second direction
par5_2d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "offset": (nfilt[0] // 2, nfilt[1] // 2),
    "axis": 2,
}  # zero phase, third direction
par6_2d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1),
    "axis": 2,
}  # non-zero phase, third direction

par1_3d = {
    "nz": 21,
    "ny": 51,
    "nx": 31,
    "nt": 5,
    "offset": (nfilt[0] // 2, nfilt[1] // 2, nfilt[2] // 2),
    "axis": 0,
}  # zero phase, all directions
par2_3d = {
    "nz": 21,
    "ny": 61,
    "nx": 31,
    "nt": 5,
    "offset": (nfilt[0] // 2 - 1, nfilt[1] // 2 + 1, nfilt[2] // 2 + 1),
    "axis": 0,
}  # non-zero phase, first direction


@pytest.mark.parametrize(
    "par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d), (par5_1d), (par6_1d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve1D(par, dtype):
    """Dot-test and inversion for Convolve1D operator"""
    np.random.seed(10)
    # 1D
    if par["axis"] == 0:
        Cop = Convolve1D(
            par["nx"], h=h1.astype(dtype), offset=par["offset"], dtype=dtype
        )
        assert dottest(
            Cop,
            par["nx"],
            par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros(par["nx"], dtype=dtype)
        x[par["nx"] // 2] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x,
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 2D
    if par["axis"] < 2:
        Cop = Convolve1D(
            (par["ny"], par["nx"]),
            h=h1.astype(dtype),
            offset=par["offset"],
            axis=par["axis"],
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        x[
            int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
            int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        ] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x.ravel(),
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 1D on 3D
    Cop = Convolve1D(
        (par["nz"], par["ny"], par["nx"]),
        h=h1.astype(dtype),
        offset=par["offset"],
        axis=par["axis"],
        dtype=dtype,
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0

    # Forward and adjoint dtype check
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # Inverse
    xlsqr = lsqr(
        Cop,
        Cop * x.ravel(),
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    assert_array_almost_equal(x, xlsqr, decimal=1)


@pytest.mark.parametrize(
    "par", [(par1_1d), (par2_1d), (par3_1d), (par4_1d), (par5_1d), (par6_1d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve1D_long(par, dtype):
    """Dot-test and inversion for Convolve1D operator with long filter"""
    np.random.seed(10)
    # 1D
    if par["axis"] == 0:
        x = np.zeros(par["nx"], dtype=dtype)
        x[par["nx"] // 2] = 1.0
        Xop = Convolve1D(nfilt[0], h=x, offset=nfilt[0] // 2, dtype=dtype)
        assert dottest(
            Xop,
            par["nx"],
            nfilt[0],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        # Forward and adjoint dtype check
        y = Xop * h1.astype(dtype)
        h1adj = Xop.H * y
        assert y.dtype == dtype
        assert h1adj.dtype == dtype

        # Inverse
        h1lsqr = lsqr(
            Xop, Xop * h1, damp=1e-20, niter=200, atol=1e-8, btol=1e-8, show=0
        )[0]
        assert_array_almost_equal(h1, h1lsqr, decimal=1)


@pytest.mark.parametrize(
    "par", [(par1_2d), (par2_2d), (par3_2d), (par4_2d), (par5_2d), (par6_2d)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve2D(par, dtype):
    """Dot-test and inversion for Convolve2D operator"""
    # 2D on 2D
    if par["axis"] == 2:
        Cop = Convolve2D(
            (par["ny"], par["nx"]),
            h=h2.astype(dtype),
            offset=par["offset"],
            dtype=dtype,
        )
        assert dottest(
            Cop,
            par["ny"] * par["nx"],
            par["ny"] * par["nx"],
            rtol=1e-4 if dtype == np.float32 else 1e-6,
            backend=backend,
        )

        x = np.zeros((par["ny"], par["nx"]), dtype=dtype)
        x[
            int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
            int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        ] = 1.0

        # Forward and adjoint dtype check
        y = Cop * x
        xadj = Cop.H * y
        assert y.dtype == dtype
        assert xadj.dtype == dtype

        # Inverse
        xlsqr = lsqr(
            Cop,
            Cop * x.ravel(),
            x0=np.zeros_like(x),
            damp=1e-20,
            niter=200,
            atol=1e-8,
            btol=1e-8,
            show=0,
        )[0]
        assert_array_almost_equal(x, xlsqr, decimal=1)

    # 2D on 3D
    axes = list(range(3))
    axes.remove(par["axis"])
    Cop = Convolve2D(
        (par["nz"], par["ny"], par["nx"]),
        h=h2.astype(dtype),
        offset=par["offset"],
        axes=axes,
        dtype="float64",
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0

    # Forward and adjoint dtype check
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # Inverse
    xlsqr = lsqr(
        Cop,
        Cop * x.ravel(),
        x0=np.zeros_like(x),
        damp=1e-20,
        niter=200,
        atol=1e-8,
        btol=1e-8,
        show=0,
    )[0]
    # due to ringing in solution we cannot use assert_array_almost_equal
    assert np.linalg.norm(xlsqr - x) / np.linalg.norm(xlsqr) < 2e-1


@pytest.mark.parametrize("par", [(par1_3d), (par2_3d)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_Convolve3D(par, dtype):
    """Dot-test and inversion for ConvolveND operator"""
    # 3D on 3D
    Cop = ConvolveND(
        (par["nz"], par["ny"], par["nx"]),
        h=h3.astype(dtype),
        offset=par["offset"],
        dtype=dtype,
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"],
        par["nz"] * par["ny"] * par["nx"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    x = np.zeros((par["nz"], par["ny"], par["nx"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
    ] = 1.0

    # Forward and adjoint dtype check
    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype

    # Inverse
    xlsqr = lsqr(
        Cop, y, x0=np.zeros_like(x), damp=1e-20, niter=400, atol=1e-8, btol=1e-8, show=0
    )[0]
    # due to ringing in solution we cannot use assert_array_almost_equal
    assert np.linalg.norm(xlsqr - x) / np.linalg.norm(xlsqr) < 2e-1

    # 3D on 4D (only modelling)
    Cop = ConvolveND(
        (par["nz"], par["ny"], par["nx"], par["nt"]),
        h=h3.astype(dtype),
        offset=par["offset"],
        axes=[0, 1, 2],
        dtype=dtype,
    )
    assert dottest(
        Cop,
        par["nz"] * par["ny"] * par["nx"] * par["nt"],
        par["nz"] * par["ny"] * par["nx"] * par["nt"],
        rtol=1e-4 if dtype == np.float32 else 1e-6,
        backend=backend,
    )

    # Forward and adjoint dtype check
    x = np.zeros((par["nz"], par["ny"], par["nx"], par["nt"]), dtype=dtype)
    x[
        int(par["nz"] / 2 - 3) : int(par["nz"] / 2 + 3),
        int(par["ny"] / 2 - 3) : int(par["ny"] / 2 + 3),
        int(par["nx"] / 2 - 3) : int(par["nx"] / 2 + 3),
        int(par["nt"] / 2 - 3) : int(par["nt"] / 2 + 3),
    ] = 1.0

    y = Cop * x
    xadj = Cop.H * y
    assert y.dtype == dtype
    assert xadj.dtype == dtype
