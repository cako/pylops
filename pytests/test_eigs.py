import os

if int(os.environ.get("TEST_CUPY_PYLOPS", 0)):
    import cupy as np

    backend = "cupy"
else:
    import numpy as np

    backend = "numpy"
import numpy as npp
import pytest

from pylops.basicoperators import MatrixMult
from pylops.optimization.eigs import power_iteration
from pylops.utils.backend import to_numpy

par1 = {"n": 21, "imag": 0, "dtype": "float32"}  # real (fp32)
par2 = {"n": 21, "imag": 0, "dtype": "float64"}  # real (fp64)
par3 = {"n": 21, "imag": 1j, "dtype": "complex64"}  # complex (cp64)
par4 = {"n": 21, "imag": 1j, "dtype": "complex128"}  # complex (cp128)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_power_iteration(par):
    """Max eigenvalue computation with power iteration method vs. numpy.linalg.eig"""
    np.random.seed(10)
    dtype = np.empty(0, dtype=par["dtype"]).real.dtype

    A = np.random.randn(par["n"], par["n"]).astype(dtype) + par[
        "imag"
    ] * np.random.randn(par["n"], par["n"]).astype(dtype)
    A1 = np.conj(A.T) @ A

    # non-symmetric
    Aop = MatrixMult(A, dtype=par["dtype"])
    eig = power_iteration(Aop, niter=200, tol=0, backend=backend)[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A))[0]))
    assert eig.dtype == par["dtype"]
    assert np.abs(np.abs(eig) - eig_np) < 1e-3

    # symmetric
    A1op = MatrixMult(A1, dtype=par["dtype"])
    eig = power_iteration(A1op, niter=200, tol=0, backend=backend)[0]
    eig_np = npp.max(npp.abs(npp.linalg.eig(to_numpy(A1))[0]))
    assert eig.dtype == par["dtype"]
    assert np.abs(np.abs(eig) - eig_np) < 1e-3
