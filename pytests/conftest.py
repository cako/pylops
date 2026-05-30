import os

# Set the default PTX compute capability for Numba to 8.6,
# currently required by the self-hosted NVIDIA GPU drivers.
os.environ["NUMBA_CUDA_DEFAULT_PTX_CC"] = "8.6"
