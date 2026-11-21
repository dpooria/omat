#!/usr/bin/env bash

echo "This file is a configuration reference. Open it and copy one example at a time."
exit 0

# Example Linux build and running configurations with different backends for omat.
# Run these examples from the omat repository root.
# The -define:OMAT_... flags also work with odin build and odin run.
# Replace the example library paths and names with those for your installation.
# Optional: use BLIS_NUM_THREADS or MKL_NUM_THREADS to tune BLAS parallelism.
# ODIN_TEST_THREADS controls the number of concurrent Odin tests.

# Default configuration: LP64 OpenBLAS (without threads) and PocketFFT (with multithreading).
# Build the C++ wrapper once before compiling Odin code.
fft_backend/pocketfft/build.sh
odin test test/

# PocketFFT build:
# Both scripts create a static library and pass extra arguments to the C++ compiler.
# by default pocketfft.a on Linux and pocketfft.lib on Windows; override the path
# with -define:OMAT_POCKETFFT=... for other installations.
# Set -define:OMAT_OPENBLAS=system:libopenblas.lib for OpenBLAS installation
# and make any required DLLs available to the executable on Windows.

# PocketFFT without threading
# On Windows, use .\fft_backend\pocketfft\build.ps1 /DPOCKETFFT_NO_MULTITHREADING.
fft_backend/pocketfft/build.sh -DPOCKETFFT_NO_MULTITHREADING
odin test test/ -define:OMAT_POCKETFFT_NO_MULTITHREADING=true

# OpenBLAS with pthreads and 64-bit BLAS integers (ILP64).
odin test test/ -define:OMAT_OPENBLAS=system:openblasp64 \
    -define:OMAT_BLAS_ILP64=true -define:OMAT_USE_OPENBLAS_THREADS=true

# OpenBLAS with OpenMP threading
odin test test/ -define:OMAT_OPENBLAS=system:openblaso -define:OMAT_USE_OPENBLAS_THREADS=true

# Netlib CBLAS and LAPACKE with 32-bit BLAS integers (LP64).
odin test test/ -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:cblas -define:OMAT_LAPACKE=system:lapacke

# FFTW with pthreads instead of PocketFFT.
# No PocketFFT wrapper build is needed when using FFTW.
# Linux defaults: fftw3, fftw3f, fftw3_threads and fftw3f_threads.
# Windows defaults: libfftw3-3.lib and libfftw3f-3.lib, with threading functions
# in those same libraries. Make the corresponding DLLs available at runtime.
# Override library paths with -define:OMAT_FFTW3=..., -define:OMAT_FFTW3F=...,
# -define:OMAT_FFTW3_THREADS=... and -define:OMAT_FFTW3F_THREADS=... as needed.
odin test test/ -define:OMAT_USE_FFTW=true

# FFTW without the pthread libraries.
odin test test/ -define:OMAT_USE_FFTW=true -define:OMAT_USE_FFTW_THREADS=false

# FFTW with OpenMP threading.
odin test test/ -define:OMAT_USE_FFTW=true \
    -define:OMAT_FFTW3_THREADS=system:fftw3_omp -define:OMAT_FFTW3F_THREADS=system:fftw3f_omp

# AMD AOCL BLIS and libFLAME (LP64). AOCL-BLAS has cblas_?gemm_batch, used by batched einsum.
# Set OMAT_BLAS_HAS_GEMM_BATCH=false in the -define flag to test the per-matrix GEMM fallback.
blis=/path/to/blis/lib
flame=/path/to/libflame/lib
utils=/path/to/aocl-utils/lib

LD_LIBRARY_PATH=$blis:$flame:$utils odin test test/ \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:blis-mt -define:OMAT_LAPACKE=system:flame \
    -define:OMAT_BLAS_HAS_GEMM_BATCH=true "-extra-linker-flags:-L$blis -L$flame -L$utils"

# AMD AOCL-FFTZ via its FFTW wrapper. Some releases call the wrapper libfftw3.so.
fftz=/path/to/aocl-fftz/lib
wrapper=libfftw3.so

LD_LIBRARY_PATH="$fftz" odin test test/ "-collection:fftz=$fftz" \
    -define:OMAT_USE_FFTW=true \
    "-define:OMAT_FFTW3=fftz:$wrapper" "-define:OMAT_FFTW3F=fftz:$wrapper" \
    -define:OMAT_USE_FFTW_THREADS=false -define:ODIN_TEST_THREADS=1

# AMD AOCL BLIS and libFLAME with the AOCL-FFTZ wrapper. Set blis, flame, utils, fftz and wrapper as above.
LD_LIBRARY_PATH="$blis:$flame:$utils:$fftz" odin test test/ \
    -define:OMAT_USE_FFTW=true \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:blis-mt -define:OMAT_LAPACKE=system:flame \
    -define:OMAT_BLAS_HAS_GEMM_BATCH=true "-collection:fftz=$fftz" \
    "-define:OMAT_FFTW3=fftz:$wrapper" "-define:OMAT_FFTW3F=fftz:$wrapper" \
    -define:OMAT_USE_FFTW_THREADS=false -define:ODIN_TEST_THREADS=1 \
    "-extra-linker-flags:-L\"$blis\" -L\"$flame\" -L\"$utils\""

# Intel oneMKL for CBLAS and LAPACKE (Linux x86-64, LP64).
# Include the oneMKL threading runtime (such as libiomp5.so) in your library search path.
# For ILP64, set MKL_INTERFACE_LAYER=ILP64 and add -define:OMAT_BLAS_ILP64=true.
# Set OMAT_BLAS_HAS_GEMM_BATCH=false in the -define flag to test the per-matrix GEMM fallback.
mkl=/opt/intel/oneapi/mkl/latest/lib  # Directory containing libmkl_rt.so.

LD_LIBRARY_PATH="$mkl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MKL_INTERFACE_LAYER=LP64 odin test test/ \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:mkl_rt -define:OMAT_LAPACKE=system:mkl_rt \
    -define:OMAT_BLAS_HAS_GEMM_BATCH=true \
    "-extra-linker-flags:-L$mkl"

# Intel oneMKL for CBLAS, LAPACKE and its built-in FFTW3 wrappers. Set mkl as above.
LD_LIBRARY_PATH="$mkl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MKL_INTERFACE_LAYER=LP64 odin test test/ \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:mkl_rt -define:OMAT_LAPACKE=system:mkl_rt \
    -define:OMAT_BLAS_HAS_GEMM_BATCH=true -define:OMAT_USE_FFTW=true \
    -define:OMAT_FFTW3=system:mkl_rt -define:OMAT_FFTW3F=system:mkl_rt \
    -define:OMAT_USE_FFTW_THREADS=false -define:ODIN_TEST_THREADS=1 "-extra-linker-flags:-L\"$mkl\""
