#!/usr/bin/env bash

echo "This file is a configuration reference. Open it and copy one example at a time."
exit 0

# Copy one example at a time from below and run it from the omat repository root.
# The -define:OMAT_... flags also work with odin build and odin run.
# Replace the example library paths and names with those for your installation.

# Default configuration: LP64 OpenBLAS and FFTW with pthreads.
odin test test/

# OpenBLAS with pthreads and 64-bit BLAS integers (ILP64).
odin test test/ -define:OMAT_OPENBLAS=system:openblasp64 \
    -define:OMAT_BLAS_ILP64=true -define:OMAT_USE_OPENBLAS_THREADS=true

# Netlib CBLAS and LAPACKE with 32-bit BLAS integers (LP64).
odin test test/ -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:cblas -define:OMAT_LAPACKE=system:lapacke

# FFTW without the pthread libraries.
odin test test/ -define:OMAT_USE_FFTW_THREADS=false

# FFTW with OpenMP libraries.
odin test test/ -define:OMAT_FFTW3_THREADS=system:fftw3_omp -define:OMAT_FFTW3F_THREADS=system:fftw3f_omp

# AMD AOCL BLIS and libFLAME (LP64).
blis=/path/to/blis/lib
flame=/path/to/libflame/lib
utils=/path/to/aocl-utils/lib

LD_LIBRARY_PATH=$blis:$flame:$utils BLIS_NUM_THREADS=1 odin test test/ \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:blis-mt -define:OMAT_LAPACKE=system:flame \
    "-extra-linker-flags:-L$blis -L$flame -L$utils"

# AMD AOCL-FFTZ via its FFTW wrapper. Some releases call the wrapper libfftw3.so.
fftz=/path/to/aocl-fftz/lib
wrapper=libfftw3xc_wrapper_fftz.so

LD_LIBRARY_PATH=$fftz odin test test/ -collection:fftz=$fftz \
    -define:OMAT_FFTW3=fftz:$wrapper -define:OMAT_FFTW3F=fftz:$wrapper \
    -define:OMAT_USE_FFTW_THREADS=false -define:ODIN_TEST_THREADS=1 "-extra-linker-flags:-L$fftz -laocl_fftz"

# Intel oneMKL for CBLAS and LAPACKE (Linux x86-64, LP64).
mkl=/opt/intel/oneapi/mkl/latest/lib  # Directory containing libmkl_rt.so.

LD_LIBRARY_PATH=$mkl MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL odin test test/ \
    -define:OMAT_DISABLE_OPENBLAS=true -define:OMAT_BLAS=system:mkl_rt -define:OMAT_LAPACKE=system:mkl_rt \
    "-extra-linker-flags:-L$mkl -lpthread -lm -ldl"

# Intel oneMKL's built-in FFTW3 wrappers (Linux x86-64, LP64).
mkl=/opt/intel/oneapi/mkl/latest/lib

LD_LIBRARY_PATH=$mkl MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL odin run examples/fft_peak.odin -file \
    -define:OMAT_FFTW3=system:mkl_rt -define:OMAT_FFTW3F=system:mkl_rt \
    -define:OMAT_USE_FFTW_THREADS=false "-extra-linker-flags:-L$mkl -lpthread -lm -ldl"
