package config

// The default library names are tested on Fedora 43
// You can change them accordingly or override them using -define
// See omat/examples/config.sh

// ------- FFT -----------
fft_backend :: enum {
    fftw,
    pocketfft, // TODO
}

OMAT_FFT_BACKEND :: fft_backend.fftw

FFTW3 :: #config(OMAT_FFTW3, "system:fftw3")
FFTW3F :: #config(OMAT_FFTW3F, "system:fftw3f")
FFTW3_THREADS :: #config(OMAT_FFTW3_THREADS, "system:fftw3_threads")
FFTW3F_THREADS :: #config(OMAT_FFTW3F_THREADS, "system:fftw3f_threads")
USE_FFTW_THREADS :: #config(OMAT_USE_FFTW_THREADS, true)

// ------- OPENBLAS -----------
OPENBLAS :: #config(OMAT_OPENBLAS, "system:openblas")
LAPACKE :: #config(OMAT_LAPACKE, OPENBLAS)
BLAS :: #config(OMAT_BLAS, OPENBLAS)
USE_OPENBLAS_THREADS :: #config(OMAT_USE_OPENBLAS_THREADS, false)
BLAS_ILP64 :: #config(OMAT_BLAS_ILP64, false)
// disables cblas extensions and linking against openblas specific bindings.
// A working omat compile still needs LAPCKE and BLAS
DISABLE_OPENBLAS :: #config(OMAT_DISABLE_OPENBLAS, false)

