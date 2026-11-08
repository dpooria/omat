package config

// This is an example configuration of omat on Fedora 43
// You can go through it and change it according to your system or override them using -define

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
USE_FFTW_THREADS :: #config(OMAT_USE_FFTW_THREADS, false)

// ------- OPENBLAS -----------
OPENBLAS :: #config(OMAT_OPENBLAS, "system:openblas64")
USE_OPENBLAS_THREADS :: #config(OMAT_USE_OPENBLAS_THREADS, false)
OPENBLAS_ILP64 :: #config(OMAT_OPENBLAS_ILP64, true)

