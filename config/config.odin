package config

// The default library names are tested on Fedora 43
// You can change them accordingly or override them using -define
// See omat/examples/config.sh

// PocketFFT is the default; opt in to FFTW with -define:OMAT_USE_FFTW=true.
USE_FFTW :: #config(OMAT_USE_FFTW, false)
when USE_FFTW {
    FFTW3 :: #config(OMAT_FFTW3, "system:libfftw3-3.lib" when ODIN_OS == .Windows else "system:fftw3")
    FFTW3F :: #config(OMAT_FFTW3F, "system:libfftw3f-3.lib" when ODIN_OS == .Windows else "system:fftw3f")
    // Windows FFTW DLLs commonly include their threading functions in the main libraries.
    FFTW3_THREADS :: #config(OMAT_FFTW3_THREADS, FFTW3 when ODIN_OS == .Windows else "system:fftw3_threads")
    FFTW3F_THREADS :: #config(OMAT_FFTW3F_THREADS, FFTW3F when ODIN_OS == .Windows else "system:fftw3f_threads")
} else {
    POCKETFFT :: #config(OMAT_POCKETFFT, "pocketfft.lib" when ODIN_OS == .Windows else "pocketfft.a")
}
USE_FFTW_THREADS :: USE_FFTW && #config(OMAT_USE_FFTW_THREADS, true)
POCKETFFT_NO_MULTITHREADING :: #config(OMAT_POCKETFFT_NO_MULTITHREADING, false)

// ------- OPENBLAS -----------
DISABLE_OPENBLAS :: #config(OMAT_DISABLE_OPENBLAS, false)
when !DISABLE_OPENBLAS {
    OPENBLAS :: #config(OMAT_OPENBLAS, "system:openblas")
    USE_OPENBLAS_THREADS :: #config(OMAT_USE_OPENBLAS_THREADS, false)
    LAPACKE :: #config(OMAT_LAPACKE, OPENBLAS)
    BLAS :: #config(OMAT_BLAS, OPENBLAS)
} else {
    LAPACKE :: #config(OMAT_LAPACKE, "system:lapacke")
    BLAS :: #config(OMAT_BLAS, "system:cblas")
}
BLAS_ILP64 :: #config(OMAT_BLAS_ILP64, false)
// disables cblas extensions and linking against openblas specific bindings.
// A working omat compile still needs LAPCKE and BLAS
// The BLAS has a working cblas_?gemm_batch for batched einsum
// disabled by default due to a bug in OpenBLAS versions 0.3.28 to 0.3.34 built with DYNAMIC_ARCH
BLAS_HAS_GEMM_BATCH :: #config(OMAT_BLAS_HAS_GEMM_BATCH, false)

