package openblas

import "../config"
import "core:c"

when !config.DISABLE_OPENBLAS do foreign import openblas {config.OPENBLAS}

blasint :: i64 when config.BLAS_ILP64 else i32
/*Set the threading backend to a custom callback.*/
openblas_dojob_callback :: proc "c" (thread_num: c.int, jobdata: rawptr, dojob_data: c.int)

openblas_threads_callback :: proc "c" (
    sync: c.int,
    dojob: openblas_dojob_callback,
    num_jobs: c.int,
    jobdata_elsize: c.size_t,
    jobdata: rawptr,
    dojob_data: c.int,
)

// openblas threading models
/* OpenBLAS is compiled for sequential use  */
SEQUENTIAL :: 0
/* OpenBLAS is compiled using normal threading model */
THREAD :: 1
/* OpenBLAS is compiled using OpenMP threading model */
OPENMP :: 2

Order :: enum c.int {
    RowMajor = 101,
    ColMajor = 102,
}

Transpose :: enum c.int {
    NoTrans     = 111,
    Trans       = 112,
    ConjTrans   = 113,
    ConjNoTrans = 114,
}

UpLo :: enum c.int {
    Upper = 121,
    Lower = 122,
}

Diag :: enum c.int {
    NonUnit = 131,
    Unit    = 132,
}

Side :: enum c.int {
    Left  = 141,
    Right = 142,
}

// Transpose control for general matrices (gels)
LATranspose :: enum c.char {
    NoTrans   = 'N', // No transpose: A
    Trans     = 'T', // Transpose: A^T
    ConjTrans = 'C', // Conjugate transpose: A^H
}

// (syevd, heevd, sygvd, hegvd, potrf, potri)
LAUpLo :: enum c.char {
    Upper = 'U',
    Lower = 'L',
}

// Job control for eigenvectors (syevd, heevd, sygvd, hegvd, geev)
JobZ :: enum c.char {
    NoVectors = 'N', // Compute eigenvalues only
    Vectors   = 'V', // Compute eigenvalues and eigenvectors
}

// Which eigenvalues to compute (syevr, heevr)
LARange :: enum c.char {
    All     = 'A', // All eigenvalues
    Values  = 'V', // Eigenvalues in the half-open interval (vl, vu]
    Indices = 'I', // The il-th through iu-th eigenvalues (1-based, ascending)
}

// Job control specifically for Divide-and-Conquer SVD (gesdd)
JobSVD :: enum c.char {
    All       = 'A', // Compute all columns of U / rows of VT
    Economy   = 'S', // Compute min(m, n) singular vectors (economy SVD)
    Overwrite = 'O', // Overwrite input matrix A with singular vectors
    None      = 'N', // Do not compute singular vectors
}


when !config.DISABLE_OPENBLAS {
    @(default_calling_convention = "c")
    foreign openblas {
        /*Set the number of threads on runtime.*/
        openblas_set_num_threads :: proc(num_threads: c.int) ---
        goto_set_num_threads :: proc(num_threads: c.int) ---
        openblas_set_num_threads_local :: proc(num_threads: c.int) -> c.int ---

        /*Get the number of threads on runtime.*/
        openblas_get_num_threads :: proc() -> c.int ---

        /*Get the number of physical processors (cores).*/
        openblas_get_num_procs :: proc() -> c.int ---

        /*Get the build configure on runtime.*/
        openblas_get_config :: proc() -> cstring ---

        /*Get the CPU corename on runtime.*/
        openblas_get_corename :: proc() -> cstring ---

        /*Set the threading backend to a custom callback.*/
        openblas_set_threads_callback_function :: proc(callback: openblas_threads_callback) ---

        /* Get the parallelization type which is used by OpenBLAS */
        openblas_get_parallel :: proc() -> c.int ---

    }
}

