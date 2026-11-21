package omat

import "core:c"
import "core:log"

import "config"
import "fft_backend"
import ob "openblas"

// Threads for OpenBLAS and for FFT plans made afterwards
set_num_threads :: proc(n: int) {

    when !config.USE_OPENBLAS_THREADS && !config.USE_FFTW_THREADS && (config.USE_FFTW || config.POCKETFFT_NO_MULTITHREADING) {
        assert(n == 1, "omat is compiled with threads disabled!")
    }
    fft_backend.set_num_threads(n)

    when !config.DISABLE_OPENBLAS {
        when config.USE_OPENBLAS_THREADS {
            if n > 1 && ob.openblas_get_parallel() == ob.SEQUENTIAL {
                log.warn(
                    "omat: linked OpenBLAS is sequential. " +
                    "Set -define:OMAT_OPENBLAS to a threaded build to use more than one thread.",
                )
            }
            ob.openblas_set_num_threads(c.int(n))
        } else {
            ob.openblas_set_num_threads(1)
        }
    }
}

