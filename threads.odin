package omat

import "core:c"
import "core:log"
import "core:sync"

import "config"
import "fftw"
import ob "openblas"

// Threads for OpenBLAS and for FFTW plans made afterwards
set_num_threads :: proc(n: int) {

    when !config.USE_OPENBLAS_THREADS && !config.USE_FFTW_THREADS {
        assert(n == 1, "omat is compiled with threads disabled!")
    }
    when config.USE_FFTW_THREADS {
        sync.mutex_lock(&fft_planner_lock)
        defer sync.mutex_unlock(&fft_planner_lock)
        @(static) fftw_threads_ready: bool
        if !fftw_threads_ready {
            fftw.init_threads()
            fftw.f_init_threads()
            fftw_threads_ready = true
        }
        fftw.plan_with_nthreads(c.int(n))
        fftw.f_plan_with_nthreads(c.int(n))
    }

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

