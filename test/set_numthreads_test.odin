package test
import "core:testing"

import om ".."
import "../config"
import "../fftw"
import ob "../openblas"

@(test)
set_num_threads_backends :: proc(t: ^testing.T) {
    defer om.set_num_threads(1)

    when config.USE_FFTW_THREADS || config.USE_OPENBLAS_THREADS {
        om.set_num_threads(2)

        when config.USE_FFTW_THREADS {
            testing.expect_value(t, int(fftw.planner_nthreads()), 2)
            testing.expect_value(t, int(fftw.f_planner_nthreads()), 2)
        }

        expected := 2 when config.USE_OPENBLAS_THREADS else 1
        testing.expect_value(t, int(ob.openblas_get_num_threads()), expected)
    }

    om.set_num_threads(1)
    testing.expect_value(t, int(ob.openblas_get_num_threads()), 1)

    when config.USE_FFTW_THREADS {
        testing.expect_value(t, int(fftw.planner_nthreads()), 1)
        testing.expect_value(t, int(fftw.f_planner_nthreads()), 1)
    }
}

@(test)
set_num_threads_rejects_multiple_threads :: proc(t: ^testing.T) {
    when !config.USE_FFTW_THREADS && !config.USE_OPENBLAS_THREADS {
        testing.expect_assert_message(t, "omat is compiled with threads disabled!")
        om.set_num_threads(2)
        testing.expect(t, false, "expected set_num_threads(2) to assert")
    }
}
