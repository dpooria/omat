package test

import "core:sync"
import "core:testing"

import "../fft_backend"
import "../fft_backend/fftw"
import "../config"

when config.USE_FFTW {
    @(test)
    fftw_raw_api :: proc(t: ^testing.T) {
        // without FFTW_threads and the init call to make_planner_thread_safe
        // Also it doesn't make the OMP variant safe and the tests fail in multi-threaded `odin test`;
        // syncing with the backend mutex to avoid that.
        sync.mutex_lock(&fft_backend.planner_lock)
        defer sync.mutex_unlock(&fft_backend.planner_lock)
        // the raw FFTW interface, as in C minus the prefix
        n0, n1, n2 :: 4, 4, 4
        data := make([]complex128, n0 * n1 * n2)
        defer delete(data)

        p := fftw.plan_dft_3d(
            n0,
            n1,
            n2,
            raw_data(data),
            raw_data(data),
            .FORWARD,
            fftw.Flags{.ESTIMATE},
        )
        defer fftw.destroy_plan(p)
        testing.expect(t, p != nil)

        for &x in data do x = 1
        fftw.execute(p)
        testing.expect_value(t, data[0], complex(f64(n0 * n1 * n2), 0))
        for x in data[1:] do testing.expect(t, abs(x) < 1e-12)

        // same bits as the C FFTW_* macros
        testing.expect_value(t, transmute(u32)fftw.Flags{}, 0)
        testing.expect_value(
            t,
            transmute(u32)(fftw.Flags{.PATIENT} | {.DESTROY_INPUT, .WISDOM_ONLY}),
            1 << 5 | 1 | 1 << 21,
        )

        // fftw.version is non-empty
        testing.expect(t, fftw.version != 0)
        testing.expect(t, fftw.f_version != 0)
    }


}
