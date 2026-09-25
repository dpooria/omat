package test

import "core:strings"
import "core:testing"

import "../fftw"

@(test)
fftw_raw_api :: proc(t: ^testing.T) {
    // the raw FFTW interface, as in C minus the prefix
    n0, n1, n2 :: 4, 4, 4
    data := make([]complex128, n0 * n1 * n2)
    defer delete(data)

    p := fftw.plan_dft_3d(n0, n1, n2, raw_data(data), raw_data(data), .FORWARD, fftw.ESTIMATE)
    defer fftw.destroy_plan(p)
    testing.expect(t, p != nil)

    for &x in data do x = 1
    fftw.execute(p)
    testing.expect_value(t, data[0], complex(f64(n0 * n1 * n2), 0))
    for x in data[1:] do testing.expect(t, abs(x) < 1e-12)

    // same bits as the C FFTW_* macros
    testing.expect_value(t, transmute(u32)fftw.MEASURE, 0)
    testing.expect_value(
        t,
        transmute(u32)(fftw.PATIENT | {.DESTROY_INPUT, .WISDOM_ONLY}),
        1 << 5 | 1 | 1 << 21,
    )

    // globals get the prefix too: f_version is fftwf_version
    testing.expect(t, strings.has_prefix(string(cstring(&fftw.version)), "fftw-"))
    testing.expect(t, strings.has_prefix(string(cstring(&fftw.f_version)), "fftw-"))
}

