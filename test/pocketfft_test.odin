package test

import "core:math"
import "core:testing"

import "../config"
import "../fft_backend/pocketfft"

when !config.USE_FFTW {
    @(test)
    pocketfft_raw_complex :: proc(t: ^testing.T) {
        pocketfft_raw_complex_check(t, complex64, 2, 1)
        pocketfft_raw_complex_check(t, complex128, 2, 1)
    }

    when !config.POCKETFFT_NO_MULTITHREADING {
        @(test)
        pocketfft_raw_complex_threaded :: proc(t: ^testing.T) {
            // Enough independent rows for PocketFFT to schedule two workers.
            pocketfft_raw_complex_check(t, complex64, 64, 2)
            pocketfft_raw_complex_check(t, complex128, 64, 2)
        }
    }

    pocketfft_raw_complex_check :: proc(t: ^testing.T, $C: typeid, $batches: int, threads: int) {
        n :: 7
        in_row, out_row :: 2 * n + 1, 3 * n + 1
        shape := [2]int{batches, n}
        si := [2]int{in_row * size_of(C), 2 * size_of(C)}
        so := [2]int{out_row * size_of(C), 3 * size_of(C)}
        input: [batches * in_row]C
        output: [batches * out_row]C
        for &v in output do v = 99
        for b in 0 ..< batches {
            offset := f64(b) / 16
            for j in 0 ..< n {
                input[b * in_row + 2 * j] = C(complex(f64(j) + offset, (f64(j) - offset) / 4))
            }
        }
        original := input
        p := pocketfft.plan_create(2, &shape[0], &si[0], &so[0], 1, 0, C == complex64, false, threads)
        defer pocketfft.destroy(p)
        testing.expect(t, p != nil)
        tol := 1e-4 when C == complex64 else 1e-11

        for _ in 0 ..< 2 {
            pocketfft.execute(p, &input[0], &output[0])
            for b in 0 ..< batches {
                for k in 0 ..< n {
                    want: complex128
                    for j in 0 ..< n {
                        angle := -2 * math.PI * f64(j * k) / n
                        want +=
                            complex128(input[b * in_row + 2 * j]) *
                            complex(math.cos(angle), math.sin(angle))
                    }
                    testing.expect(t, abs(complex128(output[b * out_row + 3 * k]) - want) < tol)
                }
            }
        }
        testing.expect_value(t, input, original)
        testing.expect_value(t, output[1], C(99))
        testing.expect_value(t, output[len(output) - 1], C(99))

        inv := pocketfft.plan_create(2, &shape[0], &so[0], &so[0], 1, 0, C == complex64, true, threads)
        defer pocketfft.destroy(inv)
        testing.expect(t, inv != nil)
        pocketfft.execute(inv, &output[0], &output[0])
        for b in 0 ..< batches {
            for j in 0 ..< n {
                testing.expect(
                    t,
                    f64(abs(output[b * out_row + 3 * j] - C(n) * input[b * in_row + 2 * j])) < tol,
                )
            }
        }
    }

    @(test)
    pocketfft_raw_real :: proc(t: ^testing.T) {
        pocketfft_raw_real_check(t, f32, complex64)
        pocketfft_raw_real_check(t, f64, complex128)
    }

    pocketfft_raw_real_check :: proc(t: ^testing.T, $R, $C: typeid) {
        n :: 7
        shape := [1]int{n}
        sr := [1]int{2 * size_of(R)}
        sc := [1]int{3 * size_of(C)}
        input, output: [2 * n]R
        spectrum: [3 * (n / 2 + 1)]C
        for j in 0 ..< n do input[2 * j] = R(j + 1)
        for &v in output do v = 99
        for &v in spectrum do v = 99
        original := input
        p := pocketfft.plan_create(1, &shape[0], &sr[0], &sc[0], 0, 1, R == f32, false, 1)
        inv := pocketfft.plan_create(1, &shape[0], &sc[0], &sr[0], 0, 2, R == f32, true, 1)
        defer pocketfft.destroy(p)
        defer pocketfft.destroy(inv)
        testing.expect(t, p != nil && inv != nil)
        pocketfft.execute(p, &input[0], &spectrum[0])
        tol := 1e-4 when R == f32 else 1e-11
        for k in 0 ..< n / 2 + 1 {
            want: complex128
            for j in 0 ..< n {
                angle := -2 * math.PI * f64(j * k) / n
                want += complex(f64(input[2 * j]), 0) * complex(math.cos(angle), math.sin(angle))
            }
            testing.expect(t, abs(complex128(spectrum[3 * k]) - want) < tol)
        }
        pocketfft.execute(inv, &spectrum[0], &output[0])
        for j in 0 ..< n {
            // Raw transforms are unnormalized.
            testing.expect(t, abs(f64(output[2 * j]) - n * f64(input[2 * j])) < tol)
            testing.expect_value(t, output[2 * j + 1], R(99))
        }
        testing.expect_value(t, input, original)
        testing.expect_value(t, spectrum[1], C(99))
        testing.expect_value(t, spectrum[len(spectrum) - 1], C(99))
    }
}
