package test

import "core:math"
import "core:math/rand"
import "core:testing"
import "core:thread"

import om ".."

random_complex :: proc($C: typeid, shape: [$N]int) -> om.Tensor(C, N) {
    t := om.zeros(C, shape)
    for &x in t.data {
        when C == complex128 {
            x = complex(rand.float64(), rand.float64())
        }
        else {
            x = complex(rand.float32(), rand.float32())
        }
    }
    return t
}

close :: proc(a, b: []$E, tol: f64) -> bool {
    if len(a) != len(b) {
        return false
    }
    for x, i in a {
        if f64(abs(x - b[i])) > tol {
            return false
        }
    }
    return true
}

@(test)
fft_matches_naive_dft :: proc(t: ^testing.T) {
    n :: 16
    x := random_complex(complex128, [1]int{n})
    defer om.tensor_destroy(&x)
    y := om.fft(x)
    defer om.tensor_destroy(&y)

    for k in 0 ..< n {
        want: complex128
        for j in 0 ..< n {
            angle := -2 * math.PI * f64(j * k) / n
            want += x.data[j] * complex(math.cos(angle), math.sin(angle))
        }
        testing.expectf(
            t,
            abs(y.data[k] - want) < 1e-10,
            "k=%d: got %v want %v",
            k,
            y.data[k],
            want,
        )
    }
}

@(test)
fft_roundtrip_3d_f32 :: proc(t: ^testing.T) {
    x := random_complex(complex64, [3]int{8, 6, 10})
    defer om.tensor_destroy(&x)
    y := om.fft(x)
    defer om.tensor_destroy(&y)
    z := om.invfft(y)
    defer om.tensor_destroy(&z)
    testing.expect(t, close(z.data, x.data, 1e-5))

    // Ortho keeps the norm (Parseval)
    o := om.fft(x, norm = .Ortho)
    defer om.tensor_destroy(&o)
    ex, eo: f64
    for v in x.data {
        ex += f64(real(v * conj(v)))
    }
    for v in o.data {
        eo += f64(real(v * conj(v)))
    }
    testing.expect(t, abs(ex - eo) < 1e-3 * ex)
}

@(test)
fft_batched :: proc(t: ^testing.T) {
    // 3 independent 1D transforms along the rows
    x := random_complex(complex128, [2]int{3, 8})
    defer om.tensor_destroy(&x)
    y := om.fft(x, {.Batched})
    defer om.tensor_destroy(&y)

    for row in 0 ..< 3 {
        r := om.view_row(x, row)
        want := om.fft(r)
        defer om.tensor_destroy(&want)
        testing.expect(t, close(y.data[row * 8:][:8], want.data, 1e-12))
    }
}

@(test)
fft_real_roundtrip :: proc(t: ^testing.T) {
    x := om.zeros(f64, [3]int{4, 6, 10})
    defer om.tensor_destroy(&x)
    sum: f64
    for &v in x.data {
        v = rand.float64()
        sum += v
    }

    y := om.fft(x) // real input: half spectrum
    defer om.tensor_destroy(&y)
    testing.expect_value(t, y.shape, [3]int{4, 6, 6})
    testing.expect(t, abs(real(y.data[0]) - sum) < 1e-10) // DC term

    z := om.invrfft(y)
    defer om.tensor_destroy(&z)
    testing.expect_value(t, z.shape, x.shape)
    testing.expect(t, close(z.data, x.data, 1e-12))
    testing.expect(t, abs(real(y.data[0]) - sum) < 1e-10) // invrfft left its input alone

    // odd length needs n
    w := om.zeros(f32, [1]int{7})
    defer om.tensor_destroy(&w)
    for &v, i in w.data {
        v = f32(i)
    }
    wf := om.fft(w)
    defer om.tensor_destroy(&wf)
    wb := om.invrfft(wf, n = 7)
    defer om.tensor_destroy(&wb)
    testing.expect(t, close(wb.data, w.data, 1e-5))
}

@(test)
fft_plan_reused :: proc(t: ^testing.T) {
    // one measured plan: a 4x5 transform for each of 3 bands, reused on new tensors
    shape :: [3]int{3, 4, 5}
    fwd := om.fft_plan(complex128, complex128, shape, {.Batched}, rigor = .Measure)
    defer om.fft_destroy(&fwd)

    for _ in 0 ..< 2 {
        x := random_complex(complex128, shape)
        defer om.tensor_destroy(&x)
        y := om.zeros(complex128, shape)
        defer om.tensor_destroy(&y)
        om.fft_execute(fwd, x, y)

        for band in 0 ..< 3 {
            b := om.view_row(x, band)
            want := om.fft(b)
            defer om.tensor_destroy(&want)
            testing.expect(t, close(y.data[band * 20:][:20], want.data, 1e-12))
        }
    }

    // in place
    inv := om.fft_plan(complex128, complex128, shape, {.Inverse, .Batched, .In_Place})
    defer om.fft_destroy(&inv)
    x := random_complex(complex128, shape)
    defer om.tensor_destroy(&x)
    orig := om.tensor_copy(x)
    defer om.tensor_destroy(&orig)
    om.fft_execute(fwd, orig, x)
    om.fft_execute(inv, x, x)
    testing.expect(t, close(x.data, orig.data, 1e-12))

    // real <-> complex, kind and direction picked from the element types
    r2c := om.fft_plan(f64, complex128, [2]int{6, 8})
    c2r := om.fft_plan(complex128, f64, [2]int{6, 8})
    defer om.fft_destroy(&r2c)
    defer om.fft_destroy(&c2r)
    testing.expect_value(t, r2c.out_shape, [2]int{6, 5})
    testing.expect_value(t, c2r.in_shape, [2]int{6, 5})
    testing.expect_value(t, c2r.scale, 1.0 / 48)
}

@(test)
fft_unaligned_view :: proc(t: ^testing.T) {
    // a view starting 8 bytes into a buffer is not 16-byte aligned
    buf := random_complex(complex64, [1]int{17})
    defer om.tensor_destroy(&buf)
    view := om.view_rows(buf, 1, 17)
    out := om.zeros(complex64, [1]int{16})
    defer om.tensor_destroy(&out)

    p := om.fft_plan(complex64, complex64, [1]int{16}, {.Unaligned})
    defer om.fft_destroy(&p)
    om.fft_execute(p, view, out)

    want := om.fft(view) // one-shots handle it themselves
    defer om.tensor_destroy(&want)
    testing.expect(t, close(out.data, want.data, 1e-4))
}

@(test)
fft_frequencies :: proc(t: ^testing.T) {
    f4 := om.fftfreq(4)
    f5 := om.fftfreq(5, 0.5)
    defer om.tensor_destroy(&f4)
    defer om.tensor_destroy(&f5)
    testing.expect(t, close(f4.data, []f64{0, 0.25, -0.5, -0.25}, 1e-15))
    testing.expect(t, close(f5.data, []f64{0, 0.4, 0.8, -0.8, -0.4}, 1e-15))
}

// plans made and destroyed from several threads at once: the planner lock keeps FFTW safe
@(test)
fft_plans_from_threads :: proc(t: ^testing.T) {
    Work :: struct {
        ok: bool,
    }
    worker :: proc(w: ^Work) {
        w.ok = true
        for i in 0 ..< 20 {
            n := 8 + i % 5
            x := om.zeros(complex128, [2]int{n, n})
            defer om.tensor_destroy(&x)
            x.data[1] = 1
            y := om.fft(x)
            defer om.tensor_destroy(&y)
            z := om.invfft(y)
            defer om.tensor_destroy(&z)
            if abs(z.data[1] - 1) > 1e-12 || abs(om.sum(z) - 1) > 1e-12 {
                w.ok = false
            }
        }
    }
    work: [4]Work
    threads: [4]^thread.Thread
    for &th, i in threads {
        th = thread.create_and_start_with_poly_data(&work[i], worker, context)
    }
    for th in threads {
        thread.join(th)
        thread.destroy(th)
    }
    for w in work {
        testing.expect(t, w.ok)
    }
}

