package test

import "core:math"
import "core:math/rand"
import "core:testing"
import "core:thread"

import om ".."

@(test)
fft_matches_naive_dft :: proc(t: ^testing.T) {
    n :: 16
    x := om.random_complex(complex128, [1]int{n})
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
    x := om.random_complex(complex64, [3]int{8, 6, 10})
    defer om.tensor_destroy(&x)
    y := om.fft(x)
    defer om.tensor_destroy(&y)
    z := om.invfft(y)
    defer om.tensor_destroy(&z)
    testing.expect(t, om.allclose(z, x, rtol = 0, atol = 1e-5))

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
    x := om.random_complex(complex128, [2]int{3, 8})
    defer om.tensor_destroy(&x)
    y := om.fft(x, {.Batched})
    defer om.tensor_destroy(&y)

    for row in 0 ..< 3 {
        r := om.view_row(x, row)
        want := om.fft(r)
        defer om.tensor_destroy(&want)
        testing.expect(t, om.allclose(om.view_row(y, row), want, rtol = 0, atol = 1e-12))
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

    z := om.fft(f64, y, {.Inverse})
    defer om.tensor_destroy(&z)
    testing.expect_value(t, z.shape, x.shape)
    testing.expect(t, om.allclose(z, x, rtol = 0, atol = 1e-12))
    testing.expect(t, abs(real(y.data[0]) - sum) < 1e-10) // inverse left its input alone

    // odd length needs n
    w := om.zeros(f32, [1]int{7})
    defer om.tensor_destroy(&w)
    for &v, i in w.data {
        v = f32(i)
    }
    wf := om.fft(w)
    defer om.tensor_destroy(&wf)
    wb := om.fft(f32, wf, {.Inverse}, n = 7)
    defer om.tensor_destroy(&wb)
    testing.expect(t, om.allclose(wb, w, rtol = 0, atol = 1e-5))
}

@(test)
fft_plan_reused :: proc(t: ^testing.T) {
    // one measured plan: a 4x5 transform for each of 3 bands, reused on new tensors
    shape :: [3]int{3, 4, 5}
    fwd := om.fft_plan(complex128, complex128, shape, {.Batched}, rigor = .Measure)
    defer om.fft_destroy(&fwd)

    for _ in 0 ..< 2 {
        x := om.random_complex(complex128, shape)
        defer om.tensor_destroy(&x)
        y := om.zeros(complex128, shape)
        defer om.tensor_destroy(&y)
        om.fft_execute(fwd, x, y)

        for band in 0 ..< 3 {
            b := om.view_row(x, band)
            want := om.fft(b)
            defer om.tensor_destroy(&want)
            testing.expect(t, om.allclose(om.view_row(y, band), want, rtol = 0, atol = 1e-12))
        }
    }

    // in place
    inv := om.fft_plan(complex128, complex128, shape, {.Inverse, .Batched, .In_Place})
    defer om.fft_destroy(&inv)
    x := om.random_complex(complex128, shape)
    defer om.tensor_destroy(&x)
    orig := om.tensor_copy(x)
    defer om.tensor_destroy(&orig)
    om.fft_execute(fwd, orig, x)
    om.fft_execute(inv, x, x)
    testing.expect(t, om.allclose(x, orig, rtol = 0, atol = 1e-12))

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
    buf := om.random_complex(complex64, [1]int{17})
    defer om.tensor_destroy(&buf)
    view := om.view_rows(buf, 1, 17)
    out := om.zeros(complex64, [1]int{16})
    defer om.tensor_destroy(&out)

    p := om.fft_plan(complex64, complex64, [1]int{16}, {.Unaligned})
    defer om.fft_destroy(&p)
    om.fft_execute(p, view, out)

    want := om.fft(view) // one-shots handle it themselves
    defer om.tensor_destroy(&want)
    testing.expect(t, om.allclose(out, want, rtol = 0, atol = 1e-4))
}

@(test)
fft_frequencies :: proc(t: ^testing.T) {
    f4 := om.fftfreq(4)
    f5 := om.fftfreq(5, 0.5)
    defer om.tensor_destroy(&f4)
    defer om.tensor_destroy(&f5)
    expect_close(t, f4.data, []f64{0, 0.25, -0.5, -0.25}, 1e-15)
    expect_close(t, f5.data, []f64{0, 0.4, 0.8, -0.8, -0.4}, 1e-15)
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


@(test)
fftfreq_int :: proc(t: ^testing.T) {
    even := om.fftfreq_int(6)
    defer om.tensor_destroy(&even)
    expect_close(t, even.data, []int{0, 1, 2, -3, -2, -1}, 0)
    odd := om.fftfreq_int(5)
    defer om.tensor_destroy(&odd)
    expect_close(t, odd.data, []int{0, 1, 2, -2, -1}, 0)
}

@(test)
fft_into_strided :: proc(t: ^testing.T) {
    x := om.random_complex(complex128, [2]int{4, 6})
    defer om.tensor_destroy(&x)
    layouts := make_layouts(x)
    defer destroy_layouts(&layouts)
    parent := om.zeros(complex128, [2]int{8, 6})
    defer om.tensor_destroy(&parent)
    om.fill(&parent, 99)
    output := om.transpose_view(om.view(parent, {1, 1}, {7, 5}))
    want := om.fft(x, norm = .Ortho)
    defer om.tensor_destroy(&want)

    p := om.fft_plan(layouts.boxed, output, norm = .Ortho, rigor = .Measure)
    defer om.fft_destroy(&p)
    om.assign(layouts.boxed, x) // measured planning may overwrite input
    for _ in 0 ..< 2 {
        om.fft_execute(p, layouts.boxed, output)
        testing.expect(t, om.allclose(output, want, rtol = 0, atol = 1e-12))
    }
    om.fft_into(layouts.transposed, output, norm = .Ortho)
    testing.expect(t, om.allclose(output, want, rtol = 0, atol = 1e-12))
    testing.expect_value(t, om.atv(parent, 0, 0), complex128(99))
    testing.expect_value(t, om.atv(parent, 7, 5), complex128(99))
    testing.expect(t, om.allclose(layouts.boxed, x, rtol = 0, atol = 0))
}

@(test)
fft_plan_from_kind :: proc(t: ^testing.T) {
    x := om.zeros(f32, [2]int{3, 7})
    defer om.tensor_destroy(&x)
    for &v, i in x.data { v = f32(i) / 7 }
    y := om.zeros(complex64, [2]int{3, 4})
    z := om.zeros(f32, x.shape)
    defer om.tensor_destroy(&y)
    defer om.tensor_destroy(&z)
    fwd := om.fft_plan(f32, .R2C, x.shape, {.Batched})
    inv := om.fft_plan(f32, .C2R, x.shape, {.Batched})
    defer om.fft_destroy(&fwd)
    defer om.fft_destroy(&inv)
    om.fft_execute(fwd, x, y)
    om.fft_execute(inv, y, z)
    testing.expect(t, om.allclose(z, x, rtol = 0, atol = 1e-5))

    c := om.random_complex(complex128, [1]int{7})
    defer om.tensor_destroy(&c)
    want := om.fft(c)
    defer om.tensor_destroy(&want)
    p := om.fft_plan(f64, .C2C, c.shape, {.In_Place})
    defer om.fft_destroy(&p)
    om.fft_execute(p, c, c)
    testing.expect(t, om.allclose(c, want, rtol = 0, atol = 1e-12))
}

@(test)
fft_into_real_batched_views :: proc(t: ^testing.T) {
    parent := om.zeros(f64, [2]int{5, 11})
    defer om.tensor_destroy(&parent)
    x := om.view(parent, {1, 2}, {4, 9})
    for i in 0 ..< 3 {
        for j in 0 ..< 7 { x.data[i * x.strides[0] + j * x.strides[1]] = f64(7 * i + j) }
    }
    spectrum_parent := om.zeros(complex128, [2]int{6, 5})
    defer om.tensor_destroy(&spectrum_parent)
    y := om.transpose_view(om.view(spectrum_parent, {1, 1}, {5, 4}))
    real_parent := om.zeros(f64, parent.shape)
    defer om.tensor_destroy(&real_parent)
    z := om.view(real_parent, {1, 2}, {4, 9})
    om.fft_into(x, y, {.Batched})
    om.fft_into(y, z, {.Inverse, .Batched})
    testing.expect(t, om.allclose(z, x, rtol = 0, atol = 1e-12))
    testing.expect_value(t, om.atv(real_parent, 0, 0), 0.0)
}

@(test)
fft_into_in_place_view :: proc(t: ^testing.T) {
    parent := om.random_complex(complex64, [2]int{7, 3})
    defer om.tensor_destroy(&parent)
    x := om.view_axis(parent, 1, 1)
    original := om.tensor_copy(x)
    defer om.tensor_destroy(&original)
    om.fft_into(x, x)
    om.fft_into(x, x, {.Inverse})
    testing.expect(t, om.allclose(x, original, rtol = 0, atol = 1e-5))
}
