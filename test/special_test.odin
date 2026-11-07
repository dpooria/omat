package test

import "core:math"
import "core:testing"

import om ".."

@(test)
special_gauss_legendre :: proc(t: ^testing.T) {
    for n in 1 ..= 12 {
        x, w := om.gauss_legendre(n)
        defer om.tensor_destroy(&x)
        defer om.tensor_destroy(&w)
        // exact for x^k, k <= 2n - 1: the integral over [-1, 1] is 2 / (k + 1) for even k
        for k in 0 ..< 2 * n {
            total := 0.0
            for i in 0 ..< n {
                total += w.data[i] * math.pow_f64(x.data[i], f64(k))
            }
            want := 2 / f64(k + 1) if k % 2 == 0 else 0
            testing.expectf(t, abs(total - want) < 1e-13, "n=%d k=%d: %v", n, k, total)
        }
        for i in 1 ..< n {
            testing.expect(t, x.data[i - 1] < x.data[i])
        }
    }
}

@(test)
special_real_spherical_harmonics :: proc(t: ^testing.T) {
    lmax :: 4
    count :: (lmax + 1) * (lmax + 1)
    ylm: [count]f64

    // l = 1 is sqrt(3 / 4pi) (y, z, x) / r, l = 0 is 1 / sqrt(4pi)
    r := [3]f64{1, -2, 2}
    om.ylm_real_into(lmax, r, ylm[:])
    c1 := math.sqrt_f64(3 / (4 * math.PI))
    testing.expect(t, abs(ylm[om.ylm_index(0, 0)] - 1 / math.sqrt_f64(4 * math.PI)) < 1e-15)
    testing.expect(t, abs(ylm[om.ylm_index(1, -1)] - c1 * -2 / 3) < 1e-15)
    testing.expect(t, abs(ylm[om.ylm_index(1, 0)] - c1 * 2 / 3) < 1e-15)
    testing.expect(t, abs(ylm[om.ylm_index(1, 1)] - c1 * 1 / 3) < 1e-15)
    // l = 2: sqrt(15 / 4pi) xy / r^2 and sqrt(5 / 16pi) (3 z^2 - r^2) / r^2
    testing.expect(
        t,
        abs(ylm[om.ylm_index(2, -2)] - math.sqrt_f64(15 / (4 * math.PI)) * -2 / 9) < 1e-15,
    )
    testing.expect(
        t,
        abs(ylm[om.ylm_index(2, 0)] - math.sqrt_f64(5 / (16 * math.PI)) * (12 - 9) / 9.0) < 1e-15,
    )

    // orthonormal on the sphere, by a product quadrature exact to degree 2 lmax
    x, w := om.gauss_legendre(lmax + 1)
    defer om.tensor_destroy(&x)
    defer om.tensor_destroy(&w)
    np := 2 * lmax + 2
    gram: [count][count]f64
    for i in 0 ..< lmax + 1 {
        st := math.sqrt_f64(1 - x.data[i] * x.data[i])
        for j in 0 ..< np {
            phi := 2 * math.PI * f64(j) / f64(np)
            om.ylm_real_into(lmax, {st * math.cos(phi), st * math.sin(phi), x.data[i]}, ylm[:])
            for a in 0 ..< count {
                for b in 0 ..< count {
                    gram[a][b] += w.data[i] * 2 * math.PI / f64(np) * ylm[a] * ylm[b]
                }
            }
        }
    }
    for a in 0 ..< count {
        for b in 0 ..< count {
            want := 1.0 if a == b else 0
            testing.expectf(t, abs(gram[a][b] - want) < 1e-13, "<%d|%d> = %v", a, b, gram[a][b])
        }
    }

    // the direction only matters, and r = 0 and the poles are fine
    om.ylm_real_into(lmax, {0, 0, 0}, ylm[:])
    testing.expect(t, abs(ylm[om.ylm_index(1, 0)] - c1) < 1e-15)
    om.ylm_real_into(lmax, {0, 0, -3}, ylm[:])
    testing.expect(t, abs(ylm[om.ylm_index(1, 0)] + c1) < 1e-15)
    testing.expect_value(t, ylm[om.ylm_index(3, 2)], 0.0)

    // tables for many points
    pts := om.zeros(f64, [2]int{2, 3})
    defer om.tensor_destroy(&pts)
    pts.data[0], pts.data[4] = 2, 5 // x and y directions
    table := om.ylm_real(1, pts)
    defer om.tensor_destroy(&table)
    testing.expect_value(t, table.shape, [2]int{2, 4})
    testing.expect(t, abs(om.atv(table, 0, om.ylm_index(1, 1)) - c1) < 1e-15)
    testing.expect(t, abs(om.atv(table, 1, om.ylm_index(1, -1)) - c1) < 1e-15)
}

@(test)
special_spherical_bessel :: proc(t: ^testing.T) {
    closed := proc(l: int, x: f64) -> f64 {
        s, c := math.sin(x), math.cos(x)
        switch l {
        case 0:
            return s / x
        case 1:
            return s / (x * x) - c / x
        case 2:
            return (3 / (x * x) - 1) * s / x - 3 * c / (x * x)
        case:
            return (15 / (x * x * x) - 6 / x) * s / x - (15 / (x * x) - 1) * c / x
        }
    }
    for x in ([]f64{0.8, 1.5, 2.7, 4.0, 7.3, 15.0, 40.0}) {
        for l in 0 ..= 3 {
            testing.expectf(t, abs(om.sph_bessel(l, x) - closed(l, x)) < 1e-12, "j_%d(%v)", l, x)
        }
    }
    // small arguments: the series, j_l(x) ~ x^l / (2l + 1)!!
    testing.expect_value(t, om.sph_bessel(0, 0), 1.0)
    testing.expect_value(t, om.sph_bessel(3, 0), 0.0)
    testing.expect(t, abs(om.sph_bessel(2, 1e-3) - 1e-6 / 15 * (1 - 1e-6 / 14)) < 1e-20)
    testing.expect(t, abs(om.sph_bessel(1, 0.3) - closed(1, 0.3)) < 1e-14)

    // high orders against the recurrence j_{l-1} + j_{l+1} = (2l + 1) / x j_l, both branches
    js: [12]f64
    for x in ([]f64{2.0, 6.5, 9.0, 12.5}) {
        om.sph_bessel_into(11, x, js[:])
        for l in 1 ..= 10 {
            lhs := js[l - 1] + js[l + 1]
            rhs := f64(2 * l + 1) / x * js[l]
            testing.expectf(t, abs(lhs - rhs) < 1e-10 * (abs(lhs) + 1e-3), "l=%d x=%v", l, x)
        }
        testing.expect(t, abs(js[3] - closed(3, x)) < 1e-11)
    }
}

@(test)
special_gaunt :: proc(t: ^testing.T) {
    // with Y_00: 1 / sqrt(4pi) delta
    for l in 0 ..= 3 {
        for m in -l ..= l {
            testing.expect(
                t,
                abs(om.gaunt(0, 0, l, m, l, m) - 1 / math.sqrt_f64(4 * math.PI)) < 1e-14,
            )
            for m2 in -l ..= l {
                if m2 != m {
                    testing.expect(t, abs(om.gaunt(0, 0, l, m, l, m2)) < 1e-14)
                }
            }
        }
    }
    // m = 0: sqrt((2l1 + 1)(2l2 + 1)(2l3 + 1) / 4pi) (l1 l2 l3; 0 0 0)^2, with (1 1 2; 0 0 0)^2 = 2/15
    testing.expect(
        t,
        abs(om.gaunt(1, 0, 1, 0, 2, 0) - math.sqrt_f64(45 / (4 * math.PI)) * 2 / 15) < 1e-14,
    )
    // x, y and xy: (3 / 4pi) sqrt(15 / 4pi) the integral of x^2 y^2, which is 4pi / 15
    want := 3 / (4 * math.PI) * math.sqrt_f64(15 / (4 * math.PI)) * 4 * math.PI / 15
    testing.expect(t, abs(om.gaunt(1, 1, 1, -1, 2, -2) - want) < 1e-14)
    // selection rules: odd l1 + l2 + l3, and the triangle rule
    testing.expect(t, abs(om.gaunt(1, 0, 1, 0, 1, 0)) < 1e-15)
    testing.expect(t, abs(om.gaunt(1, 0, 1, 0, 4, 0)) < 1e-15)
    testing.expect(t, abs(om.gaunt(2, 1, 3, -2, 1, 1)) < 1e-14)
    // symmetric in its three pairs
    testing.expect(t, abs(om.gaunt(2, 1, 3, 2, 3, 1) - om.gaunt(3, 1, 2, 1, 3, 2)) < 1e-14)
}

@(test)
special_quadrature_and_transforms :: proc(t: ^testing.T) {
    // Simpson is exact for cubics, with an odd or even number of samples
    for n in ([]int{2, 3, 4, 5, 8, 11}) {
        h := 2.0 / f64(n - 1)
        y := om.zeros(f64, [1]int{n})
        defer om.tensor_destroy(&y)
        for i in 0 ..< n {
            x := -1 + f64(i) * h
            y.data[i] = 1 + x + 3 * x * x + x * x * x if n > 2 else 1 + x
        }
        want := 4.0 if n > 2 else 2.0 // over [-1, 1]
        testing.expectf(t, abs(om.simpson(y, h) - want) < 1e-13, "n=%d", n)
        w := om.simpson_weights(n, h)
        defer om.tensor_destroy(&w)
        testing.expect(t, abs(om.sum(w) - 2) < 1e-14)
    }
    one := om.ones(f64, [1]int{1})
    defer om.tensor_destroy(&one)
    testing.expect_value(t, om.simpson(one, 0.1), 0.0)

    // trapezoid on a non-uniform grid is exact for straight lines; complex values work too
    xs := om.zeros(f64, [1]int{4})
    defer om.tensor_destroy(&xs)
    copy(xs.data, []f64{0, 0.1, 0.5, 2})
    ys := om.zeros(complex128, [1]int{4})
    defer om.tensor_destroy(&ys)
    for x, i in xs.data {
        ys.data[i] = complex(3 * x + 1, -x)
    }
    testing.expect(t, abs(om.trapezoid(ys, xs) - complex(8.0, -2.0)) < 1e-14)

    // a radial integral on a logarithmic grid r(i) = a (e^(d i) - 1): sum w[i] dr/di f
    n :: 801
    a, d := 1e-4, 0.015
    r := om.zeros(f64, [1]int{n})
    defer om.tensor_destroy(&r)
    jac := om.zeros(f64, [1]int{n})
    defer om.tensor_destroy(&jac)
    for i in 0 ..< n {
        r.data[i] = a * (math.exp(d * f64(i)) - 1)
        jac.data[i] = a * d * math.exp(d * f64(i))
    }
    w := om.simpson_weights(n, 1)
    defer om.tensor_destroy(&w)
    om.imultiply(&w, jac)
    gauss := om.zeros(f64, [1]int{n})
    defer om.tensor_destroy(&gauss)
    for x, i in r.data {
        gauss.data[i] = math.exp(-x * x)
    }
    // 4 pi integral of r^2 e^(-r^2) is pi^(3/2), the q = 0 Bessel transform
    q := om.linspace(0, 6, 7)
    defer om.tensor_destroy(&q)
    ft := om.bessel_transform(0, r, gauss, w, q)
    defer om.tensor_destroy(&ft)
    for qv, i in q.data {
        want := math.pow_f64(math.PI, 1.5) * math.exp(-qv * qv / 4)
        testing.expectf(t, abs(ft.data[i] - want) < 1e-8, "q=%v: %v vs %v", qv, ft.data[i], want)
    }
    // l = 1 of r e^(-r^2): pi^(3/2) q / 2 e^(-q^2 / 4)
    rg := om.multiply(gauss, r)
    defer om.tensor_destroy(&rg)
    ft1 := om.bessel_transform(1, r, rg, w, q)
    defer om.tensor_destroy(&ft1)
    for qv, i in q.data {
        want := math.pow_f64(math.PI, 1.5) * qv / 2 * math.exp(-qv * qv / 4)
        testing.expect(t, abs(ft1.data[i] - want) < 1e-8)
    }
}

@(test)
special_splines :: proc(t: ^testing.T) {
    // clamped with the true end slopes, a spline reproduces a cubic exactly
    cubic := proc(x: f64) -> f64 {
        return 2 - x + 0.5 * x * x - 0.25 * x * x * x
    }
    slope := proc(x: f64) -> f64 {
        return -1 + x - 0.75 * x * x
    }
    x := om.zeros(f64, [1]int{7})
    defer om.tensor_destroy(&x)
    copy(x.data, []f64{-2, -1.5, -0.2, 0, 0.7, 1.9, 3}) // uneven knots
    y := om.apply(x, cubic)
    defer om.tensor_destroy(&y)
    s := om.spline_make(x, y, [2]f64{slope(-2), slope(3)})
    defer om.spline_destroy(&s)
    for v in ([]f64{-2, -1.7, -0.1, 0.35, 1.2, 2.5, 3}) {
        testing.expect(t, abs(om.spline_eval(s, v) - cubic(v)) < 1e-12)
        testing.expect(t, abs(om.spline_derivative(s, v) - slope(v)) < 1e-12)
    }
    // cubic extrapolation just outside
    testing.expect(t, abs(om.spline_eval(s, 3.1) - cubic(3.1)) < 1e-10)

    // natural spline of sin on a fine grid, evaluated for a whole tensor
    xs := om.linspace(0, math.PI, 41)
    defer om.tensor_destroy(&xs)
    ys := om.sin(xs)
    defer om.tensor_destroy(&ys)
    ns := om.spline_make(xs, ys)
    defer om.spline_destroy(&ns)
    probe := om.linspace(0.01, 3.1, 50)
    defer om.tensor_destroy(&probe)
    vals := om.spline_eval_tensor(ns, probe)
    defer om.tensor_destroy(&vals)
    want := om.sin(probe)
    defer om.tensor_destroy(&want)
    testing.expect(t, om.allclose(vals, want, 0, 1e-5))
    testing.expect(t, abs(om.spline_eval(ns, 0) - 0) < 1e-15) // at a knot
    testing.expect(t, abs(om.spline_derivative(ns, math.PI / 2)) < 1e-4)

    // two points: a straight line
    two_x := om.linspace(0, 1, 2)
    defer om.tensor_destroy(&two_x)
    two_y := om.linspace(3, 5, 2)
    defer om.tensor_destroy(&two_y)
    line := om.spline_make(two_x, two_y)
    defer om.spline_destroy(&line)
    testing.expect(t, abs(om.spline_eval(line, 0.25) - 3.5) < 1e-15)
}

