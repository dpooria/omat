package omat

import "base:builtin"
import "core:math"

// Special functions and quadratures: real spherical
// harmonics, spherical Bessel functions, Gaunt coefficients and radial transforms.

// Nodes x and weights w of n-point Gauss-Legendre quadrature on [-1, 1]: exact for polynomials
// up to degree 2n - 1. Map to [a, b] with (b - a)/2 x + (a + b)/2 and weights times (b - a)/2.
@(require_results)
gauss_legendre :: proc(n: int, allocator := context.allocator) -> (x, w: Tensor(f64, 1)) {
    assert(n > 0, "gauss_legendre: need at least one node")
    x = zeros(f64, [1]int{n}, allocator)
    w = zeros(f64, [1]int{n}, allocator)
    for i in 0 ..< (n + 1) / 2 {
        // Newton's method on P_n from the Chebyshev-like first guess
        z := math.cos(math.PI * (f64(i) + 0.75) / (f64(n) + 0.5))
        derivative: f64
        for _ in 0 ..< 100 {
            p0, p1 := 1.0, 0.0
            for j in 1 ..= n {
                p0, p1 = ((2 * f64(j) - 1) * z * p0 - (f64(j) - 1) * p1) / f64(j), p0
            }
            derivative = f64(n) * (z * p0 - p1) / (z * z - 1)
            step := p0 / derivative
            z -= step
            if builtin.abs(step) < 1e-15 {
                break
            }
        }
        x.data[i], x.data[n - 1 - i] = -z, z
        weight := 2 / ((1 - z * z) * derivative * derivative)
        w.data[i], w.data[n - 1 - i] = weight, weight
    }
    return x, w
}

// index of Y_lm in the tables below: l^2 + l + m
ylm_index :: proc(l, m: int) -> int {
    return l * l + l + m
}

// Real spherical harmonics Y_lm(r / |r|) for l = 0 ..= lmax into out[ylm_index(l, m)], 
// which needs (lmax + 1)^2 entries.
// Orthonormal on the unit sphere, without the
// Y_l0 is real Y_l^0, m > 0 goes as cos(m phi), m < 0 as sin(|m| phi)
// so Y_1,-1, Y_10, Y_11 are sqrt(3 / 4pi) (y, z, x) / r. At r = 0 the direction is z.
ylm_real_into :: proc(lmax: int, r: [3]f64, out: []f64) {
    assert(lmax >= 0 && len(out) >= (lmax + 1) * (lmax + 1), "ylm_real: out too short")
    norm := math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z)
    ct, st := 1.0, 0.0 // cos and sin of theta
    cp, sp := 1.0, 0.0 // cos and sin of phi
    if norm > 0 {
        ct = r.z / norm
        rho := math.sqrt(r.x * r.x + r.y * r.y)
        st = rho / norm
        if rho > 0 {
            cp, sp = r.x / rho, r.y / rho
        }
    }

    // p = P_m^m, then upward in l; cos(m phi), sin(m phi) by rotation
    pmm := 1.0
    cm, sm := 1.0, 0.0
    for m in 0 ..= lmax {
        if m > 0 {
            pmm *= f64(2 * m - 1) * st
            cm, sm = cm * cp - sm * sp, sm * cp + cm * sp
        }
        p_prev, p := 0.0, pmm
        for l in m ..= lmax {
            if l > m {
                p_prev, p = p, (f64(2 * l - 1) * ct * p - f64(l + m - 1) * p_prev) / f64(l - m)
            }
            // N_lm = sqrt((2l + 1) / 4pi (l - m)! / (l + m)!)
            ratio := 1.0
            for k in l - m + 1 ..= l + m {
                ratio /= f64(k)
            }
            n := math.sqrt(f64(2 * l + 1) / (4 * math.PI) * ratio)
            if m == 0 {
                out[ylm_index(l, 0)] = n * p
            } else {
                out[ylm_index(l, m)] = math.SQRT_TWO * n * p * cm
                out[ylm_index(l, -m)] = math.SQRT_TWO * n * p * sm
            }
        }
    }
}

// Y_lm of every row of points (n, 3) as rows of a new (n, (lmax + 1)^2) table
@(require_results)
ylm_real :: proc(lmax: int, points: Tensor(f64, 2), allocator := context.allocator) -> Tensor(f64, 2) {
    assert(points.shape[1] == 3, "ylm_real: points need 3 coordinates")
    res := zeros(f64, [2]int{points.shape[0], (lmax + 1) * (lmax + 1)}, allocator)
    width := res.shape[1]
    for i in 0 ..< points.shape[0] {
        r := [3]f64{atv(points, i, 0), atv(points, i, 1), atv(points, i, 2)}
        ylm_real_into(lmax, r, res.data[i * width:][:width])
    }
    return res
}

// Spherical Bessel functions j_0(x) ..= j_lmax(x) into out: upward recurrence from j_0 and j_1
// where it is stable (x > lmax), the power series otherwise.
sph_bessel_into :: proc(lmax: int, x: f64, out: []f64) {
    assert(lmax >= 0 && len(out) > lmax, "sph_bessel: out too short")
    if x > f64(lmax) && x > 0.5 {
        s, c := math.sin(x), math.cos(x)
        out[0] = s / x
        if lmax > 0 {
            out[1] = s / (x * x) - c / x
        }
        for l in 2 ..= lmax {
            out[l] = f64(2 * l - 1) / x * out[l - 1] - out[l - 2]
        }
        return
    }
    // j_l(x) = x^l / (2l + 1)!! sum_k (-x^2 / 2)^k / (k! (2l + 3)(2l + 5)...(2l + 2k + 1))
    lead := 1.0 // x^l / (2l + 1)!!
    for l in 0 ..= lmax {
        if l > 0 {
            lead *= x / f64(2 * l + 1)
        }
        term, sum := 1.0, 1.0
        for k in 1 ..< 200 {
            term *= -x * x / (2 * f64(k) * f64(2 * l + 2 * k + 1))
            sum += term
            if builtin.abs(term) < 1e-17 * builtin.abs(sum) {
                break
            }
        }
        out[l] = lead * sum
    }
}

// j_l(x)
sph_bessel :: proc(l: int, x: f64) -> f64 {
    buffer: [64]f64
    assert(l < len(buffer), "sph_bessel: l too large")
    sph_bessel_into(l, x, buffer[:l + 1])
    return buffer[l]
}

// The Gaunt coefficient of real harmonics:  integral{Y_l1m1 Y_l2m2 Y_l3m3 dΩ},
// by a quadrature that is exact for these products. Tabulate once; each call does a small quadrature.
gaunt :: proc(l1, m1, l2, m2, l3, m3: int) -> f64 {
    L := l1 + l2 + l3
    lmax := builtin.max(l1, l2, l3)
    // products of real harmonics are polynomials of degree L on the sphere: Gauss-Legendre in
    // cos(theta), and L + 2 equally spaced phi, integrate them exactly
    nt, np := L / 2 + 1, L + 2
    x, w := gauss_legendre(nt)
    defer tensor_destroy(&x)
    defer tensor_destroy(&w)
    ylm := make([]f64, (lmax + 1) * (lmax + 1))
    defer delete(ylm)
    total := 0.0
    for i in 0 ..< nt {
        ct := x.data[i]
        st := math.sqrt(1 - ct * ct)
        for j in 0 ..< np {
            phi := 2 * math.PI * f64(j) / f64(np)
            ylm_real_into(lmax, {st * math.cos(phi), st * math.sin(phi), ct}, ylm)
            f := ylm[ylm_index(l1, m1)] * ylm[ylm_index(l2, m2)] * ylm[ylm_index(l3, m3)]
            total += w.data[i] * (2 * math.PI / f64(np)) * f
        }
    }
    return total
}

// f(q) = 4 pi sum_i weights[i] r[i]^2 j_l(q r[i]) f[i] for every q: the spherical Bessel
// transform of a radial function, with weights from a radial quadrature 
// e.g. simpson_weights times dr/di on a logarithmic grid
@(require_results)
bessel_transform :: proc(
    l: int,
    r, f, weights, q: Tensor(f64, 1),
    allocator := context.allocator,
) -> Tensor(f64, 1) {
    assert(r.shape == f.shape && r.shape == weights.shape, "bessel_transform: radial shapes differ")
    res := zeros(f64, q.shape, allocator)
    for iq in 0 ..< q.shape[0] {
        qv := atv(q, iq)
        total := 0.0
        for i in 0 ..< r.shape[0] {
            ri := atv(r, i)
            total += atv(weights, i) * ri * ri * sph_bessel(l, qv * ri) * atv(f, i)
        }
        res.data[iq] = 4 * math.PI * total
    }
    return res
}
