package omat

// Quadrature of sampled functions. For a radial integral on a non-uniform grid r(i), integrate
// f(r(i)) dr/di over the uniform index with simpson_weights: sum_i w[i] dr/di f(r(i)).

// Weights of composite Simpson's rule for n samples (with spacing h), so that the integral is
// sum_i w[i] y[i]. An even n closes with Simpson's 3/8 rule over the last three intervals.
@(require_results)
simpson_weights :: proc(n: int, h: f64, allocator := context.allocator) -> Tensor(f64, 1) {
    w := zeros(f64, [1]int{n}, allocator)
    switch {
    case n < 2:
        return w
    case n == 2:
        w.data[0], w.data[1] = h / 2, h / 2
        return w
    case n == 4:
        w.data[0], w.data[1], w.data[2], w.data[3] = 3 * h / 8, 9 * h / 8, 9 * h / 8, 3 * h / 8
        return w
    }
    simpson_end := n if n % 2 == 1 else n - 3 // samples covered by the 1-4-2-4-1 pattern
    for i in 0 ..< simpson_end {
        switch {
        case i == 0 || i == simpson_end - 1:
            w.data[i] += h / 3
        case i % 2 == 1:
            w.data[i] += 4 * h / 3
        case:
            w.data[i] += 2 * h / 3
        }
    }
    if n % 2 == 0 {
        eighths := [4]f64{3, 9, 9, 3}
        for k in 0 ..< 4 {
            w.data[n - 4 + k] += eighths[k] * h / 8
        }
    }
    return w
}

// The integral of samples y spaced h apart by composite Simpson's rule; exact for cubics
simpson :: proc(y: Tensor($T, 1), h: f64) -> T {
    w := simpson_weights(y.shape[0], h)
    defer tensor_destroy(&w)
    total := T(0)
    for i in 0 ..< y.shape[0] {
        total += T(w.data[i]) * y.data[i * y.strides[0]]
    }
    return total
}

// The integral of samples y at the increasing points x by the trapezoidal rule
trapezoid :: proc(y: Tensor($T, 1), x: Tensor(f64, 1)) -> T {
    assert(y.shape == x.shape, "trapezoid: y and x differ in length")
    total := T(0)
    for i in 1 ..< x.shape[0] {
        dx := atv(x, i) - atv(x, i - 1)
        total += T(dx / 2) * (atv(y, i) + atv(y, i - 1))
    }
    return total
}

