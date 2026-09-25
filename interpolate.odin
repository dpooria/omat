package omat

import "core:mem"
import "core:slice"

// Cubic spline through (x[i], y[i]) for increasing x: radial functions and form factor tables.
// Natural (zero curvature) at the ends unless end slopes are given. Owns its arrays.
Spline :: struct {
    allocator: mem.Allocator,
    x, y, d2:  []f64, // d2: second derivatives at the knots
}

@(require_results)
spline_make :: proc(
    x, y: Tensor(f64, 1),
    slopes: Maybe([2]f64) = nil,
    allocator := context.allocator,
) -> Spline {
    n := x.shape[0]
    assert(n >= 2 && y.shape[0] == n, "spline: need matching x and y of at least 2 points")
    s := Spline {
        allocator = allocator,
        x         = make([]f64, n, allocator),
        y         = make([]f64, n, allocator),
        d2        = make([]f64, n, allocator),
    }
    for i in 0 ..< n {
        s.x[i], s.y[i] = atv(x, i), atv(y, i)
        assert(i == 0 || s.x[i] > s.x[i - 1], "spline: x must increase")
    }

    // solved by forward elimination and back substitution
    u := make([]f64, n)
    defer delete(u)
    ends, clamped := slopes.?
    if clamped {
        h := s.x[1] - s.x[0]
        s.d2[0] = -0.5
        u[0] = 3 / h * ((s.y[1] - s.y[0]) / h - ends[0])
    }
    for i in 1 ..< n - 1 {
        sig := (s.x[i] - s.x[i - 1]) / (s.x[i + 1] - s.x[i - 1])
        p := sig * s.d2[i - 1] + 2
        s.d2[i] = (sig - 1) / p
        slope_diff :=
            (s.y[i + 1] - s.y[i]) / (s.x[i + 1] - s.x[i]) -
            (s.y[i] - s.y[i - 1]) / (s.x[i] - s.x[i - 1])
        u[i] = (6 * slope_diff / (s.x[i + 1] - s.x[i - 1]) - sig * u[i - 1]) / p
    }
    qn, un := 0.0, 0.0
    if clamped {
        h := s.x[n - 1] - s.x[n - 2]
        qn = 0.5
        un = 3 / h * (ends[1] - (s.y[n - 1] - s.y[n - 2]) / h)
    }
    s.d2[n - 1] = (un - qn * u[n - 2]) / (qn * s.d2[n - 2] + 1)
    for i := n - 2; i >= 0; i -= 1 {
        s.d2[i] = s.d2[i] * s.d2[i + 1] + u[i]
    }
    return s
}

spline_destroy :: proc(s: ^Spline) {
    delete(s.x, s.allocator)
    delete(s.y, s.allocator)
    delete(s.d2, s.allocator)
    s^ = {}
}

// the knot interval holding x and position: x = x[i] + t h; cubic extrapolation outside
@(private = "file")
spline_interval :: proc(s: Spline, x: f64) -> (i: int, h, a, b: f64) {
    n := len(s.x)
    i, _ = slice.binary_search(s.x, x)
    i = clamp(i - 1, 0, n - 2)
    h = s.x[i + 1] - s.x[i]
    a = (s.x[i + 1] - x) / h
    b = (x - s.x[i]) / h
    return i, h, a, b
}

spline_eval :: proc(s: Spline, x: f64) -> f64 {
    i, h, a, b := spline_interval(s, x)
    return(
        a * s.y[i] +
        b * s.y[i + 1] +
        ((a * a * a - a) * s.d2[i] + (b * b * b - b) * s.d2[i + 1]) * h * h / 6 \
    )
}

spline_derivative :: proc(s: Spline, x: f64) -> f64 {
    i, h, a, b := spline_interval(s, x)
    return(
        (s.y[i + 1] - s.y[i]) / h +
        ((1 - 3 * a * a) * s.d2[i] + (3 * b * b - 1) * s.d2[i + 1]) * h / 6 \
    )
}

// the spline at every point of xs, into a new tensor
@(require_results)
spline_eval_tensor :: proc(
    s: Spline,
    xs: Tensor(f64, $N),
    allocator := context.allocator,
) -> Tensor(f64, N) {
    res := tensor_copy(xs, allocator)
    for &v in res.data {
        v = spline_eval(s, v)
    }
    return res
}

