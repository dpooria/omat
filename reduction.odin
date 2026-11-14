package omat

import "base:builtin"
import "base:intrinsics"
import "core:math"

// Reductions over all elements of a tensor or view.

ReduceOp :: enum u8 {
    Sum,
    Prod,
    Max,
    Min,
}

// op over all elements; Max and Min need at least one element
reduce :: proc(t: Tensor($T, $N), $op: ReduceOp) -> T {
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    when op == .Sum {
        acc := T(0)
    } else when op == .Prod {
        acc := T(1)
    } else {
        assert(l.rows > 0, "reduce: max and min need at least one element")
        acc := t.data[0]
    }
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            x := t.data[p + k * step]
            when op == .Sum {
                acc += x
            } else when op == .Prod {
                acc *= x
            } else when op == .Max {
                acc = builtin.max(acc, x)
            } else {
                acc = builtin.min(acc, x)
            }
        }
    }
    return acc
}

sum :: proc(t: Tensor($T, $N)) -> T {
    return reduce(t, .Sum)
}

prod :: proc(t: Tensor($T, $N)) -> T {
    return reduce(t, .Prod)
}

max :: proc(t: Tensor($T, $N)) -> T where !intrinsics.type_is_complex(T) {
    return reduce(t, .Max)
}

min :: proc(t: Tensor($T, $N)) -> T where !intrinsics.type_is_complex(T) {
    return reduce(t, .Min)
}

// index of the first largest (largest = true) or smallest element
arg_extreme :: proc(
    t: Tensor($T, $N),
    largest: bool,
) -> [N]int where !intrinsics.type_is_complex(T) {
    l := plan_loop(t.shape, t.strides)
    assert(l.rows > 0, "argmax, argmin: need at least one element")
    step := loop_step(l, t.strides)
    best, best_at := t.data[0], 0
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            x := t.data[p + k * step]
            if (largest && x > best) || (!largest && x < best) {
                best, best_at = x, r * l.n + k // the row-major position
            }
        }
    }
    return unravel_index(best_at, t.shape)
}

argmax :: proc(t: Tensor($T, $N)) -> [N]int {
    return arg_extreme(t, true)
}

argmin :: proc(t: Tensor($T, $N)) -> [N]int {
    return arg_extreme(t, false)
}

mean :: proc(t: Tensor($T, $N)) -> T {
    return sum(t) / T(f64(size_of_shape(t.shape)))
}

// mean of |x - mean|^2
var :: proc(t: Tensor($T, $N)) -> T {
    mu := mean(t)
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    v := T(0)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            d := t.data[p + k * step] - mu
            v += d * conjugate(d)
        }
    }
    return v / T(f64(size_of_shape(t.shape)))
}

std :: proc(t: Tensor($T, $N)) -> T where !intrinsics.type_is_complex(T) {
    return math.sqrt(var(t))
}

// op along one axis, into a tensor of the other axes: e.g. sum_axis(abs2(psi), 0) sums over bands
@(require_results)
reduce_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    $op: ReduceOp,
    allocator := context.allocator,
) -> Tensor(T, N - 1) where N >
    1 {
    assert(0 <= axis && axis < N, "reduce_axis: bad axis")
    // move axis last, then every row of the loop is one output element, in row-major order
    axes: [N]int
    next := 0
    for d in 0 ..< N {
        if d != axis {
            axes[next] = d
            next += 1
        }
    }
    axes[N - 1] = axis
    v := permute_view(t, axes[:])
    out_shape: [N - 1]int
    for d in 0 ..< N - 1 {
        out_shape[d] = v.shape[d]
    }
    res := zeros(T, out_shape, allocator)
    n := v.shape[N - 1]
    assert(n > 0 || op == .Sum || op == .Prod, "reduce_axis: max and min need a non-empty axis")
    l := Loop(N) {
        shape = v.shape,
        rows  = len(res.data),
        n     = n,
    }
    step := v.strides[N - 1]
    for r in 0 ..< l.rows {
        p := row_offset(l, r, v.strides)
        when op == .Sum {
            acc := T(0)
        } else when op == .Prod {
            acc := T(1)
        } else {
            acc := v.data[p]
        }
        for k in 0 ..< n {
            x := v.data[p + k * step]
            when op == .Sum {
                acc += x
            } else when op == .Prod {
                acc *= x
            } else when op == .Max {
                acc = builtin.max(acc, x)
            } else {
                acc = builtin.min(acc, x)
            }
        }
        res.data[r] = acc
    }
    return res
}

@(require_results)
sum_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    allocator := context.allocator,
) -> Tensor(T, N - 1) {
    return reduce_axis(t, axis, .Sum, allocator)
}

@(require_results)
prod_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    allocator := context.allocator,
) -> Tensor(T, N - 1) {
    return reduce_axis(t, axis, .Prod, allocator)
}

@(require_results)
max_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    allocator := context.allocator,
) -> Tensor(T, N - 1) where !intrinsics.type_is_complex(T) {
    return reduce_axis(t, axis, .Max, allocator)
}

@(require_results)
min_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    allocator := context.allocator,
) -> Tensor(T, N - 1) where !intrinsics.type_is_complex(T) {
    return reduce_axis(t, axis, .Min, allocator)
}

@(require_results)
mean_axis :: proc(
    t: Tensor($T, $N),
    axis: int,
    allocator := context.allocator,
) -> Tensor(T, N - 1) {
    res := reduce_axis(t, axis, .Sum, allocator)
    idivide(&res, T(f64(t.shape[axis])))
    return res
}

// |a - b| <= atol + rtol |b| for every element, as numpy's allclose (tensor-tensor)
allclose_tt :: proc(a, b: Tensor($T, $N), rtol := 1e-5, atol := 1e-8) -> bool {
    assert(a.shape == b.shape, "allclose: shapes differ")
    l := plan_loop(a.shape, a.strides, b.strides)
    sa, sb := loop_step(l, a.strides), loop_step(l, b.strides)
    for r in 0 ..< l.rows {
        pa, pb := row_offset(l, r, a.strides), row_offset(l, r, b.strides)
        for k in 0 ..< l.n {
            x, y := a.data[pa + k * sa], b.data[pb + k * sb]
            if f64(builtin.abs(x - y)) > atol + rtol * f64(builtin.abs(y)) {
                return false
            }
        }
    }
    return true
}

// |a - b| <= atol + rtol |b| for every element, as numpy's allclose (tensor-slice)
allclose_ts :: proc(a: Tensor($T, $N), b: []T, rtol := 1e-5, atol := 1e-8) -> bool {
    assert(size_of_shape(a.shape) == len(b), "allclose: size mismatch")
    return allclose_tt(a, tensor_from_data(a.shape, b))
}

// |a - b| <= atol + rtol |b| for every element, as numpy's allclose
allclose :: proc {
    allclose_tt,
    allclose_ts,
}

