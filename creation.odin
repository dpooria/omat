package omat

import "base:intrinsics"
import "core:math"
import "core:math/rand"
import "core:slice"

@(require_results)
zeros :: proc($T: typeid, shape: [$N]int, allocator := context.allocator) -> Tensor(T, N) {
    result, ok := tensor_create(T, shape, allocator)
    assert(ok)
    return result
}

@(require_results)
ones :: proc($T: typeid, shape: [$N]int, allocator := context.allocator) -> Tensor(T, N) {
    t, ok := tensor_create(T, shape, allocator)
    assert(ok)
    slice.fill(t.data, 1)
    return t
}

@(require_results)
identity :: proc($T: typeid, n: int, allocator := context.allocator) -> Tensor(T, 2) {
    t := zeros(T, [2]int{n, n}, allocator)
    for i in 0 ..< n {
        at(t, i, i)^ = T(1)
    }
    return t
}

// square matrix with v on its diagonal
@(require_results)
diag :: proc(v: Tensor($T, 1), allocator := context.allocator) -> Tensor(T, 2) {
    n := v.shape[0]
    res := zeros(T, [2]int{n, n}, allocator)
    assign(diagonal(res), v)
    return res
}

// works on views too, e.g. zeroing a block
fill :: proc(t: ^Tensor($T, $N), value: T) {
    value := value
    assign(t^, broadcast(&value, t.shape))
}

@(require_results)
arange :: proc($T: typeid, low: T, high: T, h: T, allocator := context.allocator) -> Tensor(T, 1) {

    if (h == 0) || (h > 0 && high <= low) || (h < 0 && high >= low) {
        return zeros(T, [1]int{0}, allocator)
    }

    steps := (f64(high) - f64(low)) / f64(h)
    N := int(math.ceil(steps - 1e-10))

    res := zeros(T, [1]int{N}, allocator)
    for i in 0 ..< N {
        res.data[i] = low + (T(i) * h)
    }

    return res
}

@(require_results)
linspace :: proc(low: f64, high: f64, N: int, allocator := context.allocator) -> Tensor(f64, 1) {
    if N <= 0 {
        return zeros(f64, [1]int{0}, allocator)
    }

    res := zeros(f64, [1]int{N}, allocator)

    if N == 1 {
        res.data[0] = low
        return res
    }

    h := (high - low) / f64(N - 1)

    for i in 0 ..< (N - 1) {
        res.data[i] = low + f64(i) * h
    }

    res.data[N - 1] = high

    return res
}

@(require_results)
random_uniform :: proc(
    shape: [$N]int,
    low, high: f64,
    generator := context.random_generator,
    allocator := context.allocator,
) -> Tensor(f64, N) {
    result := zeros(f64, shape, allocator)

    for &value in result.data {
        value = rand.float64_range(low, high, generator)
    }

    return result
}

@(require_results)
random_normal :: proc(
    shape: [$N]int,
    mean: f64 = 0.0,
    stddev: f64 = 1.0,
    generator := context.random_generator,
    allocator := context.allocator,
) -> Tensor(f64, N) {
    result := zeros(f64, shape, allocator)

    for &value in result.data {
        value = rand.float64_normal(mean, stddev, generator)
    }
    return result
}

// uniform in [0, 1) for real types and for both parts of complex ones
fill_randu :: proc(t: ^Tensor($T, $N), generator := context.random_generator) {
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            t.data[p + k * step] = random_value(T, false, generator)
        }
    }
}

// standard normal for real types and for both parts of complex ones
fill_randn :: proc(t: ^Tensor($T, $N), generator := context.random_generator) {
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            t.data[p + k * step] = random_value(T, true, generator)
        }
    }
}

random_value :: proc($T: typeid, normal: bool, generator: rand.Generator) -> T {
    when intrinsics.type_is_complex(T) {
        R :: intrinsics.type_elem_type(T)
        return complex(random_value(R, normal, generator), random_value(R, normal, generator))
    } else when intrinsics.type_is_float(T) {
        if normal {
            return T(rand.float64_normal(0, 1, generator))
        }
        return T(rand.float64_range(0, 1, generator))
    } else {
        #assert(intrinsics.type_is_integer(T), "random_value: numeric types only")
        // integers: 0 or 1 uniformly; normal draws are rounded
        if normal {
            return T(math.round(rand.float64_normal(0, 1, generator)))
        }
        return T(rand.int63_max(2, generator))
    }
}

