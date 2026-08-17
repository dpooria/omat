package omat

import "core:math"
import "core:math/rand"
import "core:slice"

zeros :: proc($T: typeid, shape: [$N]int, allocator := context.allocator) -> Tensor(T, N) {
    result, ok := tensor_create(T, shape, allocator)
    assert(ok)
    return result
}

ones :: proc($T: typeid, shape: [$N]int, allocator := context.allocator) -> Tensor(T, N) {
    t, ok := tensor_create(T, shape, allocator)
    assert(ok)
    slice.fill(t.data, 1)
    return t
}

identity :: proc($T: typeid, n: int, allocator := context.allocator) -> Tensor(T, 2) {
    t := zeros(T, [2]int{n, n}, allocator)
    for i in 0 ..< n {
        a := at(t, i, i)
        a^ = T(1)
    }
    return t
}

fill :: proc(t: ^Tensor($T, $N), value: T) {
    slice.fill(t.data, value)
}

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

random_uniform :: proc(
    shape: [$N]int,
    low, high: f64,
    generator := context.random_generator,
) -> Tensor(f64, N) {
    result := zeros(f64, shape)

    for &value in result.data {
        value = rand.float64_range(low, high, generator)
    }

    return result
}

random_normal :: proc(
    shape: [$N]int,
    mean: f64 = 0.0,
    stddev: f64 = 1.0,
    generator := context.random_generator,
) -> Tensor(f64, N) {
    result := zeros(f64, shape)

    for &value in result.data {
        value = rand.float64_normal(mean, stddev, generator)
    }
    return result
}

