package omat

import "base:intrinsics"
import "core:fmt"
import "core:mem"

Tensor :: struct($T: typeid, $N: int) where intrinsics.type_is_numeric(T) {
    allocator: mem.Allocator,
    data:      []T,
    shape:     [N]int,
    // strides:   [N]int,
    // owns_data: bool,
}

@(require_results)
tensor_create :: proc(
    $T: typeid,
    shape: [$N]int,
    allocator := context.allocator,
) -> (
    t: Tensor(T, N),
    ok: bool,
) {
    t.allocator = allocator
    t.shape = shape
    size: int = 1
    for s in shape {
        size *= s
    }
    data, err := make([]T, size, allocator = t.allocator)
    ok = (err == nil)
    if !ok {
        fmt.eprintfln("Could not create an array with the shape (%v): %v", shape, err)
        return
    }
    t.data = data
    return
}

tensor_destroy :: proc(t: ^Tensor($T, $N)) {
    delete(t.data, t.allocator)
    t^ = {}
}

@(require_results)
tensor_copy :: proc(src: Tensor($T, $N)) -> (dst: Tensor(T, N), ok: bool) {
    dst, ok = tensor_create(T, src.shape, src.allocator)
    assert(ok)
    copy(dst.data, src.data)
    return dst, ok
}

tensor_from_data :: proc(
    shape: [$N]int,
    data: []$T,
    allocator := context.allocator,
) -> Tensor(T, N) {

    size: int = 1
    for dimension in shape {
        size *= dimension
    }
    assert(len(data) == size)
    return Tensor(T, N){allocator = allocator, shape = shape, data = data}
}


@(require_results)
tensor_complex64_from_parts_slice :: proc(
    shape: [$N]int,
    real_data: []f32,
    imag_data: []f32,
    allocator := context.allocator,
) -> Tensor(complex64, N) {

    N_real := len(real_data)
    assert(N_real == len(imag_data))
    size: int = 1
    for dimension in shape {
        size *= dimension
    }
    assert(N_real == size)

    res, ok := tensor_create(complex64, shape, allocator)
    assert(ok)
    for i in 0 ..< size {
        res.data[i] = complex(real_data[i], imag_data[i])
    }
    return res
}

@(require_results)
tensor_complex64_from_parts_tensor :: proc(
    real_tensor: Tensor(f32, $N),
    imag_tensor: Tensor(f32, N),
) -> Tensor(complex64, N) {

    assert(real_tensor.shape == imag_tensor.shape)
    res, ok := tensor_create(complex64, real_tensor.shape, real_tensor.allocator)
    assert(ok)

    for i in 0 ..< len(real_tensor.data) {
        res.data[i] = complex(real_tensor.data[i], imag_tensor.data[i])
    }
    return res
}

@(require_results)
tensor_complex128_from_parts_slice :: proc(
    shape: [$N]int,
    real_data: []f64,
    imag_data: []f64,
    allocator := context.allocator,
) -> Tensor(complex128, N) {

    N_real := len(real_data)
    assert(N_real == len(imag_data))
    size: int = 1
    for dimension in shape {
        size *= dimension
    }
    assert(N_real == size)

    res, ok := tensor_create(complex128, shape, allocator)
    assert(ok)
    for i in 0 ..< size {
        res.data[i] = complex(real_data[i], imag_data[i])
    }
    return res
}

@(require_results)
tensor_complex128_from_parts_tensor :: proc(
    real_tensor: Tensor(f64, $N),
    imag_tensor: Tensor(f64, N),
) -> Tensor(complex128, N) {

    assert(real_tensor.shape == imag_tensor.shape)
    res, ok := tensor_create(complex128, real_tensor.shape, real_tensor.allocator)
    assert(ok)

    for i in 0 ..< len(real_tensor.data) {
        res.data[i] = complex(real_tensor.data[i], imag_tensor.data[i])
    }
    return res
}

@(require_results)
tensor_complex_from_parts :: proc {
    tensor_complex64_from_parts_tensor,
    tensor_complex128_from_parts_tensor,
    tensor_complex64_from_parts_slice,
    tensor_complex128_from_parts_slice,
}

