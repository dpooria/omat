package omat

import "base:intrinsics"
import "core:fmt"
import "core:mem"

// An n-dimensional array: the element at index i is data[sum_d( i[d] * strides[d] )].
// Tensors made by omat are row-major and contiguous.
// A view (view, view_row, transpose_view, ...) is a borrowed tensor with other strides into another tensor data, which must outlive it;
// tensor_destroy frees nothing for borrowed tensors.
Tensor :: struct($T: typeid, $N: int) where intrinsics.type_is_numeric(T) {
    allocator: mem.Allocator,
    data:      []T, // from the first element to the last
    shape:     [N]int,
    strides:   [N]int,
    borrowed:  bool,
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
    t.strides = strides(shape)
    size: int = size_of_shape(shape)
    data, err := mem.make_aligned([]T, size, 16, t.allocator) // 16-byte alignment
    ok = (err == nil)
    if !ok {
        fmt.eprintfln("Could not create an array with the shape (%v): %v", shape, err)
        return
    }
    t.data = data
    return t, ok
}

tensor_destroy :: proc(t: ^Tensor($T, $N)) {
    if !t.borrowed {
        delete(t.data, t.allocator)
    }
    t^ = {}
}

// A contiguous copy of any tensor or view
@(require_results)
tensor_copy :: proc(src: Tensor($T, $N), allocator := context.allocator) -> Tensor(T, N) {
    dst := zeros(T, src.shape, allocator)
    assign(dst, src)
    return dst
}

// Copies t into contiguous storage from allocator and replaces it in place.
// The old data is destroyed unless if t is borrowed, then parent tensor is unaffected.
tensor_realloc :: proc(t: ^Tensor($T, $N), allocator: mem.Allocator) {
    dst := tensor_copy(t^, allocator)
    tensor_destroy(t)
    t^ = dst
}

// for FFTW and SIMD -> 16-byte aligned data
is_aligned :: proc(t: Tensor($T, $N)) -> bool {
    return uintptr(raw_data(t.data)) % 16 == 0
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
    return Tensor(T, N) {
        allocator = allocator,
        shape = shape,
        strides = strides(shape),
        data = data,
    }
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
tensor_complex64_from_parts_tensor :: proc(re, im: Tensor(f32, $N), allocator := context.allocator) -> Tensor(complex64, N) {
    res := zeros(complex64, re.shape, allocator)
    complex_into(res, re, im)
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
tensor_complex128_from_parts_tensor :: proc(re, im: Tensor(f64, $N), allocator := context.allocator) -> Tensor(complex128, N) {
    res := zeros(complex128, re.shape, allocator)
    complex_into(res, re, im)
    return res
}

// out = complex(re, im) element by element, for tensors and views of one shape
complex_into :: proc(out: Tensor($C, $N), re, im: Tensor($R, N)) {
    assert(out.shape == re.shape && re.shape == im.shape, "complex_into: shapes differ")
    l := plan_loop(re.shape, out.strides, re.strides, im.strides)
    so, sr, si := loop_step(l, out.strides), loop_step(l, re.strides), loop_step(l, im.strides)
    for r in 0 ..< l.rows {
        po, pr, pi :=
            row_offset(l, r, out.strides),
            row_offset(l, r, re.strides),
            row_offset(l, r, im.strides)
        for k in 0 ..< l.n {
            out.data[po + k * so] = complex(re.data[pr + k * sr], im.data[pi + k * si])
        }
    }
}

@(require_results)
tensor_complex_from_parts :: proc {
    tensor_complex64_from_parts_tensor,
    tensor_complex128_from_parts_tensor,
    tensor_complex64_from_parts_slice,
    tensor_complex128_from_parts_slice,
}

