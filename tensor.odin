package omat

import "base:intrinsics"
import "core:fmt"
import "core:mem"

Vector3i :: [3]int
Vector3f :: [3]f64
Vector3c :: [3]complex128

Tensor :: struct($T: typeid, $N: int) where intrinsics.type_is_numeric(T) {
    allocator: mem.Allocator,
    data:      []T,
    shape:     [N]int,
    // strides:   [N]int,
    // owns_data: bool,
}

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

tensor_copy :: proc(src: Tensor($T, $N)) -> (dst: Tensor(T, N)) {
    dst, _ = tensor_create(T, src.shape, src.allocator)
    copy(dst.data, src.data)
    return dst
}
