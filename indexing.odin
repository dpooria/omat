package omat

unravel_index_col_major :: proc(idx: int, shape: [$N]int) -> (indices: [N]int) {
    assert(idx < size_of_shape(shape))
    idx := idx
    for s, d in shape {
        indices[d] = idx % s
        idx /= s
    }
    return indices
}

unravel_index :: proc(idx: int, shape: [$N]int) -> (indices: [N]int) {
    assert(idx < size_of_shape(shape))
    idx := idx

    for dim := N - 1; dim >= 0; dim -= 1 {
        indices[dim] = idx % shape[dim]
        idx /= shape[dim]
    }

    return indices
}

ravel_multi_index_slice :: proc(indices: []int, shape: [$N]int) -> (idx: int) {
    assert(len(indices) == N)
    for i in 0 ..< N {
        assert(indices[i] < shape[i])
        idx = idx * shape[i] + indices[i]
    }
    return idx
}

ravel_multi_index_arr :: proc(indices: [$N]int, shape: [N]int) -> (idx: int) {
    for i in 0 ..< N {
        assert(indices[i] < shape[i])
        idx = idx * shape[i] + indices[i]
    }
    return idx
}

ravel_multi_index :: proc {
    ravel_multi_index_arr,
    ravel_multi_index_slice,
}

at_indices_arr :: proc(t: Tensor($T, $N), indices: [N]int) -> ^T {
    for i, d in indices {
        assert(0 <= i && i < t.shape[d], "at: index out of range")
    }
    return &t.data[offset_of_index(indices, t.strides)]
}

at_indices_int :: proc(t: Tensor($T, $N), indices: ..int) -> ^T {
    assert(len(indices) == N, "at: one index per axis")
    index: [N]int
    copy(index[:], indices)
    return at_indices_arr(t, index)
}

at :: proc {
    at_indices_int,
    at_indices_arr,
}

atv_indices_arr :: proc(t: Tensor($T, $N), indices: [N]int) -> T {
    return at_indices_arr(t, indices)^
}

atv_indices_int :: proc(t: Tensor($T, $N), indices: ..int) -> T {
    return at_indices_int(t, ..indices)^
}

atv :: proc {
    atv_indices_int,
    atv_indices_arr,
}

// row-major strides of a contiguous tensor of this shape
strides :: proc(shape: [$N]int) -> [N]int {
    strides: [N]int
    step := 1
    for dim := N - 1; dim >= 0; dim -= 1 {
        strides[dim] = step
        step *= shape[dim]
    }
    return strides
}


// A contiguous copy of t with a new shape of the same size, elements in row-major order
@(require_results)
reshape :: proc(
    t: Tensor($T, $N),
    shape: [$M]int,
    allocator := context.allocator,
) -> Tensor(T, M) {
    assert(size_of_shape(shape) == size_of_shape(t.shape), "reshape: sizes differ")
    res := zeros(T, shape, allocator)
    // res's memory seen in t's shape, so one strided pass copies in row-major order
    as_t := Tensor(T, N) {
        data     = res.data,
        shape    = t.shape,
        strides  = strides(t.shape),
        borrowed = true,
    }
    assign(as_t, t)
    return res
}

// The same tensor under a new shape, sharing the data and its ownership: destroy one of the two.
// Needs contiguous data; reshape copies any view.
ireshape :: proc(t: Tensor($T, $N), shape: [$M]int) -> Tensor(T, M) {
    assert(size_of_shape(shape) == size_of_shape(t.shape), "ireshape: sizes differ")
    assert_contiguous(t)
    return Tensor(T, M) {
        allocator = t.allocator,
        data = t.data,
        shape = shape,
        strides = strides(shape),
        borrowed = t.borrowed,
    }
}

@(private = "file")
grid_offset :: #force_inline proc(grid: Tensor($T, $N), flat: int, contiguous: bool) -> int {
    if contiguous {
        return flat
    }
    return offset_of_index(unravel_index(flat, grid.shape), grid.strides)
}

// dst[i] = src[index[i]]
gather_into :: proc(dst: Tensor($T, 1), src: Tensor(T, $N), index: []int) {
    assert(dst.shape[0] == len(index), "gather: one index per dst")
    contiguous := is_contiguous(src)
    step := dst.strides[0]
    for flat, g in index {
        dst.data[g * step] = src.data[grid_offset(src, flat, contiguous)]
    }
}

@(require_results)
gather :: proc(src: Tensor($T, $N), index: []int, allocator := context.allocator) -> Tensor(T, 1) {
    res := zeros(T, [1]int{len(index)}, allocator)
    gather_into(res, src, index)
    return res
}

// dst[index[g]] = src[g], other elements untouched (zero the dst first for a fresh transform)
scatter :: proc(dst: ^Tensor($T, $N), src: Tensor(T, 1), index: []int) {
    assert(src.shape[0] == len(index), "scatter: one index per src")
    contiguous := is_contiguous(dst^)
    step := src.strides[0]
    for flat, g in index {
        dst.data[grid_offset(dst^, flat, contiguous)] = src.data[g * step]
    }
}

// dst[index[g]] += src[g]; repeated indices accumulate
scatter_add :: proc(dst: ^Tensor($T, $N), src: Tensor(T, 1), index: []int) {
    assert(src.shape[0] == len(index), "scatter_add: one index per src")
    contiguous := is_contiguous(dst^)
    step := src.strides[0]
    for flat, g in index {
        dst.data[grid_offset(dst^, flat, contiguous)] += src.data[g * step]
    }
}
