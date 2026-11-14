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

// A view of t[..., index, ...] along axis
@(require_results)
atax :: proc(
    t: Tensor($T, $N),
    index: int,
    axis: int = 0,
) -> Tensor(T, N - 1) where N > 1 {
    assert(0 <= axis && axis < N)
    // bring axis to the front, then take its row
    axes: [N]int
    axes[0] = axis
    next := 1
    for d in 0 ..< N {
        if d != axis {
            axes[next] = d
            next += 1
        }
    }
    return view_row(permute_view(t, axes[:]), index)
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
reshape :: proc(t: Tensor($T, $N), shape: [$M]int, allocator := context.allocator) -> Tensor(T, M) {
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


// Axis i of the result is axis axes[i] of t, as numpy's transpose(t, axes)
@(require_results)
permute_axes :: proc(
    t: Tensor($T, $N),
    axes: []int,
    allocator := context.allocator,
) -> Tensor(T, N) where N >
    0 {
    return tensor_copy(permute_view(t, axes), allocator)
}

